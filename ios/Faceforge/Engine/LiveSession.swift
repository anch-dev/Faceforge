import CoreImage
import FaceforgeCore
import Foundation
import Network

/// Runs live mode: camera -> (on-device engine | PC offload) -> preview.
/// Publishes honest stats: which backend actually runs, FPS, 1%-low, latency.
final class LiveSession: ObservableObject {
    @Published private(set) var running = false
    @Published private(set) var status = "Idle"
    @Published private(set) var backend = ""
    @Published private(set) var fps = 0.0
    @Published private(set) var onePctLow = 0.0
    @Published private(set) var latencyMs = 0.0
    @Published private(set) var stageText = ""
    @Published private(set) var thermalText = "nominal"

    let renderer = PreviewRenderer()
    private let camera = CameraController()
    private var engine: LocalSwapEngine?
    private let client = OffloadClient()
    private let ops = ImageOps()           // used on the camera queue only

    // Accessed on the camera queue only (except setup while stopped).
    private var stats = FrameStats(window: 300)
    private var lastFrameTime = CFAbsoluteTimeGetCurrent()
    private var lastAccepted = 0.0
    private var policy = ThermalPolicy.forState(0, userDetectEvery: 2)
    private var mode = ProcessingMode.onDevice
    private var jpegQuality = 0.75
    private var activeEngine: LocalSwapEngine?
    private var thermalObserver: NSObjectProtocol?
    private var statsTimer: Timer?

    @MainActor
    func start(settings: AppSettings, models: ModelStore) async {
        guard !running else { return }
        guard await CameraController.requestAccess() else {
            status = "Camera access denied. Enable it in Settings > Faceforge."
            return
        }
        mode = settings.mode
        jpegQuality = settings.jpegQuality
        stats.reset()
        do {
            try camera.configure(resolution: settings.resolution)
        } catch {
            status = error.localizedDescription
            return
        }

        switch settings.mode {
        case .onDevice:
            guard let paths = models.paths else { status = "Download the models first (Models tab)."; return }
            guard AppSettings.hasSource else { status = "Pick a source face first (Source tab)."; return }
            status = "Loading models (first launch compiles them for the Neural Engine; this can take a minute)..."
            let useCoreML = settings.useCoreML, cache = models.coreMLCacheDir, every = settings.detectEvery
            do {
                let eng = try await Task.detached(priority: .userInitiated) { () throws -> LocalSwapEngine in
                    let e = try LocalSwapEngine(models: paths, useCoreML: useCoreML, cacheDir: cache)
                    e.detectEvery = every
                    guard let src = CIImage(contentsOf: AppSettings.sourceURL) else {
                        throw LocalSwapEngine.EngineError.noSourceFace
                    }
                    try e.setSource(src)
                    return e
                }.value
                engine = eng
                backend = eng.backendSummary
                camera.videoQueue.sync { activeEngine = eng }
            } catch {
                status = "Could not start: \(error.localizedDescription)"
                return
            }
        case .pc:
            guard !settings.pcHost.isEmpty else { status = "Enter your PC's address in Settings."; return }
            connectPC(host: settings.pcHost, port: settings.pcPort, token: settings.pcToken,
                      detectEvery: settings.detectEvery)
        }

        camera.onFrame = { [weak self] frame in self?.handle(frame) }
        observeThermal(userDetectEvery: settings.detectEvery)
        camera.start()
        running = true
        if case .onDevice = settings.mode { status = "Running on iPhone" }
        statsTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.publishStats() }
        }
    }

    @MainActor
    func stop() {
        camera.stop()
        camera.onFrame = nil
        camera.videoQueue.sync { activeEngine = nil }
        engine = nil
        client.disconnect()
        statsTimer?.invalidate()
        if let o = thermalObserver { NotificationCenter.default.removeObserver(o) }
        running = false
        status = "Stopped"
    }

    /// Connect using a typed host or a Bonjour-discovered endpoint.
    @MainActor
    func connectPC(host: String, port: Int, token: String, detectEvery: Int, endpoint: NWEndpoint? = nil) {
        let ep = endpoint ?? NWEndpoint.hostPort(host: NWEndpoint.Host(host),
                                                 port: NWEndpoint.Port(rawValue: UInt16(clamping: port)) ?? 47800)
        client.onState = { [weak self] s in
            Task { @MainActor in
                guard let self = self else { return }
                switch s {
                case .connecting: self.status = "Connecting to PC..."
                case .connected(let provider, let device):
                    self.status = "Connected to PC"
                    self.backend = "PC: \(provider.replacingOccurrences(of: "ExecutionProvider", with: "")) \(device)"
                    if let src = try? Data(contentsOf: AppSettings.sourceURL) { self.client.sendSource(jpeg: src) }
                    self.client.sendSettings(["detect_every": detectEvery])
                case .failed(let msg): self.status = "PC: \(msg)"
                case .idle: break
                }
            }
        }
        client.onSourceAck = { [weak self] ok, msg in
            Task { @MainActor in if !ok { self?.status = "PC: \(msg)" } }
        }
        client.onResult = { [weak self] result, rtt in
            guard let self = self, let img = CIImage(data: result.jpeg) else { return }
            self.renderer.display(img)
            self.camera.videoQueue.async {
                self.stats.add(ms: rtt)
            }
            Task { @MainActor in self.latencyMs = rtt; self.stageText = String(format: "PC %.0f ms", result.serverMs) }
        }
        client.connect(to: ep, token: token)
    }

    // MARK: camera queue

    private func handle(_ frame: CIImage) {
        let now = CFAbsoluteTimeGetCurrent()
        if policy.paused { return }
        if policy.maxFPS > 0, now - lastAccepted < 1.0 / Double(policy.maxFPS) { return }
        lastAccepted = now
        switch mode {
        case .onDevice:
            guard let eng = activeEngine else { return }
            var timing = LocalSwapEngine.Timing()
            let t0 = CFAbsoluteTimeGetCurrent()
            let out = (try? eng.process(frame, timing: &timing))?.0 ?? frame
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            stats.add(ms: ms)
            renderer.display(out)
            let text = String(format: "detect %.0f · swap %.0f · paste %.0f ms", timing.detectMs, timing.swapMs, timing.compositeMs)
            Task { @MainActor [weak self] in self?.stageText = text; self?.latencyMs = ms }
        case .pc:
            guard let jpeg = ops.jpeg(frame, quality: jpegQuality, maxSide: 960) else { return }
            _ = client.trySendFrame(jpeg: jpeg)
        }
    }

    // MARK: stats / thermal

    @MainActor
    private func publishStats() {
        let (f, low) = camera.videoQueue.sync { (stats.avgFPS, stats.onePercentLowFPS) }
        fps = f
        onePctLow = low
    }

    @MainActor
    private func observeThermal(userDetectEvery: Int) {
        let apply: () -> Void = { [weak self] in
            guard let self = self else { return }
            let raw = ProcessInfo.processInfo.thermalState.rawValue
            let p = ThermalPolicy.forState(raw, userDetectEvery: userDetectEvery)
            self.camera.videoQueue.async {
                self.policy = p
                self.activeEngine?.detectEvery = p.detectEvery
            }
            Task { @MainActor in
                self.thermalText = ["nominal", "fair", "serious", "critical"][min(raw, 3)]
                if p.paused { self.status = "Paused: iPhone is too hot. It resumes when it cools down." }
            }
        }
        apply()
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { _ in apply() }
    }
}
