import CoreImage
import FaceforgeCore
import Network
import PhotosUI
import SwiftUI

/// Still-image mode: swap the source face into a photo, on the iPhone or the PC.
struct PhotoSwapView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var models: ModelStore
    @State private var item: PhotosPickerItem?
    @State private var result: UIImage?
    @State private var busy = false
    @State private var message = ""
    @State private var engine: LocalSwapEngine?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let result = result {
                    Image(uiImage: result).resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    ContentUnavailableView("Pick a photo", systemImage: "photo",
                                           description: Text("The source face is swapped into it."))
                }
                if busy { ProgressView() }
                Text(message).font(.footnote).multilineTextAlignment(.center)
                HStack {
                    PhotosPicker("Choose photo", selection: $item, matching: .images).buttonStyle(.borderedProminent)
                    if let result = result {
                        Button("Save to Photos") {
                            UIImageWriteToSavedPhotosAlbum(result, nil, nil, nil)
                            message = "Saved to Photos."
                        }.buttonStyle(.bordered)
                    }
                }
            }
            .padding()
            .navigationTitle("Photo")
            .onChange(of: item) { _, newItem in Task { await run(newItem) } }
        }
    }

    @MainActor
    private func run(_ newItem: PhotosPickerItem?) async {
        guard let data = try? await newItem?.loadTransferable(type: Data.self),
              let ui = UIImage(data: data), let jpeg = ui.normalizedJPEG(), let input = CIImage(data: jpeg) else { return }
        guard AppSettings.hasSource, let src = CIImage(contentsOf: AppSettings.sourceURL) else {
            message = "Pick a source face first (Source tab)."
            return
        }
        busy = true
        defer { busy = false }
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            let out: CIImage
            switch settings.mode {
            case .onDevice:
                guard let paths = models.paths else { message = "Download the models first."; return }
                let useCoreML = settings.useCoreML, cache = models.coreMLCacheDir
                let fast = settings.preferFastModel ? models.fastModel : nil, target = settings.computeTarget
                let existing = engine
                let (eng, img, found) = try await Task.detached(priority: .userInitiated) {
                    () throws -> (LocalSwapEngine, CIImage, Bool) in
                    let e = try existing ?? LocalSwapEngine(models: paths, fast: fast, useCoreML: useCoreML,
                                                                   target: target, cacheDir: cache)
                    try e.setSource(src)
                    e.detectEvery = 1
                    var timing = LocalSwapEngine.Timing()
                    let (o, f) = try e.process(input, timing: &timing)
                    return (e, o, f)
                }.value
                engine = eng
                out = img
                message = found ? "" : "No face found in the photo."
            case .pc:
                out = try await PCOneShot.swap(jpeg: jpeg, source: try Data(contentsOf: AppSettings.sourceURL),
                                               settings: settings)
                message = ""
            }
            let ctx = CIContext()
            if let cg = ctx.createCGImage(out, from: out.extent) { result = UIImage(cgImage: cg) }
            message += String(format: (message.isEmpty ? "" : " ") + "Done in %.1f s.", CFAbsoluteTimeGetCurrent() - t0)
            Log.info("Photo swap (\(settings.mode.rawValue)): \(message)", "photo")
        } catch {
            message = error.localizedDescription
            Log.error("Photo swap failed (\(settings.mode.rawValue)): \(error)", "photo")
        }
    }
}

/// One frame through the PC server (connect, send source + frame, await result).
enum PCOneShot {
    static func swap(jpeg: Data, source: Data, settings: AppSettings) async throws -> CIImage {
        let client = OffloadClient()
        let host = settings.pcHost, port = settings.pcPort, token = settings.pcToken
        guard !host.isEmpty else { throw OneShotError.message("Enter your PC's address in Settings.") }
        return try await withCheckedThrowingContinuation { cont in
            var finished = false
            func finish(_ r: Result<CIImage, Error>) {
                guard !finished else { return }
                finished = true
                client.onState = nil
                client.onSourceAck = nil
                client.onResult = nil
                client.disconnect()
                cont.resume(with: r)
            }
            client.onState = { s in
                switch s {
                case .connected:
                    client.sendSource(jpeg: source)
                case .failed(let m):
                    finish(.failure(OneShotError.message(m)))
                default: break
                }
            }
            client.onSourceAck = { ok, msg in
                if !ok { finish(.failure(OneShotError.message(msg))); return }
                if !client.trySendFrame(jpeg: jpeg) { finish(.failure(OneShotError.message("Could not send."))) }
            }
            client.onResult = { r, _ in
                if let img = CIImage(data: r.jpeg) { finish(.success(img)) }
                else { finish(.failure(OneShotError.message("Bad image from PC."))) }
            }
            client.connect(to: .hostPort(host: NWEndpoint.Host(host),
                                         port: NWEndpoint.Port(rawValue: UInt16(clamping: port)) ?? 47800),
                           token: token)
        }
    }

    enum OneShotError: LocalizedError {
        case message(String)
        var errorDescription: String? { if case .message(let m) = self { return m }; return nil }
    }
}
