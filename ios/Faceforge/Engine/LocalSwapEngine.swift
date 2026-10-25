import CoreImage
import FaceforgeCore
import Foundation
import OnnxRuntimeBindings

/// On-device pipeline: SCRFD detect -> ArcFace embed (source, once) ->
/// inswapper_128 swap -> composite.
///
/// The swap is the heavy part (175 GFLOPs). The Neural Engine only runs fp16
/// models, so the publisher's fp32 file can at best use the GPU; the optional
/// fast model (fp16, made on the user's PC from their own copy) runs on the
/// Neural Engine, natively through Core ML or through ONNX Runtime's Core ML
/// EP. `backendSummary` reports what is actually used and how fast it was.
final class LocalSwapEngine {
    struct Timing { var detectMs = 0.0, swapMs = 0.0, compositeMs = 0.0 }

    /// One swapped face, ready to paste: the 128 px fake, the frame->crop
    /// transform it was made with, and the tracker box at that moment (so a
    /// later frame can re-target it to where the head is now).
    struct SwapResult {
        let fake: CIImage
        let transform: Affine2D
        let track: Box?
        let time: CFAbsoluteTime
    }

    private let env: ORTEnv
    private var detector: ORTSession
    private var detectorOnCoreML = false
    private let detectorPath: URL
    private let recognizer: ORTSession
    private let swapper: SwapRunner
    private let detInput: String, detOutputs: [String]
    private let recInput: String, recOutput: String
    private let emap: [Float]
    private let ops = ImageOps()
    let backendSummary: String
    /// Steady-state swap time measured at start-up (ms), for the log / UI.
    private(set) var benchmarkMs = 0.0

    var detSize = 320
    var detectEvery = 2
    private var frameIndex = 0
    private var cachedFace: DetectedFace?
    private var cachedTrack: Box?
    private(set) var sourceLatent: [Float]?

    init(models: ModelStore.Paths, fast: ModelStore.FastModel?, useCoreML: Bool,
         target: ComputeTarget, cacheDir: URL) throws {
        // Info level shows ORT's Core ML partitioning (how many nodes run in
        // Core ML vs on the CPU) in the log file.
        let env = try ORTEnv(loggingLevel: Log.verbose ? .info : .warning)
        Log.info("ONNX Runtime \(ORTVersion() ?? "?") · Core ML EP available: \(ORTIsCoreMLExecutionProviderAvailable())"
                 + " · compute target \(target.rawValue) · fast model \(fast?.label ?? "none")", "engine")
        self.env = env
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        /// Try Core ML with the most options first (compiled-model cache,
        /// FastPrediction, fp16 GPU accumulation), then fewer, then plain CPU.
        /// Returns the session and the label of what worked ("CPU" if no Core ML).
        func makeSession(_ path: URL, coreML: Bool, units: String, tuned: Bool) throws -> (ORTSession, String?) {
            let base: [String: String] = ["ModelFormat": "MLProgram", "MLComputeUnits": units,
                                          "RequireStaticInputShapes": "0"]
            var attempts: [(String, [String: String]?)] = []
            if coreML && ORTIsCoreMLExecutionProviderAvailable() {
                var cached = base
                cached["ModelCacheDirectory"] = cacheDir.path
                if tuned {
                    var t = cached
                    t["SpecializationStrategy"] = "FastPrediction"
                    t["AllowLowPrecisionAccumulationOnGPU"] = "1"
                    if Log.verbose { t["ProfileComputePlan"] = "1" }   // per-op device placement in the log
                    attempts.append(("CoreML[\(units)]+tuned", t))
                }
                attempts += [("CoreML[\(units)]+cache", cached), ("CoreML[\(units)]", base)]
            }
            attempts.append(("CPU", nil))
            var lastError: Error?
            for (label, cfg) in attempts {
                let t0 = CFAbsoluteTimeGetCurrent()
                do {
                    let o = try ORTSessionOptions()
                    try o.setGraphOptimizationLevel(.all)
                    try? o.setLogSeverityLevel(Log.verbose ? .info : .warning)
                    // Plain CPU fallback. Not XNNPACK: with det_10g's dynamic input it
                    // aborted the process (uncaught C++ exception during session
                    // init) on iPhone 16 Pro Max / ORT 1.28.0.
                    if let cfg = cfg { try o.appendCoreMLExecutionProvider(withOptionsV2: cfg) }
                    let session = try ORTSession(env: env, modelPath: path.path, sessionOptions: o)
                    Log.info("\(path.lastPathComponent): session ready on \(label) in "
                             + String(format: "%.1f s", CFAbsoluteTimeGetCurrent() - t0), "engine")
                    return (session, cfg == nil ? nil : label)
                } catch {
                    Log.warn("\(path.lastPathComponent): \(label) failed: \(error.localizedDescription)", "engine")
                    lastError = error
                }
            }
            throw lastError ?? EngineError.missingOutput("session")
        }

        // Detector: Core ML at 640 px, the input size det_10g's declared output
        // shapes assume (at 320 Core ML rejected them: "Invalid shape for output
        // feature '_448'"). fp32 model, so "ALL" (GPU): the Neural Engine runs
        // only fp16. If Core ML fails at run time detect() falls back to CPU @ 320.
        let (det, detCoreML) = try makeSession(models.detector, coreML: useCoreML, units: "ALL", tuned: false)
        let rec = try makeSession(models.recognizer, coreML: false, units: "ALL", tuned: false).0

        // Swapper, fastest first: native Core ML fp16 -> fp16 ONNX on Core ML
        // -> the publisher's fp32 ONNX on Core ML (GPU) -> CPU.
        var candidates: [(String, () throws -> SwapRunner)] = []
        if useCoreML, case .nativeCoreML(let url)? = fast {
            candidates.append(("native Core ML", { try NativeCoreMLSwapper(compiledModel: url, units: target.mlUnits) }))
        }
        if case .onnxFP16(let url)? = fast {
            candidates.append(("fp16 ONNX", {
                let (s, label) = try makeSession(url, coreML: useCoreML, units: target.ortValue, tuned: true)
                return try OrtSwapper(session: s, label: "ONNX fp16 · " + (label ?? "CPU"))
            }))
        }
        candidates.append(("fp32 ONNX", {
            // fp32 cannot run on the Neural Engine; ask Core ML for the GPU instead.
            if target == .neuralEngine && useCoreML {
                Log.info("fp32 swap model: the Neural Engine needs fp16, so using Core ML ALL (GPU). "
                         + "Install the fast model (Models tab) for the Neural Engine.", "engine")
            }
            let units = target == .neuralEngine ? "ALL" : target.ortValue
            let (s, label) = try makeSession(models.swapper, coreML: useCoreML, units: units, tuned: true)
            return try OrtSwapper(session: s, label: "ONNX fp32 · " + (label ?? "CPU"))
        }))
        var chosen: (SwapRunner, Double)?
        var swapError: Error?
        for (name, make) in candidates where chosen == nil {
            do {
                let r = try make()
                chosen = (r, try LocalSwapEngine.benchmark(r))
            } catch {
                Log.warn("Swap backend \(name) unusable (\(error.localizedDescription)); trying next", "engine")
                swapError = error
            }
        }
        guard let picked = chosen else { throw swapError ?? EngineError.missingOutput("swapper") }
        swapper = picked.0
        benchmarkMs = picked.1

        // Read names from locals: no access to self before every property is set.
        let detOuts = try det.outputNames()
        let detIn = try det.inputNames()[0]
        let recIn = try rec.inputNames()[0]
        let recOut = try rec.outputNames()[0]
        detector = det
        detectorPath = models.detector
        recognizer = rec
        detInput = detIn
        detOutputs = detOuts
        recInput = recIn
        recOutput = recOut

        // The emap always comes from the publisher's fp32 file.
        let em = try OnnxInitializerReader.lastInitializer(fileURL: models.swapper)
        guard em.dims == [512, 512] else { throw EngineError.badEmap(em.dims) }
        emap = em.floats
        Log.debug("detector in=\(detIn) out=\(detOuts) · recognizer in=\(recIn) out=\(recOut) · emap \(em.dims)", "engine")

        detectorOnCoreML = detCoreML != nil
        detSize = detCoreML != nil ? 640 : 320
        backendSummary = "ONNX Runtime \(ORTVersion() ?? "?") · swap: \(picked.0.label)"
            + String(format: " (%.0f ms)", picked.1)
        Log.info("Backend: \(backendSummary) · detector \(detectorOnCoreML ? "Core ML" : "CPU") @ \(detSize) px", "engine")
    }

    /// Warm up (the first Core ML run specializes the model for the device) and
    /// time a few steady-state runs. Returns the median in ms.
    private static func benchmark(_ r: SwapRunner) throws -> Double {
        let target = [Float](repeating: 0.5, count: 3 * 128 * 128)
        var latent = [Float](repeating: 0, count: 512)
        latent[0] = 1
        var times: [Double] = []
        for i in 0..<4 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try r.run(target: target, latent: latent)
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            if i == 0 { Log.info(String(format: "Swap first run (device specialisation): %.0f ms", ms), "engine") }
            else { times.append(ms) }
        }
        let med = times.sorted()[times.count / 2]
        Log.info(String(format: "Swap steady state: %.1f ms median (%@)", med,
                        times.map { String(format: "%.0f", $0) }.joined(separator: ", ")), "engine")
        return med
    }

    // MARK: - stages

    func detect(_ frame: CIImage) throws -> [DetectedFace] {
        do {
            return try detectOnce(frame)
        } catch where detectorOnCoreML {
            Log.warn("Detector on Core ML failed (\(error.localizedDescription)); switching to CPU @ 320 px", "engine")
            let o = try ORTSessionOptions()
            try o.setGraphOptimizationLevel(.all)
            detector = try ORTSession(env: env, modelPath: detectorPath.path, sessionOptions: o)
            detectorOnCoreML = false
            detSize = 320
            return try detectOnce(frame)
        }
    }

    private func detectOnce(_ frame: CIImage) throws -> [DetectedFace] {
        let (tensor, scale) = try ops.detectorTensor(frame: frame, size: detSize)
        let out = try run(detector, [detInput: (tensor, [1, 3, detSize, detSize])], outputs: detOutputs)
        return SCRFD.decode(outputs: detOutputs.map { out[$0]! }, inputSize: detSize, detScale: scale)
    }

    func embed(_ frame: CIImage, face: DetectedFace) throws -> [Float] {
        let m = FaceAlign.estimateNorm(face.kps, size: 112)
        let t = try ops.alignedTensor(frame: frame, transform: m, size: 112, mean: 127.5, std: 127.5)
        return Tensor.l2normalize(try run(recognizer, [recInput: (t, [1, 3, 112, 112])], outputs: [recOutput])[recOutput]!)
    }

    /// Set the identity to swap in, from a photo containing a face.
    @discardableResult
    func setSource(_ image: CIImage) throws -> Int {
        let faces = try detect(image)
        Log.info("Source image \(Int(image.extent.width))x\(Int(image.extent.height)): \(faces.count) face(s)", "engine")
        guard let best = faces.max(by: { $0.area < $1.area }) else { throw EngineError.noSourceFace }
        sourceLatent = Tensor.inswapperLatent(embedding: try embed(image, face: best), emap: emap)
        cachedFace = nil
        return faces.count
    }

    /// Detect (every `detectEvery` calls, else follow the cached face with the
    /// tracker box) and swap the main face. nil if no face / no source.
    func swap(_ frame: CIImage, track: Box?, timing: inout Timing) throws -> SwapResult? {
        guard let latent = sourceLatent else { return nil }
        if frameIndex % max(1, detectEvery) == 0 || cachedFace == nil {
            let t0 = CFAbsoluteTimeGetCurrent()
            cachedFace = try detect(frame).max(by: { $0.area < $1.area })
            cachedTrack = track
            timing.detectMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }
        frameIndex += 1
        guard let face = cachedFace else { return nil }
        var kps = face.kps
        if let old = cachedTrack, let now = track { kps = FaceTrack.move(kps, from: old, to: now) }

        let t1 = CFAbsoluteTimeGetCurrent()
        let m = FaceAlign.estimateNorm(kps, size: 128)
        let target = try ops.alignedTensor(frame: frame, transform: m, size: 128, mean: 0, std: 255)
        let out = try swapper.run(target: target, latent: latent)
        timing.swapMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        let fake = try ops.image(fromBGRA: Tensor.bgra(fromCHWRGB: out, width: 128, height: 128), width: 128, height: 128)
        return SwapResult(fake: fake, transform: m, track: track, time: CFAbsoluteTimeGetCurrent())
    }

    /// Swap and composite in one go (photo mode). Returns the input unchanged
    /// if no face / no source.
    func process(_ frame: CIImage, timing: inout Timing) throws -> (CIImage, Bool) {
        guard let r = try swap(frame, track: nil, timing: &timing) else { return (frame, false) }
        let t2 = CFAbsoluteTimeGetCurrent()
        let result = ops.paste(fake: r.fake, onto: frame, transform: r.transform, size: 128)
        timing.compositeMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        return (result, true)
    }

    // MARK: - ORT helpers

    private func run(_ s: ORTSession, _ inputs: [String: ([Float], [Int])], outputs: [String]) throws -> [String: [Float]] {
        try Self.runORT(s, inputs, outputs: outputs)
    }

    static func runORT(_ s: ORTSession, _ inputs: [String: ([Float], [Int])], outputs: [String]) throws -> [String: [Float]] {
        var values: [String: ORTValue] = [:]
        for (name, (data, shape)) in inputs {
            let bytes = NSMutableData(bytes: data, length: data.count * MemoryLayout<Float>.stride)
            values[name] = try ORTValue(tensorData: bytes, elementType: .float, shape: shape.map { NSNumber(value: $0) })
        }
        let res = try s.run(withInputs: values, outputNames: Set(outputs), runOptions: nil)
        var out: [String: [Float]] = [:]
        for name in outputs {
            guard let v = res[name] else { throw EngineError.missingOutput(name) }
            let d = try v.tensorData() as Data
            out[name] = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        return out
    }

    /// inswapper through ONNX Runtime (Core ML EP or CPU).
    final class OrtSwapper: SwapRunner {
        private let session: ORTSession
        private let targetName: String, sourceName: String, outputName: String
        let label: String

        init(session: ORTSession, label: String) throws {
            self.session = session
            self.label = label
            let ins = try session.inputNames()
            targetName = ins.first { $0 == "target" } ?? ins[0]
            sourceName = ins.first { $0 == "source" } ?? ins[1]
            outputName = try session.outputNames()[0]
        }

        func run(target: [Float], latent: [Float]) throws -> [Float] {
            try LocalSwapEngine.runORT(session, [targetName: (target, [1, 3, 128, 128]), sourceName: (latent, [1, 512])],
                                       outputs: [outputName])[outputName]!
        }
    }

    enum EngineError: LocalizedError {
        case badEmap([Int]), noSourceFace, missingOutput(String)
        var errorDescription: String? {
            switch self {
            case .badEmap(let d): return "inswapper file looks wrong (emap dims \(d))."
            case .noSourceFace: return "No face found in the source photo."
            case .missingOutput(let n): return "Model output \(n) missing."
            }
        }
    }
}
