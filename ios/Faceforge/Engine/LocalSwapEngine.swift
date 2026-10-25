import CoreImage
import FaceforgeCore
import Foundation
import OnnxRuntimeBindings

/// On-device pipeline: SCRFD detect -> ArcFace embed (source, once) ->
/// inswapper_128 swap -> composite. Runs ONNX Runtime with the Core ML
/// execution provider (ML Program, Neural Engine where possible).
///
/// ONNX Runtime >= 1.26 runs inswapper's reflect-Pad ops inside Core ML
/// (verified in ORT source); older builds still work but bounce those ops to
/// the CPU. `backendSummary` reports which runtime is in use.
final class LocalSwapEngine {
    struct Timing { var detectMs = 0.0, swapMs = 0.0, compositeMs = 0.0 }

    private let env: ORTEnv
    private let detector: ORTSession
    private let recognizer: ORTSession
    private let swapper: ORTSession
    private let detInput: String, detOutputs: [String]
    private let recInput: String, recOutput: String
    private let swapTarget: String, swapSource: String, swapOutput: String
    private let emap: [Float]
    private let ops = ImageOps()
    let backendSummary: String

    var detSize = 320
    var detectEvery = 2
    private var frameIndex = 0
    private var cachedFace: DetectedFace?
    private(set) var sourceLatent: [Float]?

    init(models: ModelStore.Paths, useCoreML: Bool, cacheDir: URL) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        self.env = env

        /// Try Core ML with a persistent compiled-model cache (avoids re-compiling
        /// the 554 MB swapper every launch), then Core ML without the cache
        /// (older ORT), then plain CPU. Returns the session and whether Core ML is on.
        func makeSession(_ path: URL, coreML: Bool) throws -> (ORTSession, Bool) {
            let base: [String: String] = ["ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                                          "RequireStaticInputShapes": "0"]
            var attempts: [[String: String]?] = []
            if coreML && ORTIsCoreMLExecutionProviderAvailable() {
                var cached = base
                cached["ModelCacheDirectory"] = cacheDir.path
                attempts = [cached, base]
            }
            attempts.append(nil)  // CPU
            var lastError: Error?
            for cfg in attempts {
                do {
                    let o = try ORTSessionOptions()
                    try o.setGraphOptimizationLevel(.all)
                    if let cfg = cfg { try o.appendCoreMLExecutionProvider(withOptionsV2: cfg) }
                    return (try ORTSession(env: env, modelPath: path.path, sessionOptions: o), cfg != nil)
                } catch {
                    lastError = error
                }
            }
            throw lastError ?? EngineError.missingOutput("session")
        }

        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let det = try makeSession(models.detector, coreML: useCoreML).0
        let rec = try makeSession(models.recognizer, coreML: false).0
        let (sw, coreMLActive) = try makeSession(models.swapper, coreML: useCoreML)

        // Read names from locals: no access to self before every property is set.
        let detOuts = try det.outputNames()
        let detIn = try det.inputNames()[0]
        let recIn = try rec.inputNames()[0]
        let recOut = try rec.outputNames()[0]
        let swapInputs = try sw.inputNames()
        let swapOut = try sw.outputNames()[0]
        detector = det
        recognizer = rec
        swapper = sw
        detInput = detIn
        detOutputs = detOuts
        recInput = recIn
        recOutput = recOut
        swapTarget = swapInputs.first { $0 == "target" } ?? swapInputs[0]
        swapSource = swapInputs.first { $0 == "source" } ?? swapInputs[1]
        swapOutput = swapOut

        let em = try OnnxInitializerReader.lastInitializer(fileURL: models.swapper)
        guard em.dims == [512, 512] else { throw EngineError.badEmap(em.dims) }
        emap = em.floats

        backendSummary = "ONNX Runtime \(ORTVersion() ?? "?") · " + (coreMLActive ? "Core ML (Neural Engine/GPU)" : "CPU")
    }

    // MARK: - stages

    func detect(_ frame: CIImage) throws -> [DetectedFace] {
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
        guard let best = faces.max(by: { $0.area < $1.area }) else { throw EngineError.noSourceFace }
        sourceLatent = Tensor.inswapperLatent(embedding: try embed(image, face: best), emap: emap)
        cachedFace = nil
        return faces.count
    }

    /// Swap the main face in `frame`. Returns the composited frame, or the
    /// input unchanged if no face / no source.
    func process(_ frame: CIImage, timing: inout Timing) throws -> (CIImage, Bool) {
        guard let latent = sourceLatent else { return (frame, false) }
        if frameIndex % max(1, detectEvery) == 0 || cachedFace == nil {
            let t0 = CFAbsoluteTimeGetCurrent()
            cachedFace = try detect(frame).max(by: { $0.area < $1.area })
            timing.detectMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }
        frameIndex += 1
        guard let face = cachedFace else { return (frame, false) }

        let t1 = CFAbsoluteTimeGetCurrent()
        let m = FaceAlign.estimateNorm(face.kps, size: 128)
        let target = try ops.alignedTensor(frame: frame, transform: m, size: 128, mean: 0, std: 255)
        let out = try run(swapper, [swapTarget: (target, [1, 3, 128, 128]), swapSource: (latent, [1, 512])],
                          outputs: [swapOutput])[swapOutput]!
        timing.swapMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

        let t2 = CFAbsoluteTimeGetCurrent()
        let fake = try ops.image(fromBGRA: Tensor.bgra(fromCHWRGB: out, width: 128, height: 128), width: 128, height: 128)
        let result = ops.paste(fake: fake, onto: frame, transform: m, size: 128)
        timing.compositeMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        return (result, true)
    }

    // MARK: - ORT helpers

    private func run(_ s: ORTSession, _ inputs: [String: ([Float], [Int])], outputs: [String]) throws -> [String: [Float]] {
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
