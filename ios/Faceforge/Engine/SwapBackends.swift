import CoreML
import FaceforgeCore
import Foundation

/// Runs inswapper_128 once: target crop [1,3,128,128] (RGB 0...1) + latent
/// [1,512] -> swapped crop [1,3,128,128] (RGB 0...1).
protocol SwapRunner: AnyObject {
    var label: String { get }
    func run(target: [Float], latent: [Float]) throws -> [Float]
}

/// Native Core ML (.mlmodelc compiled on this iPhone from an fp16
/// .mlpackage the user made from their own weights). Core ML schedules it on
/// the Neural Engine directly: no ONNX Runtime graph partitioning involved.
final class NativeCoreMLSwapper: SwapRunner {
    private let model: MLModel
    private let targetName: String, sourceName: String, outputName: String
    private let targetArray: MLMultiArray, sourceArray: MLMultiArray
    let label: String

    init(compiledModel url: URL, units: MLComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        let t0 = CFAbsoluteTimeGetCurrent()
        let m = try MLModel(contentsOf: url, configuration: cfg)
        model = m
        let ins = m.modelDescription.inputDescriptionsByName
        guard let out = m.modelDescription.outputDescriptionsByName.keys.sorted().first else {
            throw NativeError.badModel("no outputs")
        }
        targetName = ins["target"] != nil ? "target" : (ins.keys.sorted().first ?? "target")
        sourceName = ins["source"] != nil ? "source" : (ins.keys.sorted().last ?? "source")
        outputName = out
        targetArray = try MLMultiArray(shape: [1, 3, 128, 128], dataType: .float32)
        sourceArray = try MLMultiArray(shape: [1, 512], dataType: .float32)
        label = "Core ML native fp16 (\(Self.name(units)))"
        Log.info("Native Core ML swapper loaded in " + String(format: "%.1f s", CFAbsoluteTimeGetCurrent() - t0)
                 + " · inputs \(ins.keys.sorted()) · output \(out) · units \(Self.name(units))", "engine")
    }

    func run(target: [Float], latent: [Float]) throws -> [Float] {
        try Self.fill(targetArray, target)
        try Self.fill(sourceArray, latent)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            targetName: MLFeatureValue(multiArray: targetArray),
            sourceName: MLFeatureValue(multiArray: sourceArray),
        ])
        let result = try model.prediction(from: provider)
        guard let arr = result.featureValue(for: outputName)?.multiArrayValue else {
            throw NativeError.badModel("output \(outputName) missing")
        }
        return Self.floats(arr)
    }

    private static func fill(_ a: MLMultiArray, _ v: [Float]) throws {
        guard a.count == v.count else { throw NativeError.badModel("input size \(v.count) != \(a.count)") }
        a.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            v.withUnsafeBufferPointer { ptr.baseAddress!.update(from: $0.baseAddress!, count: v.count) }
        }
    }

    static func floats(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
        let dense = Tensor.contiguousCount(shape: shape, strides: strides)
        switch a.dataType {
        case .float32:
            return a.withUnsafeBufferPointer(ofType: Float.self) { p in
                if let n = dense { return Array(p.prefix(n)) }
                return Tensor.gather(shape: shape, strides: strides) { p[$0] }
            }
        case .float16:
            return a.withUnsafeBufferPointer(ofType: Float16.self) { p in
                if let n = dense { return p.prefix(n).map { Float($0) } }
                return Tensor.gather(shape: shape, strides: strides) { Float(p[$0]) }
            }
        default:
            return (0..<a.count).map { a[$0].floatValue }
        }
    }

    static func name(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "GPU"
        case .cpuAndNeuralEngine: return "Neural Engine"
        default: return "all"
        }
    }

    enum NativeError: LocalizedError {
        case badModel(String)
        var errorDescription: String? {
            if case .badModel(let m) = self { return "Fast swap model looks wrong: \(m)" }
            return nil
        }
    }
}

extension ComputeTarget {
    var mlUnits: MLComputeUnits {
        switch self {
        case .neuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        case .gpu: return .cpuAndGPU
        }
    }
}
