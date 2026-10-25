import SwiftUI

/// Where Core ML may run the swap model. The Neural Engine only runs fp16
/// models; "GPU" is Metal.
enum ComputeTarget: String, CaseIterable, Identifiable {
    case neuralEngine = "Neural Engine"
    case all = "All (Neural Engine + GPU + CPU)"
    case gpu = "GPU (Metal)"
    var id: String { rawValue }
    /// ONNX Runtime Core ML EP `MLComputeUnits` value.
    var ortValue: String {
        switch self {
        case .neuralEngine: return "CPUAndNeuralEngine"
        case .all: return "ALL"
        case .gpu: return "CPUAndGPU"
        }
    }
}

enum ProcessingMode: String, CaseIterable, Identifiable {
    case onDevice = "On iPhone"
    case pc = "PC offload"
    var id: String { rawValue }
}

/// Persisted user settings (UserDefaults), observable by SwiftUI.
final class AppSettings: ObservableObject {
    private static let d = UserDefaults.standard

    @Published var mode: ProcessingMode { didSet { Self.d.set(mode.rawValue, forKey: "mode") } }
    @Published var pcHost: String { didSet { Self.d.set(pcHost, forKey: "pcHost") } }
    @Published var pcPort: Int { didSet { Self.d.set(pcPort, forKey: "pcPort") } }
    @Published var pcToken: String { didSet { Self.d.set(pcToken, forKey: "pcToken") } }
    @Published var detectEvery: Int { didSet { Self.d.set(detectEvery, forKey: "detectEvery") } }
    @Published var resolution: CameraController.Resolution { didSet { Self.d.set(resolution.rawValue, forKey: "resolution") } }
    @Published var useCoreML: Bool { didSet { Self.d.set(useCoreML, forKey: "useCoreML") } }
    @Published var jpegQuality: Double { didSet { Self.d.set(jpegQuality, forKey: "jpegQuality") } }
    @Published var showStats: Bool { didSet { Self.d.set(showStats, forKey: "showStats") } }
    @Published var acceptedModelLicense: Bool { didSet { Self.d.set(acceptedModelLicense, forKey: "acceptedModelLicense") } }
    @Published var computeTarget: ComputeTarget { didSet { Self.d.set(computeTarget.rawValue, forKey: "computeTarget") } }
    /// Show camera frames at full rate and glue the latest swapped face onto
    /// the tracked head, instead of waiting for every swap.
    @Published var smoothPreview: Bool { didSet { Self.d.set(smoothPreview, forKey: "smoothPreview") } }
    @Published var preferFastModel: Bool { didSet { Self.d.set(preferFastModel, forKey: "preferFastModel") } }

    init() {
        let d = Self.d
        mode = ProcessingMode(rawValue: d.string(forKey: "mode") ?? "") ?? .onDevice
        pcHost = d.string(forKey: "pcHost") ?? ""
        pcPort = d.object(forKey: "pcPort") as? Int ?? 47800
        pcToken = d.string(forKey: "pcToken") ?? ""
        detectEvery = d.object(forKey: "detectEvery") as? Int ?? 2
        resolution = CameraController.Resolution(rawValue: d.string(forKey: "resolution") ?? "") ?? .hd
        useCoreML = d.object(forKey: "useCoreML") as? Bool ?? true
        jpegQuality = d.object(forKey: "jpegQuality") as? Double ?? 0.75
        showStats = d.object(forKey: "showStats") as? Bool ?? true
        acceptedModelLicense = d.bool(forKey: "acceptedModelLicense")
        computeTarget = ComputeTarget(rawValue: d.string(forKey: "computeTarget") ?? "") ?? .neuralEngine
        smoothPreview = d.object(forKey: "smoothPreview") as? Bool ?? true
        preferFastModel = d.object(forKey: "preferFastModel") as? Bool ?? true
    }

    /// The source identity photo, stored locally as JPEG.
    static var sourceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("source.jpg")
    }

    static var hasSource: Bool { FileManager.default.fileExists(atPath: sourceURL.path) }
}
