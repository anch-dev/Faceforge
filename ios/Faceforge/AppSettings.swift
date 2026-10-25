import SwiftUI

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
    }

    /// The source identity photo, stored locally as JPEG.
    static var sourceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("source.jpg")
    }

    static var hasSource: Bool { FileManager.default.fileExists(atPath: sourceURL.path) }
}
