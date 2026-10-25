import CoreML
import CryptoKit
import FaceforgeCore
import Foundation
import ZIPFoundation

/// Downloads the default models on the device, from InsightFace's own GitHub
/// releases (never from Faceforge), and verifies size + SHA-256.
@MainActor
final class ModelStore: NSObject, ObservableObject {
    struct Paths {
        let swapper: URL
        let detector: URL
        let recognizer: URL
    }

    enum State: Equatable {
        case missing
        case downloading(String, Double)   // label, fraction
        case verifying(String)
        case ready
        case failed(String)
    }

    @Published private(set) var state: State = .missing
    /// The optional fast (fp16, Neural Engine) swap model, made on the user's
    /// PC from their own copy of inswapper_128.
    @Published private(set) var fastState: State = .missing

    let dir: URL
    let coreMLCacheDir: URL
    private var observation: NSKeyValueObservation?

    override init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        dir = support.appendingPathComponent("Models", isDirectory: true)
        coreMLCacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CoreMLCache", isDirectory: true)
        super.init()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true      // 560 MB of models: don't push to iCloud
        var d = dir
        try? d.setResourceValues(values)
        state = paths != nil ? .ready : .missing
        fastState = fastModel != nil ? .ready : .missing
    }

    enum FastModel: Equatable {
        case nativeCoreML(URL)   // compiled .mlmodelc
        case onnxFP16(URL)
        var label: String {
            switch self {
            case .nativeCoreML: return "Native Core ML (fp16)"
            case .onnxFP16: return "ONNX fp16 via Core ML"
            }
        }
    }

    var nativeModelURL: URL { dir.appendingPathComponent("inswapper_128_fp16.mlmodelc", isDirectory: true) }
    var fp16OnnxURL: URL { dir.appendingPathComponent("inswapper_128_fp16.onnx") }

    /// Best installed fast model: native Core ML first, then fp16 ONNX.
    var fastModel: FastModel? {
        let fm = FileManager.default
        if fm.fileExists(atPath: nativeModelURL.appendingPathComponent("coremldata.bin").path) { return .nativeCoreML(nativeModelURL) }
        if fm.fileExists(atPath: fp16OnnxURL.path) { return .onnxFP16(fp16OnnxURL) }
        return nil
    }

    var paths: Paths? {
        let p = Paths(swapper: dir.appendingPathComponent(ModelCatalog.inswapper.name),
                      detector: dir.appendingPathComponent("det_10g.onnx"),
                      recognizer: dir.appendingPathComponent("w600k_r50.onnx"))
        let fm = FileManager.default
        return [p.swapper, p.detector, p.recognizer].allSatisfy { fm.fileExists(atPath: $0.path) } ? p : nil
    }

    /// Call only after the user accepted the licence shown in ModelsView.
    func downloadAll() async {
        do {
            let swapper = dir.appendingPathComponent(ModelCatalog.inswapper.name)
            if !FileManager.default.fileExists(atPath: swapper.path) {
                try await fetch(ModelCatalog.inswapper, to: swapper, label: "Face swap model")
            }
            if !FileManager.default.fileExists(atPath: dir.appendingPathComponent("w600k_r50.onnx").path) {
                let zip = dir.appendingPathComponent(ModelCatalog.buffaloL.name)
                try await fetch(ModelCatalog.buffaloL, to: zip, label: "Face detection + recognition pack")
                state = .verifying("Extracting")
                try await Task.detached(priority: .userInitiated) { [dir] in
                    try ModelStore.extractBuffalo(zip: zip, into: dir)
                }.value
                try? FileManager.default.removeItem(at: zip)
            }
            state = paths != nil ? .ready : .failed("Models incomplete after download.")
            Log.info("Models ready: \(paths != nil)", "models")
        } catch {
            state = .failed(error.localizedDescription)
            Log.error("Model download failed: \(error)", "models")
        }
    }

    func deleteAll() {
        try? FileManager.default.removeItem(at: dir)
        try? FileManager.default.removeItem(at: coreMLCacheDir)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        state = .missing
    }

    private func fetch(_ asset: ModelCatalog.Asset, to dest: URL, label: String,
                       publish: ((State) -> Void)? = nil) async throws {
        let set: (State) -> Void = publish ?? { [weak self] in self?.state = $0 }
        set(.downloading(label, 0))
        Log.info("Download start: \(asset.url.absoluteString) (\(asset.size) bytes)", "models")
        var lastDecile = -1
        let t0 = CFAbsoluteTimeGetCurrent()
        let tmp: URL = try await withCheckedThrowingContinuation { cont in
            let task = URLSession.shared.downloadTask(with: asset.url) { url, response, error in
                if let error = error { cont.resume(throwing: error); return }
                guard let url = url, (response as? HTTPURLResponse)?.statusCode == 200 else {
                    cont.resume(throwing: StoreError.http((response as? HTTPURLResponse)?.statusCode ?? -1)); return
                }
                // The temp file is deleted when this handler returns: move it now.
                let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                do { try FileManager.default.moveItem(at: url, to: keep); cont.resume(returning: keep) }
                catch { cont.resume(throwing: error) }
            }
            observation = task.progress.observe(\.fractionCompleted) { [weak self] p, _ in
                let f = p.fractionCompleted
                let decile = Int(f * 10)
                if decile != lastDecile { lastDecile = decile; Log.debug("\(asset.name): \(decile * 10)%", "models") }
                Task { @MainActor in if self != nil { set(.downloading(label, f)) } }
            }
            task.resume()
        }
        observation = nil
        Log.info(String(format: "Downloaded %@ in %.0f s; verifying SHA-256", asset.name, CFAbsoluteTimeGetCurrent() - t0), "models")
        set(.verifying(label))
        let ok = try await Task.detached(priority: .userInitiated) {
            try ModelStore.verify(tmp, size: asset.size, sha256: asset.sha256)
        }.value
        Log.info("\(asset.name) integrity: \(ok ? "OK" : "MISMATCH")", "models")
        guard ok else {
            try? FileManager.default.removeItem(at: tmp)
            throw StoreError.checksum(asset.name)
        }
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
    }

    // MARK: fast model

    func deleteFastModel() {
        try? FileManager.default.removeItem(at: nativeModelURL)
        try? FileManager.default.removeItem(at: fp16OnnxURL)
        try? FileManager.default.removeItem(at: coreMLCacheDir)   // ORT's compiled copy of the fp16 ONNX
        fastState = .missing
        Log.info("Fast swap model removed", "models")
    }

    /// Download from Send-Model-To-iPhone.bat on the PC (same Wi-Fi). Takes the
    /// best file the PC offers (native Core ML package, else fp16 ONNX) and
    /// checks it against the SHA-256 the PC lists.
    func fetchFastModel(host: String, code: String, port: Int = 47801) async {
        let h = host.trimmingCharacters(in: .whitespaces), c = code.trimmingCharacters(in: .whitespaces)
        guard !h.isEmpty, !c.isEmpty,
              let base = URL(string: "http://\(h.contains(":") ? "[\(h)]" : h):\(port)/\(c)/") else {
            fastState = .failed("Enter the PC address (Settings) and the download code.")
            return
        }
        do {
            fastState = .downloading("Asking the PC", 0)
            Log.info("Fast model: index from \(base.absoluteString)", "models")
            let (data, resp) = try await URLSession.shared.data(from: base.appendingPathComponent("index.json"))
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                throw StoreError.http((resp as? HTTPURLResponse)?.statusCode ?? -1)
            }
            let index = try JSONDecoder().decode(FastIndex.self, from: data)
            guard let file = index.files.first else { throw StoreError.missingMember("fast model") }
            Log.info("Fast model: PC offers \(index.files.map { "\($0.name) (\($0.kind), \($0.size) B)" })", "models")
            let asset = ModelCatalog.Asset(name: file.name, url: base.appendingPathComponent(file.name),
                                           size: file.size, sha256: file.sha256)
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(file.name)
            try await fetch(asset, to: tmp, label: "Fast swap model from PC", publish: { [weak self] in self?.fastState = $0 })
            try await installFastModel(from: tmp)
        } catch {
            fastState = .failed(error.localizedDescription)
            Log.error("Fast model download failed: \(error)", "models")
        }
    }

    /// Install from a file: an .mlpackage zip, an .mlpackage folder, or an fp16 .onnx.
    func installFastModel(from url: URL) async throws {
        fastState = .verifying("Installing fast model")
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let fm = FileManager.default
        let ext = url.pathExtension.lowercased()
        do {
            if ext == "onnx" {
                // Same graph as inswapper_128 (its emap is still the last initializer)?
                let em = try OnnxInitializerReader.lastInitializer(fileURL: url)
                guard em.dims == [512, 512] else {
                    throw NativeCoreMLSwapper.NativeError.badModel("not an inswapper_128 file (emap \(em.dims))")
                }
                try? fm.removeItem(at: fp16OnnxURL)
                try fm.copyItem(at: url, to: fp16OnnxURL)
                try? fm.removeItem(at: coreMLCacheDir)
            } else {
                var pkg = url
                var work: URL?
                defer { if let w = work { try? fm.removeItem(at: w) } }
                if ext == "zip" {
                    let w = fm.temporaryDirectory.appendingPathComponent("fastmodel-\(UUID().uuidString)")
                    work = w
                    try await Task.detached(priority: .userInitiated) { try fm.unzipItem(at: url, to: w) }.value
                    guard let found = fm.enumerator(at: w, includingPropertiesForKeys: nil)?
                        .compactMap({ $0 as? URL }).first(where: { $0.pathExtension == "mlpackage" }) else {
                        throw StoreError.missingMember("an .mlpackage in the zip")
                    }
                    pkg = found
                }
                guard pkg.pathExtension == "mlpackage" else { throw StoreError.missingMember(".mlpackage, .zip or .onnx file") }
                Log.info("Compiling \(pkg.lastPathComponent) for this iPhone...", "models")
                let t0 = CFAbsoluteTimeGetCurrent()
                let compiled = try await MLModel.compileModel(at: pkg)
                Log.info(String(format: "Compiled in %.1f s", CFAbsoluteTimeGetCurrent() - t0), "models")
                let m = try MLModel(contentsOf: compiled)
                let ins = m.modelDescription.inputDescriptionsByName
                guard let t = ins["target"]?.multiArrayConstraint?.shape.map(\.intValue), t == [1, 3, 128, 128],
                      let s = ins["source"]?.multiArrayConstraint?.shape.map(\.intValue), s == [1, 512] else {
                    try? fm.removeItem(at: compiled)
                    throw NativeCoreMLSwapper.NativeError.badModel("expected inputs target[1,3,128,128] and source[1,512], got \(ins.keys.sorted())")
                }
                try? fm.removeItem(at: nativeModelURL)
                try fm.moveItem(at: compiled, to: nativeModelURL)
            }
            if fm.fileExists(atPath: url.path), url.path.hasPrefix(fm.temporaryDirectory.path) { try? fm.removeItem(at: url) }
            fastState = .ready
            Log.info("Fast swap model installed: \(fastModel?.label ?? "?")", "models")
        } catch {
            fastState = .failed(error.localizedDescription)
            Log.error("Fast model install failed: \(error)", "models")
            throw error
        }
    }

    private struct FastIndex: Decodable {
        struct File: Decodable { let name: String; let size: Int64; let sha256: String; let kind: String }
        let files: [File]
    }

    nonisolated static func verify(_ url: URL, size: Int64, sha256: String) throws -> Bool {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard (attrs[.size] as? NSNumber)?.int64Value == size else { return false }
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while true {
            let chunk = try h.read(upToCount: 4 << 20) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined() == sha256
    }

    nonisolated static func extractBuffalo(zip: URL, into dir: URL) throws {
        let archive = try Archive(url: zip, accessMode: .read)
        for (name, size, sha) in ModelCatalog.buffaloMembers {
            guard let entry = archive.first(where: { ($0.path as NSString).lastPathComponent == name }) else {
                throw StoreError.missingMember(name)
            }
            let dest = dir.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dest)
            _ = try archive.extract(entry, to: dest)
            guard try verify(dest, size: size, sha256: sha) else {
                try? FileManager.default.removeItem(at: dest)
                throw StoreError.checksum(name)
            }
        }
    }

    enum StoreError: LocalizedError {
        case http(Int), checksum(String), missingMember(String)
        var errorDescription: String? {
            switch self {
            case .http(let c): return "Download failed (HTTP \(c))."
            case .checksum(let n): return "\(n) failed its integrity check. Try again."
            case .missingMember(let n): return "\(n) is missing from the downloaded pack."
            }
        }
    }
}
