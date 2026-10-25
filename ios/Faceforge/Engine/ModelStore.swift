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

    private func fetch(_ asset: ModelCatalog.Asset, to dest: URL, label: String) async throws {
        state = .downloading(label, 0)
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
                Task { @MainActor in self?.state = .downloading(label, f) }
            }
            task.resume()
        }
        observation = nil
        Log.info(String(format: "Downloaded %@ in %.0f s; verifying SHA-256", asset.name, CFAbsoluteTimeGetCurrent() - t0), "models")
        state = .verifying(label)
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
