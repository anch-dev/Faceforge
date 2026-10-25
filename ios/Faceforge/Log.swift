import Foundation
import UIKit

/// File logging, visible in the Files app: On My iPhone > Faceforge > Logs.
///
/// One file per app launch (the newest 10 are kept). stdout and stderr are
/// redirected into the same file, so anything ONNX Runtime or the system
/// prints to them lands next to Faceforge's own lines.
enum Log {
    static let verboseKey = "verboseLogging"
    /// Verbose: ONNX Runtime info-level logs (incl. which nodes run on Core ML),
    /// per-second performance lines, every error. Off: warnings and errors only.
    static var verbose: Bool {
        get { UserDefaults.standard.object(forKey: verboseKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: verboseKey) }
    }

    static let dir: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Logs", isDirectory: true)
    private(set) static var fileURL: URL?
    private static let queue = DispatchQueue(label: "faceforge.log")
    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static func start() {
        let fm = FileManager.default
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        prune(keep: 9)
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        let url = dir.appendingPathComponent("faceforge-\(f.string(from: Date())).log")
        fm.createFile(atPath: url.path, contents: nil)
        fileURL = url
        // Redirect the C streams so library output is captured too.
        freopen(url.path, "a", stderr)
        freopen(url.path, "a", stdout)
        setvbuf(stderr, nil, _IONBF, 0)
        setvbuf(stdout, nil, _IOLBF, 0)

        let b = Bundle.main
        let version = (b.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?") + " ("
            + (b.infoDictionary?["CFBundleVersion"] as? String ?? "?") + ")"
        info("Faceforge \(version) started", "app")
        info("Device \(machineID()) · \(UIDevice.current.systemName) \(UIDevice.current.systemVersion) · "
             + "RAM \(ProcessInfo.processInfo.physicalMemory / 1_048_576) MB · "
             + "thermal \(ProcessInfo.processInfo.thermalState.rawValue) · "
             + "low power \(ProcessInfo.processInfo.isLowPowerModeEnabled)", "app")
        info("Verbose logging \(verbose ? "ON" : "OFF") · log file \(url.lastPathComponent)", "app")
    }

    static func info(_ msg: @autoclosure () -> String, _ cat: String = "app") { write("INFO ", cat, msg()) }
    static func warn(_ msg: @autoclosure () -> String, _ cat: String = "app") { write("WARN ", cat, msg()) }
    static func error(_ msg: @autoclosure () -> String, _ cat: String = "app") { write("ERROR", cat, msg()) }
    static func debug(_ msg: @autoclosure () -> String, _ cat: String = "app") {
        guard verbose else { return }
        write("DEBUG", cat, msg())
    }

    private static func write(_ level: String, _ cat: String, _ msg: String) {
        let line = "\(stamp.string(from: Date())) \(level) [\(cat)] \(msg)\n"
        queue.async { fputs(line, stderr) }
    }

    /// Log files, newest first.
    static func files() -> [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return urls.filter { $0.pathExtension == "log" }.sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    static func clearOld() {
        for u in files() where u != fileURL { try? FileManager.default.removeItem(at: u) }
    }

    private static func prune(keep: Int) {
        for u in files().dropFirst(keep) { try? FileManager.default.removeItem(at: u) }
    }

    /// e.g. "iPhone17,2"
    static func machineID() -> String {
        var sys = utsname()
        uname(&sys)
        return withUnsafeBytes(of: &sys.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
}

/// Logs an error the first few times, then every 100th, so a per-frame
/// failure can't fill the disk.
final class ErrorThrottle {
    private var count = 0
    func shouldLog() -> Bool {
        count += 1
        return count <= 5 || count % 100 == 0
    }
    var total: Int { count }
}
