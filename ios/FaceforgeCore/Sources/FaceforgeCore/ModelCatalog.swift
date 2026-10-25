import Foundation

/// Default models, fetched on the device from their publisher (InsightFace's
/// GitHub releases). Faceforge never ships weights. Must match
/// faceforge_eval/models.py (enforced by tests/test_ios_sources.py).
public enum ModelCatalog {
    public static let license =
        "InsightFace models are released for non-commercial research purposes only."
    public static let licenseURL = URL(string: "https://github.com/deepinsight/insightface#license")!

    public struct Asset {
        public let name: String
        public let url: URL
        public let size: Int64
        public let sha256: String
    }

    public static let inswapper = Asset(
        name: "inswapper_128.onnx",
        url: URL(string: "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx")!,
        size: 554253681,
        sha256: "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af")

    public static let buffaloL = Asset(
        name: "buffalo_l.zip",
        url: URL(string: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")!,
        size: 288621354,
        sha256: "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f")

    /// Members extracted from buffalo_l.zip: (file, size, sha256).
    public static let buffaloMembers: [(String, Int64, String)] = [
        ("det_10g.onnx", 16923827, "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
        ("w600k_r50.onnx", 174383860, "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
    ]

    public static var totalDownloadBytes: Int64 { inswapper.size + buffaloL.size }
}
