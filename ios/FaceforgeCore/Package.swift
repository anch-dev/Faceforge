// swift-tools-version: 5.9
// Pure-Swift core of the Faceforge iOS app: geometry, detector decoding,
// ONNX initializer reading, offload protocol, stats. No UIKit / ONNX Runtime,
// so `swift test` runs on a macOS host against tests/fixtures (golden values
// produced by the verified Python reference).
import PackageDescription

let package = Package(
    name: "FaceforgeCore",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "FaceforgeCore", targets: ["FaceforgeCore"])],
    targets: [
        .target(name: "FaceforgeCore"),
        .testTarget(name: "FaceforgeCoreTests", dependencies: ["FaceforgeCore"]),
    ]
)
