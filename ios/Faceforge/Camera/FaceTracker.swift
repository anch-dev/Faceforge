import CoreImage
import FaceforgeCore
import Vision

/// Per-frame face box from Apple Vision (runs on the Neural Engine in a few
/// ms). Not used for the swap itself, only to keep an already-swapped face on
/// the head between swaps, and to follow the face between SCRFD detections.
final class FaceTracker {
    private let request: VNDetectFaceRectanglesRequest = {
        let r = VNDetectFaceRectanglesRequest()
        r.revision = VNDetectFaceRectanglesRequestRevision3
        return r
    }()
    private(set) var lastMs = 0.0
    private let failures = ErrorThrottle()

    /// Largest face, in top-left frame pixels; nil if none.
    func box(in frame: CIImage) -> Box? {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { lastMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 }
        do {
            try VNImageRequestHandler(ciImage: frame, options: [:]).perform([request])
        } catch {
            if failures.shouldLog() { Log.warn("Vision face tracker failed (#\(failures.total)): \(error)", "track") }
            return nil
        }
        guard let best = request.results?.max(by: { $0.boundingBox.width * $0.boundingBox.height
                                                    < $1.boundingBox.width * $1.boundingBox.height }) else { return nil }
        let b = best.boundingBox
        return Box.fromNormalizedBottomLeft(x: b.minX, y: b.minY, w: b.width, h: b.height,
                                            frameWidth: frame.extent.width, frameHeight: frame.extent.height)
    }
}
