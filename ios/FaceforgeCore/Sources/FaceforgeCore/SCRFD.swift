import Foundation

public struct DetectedFace: Equatable {
    public var bbox: [Float]      // x1, y1, x2, y2 (frame pixels)
    public var score: Float
    public var kps: [Point2]      // 5 landmarks (frame pixels)

    public init(bbox: [Float], score: Float, kps: [Point2]) {
        self.bbox = bbox; self.score = score; self.kps = kps
    }

    public var area: Float { max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]) }
}

/// SCRFD (InsightFace det_10g) head decoding. Mirrors faceforge_eval/detector.py,
/// which matches insightface 0.7.3 exactly on real weights.
public enum SCRFD {
    public static let inputMean: Float = 127.5
    public static let inputStd: Float = 128.0

    /// `outputs` in model order: scores x3, bbox x3, kps x3 (flattened).
    /// `detScale` = model-input pixels per frame pixel (letterbox scale).
    public static func decode(outputs: [[Float]], inputSize: Int, strides: [Int] = [8, 16, 32],
                              numAnchors: Int = 2, threshold: Float = 0.5,
                              nmsThreshold: Float = 0.4, detScale: Float = 1.0) -> [DetectedFace] {
        let fmc = strides.count
        precondition(outputs.count == fmc * 3, "expected \(fmc * 3) SCRFD outputs")
        var cands: [DetectedFace] = []
        for (idx, stride) in strides.enumerated() {
            let scores = outputs[idx], boxes = outputs[idx + fmc], kps = outputs[idx + 2 * fmc]
            let h = inputSize / stride, w = inputSize / stride
            let fs = Float(stride)
            for i in 0..<(h * w * numAnchors) where scores[i] >= threshold {
                let cell = i / numAnchors
                let cx = Float(cell % w) * fs, cy = Float(cell / w) * fs
                let b = [cx - boxes[i * 4] * fs, cy - boxes[i * 4 + 1] * fs,
                         cx + boxes[i * 4 + 2] * fs, cy + boxes[i * 4 + 3] * fs].map { $0 / detScale }
                var pts: [Point2] = []
                for k in 0..<5 {
                    pts.append(Point2(x: Double((cx + kps[i * 10 + 2 * k] * fs) / detScale),
                                      y: Double((cy + kps[i * 10 + 2 * k + 1] * fs) / detScale)))
                }
                cands.append(DetectedFace(bbox: b, score: scores[i], kps: pts))
            }
        }
        // Stable sort, descending score (ties keep decode order, like numpy argsort[::-1] on unique scores).
        cands.sort { $0.score > $1.score }
        return nms(cands, threshold: nmsThreshold)
    }

    public static func nms(_ sorted: [DetectedFace], threshold: Float) -> [DetectedFace] {
        var keep: [DetectedFace] = []
        var suppressed = [Bool](repeating: false, count: sorted.count)
        for i in sorted.indices where !suppressed[i] {
            keep.append(sorted[i])
            let a = sorted[i].bbox
            let areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
            for j in (i + 1)..<sorted.count where !suppressed[j] {
                let b = sorted[j].bbox
                let iw = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
                let ih = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
                let inter = iw * ih
                let areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
                if inter / (areaA + areaB - inter) > threshold { suppressed[j] = true }
            }
        }
        return keep
    }

    /// Letterbox geometry used to prepare the detector input: the frame is
    /// resized to fit `size` keeping aspect ratio and placed at the top-left.
    public static func letterbox(frameWidth: Int, frameHeight: Int, size: Int) -> (width: Int, height: Int, scale: Float) {
        let ratio = Float(frameHeight) / Float(frameWidth)
        if ratio > 1 {
            let nh = size, nw = Int(Float(size) / ratio)
            return (nw, nh, Float(nh) / Float(frameHeight))
        } else {
            let nw = size, nh = Int(Float(size) * ratio)
            return (nw, nh, Float(nh) / Float(frameHeight))
        }
    }
}
