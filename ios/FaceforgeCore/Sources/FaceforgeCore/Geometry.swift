import Foundation

/// 2x3 affine transform in top-left-origin pixel coordinates:
///   x' = a*x + b*y + tx
///   y' = c*x + d*y + ty
public struct Affine2D: Equatable {
    public var a, b, tx, c, d, ty: Double

    public init(a: Double, b: Double, tx: Double, c: Double, d: Double, ty: Double) {
        self.a = a; self.b = b; self.tx = tx; self.c = c; self.d = d; self.ty = ty
    }

    public static let identity = Affine2D(a: 1, b: 0, tx: 0, c: 0, d: 1, ty: 0)

    public func apply(_ p: Point2) -> Point2 {
        Point2(x: a * p.x + b * p.y + tx, y: c * p.x + d * p.y + ty)
    }

    public var inverted: Affine2D {
        let det = a * d - b * c
        let ia = d / det, ib = -b / det, ic = -c / det, id = a / det
        return Affine2D(a: ia, b: ib, tx: -(ia * tx + ib * ty),
                        c: ic, d: id, ty: -(ic * tx + id * ty))
    }

    /// Row-major [[a, b, tx], [c, d, ty]].
    public var rows: [[Double]] { [[a, b, tx], [c, d, ty]] }

    /// Uniformly scale the output space (e.g. 128-px crop -> 256-px crop).
    public func scaled(by s: Double) -> Affine2D {
        Affine2D(a: a * s, b: b * s, tx: tx * s, c: c * s, d: d * s, ty: ty * s)
    }
}

public struct Point2: Equatable {
    public var x: Double
    public var y: Double
    public init(x: Double, y: Double) { self.x = x; self.y = y }
}

public enum FaceAlign {
    /// ArcFace 112x112 reference landmarks (eyes, nose, mouth corners).
    public static let arcfaceDst: [Point2] = [
        Point2(x: 38.2946, y: 51.6963), Point2(x: 73.5318, y: 51.5014), Point2(x: 56.0252, y: 71.7366),
        Point2(x: 41.5493, y: 92.3655), Point2(x: 70.7299, y: 92.2041),
    ]

    /// Least-squares similarity transform (rotation + uniform scale +
    /// translation) mapping src onto dst. Closed form; verified equal to
    /// Umeyama / skimage SimilarityTransform to 1e-12 in the Python tests.
    public static func similarity(from src: [Point2], to dst: [Point2]) -> Affine2D {
        precondition(src.count == dst.count && src.count >= 2)
        let n = Double(src.count)
        let msx = src.map(\.x).reduce(0, +) / n, msy = src.map(\.y).reduce(0, +) / n
        let mdx = dst.map(\.x).reduce(0, +) / n, mdy = dst.map(\.y).reduce(0, +) / n
        var num1 = 0.0, num2 = 0.0, den = 0.0
        for (s, t) in zip(src, dst) {
            let sx = s.x - msx, sy = s.y - msy, dx = t.x - mdx, dy = t.y - mdy
            num1 += sx * dx + sy * dy
            num2 += sx * dy - sy * dx
            den += sx * sx + sy * sy
        }
        let a = num1 / den, b = num2 / den
        return Affine2D(a: a, b: -b, tx: mdx - (a * msx - b * msy),
                        c: b, d: a, ty: mdy - (b * msx + a * msy))
    }

    /// Transform mapping frame landmarks onto the aligned crop of `size` px.
    /// Sizes divisible by 112 scale the template; otherwise it is scaled for
    /// 128 and shifted 8 px (per 128) in x — InsightFace's convention, used by
    /// inswapper_128 (128) and FaceFusion's 'arcface_128' template (256).
    public static func estimateNorm(_ kps: [Point2], size: Int) -> Affine2D {
        let ratio: Double, diffX: Double
        if size % 112 == 0 {
            ratio = Double(size) / 112.0; diffX = 0
        } else {
            precondition(size % 128 == 0, "unsupported crop size \(size)")
            ratio = Double(size) / 128.0; diffX = 8.0 * ratio
        }
        let dst = arcfaceDst.map { Point2(x: $0.x * ratio + diffX, y: $0.y * ratio) }
        return similarity(from: kps, to: dst)
    }
}
