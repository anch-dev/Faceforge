import Foundation

/// Axis-aligned box in top-left-origin frame pixels.
public struct Box: Equatable {
    public var x, y, w, h: Double
    public init(x: Double, y: Double, w: Double, h: Double) { self.x = x; self.y = y; self.w = w; self.h = h }
    public var cx: Double { x + w / 2 }
    public var cy: Double { y + h / 2 }

    /// From a Vision-style normalized rect (origin bottom-left, 0...1).
    public static func fromNormalizedBottomLeft(x: Double, y: Double, w: Double, h: Double,
                                                frameWidth: Double, frameHeight: Double) -> Box {
        Box(x: x * frameWidth, y: (1 - y - h) * frameHeight, w: w * frameWidth, h: h * frameHeight)
    }
}

public extension Affine2D {
    /// `self` after `first`: p -> self(first(p)).
    func composed(after first: Affine2D) -> Affine2D {
        Affine2D(a: a * first.a + b * first.c, b: a * first.b + b * first.d, tx: a * first.tx + b * first.ty + tx,
                 c: c * first.a + d * first.c, d: c * first.b + d * first.d, ty: c * first.tx + d * first.ty + ty)
    }
}

/// Cheap face following between full detections / swaps.
///
/// A light per-frame tracker (Vision face rectangles) gives a box on every
/// camera frame. The motion between two boxes is modelled as uniform scale +
/// translation, which is enough to keep an already-swapped face glued to the
/// head for the few frames until the next swap lands.
public enum FaceTrack {
    /// Image-space map taking content at `old` to where it is at `new`.
    /// Identity for degenerate boxes or implausible jumps (tracker glitch).
    public static func motion(from old: Box, to new: Box, maxScaleChange: Double = 1.6) -> Affine2D {
        guard old.w > 1, old.h > 1, new.w > 1, new.h > 1 else { return .identity }
        let s = ((new.w * new.h) / (old.w * old.h)).squareRoot()
        guard s < maxScaleChange, s > 1 / maxScaleChange else { return .identity }
        return Affine2D(a: s, b: 0, tx: new.cx - s * old.cx, c: 0, d: s, ty: new.cy - s * old.cy)
    }

    /// Frame->crop transform for a crop taken when the face was at `old`,
    /// re-targeted to the face now at `new` (paste the old swap at the new spot).
    public static func retarget(_ cropTransform: Affine2D, from old: Box, to new: Box) -> Affine2D {
        cropTransform.composed(after: motion(from: old, to: new).inverted)
    }

    public static func move(_ kps: [Point2], from old: Box, to new: Box) -> [Point2] {
        let m = motion(from: old, to: new)
        return kps.map { m.apply($0) }
    }
}
