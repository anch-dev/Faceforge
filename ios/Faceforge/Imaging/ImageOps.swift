import CoreImage
import CoreVideo
import FaceforgeCore
import ImageIO
import Metal

/// Pixel plumbing between Core Image (GPU) and model tensors.
///
/// Coordinate conventions: FaceforgeCore works in top-left-origin pixel
/// coordinates with OpenCV's pixel-centre convention (matching the verified
/// Python reference). Core Image uses bottom-left origin and continuous
/// coordinates, so every affine passes through `ciTransform(...)`.
final class ImageOps {
    /// No colour management: model inputs must be the raw camera values.
    let ctx: CIContext
    private var pools: [Int: CVPixelBufferPool] = [:]
    private var masks: [Int: CIImage] = [:]

    init() {
        let opts: [CIContextOption: Any] = [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(),
                                            .cacheIntermediates: false]
        if let dev = MTLCreateSystemDefaultDevice() {
            ctx = CIContext(mtlDevice: dev, options: opts)
        } else {
            ctx = CIContext(options: opts)
        }
    }

    /// Affine (top-left, OpenCV pixel centres) frame->crop, as a Core Image
    /// transform from frame CI space to crop CI space.
    static func ciTransform(_ m: Affine2D, frameHeight: CGFloat, cropHeight: CGFloat) -> CGAffineTransform {
        // OpenCV maps integer pixel centres; CI samples at +0.5. Shift into continuous coords.
        let tx = m.tx - 0.5 * (m.a + m.b) + 0.5
        let ty = m.ty - 0.5 * (m.c + m.d) + 0.5
        let flipFrame = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: frameHeight)
        let affine = CGAffineTransform(a: m.a, b: m.c, c: m.b, d: m.d, tx: tx, ty: ty)
        let flipCrop = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: cropHeight)
        return flipFrame.concatenating(affine).concatenating(flipCrop)
    }

    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let key = width << 16 | height
        if pools[key] == nil {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
            pools[key] = pool
        }
        guard let pool = pools[key] else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        return pb
    }

    /// Render `image` (already in target CI space) into a width x height
    /// BGRA buffer and hand its bytes (row 0 = top) to `body`.
    func withBGRA<T>(_ image: CIImage, width: Int, height: Int,
                     _ body: (UnsafePointer<UInt8>, Int) throws -> T) throws -> T {
        guard let pb = pixelBuffer(width: width, height: height) else { throw ImageError.buffer }
        // Clear first: areas outside `image` must be zero, like cv2 borderValue=0.
        let clear = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        ctx.render(image.composited(over: clear), to: pb, bounds: CGRect(x: 0, y: 0, width: width, height: height),
                   colorSpace: nil)
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { throw ImageError.buffer }
        return try body(base.assumingMemoryBound(to: UInt8.self), CVPixelBufferGetBytesPerRow(pb))
    }

    /// Aligned crop of `size` px as an NCHW RGB tensor ((v - mean) / std).
    func alignedTensor(frame: CIImage, transform m: Affine2D, size: Int, mean: Float, std: Float) throws -> [Float] {
        let t = ImageOps.ciTransform(m, frameHeight: frame.extent.height, cropHeight: CGFloat(size))
        return try withBGRA(frame.transformed(by: t), width: size, height: size) { px, rowBytes in
            Tensor.chwRGB(fromBGRA: px, width: size, height: size, rowBytes: rowBytes, mean: mean, std: std)
        }
    }

    /// Letterboxed detector input (top-left aligned, zero padded), as in insightface.
    func detectorTensor(frame: CIImage, size: Int) throws -> (tensor: [Float], scale: Float) {
        let lb = SCRFD.letterbox(frameWidth: Int(frame.extent.width), frameHeight: Int(frame.extent.height), size: size)
        let sx = CGFloat(lb.width) / frame.extent.width, sy = CGFloat(lb.height) / frame.extent.height
        let placed = frame
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .transformed(by: CGAffineTransform(translationX: 0, y: CGFloat(size - lb.height)))
        let t = try withBGRA(placed, width: size, height: size) { px, rowBytes in
            Tensor.chwRGB(fromBGRA: px, width: size, height: size, rowBytes: rowBytes,
                          mean: SCRFD.inputMean, std: SCRFD.inputStd)
        }
        return (t, lb.scale)
    }

    /// CIImage from BGRA bytes (row 0 = top).
    func image(fromBGRA bytes: [UInt8], width: Int, height: Int) throws -> CIImage {
        guard let pb = pixelBuffer(width: width, height: height) else { throw ImageError.buffer }
        CVPixelBufferLockBaseAddress(pb, [])
        let dst = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        bytes.withUnsafeBufferPointer { src in
            for y in 0..<height {
                (dst + y * rowBytes).update(from: src.baseAddress! + y * width * 4, count: width * 4)
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return CIImage(cvPixelBuffer: pb)
    }

    /// Feathered oval over the face for a crop of `size` px, in ArcFace-template
    /// proportions (eyes ~46% down, mouth ~83%). The crop's corners hold
    /// background from the frame the swap was made from; a square mask pasted
    /// them back, which showed as a box that trailed behind a moving head
    /// (0.1.3). The oval fades out before reaching them.
    func softMask(size: Int) -> CIImage {
        if let m = masks[size] { return m }
        let s = CGFloat(size)
        let crop = CGRect(x: 0, y: 0, width: s, height: s)
        let gradient = CIFilter(name: "CIRadialGradient", parameters: [
            "inputCenter": CIVector(x: 0, y: 0),
            "inputRadius0": s * 0.28,          // fully swapped inside
            "inputRadius1": s * 0.40,          // fully original outside
            "inputColor0": CIColor.white,
            "inputColor1": CIColor.black,
        ])!.outputImage!
        // Stretch to an oval, centre it at 55% down (CI y is bottom-up).
        let oval = gradient
            .transformed(by: CGAffineTransform(scaleX: 1, y: 1.28))
            .transformed(by: CGAffineTransform(translationX: s * 0.5, y: s * (1 - 0.55)))
        let m = oval.composited(over: CIImage(color: .black)).cropped(to: crop)
        masks[size] = m
        return m
    }

    /// Paste an aligned fake face back over `frame`.
    func paste(fake: CIImage, onto frame: CIImage, transform m: Affine2D, size: Int) -> CIImage {
        let t = ImageOps.ciTransform(m, frameHeight: frame.extent.height, cropHeight: CGFloat(size)).inverted()
        let warpedFake = fake.clampedToExtent().transformed(by: t)
        let warpedMask = softMask(size: size).transformed(by: t)
        let f = CIFilter(name: "CIBlendWithMask")!
        f.setValue(warpedFake, forKey: kCIInputImageKey)
        f.setValue(frame, forKey: kCIInputBackgroundImageKey)
        f.setValue(warpedMask, forKey: kCIInputMaskImageKey)
        return (f.outputImage ?? frame).cropped(to: frame.extent)
    }

    func jpeg(_ image: CIImage, quality: Double, maxSide: CGFloat) -> Data? {
        let scale = min(1, maxSide / max(image.extent.width, image.extent.height))
        let img = scale < 1 ? image.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) : image
        return ctx.jpegRepresentation(of: img, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality])
    }

    enum ImageError: Error { case buffer }
}
