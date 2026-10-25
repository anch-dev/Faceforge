import Foundation

/// Conversions between 8-bit BGRA pixel buffers (what the camera / Core Image
/// produce) and NCHW float tensors (what the models take).
public enum Tensor {
    /// BGRA bytes (row-major, `rowBytes` per row) -> [1,3,h,w] float, RGB
    /// channel order, value = (pixel - mean) / std. Matches
    /// cv2.dnn.blobFromImage(img, 1/std, size, (mean,)*3, swapRB=True).
    public static func chwRGB(fromBGRA src: UnsafePointer<UInt8>, width w: Int, height h: Int, rowBytes: Int,
                              mean: Float, std: Float) -> [Float] {
        var out = [Float](repeating: 0, count: 3 * w * h)
        let plane = w * h
        let inv = 1 / std
        out.withUnsafeMutableBufferPointer { o in
            for y in 0..<h {
                let row = src + y * rowBytes
                for x in 0..<w {
                    let px = row + x * 4
                    let i = y * w + x
                    o[i] = (Float(px[2]) - mean) * inv            // R
                    o[plane + i] = (Float(px[1]) - mean) * inv    // G
                    o[2 * plane + i] = (Float(px[0]) - mean) * inv // B
                }
            }
        }
        return out
    }

    /// [1,3,h,w] RGB float in [0,1] -> BGRA bytes (alpha 255), clamped.
    public static func bgra(fromCHWRGB t: [Float], width w: Int, height h: Int) -> [UInt8] {
        var out = [UInt8](repeating: 255, count: 4 * w * h)
        let plane = w * h
        @inline(__always) func q(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v * 255))) }
        for i in 0..<plane {
            out[i * 4 + 0] = q(t[2 * plane + i])  // B
            out[i * 4 + 1] = q(t[plane + i])      // G
            out[i * 4 + 2] = q(t[i])              // R
        }
        return out
    }

    public static func l2normalize(_ v: [Float]) -> [Float] {
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return n > 0 ? v.map { $0 / n } : v
    }

    /// row vector (1 x n) times matrix (n x m, row-major) -> (1 x m).
    public static func matvec(_ v: [Float], _ m: [Float], cols: Int) -> [Float] {
        let n = v.count
        precondition(m.count == n * cols)
        var out = [Float](repeating: 0, count: cols)
        for i in 0..<n {
            let vi = v[i]
            if vi == 0 { continue }
            let rowStart = i * cols
            for j in 0..<cols { out[j] += vi * m[rowStart + j] }
        }
        return out
    }

    /// inswapper latent: normalize(normalize(embedding) . emap).
    public static func inswapperLatent(embedding: [Float], emap: [Float]) -> [Float] {
        l2normalize(matvec(l2normalize(embedding), emap, cols: embedding.count))
    }

    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrt(na) * sqrt(nb))
    }
}

public extension Tensor {
    /// Element count if `strides` describe a dense row-major layout of `shape`
    /// (so the buffer can be copied in one go), else nil.
    static func contiguousCount(shape: [Int], strides: [Int]) -> Int? {
        guard shape.count == strides.count else { return nil }
        var expected = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            if shape[i] != 1 && strides[i] != expected { return nil }
            expected *= shape[i]
        }
        return expected
    }

    /// Row-major copy of a strided tensor; `read(offset)` returns the element
    /// at a storage offset (e.g. an MLMultiArray with padded rows).
    static func gather(shape: [Int], strides: [Int], read: (Int) -> Float) -> [Float] {
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        var idx = [Int](repeating: 0, count: shape.count)
        for n in 0..<count {
            var off = 0
            for k in 0..<shape.count { off += idx[k] * strides[k] }
            out[n] = read(off)
            var k = shape.count - 1
            while k >= 0 {
                idx[k] += 1
                if idx[k] < shape[k] { break }
                idx[k] = 0
                k -= 1
            }
        }
        return out
    }
}
