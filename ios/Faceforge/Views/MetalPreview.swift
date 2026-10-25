import CoreImage
import MetalKit
import SwiftUI

/// Draws the latest processed frame (aspect-fit) with Core Image on Metal.
final class PreviewRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    private lazy var queue = device?.makeCommandQueue()
    private lazy var ctx: CIContext? = device.map { CIContext(mtlDevice: $0) }
    private var latest: CIImage?
    private let lock = NSLock()

    func display(_ image: CIImage) {
        lock.lock()
        latest = image
        lock.unlock()
    }

    func draw(in view: MTKView) {
        lock.lock()
        let img = latest
        lock.unlock()
        guard let img = img, let ctx = ctx, let drawable = view.currentDrawable,
              let cb = queue?.makeCommandBuffer() else { return }
        let ds = view.drawableSize
        let e = img.extent
        let scale = min(ds.width / e.width, ds.height / e.height)
        let scaled = img.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let se = scaled.extent
        let placed = scaled.transformed(by: CGAffineTransform(translationX: (ds.width - se.width) / 2 - se.minX,
                                                               y: (ds.height - se.height) / 2 - se.minY))
        let bounds = CGRect(origin: .zero, size: ds)
        let bg = CIImage(color: .black).cropped(to: bounds)
        ctx.render(placed.composited(over: bg), to: drawable.texture, commandBuffer: cb, bounds: bounds,
                   colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        cb.present(drawable)
        cb.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

struct MetalPreview: UIViewRepresentable {
    let renderer: PreviewRenderer

    func makeUIView(context: Context) -> MTKView {
        let v = MTKView(frame: .zero, device: renderer.device)
        v.framebufferOnly = false          // Core Image writes into the drawable texture
        v.preferredFramesPerSecond = 30
        v.enableSetNeedsDisplay = false
        v.isPaused = false
        v.delegate = renderer
        v.backgroundColor = .black
        return v
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}
