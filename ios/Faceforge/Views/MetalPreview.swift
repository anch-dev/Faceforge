import CoreImage
import MetalKit
import SwiftUI

/// Draws the latest processed frame (aspect-fit) with Core Image on Metal.
///
/// Redraws when a new frame arrives (setNeedsDisplay) instead of relying on
/// MTKView's internal display-link timer, which did not start until the view
/// was re-laid out (preview only appeared after switching tabs, 0.1.2).
final class PreviewRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    private lazy var queue = device?.makeCommandQueue()
    private lazy var ctx: CIContext? = device.map { CIContext(mtlDevice: $0) }
    private var latest: CIImage?
    private let lock = NSLock()
    private var drawPending = false
    weak var view: MTKView?

    func display(_ image: CIImage) {
        lock.lock()
        latest = image
        let schedule = !drawPending
        drawPending = true
        lock.unlock()
        guard schedule else { return }   // coalesce: at most one redraw queued
        DispatchQueue.main.async { [weak self] in self?.view?.setNeedsDisplay() }
    }

    func draw(in view: MTKView) {
        lock.lock()
        let img = latest
        drawPending = false
        lock.unlock()
        guard let img = img, let ctx = ctx, let drawable = view.currentDrawable,
              let cb = queue?.makeCommandBuffer() else { return }
        let ds = view.drawableSize
        guard ds.width > 0, ds.height > 0 else { return }
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

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.setNeedsDisplay()
    }
}

struct MetalPreview: UIViewRepresentable {
    let renderer: PreviewRenderer

    func makeUIView(context: Context) -> MTKView {
        let v = MTKView(frame: .zero, device: renderer.device)
        v.framebufferOnly = false          // Core Image writes into the drawable texture
        v.isPaused = true                  // draw on demand only
        v.enableSetNeedsDisplay = true
        v.autoResizeDrawable = true
        v.delegate = renderer
        v.backgroundColor = .black
        renderer.view = v
        return v
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        renderer.view = uiView
    }
}
