import AVFoundation
import CoreImage

/// Front camera -> upright, mirrored BGRA frames on a serial queue.
/// Late frames are discarded, so processing always works on the newest frame.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum Resolution: String, CaseIterable, Identifiable {
        case vga = "640x480", hd = "1280x720"
        var id: String { rawValue }
        var preset: AVCaptureSession.Preset { self == .vga ? .vga640x480 : .hd1280x720 }
    }

    let session = AVCaptureSession()
    let videoQueue = DispatchQueue(label: "faceforge.camera", qos: .userInteractive)
    var onFrame: ((CIImage) -> Void)?
    private let output = AVCaptureVideoDataOutput()
    private var configured = false

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func configure(resolution: Resolution, front: Bool = true) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.inputs.forEach { session.removeInput($0) }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                                   position: front ? .front : .back) else {
            throw CameraError.noCamera
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.noCamera }
        session.addInput(input)
        if session.canSetSessionPreset(resolution.preset) { session.sessionPreset = resolution.preset }

        if !configured {
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: videoQueue)
            guard session.canAddOutput(output) else { throw CameraError.noCamera }
            session.addOutput(output)
            configured = true
        }
        if let conn = output.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }   // portrait
            if conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = front
            }
        }
    }

    func start() { videoQueue.async { [session] in if !session.isRunning { session.startRunning() } } }
    func stop() { videoQueue.async { [session] in if session.isRunning { session.stopRunning() } } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(CIImage(cvPixelBuffer: pb))
    }

    enum CameraError: LocalizedError {
        case noCamera
        var errorDescription: String? { "Camera unavailable." }
    }
}
