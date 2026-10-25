import FaceforgeCore
import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var settings = AppSettings()
    @StateObject private var models = ModelStore()
    @StateObject private var live = LiveSession()

    var body: some View {
        TabView {
            LiveView(live: live, settings: settings, models: models)
                .tabItem { Label("Live", systemImage: "video") }
            SourceView()
                .tabItem { Label("Source", systemImage: "person.crop.square") }
            PhotoSwapView(settings: settings, models: models)
                .tabItem { Label("Photo", systemImage: "photo") }
            ModelsView(models: models, settings: settings)
                .tabItem { Label("Models", systemImage: "square.and.arrow.down") }
            SettingsView(settings: settings, live: live)
                .tabItem { Label("Settings", systemImage: "gear") }
        }
    }
}

struct LiveView: View {
    @ObservedObject var live: LiveSession
    @ObservedObject var settings: AppSettings
    @ObservedObject var models: ModelStore

    var body: some View {
        ZStack(alignment: .top) {
            MetalPreview(renderer: live.renderer).ignoresSafeArea()
            VStack(alignment: .leading, spacing: 4) {
                Text(live.status).font(.callout.bold())
                if settings.showStats && live.running {
                    Text(live.backend).font(.caption)
                    Text(String(format: "%.1f FPS · 1%% low %.1f · %.0f ms", live.fps, live.onePctLow, live.latencyMs))
                        .font(.caption.monospacedDigit())
                    Text(live.stageText).font(.caption2.monospacedDigit())
                    Text("Thermal: \(live.thermalText)").font(.caption2)
                }
            }
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            VStack {
                Spacer()
                Picker("Mode", selection: $settings.mode) {
                    ForEach(ProcessingMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(live.running)
                .padding(.horizontal)
                Button(live.running ? "Stop" : "Start") {
                    Task {
                        if live.running { live.stop() } else { await live.start(settings: settings, models: models) }
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.bottom, 24)
            }
        }
        .onDisappear { if live.running { live.stop() } }
    }
}

struct SourceView: View {
    @State private var item: PhotosPickerItem?
    @State private var image: UIImage? = UIImage(contentsOfFile: AppSettings.sourceURL.path)
    @State private var message = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let image = image {
                    Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 360)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    ContentUnavailableView("No source face", systemImage: "person.crop.square",
                                           description: Text("Pick a clear, front-facing photo of the face to swap in."))
                }
                PhotosPicker("Choose photo", selection: $item, matching: .images)
                    .buttonStyle(.borderedProminent)
                Text("Only use a face you have permission to use. Label anything you share as a deepfake.")
                    .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                Text(message).font(.footnote)
            }
            .padding()
            .navigationTitle("Source face")
            .onChange(of: item) { _, newItem in
                Task {
                    guard let data = try? await newItem?.loadTransferable(type: Data.self),
                          let ui = UIImage(data: data), let jpeg = ui.normalizedJPEG() else { return }
                    do {
                        try jpeg.write(to: AppSettings.sourceURL, options: .atomic)
                        image = UIImage(data: jpeg)
                        message = "Saved. Restart Live to use it."
                    } catch { message = error.localizedDescription }
                }
            }
        }
    }
}

extension UIImage {
    /// Upright JPEG (bakes EXIF orientation in, so pixel rows are top-down).
    func normalizedJPEG(quality: CGFloat = 0.92, maxSide: CGFloat = 2048) -> Data? {
        let scale = min(1, maxSide / max(size.width, size.height))
        let target = CGSize(width: size.width * scale, height: size.height * scale)
        let fmt = UIGraphicsImageRendererFormat.default()
        fmt.scale = 1
        return UIGraphicsImageRenderer(size: target, format: fmt).image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }.jpegData(compressionQuality: quality)
    }
}
