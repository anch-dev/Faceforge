import FaceforgeCore
import SwiftUI
import UniformTypeIdentifiers

struct ModelsView: View {
    @ObservedObject var models: ModelStore
    @ObservedObject var settings: AppSettings
    @State private var code = ""
    @State private var importing = false

    var body: some View {
        NavigationStack {
            Form {
                Section("On-device models") {
                    switch models.state {
                    case .ready:
                        Label("Installed", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    case .missing:
                        Text("Not installed. Needed only for \"On iPhone\" mode; PC offload works without them.")
                    case .downloading(let label, let f):
                        VStack(alignment: .leading) {
                            Text(label)
                            ProgressView(value: f)
                            Text(String(format: "%.0f%%", f * 100)).font(.caption)
                        }
                    case .verifying(let label):
                        HStack { ProgressView(); Text("Checking \(label)...") }
                    case .failed(let m):
                        Text(m).foregroundStyle(.red)
                    }
                }
                Section("Licence") {
                    Text("Downloads come straight from InsightFace's GitHub releases (about "
                         + ByteCountFormatter.string(fromByteCount: ModelCatalog.totalDownloadBytes, countStyle: .file)
                         + "). Faceforge does not host or modify them.")
                    Text(ModelCatalog.license).bold()
                    Link("InsightFace licence", destination: ModelCatalog.licenseURL)
                    Toggle("I accept these terms (personal, non-commercial use)", isOn: $settings.acceptedModelLicense)
                }
                Section {
                    Button("Download models") { Task { await models.downloadAll() } }
                        .disabled(!settings.acceptedModelLicense || isBusy || models.state == .ready)
                    Button("Delete models", role: .destructive) { models.deleteAll() }
                        .disabled(isBusy)
                } footer: {
                    Text("Use Wi-Fi and keep the app open while downloading. The first Live start afterwards compiles the models for the Neural Engine, which takes a while once.")
                }
                fastModelSection
            }
            .navigationTitle("Models")
            .fileImporter(isPresented: $importing, allowedContentTypes: Self.importTypes) { result in
                guard case .success(let url) = result else { return }
                Task { try? await models.installFastModel(from: url) }
            }
        }
    }

    @ViewBuilder private var fastModelSection: some View {
        Section {
            switch models.fastState {
            case .ready:
                Label("Installed: \(models.fastModel?.label ?? "?")", systemImage: "bolt.circle.fill").foregroundStyle(.green)
                Toggle("Use it for Live and Photo", isOn: $settings.preferFastModel)
            case .missing:
                Text("Not installed. Live uses the standard model (GPU, slower).")
            case .downloading(let label, let f):
                VStack(alignment: .leading) { Text(label); ProgressView(value: f) }
            case .verifying(let label):
                HStack { ProgressView(); Text(label + "...") }
            case .failed(let m):
                Text(m).foregroundStyle(.red)
            }
            TextField("Download code from the PC", text: $code).keyboardType(.numberPad)
            Button("Get from PC (\(settings.pcHost.isEmpty ? "set PC address in Settings" : settings.pcHost))") {
                Task { await models.fetchFastModel(host: settings.pcHost, code: code) }
            }
            .disabled(settings.pcHost.isEmpty || code.isEmpty || isFastBusy)
            Button("Import from Files...") { importing = true }.disabled(isFastBusy)
            if models.fastModel != nil {
                Button("Remove fast model", role: .destructive) { models.deleteFastModel() }.disabled(isFastBusy)
            }
        } header: {
            Text("Fast swap model (Neural Engine)")
        } footer: {
            Text("The Neural Engine only runs half-precision (fp16) models. On your PC run server\\Send-Model-To-iPhone.bat: it makes one from your own copy of the swap model (nothing is uploaded) and prints a code. Or import an .mlpackage (zipped) or fp16 .onnx made with tools/convert_coreml.py.")
        }
    }

    private static let importTypes: [UTType] = [.zip, UTType(filenameExtension: "onnx") ?? .data,
                                                UTType(filenameExtension: "mlpackage") ?? .package]

    private var isFastBusy: Bool {
        switch models.fastState {
        case .downloading, .verifying: return true
        default: return false
        }
    }

    private var isBusy: Bool {
        switch models.state {
        case .downloading, .verifying: return true
        default: return false
        }
    }
}
