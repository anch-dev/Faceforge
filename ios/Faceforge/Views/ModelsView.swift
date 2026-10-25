import FaceforgeCore
import SwiftUI

struct ModelsView: View {
    @ObservedObject var models: ModelStore
    @ObservedObject var settings: AppSettings

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
            }
            .navigationTitle("Models")
        }
    }

    private var isBusy: Bool {
        switch models.state {
        case .downloading, .verifying: return true
        default: return false
        }
    }
}
