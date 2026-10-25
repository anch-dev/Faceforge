import FaceforgeCore
import Network
import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var live: LiveSession
    @StateObject private var browser = ServerBrowser()

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("PC address (e.g. 192.168.1.20)", text: $settings.pcHost)
                        .keyboardType(.numbersAndPunctuation).textInputAutocapitalization(.never).autocorrectionDisabled()
                    TextField("Port", value: $settings.pcPort, format: .number.grouping(.never))
                        .keyboardType(.numberPad)
                    TextField("Pairing code", text: $settings.pcToken).keyboardType(.numberPad)
                } header: {
                    Text("PC offload")
                } footer: {
                    Text("Run server\\Start-Faceforge-Server.bat on your PC. It prints the address and pairing code. Phone and PC must be on the same Wi-Fi.")
                }
                Section("PCs found on this network") {
                    if browser.servers.isEmpty {
                        Text("Searching... (needs the 'zeroconf' package on the PC, which the launcher installs)")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    ForEach(browser.servers) { s in
                        Button(s.id) { resolve(s.endpoint) }
                    }
                }
                Section("Live") {
                    Picker("Camera", selection: $settings.resolution) {
                        ForEach(CameraController.Resolution.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Stepper("Detect face every \(settings.detectEvery) frames", value: $settings.detectEvery, in: 1...10)
                    Toggle("Use Neural Engine (Core ML)", isOn: $settings.useCoreML)
                    Slider(value: $settings.jpegQuality, in: 0.4...0.95) { Text("PC image quality") }
                    Toggle("Show stats", isOn: $settings.showStats)
                }
                Section("About") {
                    Text("Faceforge is for personal use. Only use faces you have permission to use, and label anything you share as a deepfake.")
                        .font(.footnote)
                }
            }
            .navigationTitle("Settings")
            .onAppear { browser.start() }
            .onDisappear { browser.stop() }
        }
    }

    /// Resolve a Bonjour service to host:port and store it.
    private func resolve(_ endpoint: NWEndpoint) {
        let c = NWConnection(to: endpoint, using: .tcp)
        c.stateUpdateHandler = { state in
            if case .ready = state, let remote = c.currentPath?.remoteEndpoint,
               case let .hostPort(host, port) = remote {
                var h = "\(host)"
                if let pct = h.firstIndex(of: "%") { h = String(h[..<pct]) }   // drop interface scope
                DispatchQueue.main.async {
                    settings.pcHost = h
                    settings.pcPort = Int(port.rawValue)
                }
                c.cancel()
            } else if case .failed = state {
                c.cancel()
            }
        }
        c.start(queue: .main)
    }
}
