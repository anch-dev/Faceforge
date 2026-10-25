import FaceforgeCore
import Foundation
import Network

/// Client side of the offload protocol (docs/offload-protocol.md). Sends
/// camera frames to the Faceforge server on the PC and receives swapped frames.
/// Callbacks arrive on the client's private queue.
final class OffloadClient {
    enum State: Equatable {
        case idle, connecting, connected(provider: String, device: String), failed(String)
    }

    var onState: ((State) -> Void)?
    var onResult: ((OffloadProtocol.Result, Double) -> Void)?   // result, round-trip ms
    var onSourceAck: ((Bool, String) -> Void)?

    private let queue = DispatchQueue(label: "faceforge.offload")
    private var connection: NWConnection?
    private var decoder = OffloadProtocol.Decoder()
    private var inFlight = 0
    private var nextFrameID: UInt32 = 0
    private var token = ""
    private(set) var state: State = .idle { didSet { onState?(state) } }

    func connect(to endpoint: NWEndpoint, token: String) {
        queue.async { [self] in
            disconnectLocked()
            self.token = token
            decoder = OffloadProtocol.Decoder()
            inFlight = 0
            let params = NWParameters.tcp
            if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options {
                tcp.noDelay = true      // small messages, latency matters more than batching
            }
            let c = NWConnection(to: endpoint, using: params)
            connection = c
            state = .connecting
            c.stateUpdateHandler = { [weak self] s in self?.handle(s) }
            c.start(queue: queue)
        }
    }

    func disconnect() { queue.async { [self] in disconnectLocked(); state = .idle } }

    func sendSource(jpeg: Data) {
        queue.async { [self] in send(OffloadProtocol.encode(.source, jpeg)) }
    }

    func sendSettings(_ s: [String: Any]) {
        queue.async { [self] in send(OffloadProtocol.encodeJSON(.settings, s)) }
    }

    /// Returns false (frame skipped) when the in-flight window is full.
    func trySendFrame(jpeg: Data) -> Bool {
        queue.sync {
            guard case .connected = state, inFlight < OffloadProtocol.maxInFlight else { return false }
            inFlight += 1
            nextFrameID &+= 1
            send(OffloadProtocol.frame(id: nextFrameID, clientTimeUS: Self.nowUS(), jpeg: jpeg))
            return true
        }
    }

    // MARK: - private (on queue)

    private func handle(_ s: NWConnection.State) {
        switch s {
        case .ready:
            send(OffloadProtocol.encodeJSON(.hello, ["protocol": OffloadProtocol.version, "token": token, "client": "ios"]))
            receive()
        case .failed(let e):
            Log.error("Connection failed: \(e)", "offload")
            state = .failed(e.localizedDescription)
        case .waiting(let e):
            state = .failed("Waiting for network: \(e.localizedDescription)")
        default:
            break
        }
    }

    private func receive() {
        connection?.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, err in
            guard let self = self else { return }
            if let data = data, !data.isEmpty {
                do {
                    for m in try self.decoder.feed(data) { self.dispatch(m) }
                } catch {
                    self.state = .failed("Protocol error: \(error)")
                    self.disconnectLocked()
                    return
                }
            }
            if let err = err { self.state = .failed(err.localizedDescription); return }
            if done { self.state = .failed("Server closed the connection."); return }
            self.receive()
        }
    }

    private func dispatch(_ m: OffloadProtocol.Message) {
        switch m.type {
        case .welcome:
            let info = (try? OffloadProtocol.parseJSON(m.payload)) ?? [:]
            state = .connected(provider: info["provider"] as? String ?? "?", device: info["device"] as? String ?? "")
        case .result:
            inFlight = max(0, inFlight - 1)
            if let r = try? OffloadProtocol.parseResult(m.payload) {
                onResult?(r, Double(Self.nowUS() &- r.clientTimeUS) / 1000)
            }
        case .sourceAck:
            let j = (try? OffloadProtocol.parseJSON(m.payload)) ?? [:]
            onSourceAck?(j["ok"] as? Bool ?? false, j["message"] as? String ?? "")
        case .error:
            let j = (try? OffloadProtocol.parseJSON(m.payload)) ?? [:]
            Log.error("Server error: \(j)", "offload")
            state = .failed(j["message"] as? String ?? "Server error")
            disconnectLocked()
        default:
            break
        }
    }

    private func send(_ data: Data) {
        connection?.send(content: data, completion: .contentProcessed { [weak self] err in
            if let err = err { self?.state = .failed(err.localizedDescription) }
        })
    }

    private func disconnectLocked() {
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        inFlight = 0
    }

    static func nowUS() -> UInt64 { UInt64(DispatchTime.now().uptimeNanoseconds / 1000) }
}

/// Finds Faceforge servers advertising _faceforge._tcp on the local network.
final class ServerBrowser: ObservableObject {
    struct Found: Identifiable, Hashable {
        let id: String
        let endpoint: NWEndpoint
    }

    @Published private(set) var servers: [Found] = []
    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let b = NWBrowser(for: .bonjour(type: "_faceforge._tcp", domain: nil), using: .tcp)
        b.browseResultsChangedHandler = { [weak self] results, _ in
            let found = results.map { r -> Found in
                if case let .service(name, _, _, _) = r.endpoint { return Found(id: name, endpoint: r.endpoint) }
                return Found(id: "\(r.endpoint)", endpoint: r.endpoint)
            }
            DispatchQueue.main.async { self?.servers = found.sorted { $0.id < $1.id } }
        }
        b.start(queue: .main)
        browser = b
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }
}
