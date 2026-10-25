import Foundation

/// Faceforge offload protocol v1 (docs/offload-protocol.md). Byte-for-byte
/// compatible with faceforge_server/protocol.py; checked against
/// tests/fixtures/protocol_v1.json in both languages.
public enum OffloadProtocol {
    public static let magic: [UInt8] = Array("FFG1".utf8)
    public static let headerSize = 12
    public static let maxPayload = 16 * 1024 * 1024
    public static let version = 1
    public static let defaultPort: UInt16 = 47800
    public static let maxInFlight = 2

    public enum MessageType: UInt8 {
        case hello = 1, welcome, source, sourceAck, frame, result, settings, error
    }

    public struct Message: Equatable {
        public let type: MessageType
        public let payload: Data
        public init(type: MessageType, payload: Data) { self.type = type; self.payload = payload }
    }

    public struct Result: Equatable {
        public let frameID: UInt32
        public let clientTimeUS: UInt64
        public let serverMs: Float
        public let faceFound: Bool
        public let jpeg: Data
    }

    public enum ProtocolError: Error, Equatable {
        case badMagic, unknownType(UInt8), tooLarge(Int), short, badJSON
    }

    // MARK: encoding

    public static func encode(_ type: MessageType, _ payload: Data = Data()) -> Data {
        var d = Data(magic)
        d.append(type.rawValue)
        d.append(contentsOf: [0, 0, 0])            // flags, reserved
        appendBE(&d, UInt32(payload.count))
        d.append(payload)
        return d
    }

    /// JSON with sorted keys and no spaces, identical to the Python encoder.
    public static func encodeJSON(_ type: MessageType, _ obj: [String: Any]) -> Data {
        let json = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])
        return encode(type, json)
    }

    public static func frame(id: UInt32, clientTimeUS: UInt64, jpeg: Data) -> Data {
        var p = Data()
        appendBE(&p, id)
        appendBE(&p, clientTimeUS)
        p.append(jpeg)
        return encode(.frame, p)
    }

    public static func parseResult(_ payload: Data) throws -> Result {
        guard payload.count >= 17 else { throw ProtocolError.short }
        let b = [UInt8](payload.prefix(17))
        let id = readBE32(b, 0)
        let ts = UInt64(readBE32(b, 4)) << 32 | UInt64(readBE32(b, 8))
        let ms = Float(bitPattern: readBE32(b, 12))
        return Result(frameID: id, clientTimeUS: ts, serverMs: ms, faceFound: b[16] != 0,
                      jpeg: payload.subdata(in: (payload.startIndex + 17)..<payload.endIndex))
    }

    public static func parseJSON(_ payload: Data) throws -> [String: Any] {
        guard let obj = try? JSONSerialization.jsonObject(with: payload), let dict = obj as? [String: Any] else {
            throw ProtocolError.badJSON
        }
        return dict
    }

    // MARK: streaming decoder

    /// Accumulates bytes from the socket and yields complete messages.
    public struct Decoder {
        private var buffer = Data()
        public init() {}

        public mutating func feed(_ bytes: Data) throws -> [Message] {
            buffer.append(bytes)
            var out: [Message] = []
            while buffer.count >= headerSize {
                let h = [UInt8](buffer.prefix(headerSize))
                guard Array(h[0..<4]) == magic else { throw ProtocolError.badMagic }
                guard let type = MessageType(rawValue: h[4]) else { throw ProtocolError.unknownType(h[4]) }
                let len = Int(readBE32(h, 8))
                guard len <= maxPayload else { throw ProtocolError.tooLarge(len) }
                guard buffer.count >= headerSize + len else { break }
                let start = buffer.startIndex
                out.append(Message(type: type, payload: buffer.subdata(in: (start + headerSize)..<(start + headerSize + len))))
                buffer = buffer.subdata(in: (start + headerSize + len)..<buffer.endIndex)
            }
            return out
        }
    }

    // MARK: helpers

    static func appendBE(_ d: inout Data, _ v: UInt32) {
        d.append(contentsOf: [UInt8(v >> 24 & 0xFF), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)])
    }

    static func appendBE(_ d: inout Data, _ v: UInt64) {
        appendBE(&d, UInt32(v >> 32))
        appendBE(&d, UInt32(v & 0xFFFF_FFFF))
    }

    static func readBE32(_ b: [UInt8], _ o: Int) -> UInt32 {
        UInt32(b[o]) << 24 | UInt32(b[o + 1]) << 16 | UInt32(b[o + 2]) << 8 | UInt32(b[o + 3])
    }
}
