import Foundation

/// Reads a float initializer straight out of an .onnx file's protobuf.
///
/// InsightFace's inswapper keeps its 512x512 "emap" as the LAST graph
/// initializer, unused by any node; insightface multiplies the source
/// embedding by it before inference. ONNX Runtime drops unused initializers,
/// so the app reads it from the file directly. Minimal wire-format parser:
/// ModelProto.graph (7) -> GraphProto.initializer (5) -> TensorProto
/// {dims 1, data_type 2, float_data 4, name 8, raw_data 9}.
public struct OnnxTensor {
    public let name: String
    public let dims: [Int]
    public let floats: [Float]
}

public enum OnnxReaderError: Error, Equatable {
    case truncated
    case noGraph
    case noInitializers
    case unsupportedType(Int)
    case sizeMismatch
}

public enum OnnxInitializerReader {
    public static func lastInitializer(fileURL: URL) throws -> OnnxTensor {
        let data = try Data(contentsOf: fileURL, options: .alwaysMapped)
        return try lastInitializer(data: data)
    }

    public static func lastInitializer(data: Data) throws -> OnnxTensor {
        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> OnnxTensor in
            let bytes = raw.bindMemory(to: UInt8.self)
            guard let graph = try findLastField(bytes, range: 0..<bytes.count, field: 7) else {
                throw OnnxReaderError.noGraph
            }
            guard let tensor = try findLastField(bytes, range: graph, field: 5) else {
                throw OnnxReaderError.noInitializers
            }
            return try parseTensor(bytes, range: tensor)
        }
    }

    // MARK: - wire format

    private static func readVarint(_ b: UnsafeBufferPointer<UInt8>, _ pos: inout Int, _ end: Int) throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while true {
            guard pos < end, shift < 64 else { throw OnnxReaderError.truncated }
            let byte = b[pos]
            pos += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 { return result }
            shift += 7
        }
    }

    /// Iterate fields in `range`; return the byte range of the LAST
    /// length-delimited field with number `field`.
    private static func findLastField(_ b: UnsafeBufferPointer<UInt8>, range: Range<Int>, field: Int) throws -> Range<Int>? {
        var pos = range.lowerBound
        var found: Range<Int>?
        while pos < range.upperBound {
            let key = try readVarint(b, &pos, range.upperBound)
            let number = Int(key >> 3), wire = Int(key & 7)
            switch wire {
            case 0: _ = try readVarint(b, &pos, range.upperBound)
            case 1: pos += 8
            case 2:
                let len = Int(try readVarint(b, &pos, range.upperBound))
                guard pos + len <= range.upperBound else { throw OnnxReaderError.truncated }
                if number == field { found = pos..<(pos + len) }
                pos += len
            case 5: pos += 4
            default: throw OnnxReaderError.truncated
            }
            guard pos <= range.upperBound else { throw OnnxReaderError.truncated }
        }
        return found
    }

    private static func parseTensor(_ b: UnsafeBufferPointer<UInt8>, range: Range<Int>) throws -> OnnxTensor {
        var pos = range.lowerBound
        var dims: [Int] = []
        var dataType = 0
        var name = ""
        var floats: [Float] = []
        var raw: Range<Int>?
        while pos < range.upperBound {
            let key = try readVarint(b, &pos, range.upperBound)
            let number = Int(key >> 3), wire = Int(key & 7)
            switch (number, wire) {
            case (1, 0): dims.append(Int(Int64(bitPattern: try readVarint(b, &pos, range.upperBound))))
            case (1, 2): // packed dims
                let len = Int(try readVarint(b, &pos, range.upperBound))
                let end = pos + len
                while pos < end { dims.append(Int(Int64(bitPattern: try readVarint(b, &pos, end)))) }
            case (2, 0): dataType = Int(try readVarint(b, &pos, range.upperBound))
            case (4, 2): // packed float_data
                let len = Int(try readVarint(b, &pos, range.upperBound))
                guard pos + len <= range.upperBound, len % 4 == 0 else { throw OnnxReaderError.truncated }
                floats = readFloats(b, pos..<(pos + len))
                pos += len
            case (8, 2):
                let len = Int(try readVarint(b, &pos, range.upperBound))
                guard pos + len <= range.upperBound else { throw OnnxReaderError.truncated }
                name = String(decoding: UnsafeBufferPointer(rebasing: b[pos..<(pos + len)]), as: UTF8.self)
                pos += len
            case (9, 2):
                let len = Int(try readVarint(b, &pos, range.upperBound))
                guard pos + len <= range.upperBound else { throw OnnxReaderError.truncated }
                raw = pos..<(pos + len)
                pos += len
            case (_, 0): _ = try readVarint(b, &pos, range.upperBound)
            case (_, 1): pos += 8
            case (_, 2):
                let len = Int(try readVarint(b, &pos, range.upperBound))
                pos += len
            case (_, 5): pos += 4
            default: throw OnnxReaderError.truncated
            }
            guard pos <= range.upperBound else { throw OnnxReaderError.truncated }
        }
        guard dataType == 1 else { throw OnnxReaderError.unsupportedType(dataType) } // FLOAT only
        if let raw = raw {
            guard raw.count % 4 == 0 else { throw OnnxReaderError.truncated }
            floats = readFloats(b, raw)
        }
        let count = dims.reduce(1, *)
        guard floats.count == count else { throw OnnxReaderError.sizeMismatch }
        return OnnxTensor(name: name, dims: dims, floats: floats)
    }

    /// Little-endian float32 values (ONNX raw_data byte order).
    private static func readFloats(_ b: UnsafeBufferPointer<UInt8>, _ r: Range<Int>) -> [Float] {
        var out = [Float](repeating: 0, count: r.count / 4)
        var i = 0
        var p = r.lowerBound
        while p < r.upperBound {
            let bits = UInt32(b[p]) | UInt32(b[p + 1]) << 8 | UInt32(b[p + 2]) << 16 | UInt32(b[p + 3]) << 24
            out[i] = Float(bitPattern: bits)
            i += 1
            p += 4
        }
        return out
    }
}
