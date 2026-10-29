import Foundation
import MLX
import zlib

public enum NumpyArchiveError: Error, LocalizedError {
    case invalidArchive(String)
    case unsupportedDataType(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArchive(let reason): "Invalid NumPy archive: \(reason)"
        case .unsupportedDataType(let descriptor): "Unsupported NumPy array type: \(descriptor)"
        }
    }
}

/// Reads numeric NumPy archives without extracting files or importing Python.
public enum NumpyArchive {
    public static func load(url: URL) throws -> [String: MLXArray] {
        try load(data: Data(contentsOf: url, options: .mappedIfSafe))
    }

    public static func load(data: Data) throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (name, raw) in try ZipArchive.members(data: data, include: { $0.hasSuffix(".npy") }) {
            result[String(name.dropLast(4))] = try array(raw)
        }
        return result
    }

    private static func array(_ data: Data) throws -> MLXArray {
        let npy = ArchiveBytes(data)
        try npy.check(0, 10)
        guard try npy.slice(0, 6) == Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]) else {
            throw invalid("invalid NPY signature")
        }
        let major = data[6], minor = data[7]
        guard minor == 0, (1...3).contains(major) else { throw invalid("unsupported NPY version") }
        let headerOffset = major == 1 ? 10 : 12
        let headerLength = major == 1 ? Int(try npy.u16(8)) : Int(try npy.u32(8))
        guard let header = String(data: try npy.slice(headerOffset, headerLength),
                                  encoding: major == 3 ? .utf8 : .isoLatin1) else {
            throw invalid("invalid NPY header text")
        }
        func field(_ pattern: String) throws -> String {
            guard let expression = try? NSRegularExpression(pattern: pattern),
                  let match = expression.firstMatch(in: header, range: NSRange(header.startIndex..., in: header)),
                  let range = Range(match.range(at: 1), in: header) else {
                throw invalid("invalid NPY header")
            }
            return String(header[range])
        }
        let descriptor = try field(#"["']descr["']\s*:\s*["']([^"']+)["']"#)
        let order = try field(#"["']fortran_order["']\s*:\s*(True|False)"#)
        let shapeText = try field(#"["']shape["']\s*:\s*\(([^)]*)\)"#)
        let shape: [Int] = try shapeText.split(separator: ",").compactMap { value in
            let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { return nil }
            guard let dimension = Int(text), dimension >= 0, dimension <= Int(Int32.max) else {
                throw invalid("invalid NPY shape")
            }
            return dimension
        }
        let kinds: [String: DType] = [
            "b1": .bool, "i1": .int8, "i2": .int16, "i4": .int32, "i8": .int64,
            "u1": .uint8, "u2": .uint16, "u4": .uint32, "u8": .uint64,
            "f2": .float16, "f4": .float32, "f8": .float64, "c8": .complex64,
        ]
        let byteOrder = descriptor.first ?? "="
        let hasOrder = "<>=|".contains(byteOrder)
        let kind = hasOrder ? String(descriptor.dropFirst()) : descriptor
        guard let dtype = kinds[kind] else { throw NumpyArchiveError.unsupportedDataType(descriptor) }
        var count = 1
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { throw invalid("NPY shape overflows") }
            count = product.partialValue
        }
        let size = count.multipliedReportingOverflow(by: dtype.size)
        guard !size.overflow else { throw invalid("NPY byte count overflows") }
        let payloadOffset = headerOffset + headerLength
        guard size.partialValue == data.count - payloadOffset else { throw invalid("NPY payload size mismatch") }
        if count == 0 { return MLXArray.zeros(shape, dtype: dtype, stream: .cpu) }
        var payload = try npy.slice(payloadOffset, size.partialValue)
        let hostLittle = UInt16(littleEndian: 1) == 1
        if (byteOrder == ">" && hostLittle) || (byteOrder == "<" && !hostLittle) {
            let unit = kind == "c8" ? 4 : dtype.size
            if unit > 1 {
                payload.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) in
                    for start in stride(from: 0, to: bytes.count, by: unit) {
                        for i in 0..<(unit / 2) { bytes.swapAt(start + i, start + unit - 1 - i) }
                    }
                }
            }
        }
        if order == "True", shape.count > 1 {
            return MLXArray(payload, Array(shape.reversed()), dtype: dtype).transposed(stream: .cpu)
        }
        return MLXArray(payload, shape, dtype: dtype)
    }

    private static func invalid(_ message: String) -> NumpyArchiveError { .invalidArchive(message) }
}

/// Reads stored or deflated ZIP members in memory, without extracting filesystem paths.
public enum ZipArchive {
    public static func load(url: URL) throws -> [String: Data] {
        try load(data: Data(contentsOf: url, options: .mappedIfSafe))
    }
    public static func load(data: Data) throws -> [String: Data] {
        try members(data: data, include: { _ in true })
    }
    fileprivate static func members(data: Data, include: (String) -> Bool) throws -> [String: Data] {
        let zip = ArchiveBytes(data)
        guard zip.count >= 22 else { throw invalid("missing ZIP directory") }
        let lowerBound = max(0, zip.count - 65_557)
        var end: Int?
        for offset in stride(from: zip.count - 22, through: lowerBound, by: -1) {
            if try zip.u32(offset) == 0x06054b50,
               offset + 22 + Int(try zip.u16(offset + 20)) == zip.count {
                end = offset
                break
            }
        }
        guard let end else { throw invalid("missing ZIP directory") }
        guard try zip.u16(end + 4) == 0, try zip.u16(end + 6) == 0,
              try zip.u16(end + 8) == zip.u16(end + 10) else {
            throw invalid("split ZIP archives are unsupported")
        }
        var entries = UInt64(try zip.u16(end + 10))
        var directorySize = UInt64(try zip.u32(end + 12))
        var directoryOffset = UInt64(try zip.u32(end + 16))
        if entries == 0xffff || directorySize == 0xffffffff || directoryOffset == 0xffffffff {
            guard end >= 20, try zip.u32(end - 20) == 0x07064b50,
                  try zip.u32(end - 16) == 0, try zip.u32(end - 4) == 1 else {
                throw invalid("missing ZIP64 directory")
            }
            let record = try integer(zip.u64(end - 12))
            try zip.check(record, 56)
            guard try zip.u32(record) == 0x06064b50, try zip.u64(record + 4) >= 44,
                  try zip.u32(record + 16) == 0, try zip.u32(record + 20) == 0,
                  try zip.u64(record + 24) == zip.u64(record + 32) else {
                throw invalid("invalid ZIP64 directory")
            }
            entries = try zip.u64(record + 32)
            directorySize = try zip.u64(record + 40)
            directoryOffset = try zip.u64(record + 48)
        }
        var offset = try integer(directoryOffset)
        let directoryLength = try integer(directorySize)
        try zip.check(offset, directoryLength)
        let directoryEnd = offset + directoryLength
        guard entries <= directorySize / 46 else { throw invalid("invalid ZIP entry count") }
        var members = [String: Data]()
        for _ in 0..<entries {
            try zip.check(offset, 46)
            guard try zip.u32(offset) == 0x02014b50 else { throw invalid("invalid ZIP entry") }
            let flags = try zip.u16(offset + 8)
            guard flags & 0x0041 == 0 else { throw invalid("encrypted ZIP entries are unsupported") }
            let method = try zip.u16(offset + 10)
            let checksum = try zip.u32(offset + 16)
            var compressed = UInt64(try zip.u32(offset + 20))
            var expanded = UInt64(try zip.u32(offset + 24))
            let nameLength = Int(try zip.u16(offset + 28))
            let extraLength = Int(try zip.u16(offset + 30))
            let commentLength = Int(try zip.u16(offset + 32))
            guard try zip.u16(offset + 34) == 0 else { throw invalid("split ZIP entry") }
            var localOffset = UInt64(try zip.u32(offset + 42))
            let entryLength = 46 + nameLength + extraLength + commentLength
            try zip.check(offset, entryLength)
            guard offset + entryLength <= directoryEnd,
                  let name = String(data: try zip.slice(offset + 46, nameLength), encoding: .utf8) else {
                throw invalid("invalid ZIP filename")
            }
            if compressed == 0xffffffff || expanded == 0xffffffff || localOffset == 0xffffffff {
                let extra = ArchiveBytes(try zip.slice(offset + 46 + nameLength, extraLength))
                var cursor = 0
                var found = false
                while cursor < extra.count {
                    let kind = try extra.u16(cursor)
                    let length = Int(try extra.u16(cursor + 2))
                    try extra.check(cursor + 4, length)
                    if kind == 1 {
                        let limit = cursor + 4 + length
                        cursor += 4
                        func next() throws -> UInt64 {
                            guard cursor + 8 <= limit else { throw invalid("truncated ZIP64 entry") }
                            defer { cursor += 8 }
                            return try extra.u64(cursor)
                        }
                        if expanded == 0xffffffff { expanded = try next() }
                        if compressed == 0xffffffff { compressed = try next() }
                        if localOffset == 0xffffffff { localOffset = try next() }
                        found = true
                        break
                    }
                    cursor += 4 + length
                }
                guard found else { throw invalid("missing ZIP64 entry sizes") }
            }
            offset += entryLength
            guard include(name), !name.hasSuffix("/") else { continue }
            let local = try integer(localOffset)
            try zip.check(local, 30)
            guard try zip.u32(local) == 0x04034b50,
                  try zip.u16(local + 6) == flags, try zip.u16(local + 8) == method else {
                throw invalid("invalid ZIP local header")
            }
            let localNameLength = Int(try zip.u16(local + 26))
            let localExtraLength = Int(try zip.u16(local + 28))
            guard try zip.slice(local + 30, localNameLength) == Data(name.utf8) else {
                throw invalid("ZIP filename mismatch")
            }
            let begin = local + 30 + localNameLength + localExtraLength
            let packedSize = try integer(compressed)
            let rawSize = try integer(expanded)
            // NumPy sidecars are arrays, not multi-gigabyte checkpoint containers.
            guard rawSize <= 2_147_483_647, packedSize <= Int(UInt32.max) else {
                throw invalid("array exceeds supported byte count")
            }
            let packed = try zip.slice(begin, packedSize)
            let raw: Data
            switch method {
            case 0:
                guard packedSize == rawSize else { throw invalid("stored ZIP size mismatch") }
                raw = packed
            case 8:
                raw = try inflate(packed, size: rawSize)
            default:
                throw invalid("unsupported ZIP compression method \(method)")
            }
            let actual = raw.withUnsafeBytes { bytes in
                UInt32(truncatingIfNeeded: crc32(0, bytes.bindMemory(to: Bytef.self).baseAddress, uInt(bytes.count)))
            }
            guard checksum == actual else { throw invalid("checksum mismatch for \(name)") }
            guard members[name] == nil else { throw invalid("duplicate ZIP member \(name)") }
            members[name] = raw
        }
        return members
    }
    private static func inflate(_ data: Data, size: Int) throws -> Data {
        var state = z_stream()
        guard inflateInit2_(&state, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
            throw invalid("could not initialize ZIP decompression")
        }
        defer { inflateEnd(&state) }
        var output = Data(count: max(size, 1))
        let status = data.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                state.next_in = UnsafeMutablePointer(mutating: source.bindMemory(to: Bytef.self).baseAddress)
                state.avail_in = uInt(source.count)
                state.next_out = destination.bindMemory(to: Bytef.self).baseAddress
                state.avail_out = uInt(destination.count)
                return zlib.inflate(&state, Z_FINISH)
            }
        }
        guard status == Z_STREAM_END, state.total_in == data.count, state.total_out == size else {
            throw invalid("truncated or invalid ZIP deflate stream")
        }
        return output.prefix(size)
    }

    private static func integer(_ value: UInt64) throws -> Int {
        guard let value = Int(exactly: value) else { throw invalid("ZIP offset overflows") }
        return value
    }

    private static func invalid(_ message: String) -> NumpyArchiveError { .invalidArchive(message) }

 }

fileprivate struct ArchiveBytes {
        let data: Data
        init(_ data: Data) { self.data = data }
        var count: Int { data.count }
        func check(_ start: Int, _ length: Int) throws {
            guard start >= 0, length >= 0, start <= count, length <= count - start else {
                throw NumpyArchiveError.invalidArchive("truncated data")
            }
        }
        func slice(_ start: Int, _ length: Int) throws -> Data {
            try check(start, length)
            return data.subdata(in: start..<(start + length))
        }
        func u16(_ start: Int) throws -> UInt16 {
            try check(start, 2)
            return UInt16(data[start]) | UInt16(data[start + 1]) << 8
        }
        func u32(_ start: Int) throws -> UInt32 {
            try check(start, 4)
            return (0..<4).reduce(0) { $0 | UInt32(data[start + $1]) << (8 * $1) }
        }
        func u64(_ start: Int) throws -> UInt64 {
            try check(start, 8)
            return (0..<8).reduce(0) { $0 | UInt64(data[start + $1]) << (8 * $1) }
        }
    }
