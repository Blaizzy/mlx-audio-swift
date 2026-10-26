import Foundation
import MLX
import Testing
import zlib
@testable import MLXAudioCore

@Suite("Native NumPy archives")
struct NumpyArchiveTests {
    @Test(arguments: [false, true])
    func storedAndDeflatedSidecars(compressed: Bool) throws {
        let mean: [Float] = [1, 2, 3, 4, 5, 6]
        let standardDeviation: [Float] = [0.5, 1.5, 2.5]
        let data = try zip([
            ("mean.npy", npy(bytes(mean), descriptor: "<f4", shape: [2, 3])),
            ("std.npy", npy(bytes(standardDeviation), descriptor: "<f4", shape: [3])),
        ], compressed: compressed)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).npz")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let arrays = try NumpyArchive.load(url: url)
        #expect(Set(arrays.keys) == ["mean", "std"])
        let loaded = try #require(arrays["mean"])
        #expect(loaded.shape == [2, 3])
        #expect(loaded.dtype == .float32)
        #expect(loaded.asArray(Float.self) == mean)
        #expect(arrays["std"]?.asArray(Float.self) == standardDeviation)
    }

    @Test func fortranEndianScalarAndEmptyArrays() throws {
        let fortran = Data([0, 1, 0, 4, 0, 2, 0, 5, 0, 3, 0, 6])
        let data = try zip([
            ("matrix.npy", npy(fortran, descriptor: ">i2", shape: [2, 3], fortran: true, version: 2)),
            ("scalar.npy", npy(bytes([Int64(42)]), descriptor: "<i8", shape: [], version: 3)),
            ("empty.npy", npy(Data(), descriptor: "<f4", shape: [0, 3])),
        ], compressed: true)
        let arrays = try NumpyArchive.load(data: data)
        let matrix = try #require(arrays["matrix"])
        #expect(matrix.shape == [2, 3])
        #expect(matrix.asArray(Int16.self) == [1, 2, 3, 4, 5, 6])
        #expect(arrays["scalar"]?.shape == [])
        #expect(arrays["scalar"]?.item(Int64.self) == 42)
        #expect(arrays["empty"]?.shape == [0, 3])
        #expect(arrays["empty"]?.size == 0)
    }

    @Test func zip64DirectoryAndEntries() throws {
        let raw = npy(bytes([Float(1), 2]), descriptor: "<f4", shape: [2])
        let data = try zip([("layer.weight.npy", raw)], compressed: true, zip64: true)
        let arrays = try NumpyArchive.load(data: data)
        #expect(arrays["layer.weight"]?.asArray(Float.self) == [1, 2])
    }

    @Test func rejectsCorruptTruncatedAndObjectData() throws {
        let raw = npy(bytes([Float(1), 2]), descriptor: "<f4", shape: [2])
        var data = try zip([("a.npy", raw)], compressed: false)
        // First member starts at 55: fixed header + name + NumPy-style ZIP64 extra.
        data[60] ^= 1
        do {
            _ = try NumpyArchive.load(data: data)
            Issue.record("Corrupt archive was accepted")
        } catch {
            #expect(error.localizedDescription.contains("checksum"))
        }
        #expect(throws: NumpyArchiveError.self) { try NumpyArchive.load(data: data.dropLast()) }
        let object = try zip([("object.npy", npy(Data(), descriptor: "|O", shape: [1]))], compressed: false)
        #expect(throws: NumpyArchiveError.self) { try NumpyArchive.load(data: object) }
        let truncated = try zip([("bad.npy", npy(Data(), descriptor: "<f4", shape: [2]))], compressed: false)
        #expect(throws: NumpyArchiveError.self) { try NumpyArchive.load(data: truncated) }
    }

    @Test func rejectsDuplicateKeysAndAllowsEmptyArchive() throws {
        let raw = npy(bytes([Int32(7)]), descriptor: "<i4", shape: [1])
        let duplicated = try zip([("key.npy", raw), ("key.npy", raw)], compressed: false)
        #expect(throws: NumpyArchiveError.self) { try NumpyArchive.load(data: duplicated) }
        #expect(try NumpyArchive.load(data: zip([], compressed: false)).isEmpty)
    }

    private func bytes<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }

    private func npy(
        _ payload: Data, descriptor: String, shape: [Int], fortran: Bool = false, version: UInt8 = 1
    ) -> Data {
        let dimensions = shape.isEmpty ? "" : shape.map(String.init).joined(separator: ", ") + ","
        var header = "{'descr': '\(descriptor)', 'fortran_order': \(fortran ? "True" : "False"), 'shape': (\(dimensions)), }"
        let prefix = version == 1 ? 10 : 12
        while (prefix + header.utf8.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var output = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, version, 0])
        if version == 1 { output.appendNumpyLE(UInt16(header.utf8.count)) }
        else { output.appendNumpyLE(UInt32(header.utf8.count)) }
        output.append(contentsOf: header.utf8)
        output.append(payload)
        return output
    }

    private func zip(_ files: [(String, Data)], compressed: Bool, zip64: Bool = false) throws -> Data {
        var output = Data(), central = Data()
        for (name, raw) in files {
            let packed = compressed ? try deflate(raw) : raw
            let checksum = raw.withUnsafeBytes { bytes in
                UInt32(truncatingIfNeeded: crc32(0, bytes.bindMemory(to: Bytef.self).baseAddress, uInt(bytes.count)))
            }
            let offset = output.count
            let filename = Data(name.utf8)
            var localExtra = Data()
            localExtra.appendNumpyLE(UInt16(1)); localExtra.appendNumpyLE(UInt16(16))
            localExtra.appendNumpyLE(UInt64(raw.count)); localExtra.appendNumpyLE(UInt64(packed.count))
            output.appendNumpyLE(UInt32(0x04034b50)); output.appendNumpyLE(UInt16(45))
            output.appendNumpyLE(UInt16(0x0800)); output.appendNumpyLE(UInt16(compressed ? 8 : 0))
            output.appendNumpyLE(UInt32(0)); output.appendNumpyLE(checksum)
            output.appendNumpyLE(UInt32.max); output.appendNumpyLE(UInt32.max)
            output.appendNumpyLE(UInt16(filename.count)); output.appendNumpyLE(UInt16(localExtra.count))
            output.append(filename); output.append(localExtra); output.append(packed)
            var extra = Data()
            if zip64 {
                extra.appendNumpyLE(UInt16(1)); extra.appendNumpyLE(UInt16(24))
                extra.appendNumpyLE(UInt64(raw.count)); extra.appendNumpyLE(UInt64(packed.count))
                extra.appendNumpyLE(UInt64(offset))
            }
            central.appendNumpyLE(UInt32(0x02014b50))
            central.appendNumpyLE(UInt16(45)); central.appendNumpyLE(UInt16(45))
            central.appendNumpyLE(UInt16(0x0800)); central.appendNumpyLE(UInt16(compressed ? 8 : 0))
            central.appendNumpyLE(UInt32(0)); central.appendNumpyLE(checksum)
            central.appendNumpyLE(zip64 ? UInt32.max : UInt32(packed.count))
            central.appendNumpyLE(zip64 ? UInt32.max : UInt32(raw.count))
            central.appendNumpyLE(UInt16(filename.count)); central.appendNumpyLE(UInt16(extra.count))
            central.appendNumpyLE(UInt16(0)); central.appendNumpyLE(UInt16(0))
            central.appendNumpyLE(UInt16(0)); central.appendNumpyLE(UInt32(0))
            central.appendNumpyLE(zip64 ? UInt32.max : UInt32(offset))
            central.append(filename); central.append(extra)
        }
        let directoryOffset = output.count
        output.append(central)
        if zip64 {
            let recordOffset = output.count
            output.appendNumpyLE(UInt32(0x06064b50)); output.appendNumpyLE(UInt64(44))
            output.appendNumpyLE(UInt16(45)); output.appendNumpyLE(UInt16(45))
            output.appendNumpyLE(UInt32(0)); output.appendNumpyLE(UInt32(0))
            output.appendNumpyLE(UInt64(files.count)); output.appendNumpyLE(UInt64(files.count))
            output.appendNumpyLE(UInt64(central.count)); output.appendNumpyLE(UInt64(directoryOffset))
            output.appendNumpyLE(UInt32(0x07064b50)); output.appendNumpyLE(UInt32(0))
            output.appendNumpyLE(UInt64(recordOffset)); output.appendNumpyLE(UInt32(1))
        }
        output.appendNumpyLE(UInt32(0x06054b50))
        output.appendNumpyLE(UInt16(0)); output.appendNumpyLE(UInt16(0))
        output.appendNumpyLE(zip64 ? UInt16.max : UInt16(files.count))
        output.appendNumpyLE(zip64 ? UInt16.max : UInt16(files.count))
        output.appendNumpyLE(zip64 ? UInt32.max : UInt32(central.count))
        output.appendNumpyLE(zip64 ? UInt32.max : UInt32(directoryOffset))
        output.appendNumpyLE(UInt16(0))
        return output
    }

    private func deflate(_ data: Data) throws -> Data {
        var state = z_stream()
        let status = deflateInit2_(&state, 6, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY,
                                   ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
        guard status == Z_OK else { throw NumpyArchiveError.invalidArchive("test compression setup") }
        defer { deflateEnd(&state) }
        var result = Data(count: Int(compressBound(uLong(data.count))))
        let finished = data.withUnsafeBytes { source in
            result.withUnsafeMutableBytes { destination in
                state.next_in = UnsafeMutablePointer(mutating: source.bindMemory(to: Bytef.self).baseAddress)
                state.avail_in = uInt(source.count)
                state.next_out = destination.bindMemory(to: Bytef.self).baseAddress
                state.avail_out = uInt(destination.count)
                return zlib.deflate(&state, Z_FINISH)
            }
        }
        guard finished == Z_STREAM_END else { throw NumpyArchiveError.invalidArchive("test compression") }
        return result.prefix(Int(state.total_out))
    }
}

private extension Data {
    mutating func appendNumpyLE<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
