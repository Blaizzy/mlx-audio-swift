import CryptoKit
import Foundation
import libzstd

/// Reader for the checksummed tar/Zstandard transport used by published Phonon checkpoints.
enum Phonon2Archive {
    struct Manifest: Decodable {
        struct File: Decodable {
            let path: String
            let original_bytes: Int
            let original_sha256: String
            let transform: Transform?
        }
        let release_format: String
        let files: [File]
    }

    struct Transform: Decodable {
        let original_bytes: Int
        let base: Int
        let payload_bytes: Int
        let plan: [[Int]]
    }

    static func extract(_ archive: URL, to directory: URL, expectedBytes: Int) throws {
        guard expectedBytes > 0, expectedBytes <= 1 << 30 else {
            throw invalid("Invalid unpacked size")
        }
        let compressed = try Data(contentsOf: archive, options: .mappedIfSafe)
        let tar = try decompress(compressed, limit: expectedBytes + (16 << 20))
        var cursor = 0
        var files: [String: Manifest.File]?
        var seen = Set<String>()
        var total = 0
        while cursor <= tar.count - 512 {
            let header = tar.subdata(in: cursor ..< (cursor + 512))
            if header.allSatisfy({ $0 == 0 }) {
                guard tar[cursor...].allSatisfy({ $0 == 0 }) else {
                    throw invalid("Trailing tar bytes")
                }
                cursor = tar.count
                break
            }
            let checksum = try octal(header.subdata(in: 148 ..< 156))
            let actual = header.enumerated().reduce(0) {
                $0 + (148 ..< 156 ~= $1.offset ? 32 : Int($1.element))
            }
            guard checksum == actual, header[156] == 48 || header[156] == 0 else {
                throw invalid("Invalid or non-file tar member")
            }
            let name = text(header.subdata(in: 0 ..< 100))
            let prefix = text(header.subdata(in: 345 ..< 500))
            let path = prefix.isEmpty ? name : "\(prefix)/\(name)"
            try validatePath(path)
            let size = try octal(header.subdata(in: 124 ..< 136))
            cursor += 512
            guard size >= 0, size <= tar.count - cursor else {
                throw invalid("Truncated tar member")
            }
            let blob = tar.subdata(in: cursor ..< (cursor + size))
            let padded = ((size + 511) / 512) * 512
            guard padded <= tar.count - cursor else { throw invalid("Truncated tar padding") }
            cursor += padded
            if files == nil {
                guard path == "bps_manifest.json", size <= 16 << 20 else {
                    throw invalid("Missing archive manifest")
                }
                let manifest = try JSONDecoder().decode(Manifest.self, from: blob)
                guard
                    ["phonon-byteplane-tar-zstd-v1", "sttg1a-byteplane-tar-zstd-v1"].contains(
                        manifest.release_format)
                else {
                    throw invalid("Unsupported archive format")
                }
                var index: [String: Manifest.File] = [:]
                for row in manifest.files {
                    try validatePath(row.path)
                    guard row.original_bytes >= 0, row.original_bytes <= expectedBytes,
                        index.updateValue(row, forKey: row.path) == nil
                    else { throw invalid("Invalid manifest member") }
                }
                files = index
                continue
            }
            let transformed = path.hasSuffix(".bps")
            let relative = transformed ? String(path.dropLast(4)) : path
            guard let row = files?[relative], seen.insert(relative).inserted,
                transformed == (row.transform != nil)
            else { throw invalid("Unexpected archive member: \(path)") }
            let restored = try row.transform.map { try joinBytePlanes(blob, transform: $0) } ?? blob
            guard restored.count == row.original_bytes, restored.count <= expectedBytes - total
            else {
                throw invalid("Archive member size mismatch")
            }
            try verify(restored, sha256: row.original_sha256)
            let target = directory.appendingPathComponent(relative)
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try restored.write(to: target)
            total += restored.count
        }
        guard cursor == tar.count, let files, seen.count == files.count, total == expectedBytes
        else {
            throw invalid("Incomplete archive")
        }
    }

    static func joinBytePlanes(_ data: Data, transform: Transform) throws -> Data {
        let base = transform.base
        let payload = transform.payload_bytes
        guard data.count == transform.original_bytes, base >= 0, base <= data.count,
            payload == data.count - base
        else { throw invalid("Invalid byte-plane metadata") }
        var output = Data(count: data.count)
        try output.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
            try data.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                destination.copyMemory(from: source)
                var cursor = 0
                for span in transform.plan {
                    guard span.count == 3 else { throw invalid("Invalid byte-plane span") }
                    let (start, end, width) = (span[0], span[1], span[2])
                    guard cursor <= start, start <= end, end <= payload, width > 1,
                        (end - start) % width == 0
                    else { throw invalid("Invalid byte-plane span") }
                    let count = (end - start) / width
                    for plane in 0 ..< width {
                        for i in 0 ..< count {
                            destination[base + start + i * width + plane] =
                                source[base + start + plane * count + i]
                        }
                    }
                    cursor = end
                }
            }
        }
        return output
    }

    static func validatePath(_ path: String) throws {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\0"),
            parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else { throw invalid("Unsafe artifact path") }
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func verify(_ data: Data, sha256: String) throws {
        guard sha256.count == 64, digest(data) == sha256.lowercased() else {
            throw invalid("SHA-256 mismatch")
        }
    }

    private static func decompress(_ data: Data, limit: Int) throws -> Data {
        guard let stream = ZSTD_createDStream() else {
            throw invalid("Could not allocate Zstandard decoder")
        }
        defer { ZSTD_freeDStream(stream) }
        guard ZSTD_isError(ZSTD_initDStream(stream)) == 0 else {
            throw invalid("Could not initialize Zstandard decoder")
        }
        var output = Data()
        var scratch = [UInt8](repeating: 0, count: 1 << 20)
        try data.withUnsafeBytes { source in
            var input = ZSTD_inBuffer(src: source.baseAddress, size: source.count, pos: 0)
            var remaining = 1
            repeat {
                let previous = input.pos
                try scratch.withUnsafeMutableBytes { destination in
                    var buffer = ZSTD_outBuffer(
                        dst: destination.baseAddress, size: destination.count, pos: 0)
                    remaining = ZSTD_decompressStream(stream, &buffer, &input)
                    guard ZSTD_isError(remaining) == 0, buffer.pos <= limit - output.count else {
                        throw invalid("Invalid or oversized Zstandard data")
                    }
                    output.append(
                        destination.bindMemory(to: UInt8.self).baseAddress!, count: buffer.pos)
                    if remaining != 0, previous == input.pos, buffer.pos == 0 {
                        throw invalid("Truncated Zstandard data")
                    }
                }
                if remaining == 0, input.pos != input.size {
                    throw invalid("Trailing Zstandard data")
                }
            } while remaining != 0
        }
        return output
    }

    private static func text(_ data: Data) -> String {
        String(decoding: data.prefix { $0 != 0 }, as: UTF8.self)
    }
    private static func octal(_ data: Data) throws -> Int {
        let value = text(data).trimmingCharacters(in: .whitespaces)
        guard let number = Int(value, radix: 8), number >= 0 else {
            throw invalid("Invalid tar size/checksum")
        }
        return number
    }
    private static func invalid(_ message: String) -> Phonon2Error { .invalidCheckpoint(message) }
}
