import Foundation
import HuggingFace
import MLX
import Testing
import libzstd

@testable import MLXAudioSTT

@Suite("Phonon published checkpoints", .serialized)
struct Phonon2ContainerTests {
    private func directory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func container(_ rows: [(String, String, [Int], Data)], at directory: URL) throws -> URL
    {
        let header = try JSONSerialization.data(withJSONObject: [
            "format": Phonon2Container.format,
            "index": rows.map { ["n": $0.0, "k": $0.1, "shape": $0.2, "b": $0.3.count] },
        ])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        for row in rows { data.append(row.3) }
        let url = directory.appendingPathComponent("model.fermion")
        try data.write(to: url)
        return url
    }

    private func half(_ values: [Float]) -> Data {
        values.map { Float16($0) }.withUnsafeBytes { Data($0) }
    }

    @Test func fiveValueDecodingPreservesSignsPalettesAndHighMask() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // 128 trits per row, padded to 130; all five values appear, including zero.
        let codes = (0 ..< 256).map { $0 % 3 }
        var trits = [UInt8]()
        for row in 0 ..< 2 {
            for group in 0 ..< 26 {
                var byte = 0
                var factor = 1
                for k in 0 ..< 5 {
                    let column = group * 5 + k
                    byte += (column < 128 ? codes[row * 128 + column] : 1) * factor
                    factor *= 3
                }
                trits.append(UInt8(byte))
            }
        }
        let count = codes.filter { $0 != 1 }.count
        var mask = [UInt8](repeating: 0, count: (count + 7) / 8)
        for i in 0 ..< count where i % 2 == 0 { mask[i / 8] |= 1 << (i % 8) }
        var blob = Data(trits + mask)
        blob.append(half([1, 2]))
        blob.append(half([3, 4]))
        let url = try container(
            [("encoder.layers.0.self_attn.q_proj", "five_value", [2, 128], blob)], at: directory)
        let result = try Phonon2Container.read(url)
        #expect(result.modules.first?.name == "encoder.layers.0.self_attn.linear_q")
        let prefix = "encoder.layers.0.self_attn.linear_q."
        let base = try #require(result.weights[prefix + "base_q"]).asArray(UInt32.self)
        let residual = try #require(result.weights[prefix + "residual_q"]).asArray(UInt32.self)
        var nonzero = 0
        for i in codes.indices {
            let code = Int((base[i / 16] >> (2 * (i % 16))) & 3)
            let high = codes[i] != 1 && nonzero % 2 == 0
            #expect(code == codes[i])
            #expect(Int((residual[i / 16] >> (2 * (i % 16))) & 3) == (high ? code : 1))
            if codes[i] != 1 { nonzero += 1 }
        }
        #expect(result.weights[prefix + "row_lo"]?.dtype == .float16)
        #expect(result.weights[prefix + "row_hi"]?.asType(.float32).asArray(Float.self) == [3, 4])
    }

    @Test func integerTablesAndLSTMBiasesMatchReferenceRounding() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Four 6-bit values [-32, -1, 0, 31], with one padded value discarded.
        let word: UInt32 = 0 | (31 << 6) | (32 << 12) | (63 << 18)
        var int6 = Data([UInt8(word & 255), UInt8((word >> 8) & 255), UInt8((word >> 16) & 255)])
        int6.append(half([0.5]))
        var int8 = Data([128, 255, 0, 127])
        int8.append(half([0.25, 2]))
        let url = try container(
            [
                ("encoder_projector.weight", "int6", [1, 3], int6),
                ("decoder.embedding.weight", "int8", [2, 2], int8),
                ("decoder.lstm.bias_ih_l0", "fp16", [2], half([0.33325, 0.125])),
                ("decoder.lstm.bias_hh_l0", "fp16", [2], half([0.6665, -0.25])),
                ("encoder.subsampling.layers.0.weight", "fp16", [1, 2, 1, 2], half([1, 2, 3, 4])),
                (
                    "encoder.layers.0.conv.depthwise_conv.weight", "fp16", [2, 1, 2],
                    half([1, 2, 3, 4])
                ),
            ], at: directory)
        let weights = try Phonon2Container.read(url).weights
        #expect(
            weights["joint.enc.weight"]?.asType(.float32).asArray(Float.self) == [-16, -0.5, 0])
        #expect(
            weights["decoder.prediction.embed.weight"]?.asType(.float32).asArray(Float.self) == [
                -32, -0.25, 0, 254,
            ])
        let expected =
            (MLXArray(half([0.33325, 0.125]), dtype: .float16).asType(.float32)
            + MLXArray(half([0.6665, -0.25]), dtype: .float16).asType(.float32)).asType(.float16)
            .asType(.bfloat16)
        #expect(
            weights["decoder.prediction.dec_rnn.lstm.0.bias"]?.asType(.float32).asArray(Float.self)
                == expected.asType(.float32).asArray(Float.self))
        #expect(weights["encoder.pre_encode.conv.0.weight"]?.shape == [1, 1, 2, 2])
        #expect(
            weights["encoder.pre_encode.conv.0.weight"]?.asType(.float32).asArray(Float.self) == [
                1, 3, 2, 4,
            ])
        #expect(weights["encoder.layers.0.conv.depthwise_conv.weight"]?.shape == [2, 2, 1])
    }

    @Test func rejectsInvalidContainersAndUnsafePaths() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        for rows: [(String, String, [Int], Data)] in [
            [("x", "int6", [1, 4], Data([1]))],
            [("x", "unknown", [1], Data())],
            [("x", "fp16", [-1], Data())],
            [("x", "fp16", [1], half([0])), ("x", "fp16", [1], half([0]))],
            [("x", "five_value", [1, 128], Data(repeating: 255, count: 30))],
        ] {
            let url = try container(rows, at: directory)
            #expect(throws: (any Error).self) { try Phonon2Container.read(url) }
        }
        for path in ["../model", "/model", "a/../b", "a\\b", "", "a//b"] {
            #expect(throws: Phonon2Error.self) { try Phonon2Archive.validatePath(path) }
        }
        let url = try container([], at: directory)
        var trailing = try Data(contentsOf: url)
        trailing.append(0)
        try trailing.write(to: url)
        #expect(throws: Phonon2Error.self) { try Phonon2Container.read(url) }
        #expect(throws: Phonon2Error.self) {
            try Phonon2Archive.verify(Data([1]), sha256: String(repeating: "0", count: 64))
        }
    }

    @Test func bytePlaneReconstructionAndBounds() throws {
        let metadata = Data(
            #"{"original_bytes":8,"base":2,"payload_bytes":6,"plan":[[0,4,2]]}"#.utf8)
        let transform = try JSONDecoder().decode(Phonon2Archive.Transform.self, from: metadata)
        #expect(
            try Phonon2Archive.joinBytePlanes(Data([9, 8, 1, 3, 2, 4, 7, 6]), transform: transform)
                == Data([9, 8, 1, 2, 3, 4, 7, 6]))
        #expect(throws: Phonon2Error.self) {
            try Phonon2Archive.joinBytePlanes(Data([1]), transform: transform)
        }
    }
    private func archive(_ members: [(String, Data)], at directory: URL) throws -> URL {
        let rows = members.map {
            [
                "path": $0.0, "original_bytes": $0.1.count,
                "original_sha256": Phonon2Archive.digest($0.1),
            ]
        }
        let manifest = try JSONSerialization.data(withJSONObject: [
            "release_format": "phonon-byteplane-tar-zstd-v1", "files": rows,
        ])
        var tar = Data()
        for (name, body) in [("bps_manifest.json", manifest)] + members {
            var header = [UInt8](repeating: 0, count: 512)
            header.replaceSubrange(0 ..< name.utf8.count, with: name.utf8)
            let size = String(format: "%011o", body.count)
            header.replaceSubrange(124 ..< 135, with: size.utf8)
            header[156] = 48
            for i in 148 ..< 156 { header[i] = 32 }
            let checksum = header.reduce(0) { $0 + Int($1) }
            let text = String(format: "%06o", checksum)
            header.replaceSubrange(148 ..< 154, with: text.utf8)
            header[154] = 0
            tar.append(contentsOf: header)
            tar.append(body)
            tar.append(Data(repeating: 0, count: (512 - body.count % 512) % 512))
        }
        tar.append(Data(repeating: 0, count: 1024))
        var compressed = Data(count: ZSTD_compressBound(tar.count))
        let count = compressed.withUnsafeMutableBytes { destination in
            tar.withUnsafeBytes { source in
                ZSTD_compress(
                    destination.baseAddress, destination.count, source.baseAddress, source.count, 1)
            }
        }
        #expect(ZSTD_isError(count) == 0)
        compressed.count = count
        let url = directory.appendingPathComponent("test.bps.tar.zst")
        try compressed.write(to: url)
        return url
    }

    @Test func publishedArchiveChecksMembersSizesAndHashes() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("output")
        let members = [("config.json", Data("{}".utf8)), ("model.fermion", Data([1, 2, 3]))]
        let url = try archive(members, at: directory)
        try Phonon2Archive.extract(url, to: output, expectedBytes: 5)
        #expect(
            try Data(contentsOf: output.appendingPathComponent("model.fermion")) == Data([1, 2, 3]))
        #expect(throws: Phonon2Error.self) {
            try Phonon2Archive.extract(url, to: output, expectedBytes: 4)
        }
        let unsafe = try archive([("../outside", Data([1]))], at: directory)
        #expect(throws: Phonon2Error.self) {
            try Phonon2Archive.extract(unsafe, to: output, expectedBytes: 1)
        }
        let duplicate = try archive([("model", Data([1])), ("model", Data([2]))], at: directory)
        #expect(throws: Phonon2Error.self) {
            try Phonon2Archive.extract(duplicate, to: output, expectedBytes: 2)
        }
        let valid = try archive(members, at: directory)
        var truncated = try Data(contentsOf: valid)
        truncated.removeLast()
        try truncated.write(to: url)
        #expect(throws: Phonon2Error.self) {
            try Phonon2Archive.extract(url, to: output, expectedBytes: 5)
        }
    }

}
