import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

@Suite("Phonon-2", .serialized)
struct Phonon2Tests {
    private var fixtures: URL {
        Bundle.module.resourceURL!.appendingPathComponent("media/phonon2")
    }

    private func compare(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float) {
        #expect(actual.shape == expected.shape)
        let difference = (actual.asType(.float32) - expected.asType(.float32)).abs()
        let maximum = difference.max().item(Float.self)
        print("Phonon-2 parity \(actual.shape): max_abs=\(maximum)")
        #expect(maximum < tolerance)
    }

    @Test func packedProjectionMatchesPython() throws {
        var weights = try MLX.loadArrays(url: fixtures.appendingPathComponent("expected/packed.safetensors"))
        let input = weights.removeValue(forKey: "input")!
        let expected = weights.removeValue(forKey: "expected")!
        let expectedBF16 = weights.removeValue(forKey: "expected_bf16")!
        let layer = Phonon2PackedLinear(inputDimensions: 128, outputDimensions: 3, bias: true)
        try layer.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        layer.materialize()
        compare(layer(input), expected, tolerance: 2e-5)
        compare(layer(input.asType(.bfloat16)), expectedBF16, tolerance: 1e-6)
        #expect(layer.baseQ.dtype == .uint32)
        #expect(layer.rowLo.dtype == .float16)
    }

    @Test func completeEncoderAndFrontendMatchPython() throws {
        let model = try Phonon2Model.fromDirectory(fixtures, computeDType: .float32)
        let reference = try MLX.loadArrays(url: fixtures.appendingPathComponent("expected/reference.safetensors"))
        let waveform = reference["waveform"]!
        let features = model.backbone.makeBatchFeatures([waveform])
        compare(features.features, reference["mel"]!, tolerance: 2e-4)
        let encoded = model.backbone.encoder(features.features, lengths: features.lengths)
        compare(encoded.0, reference["encoded"]!, tolerance: 2e-3)
        compare(encoded.1, reference["lengths"]!, tolerance: 1e-6)
        let short = model.backbone.makeBatchFeatures([waveform[0..<1]])
        compare(short.features, reference["short_mel"]!, tolerance: 2e-4)
        #expect(model.generate(audio: waveform).text.isEmpty) // Fixture forces blank output.
        #expect(model.generate(audio: MLXArray.zeros([0])).text.isEmpty)
        let parameters = Dictionary(uniqueKeysWithValues: model.backbone.parameters().flattened())
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.weight"] == nil)
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.base_q"]?.dtype == .uint32)
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.row_lo"]?.dtype == .float16)
    }

    @Test func bf16LoadPreservesLearnedPalettes() throws {
        let model = try Phonon2Model.fromDirectory(fixtures)
        let parameters = Dictionary(uniqueKeysWithValues: model.backbone.parameters().flattened())
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.row_lo"]?.dtype == .float16)
        #expect(parameters["encoder.pre_encode.conv0.weight"]?.dtype == .bfloat16)
        let reference = try MLX.loadArrays(url: fixtures.appendingPathComponent("expected/reference.safetensors"))
        #expect(model.generate(audio: reference["waveform"]!).text.isEmpty)
    }

    @Test func configurationRejectsMissingDuplicateOrMismatchedProjections() throws {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Phonon2Config.self, from: data)
        #expect(config.modules.count == 11)
        #expect(config.parakeet.decoding.durations == [0, 1, 2])
        let original = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for change in 0..<4 {
            var invalid = original
            var rows = try #require(original["phonon2_modules"] as? [[String: Any]])
            if change == 0 { rows.removeLast() }
            if change == 1 { rows[1] = rows[0] }
            if change == 2 { rows[0]["shape"] = [256, 64] }
            invalid["phonon2_modules"] = rows
            if change == 3 {
                var decoding = try #require(invalid["decoding"] as? [String: Any])
                decoding["durations"] = [0, -1, 2]
                invalid["decoding"] = decoding
            }
            let invalidData = try JSONSerialization.data(withJSONObject: invalid)
            #expect(throws: (any Error).self) { try JSONDecoder().decode(Phonon2Config.self, from: invalidData) }
        }
    }

    @Test func strictLoadingRejectsCorruptPackedTypesAndMissingWeights() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("config.json"), to: temporary.appendingPathComponent("config.json"))
        let original = try MLX.loadArrays(url: fixtures.appendingPathComponent("model.safetensors"))
        for key in ["encoder.layers.0.feed_forward1.linear1.base_q", "encoder.layers.0.feed_forward1.linear1.row_lo"] {
            var invalid = original
            invalid[key] = invalid[key]!.asType(.float32)
            try MLX.save(arrays: invalid, url: temporary.appendingPathComponent("model.safetensors"))
            #expect(throws: Phonon2Error.self) { try Phonon2Model.fromDirectory(temporary) }
        }
        var missing = original
        missing.removeValue(forKey: "encoder.layers.0.feed_forward1.linear1.base_q")
        try MLX.save(arrays: missing, url: temporary.appendingPathComponent("model.safetensors"))
        #expect(throws: (any Error).self) { try Phonon2Model.fromDirectory(temporary) }
    }

    @Test func commonLoaderDispatchesPhonon2() async throws {
        let model = try await STT.loadModel(modelRepo: fixtures.path, modelType: "phonon2")
        #expect(model is Phonon2Model)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLX_AUDIO_PHONON2_CHECKPOINT"] != nil))
    func realCheckpointParity() throws {
        let environment = ProcessInfo.processInfo.environment
        let checkpoint = URL(fileURLWithPath: try #require(environment["MLX_AUDIO_PHONON2_CHECKPOINT"]))
        let referenceDirectory = URL(fileURLWithPath: try #require(environment["MLX_AUDIO_PHONON2_REFERENCE"]))
        let model = try Phonon2Model.fromDirectory(checkpoint, computeDType: .float32)
        let reference = try MLX.loadArrays(url: referenceDirectory.appendingPathComponent("reference.safetensors"))
        let features = model.backbone.makeBatchFeatures([reference["waveform"]!])
        compare(features.features, reference["mel"]!, tolerance: 1e-3)
        let encoded = model.backbone.encoder(features.features, lengths: features.lengths)
        compare(encoded.0, reference["encoded"]!, tolerance: 1e-2)
        let output = model.generate(audio: reference["waveform"]!)
        let expected = try String(contentsOf: referenceDirectory.appendingPathComponent("transcript.txt"), encoding: .utf8)
        print("Phonon-2 real checkpoint Swift: \(output.text)")
        print("Phonon-2 real checkpoint Python: \(expected)")
        #expect(!output.text.isEmpty)
        let errorRate = wordErrorRate(output.text, reference: expected)
        print("Phonon-2 real checkpoint Python/Swift WER: \(errorRate)")
        #expect(errorRate <= 0.1)
        let bf16Model = try Phonon2Model.fromDirectory(checkpoint)
        let bf16Output = bf16Model.generate(audio: reference["waveform"]!)
        let bf16Expected = try String(contentsOf: referenceDirectory.appendingPathComponent("transcript_bf16.txt"), encoding: .utf8)
        let bf16ErrorRate = wordErrorRate(bf16Output.text, reference: bf16Expected)
        print("Phonon-2 real checkpoint BF16 Swift: \(bf16Output.text)")
        print("Phonon-2 real checkpoint BF16 Python/Swift WER: \(bf16ErrorRate)")
        #expect(!bf16Output.text.isEmpty)
        #expect(bf16ErrorRate <= 0.1)
    }

    private func wordErrorRate(_ hypothesis: String, reference: String) -> Double {
        func words(_ value: String) -> [String] {
            value.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
        }
        let expected = words(reference)
        let actual = words(hypothesis)
        var previous = Array(0...actual.count)
        for (i, word) in expected.enumerated() {
            var current = [i + 1]
            for (j, candidate) in actual.enumerated() {
                current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (word == candidate ? 0 : 1)))
            }
            previous = current
        }
        return Double(previous.last!) / Double(max(1, expected.count))
    }
}
