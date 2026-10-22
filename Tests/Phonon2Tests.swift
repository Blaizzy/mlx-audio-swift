import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

@Suite("Phonon-2", .serialized)
struct Phonon2Tests {
    private var tinyConfig: [String: Any] {
        let projections: [(String, [Int])] = [
            ("feed_forward1.linear1", [256, 128]), ("feed_forward1.linear2", [128, 256]),
            ("feed_forward2.linear1", [256, 128]), ("feed_forward2.linear2", [128, 256]),
            ("self_attn.linear_q", [128, 128]), ("self_attn.linear_k", [128, 128]),
            ("self_attn.linear_v", [128, 128]), ("self_attn.linear_out", [128, 128]),
            ("self_attn.linear_pos", [128, 128]),
            ("conv.pointwise_conv1", [256, 128]), ("conv.pointwise_conv2", [128, 128]),
        ]
        return [
            "model_type": "phonon2",
            "preprocessor": [
                "sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
                "window_stride": 0.01, "window": "hann", "features": 8, "n_fft": 512, "dither": 0,
            ],
            "encoder": [
                "feat_in": 8, "n_layers": 1, "d_model": 128, "n_heads": 4,
                "ff_expansion_factor": 2, "subsampling_factor": 2, "self_attention_model": "rel_pos",
                "subsampling": "dw_striding", "conv_kernel_size": 3, "subsampling_conv_channels": 4,
                "pos_emb_max_len": 128, "use_bias": false,
            ],
            "decoder": [
                "blank_as_pad": true, "vocab_size": 8,
                "prednet": ["pred_hidden": 16, "pred_rnn_layers": 1],
            ],
            "joint": [
                "num_classes": 8, "vocabulary": (0..<8).map { "▁word\($0)" }, "num_extra_outputs": 3,
                "jointnet": [
                    "joint_hidden": 16, "activation": "relu", "encoder_hidden": 128, "pred_hidden": 16,
                ],
            ],
            "decoding": ["model_type": "tdt", "durations": [0, 1, 2], "greedy": ["max_symbols": 4]],
            "phonon2_modules": projections.map { ["name": "encoder.layers.0.\($0.0)", "shape": $0.1] },
        ]
    }

    private func tinyWeights() throws -> [String: MLXArray] {
        let config = try JSONDecoder().decode(
            Phonon2Config.self, from: JSONSerialization.data(withJSONObject: tinyConfig))
        let modules: [(String, Module)] = [
            ("encoder", ParakeetConformer(args: config.parakeet.encoder)),
            ("decoder", ParakeetPredictNetwork(args: config.parakeet.decoder)),
            ("joint", ParakeetJointNetwork(args: config.parakeet.joint)),
        ]
        var weights: [String: MLXArray] = [:]
        for (prefix, module) in modules {
            for (name, value) in module.parameters().flattened() {
                weights["\(prefix).\(name)"] = MLXArray.zeros(value.shape, dtype: value.dtype)
            }
        }
        for projection in config.modules {
            weights.removeValue(forKey: "\(projection.name).weight")
            let layer = Phonon2PackedLinear(
                inputDimensions: projection.shape[1], outputDimensions: projection.shape[0], bias: false)
            for (name, value) in layer.parameters().flattened() {
                weights["\(projection.name).\(name)"] = value
            }
        }
        // Make the generated checkpoint emit blank tokens deterministically.
        let outputCount = config.parakeet.joint.numClasses + 1 + config.parakeet.joint.numExtraOutputs
        var bias = [Float](repeating: 0, count: outputCount)
        bias[config.parakeet.joint.vocabulary.count] = 10
        weights["joint.joint_net.bias"] = MLXArray(bias)
        return weights
    }

    private func checkpointDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: tinyConfig)
            .write(to: directory.appendingPathComponent("config.json"))
        try MLX.save(arrays: tinyWeights(), url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private func tinyModel(computeDType: DType = .bfloat16) throws -> Phonon2Model {
        let directory = try checkpointDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        return try Phonon2Model.fromDirectory(directory, computeDType: computeDType)
    }

    private func compare(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float) {
        #expect(actual.shape == expected.shape)
        let difference = (actual.asType(.float32) - expected.asType(.float32)).abs()
        let maximum = difference.max().item(Float.self)
        print("Phonon-2 parity \(actual.shape): max_abs=\(maximum)")
        #expect(maximum < tolerance)
    }

    @Test func packedProjectionRepresentsFiveValues() throws {
        let layer = Phonon2PackedLinear(inputDimensions: 128, outputDimensions: 5, bias: true)
        let baseCodes: [UInt32] = [0, 0, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF]
        let residualCodes: [UInt32] = [0, 0x55555555, 0x55555555, 0x55555555, 0x55555555]
        let bias = MLXArray([Float(0.25), -0.5, 0, 1, -1])
        try layer.update(parameters: ModuleParameters.unflattened([
            "base_q": MLXArray(baseCodes.flatMap { Array(repeating: $0, count: 8) }).reshaped(5, 8),
            "residual_q": MLXArray(residualCodes.flatMap { Array(repeating: $0, count: 8) }).reshaped(5, 8),
            "row_lo": MLXArray.ones([5], dtype: .float16),
            "row_hi": MLXArray.ones([5], dtype: .float16) * 2,
            "bias": bias.asType(.bfloat16),
        ]), verify: .all)
        layer.materialize()
        let input = arange(256, dtype: .float32).reshaped(2, 128) / 128
        let expected = input.sum(axis: -1, keepDims: true) * MLXArray([Float(-2), -1, 0, 1, 2]) + bias
        compare(layer(input), expected, tolerance: 2e-5)
        compare(layer(input.asType(.bfloat16)), expected.asType(.bfloat16), tolerance: 1e-6)
        #expect(layer.baseQ.dtype == .uint32)
        #expect(layer.rowLo.dtype == .float16)
    }

    @Test func tinyEncoderHandlesCompleteAndShortAudio() throws {
        let model = try tinyModel(computeDType: .float32)
        let waveform = sin(arange(1600, dtype: .float32) * 0.03)
        let features = model.backbone.makeBatchFeatures([waveform])
        #expect(features.features.dim(0) == 1)
        #expect(features.features.shape.last == 8)
        #expect(features.lengths.item(Int.self) > 0)
        let encoded = model.backbone.encoder(features.features, lengths: features.lengths)
        #expect(encoded.0.shape.last == 128)
        #expect(encoded.1.item(Int.self) > 0)
        #expect(MLX.all(isFinite(encoded.0)).item(Bool.self))
        let short = model.backbone.makeBatchFeatures([waveform[0..<1]])
        #expect(short.features.dim(1) > 0)
        #expect(MLX.all(isFinite(short.features)).item(Bool.self))
        #expect(model.generate(audio: waveform).text.isEmpty)
        #expect(model.generate(audio: MLXArray.zeros([0])).text.isEmpty)
        let parameters = Dictionary(uniqueKeysWithValues: model.backbone.parameters().flattened())
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.weight"] == nil)
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.base_q"]?.dtype == .uint32)
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.row_lo"]?.dtype == .float16)
    }

    @Test func bf16LoadPreservesLearnedPalettes() throws {
        let model = try tinyModel()
        let parameters = Dictionary(uniqueKeysWithValues: model.backbone.parameters().flattened())
        #expect(parameters["encoder.layers.0.feed_forward1.linear1.row_lo"]?.dtype == .float16)
        #expect(parameters["encoder.pre_encode.conv0.weight"]?.dtype == .bfloat16)
        #expect(model.generate(audio: .ones([1600])).text.isEmpty)
    }

    @Test func configurationRejectsMissingDuplicateOrMismatchedProjections() throws {
        let original = tinyConfig
        let data = try JSONSerialization.data(withJSONObject: original)
        let config = try JSONDecoder().decode(Phonon2Config.self, from: data)
        #expect(config.modules.count == 11)
        #expect(config.parakeet.decoding.durations == [0, 1, 2])
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
        try JSONSerialization.data(withJSONObject: tinyConfig)
            .write(to: temporary.appendingPathComponent("config.json"))
        let original = try tinyWeights()
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
        let directory = try checkpointDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try await STT.loadModel(modelRepo: directory.path, modelType: "phonon2")
        #expect(model is Phonon2Model)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLX_AUDIO_PHONON2_CHECKPOINT"] != nil))
    func localCheckpointTranscription() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_AUDIO_PHONON2_CHECKPOINT"])
        let model = try Phonon2Model.fromDirectory(URL(fileURLWithPath: path))
        let audioURL = try #require(Bundle.module.url(
            forResource: "conversational_a", withExtension: "wav", subdirectory: "media"))
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
        #expect(model.generate(audio: audio).text.split(separator: " ").count > 3)
    }
}
