import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

private final class ReduxTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Int] = []
    var values: [Int] { lock.withLock { frames } }
    func append(_ frame: Int) { lock.withLock { frames.append(frame) } }
    func reset() { lock.withLock { frames.removeAll() } }
}

@Suite("Parakeet Redux", .serialized)
struct ParakeetReduxTests {
    private let projections: [(String, String, [Int])] = [
        ("feed_forward1.linear1", "feed_forward1.linear1", [256, 128]),
        ("feed_forward1.linear2", "feed_forward1.linear2", [128, 256]),
        ("feed_forward2.linear1", "feed_forward2.linear1", [256, 128]),
        ("feed_forward2.linear2", "feed_forward2.linear2", [128, 256]),
        ("self_attn.q_proj", "self_attn.linear_q", [128, 128]),
        ("self_attn.k_proj", "self_attn.linear_k", [128, 128]),
        ("self_attn.v_proj", "self_attn.linear_v", [128, 128]),
        ("self_attn.o_proj", "self_attn.linear_out", [128, 128]),
        ("self_attn.relative_k_proj", "self_attn.linear_pos", [128, 128]),
        ("conv.pointwise_conv1", "conv.pointwise_conv1", [256, 128]),
        ("conv.pointwise_conv2", "conv.pointwise_conv2", [128, 128]),
    ]

    private var tinyConfig: [String: Any] {
        ["model_type": "parakeet_tdt", "redux_weight_format": "mlx-2bit",
         "encoder_config": ["hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 4, "num_mel_bins": 8,
            "conv_kernel_size": 3, "subsampling_factor": 2, "subsampling_conv_channels": 4,
            "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
            "max_position_embeddings": 128, "hidden_act": "silu"],
         "decoder_hidden_size": 8, "num_decoder_layers": 1, "hidden_act": "relu",
         "blank_token_id": 3, "vocab_size": 4, "durations": [0, 1, 2], "max_symbols_per_step": 3,
         "ternary_group_size": 128, "ternary_modules": projections.map { "encoder.layers.0." + $0.0 }]
    }

    private func checkpoint() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("redux-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let data = try JSONSerialization.data(withJSONObject: tinyConfig)
            try data.write(to: directory.appendingPathComponent("config.json"))
            let tokenizer: [String: Any] = ["model": ["vocab": ["<unk>": 0, "▁hello": 1, "▁world": 2]],
                "added_tokens": [["content": "<blank>", "id": 3]]]
            try JSONSerialization.data(withJSONObject: tokenizer).write(to: directory.appendingPathComponent("tokenizer.json"))
            let redux = try ParakeetReduxConfig.decode(data)
            let config = try ParakeetConfigParser.parseTDT(redux.modelConfig(tokenizerAt: directory.appendingPathComponent("tokenizer.json")))
            let modules: [(String, Module)] = [("encoder", ParakeetConformer(args: config.encoder)),
                ("decoder", ParakeetPredictNetwork(args: config.decoder)), ("joint", ParakeetJointNetwork(args: config.joint))]
            var weights: [String: MLXArray] = [:]
            for (prefix, module) in modules {
                for (key, value) in module.parameters().flattened() {
                    let name = prefix + "." + key
                    let phase = Float(name.utf8.reduce(0) { $0 + Int($1) } % 31)
                    var weight = sin(arange(value.size, dtype: .float32) * 0.13 + phase).reshaped(value.shape) * 0.02
                    if key.contains("norm"), key.hasSuffix("weight") || key.hasSuffix("running_var") { weight = weight + 1 }
                    weights[name] = weight
                }
            }
            for (_, target, shape) in projections {
                let name = "encoder.layers.0." + target
                weights[name + ".weight"] = MLXArray(Array(repeating: UInt32(0x11111111), count: shape[0] * shape[1] / 16)).reshaped(shape[0], shape[1] / 16)
                weights[name + ".scales"] = MLXArray.ones([shape[0], shape[1] / 128], dtype: .float16) * 0.01
                weights[name + ".biases"] = -weights[name + ".scales"]!
            }
            // Force blank with duration zero to exercise Redux's immediate advance.
            weights["joint.joint_net.weight"] = MLXArray.zeros([7, 8])
            weights["joint.joint_net.bias"] = MLXArray([Float(0), 0, 0, 10, 10, 0, 0])
            try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
            return directory
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private func publishedCheckpoint() throws -> URL {
        let directory = try checkpoint()
        do {
            let file = directory.appendingPathComponent("model.safetensors")
            let weights = try MLX.loadArrays(url: file)
            eval(Array(weights.values))
            var source: [String: MLXArray] = [:]
            let quantized = Set(projections.map { "encoder.layers.0." + $0.1 })
            for (name, original) in weights {
                if name.hasSuffix(".biases") { continue }
                var value = original
                var target = name
                if name.hasSuffix(".weight"), quantized.contains(String(name.dropLast(7))) {
                    let module = String(name.dropLast(7))
                    let projection = try #require(projections.first { module == "encoder.layers.0." + $0.1 })
                    let words = value.asArray(UInt32.self)
                    let rows = projection.2[0], columns = projection.2[1], rowBytes = (columns + 4) / 5
                    var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
                    let powers = [1, 3, 9, 27, 81]
                    for row in 0..<rows {
                        for column in 0..<columns {
                            let code = (words[row * (columns / 16) + column / 16] >> (2 * (column % 16))) & 3
                            bytes[row * rowBytes + column / 5] += UInt8(Int(code) * powers[column % 5])
                        }
                    }
                    source["encoder.layers.0." + projection.0 + ".qweight"] = MLXArray(bytes).reshaped(rows, rowBytes)
                    continue
                }
                if name == "decoder.prediction.dec_rnn.lstm.0.bias" {
                    source["decoder.lstm.bias_ih_l0"] = value * 0.5
                    source["decoder.lstm.bias_hh_l0"] = value * 0.5
                    continue
                }
                let mappings = [
                    ("encoder.pre_encode.conv0", "encoder.subsampling.layers.0"),
                    ("encoder.pre_encode.out", "encoder.subsampling.linear"),
                    (".conv.batch_norm.", ".conv.norm."),
                    (".self_attn.linear_q", ".self_attn.q_proj"), (".self_attn.linear_k", ".self_attn.k_proj"),
                    (".self_attn.linear_v", ".self_attn.v_proj"), (".self_attn.linear_out", ".self_attn.o_proj"),
                    (".self_attn.linear_pos", ".self_attn.relative_k_proj"),
                    (".self_attn.posBiasU", ".self_attn.bias_u"), (".self_attn.posBiasV", ".self_attn.bias_v"),
                    ("decoder.prediction.embed.", "decoder.embedding."),
                    ("decoder.prediction.dec_rnn.lstm.0.Wx", "decoder.lstm.weight_ih_l0"),
                    ("decoder.prediction.dec_rnn.lstm.0.Wh", "decoder.lstm.weight_hh_l0"),
                    ("joint.pred.", "decoder.decoder_projector."), ("joint.enc.", "encoder_projector."),
                    ("joint.joint_net.", "joint.head."),
                ]
                for (old, new) in mappings { target = target.replacingOccurrences(of: old, with: new) }
                if target.hasSuffix(".weight") {
                    if value.ndim == 3 { value = value.transposed(0, 2, 1) }
                    if value.ndim == 4 { value = value.transposed(0, 3, 1, 2) }
                }
                source[target] = value
            }
            source["encoder.layers.0.conv.norm.num_batches_tracked"] = MLXArray(Int64(0))
            for layer in ["proj", "ctx", "out"] {
                for parameter in ["weight", "bias"] { source["vad_head.\(layer).\(parameter)"] = MLXArray.zeros([1]) }
            }
            var config = tinyConfig
            config.removeValue(forKey: "redux_weight_format")
            try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
            let manifest: [String: Any] = ["format": "thrush-ternary-v2", "names": "hf",
                "quant": ["mode": "ternary", "group_size": 128],
                "packing": ["base": 3, "elements_per_byte": 5, "code_offset": 1],
                "quantized_modules": projections.map {
                    ["name": "encoder.layers.0." + $0.0, "out_features": $0.2[0], "in_features": $0.2[1],
                     "group_size": 128, "has_bias": false, "as_conv1d": $0.0.hasPrefix("conv.")]
                }]
            try JSONSerialization.data(withJSONObject: manifest).write(to: directory.appendingPathComponent("ternary.json"))
            try MLX.save(arrays: source, url: file)
            return directory
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    @Test func ternaryRepackingPreservesCodesAcrossByteAndWordBoundaries() throws {
        let rows = 3, columns = 128, rowBytes = (columns + 4) / 5
        let powers = [1, 3, 9, 27, 81]
        var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
        var expected = [UInt32](repeating: 0, count: rows * columns / 16)
        for row in 0..<rows {
            for column in 0..<rowBytes * 5 {
                let code = (row + column) % 3
                bytes[row * rowBytes + column / 5] += UInt8(code * powers[column % 5])
                if column < columns { expected[row * (columns / 16) + column / 16] |= UInt32(code) << (2 * (column % 16)) }
            }
        }
        let packed = try ParakeetReduxConfig.repackTernary(MLXArray(bytes).reshaped(rows, rowBytes), inputDimensions: columns)
        #expect(packed.dtype == .uint32)
        #expect(packed.asArray(UInt32.self) == expected)
        bytes[0] = 243
        #expect(throws: STTError.self) {
            try ParakeetReduxConfig.repackTernary(MLXArray(bytes).reshaped(rows, rowBytes), inputDimensions: columns)
        }
    }

    @Test func publishedWeightsMatchPreparedWeightsExactly() async throws {
        let source = try publishedCheckpoint(), prepared = try checkpoint()
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: prepared)
        }
        let model = try #require(try await STT.loadModel(modelRepo: source.path) as? ParakeetModel)
        let expected = try ParakeetModel.fromDirectory(prepared)
        let actualWeights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let expectedWeights = Dictionary(uniqueKeysWithValues: expected.parameters().flattened())
        #expect(Set(actualWeights.keys) == Set(expectedWeights.keys))
        for (name, actual) in actualWeights {
            let reference = try #require(expectedWeights[name])
            #expect(actual.shape == reference.shape)
            #expect(actual.dtype == reference.dtype)
            #expect(MLX.all(actual .== reference).item(Bool.self), "\(name)")
        }
        #expect(model.generate(audio: MLXArray.ones([1600])).text.isEmpty)
    }

    @Test func rejectsUnsupportedManifestPacking() throws {
        let directory = try publishedCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("ternary.json")
        var manifest = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        manifest["packing"] = ["base": 2, "elements_per_byte": 5, "code_offset": 1]
        try JSONSerialization.data(withJSONObject: manifest).write(to: file)
        #expect(throws: STTError.self) { try ParakeetModel.fromDirectory(directory) }
    }

    @Test func preparedWeightsLoadThroughSharedModelType() async throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let loaded = try await STT.loadModel(modelRepo: directory.path)
        let model = try #require(loaded as? ParakeetModel)
        #expect(model.computeDType == .float32)
        #expect(model.preprocessConfig.normalizeValidFrames)
        #expect(model.encoderConfig.maskPadding)
        #expect(model.vocabulary == ["<unk>", "▁hello", "▁world"])
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(params["encoder.layers.0.feed_forward1.linear1.weight"]?.dtype == .uint32)
        #expect(params["encoder.layers.0.feed_forward1.linear1.scales"]?.dtype == .float16)
        let bf16 = try ParakeetModel.fromDirectory(directory, computeDType: .bfloat16)
        let bf16Params = Dictionary(uniqueKeysWithValues: bf16.parameters().flattened())
        #expect(bf16Params["encoder.layers.0.feed_forward1.linear1.scales"]?.dtype == .float16)
        #expect(bf16Params["encoder.pre_encode.conv0.weight"]?.dtype == .bfloat16)
    }

    @Test func frontendNormalizesOnlyValidFrames() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try ParakeetModel.fromDirectory(directory)
        let audio = sin(arange(1600, dtype: .float32) * 0.03)
        let mel = ParakeetAudio.logMelSpectrogram(audio, config: model.preprocessConfig)
        #expect(mel.shape == [1, 11, 8])
        #expect(mel[0, 10, 0...].abs().max().item(Float.self) == 0)
        #expect(mel[0, 0..<10, 0...].mean(axis: 0).abs().max().item(Float.self) < 1e-5)
        #expect(model.makeBatchFeatures([audio]).lengths.item(Int.self) == 10)
        #expect(model.generate(audio: MLXArray.ones([1])).text.isEmpty)
    }

    @Test func paddingDoesNotChangeValidEncoderFrames() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try ParakeetModel.fromDirectory(directory)
        let audio = sin(arange(1600, dtype: .float32) * 0.03)
        let single = model.makeBatchFeatures([audio])
        let batch = model.makeBatchFeatures([audio, concatenated([audio, audio])])
        let expected = model.encodeBatchFeatures(single.features, lengths: single.lengths)
        let actual = model.encodeBatchFeatures(batch.features, lengths: batch.lengths)
        let valid = expected.1.item(Int.self)
        #expect(valid == 5)
        #expect(actual.1[0].item(Int.self) == valid)
        let error = (actual.0[0, 0..<valid, 0...] - expected.0[0, 0..<valid, 0...]).abs().max().item(Float.self)
        #expect(error < 1e-4)
        let results = try model.generateBatch(audios: [audio, concatenated([audio, audio])])
        #expect(results.map(\.text) == ["", ""])
    }

    @Test func blankAdvancesAndZeroDurationEmissionsAreBounded() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try ParakeetModel.fromDirectory(directory)
        let trace = ReduxTrace()
        model.tdtTraceEmitter = { step in trace.append(step.time) }
        #expect(model.generate(audio: MLXArray.ones([1600])).text.isEmpty)
        #expect(trace.values == [0, 1, 2, 3, 4])
        trace.reset()
        try model.update(parameters: ModuleParameters.unflattened([
            "joint.joint_net.bias": MLXArray([Float(0), 10, 0, 0, 10, 0, 0]),
        ]), verify: .noUnusedKeys)
        #expect(!model.generate(audio: MLXArray.ones([1600])).text.isEmpty)
        #expect(trace.values.count == 15)
        #expect(trace.values.allSatisfy { $0 == 0 })
    }

    @Test func rejectsUnknownFormatsAndInvalidVocabulary() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        var invalid = tinyConfig
        invalid["redux_weight_format"] = "unknown"
        #expect(throws: STTError.self) { try ParakeetReduxConfig.decode(JSONSerialization.data(withJSONObject: invalid)) }
        let tokenizer: [String: Any] = ["model": ["vocab": ["hello": 0, "world": 2]], "added_tokens": [["content": "<blank>", "id": 3]]]
        try JSONSerialization.data(withJSONObject: tokenizer).write(to: directory.appendingPathComponent("tokenizer.json"))
        #expect(throws: STTError.self) { try ParakeetModel.fromDirectory(directory) }
    }

    @Test func rejectsMissingOrIncorrectlyTypedPackedWeights() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: file)
        eval(Array(weights.values))
        let name = "encoder.layers.0.feed_forward1.linear1.weight"
        var invalid = weights
        invalid[name] = invalid[name]!.asType(.float32)
        try MLX.save(arrays: invalid, url: file)
        #expect(throws: STTError.self) { try ParakeetModel.fromDirectory(directory) }
        invalid = weights
        invalid.removeValue(forKey: "encoder.layers.0.conv.pointwise_conv1.scales")
        try MLX.save(arrays: invalid, url: file)
        #expect(throws: STTError.self) { try ParakeetModel.fromDirectory(directory) }
    }
}

@Suite("Parakeet Redux checkpoint", .serialized)
struct ParakeetReduxCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["PARAKEET_REDUX_MODEL_DIR"] != nil))
    func localCheckpointTranscription() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["PARAKEET_REDUX_MODEL_DIR"])
        let loaded = try await STT.loadModel(modelRepo: path)
        let model = try #require(loaded as? ParakeetModel)
        for name in ["intention", "conversational_a", "noisy_audio"] {
            let url = try #require(Bundle.module.url(forResource: name, withExtension: "wav", subdirectory: "media"))
            let (_, audio) = try loadAudioArray(from: url, sampleRate: 16000)
            let parameters = STTGenerateParameters(chunkDuration: 30)
            let output = model.generate(audio: audio, generationParameters: parameters)
            print("REDUX_TRANSCRIPT \(name): \(output.text)")
            #expect(!output.text.isEmpty)
            var streamed: STTOutput?
            for try await event in model.generateStream(audio: audio, generationParameters: parameters) {
                if case .result(let result) = event { streamed = result }
            }
            #expect(streamed?.text == output.text)
        }
    }
}
