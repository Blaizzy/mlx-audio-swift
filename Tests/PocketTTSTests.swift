import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs

@Suite("Pocket TTS and Phonon", .serialized)
struct PocketTTSTests {
    private func configuration() -> [String: Any] {
        [
            "model_type": "pocket_tts",
            "flow_lm": [
                "d_model": 32, "num_heads": 4, "num_layers": 1,
                "dim_feedforward": 80, "max_period": 10000,
                "n_bins": 8, "lut_dim": 32, "flow_dim": 8, "flow_depth": 1, "ldim": 4,
            ],
            "mimi": [
                "channels": 1, "sample_rate": 16, "frame_rate": 2,
                "dimension": 8, "quantizer_dimension": 4, "quantizer_output_dimension": 8,
                "n_filters": 2, "n_residual_layers": 1, "ratios": [2],
                "kernel_size": 3, "last_kernel_size": 3, "residual_kernel_size": 3,
                "dilation_base": 2, "compress": 2,
                "transformer_d_model": 8, "transformer_num_heads": 2, "transformer_num_layers": 1,
                "transformer_layer_scale": 0.01, "transformer_context": 16,
                "transformer_max_period": 10000, "transformer_dim_feedforward": 16,
                "encoder_enabled": false,
            ],
            "temp": 0.8, "lsd_decode_steps": 2, "eos_threshold": 4.0,
            "conditioners": [["name": "num_speakers", "type": "lut", "lut": ["n_bins": 3, "dim": 4]]],
            "fuser": ["sum": ["num_speakers"], "prepend": ["transcript_in_segment", "speaker_wavs"]],
        ]
    }

    private func decode(_ values: [String: Any]) throws -> PocketTTSModelConfig {
        try JSONDecoder().decode(PocketTTSModelConfig.self, from: JSONSerialization.data(withJSONObject: values))
    }

    private func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let tokenizer = """
        {"model":{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",0],["▁Hello",-1],["▁world",-1],[".",-1],["!",-1],["?",-1],["▁",-2],["x",-3]
        ]}}
        """
        try Data(tokenizer.utf8).write(to: folder.appendingPathComponent("tokenizer.json"))
        return folder
    }

    @Test func flatAndNestedConfigsPreserveDimensionsAndDefaults() throws {
        let flat = try decode(configuration())
        #expect(flat.flowLM.transformer.hiddenScale == 2.5)
        #expect(flat.flowLM.ldim == 4)
        #expect(flat.flowLM.timeRmsNorm == "rms_var" && flat.flowLM.transformer.gelu == "erf")
        #expect(flat.mimi.gelu == "tanh")
        #expect(flat.mimi.seanet.padMode == "constant")
        #expect(flat.mimi.transformer.outputDimensions == [8])
        #expect(!flat.mimi.encoderEnabled)
        #expect(flat.temp == 0.8 && flat.lsdDecodeSteps == 2 && flat.eosThreshold == 4)

        var nested = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(flat)) as? [String: Any])
        for key in ["temp", "lsd_decode_steps", "eos_threshold", "conditioners", "fuser"] {
            nested.removeValue(forKey: key)
        }
        var mimi = try #require(nested["mimi"] as? [String: Any])
        mimi.removeValue(forKey: "encoder_enabled")
        nested["mimi"] = mimi
        let legacy = try decode(nested)
        #expect(legacy.flowLM.transformer.hiddenScale == 2.5)
        #expect(legacy.mimi.encoderEnabled)
        #expect(legacy.temp == 0.7 && legacy.lsdDecodeSteps == 1 && legacy.eosThreshold == -4)
        #expect(legacy.conditioners.isEmpty)
    }

    @Test func speakerLatentsAreProjectedAndConditioningIsCached() async throws {
        let folder = try makeFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let model = try await PocketTTSModel.fromConfig(decode(configuration()), modelFolder: folder)
        #expect(model.mimi.encoder == nil && model.mimi.encoder_transformer == nil && model.mimi.downsample == nil)
        #expect(model.defaultGenerationParameters.temperature == 0.8)
        #expect(model.flow_lm.transformer.layers[0].linear1.weight.shape == [80, 32])
        model.speaker_proj_weight = MLXArray.ones([32, 8])
        let projected = try model.voiceConditioning(["speaker_wavs": MLXArray.ones([8, 3])])
        #expect(projected.shape == [1, 3, 32])
        #expect(projected.asArray(Float.self).allSatisfy { $0 == 8 })

        let speakers = try #require(model.flow_lm.num_speakers)
        try speakers.update(parameters: .unflattened([
            "embed.weight": MLXArray.ones([4, 4]) * 2,
            "output_proj.weight": MLXArray.ones([32, 4]),
        ]), verify: .all)
        model.flow_lm.cacheConditioning()
        let conditioned = model.flow_lm.conditionAudioInput(MLXArray.zeros([1, 1, 32]))
        #expect(conditioned.asArray(Float.self).allSatisfy { $0 == 8 })
        #expect(!model.parameters().flattened().contains { $0.0.contains("conditioningEmbedding") })
        try speakers.output_proj.update(parameters: .unflattened(["weight": MLXArray.ones([32, 4]) * 3]), verify: .all)
        model.flow_lm.cacheConditioning()
        #expect(model.flow_lm.conditionAudioInput(MLXArray.zeros([1, 1, 32])).asArray(Float.self).allSatisfy { $0 == 24 })
    }

    @Test(arguments: ["rms", "rms_var"])
    func timestepNormalizationMatchesReference(mode: String) {
        let norm = PocketRMSNorm(dim: 4, mode: mode)
        norm.alpha = MLXArray.ones([4]) * 2
        let values: [Float] = [1, 2, 4, 8]
        let denominator: Float = mode == "rms" ? 21.25 : 28.75 / 3
        let expected = MLXArray(values.map { $0 * 2 / sqrt(denominator + 1e-5) }).reshaped([1, 4])
        let actual = norm(MLXArray(values).reshaped([1, 4]))
        #expect(MLX.max(MLX.abs(actual - expected)).item(Float.self) < 1e-6)
    }

    @Test func continuousConditioningMatchesReferenceAndPreservesProjectionDtype() throws {
        let conditioner = ContinuousConditioner(dim: 4, outputDim: 4, scaleFactor: 2)
        let expected = MLXArray([
            Float(1), 1, 0, 0,
            cos(Float(2)), cos(Float(0.0002)), sin(Float(2)), sin(Float(0.0002)),
        ]).reshaped([2, 4])
        let actual = conditioner(MLXArray([Float(0), 1]))
        #expect(MLX.max(MLX.abs(actual - expected)).item(Float.self) < 1e-6)
        let projected = ContinuousConditioner(dim: 4, outputDim: 8, scaleFactor: 1000)
        try projected.update(parameters: .unflattened([
            "output_proj.weight": MLXArray.ones([8, 4]).asType(.bfloat16),
        ]), verify: .all)
        let output = projected(MLXArray([Float(0)]))
        #expect(output.dtype == .bfloat16)
        #expect(output.asArray(Float.self).allSatisfy { $0 == 2 })
    }

    @Test(arguments: ["erf", "tanh"])
    func codecActivationMatchesConfiguration(activation: String) async throws {
        let folder = try makeFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        var values = configuration()
        var mimi = try #require(values["mimi"] as? [String: Any])
        mimi["gelu"] = activation
        values["mimi"] = mimi
        let model = try await PocketTTSModel.fromConfig(decode(values), modelFolder: folder)
        let mlp = try #require(model.mimi.decoder_transformer.transformer.layers[0].gating as? MlpNoGating)
        try mlp.update(parameters: .unflattened([
            "linear1.weight": concatenated([eye(8), MLXArray.zeros([8, 8])], axis: 0),
            "linear2.weight": concatenated([eye(8), MLXArray.zeros([8, 8])], axis: 1),
        ]), verify: .all)
        let inputs: [Float] = [-2, -1, 1, 2, -2, -1, 1, 2]
        let expected = inputs.map { x in
            activation == "erf"
                ? 0.5 * x * (1 + erf(x / sqrt(2)))
                : 0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
        }
        let actual = mlp(MLXArray(inputs).reshaped([1, 8]))
        #expect(MLX.max(MLX.abs(actual - MLXArray(expected))).item(Float.self) < 1e-6)
    }

    // The tiny decoder has one upsampling stage. These fixed Sequential names
    // are independent of the runtime mapping and follow the reference layout.
    private func upstreamWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let decoderNames = [
            "mimi.decoder.init_conv1d": "mimi.decoder.model.0",
            "mimi.decoder.layers.0.upsample": "mimi.decoder.model.2",
            "mimi.decoder.layers.0.residuals.0.block.0": "mimi.decoder.model.3.block.1",
            "mimi.decoder.layers.0.residuals.0.block.1": "mimi.decoder.model.3.block.3",
            "mimi.decoder.final_conv1d": "mimi.decoder.model.5",
        ]
        var raw: [String: MLXArray] = [:]
        for (key, tensor) in weights {
            var source = key
            var value = tensor
            if key.hasPrefix("mimi.decoder.") {
                for (target, original) in decoderNames where key.hasPrefix(target + ".") {
                    source = original + key.dropFirst(target.count)
                    break
                }
                source = source.replacingOccurrences(of: ".conv.conv.", with: ".conv.")
                    .replacingOccurrences(of: ".convtr.convtr.", with: ".convtr.")
                if value.ndim == 3 {
                    value = source.contains(".convtr.") ? value.transposed(2, 0, 1) : value.transposed(0, 2, 1)
                }
            } else if value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
            source = source.replacingOccurrences(of: "mimi.upsample.convtr.convtr.convtr.", with: "mimi.upsample.convtr.convtr.")
                .replacingOccurrences(of: ".gating.linear", with: ".linear")
                .replacingOccurrences(of: ".in_proj.weight", with: ".in_proj_weight")
                .replacingOccurrences(of: "flow_lm.transformer.", with: "flow_lm.backbone.")
                .replacingOccurrences(of: "flow_lm.flow_net.", with: "flow_lm.flow.")
                .replacingOccurrences(of: "speaker_proj_weight", with: "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight")
            for (name, original) in [("conditioner", "transcript_in_segment"), ("num_speakers", "num_speakers"), ("duration_delta", "duration_delta")] {
                source = source.replacingOccurrences(of: "flow_lm.\(name).", with: "flow_lm.condition_provider.conditioners.\(original).")
            }
            raw[source] = value
        }
        return raw
    }

    @Test(arguments: [6, 9])
    func officialCheckpointLoadsWithoutModelTypeAndUsesDefaultVoice(layers: Int) async throws {
        let folder = try makeFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        var values = configuration()
        values.removeValue(forKey: "model_type")
        var flow = try #require(values["flow_lm"] as? [String: Any])
        flow["num_layers"] = layers
        flow["time_rms_norm"] = "rms"
        flow["gelu"] = "erf"
        values["flow_lm"] = flow
        var mimi = try #require(values["mimi"] as? [String: Any])
        mimi["gelu"] = "erf"
        mimi.removeValue(forKey: "encoder_enabled")
        values["mimi"] = mimi
        var conditioners = try #require(values["conditioners"] as? [[String: Any]])
        conditioners.append([
            "name": "duration_delta", "type": "continuous",
            "continuous": ["dim": 4, "scale_factor": 1000.0, "max_period": 10000.0],
        ])
        values["conditioners"] = conditioners
        values["fuser"] = ["sum": ["num_speakers", "duration_delta"]]
        let reference = try await PocketTTSModel.fromConfig(decode(values), modelFolder: folder, encoderEnabled: false)
        let expected = Dictionary(uniqueKeysWithValues: reference.parameters().flattened().map { key, value in
            (key, key.hasPrefix("flow_lm.") ? value.asType(.bfloat16) : value)
        })
        var raw = upstreamWeights(expected)
        raw["flow_lm.condition_provider.conditioners.duration_delta.learnt_padding"] = MLXArray.zeros([1, 1, 32])
        raw["speaker_wavs"] = MLXArray.zeros([1, 8, 2])
        try MLX.save(arrays: raw, url: folder.appendingPathComponent("model.safetensors"))
        let voice = folder.appendingPathComponent("default-voice.safetensors")
        try MLX.save(arrays: ["speaker_wavs": MLXArray.ones([1, 8, 2])], url: voice)
        try JSONSerialization.data(withJSONObject: values).write(to: folder.appendingPathComponent("config.json"))

        let loaded = try await TTS.loadModel(modelRepo: folder.path)
        let model = try #require(loaded as? PocketTTSModel)
        let actual = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(Set(actual.keys) == Set(expected.keys))
        for (key, tensor) in expected {
            let mapped = try #require(actual[key])
            #expect(mapped.dtype == tensor.dtype, "\(key)")
            #expect(mapped.shape == tensor.shape, "\(key)")
            #expect(MLX.max(MLX.abs(mapped - tensor)).item(Float.self) == 0, "\(key)")
        }
        #expect(model.mimi.encoder == nil)
        #expect(model.flow_lm.flow_net.time_embed[0].mlp[3] is PocketRMSNorm)
        let duration = try #require(model.flow_lm.duration_delta)
        let speakers = try #require(model.flow_lm.num_speakers)
        let expectedCondition = speakers(MLXArray([Int32(1)])) + duration(MLXArray([Float(0)]))
        let conditioned = model.flow_lm.conditionAudioInput(MLXArray.zeros([1, 1, 32]))
        #expect(MLX.max(MLX.abs(conditioned - expectedCondition)).item(Float.self) == 0)
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let audio = try await model.generate(text: "Hello world.", voice: nil, generationParameters: parameters)
        let named = try await model.generate(text: "Hello world.", voice: "default", generationParameters: parameters)
        let local = try await model.generate(text: "Hello world.", voice: voice.path, generationParameters: parameters)
        #expect(audio.size > 0 && audio.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(MLX.max(MLX.abs(audio - named)).item(Float.self) == 0)
        #expect(MLX.max(MLX.abs(audio - local)).item(Float.self) == 0)
    }

    @Test func quantizedCheckpointLoadsAndStreamsTheSameAudio() async throws {
        let folder = try makeFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        var values = configuration()
        values["eos_threshold"] = 1e6
        values["weights_name"] = "phonon.safetensors"
        let path = "flow_lm.transformer.layers.0.self_attn.in_proj"
        values["quantization_config"] = ["group_size": 32, "bits": 8, path: ["group_size": 32, "bits": 4]]
        let reference = try await PocketTTSModel.fromConfig(decode(values), modelFolder: folder)
        quantize(model: reference) { name, _ in name == path ? (32, 4, .affine) : nil }
        var weights = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        weights["flow_lm.conditioner.learnt_padding"] = MLXArray.zeros([32])
        try MLX.save(arrays: weights, url: folder.appendingPathComponent("phonon.safetensors"))
        try JSONSerialization.data(withJSONObject: values).write(to: folder.appendingPathComponent("config.json"))
        let voices = folder.appendingPathComponent("voices")
        try FileManager.default.createDirectory(at: voices, withIntermediateDirectories: true)
        try MLX.save(arrays: ["speaker_wavs": MLXArray.ones([1, 8, 3])], url: voices.appendingPathComponent("Freya.safetensors"))
        // A root voice file must never be merged into model weights.
        try MLX.save(arrays: ["emb": MLXArray.zeros([1, 3, 32])], url: folder.appendingPathComponent("unused.safetensors"))

        let model = try await TTS.loadModel(modelRepo: folder.path, modelType: "phonon")
        let pocket = try #require(model as? PocketTTSModel)
        let linear = try #require(pocket.flow_lm.transformer.layers[0].self_attn.in_proj as? QuantizedLinear)
        #expect(linear.bits == 4 && linear.groupSize == 32)
        #expect(!(pocket.flow_lm.transformer.layers[0].linear1 is QuantizedLinear))
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let audio = try await pocket.generate(text: "Hello world.", voice: nil, generationParameters: parameters)
        #expect(audio.shape == [24])
        #expect(audio.asArray(Float.self).allSatisfy { $0.isFinite })
        var chunks: [MLXArray] = []
        for try await event in pocket.generateStream(
            text: "Hello world.", voice: "fReYa", refAudio: nil, refText: nil, language: nil,
            generationParameters: parameters, streamingInterval: 0.25
        ) {
            if case .audio(let chunk) = event { chunks.append(chunk) }
        }
        #expect(chunks.count == 3)
        #expect(chunks.allSatisfy { $0.shape == [8] })
        let streamed = concatenated(chunks, axis: 0)
        #expect(MLX.max(MLX.abs(audio - streamed)).item(Float.self) < 1e-6)
    }

    @Test func missingDecoderWeightsFailStrictLoading() async throws {
        let folder = try makeFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let values = configuration()
        let model = try await PocketTTSModel.fromConfig(decode(values), modelFolder: folder)
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        weights.removeValue(forKey: "mimi.decoder.final_conv1d.conv.conv.weight")
        try MLX.save(arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: values).write(to: folder.appendingPathComponent("config.json"))
        do {
            _ = try await PocketTTSModel.fromModelDirectory(folder)
            Issue.record("Missing decoder weights should fail loading")
        } catch {
            #expect(String(describing: error).contains("weight"))
        }
    }

    @Test func rejectsUnknownVoicesAndReferenceAudioWithoutEncoder() async throws {
        let folder = try makeFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let model = try await PocketTTSModel.fromConfig(decode(configuration()), modelFolder: folder)
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        do {
            _ = try await model.generate(text: "Hello", voice: "missing", generationParameters: parameters)
            Issue.record("Unknown voices should fail")
        } catch {
            #expect(error.localizedDescription.contains("Unknown voice"))
        }
        #expect(model.temp == 0.8)
        do {
            _ = try await model.generate(
                text: "Hello", voice: nil, refAudio: MLXArray.zeros([16]), generationParameters: parameters)
            Issue.record("Reference audio requires the speaker encoder")
        } catch {
            #expect(error.localizedDescription.contains("omits the speaker encoder"))
        }
    }
}
