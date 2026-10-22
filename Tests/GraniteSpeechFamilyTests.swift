import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private func tinyGraniteConfig(plus: Bool = false) throws -> GraniteSpeechModelConfig {
    let json =
        #"{"audio_token_index":15,"encoder_config":{"input_dim":160,"hidden_dim":8,"num_layers":4,"num_heads":2,"dim_head":4,"feedforward_mult":2,"output_dim":6,"context_size":4,"max_pos_emb":4,"conv_kernel_size":3},"projector_config":{"hidden_size":8,"encoder_hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16},"text_config":{"vocab_size":16,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"tie_word_embeddings":true},"window_size":3,"downsample_rate":3}"#
    var config = try JSONDecoder().decode(GraniteSpeechModelConfig.self, from: Data(json.utf8))
    if plus {
        config.modelType = "granite_speech_plus"
        config.encoderConfig.catHiddenLayers = [3]
        config.projectorConfig.encoderHiddenSize = 16
    }
    return config
}

private func tinyNARConfig() throws -> GraniteSpeechNARConfig {
    let json =
        #"{"blank_token_id":15,"min_edit_sequence_length":4,"encoder_layer_indices":[1,-1],"encoder_config":{"input_dim":160,"hidden_dim":8,"num_layers":2,"num_heads":2,"dim_head":4,"feedforward_mult":2,"output_dim":6,"context_size":4,"max_pos_emb":4,"conv_kernel_size":3,"bpe_output_dim":16,"bpe_pooling_window":4,"self_conditioning_layer":1},"projector_config":{"hidden_size":8,"num_heads":2,"num_layers":1,"num_encoder_layers":2,"block_size":3,"downsample_rate":3,"encoder_dim":8,"llm_dim":8,"mlp_ratio":2},"text_config":{"vocab_size":16,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"tie_word_embeddings":true}}"#
    return try JSONDecoder().decode(GraniteSpeechNARConfig.self, from: Data(json.utf8))
}

@Suite(.serialized)
struct GraniteSpeechFamilyTests {
    @Test func ropeSchemasAndRoundTrip() throws {
        for key in ["rope_scaling", "rope_parameters"] {
            let json =
                "{\"\(key)\":{\"rope_type\":\"yarn\",\"factor\":4,\"original_max_position_embeddings\":4096,\"rope_theta\":20000}}"
            let config = try JSONDecoder().decode(
                GraniteSpeechTextConfig.self, from: Data(json.utf8))
            #expect(config.ropeScaling?.ropeType == "yarn")
            #expect(config.ropeScaling?.factor == 4)
            #expect(config.ropeTheta == 20000)
            let decoded = try JSONDecoder().decode(
                GraniteSpeechTextConfig.self, from: JSONEncoder().encode(config))
            #expect(decoded.ropeScaling?.originalMaxPositionEmbeddings == 4096)
        }
        let flat = try JSONDecoder().decode(
            GraniteSpeechTextConfig.self,
            from: Data(#"{"rope_type":"yarn","factor":4}"#.utf8))
        #expect(flat.ropeScaling?.factor == 4)
    }

    @Test func gpt2TokenizerUsesItsDeclaredPreTokenizer() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data =
            #"{"model":{"type":"BPE","vocab":{"<unk>":0,"?":1,"Ċ":2,"?Ċ":3},"merges":["? Ċ"],"unk_token":"<unk>"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\S+\\s*"},"behavior":"Removed","invert":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}}"#
        try Data(data.utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        for (name, expected) in [("GPT2Tokenizer", [1, 2]), ("TokenizersBackend", [3])] {
            let config = ["tokenizer_class": name, "unk_token": "<unk>"]
            try JSONSerialization.data(withJSONObject: config).write(
                to: directory.appendingPathComponent("tokenizer_config.json"))
            let tokenizer = try await loadGraniteSpeechTokenizer(from: directory)
            #expect(tokenizer.encode(text: "?\n") == expected)
        }
    }

    @Test func yarnRotatesAtAbsoluteCachePositions() throws {
        let config = try JSONDecoder().decode(
            GraniteSpeechTextConfig.self,
            from: Data(#"{"rope_scaling":{"rope_type":"yarn","factor":4}}"#.utf8))
        let rope = GraniteSpeechRotaryEmbedding(dimensions: 128, config: config)
        #expect(abs(rope.magnitude - (1 + 0.1 * log(Float(4)))) < 1e-7)
        let x = sin(arange(3 * 128).asType(.float32)).reshaped(1, 1, 3, 128)
        let whole = rope(x, offset: 5000)
        let tail = rope(x[0..., 0..., 2..., 0...], offset: 5002)
        #expect(abs(whole[0..., 0..., 2..., 0...] - tail).max().item(Float.self) < 1e-5)
        let frequencies = try #require(rope.frequencies)
        #expect(abs(frequencies[0].item(Float.self) - 1) < 1e-7)
        #expect(
            abs(frequencies[-1].item(Float.self) / pow(Float(10000), Float(126) / 128) - 4) < 1e-5)
    }

    @Test func partialAttentionBlocksIncludeZeroPadding() throws {
        var config = try tinyGraniteConfig().encoderConfig
        config.hiddenDim = 2
        config.numHeads = 1
        config.dimHead = 2
        let attention = GraniteSpeechConformerAttention(config, nonAutoregressive: true)
        let weights: [String: MLXArray] = [
            "pre_norm.weight": .ones([2]), "pre_norm.bias": .zeros([2]),
            "to_q.weight": .zeros([2, 2]),
            "to_kv.weight": MLXArray([Float(0), 0, 0, 0, 1, 0, 0, 1]).reshaped(4, 2),
            "to_out.weight": MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
            "to_out.bias": .zeros([2]),
            "rel_pos_emb.weight": .zeros([9, 2]),
        ]
        try attention.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let output = attention(
            MLXArray([Float(1), 2, 1, 2]).reshaped(1, 2, 2),
            attentionDists: .zeros([4, 4], dtype: .int32))
        #expect(
            abs(output - MLXArray([Float(-0.5), 0.5]).reshaped(1, 1, 2)).max().item(Float.self)
                < 5e-4)
    }

    @Test func plusFeaturesIncludeIntermediateStatesAfterConditioning() throws {
        var config = try tinyGraniteConfig(plus: true)
        config.encoderConfig.catHiddenLayers = [0, 2]
        config.projectorConfig.encoderHiddenSize = 24
        let model = GraniteSpeechModel(config)
        let features = sin(arange(5 * 160).asType(.float32)).reshaped(1, 5, 160)
        let encoded = model.encoder(features)
        #expect(encoded.shape == [1, 5, 24])
        #expect(
            abs(encoded[0..., 0..., ..<8] - model.encoder.inputLinear(features)).max().item(
                Float.self) == 0)
        #expect(model.getAudioFeatures(features).shape == [1, 2, 8])
    }

    @Test func plusPromptsAndModelSpecificControls() throws {
        let model = GraniteSpeechModel(try tinyGraniteConfig(plus: true))
        let instruction = model.resolvePrompt(
            prompt: nil, language: nil, task: .speakerAttribution,
            sourceLanguage: nil, hotwords: [], punctuate: false)
        let messages = model.promptMessages(
            numAudioTokens: 2, instruction: instruction, systemPrompt: nil)
        #expect(messages.count == 2)
        #expect(messages[0]["content"] as? String == GraniteSpeechModel.plusSystemPrompt)
        #expect(
            (messages[1]["content"] as? String)?.hasPrefix(
                "<|audio|><|audio|> Speaker attribution:") == true)
        #expect(
            model.resolvePrompt(
                prompt: nil, language: nil, task: .transcription,
                sourceLanguage: "French", hotwords: [" MLX ", ""], punctuate: true)
                == "The speech is in French. Transcribe it exactly; do not translate. transcribe the speech with proper punctuation and capitalization. Keywords: MLX"
        )
        #expect(
            model.promptMessages(numAudioTokens: 1, instruction: "test", systemPrompt: "").count
                == 1)
    }

    @Test func speakerParsingRejectsMalformedTurns() throws {
        let segments = try #require(
            GraniteSpeechModel.parseSegments(
                "[Speaker 1]: Hello. [Speaker 2]: Hi. [Speaker 1]: Again.",
                task: .speakerAttribution))
        #expect(segments.count == 3)
        #expect(segments[1]["speaker_id"] as? Int == 2)
        #expect(segments[0]["text"] as? String == "Hello.")
        for malformed in [
            "hello", "[Speaker 2]: Hi.", "prefix [Speaker 1]: Hello.", "[Speaker 1]:",
        ] {
            #expect(GraniteSpeechModel.parseSegments(malformed, task: .speakerAttribution) == nil)
        }
    }

    @Test func timestampParsingHandlesSilenceAndWraps() throws {
        let segments = try #require(
            GraniteSpeechModel.parseSegments(
                "Hello [T:980] _ [T:20] world [T:60]", task: .timestamps))
        let words = try #require(segments[0]["words"] as? [[String: Any]])
        #expect(words.count == 2)
        #expect(words[1]["start"] as? Double == 10.2)
        #expect(words[1]["end"] as? Double == 10.6)
        #expect(segments[0]["text"] as? String == "Hello world")
        #expect(GraniteSpeechModel.parseSegments("_ [T:80]", task: .timestamps)?.isEmpty == true)
        for malformed in ["two words [T:40]", "word [T:40] tail", "word [T:2000] back [T:1500]"] {
            #expect(GraniteSpeechModel.parseSegments(malformed, task: .timestamps) == nil)
        }
    }

    @Test func sourceAndPreparedWeightsSanitizeIdempotently() {
        let source: [String: MLXArray] = [
            "language_model.model.model.embed_tokens.weight": .ones([16, 8]),
            "encoder.layers.0.conv.depth_conv.conv.weight": .zeros([16, 1, 3]),
            "encoder.layers.0.conv.up_conv.weight": .zeros([32, 8, 1]),
            "encoder.layers.0.conv.batch_norm.num_batches_tracked": MLXArray(0),
            "weight": .ones([16, 8]),
        ]
        let prepared: [String: MLXArray] = [
            "editor.embed_tokens.weight": .ones([16, 8]),
            "encoder.layers.0.conv.depth_conv.weight": .zeros([16, 3, 1]),
            "encoder.layers.0.conv.bn.weight": .ones([16]),
        ]
        for weights in [source, prepared] {
            let sanitized = GraniteSpeechNARModel.sanitize(weights: weights)
            #expect(sanitized["editor.model.embed_tokens.weight"]?.shape == [16, 8])
            #expect(sanitized["encoder.layers.0.conv.depth_conv.conv.weight"]?.shape == [16, 3, 1])
            #expect(sanitized["weight"] == nil)
            #expect(!sanitized.keys.contains { $0.hasSuffix("num_batches_tracked") })
            let twice = GraniteSpeechNARModel.sanitize(weights: sanitized)
            #expect(twice.keys.sorted() == sanitized.keys.sorted())
            for key in sanitized.keys { #expect(twice[key]?.shape == sanitized[key]?.shape) }
        }
    }

    @Test func narPoolingCTCAndInsertionSlots() {
        #expect(GraniteSpeechNARModel.collapse([15, 2, 2, 15, 2, 3, 3], blankId: 15) == [2, 2, 3])
        #expect(
            GraniteSpeechNARModel.insertionSlots([2, 3], blankId: 15, minimumLength: 8) == [
                15, 2, 15, 3, 15, 15, 15, 15,
            ])
        #expect(
            GraniteSpeechNARModel.insertionSlots([], blankId: 15, minimumLength: 4) == [
                15, 15, 15, 15,
            ])
        let pooled = GraniteSpeechNAREncoder.posteriorWeightedPool(
            MLXArray([Float(2), 4, 8, 10, 20]).reshaped(1, 5, 1),
            blankProbabilities: MLXArray([Float(1), 0, 0, 1, 0]).reshaped(1, 5), window: 4)
        #expect(pooled.asArray(Float.self) == [6, 20])
    }

    @Test func narTinyForwardAndConfigRoundTrip() throws {
        let config = try tinyNARConfig()
        try config.validate()
        let decoded = try JSONDecoder().decode(
            GraniteSpeechNARConfig.self, from: JSONEncoder().encode(config))
        #expect(decoded.encoderConfig.conformer.numLayers == 2)
        #expect(decoded.encoderConfig.selfConditioningLayer == 1)
        let model = GraniteSpeechNARModel(config)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let features = GraniteSpeechNARFrontend.features(.zeros([1600])).expandedDimensions(axis: 0)
        #expect(features.shape == [1, 5, 160])
        let (logits, audioTokens) = model.editedLogits(features)
        #expect(logits.dim(2) == 16)
        #expect(audioTokens == 2)
        #expect(MLX.all(isFinite(logits)).item(Bool.self))
        #expect(GraniteSpeechNARFrontend.features(.zeros([480])).shape == [1, 160])
    }

    @Test(arguments: [false, true])
    func sharedLoaderReadsTinyLocalModels(nar: Bool) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original: Module
        let config: Data
        if nar {
            original = GraniteSpeechNARModel(try tinyNARConfig())
            config = try JSONEncoder().encode(tinyNARConfig())
        } else {
            original = GraniteSpeechModel(try tinyGraniteConfig(plus: true))
            config = try JSONEncoder().encode(tinyGraniteConfig(plus: true))
        }
        try config.write(to: directory.appendingPathComponent("config.json"))
        let pieces = ["<unk>"] + (1 ..< 15).map { "token\($0)" } + ["<|audio|>"]
        let vocabulary = Dictionary(
            uniqueKeysWithValues: pieces.enumerated().map { ($0.element, $0.offset) })
        let tokenizer: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [], "unk_token": "<unk>"],
            "added_tokens": [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 15, "content": "<|audio|>", "special": true],
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer).write(
            to: directory.appendingPathComponent("tokenizer.json"))
        try Data(
            #"{"tokenizer_class":"GPT2Tokenizer","unk_token":"<unk>","eos_token":"<unk>"}"#.utf8
        )
        .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        eval(Array(weights.values))
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await STT.loadModel(modelRepo: directory.path)
        #expect((loaded is GraniteSpeechNARModel) == nar)
        #expect((loaded as? GraniteSpeechModel)?.isPlus == (nar ? nil : true))
        let parameters = STTGenerateParameters(maxTokens: 1)
        let audio = MLXArray.zeros([1600])
        let output = loaded.generate(audio: audio, generationParameters: parameters)
        var finals = 0
        for try await event in loaded.generateStream(audio: audio, generationParameters: parameters)
        {
            if case .result(let result) = event {
                finals += 1
                #expect(result.text == output.text)
            }
        }
        #expect(finals == 1)
    }

    @Test func quantizedEditorKeepsTiedEmbeddingHead() throws {
        var config = try tinyNARConfig()
        config.textConfig.hiddenSize = 64
        config.textConfig.intermediateSize = 128
        config.textConfig.numAttentionHeads = 4
        config.projectorConfig.llmDim = 64
        let model = GraniteSpeechNARModel(config)
        quantize(model: model) { path, _ in path.hasPrefix("editor.") ? (32, 5, .affine) : nil }
        #expect(model.editor.embedTokens.weight.dtype == .uint32)
        let (logits, _) = model.editedLogits(.zeros([1, 5, 160]))
        #expect(logits.dim(2) == 16)
        #expect(MLX.all(isFinite(logits)).item(Bool.self))
    }

    @Test func editorUsesFutureContextWhileAutoregressiveDecoderIsCausal() throws {
        let model = GraniteSpeechLanguageModel(try tinyNARConfig().textConfig)
        let x = sin(arange(4 * 8).asType(.float32)).reshaped(1, 4, 8)
        let changed = concatenated([x[0..., ..<3, 0...], x[0..., 3..., 0...] * -10], axis: 1)
        let causal = model.model(inputEmbeddings: x)
        let causalChanged = model.model(inputEmbeddings: changed)
        #expect(
            abs(causal[0..., ..<3, 0...] - causalChanged[0..., ..<3, 0...]).max().item(Float.self)
                < 1e-6)
        let edited = model.model(inputEmbeddings: x, bidirectional: true)
        let editedChanged = model.model(inputEmbeddings: changed, bidirectional: true)
        #expect(
            abs(edited[0..., ..<3, 0...] - editedChanged[0..., ..<3, 0...]).max().item(Float.self)
                > 1e-5)
    }
}

@Suite(.serialized)
struct GraniteSpeechFamilyCheckpointTests {
    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["GRANITE_FAMILY_CHECKPOINTS"] != nil),
        arguments: [
            "granite-speech-4.1-2b", "granite-speech-4.1-2b-plus", "granite-speech-5.0-2b",
            "granite-speech-4.1-2b-nar", "granite-speech-4.1-2b-nar-mlx-5bit",
        ])
    func publishedCheckpointTranscribes(modelName: String) async throws {
        let root = try #require(ProcessInfo.processInfo.environment["GRANITE_FAMILY_CHECKPOINTS"])
        let directory = URL(fileURLWithPath: root).appendingPathComponent(modelName)
        let model = try await STT.loadModel(modelRepo: directory.path)
        #expect((model is GraniteSpeechNARModel) == modelName.contains("-nar"))
        let audioURL = try #require(
            Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")
        )
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
        let parameters = STTGenerateParameters(maxTokens: 32, temperature: 0)
        let output = model.generate(audio: audio, generationParameters: parameters)
        print("\(modelName): \(output.text)")
        let words = output.text.lowercased().components(
            separatedBy: CharacterSet.alphanumerics.inverted
        ).filter { !$0.isEmpty }
        #expect(words == ["intention"])
        var streamed = ""
        var finals = 0
        for try await event in model.generateStream(audio: audio, generationParameters: parameters)
        {
            switch event {
            case .token(let text): streamed += text
            case .info: break
            case .result(let result):
                finals += 1
                #expect(result.text == output.text)
            }
        }
        #expect(finals == 1)
        #expect(!streamed.isEmpty)
    }
}
