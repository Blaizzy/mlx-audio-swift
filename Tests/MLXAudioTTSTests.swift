//  Run the TTS suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/SopranoTextCleaningTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests/testTextCleaning()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Testing
import MLX
import Metal
import MLXLMCommon
import Foundation

private let metalAvailable: Bool = {
    #if canImport(Metal)
    return MTLCreateSystemDefaultDevice() != nil
    #else
    return false
    #endif
}()

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs

private func loadTTSNetworkFixture(sampleRate: Int, maxSamples: Int) throws -> MLXArray {
    let audioURL = Bundle.module.url(
        forResource: "intention",
        withExtension: "wav",
        subdirectory: "media"
    )!
    let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: sampleRate)
    let sampleCount = min(audio.shape[0], maxSamples)
    return audio[0..<sampleCount]
}

private struct FakeFishTokenizer: FishSpeechTokenizing {
    let vocabSize = 8_192
    let eosTokenID = 99
    let padTokenID = 0
    let semanticBeginID = 1_000
    let semanticEndID = 5_095

    func encode(_ text: String, addSpecialTokens: Bool) -> [Int] {
        switch text {
        case "\(fishSpeechIMStartToken)\(FishSpeechRole.assistant.rawValue)\n\(fishSpeechVoiceModalityToken)":
            return [11]
        case "\(fishSpeechIMEndToken)\n":
            return [12]
        case "hi":
            return [13, 14]
        default:
            return text.utf8.map(Int.init)
        }
    }

    func decode(_ tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map(String.init).joined(separator: ",")
    }

    func tokenID(for token: String) -> Int? {
        switch token {
        case fishSpeechEOSToken:
            return eosTokenID
        case fishSpeechPadToken:
            return padTokenID
        case fishSpeechIMEndToken:
            return 12
        default:
            return nil
        }
    }
}

private func makeTinyFishSpeechConfig() -> FishSpeechConfig {
    FishSpeechConfig(
        textConfig: FishTextConfig(
            vocabSize: 128,
            nLayer: 1,
            nHead: 2,
            dim: 8,
            intermediateSize: 16,
            nLocalHeads: 2,
            headDim: 4,
            maxSeqLen: 64
        ),
        audioDecoderConfig: FishAudioDecoderConfig(
            vocabSize: 32,
            nLayer: 1,
            nHead: 2,
            dim: 8,
            intermediateSize: 16,
            nLocalHeads: 2,
            headDim: 4,
            maxSeqLen: 8,
            textDim: 8,
            numCodebooks: 2
        )
    )
}


// MARK: - Text Cleaning Unit Tests

struct SopranoTextCleaningTests {

    @Test func testTextCleaning() {
        // Test number normalization
        let text1 = "I have $100 and 50 cents."
        let cleaned1 = cleanTextForSoprano(text1)
        #expect(cleaned1.contains("one hundred dollars"), "Should expand dollar amounts")

        // Test abbreviations
        let text2 = "Dr. Smith went to the API conference."
        let cleaned2 = cleanTextForSoprano(text2)
        #expect(cleaned2.contains("doctor"), "Should expand Dr. to doctor")
        #expect(cleaned2.contains("a p i"), "Should expand API")

        // Test ordinals
        let text3 = "This is the 1st and 2nd test."
        let cleaned3 = cleanTextForSoprano(text3)
        #expect(cleaned3.contains("first"), "Should expand 1st to first")
        #expect(cleaned3.contains("second"), "Should expand 2nd to second")

        print("\u{001B}[32mText cleaning tests passed!\u{001B}[0m")
    }
}

struct EchoTTSTests {

    @Test func testTextNormalization() {
        let normalized = echoTtsNormalizeTextPrompt("Hello: world\nnew line")
        #expect(normalized.hasPrefix("[S1] "))
        #expect(normalized.contains(","))
        #expect(!normalized.contains("\n"))
    }

    @Test func testTokenizerEncode() {
        let tokens = echoTtsTokenizerEncode("hello", appendBOS: true, normalize: false)
        #expect(tokens.shape == [6])
        #expect(tokens[0].item(Int32.self) == 0)
    }

    @Test func testTextInputIDsAndMask() {
        let result = echoTtsTextInputIDsAndMask(
            ["hello", "world"],
            maxLength: 10,
            normalize: true,
            padToMax: true
        )
        #expect(result.inputIDs.shape == [2, 10])
        #expect(result.mask.shape == [2, 10])
        #expect(result.normalizedTexts.count == 2)
    }

    @Test func testEchoDiTForwardShapes() {
        let config = EchoDiTConfig(
            latentSize: 8,
            modelSize: 32,
            numLayers: 2,
            numHeads: 4,
            intermediateSize: 64,
            normEps: 1e-5,
            textVocabSize: 256,
            textModelSize: 32,
            textNumLayers: 1,
            textNumHeads: 4,
            textIntermediateSize: 64,
            speakerPatchSize: 2,
            speakerModelSize: 32,
            speakerNumLayers: 1,
            speakerNumHeads: 4,
            speakerIntermediateSize: 64,
            timestepEmbedSize: 16,
            adalnRank: 8
        )
        let model = EchoDiT(
            latentSize: config.latentSize,
            modelSize: config.modelSize,
            numLayers: config.numLayers,
            numHeads: config.numHeads,
            intermediateSize: config.intermediateSize,
            normEps: config.normEps,
            textVocabSize: config.textVocabSize,
            textModelSize: config.textModelSize,
            textNumLayers: config.textNumLayers,
            textNumHeads: config.textNumHeads,
            textIntermediateSize: config.textIntermediateSize,
            speakerPatchSize: config.speakerPatchSize,
            speakerModelSize: config.speakerModelSize,
            speakerNumLayers: config.speakerNumLayers,
            speakerNumHeads: config.speakerNumHeads,
            speakerIntermediateSize: config.speakerIntermediateSize,
            timestepEmbedSize: config.timestepEmbedSize,
            adalnRank: config.adalnRank
        )

        let x = MLXRandom.normal([1, 6, config.latentSize])
        let t = MLXArray([Float(0.7)])
        let textInputIDs = MLXArray([Int32(0), 1, 2, 3, 4]).reshaped([1, 5])
        let textMask = MLXArray([true, true, true, true, true]).reshaped([1, 5])
        let speakerLatent = MLXRandom.normal([1, 8, config.latentSize])
        let speakerMask = MLXArray.ones([1, 8], dtype: .bool)

        let kvText = model.getKVCacheText(textInputIDs, textMask: textMask)
        let kvSpeaker = model.getKVCacheSpeaker(speakerLatent)
        let output = model(
            x: x,
            t: t,
            textMask: textMask,
            speakerMask: speakerMask,
            kvCacheText: kvText,
            kvCacheSpeaker: kvSpeaker
        )

        #expect(output.shape == [1, 6, config.latentSize])
    }

    @Test func testSanitizeAndGenerateSmoke() throws {
        final class FakeFishAE: EchoTTSAudioCodec {
            func encodeZQ(_ audioData: MLXArray) -> MLXArray {
                MLXArray.zeros([audioData.shape[0], 8, max(audioData.shape[2] / 2_048, 1)], dtype: .float32)
            }

            func decodeZQ(_ zQ: MLXArray) -> MLXArray {
                MLXArray.zeros([zQ.shape[0], 1, zQ.shape[2] * 2_048], dtype: .float32)
            }
        }

        let config = EchoTTSConfig(
            dit: EchoDiTConfig(
                latentSize: 8,
                modelSize: 32,
                numLayers: 2,
                numHeads: 4,
                intermediateSize: 64,
                normEps: 1e-5,
                textVocabSize: 256,
                textModelSize: 32,
                textNumLayers: 1,
                textNumHeads: 4,
                textIntermediateSize: 64,
                speakerPatchSize: 2,
                speakerModelSize: 32,
                speakerNumLayers: 1,
                speakerNumHeads: 4,
                speakerIntermediateSize: 64,
                timestepEmbedSize: 16,
                adalnRank: 8
            ),
            sampler: EchoTTSSamplerConfig(
                numSteps: 1,
                cfgScaleText: 1,
                cfgScaleSpeaker: 1,
                sequenceLength: 4
            )
        )
        let model = EchoTTSModel(
            config: config,
            fishAE: FakeFishAE(),
            pcaState: EchoTTSPCAState(
                pcaComponents: MLXArray.eye(8, dtype: .float32),
                pcaMean: MLXArray.zeros([8], dtype: .float32),
                latentScale: 1
            )
        )

        let sanitized = model.sanitize(weights: [
            "cond_module.0.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "pca_components": MLXArray.zeros([1], dtype: .float32),
        ])
        #expect(sanitized["model.condModule.layers.0.weight"] != nil)
        #expect(sanitized["model.pca_components"] == nil)

        let result = try model.generateDetailed(
            text: "hi",
            refAudio: nil,
            rngSeed: 0,
            numSteps: 1,
            sequenceLength: 4
        )
        #expect(model.sampleRate == 44_100)
        #expect(result.audio.shape[0] > 0)
    }

    @Test func testDeleteBlockwiseModules() throws {
        let config = EchoTTSConfig(
            deleteBlockwiseModules: true,
            dit: EchoDiTConfig(
                latentSize: 8,
                modelSize: 32,
                numLayers: 2,
                numHeads: 4,
                intermediateSize: 64,
                normEps: 1e-5,
                textVocabSize: 256,
                textModelSize: 32,
                textNumLayers: 1,
                textNumHeads: 4,
                textIntermediateSize: 64,
                speakerPatchSize: 2,
                speakerModelSize: 32,
                speakerNumLayers: 1,
                speakerNumHeads: 4,
                speakerIntermediateSize: 64,
                timestepEmbedSize: 16,
                adalnRank: 8
            ),
            sampler: EchoTTSSamplerConfig(numSteps: 1, sequenceLength: 4)
        )
        let model = EchoTTSModel(config: config)

        let sanitized = model.sanitize(weights: [
            "latent_encoder.in_proj.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "blocks.0.attention.wk_latent.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "blocks.0.attention.wv_latent.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "out_proj.weight": MLXArray.zeros([8, 32], dtype: .float32),
        ])
        #expect(sanitized["model.outProj.weight"] != nil)
        #expect(!sanitized.keys.contains(where: { $0.contains("latent_encoder") }))
        #expect(!sanitized.keys.contains(where: { $0.contains("wk_latent") }))
        #expect(!sanitized.keys.contains(where: { $0.contains("wv_latent") }))

        #expect(throws: AudioGenerationError.self) {
            try model.model.getKVCacheLatent(MLXArray.zeros([1, 0, 8], dtype: .float32))
        }

        #expect(throws: AudioGenerationError.self) {
            try model.generateLatents(text: "hi", blockSizes: [2], numSteps: 1, sequenceLength: 4)
        }
    }
}

struct FishSpeechTests {

    @Test func testConfigDecodesQuantizationAlias() throws {
        let data = Data(
            """
            {
              "model_type": "fish_qwen3_omni",
              "quantization_config": {
                "group_size": 64,
                "bits": 4
              }
            }
            """.utf8
        )

        let config = try JSONDecoder().decode(FishSpeechConfig.self, from: data)

        #expect(config.modelType == "fish_qwen3_omni")
        #expect(config.sampleRate == 44_100)
        #expect(config.quantization == BaseConfiguration.Quantization(groupSize: 64, bits: 4))
    }

    @Test func testConversationEncodingInterleavesSemanticAndCodebookRows() {
        let tokenizer = FakeFishTokenizer()
        let codes = MLXArray([Int32(1), 2, 10, 20]).reshaped([2, 2])
        let conversation = FishSpeechConversation(messages: [
            FishSpeechMessage(
                role: .assistant,
                parts: [
                    .text(FishSpeechTextPart(text: "hi")),
                    .vq(FishSpeechVQPart(codes)),
                ],
                addIMStart: true,
                addIMEnd: true,
                modality: .voice
            )
        ])

        let encoded = conversation.encodeForInference(tokenizer: tokenizer, numCodebooks: 2)

        #expect(encoded.shape == [3, 6])
        #expect(encoded[0].asArray(Int32.self) == [11, 13, 14, 1_001, 1_002, 12])
        #expect(encoded[1].asArray(Int32.self) == [0, 0, 0, 1, 2, 0])
        #expect(encoded[2].asArray(Int32.self) == [0, 0, 0, 10, 20, 0])
    }

    @Test func testSpeakerSplitAndBatching() {
        let text = "<|speaker:0|>hello\n<|speaker:1|>world\n<|speaker:2|>again"
        let turns = fishSpeechSplitTextBySpeaker(text)
        let batches = fishSpeechGroupTurnsIntoBatches(turns, maxSpeakers: 2, maxBytes: 1_000)

        #expect(turns == ["<|speaker:0|>hello", "<|speaker:1|>world", "<|speaker:2|>again"])
        #expect(batches == ["<|speaker:0|>hello\n<|speaker:1|>world", "<|speaker:2|>again"])
    }

    @Test func testSanitizeRemapsFishWeightPrefixes() {
        let model = FishSpeechModel(config: makeTinyFishSpeechConfig())
        let sanitized = model.sanitize(weights: [
            "text_model.model.embeddings.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "audio_decoder.codebook_embeddings.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "audio_decoder.layers.0.attention.wqkv.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "model.norm.weight": MLXArray.zeros([1], dtype: .float32),
        ])

        #expect(sanitized["model.embeddings.weight"] != nil)
        #expect(sanitized["model.codebook_embeddings.weight"] != nil)
        #expect(sanitized["model.fast_layers.0.attention.wqkv.weight"] != nil)
        #expect(sanitized["model.norm.weight"] != nil)
    }

    @Test func testDefaultRepositoryID() {
        #expect(FishSpeechModel.defaultRepositoryID == "mlx-community/fish-audio-s2-pro-8bit")
    }

    @Test func testCachedTokenizerMatchesReferenceSpecialTokenEncoding() async throws {
        let modelURL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub/mlx-audio/mlx-community_fish-audio-s2-pro-8bit")
        guard FileManager.default.fileExists(atPath: modelURL.path) else { return }

        let tokenizer = try await FishSpeechTokenizer.fromModelDirectory(modelURL)

        #expect(
            tokenizer.encode("\(fishSpeechIMEndToken)\n", addSpecialTokens: false)
                == [151_645, 198]
        )
        #expect(
            tokenizer.encode(
                "\(fishSpeechIMStartToken)assistant\n\(fishSpeechVoiceModalityToken)",
                addSpecialTokens: false
            ) == [151_644, 77_091, 198, 151_673]
        )
    }

    @Test func testCachedConversationPromptMatchesReference() async throws {
        let modelURL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub/mlx-audio/mlx-community_fish-audio-s2-pro-8bit")
        guard FileManager.default.fileExists(atPath: modelURL.path) else { return }

        let tokenizer = try await FishSpeechTokenizer.fromModelDirectory(modelURL)
        let conversation = FishSpeechConversation(messages: [
            FishSpeechMessage(
                role: .system,
                parts: [.text(FishSpeechTextPart(text: "convert the provided text to speech"))],
                addIMStart: true,
                addIMEnd: true,
                modality: nil
            ),
            FishSpeechMessage(
                role: .user,
                parts: [.text(FishSpeechTextPart(text: "This is a Fish S2 Pro generation test from the Swift port."))],
                addIMStart: true,
                addIMEnd: true,
                modality: nil
            ),
            FishSpeechMessage(
                role: .assistant,
                parts: [],
                addIMStart: true,
                addIMEnd: false,
                modality: .voice
            ),
        ])

        let prompt = conversation.encodeForInference(tokenizer: tokenizer, numCodebooks: 10)

        #expect(prompt.shape == [11, 34])
        #expect(prompt[0].asArray(Int32.self) == [
            151_644, 8_948, 198, 14_166, 279, 3_897, 1_467, 311, 8_806, 151_645, 198,
            151_644, 872, 198, 1_986, 374, 264, 16_608, 328, 17, 1_298, 9_471, 1_273,
            504, 279, 23_670, 2_635, 13, 151_645, 198, 151_644, 77_091, 198, 151_673,
        ])
    }

    @Test func testCachedFirstGreedyStepMatchesReference() async throws {
        let modelURL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub/mlx-audio/mlx-community_fish-audio-s2-pro-8bit")
        guard FileManager.default.fileExists(atPath: modelURL.path) else { return }

        let model = try await FishSpeechModel.fromPretrained()
        let tokenizer = try #require(model.tokenizer)
        let semanticBias = try #require(model.semanticLogitBias)

        var conversation = FishSpeechConversation()
        conversation.append(FishSpeechMessage(
            role: .system,
            parts: [.text(FishSpeechTextPart(text: "convert the provided text to speech"))],
            addIMStart: true,
            addIMEnd: true,
            modality: nil
        ))
        conversation.append(FishSpeechMessage(
            role: .user,
            parts: [.text(FishSpeechTextPart(text: "This is a Fish S2 Pro generation test from the Swift port."))],
            addIMStart: true,
            addIMEnd: true,
            modality: nil
        ))
        conversation.append(FishSpeechMessage(
            role: .assistant,
            parts: [],
            addIMStart: true,
            addIMEnd: false,
            modality: .voice
        ))

        let prompt = conversation.encodeForInference(tokenizer: tokenizer, numCodebooks: model.model.numCodebooks)
            .expandedDimensions(axis: 0)
        let cache = model.model.makeCache()
        let result = model.model(prompt, cache: cache)
        let logits = result.logits[0..., (result.logits.dim(1) - 1)..<result.logits.dim(1), 0...]
            .squeezed(axis: 1)
        let biased = logits + semanticBias.asType(logits.dtype)
        func firstMax(_ logits: MLXArray) -> MLXArray {
            let maxValues = MLX.max(logits, axis: -1, keepDims: true)
            var indices = MLXArray(0 ..< logits.dim(logits.ndim - 1)).reshaped([1, -1]).asType(.int32)
            if logits.ndim > 1 {
                indices = MLX.broadcast(indices, to: logits.shape)
            }
            let firstMaxIndices = MLX.where(logits .== maxValues, indices, MLXArray(Int32.max))
            return MLX.min(firstMaxIndices, axis: -1).asType(.int32)
        }

        let greedy = firstMax(biased)
        let sorted = argSort(-biased, axis: -1)
        eval(greedy, sorted)

        let firstToken = Int(greedy.item(Int32.self))
        let top10 = Array(sorted[0].asArray(Int32.self).prefix(10)).map(Int.init)

        let semanticCode = clip(
            greedy - Int32(model.config.semanticStartTokenID),
            min: 0,
            max: Int32(model.config.audioDecoderConfig.vocabSize - 1)
        ).asType(.int32)
        var codebooks = [Int(semanticCode.item(Int32.self))]
        let fastCache = model.model.makeFastCache()
        let fastPrefill = model.model.fastForwardCached(
            result.hiddenStates[0..., (result.hiddenStates.dim(1) - 1)..<result.hiddenStates.dim(1), 0...]
                .squeezed(axis: 1),
            cache: fastCache
        )
        eval(fastPrefill)
        var fastHidden = model.model.fastEmbeddings(semanticCode)
        for _ in 0 ..< (model.model.numCodebooks - 1) {
            let residualLogits = model.model.fastForwardCached(fastHidden, cache: fastCache)
            let residualToken = firstMax(residualLogits).asType(.int32)
            eval(residualToken)
            codebooks.append(Int(residualToken.item(Int32.self)))
            fastHidden = model.model.fastEmbeddings(residualToken)
        }

        #expect(result.logits.dtype == .bfloat16)
        #expect(result.hiddenStates.dtype == .bfloat16)
        #expect(model.model.embeddings.weight.dtype == .uint32)
        #expect(firstToken == 153_005)
        #expect(top10 == [153_005, 153_352, 154_140, 155_645, 153_743, 154_165, 154_636, 153_616, 155_380, 155_668])
        #expect(codebooks == [1327, 917, 130, 446, 138, 836, 850, 370, 643, 383])
    }
}

@Suite("Echo TTS Network Tests", .serialized)
struct EchoTTSNetworkTests {

    @Test func echoTTSBaseLoadsConfiguredCodecAndGeneratesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network Echo TTS test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_ECHO_TTS_REPO"] ?? "mlx-community/echo-tts-base"
        let model = try await EchoTTSModel.fromPretrained(repo)
        let refAudio = try loadTTSNetworkFixture(sampleRate: model.sampleRate, maxSamples: model.sampleRate / 4)

        if repo == "mlx-community/echo-tts-base" {
            #expect(model.config.fishCodecRepo == "jordand/fish-s1-dac-min")
        }

        let result = try model.generateDetailed(
            text: "hello",
            refAudio: refAudio,
            rngSeed: 0,
            numSteps: 1,
            sequenceLength: 8
        )

        #expect(result.audio.shape[0] > 0)
        #expect(result.info.generationTokenCount == 8)
        #expect(model.fishAE != nil)
        #expect(model.pcaState != nil)
    }
}

// MARK: - KittenTTS Tests

@Suite("KittenTTS")
struct KittenTTSTests {
    @Test func textCleanerMapsIPASymbols() {
        let tokens = KittenTTSTextCleaner.cleanText("hello")
        #expect(tokens.count == 5)
        #expect(tokens.allSatisfy { $0 >= 0 })

        let ipaTokens = KittenTTSTextCleaner.cleanText("həlˈoʊ")
        #expect(ipaTokens.count > 0)
        #expect(ipaTokens.allSatisfy { $0 >= 0 })
    }

    @Test func configDecodesFromJSON() throws {
        let json = """
        {
            "model_type": "kitten_tts",
            "hidden_dim": 128, "max_conv_dim": 256, "max_dur": 50,
            "n_layer": 2, "n_mels": 80, "n_token": 178, "style_dim": 128,
            "text_encoder_kernel_size": 5, "asr_res_dim": 64, "sample_rate": 24000,
            "plbert": {
                "num_hidden_layers": 12, "num_attention_heads": 12,
                "hidden_size": 768, "intermediate_size": 2048,
                "max_position_embeddings": 512, "embedding_size": 128,
                "inner_group_num": 1, "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2, "layer_norm_eps": 1e-12
            },
            "istftnet": {
                "resblock_kernel_sizes": [3, 3], "upsample_rates": [10, 6],
                "upsample_initial_channel": 256,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes": [20, 12],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
            },
            "voice_aliases": {"Bella": "expr-voice-2-f"},
            "voices_path": "voices.npz"
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(KittenTTSConfig.self, from: json)
        #expect(config.modelType == "kitten_tts")
        #expect(config.sampleRate == 24000)
        #expect(config.hiddenDim == 128)
        #expect(config.plbert.numHiddenLayers == 12)
        #expect(config.istftnet.upsampleRates == [10, 6])
        #expect(config.voiceAliases?["Bella"] == "expr-voice-2-f")
    }

    @Test func modelStructureMatchesWeightKeys() throws {
        // Integration test: requires model downloaded locally. Set MLXAUDIO_TEST_MODEL_DIR or skip.
        guard let dirPath = ProcessInfo.processInfo.environment["MLXAUDIO_TEST_MODEL_DIR"] else {
            print("⚠️ Skipping: set MLXAUDIO_TEST_MODEL_DIR to model directory")
            return
        }
        let modelDir = URL(fileURLWithPath: dirPath)
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("⚠️ Skipping: config.json not found at \(configURL.path)")
            return
        }

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(KittenTTSConfig.self, from: configData)
        let model = KittenTTSModel.testInit(config: config)

        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let sanitized = model.sanitize(weights: rawWeights)

        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let weightKeys = Set(sanitized.keys)

        let missingInModel = weightKeys.subtracting(modelKeys)
        let missingInWeights = modelKeys.subtracting(weightKeys)

        if !missingInModel.isEmpty {
            print("❌ Weight keys not found in model (\(missingInModel.count)):")
            for k in missingInModel.sorted().prefix(20) { print("  \(k)") }
        }
        if !missingInWeights.isEmpty {
            print("⚠️ Model keys not in weights (\(missingInWeights.count)):")
            for k in missingInWeights.sorted().prefix(20) { print("  \(k)") }
        }

        #expect(missingInModel.count == 0, "Weight keys not matched by model structure")
    }

    @Test func textCleanerHandlesSpecialCharacters() {
        let empty = KittenTTSTextCleaner.cleanText("")
        #expect(empty.isEmpty)

        let punctuation = KittenTTSTextCleaner.cleanText("Hello, world!")
        #expect(punctuation.count == "Hello, world!".count)

        let unknown = KittenTTSTextCleaner.cleanText("日本語")
        #expect(unknown.isEmpty)
    }

    @Test func textCleanerSymbolTableIsComplete() {
        let symbolTable = KittenTTSTextCleaner.symbolToIndex
        #expect(symbolTable.count >= 170)
        #expect(symbolTable["$"] == 0)
        #expect(symbolTable[";"] == 1)
        #expect(symbolTable["A"] != nil)
        #expect(symbolTable["ɑ"] != nil)
        #expect(symbolTable["ᵻ"] != nil)
    }

    @Test func configDefaultValues() throws {
        let minimalJSON = """
        {
            "model_type": "kitten_tts",
            "hidden_dim": 128, "max_conv_dim": 256, "max_dur": 50,
            "n_layer": 2, "n_mels": 80, "n_token": 178, "style_dim": 128,
            "text_encoder_kernel_size": 5, "asr_res_dim": 64,
            "plbert": {
                "num_hidden_layers": 12, "num_attention_heads": 12,
                "hidden_size": 768, "intermediate_size": 2048,
                "max_position_embeddings": 512, "embedding_size": 128,
                "inner_group_num": 1, "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2, "layer_norm_eps": 1e-12
            },
            "istftnet": {
                "resblock_kernel_sizes": [3, 3], "upsample_rates": [10, 6],
                "upsample_initial_channel": 256,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes": [20, 12],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(KittenTTSConfig.self, from: minimalJSON)
        #expect(config.sampleRate == 24000)
        #expect(config.voicesPath == "voices.npz")
        #expect(config.voiceAliases == nil)
        #expect(config.speedPriors == nil)
        #expect(config.decoderOutDim == nil)
    }

    @Test func voiceAliasResolution() throws {
        let config = try JSONDecoder().decode(KittenTTSConfig.self, from: """
        {
            "model_type": "kitten_tts",
            "hidden_dim": 128, "max_conv_dim": 256, "max_dur": 50,
            "n_layer": 2, "n_mels": 80, "n_token": 178, "style_dim": 128,
            "text_encoder_kernel_size": 5, "asr_res_dim": 64,
            "voice_aliases": {"Bella": "expr-voice-2-f", "Luna": "expr-voice-3-f"},
            "plbert": {
                "num_hidden_layers": 12, "num_attention_heads": 12,
                "hidden_size": 768, "intermediate_size": 2048,
                "max_position_embeddings": 512, "embedding_size": 128,
                "inner_group_num": 1, "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2, "layer_norm_eps": 1e-12
            },
            "istftnet": {
                "resblock_kernel_sizes": [3, 3], "upsample_rates": [10, 6],
                "upsample_initial_channel": 256,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes": [20, 12],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
            }
        }
        """.data(using: .utf8)!)
        #expect(config.voiceAliases?["Bella"] == "expr-voice-2-f")
        #expect(config.voiceAliases?["Luna"] == "expr-voice-3-f")
        #expect(config.voiceAliases?["Hugo"] == nil)
    }

    @Test func factoryInfersKittenModelType() throws {
        let resolved = TTS.resolveModelType(modelRepo: "mlx-community/kitten-tts-nano-0.8-8bit")
        #expect(resolved == "kitten_tts")
        let resolved2 = TTS.resolveModelType(modelRepo: "mlx-community/kitten-tts-mini-0.8")
        #expect(resolved2 == "kitten_tts")
    }
}


// MARK: - Kokoro TTS Tests

private let kokoroConfigJSON = """
{
    "model_type": "kokoro",
    "hidden_dim": 512, "n_token": 178, "dim_in": 64, "dropout": 0.2,
    "max_conv_dim": 512, "max_dur": 50, "multispeaker": false,
    "n_layer": 3, "n_mels": 80, "style_dim": 128,
    "text_encoder_kernel_size": 5, "asr_res_dim": 64,
    "vocab": {"a": 1, "b": 2, "c": 3, "h": 4, "e": 5, "l": 6, "o": 7, " ": 8},
    "plbert": {
        "num_hidden_layers": 12, "num_attention_heads": 12,
        "hidden_size": 768, "intermediate_size": 2048,
        "max_position_embeddings": 512, "embedding_size": 128,
        "inner_group_num": 1, "num_hidden_groups": 1,
        "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
        "type_vocab_size": 2, "layer_norm_eps": 1e-12
    },
    "istftnet": {
        "resblock_kernel_sizes": [3, 7, 11], "upsample_rates": [10, 6],
        "upsample_initial_channel": 512,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_kernel_sizes": [20, 12],
        "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
    }
}
"""

private func makeKokoroConfig() throws -> KokoroConfig {
    try JSONDecoder().decode(KokoroConfig.self, from: kokoroConfigJSON.data(using: .utf8)!)
}

@Suite("KokoroTTS")
struct KokoroTTSTests {

    @Test func configDecodesFromJSON() throws {
        let config = try makeKokoroConfig()
        #expect(config.modelType == "kokoro")
        #expect(config.hiddenDim == 512)
        #expect(config.nToken == 178)
        #expect(config.dimIn == 64)
        #expect(config.dropout == 0.2)
        #expect(config.multispeaker == false)
        #expect(config.styleDim == 128)
        #expect(config.vocab["a"] == 1)
        #expect(config.vocab.count == 8)
        #expect(config.plbert.numHiddenLayers == 12)
        #expect(config.istftnet.upsampleRates == [10, 6])
    }

    @Test func configDefaultValues() throws {
        let json = """
        {
            "model_type": "kokoro", "hidden_dim": 512, "n_token": 178,
            "vocab": {"x": 1},
            "plbert": {
                "num_hidden_layers": 12, "num_attention_heads": 12,
                "hidden_size": 768, "intermediate_size": 2048,
                "max_position_embeddings": 512, "embedding_size": 128,
                "inner_group_num": 1, "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2, "layer_norm_eps": 1e-12
            },
            "istftnet": {
                "resblock_kernel_sizes": [3], "upsample_rates": [10, 6],
                "upsample_initial_channel": 512,
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_kernel_sizes": [20, 12],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(KokoroConfig.self, from: json)
        #expect(config.sampleRate == 24000)
        #expect(config.dimIn == 64)
        #expect(config.dropout == 0.2)
        #expect(config.maxConvDim == 512)
        #expect(config.maxDur == 50)
        #expect(config.nLayer == 3)
        #expect(config.nMels == 80)
        #expect(config.styleDim == 128)
        #expect(config.textEncoderKernelSize == 5)
        #expect(config.asrResDim == 64)
        #expect(config.voicesPath == nil)
        #expect(config.voiceAliases == nil)
        #expect(config.speedPriors == nil)
        #expect(config.quantization == nil)
    }

    @Test func configDecodesQuantizationAlias() throws {
        let json = """
        {
            "model_type": "kokoro", "hidden_dim": 512, "n_token": 178,
            "vocab": {"x": 1},
            "quantization_config": {"group_size": 64, "bits": 4},
            "plbert": {
                "num_hidden_layers": 12, "num_attention_heads": 12,
                "hidden_size": 768, "intermediate_size": 2048,
                "max_position_embeddings": 512, "embedding_size": 128,
                "inner_group_num": 1, "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2, "layer_norm_eps": 1e-12
            },
            "istftnet": {
                "resblock_kernel_sizes": [3], "upsample_rates": [10, 6],
                "upsample_initial_channel": 512,
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_kernel_sizes": [20, 12],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(KokoroConfig.self, from: json)
        #expect(config.quantization == BaseConfiguration.Quantization(groupSize: 64, bits: 4))
    }

    @Test func tokenizerConvertsCharsToIDs() throws {
        let config = try makeKokoroConfig()
        let tokens = "hello".compactMap { config.vocab[String($0)] }
        #expect(tokens == [4, 5, 6, 6, 7])
    }

    @Test func tokenizerSkipsUnknownChars() throws {
        let config = try makeKokoroConfig()
        let tokens = "a日b".compactMap { config.vocab[String($0)] }
        #expect(tokens == [1, 2])
    }

    @Test func tokenizerEmptyString() throws {
        let config = try makeKokoroConfig()
        let tokens = "".compactMap { config.vocab[String($0)] }
        #expect(tokens.isEmpty)
    }

    @Test func sanitizeSkipsPositionIds() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let sanitized = model.sanitize(weights: [
            "bert.embeddings.position_ids": MLXArray.zeros([1, 512]),
            "bert.encoder.position_ids": MLXArray.zeros([1]),
            "bert.embeddings.word_embeddings.weight": MLXArray.zeros([1, 1]),
        ])
        #expect(sanitized["bert.embeddings.position_ids"] == nil)
        #expect(sanitized["bert.encoder.position_ids"] == nil)
        #expect(sanitized["bert.embeddings.word_embeddings.weight"] != nil)
    }

    @Test func sanitizeRemapsLSTMKeys() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let sanitized = model.sanitize(weights: [
            "predictor.lstm.weight_ih_l0": MLXArray.zeros([1]),
            "predictor.lstm.weight_hh_l0": MLXArray.zeros([1]),
            "predictor.lstm.bias_ih_l0": MLXArray.zeros([1]),
            "predictor.lstm.bias_hh_l0": MLXArray.zeros([1]),
            "predictor.lstm.weight_ih_l0_reverse": MLXArray.zeros([1]),
            "predictor.lstm.weight_hh_l0_reverse": MLXArray.zeros([1]),
            "predictor.lstm.bias_ih_l0_reverse": MLXArray.zeros([1]),
            "predictor.lstm.bias_hh_l0_reverse": MLXArray.zeros([1]),
        ])
        #expect(sanitized["predictor.lstm.Wx_forward"] != nil)
        #expect(sanitized["predictor.lstm.Wh_forward"] != nil)
        #expect(sanitized["predictor.lstm.bias_ih_forward"] != nil)
        #expect(sanitized["predictor.lstm.bias_hh_forward"] != nil)
        #expect(sanitized["predictor.lstm.Wx_backward"] != nil)
        #expect(sanitized["predictor.lstm.Wh_backward"] != nil)
        #expect(sanitized["predictor.lstm.bias_ih_backward"] != nil)
        #expect(sanitized["predictor.lstm.bias_hh_backward"] != nil)
        #expect(sanitized["predictor.lstm.weight_ih_l0"] == nil)
        #expect(sanitized["predictor.lstm.weight_ih_l0_reverse"] == nil)
    }

    @Test func sanitizeRemapsLayerNormKeys() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let sanitized = model.sanitize(weights: [
            "bert.embeddings.LayerNorm.gamma": MLXArray.zeros([1]),
            "bert.embeddings.LayerNorm.beta": MLXArray.zeros([1]),
        ])
        #expect(sanitized["bert.embeddings.LayerNorm.weight"] != nil)
        #expect(sanitized["bert.embeddings.LayerNorm.bias"] != nil)
        #expect(sanitized["bert.embeddings.LayerNorm.gamma"] == nil)
    }

    @Test func sanitizeRemapsAlphaKeys() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let sanitized = model.sanitize(weights: [
            "decoder.generator.resblocks.0.alpha1.0": MLXArray.zeros([1]),
            "decoder.generator.resblocks.0.alpha2.0": MLXArray.zeros([1]),
        ])
        #expect(sanitized["decoder.generator.resblocks.0.alpha1_0"] != nil)
        #expect(sanitized["decoder.generator.resblocks.0.alpha2_0"] != nil)
    }

    @Test func sanitizeTransposesF0NProj() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let w = MLXArray(Array(stride(from: Float(0), to: 24, by: 1))).reshaped([2, 3, 4])
        let sanitized = model.sanitize(weights: [
            "predictor.F0_proj.weight": w,
            "predictor.N_proj.weight": w,
        ])
        #expect(sanitized["predictor.F0_proj.weight"]!.shape == [2, 4, 3])
        #expect(sanitized["predictor.N_proj.weight"]!.shape == [2, 4, 3])
    }

    @Test func sanitizeTransposesNoiseConvs() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let w = MLXArray(Array(stride(from: Float(0), to: 24, by: 1))).reshaped([2, 3, 4])
        let sanitized = model.sanitize(weights: [
            "decoder.generator.noise_convs.0.weight": w,
        ])
        #expect(sanitized["decoder.generator.noise_convs.0.weight"]!.shape == [2, 4, 3])
    }

    @Test func sanitizeTransposesNonCanonicalWeightV() throws {
        guard metalAvailable else { return }
        let config = try makeKokoroConfig()
        let model = KokoroModel.testInit(config: config)
        let canonical = MLXArray.zeros([8, 3, 3])
        let nonCanonical = MLXArray.zeros([3, 8, 1])
        let sanitized = model.sanitize(weights: [
            "text_encoder.cnn.0.0.weight_v": canonical,
            "decoder.encode.conv1.weight_v": nonCanonical,
        ])
        #expect(sanitized["text_encoder.cnn.0.0.weight_v"]!.shape == [8, 3, 3])
        #expect(sanitized["decoder.encode.conv1.weight_v"]!.shape == [3, 1, 8])
    }

    @Test func factoryInfersKokoroModelType() {
        let repoNames = ["mlx-community/Kokoro-82M-bf16", "mlx-community/kokoro-v1-8bit"]
        for name in repoNames {
            #expect(name.lowercased().contains("kokoro"))
        }
    }

    @Test func configDecodesMinimalPLBert() throws {
        let json = """
        {
            "model_type": "kokoro", "hidden_dim": 512, "n_token": 178,
            "vocab": {"x": 1},
            "plbert": {
                "hidden_size": 768, "num_attention_heads": 12,
                "intermediate_size": 2048, "max_position_embeddings": 512,
                "num_hidden_layers": 12, "dropout": 0.1
            },
            "istftnet": {
                "resblock_kernel_sizes": [3], "upsample_rates": [10, 6],
                "upsample_initial_channel": 512,
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_kernel_sizes": [20, 12],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(KokoroConfig.self, from: json)
        #expect(config.plbert.hiddenSize == 768)
        #expect(config.plbert.embeddingSize == 128)
        #expect(config.plbert.innerGroupNum == 1)
        #expect(config.plbert.numHiddenGroups == 1)
        #expect(config.plbert.hiddenDropoutProb == 0.1)
        #expect(config.plbert.typeVocabSize == 2)
    }

    @Test func modelStructureMatchesWeightKeys() throws {
        guard metalAvailable else { return }
        guard let dirPath = ProcessInfo.processInfo.environment["MLXAUDIO_KOKORO_MODEL_DIR"] else {
            print("⚠️ Skipping: set MLXAUDIO_KOKORO_MODEL_DIR to model directory")
            return
        }
        let modelDir = URL(fileURLWithPath: dirPath)
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("⚠️ Skipping: config.json not found at \(configURL.path)")
            return
        }

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(KokoroConfig.self, from: configData)
        let model = KokoroModel.testInit(config: config)

        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let sanitized = model.sanitize(weights: rawWeights)

        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let weightKeys = Set(sanitized.keys)

        let missingInModel = weightKeys.subtracting(modelKeys)
        let missingInWeights = modelKeys.subtracting(weightKeys)

        if !missingInModel.isEmpty {
            print("❌ Weight keys not found in model (\(missingInModel.count)):")
            for k in missingInModel.sorted().prefix(20) { print("  \(k)") }
        }
        if !missingInWeights.isEmpty {
            print("⚠️ Model keys not in weights (\(missingInWeights.count)):")
            for k in missingInWeights.sorted().prefix(20) { print("  \(k)") }
        }

        #expect(missingInModel.count == 0, "Weight keys not matched by model structure")
    }
}

// MARK: - Kokoro Multilingual Processor Tests

@Suite("KokoroMultilingualProcessor")
struct KokoroMultilingualProcessorTests {

    @Test func voiceLanguageMapCoversAllPrefixes() {
        let map = KokoroMultilingualProcessor.voiceLanguageMap
        #expect(map["a"] == "en-us")
        #expect(map["b"] == "en-gb")
        #expect(map["e"] == "es")
        #expect(map["f"] == "fr")
        #expect(map["h"] == "hi")
        #expect(map["i"] == "it")
        #expect(map["j"] == "ja")
        #expect(map["p"] == "pt")
        #expect(map["z"] == "cmn")
        #expect(map.count == 9)
    }

    @Test func languageForVoiceInfersCorrectly() {
        #expect(KokoroMultilingualProcessor.languageForVoice("af_heart") == "en-us")
        #expect(KokoroMultilingualProcessor.languageForVoice("am_adam") == "en-us")
        #expect(KokoroMultilingualProcessor.languageForVoice("bf_emma") == "en-gb")
        #expect(KokoroMultilingualProcessor.languageForVoice("ef_dora") == "es")
        #expect(KokoroMultilingualProcessor.languageForVoice("ff_siwis") == "fr")
        #expect(KokoroMultilingualProcessor.languageForVoice("hf_alpha") == "hi")
        #expect(KokoroMultilingualProcessor.languageForVoice("if_sara") == "it")
        #expect(KokoroMultilingualProcessor.languageForVoice("jf_alpha") == "ja")
        #expect(KokoroMultilingualProcessor.languageForVoice("pf_dora") == "pt")
        #expect(KokoroMultilingualProcessor.languageForVoice("zf_xiaobei") == "cmn")
    }

    @Test func languageForVoiceReturnsNilForEmpty() {
        #expect(KokoroMultilingualProcessor.languageForVoice("") == nil)
    }

    @Test func languageForVoiceReturnsNilForUnknownPrefix() {
        #expect(KokoroMultilingualProcessor.languageForVoice("xf_unknown") == nil)
    }

    @Test func processEnglishDelegatesToMisaki() async throws {
        guard metalAvailable else { return }
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }
        let processor = KokoroMultilingualProcessor()
        try await processor.prepare(for: "en-us")
        let result = try processor.process(text: "hello", language: "en-us")
        #expect(!result.isEmpty)
        #expect(result != "hello")
    }

    @Test func processEnglishGBDelegatesToMisaki() async throws {
        guard metalAvailable else { return }
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }
        let processor = KokoroMultilingualProcessor()
        try await processor.prepare(for: "en-gb")
        let result = try processor.process(text: "hello", language: "en-gb")
        #expect(!result.isEmpty)
        #expect(result != "hello")
    }

    @Test func processNilLanguageDefaultsToEnglish() async throws {
        guard metalAvailable else { return }
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }
        let processor = KokoroMultilingualProcessor()
        try await processor.prepare(for: "en-us")
        let result = try processor.process(text: "hello", language: nil)
        #expect(!result.isEmpty)
        #expect(result != "hello")
    }

    @Test func processUnsupportedLanguageThrows() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        #expect(throws: LexiconError.self) {
            try processor.process(text: "test", language: "xyz")
        }
    }

    @Test func processLexiconLangWithoutDownloadThrows() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor(lexiconRepo: "nonexistent/repo")
        #expect(throws: LexiconError.self) {
            try processor.process(text: "hola", language: "es")
        }
    }

    @Test func processNeuralLangWithoutModelThrows() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor(neuralG2PRepo: "nonexistent/repo")
        #expect(throws: LexiconError.self) {
            try processor.process(text: "こんにちは", language: "ja")
        }
    }

    @Test func splitWordsCJKSplitsByCharacter() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let tokens = processor.splitWords(text: "こんにちは世界", lang: "ja")
        #expect(tokens == ["こ", "ん", "に", "ち", "は", "世", "界"])
    }

    @Test func splitWordsCJKSkipsWhitespace() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let tokens = processor.splitWords(text: "你好 世界", lang: "cmn")
        #expect(tokens == ["你", "好", "世", "界"])
    }

    @Test func splitWordsRegularSplitsByPunctuation() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let tokens = processor.splitWords(text: "hola, mundo!", lang: "es")
        #expect(tokens == ["hola", ",", "mundo", "!"])
    }

    @Test func splitWordsHindiSplitsBySpace() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let tokens = processor.splitWords(text: "नमस्ते दुनिया", lang: "hi")
        #expect(tokens == ["नमस्ते", "दुनिया"])
    }

    @Test func lookupWordDirectMatch() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let lexicon = ["hola": "ˈola", "mundo": "ˈmundo"]
        #expect(processor.lookupWord("hola", lexicon: lexicon) == "ˈola")
        #expect(processor.lookupWord("mundo", lexicon: lexicon) == "ˈmundo")
    }

    @Test func lookupWordFallsBackToOriginal() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let lexicon = ["hola": "ˈola"]
        #expect(processor.lookupWord("unknown", lexicon: lexicon) == "unknown")
    }

    @Test func lookupWordAccentStrippedFallback() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let lexicon = ["mas": "ˈmas"]
        #expect(processor.lookupWord("más", lexicon: lexicon) == "ˈmas")
    }

    @Test func phonemizePreservesPunctuation() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let lexicon = ["hola": "ˈola", "mundo": "ˈmundo"]
        let result = processor.phonemize(text: "Hola, mundo!", lexicon: lexicon)
        #expect(result == "ˈola , ˈmundo !")
    }

    @Test func phonemizePassesThroughUnknownWords() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let lexicon = ["hola": "ˈola"]
        let result = processor.phonemize(text: "hola xyz", lexicon: lexicon)
        #expect(result.contains("ˈola"))
        #expect(result.contains("xyz"))
    }

    @Test func phonemizeHandlesEmptyText() {
        guard metalAvailable else { return }
        let processor = KokoroMultilingualProcessor()
        let result = processor.phonemize(text: "", lexicon: [:])
        #expect(result.isEmpty)
    }

    @Test func initDefaultRepos() async throws {
        guard metalAvailable else { return }
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }
        let processor = KokoroMultilingualProcessor()
        try await processor.prepare(for: "en-us")
        let result = try processor.process(text: "hello", language: "en-us")
        #expect(!result.isEmpty)
    }

    @Test func prepareForEnglishSupportsAllEnglishVariants() async throws {
        guard metalAvailable else { return }
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }
        let processor = KokoroMultilingualProcessor()
        try await processor.prepare(for: "en-us")
        try await processor.prepare(for: "en-gb")
        try await processor.prepare(for: "en")
    }
}
