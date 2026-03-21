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
import MLXLMCommon
import Foundation

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
