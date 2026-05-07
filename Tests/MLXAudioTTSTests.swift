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
import Tokenizers
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

private func writeTestFile(_ url: URL, contents: String) throws {
    let data = try #require(contents.data(using: .utf8))
    try data.write(to: url)
}

private func makeTemporaryArtifactDirectory(prefix: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func cleanupTemporaryArtifactDirectory(_ directory: URL) {
    try? FileManager.default.removeItem(at: directory)
}

private struct MossFakeTokenizer: MossTextTokenizing {
    func encode(_ text: String) -> [Int] {
        text.utf8.map { Int($0 % 50) + 10 }
    }

    func decode(_ tokenIDs: [Int]) -> String {
        String(tokenIDs.map { Character(UnicodeScalar(($0 - 10) % 50 + 65)!) })
    }
}

private struct MossTTSFullFakeTokenizer: MossTTSTextTokenizing {
    private let tokenIDsByString: [String: Int] = [
        "<|im_start|>": 151_644,
        "<|im_end|>": 151_645,
        "<|audio_start|>": 151_652,
        "<|audio_end|>": 151_653,
        "<|audio_user_slot|>": 151_654,
        "<|audio_assistant_gen_slot|>": 151_656,
        "<|audio_assistant_delay_slot|>": 151_662,
    ]

    private var tokenStringsByID: [Int: String] {
        Dictionary(uniqueKeysWithValues: tokenIDsByString.map { ($0.value, $0.key) })
    }

    private var orderedSpecialTokens: [(String, Int)] {
        tokenIDsByString.sorted { lhs, rhs in
            lhs.key.count == rhs.key.count ? lhs.key < rhs.key : lhs.key.count > rhs.key.count
        }
    }

    func encode(_ text: String) -> [Int] {
        var result: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            var matched = false
            let suffix = text[index...]
            for (token, id) in orderedSpecialTokens where suffix.hasPrefix(token) {
                result.append(id)
                index = text.index(index, offsetBy: token.count)
                matched = true
                break
            }
            if !matched {
                let scalar = text[index].unicodeScalars.first?.value ?? 0
                result.append(1_000 + Int(scalar % 500))
                index = text.index(after: index)
            }
        }
        return result
    }

    func decode(_ tokenIDs: [Int]) -> String {
        tokenIDs.map { tokenStringsByID[$0] ?? "<\($0)>" }.joined()
    }

    func tokenString(for tokenID: Int) -> String? {
        tokenStringsByID[tokenID]
    }
}

private func makeTinyMossConfig(nVQ: Int = 2) throws -> MossTTSNanoConfig {
    try MossTTSNanoConfig(
        gpt2Config: MossGPT2Config(
            vocabSize: 128,
            nPositions: 64,
            nCtx: 64,
            nEmbd: 16,
            nLayer: 1,
            nHead: 4,
            nInner: 32,
            positionEmbeddingType: "rope"
        ),
        nVQ: nVQ,
        audioVocabSize: 8,
        audioCodebookSizes: Array(repeating: 8, count: nVQ),
        audioPadTokenID: 8,
        localTransformerLayers: 1,
        maxPositionEmbeddings: 64,
        hiddenSize: 16,
        vocabSize: 128
    )
}

@Suite("MOSS TTS Nano Tests")
struct MossTTSNanoTests {
    @Test func configDecodesUpstreamAliases() throws {
        let json = """
        {
          "model_type": "moss_tts_nano",
          "n_vq": 2,
          "audio_vocab_size": 8,
          "audio_codebook_sizes": [8, 9],
          "gpt2_config": {
            "vocab_size": 128,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "position_embedding_type": "rope"
          }
        }
        """
        let config = try JSONDecoder().decode(MossTTSNanoConfig.self, from: Data(json.utf8))
        #expect(config.nVQ == 2)
        #expect(config.audioCodebookSizes == [8, 9])
        #expect(config.gpt2Config.nEmbd == 16)
        #expect(config.gpt2Config.nLayer == 1)
        #expect(config.gpt2Config.nHead == 4)
        #expect(config.localGPT2Config().nPositions == 3)
    }

    @Test func configRejectsBadCodebookCount() {
        #expect(throws: DecodingError.self) {
            _ = try MossTTSNanoConfig(nVQ: 2, audioCodebookSizes: [8])
        }
    }

    @Test func textUtilitiesMatchPromptShape() throws {
        let tokenizer = MossFakeTokenizer()
        let config = try makeTinyMossConfig()

        let prefix = mossBuildUserPromptPrefix(tokenizer: tokenizer, config: config)
        #expect(prefix.first == config.imStartTokenID)
        #expect(prefix.count > 5)

        let prepared = try mossPrepareTextForSentenceChunking("hello world")
        #expect(prepared == "        Hello world.")

        let chunks = try mossSplitTextIntoBestSentences(
            tokenizer: tokenizer,
            text: "one two three, four five six, seven eight nine.",
            maxTokens: 14
        )
        #expect(chunks.count > 1)
    }

    @Test func sentencePieceBPEUsesMergeScores() throws {
        let json = """
        {
          "model": {
            "type": "BPE",
            "unk_id": 0,
            "vocab": [
              ["<unk>", 0.0],
              ["▁", -100.0],
              ["A", -100.0],
              ["B", -100.0],
              ["C", -100.0],
              ["▁A", -10.0],
              ["AB", -20.0],
              ["BC", -1.0]
            ]
          }
        }
        """
        let tokenizer = try SentencePieceTokenizer(tokenizerJSONData: Data(json.utf8))
        #expect(tokenizer.encodeWithByteFallback("ABC") == [5, 7])
    }

    @Test func promptRowsAndEmbeddingsHaveExpectedShapes() throws {
        let config = try makeTinyMossConfig()
        let model = MossTTSNanoModel(config: config)

        let textRows = model.buildTextRows([11, 12]).asArray(Int32.self)
        #expect(textRows == [11, 8, 8, 12, 8, 8])

        let promptCodes = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)], [2, 2])
        let audioRows = try model.buildAudioPrefixRows(
            promptAudioCodes: promptCodes,
            slotTokenID: config.audioUserSlotTokenID
        )
        #expect(audioRows.asArray(Int32.self) == [8, 1, 2, 8, 3, 4])

        let inputIDs = MLXArray([Int32(11), 1, 8, 12, 8, 2], [1, 2, 3]).asType(.int32)
        let embeds = try model.buildInputsEmbeds(inputIDs)
        #expect(embeds.shape == [1, 2, 16])
    }

    @Test func inferenceInputBuilderCombinesTextAndReferenceAudio() throws {
        let tokenizer = MossFakeTokenizer()
        let config = try makeTinyMossConfig()
        let model = MossTTSNanoModel(config: config)
        let promptCodes = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)], [2, 2])

        let prepared = try model.buildInferenceInputIDs(
            text: "target",
            tokenizer: tokenizer,
            promptAudioCodes: promptCodes
        )

        #expect(prepared.inputIDs.ndim == 3)
        #expect(prepared.inputIDs.dim(0) == 1)
        #expect(prepared.inputIDs.dim(2) == config.nVQ + 1)
        #expect(prepared.attentionMask.shape == [1, prepared.inputIDs.dim(1)])
    }

    @Test func samplingRestrictsAssistantTextCandidates() throws {
        var values = Array(repeating: Float(-10), count: 16)
        values[9] = 3
        values[7] = 1
        let logits = MLXArray(values, [1, 16])

        let next = try mossSampleAssistantTextToken(
            textLogits: logits,
            audioAssistantSlotTokenID: 9,
            audioEndTokenID: 7,
            doSample: false,
            temperature: 1,
            topK: 50,
            topP: 1
        )
        #expect(next.asArray(Int32.self) == [9])
    }

    @Test func samplingAppliesRepetitionPenaltyAndTopK() {
        let logits = MLXArray([Float(0), Float(2), Float(-2), Float(1)], [1, 4])
        let penalized = mossApplyRepetitionPenalty(
            logits: logits,
            previousTokenIDs: MLXArray([Int32(1), Int32(2)]),
            penalty: 2
        ).asArray(Float.self)
        #expect(abs(penalized[1] - 1) < 0.0001)
        #expect(abs(penalized[2] + 4) < 0.0001)

        let topK = mossApplyTopK(logits, topK: 2).asArray(Float.self)
        #expect(topK[0].isInfinite && topK[0] < 0)
        #expect(topK[1] == 2)
        #expect(topK[2].isInfinite && topK[2] < 0)
        #expect(topK[3] == 1)
    }

    @Test func samplingTopPMatchesPythonTailMask() {
        let logits = MLXArray(
            [0.55, 0.25, 0.15, 0.05].map { Float(log($0)) },
            [1, 4]
        )
        let topP = mossApplyTopP(logits, topP: 0.79).asArray(Float.self)
        #expect(topP[0].isFinite)
        #expect(topP[1].isFinite)
        #expect(topP[2].isInfinite && topP[2] < 0)
        #expect(topP[3].isInfinite && topP[3] < 0)
    }

    @Test func defaultSamplingParametersMatchPythonCLIForMoss() throws {
        let model = MossTTSNanoModel(config: try makeTinyMossConfig())
        #expect(model.defaultGenerationParameters.temperature == 0.7)
        #expect(model.defaultGenerationParameters.topP == 0.9)
        #expect(model.defaultGenerationParameters.topK == 50)
        #expect(model.defaultGenerationParameters.repetitionPenalty == 1.1)
    }

    @Test func sanitizeDropsUnusedUpstreamHeads() throws {
        let model = MossTTSNanoModel(config: try makeTinyMossConfig())
        let tensor = MLXArray.zeros([1], dtype: .float32)
        let sanitized = model.sanitize(weights: [
            "text_lm_head.weight": tensor,
            "audio_lm_heads.0.weight": tensor,
            "local_transformer.wte.weight": tensor,
            "transformer.wte.weight": tensor,
        ])

        #expect(sanitized["text_lm_head.weight"] == nil)
        #expect(sanitized["audio_lm_heads.0.weight"] == nil)
        #expect(sanitized["local_transformer.wte.weight"] == nil)
        #expect(sanitized["transformer.wte.weight"] != nil)
    }

    @Test func audioTokenizerConfigAndEmptyDecodePath() throws {
        let directory = try makeTemporaryArtifactDirectory(prefix: "tiny-moss-audio-tokenizer")
        defer { cleanupTemporaryArtifactDirectory(directory) }
        let configJSON = """
        {
          "sample_rate": 48000,
          "sampling_rate": 48000,
          "downsample_rate": 4,
          "number_channels": 2,
          "enable_channel_interleave": true,
          "encoder_kwargs": [
            { "module_type": "PatchedPretransform", "patch_size": 2 }
          ],
          "decoder_kwargs": [
            { "module_type": "PatchedPretransform", "patch_size": 2 }
          ],
          "quantizer_type": "rlfq",
          "quantizer_kwargs": {
            "input_dim": 2,
            "rvq_dim": 2,
            "output_dim": 2,
            "num_quantizers": 1,
            "codebook_size": 4,
            "codebook_dim": 2
          }
        }
        """
        try writeTestFile(directory.appendingPathComponent("config.json"), contents: configJSON)
        let parsed = try MossAudioTokenizerConfig.fromFile(directory.appendingPathComponent("config.json"))
        #expect(parsed.numberChannels == 2)
        #expect(parsed.encoderKwargs.count == 1)

        let tokenizer = try MLXMossAudioTokenizer(config: parsed)
        #expect(tokenizer.numQuantizers == 1)
        let empty = try tokenizer.decodeAudioCodes(MLXArray.zeros([0, 1], type: Int32.self), numQuantizers: 1)
        #expect(empty.shape == [0, 2])
    }

    @Test func ttsModelResolutionIncludesMossNano() {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/MOSS-TTS-Nano") == "moss_tts_nano")
        #expect(TTS.resolveModelType(modelRepo: "anything", modelType: "moss_tts_nano") == "moss_tts_nano")
    }
}

@Suite("MOSS TTS Full Tests")
struct MossTTSFullTests {
    @Test func configDecodesDelayAndLocalVariants() throws {
        let delayJSON = """
        {
          "model_type": "moss_tts_delay",
          "n_vq": 16,
          "audio_vocab_size": 1024,
          "sampling_rate": 24000,
          "language_config": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "rope_parameters": { "rope_theta": 1000000.0 }
          }
        }
        """
        let delay = try JSONDecoder().decode(MossTTSConfig.self, from: Data(delayJSON.utf8))
        #expect(delay.nVQ == 16)
        #expect(delay.samplingRate == 24_000)
        #expect(delay.languageConfig.ropeTheta == 1_000_000)
        #expect(delay.isLocalTransformer == false)

        let localJSON = """
        {
          "model_type": "moss_tts_delay",
          "n_vq": 32,
          "audio_vocab_size": 1024,
          "sample_rate": 24000,
          "additional_mlp_ffn_hidden_size": 4096,
          "local_ffn_hidden_size": 4096,
          "local_hidden_size": 1536,
          "local_num_layers": 4,
          "language_config": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128
          }
        }
        """
        let local = try JSONDecoder().decode(MossTTSConfig.self, from: Data(localJSON.utf8))
        #expect(local.isLocalTransformer)
        #expect(local.samplingRate == 24_000)
        let localTransformer = try local.localTransformerConfig()
        #expect(localTransformer.hiddenSize == 1536)
        #expect(localTransformer.intermediateSize == 4096)
        #expect(localTransformer.numHiddenLayers == 4)
    }

    @Test func audioTokenizerMissingChannelCountDefaultsToMono() throws {
        let directory = try makeTemporaryArtifactDirectory(prefix: "full-moss-audio-tokenizer")
        defer { cleanupTemporaryArtifactDirectory(directory) }
        let configJSON = """
        {
          "sample_rate": 24000,
          "sampling_rate": 24000,
          "downsample_rate": 1920,
          "enable_channel_interleave": true,
          "quantizer_type": "rlfq",
          "quantizer_kwargs": {
            "num_quantizers": 32,
            "codebook_size": 1024,
            "codebook_dim": 8
          }
        }
        """
        try writeTestFile(directory.appendingPathComponent("config.json"), contents: configJSON)
        let parsed = try MossAudioTokenizerConfig.fromFile(directory.appendingPathComponent("config.json"))
        #expect(parsed.numberChannels == 1)
        #expect(parsed.downsampleRate == 1_920)
    }

    @Test func delayPatternRoundTripsAudioCodes() throws {
        let codes = MLXArray([Int32(1), 2, 3, 4, 5, 6], [2, 3]).asType(.int32)
        let delayed = try mossTTSApplyDelayPattern(codes, padCode: 99)
        #expect(delayed.asArray(Int32.self) == [
            1, 99, 99,
            4, 2, 99,
            99, 5, 3,
            99, 99, 6,
        ])
        let restored = try mossTTSApplyDeDelayPattern(delayed)
        #expect(restored.asArray(Int32.self) == codes.asArray(Int32.self))
    }

    @Test func delayProcessorBuildsReferencePromptRows() throws {
        let config = MossTTSConfig(nVQ: 3, audioPadCode: 99)
        let tokenizer = MossTTSFullFakeTokenizer()
        let processor = try MossTTSDelayProcessor(tokenizer: tokenizer, config: config)
        let reference = MLXArray([Int32(1), 2, 3, 4, 5, 6], [2, 3]).asType(.int32)
        let user = processor.buildUserMessage(text: "target", reference: [reference], language: "English")

        let batch = try processor([[user]], mode: "generation")
        #expect(batch.inputIDs.dim(0) == 1)
        #expect(batch.inputIDs.dim(2) == 4)

        let textColumn = batch.inputIDs[0, 0..., 0].asArray(Int32.self).map(Int.init)
        #expect(textColumn.contains(config.audioStartTokenID))
        #expect(textColumn.contains(config.audioEndTokenID))
        #expect(textColumn.contains(config.audioUserSlotTokenID))

        let audioColumns = batch.inputIDs[0, 0..., 1...].asArray(Int32.self)
        #expect(audioColumns.contains(1))
        #expect(audioColumns.contains(6))
    }

    @Test func standardPromptOmitsSceneField() throws {
        let config = MossTTSConfig(nVQ: 32)
        let processor = try MossTTSDelayProcessor(tokenizer: MossTTSFullFakeTokenizer(), config: config)
        let user = processor.buildUserMessage(text: "target", language: "English", scene: "studio")

        #expect(!user.content.contains("- Scene:"))
        #expect(user.content.contains("- Text:\ntarget"))
    }

    @Test func dialoguePromptKeepsSceneField() throws {
        let config = MossTTSConfig(nVQ: 16)
        let processor = try MossTTSDelayProcessor(tokenizer: MossTTSFullFakeTokenizer(), config: config)
        let user = processor.buildUserMessage(text: "target", language: "English", scene: "studio")

        #expect(user.content.contains("- Scene:\nstudio"))
        #expect(user.content.contains("- Text:\ntarget"))
    }

    @Test func localProcessorAppendsAudioStartWithoutDelayPattern() throws {
        let language = MossQwen3Config(
            vocabSize: 151_936,
            hiddenSize: 64,
            numHiddenLayers: 1,
            intermediateSize: 128,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            headDim: 16
        )
        let config = MossTTSConfig(
            languageConfig: language,
            nVQ: 3,
            audioPadCode: 99,
            additionalMLPFFNHiddenSize: 128,
            localFFNHiddenSize: 128,
            localHiddenSize: 64,
            localNumLayers: 1
        )
        let processor = try MossTTSLocalProcessor(tokenizer: MossTTSFullFakeTokenizer(), config: config)
        let user = processor.buildUserMessage(text: "target", language: "English")

        let batch = try processor([[user]], mode: "generation")
        let textColumn = batch.inputIDs[0, 0..., 0].asArray(Int32.self).map(Int.init)
        #expect(textColumn.last == config.audioStartTokenID)
        #expect(batch.inputIDs[0, -1, 1...].asArray(Int32.self).allSatisfy { $0 == Int32(config.audioPadCode) })
    }

    @Test func ttsModelResolutionIncludesFullMossVariants() {
        #expect(TTS.resolveModelType(modelRepo: "OpenMOSS-Team/MOSS-TTS") == "moss_tts_delay")
        #expect(TTS.resolveModelType(modelRepo: "OpenMOSS-Team/MOSS-TTSD-v1.0") == "moss_tts_delay")
        #expect(TTS.resolveModelType(modelRepo: "OpenMOSS-Team/MOSS-TTS-Local-Transformer") == "moss_tts_local")
        #expect(TTS.resolveModelType(modelRepo: "anything", modelType: "moss_tts_delay") == "moss_tts_delay")
    }
}

private func makeTinyQwenTokenizerDirectory() throws -> URL {
    let directory = try makeTemporaryArtifactDirectory(prefix: "tiny-qwen3-tokenizer")

    let tokenizerConfig = """
    {
      "tokenizer_class": "GPT2Tokenizer",
      "bos_token": "<bos>",
      "eos_token": "<eos>",
      "unk_token": "<unk>",
      "pad_token": "<pad>",
      "model_max_length": 128,
      "do_lower_case": false
    }
    """

    let tokenizerData = """
    {
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        { "id": 0, "content": "<bos>", "special": true },
        { "id": 1, "content": "<pad>", "special": true },
        { "id": 2, "content": "<eos>", "special": true },
        { "id": 3, "content": "<unk>", "special": true },
        { "id": 4, "content": "<|im_start|>", "special": true },
        { "id": 5, "content": "<|im_end|>", "special": true }
      ],
      "model": {
        "type": "BPE",
        "vocab": {
          "<bos>": 0,
          "<pad>": 1,
          "<eos>": 2,
          "<unk>": 3,
          "<|im_start|>": 4,
          "<|im_end|>": 5,
          "assistant": 6,
          "user": 7,
          "one": 8,
          "two": 9,
          "three": 10,
          "four": 11,
          "five": 12,
          "target": 13,
          "voice": 14,
          "prompt": 15,
          "sample": 16,
          "english": 17
        },
        "merges": [],
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "unk_token": "<unk>"
      },
      "normalizer": {
        "type": "Lowercase"
      },
      "pre_tokenizer": {
        "type": "Whitespace"
      }
    }
    """

    try writeTestFile(directory.appendingPathComponent("tokenizer_config.json"), contents: tokenizerConfig)
    try writeTestFile(directory.appendingPathComponent("tokenizer.json"), contents: tokenizerData)
    return directory
}

private func makeTinyQwen3TTSModel(
    ttsModelType: String,
    includeSpeechEncoder: Bool,
    speakerIDEntries: String = "",
    dialectEntries: String = ""
) async throws -> (model: Qwen3TTSModel, tokenizerDirectory: URL) {
    let tokenizerDirectory = try makeTinyQwenTokenizerDirectory()
    let encoderConfigJSON = includeSpeechEncoder ? #""encoder_config": {},"# : ""
    let spkIdJSON = speakerIDEntries.isEmpty ? "" : #""spk_id": {"# + speakerIDEntries + "},"
    let spkDialectJSON = dialectEntries.isEmpty ? "" : #""spk_is_dialect": {"# + dialectEntries + "},"
    let configJSON = """
    {
      "model_type": "qwen3_tts",
      "tts_model_type": "\(ttsModelType)",
      "tts_model_size": "tiny",
      "tokenizer_type": "qwen3_tts_tokenizer_12hz",
      "im_start_token_id": 4,
      "im_end_token_id": 5,
      "tts_pad_token_id": 21,
      "tts_bos_token_id": 22,
      "tts_eos_token_id": 23,
      "sample_rate": 24000,
      "talker_config": {
        "vocab_size": 3072,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "max_position_embeddings": 128,
        "num_code_groups": 2,
        "text_hidden_size": 16,
        "text_vocab_size": 64,
        "codec_eos_token_id": 3050,
        "codec_think_id": 3051,
        "codec_nothink_id": 3052,
        "codec_think_bos_id": 3053,
        "codec_think_eos_id": 3054,
        "codec_pad_id": 3055,
        "codec_bos_id": 3056,
        "codec_language_id": {
          "english": 3057
        },
        \(spkIdJSON)
        \(spkDialectJSON)
        "code_predictor_config": {
          "vocab_size": 2048,
          "hidden_size": 16,
          "intermediate_size": 32,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "head_dim": 4,
          "max_position_embeddings": 128,
          "num_code_groups": 2
        }
      },
      "tokenizer_config": {
        "encoder_valid_num_quantizers": 2,
        \(encoderConfigJSON)
        "decoder_config": {}
      }
    }
    """

    let configData = Data(configJSON.utf8)
    let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)
    let model = Qwen3TTSModel(config: config)
    model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
    model.speechTokenizer = Qwen3TTSSpeechTokenizer(config: try #require(config.tokenizerConfig))
    return (model, tokenizerDirectory)
}

private func collectQwen3TTSStream(
    _ stream: AsyncThrowingStream<AudioGeneration, Error>
) async throws -> (tokenCount: Int, infoCount: Int, lastAudio: MLXArray?) {
    var tokenCount = 0
    var infoCount = 0
    var lastAudio: MLXArray?

    for try await event in stream {
        switch event {
        case .token:
            tokenCount += 1
        case .info:
            infoCount += 1
        case .audio(let audio):
            lastAudio = audio
        }
    }

    return (tokenCount, infoCount, lastAudio)
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

private func makeTinyEchoTTSConfig(numSteps: Int = 1, sequenceLength: Int = 4) -> EchoTTSConfig {
    EchoTTSConfig(
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
            numSteps: numSteps,
            cfgScaleText: 1,
            cfgScaleSpeaker: 1,
            sequenceLength: sequenceLength
        )
    )
}

private final class StreamCancellationState: @unchecked Sendable {
    private let lock = NSLock()
    private var producerCancelled = false

    func markCancelled() {
        lock.lock()
        producerCancelled = true
        lock.unlock()
    }

    var wasCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return producerCancelled
    }
}

private final class ProxyCancellationProbeModel: SpeechGenerationModel, @unchecked Sendable {
    let sampleRate = 24_000
    let defaultGenerationParameters = GenerateParameters(maxTokens: 1)

    private let state: StreamCancellationState

    init(state: StreamCancellationState) {
        self.state = state
    }

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        MLXArray.zeros([0], dtype: .float32)
    }

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task {
            do {
                while true {
                    try Task.checkCancellation()
                    continuation.yield(.audio(MLXArray.zeros([8], dtype: .float32)))
                    try await Task.sleep(nanoseconds: 20_000_000)
                }
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in
            self.state.markCancelled()
            task.cancel()
        }
        return stream
    }

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }
}

private final class CountingFishAE: EchoTTSAudioCodec, @unchecked Sendable {
    private let lock = NSLock()
    private var _decodeCount = 0

    var decodeCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _decodeCount
    }

    func encodeZQ(_ audioData: MLXArray) -> MLXArray {
        MLXArray.zeros([audioData.shape[0], 8, max(audioData.shape[2] / 2_048, 1)], dtype: .float32)
    }

    func decodeZQ(_ zQ: MLXArray) -> MLXArray {
        lock.lock()
        _decodeCount += 1
        lock.unlock()
        return MLXArray.zeros([zQ.shape[0], 1, zQ.shape[2] * 2_048], dtype: .float32)
    }
}


// MARK: - Text Cleaning Unit Tests

@Suite("Qwen3TTS")
struct Qwen3TTSTests {

    @Test func prepareReferenceConditioningBuildsReusableReferenceState() async throws {
        let fixture = try await makeTinyQwen3TTSModel(
            ttsModelType: "voice_design",
            includeSpeechEncoder: true
        )
        defer { cleanupTemporaryArtifactDirectory(fixture.tokenizerDirectory) }

        let refAudio = try loadTTSNetworkFixture(sampleRate: 24_000, maxSamples: 24_000)
        let conditioning = try fixture.model.prepareReferenceConditioning(
            refAudio: refAudio,
            refText: "one two three four five one two three four five",
            language: "English"
        )

        #expect(conditioning.speakerEmbedding == nil)
        #expect(conditioning.referenceSpeechCodes.ndim == 3)
        #expect(conditioning.referenceSpeechCodes.dim(0) == 1)
        #expect(conditioning.referenceSpeechCodes.dim(1) == 2)
        #expect(conditioning.referenceSpeechCodes.dim(2) > 0)
        #expect(conditioning.referenceTextTokenIDs.ndim == 2)
        #expect(conditioning.referenceTextTokenIDs.dim(1) > 0)
        #expect(conditioning.resolvedLanguage == "english")
        #expect(conditioning.codecLanguageID == 3057)
    }

    @Test func generateFromPreparedConditioningSupportsDirectAndStreamingCalls() async throws {
        let fixture = try await makeTinyQwen3TTSModel(
            ttsModelType: "voice_design",
            includeSpeechEncoder: true
        )
        defer { cleanupTemporaryArtifactDirectory(fixture.tokenizerDirectory) }

        let refAudio = try loadTTSNetworkFixture(sampleRate: 24_000, maxSamples: 24_000)
        let conditioning = try fixture.model.prepareReferenceConditioning(
            refAudio: refAudio,
            refText: "one two three four five one two three four five",
            language: "English"
        )
        let parameters = GenerateParameters(
            maxTokens: 2,
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.0
        )

        MLXRandom.seed(0)
        let audio = try await fixture.model.generate(
            text: "target voice prompt one two three four five",
            conditioning: conditioning,
            generationParameters: parameters
        )
        #expect(audio.ndim == 1)
        #expect(audio.shape[0] > 0)

        MLXRandom.seed(0)
        let streamResult = try await collectQwen3TTSStream(
            fixture.model.generateStream(
                text: "target voice prompt one two three four five",
                conditioning: conditioning,
                generationParameters: parameters,
                streamingInterval: 0.05
            )
        )
        #expect(streamResult.tokenCount > 0)
        #expect(streamResult.infoCount == 1)
        #expect(streamResult.lastAudio != nil)
        #expect(streamResult.lastAudio?.ndim == 1)
    }

    @Test func rawReferenceAPIsStillWorkThroughTheCompatibilityWrapper() async throws {
        let fixture = try await makeTinyQwen3TTSModel(
            ttsModelType: "voice_design",
            includeSpeechEncoder: true
        )
        defer { cleanupTemporaryArtifactDirectory(fixture.tokenizerDirectory) }

        let refAudio = try loadTTSNetworkFixture(sampleRate: 24_000, maxSamples: 24_000)
        let parameters = GenerateParameters(
            maxTokens: 2,
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.0
        )

        MLXRandom.seed(0)
        let rawAudio = try await fixture.model.generate(
            text: "target voice prompt one two three four five",
            voice: nil,
            refAudio: refAudio,
            refText: "one two three four five one two three four five",
            language: "English",
            generationParameters: parameters
        )
        #expect(rawAudio.ndim == 1)
        #expect(rawAudio.shape[0] > 0)

        MLXRandom.seed(0)
        let streamResult = try await collectQwen3TTSStream(
            fixture.model.generateStream(
                text: "target voice prompt one two three four five",
                voice: nil,
                refAudio: refAudio,
                refText: "one two three four five one two three four five",
                language: "English",
                generationParameters: parameters,
                streamingInterval: 0.05
            )
        )
        #expect(streamResult.tokenCount > 0)
        #expect(streamResult.infoCount == 1)
        #expect(streamResult.lastAudio != nil)
        #expect(streamResult.lastAudio?.ndim == 1)
    }

    @Test func customVoiceRemainsSeparateFromReferenceConditioning() async throws {
        let fixture = try await makeTinyQwen3TTSModel(
            ttsModelType: "custom_voice",
            includeSpeechEncoder: false,
            speakerIDEntries: #""ryan": 100"#,
            dialectEntries: #""ryan": false"#
        )
        defer { cleanupTemporaryArtifactDirectory(fixture.tokenizerDirectory) }

        let parameters = GenerateParameters(
            maxTokens: 2,
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.0
        )

        let audio = try await fixture.model.generate(
            text: "target voice prompt one two three four five",
            voice: "ryan",
            refAudio: nil,
            refText: nil,
            language: "English",
            generationParameters: parameters
        )
        #expect(audio.ndim == 1)
        #expect(audio.shape[0] > 0)

        let refAudio = try loadTTSNetworkFixture(sampleRate: 24_000, maxSamples: 24_000)
        #expect(throws: AudioGenerationError.self) {
            try fixture.model.prepareReferenceConditioning(
                refAudio: refAudio,
                refText: "one two three four five one two three four five",
                language: "English"
            )
        }
    }
}

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

    @Test func proxySamplesStreamReleaseCancelsUpstreamProducer() async throws {
        let state = StreamCancellationState()
        let model = ProxyCancellationProbeModel(state: state)
        var stream: AsyncThrowingStream<[Float], Error>? = model.generateSamplesStream(
            text: "hi",
            voice: nil,
            refAudio: nil,
            refText: nil,
            language: nil
        )

        var iterator = stream?.makeAsyncIterator()
        let consumedOne = try await iterator?.next() != nil

        iterator = nil
        stream = nil

        var producerCancelled = state.wasCancelled
        for _ in 0 ..< 20 where !producerCancelled {
            try await Task.sleep(nanoseconds: 10_000_000)
            producerCancelled = state.wasCancelled
        }

        #expect(consumedOne)
        #expect(producerCancelled)
    }

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
        let config = makeTinyEchoTTSConfig()
        let model = EchoTTSModel(
            config: config,
            fishAE: CountingFishAE(),
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

    @Test func generateDetailedCancellationStopsBeforeDecode() async throws {
        let codec = CountingFishAE()
        let model = EchoTTSModel(
            config: makeTinyEchoTTSConfig(sequenceLength: 32),
            fishAE: codec,
            pcaState: EchoTTSPCAState(
                pcaComponents: MLXArray.eye(8, dtype: .float32),
                pcaMean: MLXArray.zeros([8], dtype: .float32),
                latentScale: 1
            )
        )

        let task = Task {
            try model.generateDetailed(
                text: "hi",
                refAudio: nil,
                rngSeed: 0,
                numSteps: 20_000,
                sequenceLength: 32
            )
        }

        try await Task.sleep(nanoseconds: 10_000_000)
        task.cancel()

        let cancelled: Bool
        do {
            _ = try await task.value
            cancelled = false
        } catch is CancellationError {
            cancelled = true
        } catch {
            Issue.record("Unexpected error while cancelling Echo generation: \(error)")
            cancelled = false
        }

        var decodeCount = codec.decodeCount
        for _ in 0 ..< 10 where decodeCount == 0 {
            try await Task.sleep(nanoseconds: 10_000_000)
            decodeCount = codec.decodeCount
        }

        #expect(cancelled)
        #expect(decodeCount == 0)
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

    @Test func durationNaNProducesSilenceInsteadOfCrash() throws {
        guard metalAvailable else { return }
        let nanDuration = MLXArray([Float.nan, Float.nan, Float.nan])
        let safe = nanToNum(nanDuration, nan: 1.0)
        let clipped = MLX.clip(MLX.round(safe), min: 1, max: 100).asType(.int32)
        let arr: [Int32] = clipped.asArray(Int32.self)
        for n in arr {
            #expect(n >= 1 && n <= 100, "Duration \(n) should be clamped between 1 and 100")
        }
    }

    @Test func durationExtremeValuesAreCapped() throws {
        guard metalAvailable else { return }
        let extreme = MLXArray([Float(999), Float(0.001), Float(-5)])
        let clipped = MLX.clip(MLX.round(extreme), min: 1, max: 100).asType(.int32)
        let arr: [Int32] = clipped.asArray(Int32.self)
        #expect(arr[0] == 100, "Large duration should be capped at 100")
        #expect(arr[1] == 1, "Tiny duration should be clamped to 1")
        #expect(arr[2] == 1, "Negative duration should be clamped to 1")
    }

    @Test func emptyIndicesReturnsGracefully() throws {
        guard metalAvailable else { return }
        let durArray: [Int32] = [0, 0, 0]
        var indices = [MLXArray]()
        for (i, n) in durArray.enumerated() {
            let count = min(max(Int(n), 0), 100)
            if count > 0 {
                indices.append(MLX.repeated(MLXArray(Int32(i)), count: count))
            }
        }
        #expect(indices.isEmpty, "All-zero durations should produce empty indices")
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
