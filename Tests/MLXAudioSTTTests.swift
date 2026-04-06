//  Run the STT suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/GLMASRModuleSetupTests \
//      -only-testing:MLXAudioTests/Qwen3ASRModuleSetupTests \
//      -only-testing:MLXAudioTests/ForceAlignProcessorTests \
//      -only-testing:MLXAudioTests/ForcedAlignResultTests \
//      -only-testing:MLXAudioTests/Qwen3ASRHelperTests \
//      -only-testing:MLXAudioTests/SplitAudioIntoChunksTests \
//      -only-testing:MLXAudioTests/FireRedASR2Tests \
//      -only-testing:MLXAudioTests/FireRedASR2NetworkTests \
//      -only-testing:MLXAudioTests/SenseVoiceTests \
//      -only-testing:MLXAudioTests/SenseVoiceNetworkTests \
//      -only-testing:MLXAudioTests/ParakeetSTTTests \
//      -only-testing:MLXAudioTests/VoxtralRealtimeSTTTests \
//      -only-testing:MLXAudioTests/CohereTranscribeModuleSetupTests \
//      -only-testing:MLXAudioTests/CohereTranscribeSTTTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/GLMASRModuleSetupTests'
//    -only-testing:'MLXAudioTests/Qwen3ASRModuleSetupTests'
//    -only-testing:'MLXAudioTests/ForceAlignProcessorTests'
//    -only-testing:'MLXAudioTests/ForcedAlignResultTests'
//    -only-testing:'MLXAudioTests/Qwen3ASRHelperTests'
//    -only-testing:'MLXAudioTests/SplitAudioIntoChunksTests'
//    -only-testing:'MLXAudioTests/FireRedASR2Tests'
//    -only-testing:'MLXAudioTests/FireRedASR2NetworkTests'
//    -only-testing:'MLXAudioTests/SenseVoiceTests'
//    -only-testing:'MLXAudioTests/SenseVoiceNetworkTests'
//    -only-testing:'MLXAudioTests/ParakeetSTTTests'
//    -only-testing:'MLXAudioTests/VoxtralRealtimeSTTTests'
//    -only-testing:'MLXAudioTests/CohereTranscribeModuleSetupTests'
//    -only-testing:'MLXAudioTests/CohereTranscribeSTTTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/GLMASRModuleSetupTests/whisperConfigDefaults()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioSTT

private func loadSTTNetworkFixture(sampleRate: Int, maxSamples: Int? = nil) throws -> MLXArray {
    let audioURL = Bundle.module.url(
        forResource: "intention",
        withExtension: "wav",
        subdirectory: "media"
    )!
    let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: sampleRate)
    if let maxSamples {
        let sampleCount = min(audio.shape[0], maxSamples)
        return audio[0..<sampleCount]
    }
    return audio
}

private func encodeProtobufVarint(_ value: UInt64) -> [UInt8] {
    var value = value
    var bytes: [UInt8] = []
    repeat {
        var byte = UInt8(value & 0x7f)
        value >>= 7
        if value != 0 {
            byte |= 0x80
        }
        bytes.append(byte)
    } while value != 0
    return bytes
}

private func makeSentencePieceModelData(_ pieces: [(token: String, score: Float, type: Int)]) -> Data {
    var data = Data()
    for piece in pieces {
        var pieceData = Data()
        let tokenData = Data(piece.token.utf8)
        pieceData.append(contentsOf: encodeProtobufVarint(UInt64((1 << 3) | 2)))
        pieceData.append(contentsOf: encodeProtobufVarint(UInt64(tokenData.count)))
        pieceData.append(tokenData)

        let scoreBits = piece.score.bitPattern.littleEndian
        pieceData.append(contentsOf: encodeProtobufVarint(UInt64((2 << 3) | 5)))
        pieceData.append(UInt8(truncatingIfNeeded: scoreBits))
        pieceData.append(UInt8(truncatingIfNeeded: scoreBits >> 8))
        pieceData.append(UInt8(truncatingIfNeeded: scoreBits >> 16))
        pieceData.append(UInt8(truncatingIfNeeded: scoreBits >> 24))

        pieceData.append(contentsOf: encodeProtobufVarint(UInt64(3 << 3)))
        pieceData.append(contentsOf: encodeProtobufVarint(UInt64(piece.type)))

        data.append(contentsOf: encodeProtobufVarint(UInt64((1 << 3) | 2)))
        data.append(contentsOf: encodeProtobufVarint(UInt64(pieceData.count)))
        data.append(pieceData)
    }
    return data
}

private func makeCohereFixtureConfigJSON(vocabSize: Int = 32, hiddenSize: Int = 16, numLayers: Int = 0, maxSequenceLength: Int = 32) -> String {
    """
    {
      "model_type": "cohere_asr",
      "vocab_size": \(vocabSize),
      "sample_rate": 16000,
      "max_audio_clip_s": 35,
      "encoder": {
        "d_model": \(hiddenSize),
        "ff_expansion_factor": 2,
        "n_heads": 2,
        "conv_kernel_size": 9,
        "n_layers": \(numLayers),
        "pos_emb_max_len": 64,
        "subsampling_conv_channels": 8,
        "subsampling_factor": 8,
        "feat_in": 128
      },
      "transf_decoder": {
        "config_dict": {
          "hidden_size": \(hiddenSize),
          "inner_size": \(hiddenSize * 2),
          "num_attention_heads": 2,
          "num_layers": \(numLayers),
          "max_sequence_length": \(maxSequenceLength)
        }
      }
    }
    """
}


struct GLMASRModuleSetupTests {

    // MARK: - Configuration Tests

    @Test func whisperConfigDefaults() {
        let config = WhisperConfig()

        #expect(config.modelType == "whisper")
        #expect(config.activationFunction == "gelu")
        #expect(config.dModel == 1280)
        #expect(config.encoderAttentionHeads == 20)
        #expect(config.encoderFfnDim == 5120)
        #expect(config.encoderLayers == 32)
        #expect(config.numMelBins == 128)
        #expect(config.maxSourcePositions == 1500)
        #expect(config.ropeTraditional)
    }

    @Test func whisperConfigCustom() {
        let config = WhisperConfig(
            dModel: 512,
            encoderAttentionHeads: 8,
            encoderLayers: 6,
            numMelBins: 80
        )

        #expect(config.dModel == 512)
        #expect(config.encoderAttentionHeads == 8)
        #expect(config.encoderLayers == 6)
        #expect(config.numMelBins == 80)
    }

    @Test func llamaConfigDefaults() {
        let config = LlamaConfig()

        #expect(config.modelType == "llama")
        #expect(config.vocabSize == 59264)
        #expect(config.hiddenSize == 2048)
        #expect(config.intermediateSize == 6144)
        #expect(config.numHiddenLayers == 28)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 4)
        #expect(config.hiddenAct == "silu")
        #expect(config.eosTokenId == [59246, 59253, 59255])
    }

    @Test func llamaConfigCustom() {
        let config = LlamaConfig(
            vocabSize: 32000,
            hiddenSize: 1024,
            numHiddenLayers: 12
        )

        #expect(config.vocabSize == 32000)
        #expect(config.hiddenSize == 1024)
        #expect(config.numHiddenLayers == 12)
    }

    @Test func glmASRModelConfigDefaults() {
        let config = GLMASRModelConfig()

        #expect(config.modelType == "glmasr")
        #expect(config.adapterType == "mlp")
        #expect(config.mergeFactor == 4)
        #expect(config.useRope)
        #expect(config.maxWhisperLength == 1500)
    }

    @Test func glmASRModelConfigWithNestedConfigs() {
        let whisperConfig = WhisperConfig(dModel: 512, encoderLayers: 6)
        let llamaConfig = LlamaConfig(hiddenSize: 1024, numHiddenLayers: 12)

        let config = GLMASRModelConfig(
            whisperConfig: whisperConfig,
            lmConfig: llamaConfig,
            mergeFactor: 2
        )

        #expect(config.whisperConfig.dModel == 512)
        #expect(config.whisperConfig.encoderLayers == 6)
        #expect(config.lmConfig.hiddenSize == 1024)
        #expect(config.lmConfig.numHiddenLayers == 12)
        #expect(config.mergeFactor == 2)
    }

    // MARK: - Layer Tests

    @Test func whisperAttentionShape() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderLayers: 2
        )

        let attention = WhisperAttention(config: config, useRope: false)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = attention(hiddenStates)

        #expect(output.shape == [batchSize, seqLen, config.dModel])
    }

    @Test func whisperAttentionWithRoPE() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderLayers: 2,
            ropeTraditional: true
        )

        let attention = WhisperAttention(config: config, useRope: true)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = attention(hiddenStates)

        #expect(output.shape == [batchSize, seqLen, config.dModel])
    }

    @Test func whisperEncoderLayerShape() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            encoderLayers: 1
        )

        let layer = WhisperEncoderLayer(config: config, useRope: false)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = layer(hiddenStates)

        #expect(output.shape == [batchSize, seqLen, config.dModel])
    }

    @Test func whisperEncoderShape() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            encoderLayers: 2,
            numMelBins: 80,
            maxSourcePositions: 100
        )

        let encoder = WhisperEncoder(config: config, useRope: false)

        let batchSize = 2
        let seqLen = 100
        let inputFeatures = MLXArray.ones([batchSize, seqLen, config.numMelBins])

        let output = encoder(inputFeatures)

        // After conv2 with stride 2, sequence length is halved
        let expectedSeqLen = seqLen / 2
        #expect(output.shape[0] == batchSize)
        #expect(output.shape[1] == expectedSeqLen)
        #expect(output.shape[2] == config.dModel)
    }

    @Test func adaptingMLPShape() {
        let inputDim = 512
        let intermediateDim = 1024
        let outputDim = 256

        let mlp = AdaptingMLP(inputDim: inputDim, intermediateDim: intermediateDim, outputDim: outputDim)

        let batchSize = 2
        let seqLen = 10
        let input = MLXArray.ones([batchSize, seqLen, inputDim])

        let output = mlp(input)

        #expect(output.shape == [batchSize, seqLen, outputDim])
    }

    @Test func audioEncoderShape() {
        let whisperConfig = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            encoderLayers: 2,
            numMelBins: 80,
            maxSourcePositions: 100
        )

        let llamaConfig = LlamaConfig(
            hiddenSize: 512,
            numHiddenLayers: 2
        )

        let config = GLMASRModelConfig(
            whisperConfig: whisperConfig,
            lmConfig: llamaConfig,
            mergeFactor: 4,
            maxWhisperLength: 100
        )

        let audioEncoder = AudioEncoder(config: config)

        let batchSize = 2
        let seqLen = 100
        let inputFeatures = MLXArray.ones([batchSize, seqLen, whisperConfig.numMelBins])

        let (output, audioLen) = audioEncoder(inputFeatures)

        #expect(output.shape[0] == batchSize)
        #expect(output.shape[2] == llamaConfig.hiddenSize)
        #expect(audioLen > 0)
    }

    @Test func audioEncoderBoaEoaTokens() {
        let whisperConfig = WhisperConfig(dModel: 256, encoderAttentionHeads: 4, encoderLayers: 1)
        let llamaConfig = LlamaConfig(hiddenSize: 512)
        let config = GLMASRModelConfig(whisperConfig: whisperConfig, lmConfig: llamaConfig)

        let audioEncoder = AudioEncoder(config: config)

        let (boa, eoa) = audioEncoder.getBoaEoaTokens()

        #expect(boa.shape == [1, llamaConfig.hiddenSize])
        #expect(eoa.shape == [1, llamaConfig.hiddenSize])
    }

    // MARK: - STTOutput Tests

    @Test func sttOutputCreation() {
        let output = STTOutput(
            text: "Hello world",
            promptTokens: 100,
            generationTokens: 50,
            totalTokens: 150,
            promptTps: 100.0,
            generationTps: 50.0,
            totalTime: 1.5
        )

        #expect(output.text == "Hello world")
        #expect(output.promptTokens == 100)
        #expect(output.generationTokens == 50)
        #expect(output.totalTokens == 150)
        #expect(output.promptTps == 100.0)
        #expect(output.generationTps == 50.0)
        #expect(output.totalTime == 1.5)
    }

    @Test func sttOutputDefaults() {
        let output = STTOutput(text: "Test")

        #expect(output.text == "Test")
        #expect(output.segments == nil)
        #expect(output.language == nil)
        #expect(output.promptTokens == 0)
        #expect(output.generationTokens == 0)
        #expect(output.totalTokens == 0)
    }

    @Test func sttOutputDescription() {
        let output = STTOutput(
            text: "Test transcription",
            language: "en",
            promptTokens: 50,
            generationTokens: 25,
            totalTokens: 75,
            totalTime: 0.5
        )

        let description = output.description

        #expect(description.contains("Test transcription"))
        #expect(description.contains("en"))
        #expect(description.contains("50"))
        #expect(description.contains("25"))
        #expect(description.contains("75"))
    }

    // MARK: - Config Decoding Tests

    @Test func whisperConfigDecoding() throws {
        let json = """
        {
            "model_type": "whisper",
            "d_model": 512,
            "encoder_attention_heads": 8,
            "encoder_layers": 6,
            "num_mel_bins": 80
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(WhisperConfig.self, from: data)

        #expect(config.modelType == "whisper")
        #expect(config.dModel == 512)
        #expect(config.encoderAttentionHeads == 8)
        #expect(config.encoderLayers == 6)
        #expect(config.numMelBins == 80)
    }

    @Test func llamaConfigDecoding() throws {
        let json = """
        {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "eos_token_id": [1, 2, 3]
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(LlamaConfig.self, from: data)

        #expect(config.modelType == "llama")
        #expect(config.vocabSize == 32000)
        #expect(config.hiddenSize == 1024)
        #expect(config.numHiddenLayers == 12)
        #expect(config.eosTokenId == [1, 2, 3])
    }

    @Test func glmASRModelConfigDecoding() throws {
        let json = """
        {
            "model_type": "glmasr",
            "adapter_type": "mlp",
            "merge_factor": 2,
            "use_rope": true,
            "whisper_config": {
                "d_model": 512,
                "encoder_layers": 6
            },
            "lm_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12
            }
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(GLMASRModelConfig.self, from: data)

        #expect(config.modelType == "glmasr")
        #expect(config.adapterType == "mlp")
        #expect(config.mergeFactor == 2)
        #expect(config.useRope)
        #expect(config.whisperConfig.dModel == 512)
        #expect(config.whisperConfig.encoderLayers == 6)
        #expect(config.lmConfig.hiddenSize == 1024)
        #expect(config.lmConfig.numHiddenLayers == 12)
    }

    // MARK: - AnyCodable Tests

    @Test func anyCodableWithInt() throws {
        let json = """
        {"value": 42}
        """

        struct Container: Codable {
            let value: AnyCodable
        }

        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        #expect(container.value.value as? Int == 42)
    }

    @Test func anyCodableWithString() throws {
        let json = """
        {"value": "hello"}
        """

        struct Container: Codable {
            let value: AnyCodable
        }

        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        #expect(container.value.value as? String == "hello")
    }

    @Test func anyCodableWithArray() throws {
        let json = """
        {"value": [1, 2, 3]}
        """

        struct Container: Codable {
            let value: AnyCodable
        }

        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        let array = container.value.value as? [Any]
        #expect(array?.count == 3)
    }
}


// MARK: - Qwen3 ASR Module Setup Tests

struct Qwen3ASRModuleSetupTests {

    // MARK: - Audio Encoder Config Tests

    @Test func qwen3AudioEncoderConfigDefaults() {
        let config = Qwen3AudioEncoderConfig()

        #expect(config.numMelBins == 128)
        #expect(config.encoderLayers == 24)
        #expect(config.encoderAttentionHeads == 16)
        #expect(config.encoderFfnDim == 4096)
        #expect(config.dModel == 1024)
        #expect(config.dropout == 0.0)
        #expect(config.activationFunction == "gelu")
        #expect(config.maxSourcePositions == 1500)
        #expect(config.nWindow == 50)
        #expect(config.outputDim == 2048)
        #expect(config.nWindowInfer == 800)
        #expect(config.convChunksize == 500)
        #expect(config.downsampleHiddenSize == 480)
    }

    @Test func qwen3AudioEncoderConfigCustom() {
        let config = Qwen3AudioEncoderConfig(
            numMelBins: 80,
            encoderLayers: 12,
            encoderAttentionHeads: 8,
            dModel: 512,
            outputDim: 1024
        )

        #expect(config.numMelBins == 80)
        #expect(config.encoderLayers == 12)
        #expect(config.encoderAttentionHeads == 8)
        #expect(config.dModel == 512)
        #expect(config.outputDim == 1024)
    }

    @Test func qwen3AudioEncoderConfigDecoding() throws {
        let json = """
        {
            "num_mel_bins": 80,
            "encoder_layers": 12,
            "encoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "d_model": 512,
            "n_window": 25,
            "output_dim": 1024,
            "n_window_infer": 400,
            "conv_chunksize": 250,
            "downsample_hidden_size": 240
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3AudioEncoderConfig.self, from: data)

        #expect(config.numMelBins == 80)
        #expect(config.encoderLayers == 12)
        #expect(config.encoderAttentionHeads == 8)
        #expect(config.encoderFfnDim == 2048)
        #expect(config.dModel == 512)
        #expect(config.nWindow == 25)
        #expect(config.outputDim == 1024)
        #expect(config.nWindowInfer == 400)
        #expect(config.convChunksize == 250)
        #expect(config.downsampleHiddenSize == 240)
    }

    @Test func qwen3AudioEncoderConfigDecodingDefaults() throws {
        // Empty JSON should use all defaults
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3AudioEncoderConfig.self, from: data)

        #expect(config.numMelBins == 128)
        #expect(config.encoderLayers == 24)
        #expect(config.dModel == 1024)
    }

    // MARK: - Text Config Tests

    @Test func qwen3TextConfigDefaults() {
        let config = Qwen3TextConfig()

        #expect(config.modelType == "qwen3")
        #expect(config.vocabSize == 151936)
        #expect(config.hiddenSize == 1024)
        #expect(config.intermediateSize == 3072)
        #expect(config.numHiddenLayers == 28)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.hiddenAct == "silu")
        #expect(config.rmsNormEps == 1e-6)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.ropeTheta == 1000000.0)
        #expect(config.attentionBias == false)
    }

    @Test func qwen3TextConfigDecoding() throws {
        let json = """
        {
            "model_type": "qwen3",
            "vocab_size": 152064,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": false,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TextConfig.self, from: data)

        #expect(config.vocabSize == 152064)
        #expect(config.hiddenSize == 1024)
        #expect(config.numHiddenLayers == 28)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.tieWordEmbeddings == false)
    }

    // MARK: - Qwen3ASRConfig Tests

    @Test func qwen3ASRConfigDefaults() {
        let config = Qwen3ASRConfig()

        #expect(config.modelType == "qwen3_asr")
        #expect(config.audioTokenId == 151676)
        #expect(config.audioStartTokenId == 151669)
        #expect(config.audioEndTokenId == 151670)
        #expect(config.supportLanguages.isEmpty)
        #expect(config.isForcedAligner == false)
        #expect(config.timestampTokenId == nil)
        #expect(config.timestampSegmentTime == nil)
        #expect(config.classifyNum == nil)
    }

    @Test func qwen3ASRConfigForcedAlignerDetection() {
        // Via model_type
        let config1 = Qwen3ASRConfig(modelType: "qwen3_forced_aligner")
        #expect(config1.isForcedAligner == true)

        // Via classify_num
        let config2 = Qwen3ASRConfig(classifyNum: 5000)
        #expect(config2.isForcedAligner == true)

        // Regular ASR
        let config3 = Qwen3ASRConfig()
        #expect(config3.isForcedAligner == false)
    }

    @Test func qwen3ASRConfigFlatDecoding() throws {
        let json = """
        {
            "model_type": "qwen3_asr",
            "audio_config": {
                "num_mel_bins": 128,
                "encoder_layers": 18,
                "d_model": 896
            },
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 1024,
                "tie_word_embeddings": true
            },
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "support_languages": ["English", "Chinese"]
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: data)

        #expect(config.modelType == "qwen3_asr")
        #expect(config.audioConfig.numMelBins == 128)
        #expect(config.audioConfig.encoderLayers == 18)
        #expect(config.audioConfig.dModel == 896)
        #expect(config.textConfig.vocabSize == 151936)
        #expect(config.textConfig.hiddenSize == 1024)
        #expect(config.textConfig.tieWordEmbeddings == true)
        #expect(config.audioTokenId == 151676)
        #expect(config.supportLanguages == ["English", "Chinese"])
        #expect(config.isForcedAligner == false)
    }

    @Test func qwen3ASRConfigThinkerDecoding() throws {
        // HuggingFace nested thinker_config format
        let json = """
        {
            "model_type": "qwen3_asr",
            "thinker_config": {
                "model_type": "qwen3_asr",
                "audio_config": {
                    "num_mel_bins": 128,
                    "encoder_layers": 18,
                    "encoder_attention_heads": 14,
                    "d_model": 896,
                    "output_dim": 1024
                },
                "text_config": {
                    "vocab_size": 151936,
                    "hidden_size": 1024,
                    "num_hidden_layers": 28,
                    "tie_word_embeddings": true
                },
                "audio_token_id": 151676,
                "audio_start_token_id": 151669,
                "audio_end_token_id": 151670
            },
            "support_languages": ["English", "Chinese", "Japanese"]
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: data)

        #expect(config.audioConfig.encoderLayers == 18)
        #expect(config.audioConfig.encoderAttentionHeads == 14)
        #expect(config.audioConfig.dModel == 896)
        #expect(config.audioConfig.outputDim == 1024)
        #expect(config.textConfig.vocabSize == 151936)
        #expect(config.textConfig.hiddenSize == 1024)
        #expect(config.textConfig.tieWordEmbeddings == true)
        #expect(config.audioTokenId == 151676)
        #expect(config.supportLanguages == ["English", "Chinese", "Japanese"])
    }

    @Test func qwen3ASRConfigForcedAlignerThinkerDecoding() throws {
        // HuggingFace ForcedAligner config with thinker_config
        let json = """
        {
            "model_type": "qwen3_forced_aligner",
            "timestamp_token_id": 151705,
            "timestamp_segment_time": 80.0,
            "thinker_config": {
                "model_type": "qwen3_forced_aligner",
                "audio_config": {
                    "num_mel_bins": 128,
                    "encoder_layers": 24,
                    "d_model": 1024
                },
                "text_config": {
                    "vocab_size": 152064,
                    "hidden_size": 1024,
                    "tie_word_embeddings": false
                },
                "audio_token_id": 151676,
                "classify_num": 5000,
                "timestamp_token_id": 151705,
                "timestamp_segment_time": 80.0
            }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: data)

        #expect(config.isForcedAligner == true)
        #expect(config.modelType == "qwen3_forced_aligner")
        #expect(config.classifyNum == 5000)
        #expect(config.timestampTokenId == 151705)
        #expect(config.timestampSegmentTime == 80.0)
        #expect(config.textConfig.tieWordEmbeddings == false)
        #expect(config.textConfig.vocabSize == 152064)
        #expect(config.audioConfig.encoderLayers == 24)
        #expect(config.audioConfig.dModel == 1024)
    }

    // MARK: - StringAnyCodable Tests

    @Test func stringAnyCodableInt() throws {
        let json = """
        {"value": 42}
        """
        struct Container: Codable { let value: StringAnyCodable }
        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        if case .int(let v) = container.value.value {
            #expect(v == 42)
        } else {
            #expect(Bool(false), "Expected .int")
        }
    }

    @Test func stringAnyCodableString() throws {
        let json = """
        {"value": "hello"}
        """
        struct Container: Codable { let value: StringAnyCodable }
        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        if case .string(let v) = container.value.value {
            #expect(v == "hello")
        } else {
            #expect(Bool(false), "Expected .string")
        }
    }

    @Test func stringAnyCodableDouble() throws {
        let json = """
        {"value": 3.14}
        """
        struct Container: Codable { let value: StringAnyCodable }
        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        if case .double(let v) = container.value.value {
            #expect(abs(v - 3.14) < 0.001)
        } else {
            #expect(Bool(false), "Expected .double")
        }
    }

    @Test func stringAnyCodableBool() throws {
        let json = """
        {"value": true}
        """
        struct Container: Codable { let value: StringAnyCodable }
        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        if case .bool(let v) = container.value.value {
            #expect(v == true)
        } else {
            #expect(Bool(false), "Expected .bool")
        }
    }

    @Test func stringAnyCodableRoundTrip() throws {
        let json = """
        {"value": "test_string"}
        """
        struct Container: Codable { let value: StringAnyCodable }
        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        // Re-encode
        let encoded = try JSONEncoder().encode(container)
        let decoded = try JSONDecoder().decode(Container.self, from: encoded)

        if case .string(let v) = decoded.value.value {
            #expect(v == "test_string")
        } else {
            #expect(Bool(false), "Expected .string after round trip")
        }
    }

    // MARK: - Audio Layer Shape Tests

    @Test func qwen3SinusoidalPEShape() {
        let length = 100
        let channels = 64
        let pe = Qwen3ASRSinusoidalPE(length: length, channels: channels)

        let output = pe(50)
        #expect(output.shape == [50, channels])

        let outputFull = pe(length)
        #expect(outputFull.shape == [length, channels])
    }

    @Test func qwen3AudioAttentionShape() {
        let config = Qwen3AudioEncoderConfig(
            encoderAttentionHeads: 4,
            dModel: 256
        )

        let attention = Qwen3ASRAttention(config)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = attention(hiddenStates)

        #expect(output.shape == [batchSize, seqLen, config.dModel])
    }

    @Test func qwen3AudioAttentionWithMask() {
        let config = Qwen3AudioEncoderConfig(
            encoderAttentionHeads: 4,
            dModel: 256
        )

        let attention = Qwen3ASRAttention(config)

        let batchSize = 1
        let seqLen = 8
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        // Create a simple mask
        let mask = MLX.zeros([seqLen, seqLen])
        let output = attention(hiddenStates, mask: mask)

        #expect(output.shape == [batchSize, seqLen, config.dModel])
    }

    @Test func qwen3AudioEncoderLayerShape() {
        let config = Qwen3AudioEncoderConfig(
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            dModel: 256
        )

        let layer = Qwen3ASRAudioEncoderLayer(config)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = layer(hiddenStates)

        #expect(output.shape == [batchSize, seqLen, config.dModel])
    }

    // MARK: - Text Layer Shape Tests

    @Test func qwen3TextMLPShape() {
        let config = Qwen3TextConfig(
            hiddenSize: 256,
            intermediateSize: 512
        )

        let mlp = Qwen3ASRTextMLP(config)

        let batchSize = 2
        let seqLen = 10
        let input = MLXArray.ones([batchSize, seqLen, config.hiddenSize])

        let output = mlp(input)

        #expect(output.shape == [batchSize, seqLen, config.hiddenSize])
    }

    @Test func qwen3TextAttentionShape() {
        let config = Qwen3TextConfig(
            hiddenSize: 256,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64
        )

        let attention = Qwen3ASRTextAttention(config, layerIdx: 0)

        let batchSize = 1
        let seqLen = 8
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.hiddenSize])

        let output = attention(hiddenStates, mask: .none, cache: nil)

        #expect(output.shape == [batchSize, seqLen, config.hiddenSize])
    }

    @Test func qwen3TextDecoderLayerShape() {
        let config = Qwen3TextConfig(
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64
        )

        let layer = Qwen3ASRTextDecoderLayer(config, layerIdx: 0)

        let batchSize = 1
        let seqLen = 8
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.hiddenSize])

        let output = layer(hiddenStates, mask: .none, cache: nil)

        #expect(output.shape == [batchSize, seqLen, config.hiddenSize])
    }

    @Test func qwen3TextModelShape() {
        let config = Qwen3TextConfig(
            vocabSize: 1000,
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64
        )

        let textModel = Qwen3ASRTextModel(config)

        let batchSize = 1
        let seqLen = 8
        let inputIds = MLXArray.zeros([batchSize, seqLen]).asType(.int32)

        let output = textModel(inputIds: inputIds)

        #expect(output.shape == [batchSize, seqLen, config.hiddenSize])
    }

    @Test func qwen3TextModelWithEmbeddings() {
        let config = Qwen3TextConfig(
            vocabSize: 1000,
            hiddenSize: 256,
            intermediateSize: 512,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64
        )

        let textModel = Qwen3ASRTextModel(config)

        let batchSize = 1
        let seqLen = 8
        let embeddings = MLXArray.ones([batchSize, seqLen, config.hiddenSize])

        let output = textModel(inputsEmbeds: embeddings)

        #expect(output.shape == [batchSize, seqLen, config.hiddenSize])
    }

    // MARK: - Model Construction Tests

    @Test func qwen3ASRModelConstruction() {
        let config = Qwen3ASRConfig(
            audioConfig: Qwen3AudioEncoderConfig(
                encoderLayers: 1,
                encoderAttentionHeads: 4,
                encoderFfnDim: 512,
                dModel: 256,
                maxSourcePositions: 100,
                outputDim: 128
            ),
            textConfig: Qwen3TextConfig(
                vocabSize: 1000,
                hiddenSize: 128,
                intermediateSize: 256,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                numKeyValueHeads: 2,
                headDim: 32,
                tieWordEmbeddings: true
            )
        )

        let model = Qwen3ASRModel(config)

        // With tieWordEmbeddings=true, lmHead should be nil
        #expect(model.config.textConfig.tieWordEmbeddings == true)
    }

    @Test func qwen3ASRModelConstructionWithLmHead() {
        let config = Qwen3ASRConfig(
            audioConfig: Qwen3AudioEncoderConfig(
                encoderLayers: 1,
                encoderAttentionHeads: 4,
                encoderFfnDim: 512,
                dModel: 256,
                outputDim: 128
            ),
            textConfig: Qwen3TextConfig(
                vocabSize: 1000,
                hiddenSize: 128,
                intermediateSize: 256,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                numKeyValueHeads: 2,
                headDim: 32,
                tieWordEmbeddings: false
            )
        )

        let model = Qwen3ASRModel(config)

        // With tieWordEmbeddings=false, lmHead should exist
        #expect(model.config.textConfig.tieWordEmbeddings == false)
    }

    @Test func qwen3ASRDefaultGenerationParametersAllowAutoLanguage() {
        let model = Qwen3ASRModel(Qwen3ASRConfig())

        #expect(model.defaultGenerationParameters.language == nil)
    }

    @Test func qwen3ASRNormalizesLanguageAliases() {
        let model = Qwen3ASRModel(
            Qwen3ASRConfig(supportLanguages: ["Chinese", "English", "Japanese"])
        )

        #expect(model.normalizeLanguageName("en") == "English")
        #expect(model.normalizeLanguageName(" chinese ") == "Chinese")
        #expect(model.normalizeLanguageName("ja") == "Japanese")
    }

    @Test func qwen3ASRParsesDetectedChunkLanguage() {
        let model = Qwen3ASRModel(
            Qwen3ASRConfig(supportLanguages: ["Chinese", "English", "Japanese"])
        )

        let parsed = model.parseGeneratedChunk(
            "language chinese<asr_text>你好世界",
            forcedLanguage: nil
        )

        #expect(parsed.language == "Chinese")
        #expect(parsed.text == "你好世界")
    }

    @Test func qwen3ASRChunkParsingFallsBackToEnglishForBareText() {
        let model = Qwen3ASRModel(Qwen3ASRConfig())

        let parsed = model.parseGeneratedChunk("hello world", forcedLanguage: nil)

        #expect(parsed.language == "English")
        #expect(parsed.text == "hello world")
    }

    @Test func qwen3ASRMergeLanguagesDeduplicatesInOrder() {
        let merged = Qwen3ASRModel.mergeLanguages(["Chinese", "", "English", "Chinese", nil])

        #expect(merged == "Chinese,English")
    }

    @Test func qwen3ForcedAlignerModelConstruction() {
        let config = Qwen3ASRConfig(
            audioConfig: Qwen3AudioEncoderConfig(
                encoderLayers: 1,
                encoderAttentionHeads: 4,
                encoderFfnDim: 512,
                dModel: 256,
                outputDim: 128
            ),
            textConfig: Qwen3TextConfig(
                vocabSize: 1000,
                hiddenSize: 128,
                intermediateSize: 256,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                numKeyValueHeads: 2,
                headDim: 32,
                tieWordEmbeddings: false
            ),
            modelType: "qwen3_forced_aligner",
            classifyNum: 5000
        )

        let model = Qwen3ForcedAlignerModel(config)

        #expect(config.isForcedAligner == true)
        #expect(config.classifyNum == 5000)
        // Model should have been created without error
        _ = model
    }

    // MARK: - Cache Tests

    @Test func qwen3ASRModelMakeCache() {
        let config = Qwen3ASRConfig(
            textConfig: Qwen3TextConfig(
                numHiddenLayers: 4
            )
        )

        let model = Qwen3ASRModel(config)
        let cache = model.makeCache()

        #expect(cache.count == 4)
    }

    // MARK: - Weight Sanitization Tests

    @Test func qwen3ASRSanitizeStripsThinkerPrefix() {
        let weights: [String: MLXArray] = [
            "thinker.model.layers.0.self_attn.q_proj.weight": MLXArray.ones([64, 64]),
            "thinker.model.layers.0.self_attn.k_proj.weight": MLXArray.ones([64, 64]),
            "thinker.audio_tower.conv2d1.weight": MLXArray.ones([32, 3, 3, 1]),
        ]

        let sanitized = Qwen3ASRModel.sanitize(weights: weights)

        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.self_attn.k_proj.weight"] != nil)
        #expect(sanitized["audio_tower.conv2d1.weight"] != nil)
        #expect(sanitized["thinker.model.layers.0.self_attn.q_proj.weight"] == nil)
    }

    @Test func qwen3ASRSanitizeSkipsLmHead() {
        let weights: [String: MLXArray] = [
            "thinker.lm_head.weight": MLXArray.ones([1000, 128]),
            "thinker.model.norm.weight": MLXArray.ones([128]),
        ]

        // skipLmHead = true (default)
        let sanitized = Qwen3ASRModel.sanitize(weights: weights, skipLmHead: true)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.norm.weight"] != nil)

        // skipLmHead = false
        let sanitizedKeep = Qwen3ASRModel.sanitize(weights: weights, skipLmHead: false)
        #expect(sanitizedKeep["lm_head.weight"] != nil)
    }

    @Test func qwen3ASRSanitizeTransposesConv2d() {
        // Simulate PyTorch conv2d weights: (O, I, H, W) shape
        let weights: [String: MLXArray] = [
            "thinker.audio_tower.conv2d1.weight": MLXArray.ones([32, 1, 3, 3]),
        ]

        let sanitized = Qwen3ASRModel.sanitize(weights: weights)

        // Should be transposed to (O, H, W, I)
        let w = sanitized["audio_tower.conv2d1.weight"]!
        #expect(w.shape == [32, 3, 3, 1])
    }

    @Test func qwen3ForcedAlignerSanitizeKeepsLmHead() {
        let weights: [String: MLXArray] = [
            "thinker.lm_head.weight": MLXArray.ones([5000, 128]),
            "thinker.model.norm.weight": MLXArray.ones([128]),
        ]

        let sanitized = Qwen3ForcedAlignerModel.sanitize(weights: weights)

        // ForcedAligner should keep lm_head
        #expect(sanitized["lm_head.weight"] != nil)
        #expect(sanitized["model.norm.weight"] != nil)
    }
}

struct CohereTranscribeModuleSetupTests {

    @Test func cohereConfigDecoding() throws {
        let json = """
        {
          "model_type": "cohere_asr",
          "vocab_size": 16384,
          "sample_rate": 16000,
          "max_audio_clip_s": 35,
          "quantization_config": {
            "group_size": 64,
            "bits": 8
          },
          "encoder": {
            "d_model": 1280,
            "ff_expansion_factor": 4,
            "n_heads": 8,
            "conv_kernel_size": 9,
            "n_layers": 48,
            "pos_emb_max_len": 5000,
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "feat_in": 128
          },
          "transf_decoder": {
            "config_dict": {
              "hidden_size": 1024,
              "inner_size": 4096,
              "num_attention_heads": 8,
              "num_layers": 8,
              "max_sequence_length": 1024
            }
          }
        }
        """

        let config = try JSONDecoder().decode(CohereTranscribeConfig.self, from: Data(json.utf8))
        #expect(config.modelType == "cohere_asr")
        #expect(config.vocabSize == 16384)
        #expect(config.sampleRate == 16000)
        #expect(config.encoder.dModel == 1280)
        #expect(config.encoder.featIn == 128)
        #expect(config.decoder.hiddenSize == 1024)
        #expect(config.decoder.numLayers == 8)
        #expect(config.decoder.maxSequenceLength == 1024)
        #expect(config.quantization?.bits == 8)
    }

    @Test func cohereAudioFeaturesShape() {
        let audio = MLXArray(Array(repeating: Float(0), count: 16_000))
        let filters = CohereTranscribeAudio.computeMelFilters()
        let features = CohereTranscribeAudio.computeFeatures(audio: audio, melFilters: filters)

        #expect(features.ndim == 3)
        #expect(features.shape[0] == 1)
        #expect(features.shape[1] == 128)
        #expect(features.shape[2] > 0)
        #expect(features.reshaped(-1)[0].item(Float.self).isFinite)
    }

    @Test func cohereTokenizerBuildsPromptTokens() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cohere-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let tokenizerData = makeSentencePieceModelData([
            ("<unk>", 0, 2),
            ("<s>", 0, 3),
            ("</s>", 0, 3),
            ("▁hello", -0.1, 1),
            ("▁world", -0.2, 1),
        ])
        try tokenizerData.write(to: fixtureDir.appendingPathComponent("tokenizer.model"))

        let tokenizerConfig = """
        {
          "added_tokens_decoder": {
            "3": {"content": "<|endoftext|>"},
            "4": {"content": "<|startofcontext|>"},
            "5": {"content": "<|startoftranscript|>"},
            "6": {"content": "<|en|>"},
            "7": {"content": "<|pnc|>"},
            "8": {"content": "<|notimestamp|>"},
            "9": {"content": "<|nodiarize|>"},
            "10": {"content": "<|noitn|>"},
            "11": {"content": "<|emo:undefined|>"},
            "12": {"content": "<|timestamp|>"},
            "13": {"content": "<|nopnc|>"}
          }
        }
        """
        try tokenizerConfig.write(
            to: fixtureDir.appendingPathComponent("tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )

        let config = try JSONDecoder().decode(
            CohereTranscribeConfig.self,
            from: Data(makeCohereFixtureConfigJSON().utf8)
        )

        let tokenizer = try CohereTranscribeTokenizer(modelDir: fixtureDir, config: config)
        let prompt = tokenizer.buildPromptTokens(language: "en")

        #expect(prompt.count == 9)
        #expect(prompt[0] == 4)
        #expect(prompt[1] == 5)
        #expect(prompt[2] == 11)
        #expect(prompt[3] == 6)
        #expect(prompt[4] == 6)
        #expect(prompt[5] == 7)
        #expect(prompt[6] == 10)
        #expect(prompt[7] == 8)
        #expect(prompt[8] == 9)
        #expect(tokenizer.encode(text: "<|endoftext|>") == [3])
        #expect(tokenizer.decode(tokens: [3, 3]).isEmpty)
    }
}

struct CohereTranscribeSTTTests {

    @Test func fromDirectoryFixtureSmokeTest() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cohere-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = makeCohereFixtureConfigJSON()
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizerData = makeSentencePieceModelData([
            ("<unk>", 0, 2),
            ("<s>", 0, 3),
            ("</s>", 0, 3),
            ("▁a", -0.1, 1),
            ("▁b", -0.2, 1),
            ("▁c", -0.3, 1),
        ])
        try tokenizerData.write(to: fixtureDir.appendingPathComponent("tokenizer.model"))

        let tokenizerConfig = """
        {
          "added_tokens_decoder": {
            "3": {"content": "<|endoftext|>"},
            "4": {"content": "<|startofcontext|>"},
            "5": {"content": "<|startoftranscript|>"},
            "6": {"content": "<|en|>"},
            "7": {"content": "<|pnc|>"},
            "8": {"content": "<|notimestamp|>"},
            "9": {"content": "<|nodiarize|>"},
            "10": {"content": "<|noitn|>"},
            "11": {"content": "<|emo:undefined|>"},
            "12": {"content": "<|timestamp|>"},
            "13": {"content": "<|nopnc|>"}
          }
        }
        """
        try tokenizerConfig.write(
            to: fixtureDir.appendingPathComponent("tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )

        let seedModel = CohereTranscribeModel(try JSONDecoder().decode(
            CohereTranscribeConfig.self,
            from: Data(configJSON.utf8)
        ))
        var weights = Dictionary(uniqueKeysWithValues: seedModel.parameters().flattened().map { key, value in
            (key, MLXArray.zeros(value.shape, type: Float.self).asType(value.dtype))
        })
        weights["lm_head.bias"] = MLXArray(
            (0..<32).map { $0 == 3 ? Float(1000) : Float(0) }
        )
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try CohereTranscribeModel.fromDirectory(fixtureDir)
        let output = model.generate(
            audio: MLXArray(Array(repeating: Float(0), count: 16_000)),
            generationParameters: STTGenerateParameters(language: "en")
        )

        #expect(output.text.isEmpty)
        #expect(output.promptTokens == 9)
        #expect(output.generationTokens == 0)
        #expect(output.totalTokens == 9)
    }
}

// MARK: - Force Align Processor Tests

struct ForceAlignProcessorTests {

    @Test func isKeptChar() {
        let processor = ForceAlignProcessor()

        #expect(processor.isKeptChar("a") == true)
        #expect(processor.isKeptChar("Z") == true)
        #expect(processor.isKeptChar("5") == true)
        #expect(processor.isKeptChar("'") == true)
        #expect(processor.isKeptChar(" ") == false)
        #expect(processor.isKeptChar(",") == false)
        #expect(processor.isKeptChar(".") == false)
    }

    @Test func cleanToken() {
        let processor = ForceAlignProcessor()

        #expect(processor.cleanToken("hello!") == "hello")
        #expect(processor.cleanToken("it's") == "it's")
        #expect(processor.cleanToken("...test...") == "test")
        #expect(processor.cleanToken("hello world") == "helloworld")
        #expect(processor.cleanToken("") == "")
    }

    @Test func isCJKChar() {
        let processor = ForceAlignProcessor()

        // Chinese characters
        #expect(processor.isCJKChar("\u{4E00}") == true)  // 一
        #expect(processor.isCJKChar("\u{9FFF}") == true)
        #expect(processor.isCJKChar("中") == true)
        #expect(processor.isCJKChar("文") == true)

        // Non-CJK
        #expect(processor.isCJKChar("a") == false)
        #expect(processor.isCJKChar("1") == false)
        #expect(processor.isCJKChar(" ") == false)
    }

    @Test func tokenizeSpaceLangEnglish() {
        let processor = ForceAlignProcessor()

        let tokens = processor.tokenizeSpaceLang("Hello, world! This is a test.")
        #expect(tokens == ["Hello", "world", "This", "is", "a", "test"])
    }

    @Test func tokenizeSpaceLangEmpty() {
        let processor = ForceAlignProcessor()

        let tokens = processor.tokenizeSpaceLang("")
        #expect(tokens.isEmpty)
    }

    @Test func tokenizeSpaceLangWithPunctuation() {
        let processor = ForceAlignProcessor()

        let tokens = processor.tokenizeSpaceLang("I'm don't can't")
        #expect(tokens == ["I'm", "don't", "can't"])
    }

    @Test func tokenizeChineseMixed() {
        let processor = ForceAlignProcessor()

        let tokens = processor.tokenizeChineseMixed("你好world")
        #expect(tokens == ["你", "好", "world"])
    }

    @Test func tokenizeChinesePure() {
        let processor = ForceAlignProcessor()

        let tokens = processor.tokenizeChineseMixed("你好世界")
        #expect(tokens == ["你", "好", "世", "界"])
    }

    @Test func tokenizeChineseMixedWithSpaces() {
        let processor = ForceAlignProcessor()

        let tokens = processor.tokenizeChineseMixed("hello 你好 world")
        #expect(tokens == ["hello", "你", "好", "world"])
    }

    @Test func encodeTimestampEnglish() {
        let processor = ForceAlignProcessor()

        let (wordList, inputText) = processor.encodeTimestamp(
            text: "Hello world",
            language: "English"
        )

        #expect(wordList == ["Hello", "world"])
        #expect(inputText.contains("<|audio_start|>"))
        #expect(inputText.contains("<|audio_pad|>"))
        #expect(inputText.contains("<|audio_end|>"))
        #expect(inputText.contains("<timestamp>"))
        #expect(inputText.contains("Hello"))
        #expect(inputText.contains("world"))
    }

    @Test func encodeTimestampChinese() {
        let processor = ForceAlignProcessor()

        let (wordList, inputText) = processor.encodeTimestamp(
            text: "你好世界",
            language: "Chinese"
        )

        #expect(wordList == ["你", "好", "世", "界"])
        #expect(inputText.contains("<timestamp>"))
    }

    @Test func fixTimestampMonotonic() {
        let processor = ForceAlignProcessor()

        // Already monotonic
        let result = processor.fixTimestamp([100, 200, 300, 400])
        #expect(result == [100, 200, 300, 400])
    }

    @Test func fixTimestampEmpty() {
        let processor = ForceAlignProcessor()

        let result = processor.fixTimestamp([])
        #expect(result.isEmpty)
    }

    @Test func fixTimestampSingleElement() {
        let processor = ForceAlignProcessor()

        let result = processor.fixTimestamp([500])
        #expect(result == [500])
    }

    @Test func fixTimestampNonMonotonic() {
        let processor = ForceAlignProcessor()

        // Non-monotonic: 300 is out of place
        let result = processor.fixTimestamp([100, 200, 300, 150, 400, 500])

        // Result should be monotonically non-decreasing
        for i in 1..<result.count {
            #expect(result[i] >= result[i - 1], "Timestamps should be non-decreasing at index \(i)")
        }
    }

    @Test func parseTimestamp() {
        let processor = ForceAlignProcessor()

        let wordList = ["Hello", "world"]
        // 4 timestamps: start1, end1, start2, end2
        let timestamps: [Double] = [1000, 2000, 2500, 3500]

        let items = processor.parseTimestamp(wordList: wordList, timestamp: timestamps)

        #expect(items.count == 2)
        #expect(items[0].text == "Hello")
        #expect(items[0].startTime == 1.0)
        #expect(items[0].endTime == 2.0)
        #expect(items[1].text == "world")
        #expect(items[1].startTime == 2.5)
        #expect(items[1].endTime == 3.5)
    }
}

// MARK: - ForcedAlignResult Tests

struct ForcedAlignResultTests {

    @Test func forcedAlignResultText() {
        let result = ForcedAlignResult(items: [
            ForcedAlignItem(text: "Hello", startTime: 0.0, endTime: 0.5),
            ForcedAlignItem(text: "world", startTime: 0.5, endTime: 1.0),
        ])

        #expect(result.text == "Hello world")
    }

    @Test func forcedAlignResultSegments() {
        let result = ForcedAlignResult(items: [
            ForcedAlignItem(text: "Hello", startTime: 0.0, endTime: 0.5),
            ForcedAlignItem(text: "world", startTime: 0.5, endTime: 1.0),
        ])

        let segments = result.segments
        #expect(segments.count == 2)
        #expect(segments[0]["text"] as? String == "Hello")
        #expect(segments[0]["start"] as? Double == 0.0)
        #expect(segments[0]["end"] as? Double == 0.5)
        #expect(segments[1]["text"] as? String == "world")
    }

    @Test func forcedAlignResultEmpty() {
        let result = ForcedAlignResult(items: [])
        #expect(result.text == "")
        #expect(result.segments.isEmpty)
    }
}

// MARK: - Helper Function Tests

struct Qwen3ASRHelperTests {

    @Test func getFeatExtractOutputLengthsBasic() {
        // Test with a known input length
        let inputLengths = MLXArray([Int32(200)])
        let output = getFeatExtractOutputLengths(inputLengths)
        let result = Int(output[0].item(Int32.self))

        // Should produce a positive output length
        #expect(result > 0)
    }

    @Test func getFeatExtractOutputLengthsMultiple() {
        let inputLengths = MLXArray([Int32(100), Int32(200), Int32(300)])
        let output = getFeatExtractOutputLengths(inputLengths)

        // All output lengths should be positive
        for i in 0..<3 {
            let result = Int(output[i].item(Int32.self))
            #expect(result > 0, "Output length at index \(i) should be positive")
        }

        // Longer input should produce longer or equal output
        let len1 = Int(output[0].item(Int32.self))
        let len2 = Int(output[1].item(Int32.self))
        let len3 = Int(output[2].item(Int32.self))
        #expect(len2 >= len1)
        #expect(len3 >= len2)
    }

    @Test func getFeatExtractOutputLengthsChunkBoundary() {
        // Test at chunk boundary (100)
        let inputLengths = MLXArray([Int32(100)])
        let output = getFeatExtractOutputLengths(inputLengths)
        let result = Int(output[0].item(Int32.self))

        // At boundary of 100, should get 13 tokens from the chunk
        #expect(result == 13)
    }

    @Test func computeChunkedEncoderWindowLengthsMatchesChunkedOutputs() {
        let windowLengths = computeChunkedEncoderWindowLengths(
            chunkFeatureLengthsAfterCnn: Array(repeating: 13, count: 21) + [8],
            chunkCountsPerInput: [22],
            chunksPerWindow: 8
        )

        #expect(windowLengths == [104, 104, 73])
    }

    @Test func computeChunkedEncoderWindowLengthsPreservesSplitFixtureShape() {
        let windowLengths = computeChunkedEncoderWindowLengths(
            chunkFeatureLengthsAfterCnn: Array(repeating: 13, count: 21) + [8]
                + Array(repeating: 13, count: 8) + [5],
            chunkCountsPerInput: [22, 9],
            chunksPerWindow: 8
        )

        #expect(windowLengths == [104, 104, 73, 104, 5])
        #expect(windowLengths.reduce(0, +) == 390)
    }
}

// MARK: - Audio Chunking Tests

struct SplitAudioIntoChunksTests {

    @Test func shortAudioReturnsOneChunk() {
        // 1 second of audio at 16kHz
        let sampleRate = 16000
        let audio = MLXArray(Array(repeating: Float(0.5), count: sampleRate))

        let chunks = splitAudioIntoChunks(audio, sampleRate: sampleRate, chunkDuration: 1200.0)

        #expect(chunks.count == 1)
        #expect(chunks[0].1 == 0.0, "Offset should be 0")
        #expect(chunks[0].0.dim(0) == sampleRate)
    }

    @Test func veryShortAudioGetsPadded() {
        // 0.1 seconds at 16kHz = 1600 samples
        let sampleRate = 16000
        let audio = MLXArray(Array(repeating: Float(0.1), count: 1600))

        let chunks = splitAudioIntoChunks(
            audio,
            sampleRate: sampleRate,
            chunkDuration: 1200.0,
            minChunkDuration: 1.0
        )

        #expect(chunks.count == 1)
        // Should be padded to at least 1.0 second = 16000 samples
        #expect(chunks[0].0.dim(0) >= sampleRate)
    }

    @Test func longAudioGetsSplit() {
        // 10 seconds of audio, chunk at 3 seconds
        let sampleRate = 16000
        let totalSamples = sampleRate * 10
        let audio = MLXArray(Array(repeating: Float(0.3), count: totalSamples))

        let chunks = splitAudioIntoChunks(
            audio,
            sampleRate: sampleRate,
            chunkDuration: 3.0,
            minChunkDuration: 0.5
        )

        // Should have multiple chunks
        #expect(chunks.count > 1, "10 seconds of audio with 3s chunk duration should produce multiple chunks")

        // All offsets should be non-negative and increasing
        for i in 1..<chunks.count {
            #expect(chunks[i].1 > chunks[i - 1].1, "Offsets should be increasing")
        }

        // First offset should be 0
        #expect(chunks[0].1 == 0.0)
    }

    @Test func chunksSplitAtLowEnergy() {
        let sampleRate = 16000

        // Create audio with loud and silent sections:
        // 2s loud -> 1s silence -> 2s loud -> 1s silence -> 2s loud = 8s total
        var samples = [Float]()
        for i in 0..<(sampleRate * 8) {
            let t = Float(i) / Float(sampleRate)
            if t < 2.0 || (t >= 3.0 && t < 5.0) || t >= 6.0 {
                // Loud section: sine wave
                samples.append(sin(t * 440.0 * 2.0 * .pi) * 0.8)
            } else {
                // Silent section
                samples.append(0.0)
            }
        }
        let audio = MLXArray(samples)

        let chunks = splitAudioIntoChunks(
            audio,
            sampleRate: sampleRate,
            chunkDuration: 3.0,
            minChunkDuration: 0.5,
            searchExpandSec: 1.5,
            minWindowMs: 50.0
        )

        // Should produce multiple chunks
        #expect(chunks.count >= 2, "Should split into at least 2 chunks")

        // Each chunk should have positive length
        for (chunk, _) in chunks {
            #expect(chunk.dim(0) > 0, "Chunk should not be empty")
        }
    }

    @Test func multidimensionalAudioReduced() {
        // Stereo audio (2D)
        let sampleRate = 16000
        let left = Array(repeating: Float(0.5), count: sampleRate)
        let right = Array(repeating: Float(0.3), count: sampleRate)
        let stereo = MLXArray(left + right).reshaped(2, sampleRate).transposed()
        // shape: [sampleRate, 2]

        let chunks = splitAudioIntoChunks(stereo, sampleRate: sampleRate)

        #expect(chunks.count == 1)
        // After mean(axis: -1), should be 1D
        #expect(chunks[0].0.ndim == 1)
    }

    @Test func exactChunkBoundary() {
        // Audio exactly at chunk duration
        let sampleRate = 16000
        let chunkDuration: Float = 5.0
        let totalSamples = Int(chunkDuration * Float(sampleRate))
        let audio = MLXArray(Array(repeating: Float(0.2), count: totalSamples))

        let chunks = splitAudioIntoChunks(
            audio,
            sampleRate: sampleRate,
            chunkDuration: chunkDuration
        )

        // Should be exactly 1 chunk since totalSec <= chunkDuration
        #expect(chunks.count == 1)
    }

    @Test func allChunksCoverFullAudio() {
        // Verify no samples are lost
        let sampleRate = 16000
        let totalSamples = sampleRate * 7  // 7 seconds
        let audio = MLXArray(Array(repeating: Float(0.1), count: totalSamples))

        let chunks = splitAudioIntoChunks(
            audio,
            sampleRate: sampleRate,
            chunkDuration: 2.0,
            minChunkDuration: 0.5
        )

        // Sum of chunk samples should be >= total (may include padding)
        let totalChunkSamples = chunks.reduce(0) { $0 + $1.0.dim(0) }
        #expect(totalChunkSamples >= totalSamples, "Chunks should cover all audio samples")
    }
}
struct ParakeetSTTTests {

    @Test func variantResolutionAndTypedParsing() throws {
        let tdtJSON = """
        {
          "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
          "model_defaults": {"tdt_durations": [0, 1, 2]},
          "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "dither": 0.0
          },
          "encoder": {
            "feat_in": 80,
            "n_layers": 2,
            "d_model": 32,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 15,
            "subsampling_conv_channels": 32,
            "pos_emb_max_len": 512
          },
          "decoder": {
            "blank_as_pad": true,
            "vocab_size": 4,
            "prednet": {
              "pred_hidden": 32,
              "pred_rnn_layers": 1
            }
          },
          "joint": {
            "num_classes": 4,
            "vocabulary": ["▁", "a", "b", "."],
            "jointnet": {
              "joint_hidden": 32,
              "activation": "relu",
              "encoder_hidden": 32,
              "pred_hidden": 32
            }
          },
          "decoding": {
            "model_type": "tdt",
            "durations": [0, 1, 2],
            "greedy": {"max_symbols": 10}
          }
        }
        """

        let raw = try JSONDecoder().decode(ParakeetRawConfig.self, from: Data(tdtJSON.utf8))
        let variant = try ParakeetVariantResolver.resolve(raw)
        #expect(variant == .tdt)

        let typed = try ParakeetConfigParser.parseTDT(raw)
        #expect(typed.preprocessor.sampleRate == 16000)
        #expect(typed.encoder.subsampling == "dw_striding")
        #expect(typed.decoding.durations == [0, 1, 2])
        #expect(typed.decoding.greedy?.maxSymbols == 10)
    }

    @Test func ctcVariantResolution() throws {
        let ctcJSON = """
        {
          "target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
          "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "dither": 0.0
          },
          "encoder": {
            "feat_in": 80,
            "n_layers": 2,
            "d_model": 32,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 15,
            "subsampling_conv_channels": 32,
            "pos_emb_max_len": 512
          },
          "decoder": {
            "feat_in": 32,
            "num_classes": 4,
            "vocabulary": ["▁", "a", "b", "."]
          },
          "decoding": {"greedy": {"max_symbols": 8}}
        }
        """

        let raw = try JSONDecoder().decode(ParakeetRawConfig.self, from: Data(ctcJSON.utf8))
        #expect(try ParakeetVariantResolver.resolve(raw) == .ctc)
        let typed = try ParakeetConfigParser.parseCTC(raw)
        #expect(typed.decoder.featIn == 32)
        #expect(typed.decoder.vocabulary.count == 4)
    }

    @Test func tokenizerDecodesSentencePieceMarker() {
        let vocab = ["▁", "h", "e", "l", "o", "."]
        let text = ParakeetTokenizer.decode(tokens: [0, 1, 2, 3, 3, 4, 5], vocabulary: vocab)
        #expect(text == " hello.")
    }

    @Test func alignmentSentenceAndMergeUtilities() throws {
        let tokens: [ParakeetAlignedToken] = [
            .init(id: 1, text: "Hi", start: 0.0, duration: 0.2),
            .init(id: 2, text: ".", start: 0.2, duration: 0.1),
            .init(id: 3, text: " Next", start: 0.5, duration: 0.2),
            .init(id: 4, text: "!", start: 0.7, duration: 0.1),
        ]
        let sentences = ParakeetAlignment.tokensToSentences(tokens)
        #expect(sentences.count == 2)
        #expect(sentences[0].text == "Hi.")
        #expect(sentences[1].text == " Next!")

        let a: [ParakeetAlignedToken] = [
            .init(id: 1, text: " a", start: 0.0, duration: 0.2),
            .init(id: 2, text: " b", start: 0.2, duration: 0.2),
            .init(id: 3, text: " c", start: 0.4, duration: 0.2),
        ]
        let b: [ParakeetAlignedToken] = [
            .init(id: 2, text: " b", start: 0.21, duration: 0.2),
            .init(id: 3, text: " c", start: 0.41, duration: 0.2),
            .init(id: 4, text: " d", start: 0.61, duration: 0.2),
        ]
        let mergedContiguous = try ParakeetAlignment.mergeLongestContiguous(a, b, overlapDuration: 0.6)
        #expect(mergedContiguous.map(\.id) == [1, 2, 3, 4])

        let mergedLCS = ParakeetAlignment.mergeLongestCommonSubsequence(a, b, overlapDuration: 0.6)
        #expect(mergedLCS.map(\.id) == [1, 2, 3, 4])
    }

    @Test func melPreprocessingProducesExpectedShape() {
        let config = ParakeetPreprocessConfig(
            sampleRate: 16000,
            normalize: "per_feature",
            windowSize: 0.02,
            windowStride: 0.01,
            window: "hann",
            features: 80,
            nFft: 512,
            dither: 0,
            padTo: 0,
            padValue: 0,
            preemph: 0.97
        )

        let audio = MLXArray(Array(repeating: Float(0.0), count: 16000))
        let mel = ParakeetAudio.logMelSpectrogram(audio, config: config)

        #expect(mel.ndim == 3)
        #expect(mel.shape[0] == 1)
        #expect(mel.shape[2] == 80)
        #expect(mel.shape[1] > 0)
    }

    @Test func deterministicRNNTAndTDTControlFlow() {
        let rnntBlank = 10
        let rnntStep1 = ParakeetDecodingLogic.rnntStep(
            predictedToken: rnntBlank,
            blankToken: rnntBlank,
            time: 5,
            newSymbols: 2,
            maxSymbols: 4
        )
        #expect(rnntStep1.nextTime == 6)
        #expect(rnntStep1.nextNewSymbols == 0)
        #expect(rnntStep1.emittedToken == false)

        let rnntStep2 = ParakeetDecodingLogic.rnntStep(
            predictedToken: 2,
            blankToken: rnntBlank,
            time: 8,
            newSymbols: 3,
            maxSymbols: 4
        )
        #expect(rnntStep2.nextTime == 9)
        #expect(rnntStep2.nextNewSymbols == 0)
        #expect(rnntStep2.emittedToken == true)

        let tdtStep1 = ParakeetDecodingLogic.tdtStep(
            predictedToken: 1,
            blankToken: 5,
            decisionIndex: 1,
            durations: [0, 2, 4],
            time: 10,
            newSymbols: 0,
            maxSymbols: 4
        )
        #expect(tdtStep1.nextTime == 12)
        #expect(tdtStep1.nextNewSymbols == 0)
        #expect(tdtStep1.jump == 2)
        #expect(tdtStep1.emittedToken == true)

        let tdtStep2 = ParakeetDecodingLogic.tdtStep(
            predictedToken: 1,
            blankToken: 5,
            decisionIndex: 0,
            durations: [0, 2, 4],
            time: 3,
            newSymbols: 3,
            maxSymbols: 4
        )
        #expect(tdtStep2.nextTime == 4)  // zero-duration + max_symbols fallback
        #expect(tdtStep2.nextNewSymbols == 0)
        #expect(tdtStep2.jump == 0)
    }

    @Test func deterministicCTCCollapseSpans() {
        let spans = ParakeetDecodingLogic.ctcSpans(
            bestTokens: [5, 5, 9, 2, 2, 9, 2, 3, 3],
            blankToken: 9
        )
        #expect(spans == [
            .init(token: 5, startFrame: 0, endFrame: 2),
            .init(token: 2, startFrame: 3, endFrame: 5),
            .init(token: 2, startFrame: 6, endFrame: 7),
            .init(token: 3, startFrame: 7, endFrame: 9),
        ])
    }

    @Test func tokenizerFiltersSpecialTokens() {
        let vocab = ["<unk>", "<|nospeech|>", "<pad>", "<|startoftranscript|>",
                     "<|pnc|>", "<|ru|>", "▁", "h", "e", "l", "o"]
        let tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10]
        let text = ParakeetTokenizer.decode(tokens: tokens, vocabulary: vocab)
        #expect(text == " hello")

        #expect(ParakeetTokenizer.isSpecialToken(0, vocabulary: vocab) == true)   // <unk>
        #expect(ParakeetTokenizer.isSpecialToken(2, vocabulary: vocab) == true)   // <pad>
        #expect(ParakeetTokenizer.isSpecialToken(3, vocabulary: vocab) == true)   // <|startoftranscript|>
        #expect(ParakeetTokenizer.isSpecialToken(5, vocabulary: vocab) == true)   // <|ru|>
        #expect(ParakeetTokenizer.isSpecialToken(7, vocabulary: vocab) == false)  // h
        #expect(ParakeetTokenizer.isSpecialToken(99, vocabulary: vocab) == false) // out of range
    }

    @Test func modelIsInEvalModeAfterLoading() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("parakeet-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)

        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = """
        {
          "target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
          "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "dither": 0.0
          },
          "encoder": {
            "feat_in": 80,
            "n_layers": 0,
            "d_model": 16,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 2,
            "self_attention_model": "abs_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 15,
            "subsampling_conv_channels": 16,
            "pos_emb_max_len": 128
          },
          "decoder": {
            "feat_in": 16,
            "num_classes": 4,
            "vocabulary": ["▁", "a", "b", "."]
          },
          "decoding": {"greedy": {"max_symbols": 8}}
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let weights: [String: MLXArray] = [
            "encoder.pre_encode.conv0.weight": MLXArray.zeros([16, 3, 3, 1], type: Float.self),
            "encoder.pre_encode.conv0.bias": MLXArray.zeros([16], type: Float.self),
            "encoder.pre_encode.out.weight": MLXArray.zeros([16, 640], type: Float.self),
            "encoder.pre_encode.out.bias": MLXArray.zeros([16], type: Float.self),
            "decoder.decoder_layers.0.weight": MLXArray.zeros([5, 1, 16], type: Float.self),
            "decoder.decoder_layers.0.bias": MLXArray.zeros([5], type: Float.self),
        ]
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try ParakeetModel.fromDirectory(fixtureDir)
        #expect(model.training == false)

        let audio = MLXArray(Array(repeating: Float(0), count: 3200))
        let output = model.generate(audio: audio)

        #expect(model.variant == .ctc)
        #expect(model.vocabulary.count == 4)
        #expect(output.text.count >= 0)
    }

    @Test func melPreprocessingNumericInvariants() {
        let config = ParakeetPreprocessConfig(
            sampleRate: 16000, normalize: "per_feature",
            windowSize: 0.025, windowStride: 0.01, window: "hann",
            features: 128, nFft: 512, dither: 0
        )

        let audio = MLXArray(Array(repeating: Float(0.1), count: 16000))
        let mel = ParakeetAudio.logMelSpectrogram(audio, config: config)

        #expect(mel.ndim == 3)
        #expect(mel.shape[0] == 1)
        #expect(mel.shape[2] == 128)

        let mean = MLX.mean(mel).item(Float.self)
        let std = MLX.std(mel).item(Float.self)
        #expect(abs(mean) < 0.01, "Per-feature normalized mel should have near-zero mean, got \(mean)")
        #expect(abs(std - 1.0) < 0.05, "Per-feature normalized mel should have near-unit std, got \(std)")
        #expect(mel.min().item(Float.self).isFinite)
        #expect(mel.max().item(Float.self).isFinite)
    }

    @Test func melPreprocessingShortAudioDoesNotCrash() {
        let config = ParakeetPreprocessConfig(
            sampleRate: 16000, normalize: "per_feature",
            windowSize: 0.025, windowStride: 0.01, window: "hann",
            features: 80, nFft: 512, dither: 0
        )

        let shortAudio = MLXArray([Float(0.1), 0.2, 0.3])
        let mel = ParakeetAudio.logMelSpectrogram(shortAudio, config: config)

        #expect(mel.ndim == 3)
        #expect(mel.reshaped(-1)[0].item(Float.self).isFinite)
    }
}

struct VoxtralRealtimeSTTTests {
    @Test func configDecodesNestedAudioEncodingArgs() throws {
        let json = """
        {
          "model_type": "voxtral_realtime",
          "encoder_args": {
            "dim": 1280,
            "audio_encoding_args": {
              "sampling_rate": 16000,
              "num_mel_bins": 128,
              "window_size": 400,
              "hop_length": 160,
              "global_log_mel_max": 1.5
            }
          },
          "decoder": {
            "dim": 3072,
            "n_layers": 2,
            "n_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "hidden_dim": 9216,
            "vocab_size": 131072
          }
        }
        """

        let config = try JSONDecoder().decode(VoxtralRealtimeConfig.self, from: Data(json.utf8))
        #expect(config.modelType == "voxtral_realtime")
        #expect(config.audioEncodingArgs.numMelBins == 128)
        #expect(config.decoder.dim == 3072)
        #expect(config.vocabSize == 131072)
    }

    @Test func tokenizerDecodeSkipsSpecialTokens() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-tekken-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let tekkenJSON = """
        {
          "vocab": [
            {"token_bytes": "aGVs"},
            {"token_bytes": "bG8="}
          ],
          "config": {"default_num_special_tokens": 1000},
          "special_tokens": [{"rank": 2}, {"rank": 32}]
        }
        """
        try tekkenJSON.write(
            to: fixtureDir.appendingPathComponent("tekken.json"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizer = try VoxtralRealtimeTokenizer.fromModelDirectory(fixtureDir)
        let text = tokenizer.decode(tokenIds: [1, 1000, 1001, 2, 32])
        #expect(text == "hello")
    }

    @Test func audioMelProducesExpectedShape() {
        let audio = MLXArray(Array(repeating: Float(0), count: 16000))
        let filters = VoxtralRealtimeAudio.computeMelFilters()
        let mel = VoxtralRealtimeAudio.computeMelSpectrogram(
            audio: audio,
            melFilters: filters,
            windowSize: 400,
            hopLength: 160,
            globalLogMelMax: 1.5
        )

        #expect(mel.ndim == 2)
        #expect(mel.shape[0] == 128)
        #expect(mel.shape[1] > 0)
    }

    @Test func sanitizeRemapsAndTransposesConvWeights() {
        let convWeight = MLXArray.zeros([8, 4, 3], type: Float.self)
        let weights: [String: MLXArray] = [
            "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight": convWeight,
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": MLXArray.zeros([32, 16], type: Float.self),
            "layers.0.feed_forward.w1.weight": MLXArray.zeros([4, 4], type: Float.self),
            "layers.0.ada_rms_norm_t_cond.0.weight": MLXArray.zeros([4, 4], type: Float.self),
        ]

        let sanitized = VoxtralRealtimeModel.sanitize(weights: weights)

        let mappedConv = sanitized["encoder.conv_layers_0_conv.conv.weight"]
        #expect(mappedConv != nil)
        #expect(mappedConv?.shape == [8, 3, 4])
        #expect(sanitized["decoder.tok_embeddings.weight"] != nil)
        #expect(sanitized["decoder.layers.0.feed_forward_w1.weight"] != nil)
        #expect(sanitized["decoder.layers.0.ada_rms_norm_t_cond.ada_down.weight"] != nil)
    }

    @Test func fromDirectoryAndGenerateEOSSmoke() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = """
        {
          "model_type": "voxtral_realtime",
          "encoder_args": {
            "dim": 16,
            "n_layers": 0,
            "n_heads": 2,
            "head_dim": 8,
            "hidden_dim": 32,
            "n_kv_heads": 2,
            "norm_eps": 1e-5,
            "rope_theta": 1000000,
            "sliding_window": 64,
            "causal": true,
            "use_biases": true,
            "downsample_factor": 4
          },
          "decoder": {
            "dim": 16,
            "n_layers": 0,
            "n_heads": 2,
            "n_kv_heads": 2,
            "head_dim": 8,
            "hidden_dim": 32,
            "vocab_size": 8,
            "norm_eps": 1e-5,
            "rope_theta": 1000000,
            "sliding_window": 64,
            "tied_embeddings": true,
            "ada_rms_norm_t_cond": false,
            "ada_rms_norm_t_cond_dim": 4
          },
          "audio_encoding_args": {
            "sampling_rate": 16000,
            "frame_rate": 12.5,
            "num_mel_bins": 128,
            "hop_length": 160,
            "window_size": 400,
            "global_log_mel_max": 1.5
          },
          "transcription_delay_ms": 0,
          "bos_token_id": 1,
          "eos_token_id": 0,
          "streaming_pad_token_id": 2,
          "n_left_pad_tokens": 1
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let tekkenJSON = """
        {
          "vocab": [
            {"token_bytes":"YQ=="},
            {"token_bytes":"Yg=="},
            {"token_bytes":"Yw=="},
            {"token_bytes":"ZA=="},
            {"token_bytes":"ZQ=="},
            {"token_bytes":"Zg=="},
            {"token_bytes":"Zw=="},
            {"token_bytes":"aA=="}
          ],
          "config":{"default_num_special_tokens":0},
          "special_tokens":[]
        }
        """
        try tekkenJSON.write(
            to: fixtureDir.appendingPathComponent("tekken.json"),
            atomically: true,
            encoding: .utf8
        )

        // All-zero embeddings force argmax token 0 immediately, matching EOS id above.
        let weights: [String: MLXArray] = [
            "encoder.conv_layers_0_conv.conv.weight": MLXArray.zeros([16, 3, 128], type: Float.self),
            "encoder.conv_layers_0_conv.conv.bias": MLXArray.zeros([16], type: Float.self),
            "encoder.conv_layers_1_conv.conv.weight": MLXArray.zeros([16, 3, 16], type: Float.self),
            "encoder.conv_layers_1_conv.conv.bias": MLXArray.zeros([16], type: Float.self),
            "encoder.transformer_norm.weight": MLXArray.ones([16], type: Float.self),
            "encoder.audio_language_projection_0.weight": MLXArray.zeros([16, 64], type: Float.self),
            "encoder.audio_language_projection_2.weight": MLXArray.zeros([16, 16], type: Float.self),
            "decoder.tok_embeddings.weight": MLXArray.zeros([8, 16], type: Float.self),
            "decoder.norm.weight": MLXArray.ones([16], type: Float.self),
        ]
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let audio = MLXArray(Array(repeating: Float(0), count: 16000))
        let output = model.generate(
            audio: audio,
            generationParameters: STTGenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        #expect(output.promptTokens > 0)
        #expect(output.generationTokens == 0)
        #expect(output.totalTokens == output.promptTokens)
        #expect(output.text == "")
    }
}

@Suite("FireRed ASR 2 Tests", .serialized)
struct FireRedASR2Tests {

    @Test func configDefaultsAndDecoding() throws {
        let defaults = FireRedASR2Config()
        #expect(defaults.modelType == "fireredasr2")
        #expect(defaults.idim == 80)
        #expect(defaults.odim == 8667)
        #expect(defaults.sosID == 3)
        #expect(defaults.eosID == 4)
        #expect(defaults.encoder.nLayers == 16)
        #expect(defaults.decoder.nHead == 20)

        let json = """
        {
          "model_type": "fireredasr2",
          "idim": 80,
          "odim": 32,
          "sos_id": 7,
          "eos_id": 8,
          "encoder": {
            "n_layers": 2,
            "n_head": 4,
            "d_model": 32,
            "kernel_size": 15,
            "pe_maxlen": 256
          },
          "decoder": {
            "n_layers": 3,
            "n_head": 4,
            "d_model": 32,
            "pe_maxlen": 512
          }
        }
        """
        let decoded = try JSONDecoder().decode(FireRedASR2Config.self, from: Data(json.utf8))
        #expect(decoded.odim == 32)
        #expect(decoded.sosID == 7)
        #expect(decoded.eosID == 8)
        #expect(decoded.encoder.nLayers == 2)
        #expect(decoded.encoder.kernelSize == 15)
        #expect(decoded.decoder.nLayers == 3)
        #expect(decoded.decoder.peMaxlen == 512)
    }

    @Test func fbankExtractionShape() {
        let audio = MLXArray(Array(repeating: Float(0), count: 16000))
        let fbank = FireRedASR2Audio.extractFbank(audio)

        #expect(fbank.ndim == 2)
        #expect(fbank.shape[0] == 98)
        #expect(fbank.shape[1] == 80)
    }

    @Test func tokenizerCleanup() {
        let tokenizer = FireRedASR2Tokenizer(vocabulary: ["<blank>", "\u{2581}Hello", "<sil>", "World"])
        let text = tokenizer.decode(tokenIds: [0, 1, 2, 3])
        #expect(text == "helloworld")
    }

    @Test func encoderAndDecoderShapes() {
        let config = FireRedASR2Config(
            odim: 16,
            dModel: 16,
            encoder: FireRedASR2EncoderConfig(
                nLayers: 1,
                nHead: 4,
                dModel: 16,
                kernelSize: 15,
                peMaxlen: 128
            ),
            decoder: FireRedASR2DecoderConfig(
                nLayers: 1,
                nHead: 4,
                dModel: 16,
                peMaxlen: 128
            )
        )
        let model = FireRedASR2Model(config)

        let features = MLXArray.zeros([1, 12, 80], type: Float.self)
        let encoderOutput = model.encode(features)
        #expect(encoderOutput.shape == [1, 3, 16])

        let tokens = MLXArray([Int32(config.sosID), 1]).reshaped([1, 2])
        let (logits, cache) = model.decodeOneStep(tokens, encoderOutput: encoderOutput)
        #expect(logits.shape == [1, config.odim])
        #expect(cache.count == config.decoder.nLayers)
        #expect(cache[0]?.shape == [1, 2, 16])
    }

    @Test func sanitizeRemapsAndTransposesWeights() {
        let weights: [String: MLXArray] = [
            "encoder.input_preprocessor.conv.0.weight": MLXArray.zeros([8, 1, 3, 3], type: Float.self),
            "encoder.layer_stack.0.ffn1.net.1.weight": MLXArray.zeros([16, 8], type: Float.self),
            "encoder.layer_stack.0.conv.pointwise_conv1.weight": MLXArray.zeros([8, 4, 3], type: Float.self),
            "decoder.tgt_word_emb.weight": MLXArray.zeros([6, 8], type: Float.self),
        ]

        let sanitized = FireRedASR2Model.sanitize(weights: weights)
        #expect(sanitized["encoder.input_preprocessor.conv1.weight"]?.shape == [8, 3, 3, 1])
        #expect(sanitized["encoder.layer_stack.0.ffn1.net_1.weight"] != nil)
        #expect(sanitized["encoder.layer_stack.0.conv.pointwise_conv1.weight"]?.shape == [8, 3, 4])
        #expect(sanitized["decoder.tgt_word_prj.weight"]?.shape == [6, 8])
    }

    @Test func fromDirectoryFixtureSmokeTest() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("firered-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = """
        {
          "model_type": "fireredasr2",
          "idim": 80,
          "odim": 6,
          "d_model": 8,
          "sos_id": 3,
          "eos_id": 4,
          "pad_id": 2,
          "blank_id": 0,
          "encoder": {
            "n_layers": 0,
            "n_head": 2,
            "d_model": 8,
            "kernel_size": 15,
            "pe_maxlen": 64
          },
          "decoder": {
            "n_layers": 0,
            "n_head": 2,
            "d_model": 8,
            "pe_maxlen": 64
          }
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let cmvnJSON = """
        {
          "means": \(Array(repeating: 0.0, count: 80)),
          "istd": \(Array(repeating: 1.0, count: 80))
        }
        """
        try cmvnJSON.write(
            to: fixtureDir.appendingPathComponent("cmvn.json"),
            atomically: true,
            encoding: .utf8
        )

        let dict = """
        <blank> 0
        ▁a 1
        ▁b 2
        <sos/eos> 3
        <eos> 4
        ▁c 5
        """
        try dict.write(
            to: fixtureDir.appendingPathComponent("dict.txt"),
            atomically: true,
            encoding: .utf8
        )

        let weights: [String: MLXArray] = [
            "encoder.input_preprocessor.conv1.weight": MLXArray.zeros([32, 1, 3, 3], type: Float.self),
            "encoder.input_preprocessor.conv1.bias": MLXArray.zeros([32], type: Float.self),
            "encoder.input_preprocessor.conv2.weight": MLXArray.zeros([32, 32, 3, 3], type: Float.self),
            "encoder.input_preprocessor.conv2.bias": MLXArray.zeros([32], type: Float.self),
            "encoder.input_preprocessor.out.weight": MLXArray.zeros([8, 608], type: Float.self),
            "encoder.input_preprocessor.out.bias": MLXArray.zeros([8], type: Float.self),
            "decoder.tgt_word_emb.weight": MLXArray.zeros([6, 8], type: Float.self),
            "decoder.layer_norm_out.weight": MLXArray.ones([8], type: Float.self),
            "decoder.layer_norm_out.bias": MLXArray.zeros([8], type: Float.self),
        ]
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try FireRedASR2Model.fromDirectory(fixtureDir)
        let audio = MLXArray(Array(repeating: Float(0), count: 16000))
        let output = model.generate(
            audio: audio,
            generationParameters: STTGenerateParameters(maxTokens: 2, temperature: 0.0)
        )

        #expect(model.vocabulary.count == 6)
        #expect(output.generationTokens >= 0)
        #expect(output.totalTime >= 0)
    }
}

@Suite("FireRed ASR 2 Network Tests", .serialized)
struct FireRedASR2NetworkTests {

    @Test func fireredFromPretrainedLoadsRealWeightsAndTranscribesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network FireRed test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_FIRERED_REPO"] ?? "mlx-community/FireRedASR2-AED-mlx"
        let model = try await FireRedASR2Model.fromPretrained(repo)
        let audio = try loadSTTNetworkFixture(sampleRate: 16000)
        let output = model.generate(
            audio: audio,
            beamSize: 3,
            softmaxSmoothing: 1.25,
            lengthPenalty: 0.6,
            eosPenalty: 1.0,
            maxLen: 128,
            language: "English"
        )

        #expect(model.config.modelType == "fireredasr2")
        #expect(model.cmvnMeans != nil)
        #expect(model.cmvnIstd != nil)
        #expect(!model.vocabulary.isEmpty)
        #expect(!output.text.isEmpty)
        #expect(output.generationTokens > 0)
    }
}

@Suite("SenseVoice Tests", .serialized)
struct SenseVoiceTests {

    @Test func configDefaultsAndTypoDecoding() throws {
        let defaults = SenseVoiceConfig()
        #expect(defaults.modelType == "sensevoice")
        #expect(defaults.vocabSize == 25055)
        #expect(defaults.inputSize == 560)
        #expect(defaults.encoderConf.outputSize == 512)
        #expect(defaults.encoderConf.numBlocks == 50)
        #expect(defaults.frontendConf.lfrM == 7)
        #expect(defaults.frontendConf.lfrN == 6)

        let json = """
        {
          "model_type": "sensevoice",
          "vocab_size": 100,
          "input_size": 560,
          "encoder_conf": {
            "output_size": 64,
            "attention_heads": 2,
            "linear_units": 128,
            "num_blocks": 3,
            "tp_blocks": 2,
            "kernel_size": 5,
            "sanm_shfit": 2
          },
          "frontend_conf": {
            "fs": 16000,
            "n_mels": 80,
            "lfr_m": 7,
            "lfr_n": 6
          }
        }
        """
        let decoded = try JSONDecoder().decode(SenseVoiceConfig.self, from: Data(json.utf8))
        #expect(decoded.vocabSize == 100)
        #expect(decoded.encoderConf.outputSize == 64)
        #expect(decoded.encoderConf.sanmShift == 2)
        #expect(decoded.frontendConf.nMels == 80)
    }

    @Test func lfrAndCMVNBehaveLikeReference() {
        let feats = MLXArray((0..<20).map(Float.init)).reshaped([20, 1])
        let lfr = SenseVoiceAudio.applyLFR(feats, lfrM: 7, lfrN: 6)

        #expect(lfr.shape == [4, 7])
        let firstFrame = lfr[0].asArray(Float.self)
        #expect(firstFrame == [0, 0, 0, 0, 1, 2, 3])

        let cmvn = SenseVoiceAudio.applyCMVN(
            MLXArray([1.0, 2.0] as [Float]).reshaped([1, 2]),
            means: MLXArray([0.5, -1.0] as [Float]),
            istd: MLXArray([2.0, 4.0] as [Float])
        )
        #expect(cmvn.asArray(Float.self) == [3.0, 4.0])
    }

    @Test func parseAMMVNAndSentencePieceTokenizer() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sensevoice-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let mvn = """
        <AddShift> 80 80
        <LearnRateCoef> 1 [ -1.0 0.5 ]
        <Rescale> 80 80
        <LearnRateCoef> 1 [ 2.0 4.0 ]
        """
        try mvn.write(
            to: fixtureDir.appendingPathComponent("am.mvn"),
            atomically: true,
            encoding: .utf8
        )

        let modelData = makeSentencePieceModelData([
            ("<unk>", 0, 2),
            ("<s>", 0, 3),
            ("</s>", 0, 3),
            ("▁hello", -0.1, 1),
            ("▁world", -0.2, 1),
        ])
        try modelData.write(to: fixtureDir.appendingPathComponent("toy.model"))

        let parsed = try SenseVoiceAudio.parseAMMVN(fixtureDir.appendingPathComponent("am.mvn"))
        #expect(parsed.means == [-1.0, 0.5])
        #expect(parsed.istd == [2.0, 4.0])

        let tokenizer = try SenseVoiceTokenizer(modelDirectory: fixtureDir)
        #expect(tokenizer.tokenizer != nil)
        #expect(tokenizer.decode([3, 4]) == "hello world")
    }

    @Test func forwardShapeAndSanitize() {
        let config = SenseVoiceConfig(
            vocabSize: 32,
            inputSize: 560,
            encoderConf: SenseVoiceEncoderConfig(
                outputSize: 64,
                attentionHeads: 2,
                linearUnits: 128,
                numBlocks: 3,
                tpBlocks: 2,
                kernelSize: 5
            )
        )
        let model = SenseVoiceModel(config)

        let feats = MLXArray.zeros([1, 10, 560], type: Float.self)
        let logProbs = model(feats, language: "en")
        #expect(logProbs.shape == [1, 14, 32])

        let sanitized = SenseVoiceModel.sanitize(weights: [
            "ctc.ctc_lo.weight": MLXArray.zeros([32, 64], type: Float.self),
            "encoder.encoders.0.self_attn.fsmn_block.weight": MLXArray.zeros([64, 1, 5], type: Float.self),
        ])
        #expect(sanitized["ctc_lo.weight"]?.shape == [32, 64])
        #expect(sanitized["encoder.encoders.0.self_attn.fsmn_block.weight"]?.shape == [64, 5, 1])
    }

    @Test func fromDirectoryFixtureSmokeTest() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sensevoice-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = """
        {
          "model_type": "sensevoice",
          "vocab_size": 6,
          "input_size": 560,
          "encoder_conf": {
            "output_size": 8,
            "attention_heads": 2,
            "linear_units": 16,
            "num_blocks": 1,
            "tp_blocks": 0,
            "kernel_size": 5
          }
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let mvn = """
        <AddShift> 560 560
        <LearnRateCoef> 1 [ \(Array(repeating: "0.0", count: 560).joined(separator: " ")) ]
        <Rescale> 560 560
        <LearnRateCoef> 1 [ \(Array(repeating: "1.0", count: 560).joined(separator: " ")) ]
        """
        try mvn.write(
            to: fixtureDir.appendingPathComponent("am.mvn"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizerData = makeSentencePieceModelData([
            ("<unk>", 0, 2),
            ("<s>", 0, 3),
            ("</s>", 0, 3),
            ("▁a", -0.1, 1),
            ("▁b", -0.2, 1),
            ("▁c", -0.3, 1),
        ])
        try tokenizerData.write(to: fixtureDir.appendingPathComponent("toy.model"))

        let seedModel = SenseVoiceModel(try JSONDecoder().decode(
            SenseVoiceConfig.self,
            from: Data(configJSON.utf8)
        ))
        var weights = Dictionary(uniqueKeysWithValues: seedModel.parameters().flattened())
        if let ctcWeight = weights.removeValue(forKey: "ctc_lo.weight") {
            weights["ctc.ctc_lo.weight"] = ctcWeight
        }
        if let ctcBias = weights.removeValue(forKey: "ctc_lo.bias") {
            weights["ctc.ctc_lo.bias"] = ctcBias
        }
        for (key, value) in Array(weights) where key.contains("fsmn_block.weight") && value.ndim == 3 {
            weights[key] = value.transposed(0, 2, 1)
        }
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try SenseVoiceModel.fromDirectory(fixtureDir)
        let output = model.generate(
            audio: MLXArray(Array(repeating: Float(0), count: 16_000)),
            generationParameters: STTGenerateParameters(language: "en")
        )

        #expect(model.cmvnMeans != nil)
        #expect(model.cmvnIstd != nil)
        #expect(model.tokenizer?.tokenizer != nil)
        #expect(output.segments?.count == 1)
    }
}

@Suite("SenseVoice Network Tests", .serialized)
struct SenseVoiceNetworkTests {

    @Test func senseVoiceFromPretrainedLoadsRealWeightsAndTranscribesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network SenseVoice test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_SENSEVOICE_REPO"] ?? "mlx-community/SenseVoiceSmall"
        let model = try await SenseVoiceModel.fromPretrained(repo)
        let audio = try loadSTTNetworkFixture(sampleRate: 16000)
        let output = model.generate(
            audio: audio,
            generationParameters: STTGenerateParameters(verbose: false, language: "en")
        )

        #expect(model.config.modelType == "sensevoice")
        #expect(model.cmvnMeans != nil)
        #expect(model.cmvnIstd != nil)
        #expect(model.tokenizer?.tokenizer != nil)
        #expect(output.language == "en" || output.language == "unknown")
        #expect(!output.text.isEmpty)
    }
}

// MARK: - Granite Speech Tests

struct GraniteSpeechConfigTests {

    @Test func encoderConfigDefaults() throws {
        let json = """
        {}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(GraniteSpeechEncoderConfig.self, from: json)

        #expect(config.inputDim == 160)
        #expect(config.numLayers == 10)
        #expect(config.hiddenDim == 1024)
        #expect(config.feedforwardMult == 4)
        #expect(config.numHeads == 8)
        #expect(config.dimHead == 128)
        #expect(config.outputDim == 42)
        #expect(config.contextSize == 200)
        #expect(config.maxPosEmb == 512)
        #expect(config.convKernelSize == 15)
        #expect(config.convExpansionFactor == 2)
    }

    @Test func projectorConfigDefaults() throws {
        let json = """
        {}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(GraniteSpeechProjectorConfig.self, from: json)

        #expect(config.hiddenSize == 1024)
        #expect(config.numHiddenLayers == 2)
        #expect(config.numAttentionHeads == 16)
        #expect(config.intermediateSize == 4096)
        #expect(config.hiddenAct == "gelu")
        #expect(config.encoderHiddenSize == 1024)
    }

    @Test func textConfigDefaults() throws {
        let json = """
        {}
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(GraniteSpeechTextConfig.self, from: json)

        #expect(config.vocabSize == 100353)
        #expect(config.hiddenSize == 2048)
        #expect(config.numHiddenLayers == 40)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 4)
        #expect(config.attentionMultiplier == 0.0078125)
        #expect(config.embeddingMultiplier == 12.0)
        #expect(config.residualMultiplier == 0.22)
        #expect(config.logitsScaling == 8.0)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test func modelConfigParsing() throws {
        let json = """
        {
            "model_type": "granite_speech",
            "audio_token_index": 100352,
            "downsample_rate": 5,
            "window_size": 15,
            "encoder_config": {"input_dim": 160, "num_layers": 10},
            "projector_config": {"hidden_size": 1024},
            "text_config": {"vocab_size": 100353, "hidden_size": 2048}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(GraniteSpeechModelConfig.self, from: json)

        #expect(config.modelType == "granite_speech")
        #expect(config.audioTokenIndex == 100352)
        #expect(config.downsampleRate == 5)
        #expect(config.windowSize == 15)
        #expect(config.encoderConfig.inputDim == 160)
        #expect(config.encoderConfig.numLayers == 10)
        #expect(config.projectorConfig.hiddenSize == 1024)
        #expect(config.textConfig.vocabSize == 100353)
    }
}

struct GraniteSpeechModuleTests {

    @Test func ctcEncoderCreation() {
        let config = try! JSONDecoder().decode(
            GraniteSpeechEncoderConfig.self,
            from: "{}".data(using: .utf8)!
        )
        let encoder = GraniteSpeechCTCEncoder(config)
        #expect(encoder.numLayers == 10)

        // Verify forward pass with small input
        let input = MLXArray.zeros([1, 10, 160])
        let output = encoder(input)
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 10)
        #expect(output.shape[2] == 1024)
    }

    @Test func encoderProjectorCreation() throws {
        let json = """
        {
            "encoder_config": {},
            "projector_config": {"hidden_size": 1024},
            "text_config": {"hidden_size": 2048}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(GraniteSpeechModelConfig.self, from: json)
        let projector = GraniteSpeechEncoderProjector(config)

        #expect(projector.numQueries == 3)  // window_size(15) / downsample_rate(5)

        let input = MLXArray.zeros([1, 30, 1024])
        let output = projector(input)
        // 30 frames / window_size(15) = 2 blocks, 2 * 3 queries = 6 tokens
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 6)
        #expect(output.shape[2] == 2048)
    }


}
