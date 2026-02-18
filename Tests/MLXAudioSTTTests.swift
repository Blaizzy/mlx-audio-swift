//
//  MLXAudioSTTTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioSTT


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
<<<<<<< Updated upstream
=======
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

    @Test func fromDirectorySmokeTestWithFixtureConfigAndWeights() async throws {
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
        let audio = MLXArray(Array(repeating: Float(0), count: 3200))
        let output = model.generate(audio: audio)

        #expect(model.variant == .ctc)
        #expect(model.vocabulary.count == 4)
        #expect(output.text.count >= 0)
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
>>>>>>> Stashed changes
