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

    @Test func whisperConfigDefaults() async throws {
        try await withCPU {
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
    }

    @Test func whisperConfigCustom() async throws {
        try await withCPU {
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
    }

    @Test func llamaConfigDefaults() async throws {
        try await withCPU {
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
    }

    @Test func llamaConfigCustom() async throws {
        try await withCPU {
                    let config = LlamaConfig(
                        vocabSize: 32000,
                        hiddenSize: 1024,
                        numHiddenLayers: 12
                    )

                    #expect(config.vocabSize == 32000)
                    #expect(config.hiddenSize == 1024)
                    #expect(config.numHiddenLayers == 12)
        }
    }

    @Test func glmASRModelConfigDefaults() async throws {
        try await withCPU {
                    let config = GLMASRModelConfig()

                    #expect(config.modelType == "glmasr")
                    #expect(config.adapterType == "mlp")
                    #expect(config.mergeFactor == 4)
                    #expect(config.useRope)
                    #expect(config.maxWhisperLength == 1500)
        }
    }

    @Test func glmASRModelConfigWithNestedConfigs() async throws {
        try await withCPU {
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
    }

    // MARK: - Layer Tests

    @Test func whisperAttentionShape() async throws {
        try await withCPU {
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
    }

    @Test func whisperAttentionWithRoPE() async throws {
        try await withCPU {
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
    }

    @Test func whisperEncoderLayerShape() async throws {
        try await withCPU {
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
    }

    @Test func whisperEncoderShape() async throws {
        try await withCPU {
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
    }

    @Test func adaptingMLPShape() async throws {
        try await withCPU {
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
    }

    @Test func audioEncoderShape() async throws {
        try await withCPU {
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
    }

    @Test func audioEncoderBoaEoaTokens() async throws {
        try await withCPU {
                    let whisperConfig = WhisperConfig(dModel: 256, encoderAttentionHeads: 4, encoderLayers: 1)
                    let llamaConfig = LlamaConfig(hiddenSize: 512)
                    let config = GLMASRModelConfig(whisperConfig: whisperConfig, lmConfig: llamaConfig)

                    let audioEncoder = AudioEncoder(config: config)

                    let (boa, eoa) = audioEncoder.getBoaEoaTokens()

                    #expect(boa.shape == [1, llamaConfig.hiddenSize])
                    #expect(eoa.shape == [1, llamaConfig.hiddenSize])
        }
    }

    // MARK: - STTOutput Tests

    @Test func sttOutputCreation() async throws {
        try await withCPU {
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
    }

    @Test func sttOutputDefaults() async throws {
        try await withCPU {
                    let output = STTOutput(text: "Test")

                    #expect(output.text == "Test")
                    #expect(output.segments == nil)
                    #expect(output.language == nil)
                    #expect(output.promptTokens == 0)
                    #expect(output.generationTokens == 0)
                    #expect(output.totalTokens == 0)
        }
    }

    @Test func sttOutputDescription() async throws {
        try await withCPU {
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
    }

    // MARK: - Config Decoding Tests

    @Test func whisperConfigDecoding() async throws {
        try await withCPU {
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
    }

    @Test func llamaConfigDecoding() async throws {
        try await withCPU {
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
    }

    @Test func glmASRModelConfigDecoding() async throws {
        try await withCPU {
                    let json = """
                    {
                        "model_type": "glmasr",
                        "adapter_type": "mlp",
                        "merge_factor": 2,
                        "use_rope": true,
                        "audio_config": {
                            "d_model": 512,
                            "encoder_layers": 6
                        },
                        "text_config": {
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
    }

    // MARK: - AnyCodable Tests

    @Test func anyCodableWithInt() async throws {
        try await withCPU {
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
    }

    @Test func anyCodableWithString() async throws {
        try await withCPU {
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
    }

    @Test func anyCodableWithArray() async throws {
        try await withCPU {
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
}


// Run GLMASR tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/GLMASRTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run|Transcription)"


struct GLMASRTests {

    /// Test basic transcription with GLM-ASR model
    @Test func glmASRTranscribe() async throws {
        try await withCPU {
                    let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
                    let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
                    print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

                    print("\u{001B}[33mLoading GLMASR model...\u{001B}[0m")
                    let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
                    print("\u{001B}[32mGLMASR model loaded!\u{001B}[0m")

                    let output = model.generate(audio: audioData)
                    print("\u{001B}[32m GLMASR Transcription: \(output.text)\u{001B}[0m")
                    print("\u{001B}[32m GLMASR Generation Stats: \(output)\u{001B}[0m")

                    #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
        }
    }

    /// Test streaming transcription with GLM-ASR model
    @Test func glmASRTranscribeStream() async throws {
        try await withCPU {
                    let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
                    let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
                    print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

                    print("\u{001B}[33mLoading GLMASR model...\u{001B}[0m")
                    let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
                    print("\u{001B}[32mGLMASR model loaded!\u{001B}[0m")

                    print("\u{001B}[33mStreaming transcription ...\u{001B}[0m")

                    var tokenCount = 0
                    var transcribedText = ""
                    var finalOutput: STTOutput?
                    var generationInfo: STTGenerationInfo?

                    for try await event in model.generateStream(audio: audioData) {
                        switch event {
                        case .token(let token):
                            tokenCount += 1
                            transcribedText += token
                        case .info(let info):
                            generationInfo = info
                            print("\n\u{001B}[36m\(info.summary)\u{001B}[0m")
                        case .result(let output):
                            finalOutput = output
                            print("\u{001B}[32m GLMASR Streaming Transcription: \(output.text)\u{001B}[0m")
                            print("\u{001B}[32m GLMASR Streaming Stats: \(output)\u{001B}[0m")
                        }
                    }

                    #expect(tokenCount > 0, "Should have generated tokens")
                    #expect(finalOutput != nil, "Should have received final output")
                    #expect(generationInfo != nil, "Should have received generation info")

                    if let output = finalOutput {
                        #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
                        print("\u{001B}[32m\(output)\u{001B}[0m")
                    }
        }
    }
}
