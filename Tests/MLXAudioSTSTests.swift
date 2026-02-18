//
//  MLXAudioSTSTests.swift
//  MLXAudioTests
//
//  Created by Claude on 17/02/2026.
//

import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCore
@testable import MLXAudioSTS


// Run config tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:'MLXAudioTests/LFMAudioConfigTests' \
// 2>&1 | grep -E "(Suite.*started|Test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED)"

struct LFMAudioConfigTests {

    // MARK: - PreprocessorConfig

    @Test func preprocessorConfigDefaults() {
        let config = PreprocessorConfig()

        #expect(config.sampleRate == 16000)
        #expect(config.normalize == "per_feature")
        #expect(config.windowSize == 0.025)
        #expect(config.windowStride == 0.01)
        #expect(config.window == "hann")
        #expect(config.features == 128)
        #expect(config.nFft == 512)
        #expect(config.log == true)
        #expect(config.frameSplicing == 1)
        #expect(config.dither == 1e-05)
        #expect(config.padTo == 0)
        #expect(config.padValue == 0.0)
        #expect(config.preemph == 0.97)
    }

    @Test func preprocessorConfigComputedProperties() {
        let config = PreprocessorConfig()

        // hopLength = Int(16000 * 0.01) = 160
        #expect(config.hopLength == 160)
        // winLength = Int(16000 * 0.025) = 400
        #expect(config.winLength == 400)
    }

    @Test func preprocessorConfigDecoding() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(PreprocessorConfig.self, from: data)

        #expect(config.sampleRate == 16000)
        #expect(config.features == 128)
        #expect(config.nFft == 512)
        #expect(config.preemph == 0.97)
    }

    // MARK: - ConformerEncoderConfig

    @Test func conformerEncoderConfigDefaults() {
        let config = ConformerEncoderConfig()

        #expect(config.featIn == 128)
        #expect(config.featOut == -1)
        #expect(config.nLayers == 17)
        #expect(config.dModel == 512)
        #expect(config.subsampling == "dw_striding")
        #expect(config.subsamplingFactor == 8)
        #expect(config.subsamplingConvChannels == 256)
        #expect(config.causalDownsampling == false)
        #expect(config.ffExpansionFactor == 4)
        #expect(config.selfAttentionModel == "rel_pos")
        #expect(config.nHeads == 8)
        #expect(config.attContextSize == [-1, -1])
        #expect(config.xscaling == false)
        #expect(config.untieBiases == true)
        #expect(config.posEmbMaxLen == 5000)
        #expect(config.convKernelSize == 9)
        #expect(config.convNormType == "batch_norm")
        #expect(config.dropout == 0.1)
    }

    @Test func conformerEncoderConfigDecoding() throws {
        let json = """
        {
            "feat_in": 80,
            "n_layers": 12,
            "d_model": 256,
            "n_heads": 4
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ConformerEncoderConfig.self, from: data)

        #expect(config.featIn == 80)
        #expect(config.nLayers == 12)
        #expect(config.dModel == 256)
        #expect(config.nHeads == 4)
        // Defaults for unspecified fields
        #expect(config.subsamplingFactor == 8)
        #expect(config.convKernelSize == 9)
    }

    // MARK: - DepthformerConfig

    @Test func depthformerConfigDefaults() {
        let config = DepthformerConfig()

        #expect(config.layers == 6)
        #expect(config.dim == 1024)
        #expect(config.numHeads == 32)
        #expect(config.numKvHeads == 8)
        #expect(config.tie == true)
    }

    @Test func depthformerConfigDecoding() throws {
        let json = """
        {
            "layers": 4,
            "dim": 512,
            "num_heads": 16,
            "num_kv_heads": 4,
            "tie": false
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(DepthformerConfig.self, from: data)

        #expect(config.layers == 4)
        #expect(config.dim == 512)
        #expect(config.numHeads == 16)
        #expect(config.numKvHeads == 4)
        #expect(config.tie == false)
    }

    // MARK: - DetokenizerConfig

    @Test func detokenizerConfigDefaults() {
        let config = DetokenizerConfig()

        #expect(config.hiddenSize == 512)
        #expect(config.numHiddenLayers == 8)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.slidingWindow == 30)
        #expect(config.intermediateSize == 2304)
        #expect(config.normEps == 1e-5)
        #expect(config.ropeTheta == 1000000.0)
        #expect(config.outputSize == 1282)
        #expect(config.numCodebooks == 8)
        #expect(config.vocabSize == 2048)
        #expect(config.nFft == 1280)
        #expect(config.hopLength == 320)
        #expect(config.upsampleFactor == 6)
    }

    @Test func detokenizerConfigLayerTypes() {
        let config = DetokenizerConfig()

        #expect(config.layerTypes.count == 8)
        // Pattern: conv, conv, sliding_attention, conv, sliding_attention, conv, sliding_attention, conv
        #expect(config.layerTypes[0] == "conv")
        #expect(config.layerTypes[2] == "sliding_attention")
        #expect(config.layerTypes[7] == "conv")
    }

    // MARK: - LFMGenerationConfig

    @Test func generationConfigDefaults() {
        let config = LFMGenerationConfig()

        #expect(config.maxNewTokens == 512)
        #expect(config.temperature == 1.0)
        #expect(config.topK == 50)
        #expect(config.topP == 1.0)
        #expect(config.audioTemperature == 1.0)
        #expect(config.audioTopK == 4)
    }

    @Test func generationConfigCustom() {
        let config = LFMGenerationConfig(
            maxNewTokens: 2048,
            temperature: 0.8,
            topK: 30,
            audioTemperature: 0.7,
            audioTopK: 10
        )

        #expect(config.maxNewTokens == 2048)
        #expect(config.temperature == 0.8)
        #expect(config.topK == 30)
        #expect(config.audioTemperature == 0.7)
        #expect(config.audioTopK == 10)
    }
}


// MARK: - Module Setup Tests

struct LFMAudioModuleSetupTests {

    @Test func modalityConstants() {
        #expect(LFMModality.text.rawValue == 1)
        #expect(LFMModality.audioIn.rawValue == 2)
        #expect(LFMModality.audioOut.rawValue == 3)
    }

    @Test func specialTokenConstants() {
        #expect(lfmAudioStartToken == 128)
        #expect(lfmImEndToken == 7)
        #expect(lfmTextEndToken == 130)
        #expect(lfmAudioEOSToken == 2048)
    }

    @Test func audioEmbeddingShape() {
        let vocabSize = 2049
        let dim = 64
        let numCodebooks = 8
        let emb = AudioEmbedding(vocabSize: vocabSize, dim: dim, numCodebooks: numCodebooks)

        // Input: (B, K) where K = numCodebooks, values in [0, vocabSize)
        let input = MLXArray([0, 1, 2, 3, 4, 5, 6, 7]).expandedDimensions(axis: 0) // (1, 8)
        let output = emb(input)

        // Output should be (1, dim) after summing over codebooks
        #expect(output.shape == [1, dim])
    }

    @Test func audioEmbeddingWithNormShape() {
        let vocabSize = 2049
        let dim = 64
        let emb = AudioEmbeddingWithNorm(vocabSize: vocabSize, dim: dim)

        // embed: (B,) -> (B, dim)
        let input = MLXArray([Int32(42)]).expandedDimensions(axis: 0) // (1, 1)
        let embedded = emb.embed(input.squeezed(axis: 1))
        #expect(embedded.shape == [1, dim])

        // logits: (B, dim) -> (B, vocabSize)
        let hidden = MLXArray.zeros([1, dim])
        let logits = emb.logits(hidden)
        #expect(logits.shape == [1, vocabSize])
    }

    @Test func conformerEncoderConstruction() {
        let config = ConformerEncoderConfig()
        let encoder = ConformerEncoder(config)

        // Verify the encoder was constructed (it has layers)
        #expect(encoder.layers.count == config.nLayers)
    }

    @Test func depthformerConstruction() {
        let config = DepthformerConfig()
        let depthformer = Depthformer(
            layers: config.layers, dim: config.dim,
            numHeads: config.numHeads, numKvHeads: config.numKvHeads
        )

        #expect(depthformer.layersCount == config.layers)
    }
}


// MARK: - Inference Tests

// Run inference tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:'MLXAudioTests/LFMAudioInferenceTests' \
// 2>&1 | grep -E "(Suite.*started|Test.*started|Loading|Loaded|Generated|Text|Audio|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

@Suite(.serialized)
struct LFMAudioInferenceTests {

    static let modelName = "mlx-community/LFM2.5-Audio-1.5B-bf16"

    // MARK: - Text-to-Text

    @Test func testTextToText() async throws {
        print("\u{001B}[33mLoading LFM2.5-Audio model...\u{001B}[0m")
        let model = try await LFM2AudioModel.fromPretrained(Self.modelName)
        let processor = model.processor!
        print("\u{001B}[32mModel loaded!\u{001B}[0m")

        // Build chat: system → user (text question) → assistant
        let chat = ChatState(processor: processor)
        chat.newTurn(role: "system")
        chat.addText("Answer briefly in one sentence.")
        chat.endTurn()
        chat.newTurn(role: "user")
        chat.addText("What is 2 + 2?")
        chat.endTurn()
        chat.newTurn(role: "assistant")

        let genConfig = LFMGenerationConfig(
            maxNewTokens: 64,
            temperature: 0.8,
            topK: 50
        )

        print("\u{001B}[33mGenerating text-to-text response...\u{001B}[0m")

        var textTokens: [Int] = []
        for try await (token, modality) in model.generateInterleaved(
            textTokens: chat.getTextTokens(),
            audioFeatures: chat.getAudioFeatures(),
            modalities: chat.getModalities(),
            config: genConfig
        ) {
            eval(token)
            if modality == .text {
                textTokens.append(token.item(Int.self))
            }
        }

        let decodedText = processor.decodeText(textTokens)
        print("\u{001B}[32mText-to-Text output: \(decodedText)\u{001B}[0m")
        print("\u{001B}[32mGenerated \(textTokens.count) text tokens\u{001B}[0m")

        #expect(textTokens.count > 0, "Should generate at least one text token")
        #expect(!decodedText.isEmpty, "Decoded text should not be empty")
    }

    // MARK: - Text-to-Speech

    @Test func testTextToSpeech() async throws {
        print("\u{001B}[33mLoading LFM2.5-Audio model...\u{001B}[0m")
        let model = try await LFM2AudioModel.fromPretrained(Self.modelName)
        let processor = model.processor!
        print("\u{001B}[32mModel loaded!\u{001B}[0m")

        // Build chat: system (TTS instruction) → user (text) → assistant + audio_start
        let chat = ChatState(processor: processor)
        chat.newTurn(role: "system")
        chat.addText("Perform TTS. Use a UK male voice.")
        chat.endTurn()
        chat.newTurn(role: "user")
        chat.addText("Hello, welcome to MLX Audio!")
        chat.endTurn()
        chat.newTurn(role: "assistant")
        chat.addAudioStartToken()

        let genConfig = LFMGenerationConfig(
            maxNewTokens: 256,
            temperature: 0.8,
            topK: 50,
            audioTemperature: 0.7,
            audioTopK: 30
        )

        print("\u{001B}[33mGenerating text-to-speech response...\u{001B}[0m")

        var audioCodes: [MLXArray] = []
        for try await (token, modality) in model.generateSequential(
            textTokens: chat.getTextTokens(),
            audioFeatures: chat.getAudioFeatures(),
            modalities: chat.getModalities(),
            config: genConfig
        ) {
            eval(token)
            if modality == .audioOut {
                if token[0].item(Int.self) == lfmAudioEOSToken {
                    break
                }
                audioCodes.append(token)
            }
        }

        print("\u{001B}[32mText-to-Speech: generated \(audioCodes.count) audio frames\u{001B}[0m")

        #expect(audioCodes.count > 0, "Should generate at least one audio frame")

        // Each audio frame should have shape (8,) for 8 codebooks
        if let firstFrame = audioCodes.first {
            #expect(firstFrame.shape == [8], "Audio frame should have 8 codebook values")
        }
    }

    // MARK: - Speech-to-Text

    @Test func testSpeechToText() async throws {
        // Load test audio
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading LFM2.5-Audio model...\u{001B}[0m")
        let model = try await LFM2AudioModel.fromPretrained(Self.modelName)
        let processor = model.processor!
        print("\u{001B}[32mModel loaded!\u{001B}[0m")

        // Build chat: user (audio + transcription request) → assistant
        let chat = ChatState(processor: processor)
        chat.newTurn(role: "user")
        chat.addAudio(audioData, sampleRate: sampleRate)
        chat.addText("Transcribe the audio.")
        chat.endTurn()
        chat.newTurn(role: "assistant")

        let genConfig = LFMGenerationConfig(
            maxNewTokens: 256,
            temperature: 0.8,
            topK: 50
        )

        print("\u{001B}[33mGenerating speech-to-text response...\u{001B}[0m")

        var textTokens: [Int] = []
        for try await (token, modality) in model.generateInterleaved(
            textTokens: chat.getTextTokens(),
            audioFeatures: chat.getAudioFeatures(),
            modalities: chat.getModalities(),
            config: genConfig
        ) {
            eval(token)
            if modality == .text {
                textTokens.append(token.item(Int.self))
                print(processor.decodeText([token.item(Int.self)]), terminator: "")
            }
        }

        let decodedText = processor.decodeText(textTokens)
        print("\n\u{001B}[32mSpeech-to-Text transcription: \(decodedText)\u{001B}[0m")
        print("\u{001B}[32mGenerated \(textTokens.count) text tokens\u{001B}[0m")

        #expect(textTokens.count > 0, "Should generate at least one text token")
        #expect(!decodedText.isEmpty, "Transcription should not be empty")
    }

    // MARK: - Speech-to-Speech

    @Test func testSpeechToSpeech() async throws {
        // Load test audio
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading LFM2.5-Audio model...\u{001B}[0m")
        let model = try await LFM2AudioModel.fromPretrained(Self.modelName)
        let processor = model.processor!
        print("\u{001B}[32mModel loaded!\u{001B}[0m")

        // Build chat: system → user (audio) → assistant
        let chat = ChatState(processor: processor)
        chat.newTurn(role: "system")
        chat.addText("Respond with interleaved text and audio.")
        chat.endTurn()
        chat.newTurn(role: "user")
        chat.addAudio(audioData, sampleRate: sampleRate)
        chat.endTurn()
        chat.newTurn(role: "assistant")

        let genConfig = LFMGenerationConfig(
            maxNewTokens: 512,
            temperature: 0.8,
            topK: 50,
            audioTemperature: 0.7,
            audioTopK: 30
        )

        print("\u{001B}[33mGenerating speech-to-speech response...\u{001B}[0m")

        var textTokens: [Int] = []
        var audioCodes: [MLXArray] = []
        for try await (token, modality) in model.generateInterleaved(
            textTokens: chat.getTextTokens(),
            audioFeatures: chat.getAudioFeatures(),
            modalities: chat.getModalities(),
            config: genConfig
        ) {
            eval(token)
            if modality == .text {
                textTokens.append(token.item(Int.self))
            } else if modality == .audioOut {
                audioCodes.append(token)
            }
        }

        let decodedText = processor.decodeText(textTokens)
        print("\u{001B}[32mSpeech-to-Speech text: \(decodedText)\u{001B}[0m")
        print("\u{001B}[32mGenerated \(textTokens.count) text tokens, \(audioCodes.count) audio frames\u{001B}[0m")

        // The model should generate at least some tokens (text or audio)
        let totalTokens = textTokens.count + audioCodes.count
        #expect(totalTokens > 0, "Should generate at least one token (text or audio)")
    }

    // MARK: - TTS with Audio Decode & WAV Save

    @Test func testTTSDecodeAndSave() async throws {
        let model = try await LFM2AudioModel.fromPretrained(Self.modelName)
        let processor = model.processor!

        // Build TTS chat
        let chat = ChatState(processor: processor)
        chat.newTurn(role: "system")
        chat.addText("Perform TTS. Use a UK male voice.")
        chat.endTurn()
        chat.newTurn(role: "user")
        chat.addText("Hello, welcome to MLX Audio Swift! This is a test of the text to speech system.")
        chat.endTurn()
        chat.newTurn(role: "assistant")
        chat.addAudioStartToken()

        let genConfig = LFMGenerationConfig(
            maxNewTokens: 512,
            temperature: 0.8,
            topK: 50,
            audioTemperature: 0.7,
            audioTopK: 30
        )

        print("\u{001B}[33mGenerating TTS audio codes...\u{001B}[0m")

        var audioCodes: [MLXArray] = []
        for try await (token, modality) in model.generateSequential(
            textTokens: chat.getTextTokens(),
            audioFeatures: chat.getAudioFeatures(),
            modalities: chat.getModalities(),
            config: genConfig
        ) {
            eval(token)
            if modality == .audioOut {
                if token[0].item(Int.self) == lfmAudioEOSToken {
                    break
                }
                audioCodes.append(token)
            }
        }

        print("\u{001B}[32mGenerated \(audioCodes.count) audio frames\u{001B}[0m")
        #expect(audioCodes.count > 0)

        // Stack codes into (1, 8, T) for detokenizer
        let stacked = MLX.stacked(audioCodes, axis: 0) // (T, 8)
        let codesInput = stacked.transposed(1, 0).expandedDimensions(axis: 0) // (1, 8, T)
        eval(codesInput)
        print("\u{001B}[33mDecoding audio codes \(codesInput.shape)...\u{001B}[0m")

       
        // Load detokenizer
        let detokenizer = try LFM2AudioDetokenizer.fromPretrained(modelPath: model.modelDirectory!)

        // Decode codes → waveform
        let waveform = detokenizer(codesInput) // (1, samples)
        eval(waveform)
        let samples = waveform[0].asArray(Float.self)
        print("\u{001B}[32mDecoded \(samples.count) audio samples (\(String(format: "%.1f", Double(samples.count) / 24000.0))s at 24kHz)\u{001B}[0m")

        // Save to Desktop
        let outputURL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Desktop/lfm_tts_output.wav")
        try AudioUtils.writeWavFile(samples: samples, sampleRate: 24000, fileURL: outputURL)
        print("\u{001B}[32mSaved WAV to: \(outputURL.path)\u{001B}[0m")
    }


}
