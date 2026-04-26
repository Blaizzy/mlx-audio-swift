// Test OmniVoice model loading and generation
//
// Run with:
//   swift test --filter OmniVoiceTests
//   MLXAUDIO_ENABLE_NETWORK_TESTS=1 swift test --filter OmniVoiceTests

import Testing
import MLX
import Foundation
import AVFoundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXLMCommon

@Suite("OmniVoice Configuration Tests")
struct OmniVoiceConfigTests {

    @Test func testConfigDecodesFromJSON() throws {
        let json = """
        {
            "architectures": ["OmniVoice"],
            "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
            "audio_mask_id": 1024,
            "audio_vocab_size": 1025,
            "bos_token_id": null,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "llm_config": {
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": false,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "chunk_size_feed_forward": 0,
                "dtype": "float32",
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "is_encoder_decoder": false,
                "layer_types": ["full_attention"],
                "max_position_embeddings": 40960,
                "max_window_layers": 28,
                "model_type": "qwen3",
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                "tie_word_embeddings": true,
                "use_cache": true,
                "vocab_size": 151676
            },
            "model_type": "omnivoice",
            "num_audio_codebook": 8,
            "pad_token_id": 151643,
            "transformers_version": "5.3.0"
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: json)

        #expect(config.modelType == "omnivoice")
        #expect(config.architectures == ["OmniVoice"])
        #expect(config.audioVocabSize == 1025)
        #expect(config.audioMaskId == 1024)
        #expect(config.numAudioCodebook == 8)
        #expect(config.audioCodebookWeights == [8, 8, 6, 6, 4, 4, 2, 2])
        #expect(config.eosTokenId == 151645)
        #expect(config.padTokenId == 151643)
        #expect(config.sampleRate == 24000)

        // LLM config checks
        #expect(config.llmConfig.hiddenSize == 1024)
        #expect(config.llmConfig.numHiddenLayers == 28)
        #expect(config.llmConfig.numAttentionHeads == 16)
        #expect(config.llmConfig.numKeyValueHeads == 8)
        #expect(config.llmConfig.vocabSize == 151676)
        #expect(config.llmConfig.ropeTheta == 1_000_000)
    }

    @Test func testGenerateParametersDefaults() {
        let params = OmniVoiceGenerateParameters()

        #expect(params.numStep == 32)
        #expect(params.guidanceScale == 2.0)
        #expect(params.speed == 1.0)
        #expect(params.duration == nil)
        #expect(params.tShift == 0.1)
        #expect(params.denoise == true)
        #expect(params.postprocessOutput == true)
        #expect(params.layerPenaltyFactor == 5.0)
        #expect(params.positionTemperature == 5.0)
        #expect(params.classTemperature == 0.0)
    }

    @Test func testFastPreset() {
        let params = OmniVoiceGenerateParameters.fast

        #expect(params.numStep == 16)
        #expect(params.guidanceScale == 1.5)
    }

    @Test func testHighQualityPreset() {
        let params = OmniVoiceGenerateParameters.highQuality

        #expect(params.numStep == 64)
        #expect(params.guidanceScale == 2.5)
    }

    @Test func testCustomParameters() {
        let params = OmniVoiceGenerateParameters(
            numStep: 48,
            guidanceScale: 3.0,
            speed: 0.8,
            duration: 5.0,
            tShift: 0.2,
            denoise: false,
            postprocessOutput: false,
            layerPenaltyFactor: 3.0,
            positionTemperature: 3.0,
            classTemperature: 0.5
        )

        #expect(params.numStep == 48)
        #expect(params.guidanceScale == 3.0)
        #expect(params.speed == 0.8)
        #expect(params.duration == 5.0)
        #expect(params.tShift == 0.2)
        #expect(params.denoise == false)
        #expect(params.postprocessOutput == false)
        #expect(params.layerPenaltyFactor == 3.0)
        #expect(params.positionTemperature == 3.0)
        #expect(params.classTemperature == 0.5)
    }
}

@Suite("OmniVoice TTS Factory Tests")
struct OmniVoiceFactoryTests {

    @Test func testTTSResolveModelTypeOmniVoice() {
        let modelType = TTS.resolveModelType(modelRepo: "mlx-community/OmniVoice-bf16")
        #expect(modelType == "omnivoice")
    }

    @Test func testTTSResolveModelTypeOmniVoiceCaseInsensitive() {
        let modelType = TTS.resolveModelType(modelRepo: "mlx-community/omnivoice-bf16")
        #expect(modelType == "omnivoice")
    }

    @Test func testTTSResolveModelTypeOmniVoiceWithPrefix() {
        let modelType = TTS.resolveModelType(modelRepo: "k2-fsa/OmniVoice")
        #expect(modelType == "omnivoice")
    }
}

@Suite("OmniVoice Model Tests", .serialized)
struct OmniVoiceModelTests {

    @Test func testModelTypeRegistered() async throws {
        // Verify that OmniVoice is registered in the TTS factory
        // This doesn't load the model, just checks the factory
        let modelType = TTS.resolveModelType(modelRepo: "mlx-community/OmniVoice-bf16")
        #expect(modelType == "omnivoice")
    }

    @Test func testAutoVoiceGeneration() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("⚠️ Skipping network OmniVoice test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_OMNIVOICE_REPO"] ?? "mlx-community/OmniVoice-bf16"

        print("Loading OmniVoice model from \(repo)...")
        let model = try await TTS.loadModel(modelRepo: repo)

        #expect(model.sampleRate == 24000)
        print("Model loaded successfully with sample rate: \(model.sampleRate)")

        // Test auto voice generation (no ref_audio, no instruct)
        let text = "Hello, this is a test of OmniVoice text-to-speech."
        print("Generating audio for: \(text)")

        let generationParams = GenerateParameters(
            maxTokens: 2048,
            temperature: 1.0,
            topP: 0.95,
            repetitionPenalty: 1.05
        )

        let startTime = CFAbsoluteTimeGetCurrent()
        let audio = try await model.generate(
            text: text,
            voice: nil,
            refAudio: nil,
            refText: nil,
            language: "English",
            generationParameters: generationParams
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        // Verify output
        #expect(audio.shape[0] > 0, "Generated audio should have samples")
        print("Generated \(audio.shape[0]) samples in \(String(format: "%.2f", elapsed))s")

        // Calculate audio duration
        let duration = Double(audio.shape[0]) / Double(model.sampleRate)
        print("Audio duration: \(String(format: "%.2f", duration))s")
        #expect(duration > 0.5, "Audio should be at least 0.5 seconds")

        // Save to temp file for manual verification
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent("omnivoice_auto_voice_test.wav")
        try writeWavFile(samples: audio.asArray(Float.self), sampleRate: Double(model.sampleRate), outputURL: outputURL)
        print("Saved audio to: \(outputURL.path)")
    }

    @Test func testVoiceDesignGeneration() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("⚠️ Skipping network OmniVoice test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_OMNIVOICE_REPO"] ?? "mlx-community/OmniVoice-bf16"

        print("Loading OmniVoice model from \(repo)...")
        let model = try await TTS.loadModel(modelRepo: repo)

        // Test voice design generation with instruct
        let text = "Hello, this is a voice design test."
        let instruct = "male, British accent"
        print("Generating audio with instruct: \(instruct)")
        print("Text: \(text)")

        let startTime = CFAbsoluteTimeGetCurrent()
        let audio = try await (model as! OmniVoiceModel).generate(
            text: text,
            voice: instruct,
            ovParameters: OmniVoiceGenerateParameters(
                numStep: 32,
                guidanceScale: 2.0,
                speed: 1.0,
                tShift: 0.1,
                denoise: true,
                postprocessOutput: true
            )
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        #expect(audio.shape[0] > 0, "Generated audio should have samples")
        let duration = Double(audio.shape[0]) / Double(model.sampleRate)
        print("Generated \(String(format: "%.2f", duration))s of audio in \(String(format: "%.2f", elapsed))s")

        // Save to temp file
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent("omnivoice_voice_design_test.wav")
        try writeWavFile(samples: audio.asArray(Float.self), sampleRate: Double(model.sampleRate), outputURL: outputURL)
        print("Saved audio to: \(outputURL.path)")
    }

    @Test func testVoiceCloningGeneration() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("⚠️ Skipping network OmniVoice test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_OMNIVOICE_REPO"] ?? "mlx-community/OmniVoice-bf16"

        print("Loading OmniVoice model from \(repo)...")
        let model = try await TTS.loadModel(modelRepo: repo)

        // Load reference audio from test media
        let bundle = Bundle.module
        guard let audioURL = bundle.url(forResource: "intention", withExtension: "wav") else {
            print("⚠️ Test audio file not found, skipping test")
            return
        }

        let (refSampleRate, refAudio) = try loadAudioArray(from: audioURL, sampleRate: model.sampleRate)
        print("Loaded reference audio: \(refAudio.shape[0]) samples at \(refSampleRate)Hz")

        let text = "This is a voice cloning test."
        let refText = "intention"  // Simple reference text
        print("Generating with voice cloning...")
        print("Text: \(text)")
        print("Ref text: \(refText)")

        let startTime = CFAbsoluteTimeGetCurrent()
        let audio = try await (model as! OmniVoiceModel).generate(
            text: text,
            refAudio: refAudio,
            refText: refText,
            ovParameters: OmniVoiceGenerateParameters(
                numStep: 32,
                guidanceScale: 2.0,
                speed: 1.0
            )
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        #expect(audio.shape[0] > 0, "Generated audio should have samples")
        let duration = Double(audio.shape[0]) / Double(model.sampleRate)
        print("Generated \(String(format: "%.2f", duration))s of audio in \(String(format: "%.2f", elapsed))s")

        // Save to temp file
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent("omnivoice_voice_cloning_test.wav")
        try writeWavFile(samples: audio.asArray(Float.self), sampleRate: Double(model.sampleRate), outputURL: outputURL)
        print("Saved audio to: \(outputURL.path)")
    }

    @Test func testAudioTokenizerRoundTrip() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("⚠️ Skipping network OmniVoice test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_OMNIVOICE_REPO"] ?? "mlx-community/OmniVoice-bf16"
        print("Loading OmniVoice model from \(repo)...")
        let model = try await TTS.loadModel(modelRepo: repo)
        guard let audioTokenizer = (model as? OmniVoiceModel)?.audioTokenizer else {
            print("⚠️ Audio tokenizer not available")
            return
        }

        let bundle = Bundle.module
        guard let audioURL = bundle.url(forResource: "intention", withExtension: "wav") else {
            print("⚠️ Test audio file not found, skipping test")
            return
        }

        let (refSampleRate, refAudio) = try loadAudioArray(from: audioURL, sampleRate: model.sampleRate)
        print("Loaded reference audio: \(refAudio.shape[0]) samples at \(refSampleRate)Hz")

        // Encode -> Decode roundtrip to isolate vocoder issues
        let tokens = try audioTokenizer.encode(refAudio)
        print("Encoded tokens shape: \(tokens.shape)")
        let tokenMin = tokens.min().item(Int32.self)
        let tokenMax = tokens.max().item(Int32.self)
        print("Token range: min=\(tokenMin), max=\(tokenMax)")

        let decodedAudio = try audioTokenizer.decode(tokens)
        print("Decoded audio shape: \(decodedAudio.shape)")

        let tempDir = FileManager.default.temporaryDirectory
        let originalURL = tempDir.appendingPathComponent("omnivoice_roundtrip_original.wav")
        let decodedURL = tempDir.appendingPathComponent("omnivoice_roundtrip_decoded.wav")
        try writeWavFile(samples: refAudio.asArray(Float.self), sampleRate: Double(model.sampleRate), outputURL: originalURL)
        try writeWavFile(samples: decodedAudio.asArray(Float.self), sampleRate: Double(model.sampleRate), outputURL: decodedURL)
        print("Saved original to: \(originalURL.path)")
        print("Saved decoded to: \(decodedURL.path)")

        // Simple numeric sanity check: correlation should be positive if the codec is working
        let originalSamples = refAudio.asArray(Float.self)
        let decodedSamples = decodedAudio.asArray(Float.self)
        let minLen = min(originalSamples.count, decodedSamples.count)
        var dot: Float = 0
        var origNorm: Float = 0
        var decNorm: Float = 0
        for i in 0..<minLen {
            dot += originalSamples[i] * decodedSamples[i]
            origNorm += originalSamples[i] * originalSamples[i]
            decNorm += decodedSamples[i] * decodedSamples[i]
        }
        let correlation = dot / (sqrt(origNorm) * sqrt(decNorm) + 1e-8)
        print("Correlation between original and decoded: \(correlation)")
        #expect(correlation > 0.1, "Decoded audio should correlate with original; low correlation indicates vocoder bug")
    }

    @Test func testStreamingGeneration() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("⚠️ Skipping network OmniVoice test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_OMNIVOICE_REPO"] ?? "mlx-community/OmniVoice-bf16"

        print("Loading OmniVoice model from \(repo)...")
        let model = try await TTS.loadModel(modelRepo: repo)

        let text = "Testing streaming generation."
        print("Streaming generation for: \(text)")

        let generationParams = GenerateParameters(
            maxTokens: 2048,
            temperature: 1.0,
            topP: 0.95
        )

        var totalSamples = 0
        var chunkCount = 0
        let startTime = CFAbsoluteTimeGetCurrent()

        for try await event in model.generateStream(
            text: text,
            voice: nil,
            refAudio: nil,
            refText: nil,
            language: "English",
            generationParameters: generationParams,
            streamingInterval: 0.5
        ) {
            switch event {
            case .token(let tokenId):
                chunkCount += 1
                if chunkCount % 10 == 0 {
                    print("Generated \(chunkCount) tokens so far")
                }
            case .info(let info):
                print("Generation info: \(info.summary)")
            case .audio(let chunk):
                let samples = chunk.asArray(Float.self)
                totalSamples += samples.count
                print("Received audio chunk: \(samples.count) samples (total: \(totalSamples))")
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("Streaming completed: \(totalSamples) samples in \(String(format: "%.2f", elapsed))s")

        #expect(totalSamples > 0, "Should have received audio samples")

        // Save streamed audio
        if totalSamples > 0 {
            let tempDir = FileManager.default.temporaryDirectory
            let outputURL = tempDir.appendingPathComponent("omnivoice_streaming_test.wav")
            // Note: In real test, we'd collect all chunks and save
            print("Streaming test passed with \(totalSamples) samples")
        }
    }
}

// MARK: - Helper Functions

private func writeWavFile(samples: [Float], sampleRate: Double, outputURL: URL) throws {
    let frameCount = AVAudioFrameCount(samples.count)
    guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
          let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
          let channelData = buffer.floatChannelData else {
        throw NSError(domain: "WavWrite", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    buffer.frameLength = frameCount
    for i in 0..<samples.count {
        channelData[0][i] = samples[i]
    }

    let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
    try audioFile.write(from: buffer)
}
