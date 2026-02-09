//
//  MLXAudioTTSTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 31/12/2025.
//

import Testing
import MLX
import MLXLMCommon
import Foundation
import AVFoundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs


// Run Qwen3 tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


// MARK: - Qwen3-TTS Speech Tokenizer Unit Tests (no model download required)

// Run Qwen3TTSSpeechTokenizerTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSSpeechTokenizerTests {

    /// Test that hasEncoder defaults to false when no encoder is loaded
    @Test func testHasEncoderDefaultsFalse() throws {
        // Create a minimal tokenizer config from empty JSON (all defaults)
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: json)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        #expect(tokenizer.hasEncoder == false, "hasEncoder should default to false when no encoder is loaded")
    }

    /// Test that hasEncoder can be set to true (simulating encoder load)
    @Test func testHasEncoderCanBeSetTrue() throws {
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: json)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        // Simulate what Task 7 will do when the encoder is loaded
        tokenizer.hasEncoder = true
        #expect(tokenizer.hasEncoder == true, "hasEncoder should be true after encoder is loaded")
    }
}


// MARK: - Qwen3-TTS Language Resolution Unit Tests (no model download required)

// Run Qwen3TTSLanguageTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSLanguageTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSLanguageTests {

    /// Test ISO 639-1 code "en" resolves to "english" without config
    @Test func testResolveLanguageEnglishISO() {
        let result = Qwen3TTSModel.resolveLanguage("en")
        #expect(result == "english", "ISO code 'en' should resolve to 'english'")
    }

    /// Test ISO 639-1 code "zh" resolves to "chinese" without config
    @Test func testResolveLanguageChineseISO() {
        let result = Qwen3TTSModel.resolveLanguage("zh")
        #expect(result == "chinese", "ISO code 'zh' should resolve to 'chinese'")
    }

    /// Test ISO 639-1 code "ja" resolves to "japanese" without config
    @Test func testResolveLanguageJapaneseISO() {
        let result = Qwen3TTSModel.resolveLanguage("ja")
        #expect(result == "japanese", "ISO code 'ja' should resolve to 'japanese'")
    }

    /// Test ISO 639-1 code "ko" resolves to "korean" without config
    @Test func testResolveLanguageKoreanISO() {
        let result = Qwen3TTSModel.resolveLanguage("ko")
        #expect(result == "korean", "ISO code 'ko' should resolve to 'korean'")
    }

    /// Test all supported ISO 639-1 codes resolve correctly
    @Test func testResolveLanguageAllISO() {
        let expected: [String: String] = [
            "en": "english",
            "zh": "chinese",
            "ja": "japanese",
            "ko": "korean",
            "de": "german",
            "fr": "french",
            "ru": "russian",
            "pt": "portuguese",
            "es": "spanish",
            "it": "italian",
        ]
        for (iso, name) in expected {
            let result = Qwen3TTSModel.resolveLanguage(iso)
            #expect(result == name, "ISO code '\(iso)' should resolve to '\(name)', got '\(result ?? "nil")'")
        }
    }

    /// Test full language name "english" passes through without config
    @Test func testResolveLanguageFullNamePassthrough() {
        let result = Qwen3TTSModel.resolveLanguage("english")
        #expect(result == "english", "Full language name 'english' should pass through")
    }

    /// Test full language name "chinese" passes through without config
    @Test func testResolveLanguageChineseFullName() {
        let result = Qwen3TTSModel.resolveLanguage("chinese")
        #expect(result == "chinese", "Full language name 'chinese' should pass through")
    }

    /// Test "auto" passes through as a special value
    @Test func testResolveLanguageAuto() {
        let result = Qwen3TTSModel.resolveLanguage("auto")
        #expect(result == "auto", "'auto' should pass through unchanged")
    }

    /// Test "Auto" (mixed case) passes through
    @Test func testResolveLanguageAutoMixedCase() {
        let result = Qwen3TTSModel.resolveLanguage("Auto")
        #expect(result == "auto", "'Auto' (mixed case) should resolve to 'auto'")
    }

    /// Test unsupported code returns nil
    @Test func testResolveLanguageUnsupportedCode() {
        let result = Qwen3TTSModel.resolveLanguage("xx")
        #expect(result == nil, "Unsupported code 'xx' should return nil")
    }

    /// Test empty string returns nil
    @Test func testResolveLanguageEmptyString() {
        let result = Qwen3TTSModel.resolveLanguage("")
        #expect(result == nil, "Empty string should return nil")
    }

    /// Test case insensitivity for ISO codes
    @Test func testResolveLanguageCaseInsensitive() {
        let result = Qwen3TTSModel.resolveLanguage("EN")
        #expect(result == "english", "Uppercase 'EN' should resolve to 'english'")
    }

    /// Test case insensitivity for full names
    @Test func testResolveLanguageFullNameCaseInsensitive() {
        let result = Qwen3TTSModel.resolveLanguage("English")
        #expect(result == "english", "'English' (capitalized) should resolve to 'english'")
    }

    /// Test ISO code with config validation — language exists in config
    @Test func testResolveLanguageWithConfigValid() throws {
        // Build a minimal talker config with codecLanguageId containing "english"
        let json = """
        {
            "codec_language_id": {"english": 2158, "chinese": 2159}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)

        let result = Qwen3TTSModel.resolveLanguage("en", config: config)
        #expect(result == "english", "ISO code 'en' should resolve to 'english' when validated against config")
    }

    /// Test ISO code with config validation — language NOT in config
    @Test func testResolveLanguageWithConfigInvalid() throws {
        // Build a config that only supports "chinese" — not "english"
        let json = """
        {
            "codec_language_id": {"chinese": 2159}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)

        let result = Qwen3TTSModel.resolveLanguage("en", config: config)
        #expect(result == nil, "ISO code 'en' should return nil when 'english' is not in config's codecLanguageId")
    }

    /// Test full name with config validation — passes through when in config
    @Test func testResolveLanguageFullNameWithConfigValid() throws {
        let json = """
        {
            "codec_language_id": {"english": 2158, "chinese": 2159}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)

        let result = Qwen3TTSModel.resolveLanguage("english", config: config)
        #expect(result == "english", "Full name 'english' should pass through when in config")
    }

    /// Test "auto" with config — always passes through regardless of config
    @Test func testResolveLanguageAutoWithConfig() throws {
        let json = """
        {
            "codec_language_id": {"english": 2158}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)

        let result = Qwen3TTSModel.resolveLanguage("auto", config: config)
        #expect(result == "auto", "'auto' should always pass through, even with config")
    }

    /// Test config with dialect language — pass through a dialect string in codecLanguageId
    @Test func testResolveLanguageDialectInConfig() throws {
        let json = """
        {
            "codec_language_id": {"english": 2158, "sichuan_dialect": 2170}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)

        let result = Qwen3TTSModel.resolveLanguage("sichuan_dialect", config: config)
        #expect(result == "sichuan_dialect", "Dialect strings should pass through when in config")
    }
}


struct Qwen3TTSTests {

    /// Test basic text-to-speech generation with Qwen3 model
    @Test func testQwen3Generate() async throws {
        // 1. Load Qwen3 model from HuggingFace
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
        print("\u{001B}[32mQwen3 model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Hello, this is a test of the Qwen3 text to speech model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 500,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with Qwen3 model
    @Test func testQwen3GenerateStream() async throws {
        // 1. Load Qwen3 model from HuggingFace
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
        print("\u{001B}[32mQwen3 model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Qwen3 model."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: Qwen3GenerationInfo?

        for try await event in model.generateStream(text: text, parameters: parameters) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen3_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }


}


// Run LlamaTTS tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/LlamaTTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct LlamaTTSTests {

    /// Test basic text-to-speech generation with LlamaTTS model (Orpheus)
    @Test func testLlamaTTSGenerate() async throws {
        // 1. Load LlamaTTS model from HuggingFace
        print("\u{001B}[33mLoading LlamaTTS (Orpheus) model...\u{001B}[0m")
        let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
        print("\u{001B}[32mLlamaTTS model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Hello, this is a test of the Orpheus text to speech model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 800,
            temperature: 0.7,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: "tara",
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("llama_tts_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with LlamaTTS model (Orpheus)
    @Test func testLlamaTTSGenerateStream() async throws {
        // 1. Load LlamaTTS model from HuggingFace
        print("\u{001B}[33mLoading LlamaTTS (Orpheus) model...\u{001B}[0m")
        let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
        print("\u{001B}[32mLlamaTTS model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Orpheus model."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: LlamaTTSGenerationInfo?

        for try await event in model.generateStream(text: text, voice: "tara", parameters: parameters) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("llama_tts_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }




}

// Run Qwen3-TTS VoiceDesign tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSVoiceDesignTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct Qwen3TTSVoiceDesignTests {

    /// Test VoiceDesign model loading and audio generation.
    /// This downloads the 1.7B VoiceDesign model (~3.4GB), so it requires
    /// sufficient disk space and RAM. Expect this test to take several minutes
    /// on first run due to model download.
    @Test func testVoiceDesignGenerateAudio() async throws {
        // 1. Load Qwen3-TTS VoiceDesign model
        print("\u{001B}[33mLoading Qwen3-TTS VoiceDesign model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(
            "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
        )
        print("\u{001B}[32mQwen3-TTS VoiceDesign model loaded!\u{001B}[0m")

        #expect(model.sampleRate == 24000, "VoiceDesign model should output 24kHz audio")

        // 2. Generate audio with a voice description (instruct)
        let text = "Hello, this is a test"
        let instruct = "A calm female voice with low register"
        print("\u{001B}[33mGenerating audio for: \"\(text)\" with instruct: \"\(instruct)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        // VoiceDesign uses 'voice' parameter as the instruct (voice description)
        let audio = try await model.generate(
            text: text,
            voice: instruct,
            refAudio: nil,
            refText: nil,
            language: "en",
            generationParameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Verify audio is non-empty
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save to WAV and verify file
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tts_voicedesign_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved VoiceDesign audio to\u{001B}[0m: \(outputURL.path)")

        // 5. Verify the WAV file exists and has content
        let fileData = try Data(contentsOf: outputURL)
        #expect(fileData.count > 44, "WAV file should be larger than just the header (44 bytes)")

        // 6. Verify sample rate by reading back with AVFoundation
        let audioFile = try AVAudioFile(forReading: outputURL)
        let actualSampleRate = audioFile.processingFormat.sampleRate
        #expect(actualSampleRate == 24000.0, "Output WAV should be 24kHz, got \(actualSampleRate)")

        // 7. Verify non-zero duration
        let duration = Double(audioFile.length) / actualSampleRate
        #expect(duration > 0.1, "Audio duration should be > 0.1s, got \(duration)s")
        print("\u{001B}[32mAudio duration: \(String(format: "%.2f", duration))s at \(Int(actualSampleRate))Hz\u{001B}[0m")
    }

    /// Test VoiceDesign streaming generation
    @Test func testVoiceDesignStreamGenerate() async throws {
        // 1. Load Qwen3-TTS VoiceDesign model
        print("\u{001B}[33mLoading Qwen3-TTS VoiceDesign model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(
            "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
        )
        print("\u{001B}[32mQwen3-TTS VoiceDesign model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "This is a streaming test of VoiceDesign."
        let instruct = "A deep male voice speaking slowly"
        print("\u{001B}[33mStreaming generation for: \"\(text)\" with instruct: \"\(instruct)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: AudioGenerationInfo?

        for try await event in model.generateStream(
            text: text,
            voice: instruct,
            refAudio: nil,
            refText: nil,
            language: "en",
            generationParameters: parameters
        ) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen3_tts_voicedesign_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved VoiceDesign streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }

    /// Test loading VoiceDesign model via TTSModelUtils
    @Test func testVoiceDesignModelRouting() async throws {
        // Verify the model loads correctly via the generic utility
        print("\u{001B}[33mLoading VoiceDesign model via TTSModelUtils...\u{001B}[0m")
        let model = try await TTSModelUtils.loadModel(
            modelRepo: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            modelType: "qwen3_tts"
        )
        print("\u{001B}[32mModel loaded via TTSModelUtils!\u{001B}[0m")

        #expect(model.sampleRate == 24000, "VoiceDesign model should output 24kHz audio")
        #expect(model is Qwen3TTSModel, "Model should be Qwen3TTSModel instance")
    }
}


// Run PocketTTS tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/PocketTTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct PocketTTSTests {

    /// Test basic text-to-speech generation with PocketTTS model
    @Test func testPocketTTSGenerate() async throws {
        // 1. Load PocketTTS model from HuggingFace
        print("\u{001B}[33mLoading PocketTTS model...\u{001B}[0m")
        let model = try await PocketTTSModel.fromPretrained("mlx-community/pocket-tts")
        print("\u{001B}[32mPocketTTS model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Hello, this is a test of the PocketTTS model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let audio = try await model.generate(
            text: text,
            voice: "alba",
            generationParameters: GenerateParameters(temperature: 0.7)
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("pocket_tts_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }
}


// Run Soprano tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/SopranoTTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct SopranoTTSTests {

    /// Test basic text-to-speech generation with Soprano model
    @Test func testSopranoGenerate() async throws {
        // 1. Load Soprano model from HuggingFace
        print("\u{001B}[33mLoading Soprano TTS model...\u{001B}[0m")
        let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")
        print("\u{001B}[32mSoprano model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Performance Optimization: Automatic model quantization and hardware optimization that delivers 30%-100% faster inference than standard implementations."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        // Use temperature=0.0 for deterministic generation (same as hello world test)
        let parameters = GenerateParameters(
            maxTokens: 200,
            temperature: 0.3,
            topP: 0.95,
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("soprano_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with Soprano model
    @Test func testSopranoGenerateStream() async throws {
        // 1. Load Soprano model from HuggingFace
        print("\u{001B}[33mLoading Soprano TTS model...\u{001B}[0m")
        let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")
        print("\u{001B}[32mSoprano model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Soprano model. I think it's working."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        // Use temperature=0.0 for deterministic generation
        let parameters = GenerateParameters(
            maxTokens: 100,
            temperature: 0.3,
            topP: 1.0
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: SopranoGenerationInfo?

        for try await event in model.generateStream(text: text, parameters: parameters) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("soprano_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }

    /// Test text cleaning utilities
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
