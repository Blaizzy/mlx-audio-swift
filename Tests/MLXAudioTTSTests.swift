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


// MARK: - Qwen3-TTS Config Parsing Tests (no model download required)

// Run Qwen3TTSConfigTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSConfigTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSConfigTests {

    // MARK: - Test JSON Fixtures

    /// Minimal Base model config JSON matching mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
    static let baseConfigJSON = """
    {
        "model_type": "qwen3_tts",
        "tts_model_type": "base",
        "tts_model_size": "1b7",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "tts_pad_token_id": 151671,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "sample_rate": 24000,
        "speaker_encoder_config": {
            "enc_dim": 2048,
            "sample_rate": 24000
        },
        "talker_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 6144,
            "hidden_act": "silu",
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "text_hidden_size": 2048,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_pad_id": 2148,
            "codec_think_id": 2154,
            "codec_nothink_id": 2155,
            "codec_think_bos_id": 2156,
            "codec_think_eos_id": 2157,
            "codec_language_id": {
                "chinese": 2055,
                "english": 2050,
                "german": 2053,
                "italian": 2070,
                "portuguese": 2071,
                "spanish": 2054,
                "japanese": 2058,
                "korean": 2064,
                "french": 2061,
                "russian": 2069
            },
            "spk_id": {},
            "spk_is_dialect": {}
        }
    }
    """

    /// Minimal VoiceDesign config JSON matching mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
    static let voiceDesignConfigJSON = """
    {
        "model_type": "qwen3_tts",
        "tts_model_type": "voice_design",
        "tts_model_size": "1b7",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "tts_pad_token_id": 151671,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "sample_rate": 24000,
        "talker_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 6144,
            "hidden_act": "silu",
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "text_hidden_size": 2048,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_pad_id": 2148,
            "codec_think_id": 2154,
            "codec_nothink_id": 2155,
            "codec_think_bos_id": 2156,
            "codec_think_eos_id": 2157,
            "codec_language_id": {
                "chinese": 2055,
                "english": 2050,
                "german": 2053,
                "italian": 2070,
                "portuguese": 2071,
                "spanish": 2054,
                "japanese": 2058,
                "korean": 2064,
                "french": 2061,
                "russian": 2069
            },
            "spk_id": {},
            "spk_is_dialect": {}
        }
    }
    """

    /// Minimal CustomVoice config JSON matching mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
    /// Note: spk_id values use [Int] arrays to match the Swift Codable type declaration.
    /// The actual HuggingFace JSON uses plain Int values; the Python type annotation is
    /// Dict[str, List[int]]. This discrepancy will be addressed in a future task.
    static let customVoiceConfigJSON = """
    {
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "tts_pad_token_id": 151671,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "sample_rate": 24000,
        "talker_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 6144,
            "hidden_act": "silu",
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "text_hidden_size": 2048,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_pad_id": 2148,
            "codec_think_id": 2154,
            "codec_nothink_id": 2155,
            "codec_think_bos_id": 2156,
            "codec_think_eos_id": 2157,
            "codec_language_id": {
                "chinese": 2055,
                "english": 2050,
                "german": 2053,
                "italian": 2070,
                "portuguese": 2071,
                "spanish": 2054,
                "japanese": 2058,
                "korean": 2064,
                "french": 2061,
                "russian": 2069,
                "beijing_dialect": 2074,
                "sichuan_dialect": 2062
            },
            "spk_id": {
                "serena": [3066],
                "vivian": [3065],
                "uncle_fu": [3010],
                "ryan": [3061],
                "aiden": [2861],
                "ono_anna": [2873],
                "sohee": [2864],
                "eric": [2875],
                "dylan": [2878]
            },
            "spk_is_dialect": {
                "eric": "sichuan_dialect",
                "dylan": "beijing_dialect"
            }
        }
    }
    """

    // MARK: - Helper

    private func parseConfig(_ json: String) throws -> Qwen3TTSModelConfig {
        let data = json.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: data)
    }

    // MARK: - Speaker Encoder Config Tests

    /// Parse Base model config JSON, verify speakerEncoderConfig is non-nil
    /// with encDim == 2048, sampleRate == 24000
    @Test func testBaseConfigHasSpeakerEncoderConfig() throws {
        let config = try parseConfig(Self.baseConfigJSON)
        #expect(config.speakerEncoderConfig != nil,
                "Base model config should have speaker_encoder_config")
        #expect(config.speakerEncoderConfig!.encDim == 2048,
                "Base model enc_dim should be 2048")
        #expect(config.speakerEncoderConfig!.sampleRate == 24000,
                "Base model speaker encoder sample_rate should be 24000")
    }

    /// Verify that fields not specified in the JSON get their correct defaults
    @Test func testBaseConfigSpeakerEncoderDefaults() throws {
        let config = try parseConfig(Self.baseConfigJSON)
        let sec = config.speakerEncoderConfig!
        // The Base HF JSON only specifies enc_dim and sample_rate; all other fields
        // should fall back to their defaults from the Python config class.
        #expect(sec.melDim == 128, "Default melDim should be 128")
        #expect(sec.encChannels == [512, 512, 512, 512, 1536],
                "Default encChannels should match Python defaults")
        #expect(sec.encKernelSizes == [5, 3, 3, 3, 1],
                "Default encKernelSizes should match Python defaults")
        #expect(sec.encDilations == [1, 2, 3, 4, 1],
                "Default encDilations should match Python defaults")
        #expect(sec.encAttentionChannels == 128,
                "Default encAttentionChannels should be 128")
        #expect(sec.encRes2netScale == 8,
                "Default encRes2netScale should be 8")
        #expect(sec.encSeChannels == 128,
                "Default encSeChannels should be 128")
    }

    /// Parse VoiceDesign config JSON, verify speakerEncoderConfig is nil
    @Test func testVoiceDesignConfigHasNoSpeakerEncoderConfig() throws {
        let config = try parseConfig(Self.voiceDesignConfigJSON)
        #expect(config.speakerEncoderConfig == nil,
                "VoiceDesign config should not have speaker_encoder_config")
    }

    /// Parse CustomVoice config JSON, verify speakerEncoderConfig is nil
    @Test func testCustomVoiceConfigHasNoSpeakerEncoderConfig() throws {
        let config = try parseConfig(Self.customVoiceConfigJSON)
        #expect(config.speakerEncoderConfig == nil,
                "CustomVoice config should not have speaker_encoder_config")
    }

    // MARK: - Model Type and spkId Tests

    /// Parse Base config: ttsModelType == "base", spkId empty
    @Test func testBaseConfigModelType() throws {
        let config = try parseConfig(Self.baseConfigJSON)
        #expect(config.ttsModelType == "base",
                "Base config tts_model_type should be 'base'")
        #expect(config.talkerConfig?.spkId?.isEmpty == true,
                "Base config spk_id should be empty")
    }

    /// Parse VoiceDesign config: ttsModelType == "voice_design", spkId empty
    @Test func testVoiceDesignConfigModelType() throws {
        let config = try parseConfig(Self.voiceDesignConfigJSON)
        #expect(config.ttsModelType == "voice_design",
                "VoiceDesign config tts_model_type should be 'voice_design'")
        #expect(config.talkerConfig?.spkId?.isEmpty == true,
                "VoiceDesign config spk_id should be empty")
    }

    /// Parse CustomVoice config: ttsModelType == "custom_voice", spkId has entries
    @Test func testCustomVoiceConfigModelType() throws {
        let config = try parseConfig(Self.customVoiceConfigJSON)
        #expect(config.ttsModelType == "custom_voice",
                "CustomVoice config tts_model_type should be 'custom_voice'")
        let spkId = config.talkerConfig?.spkId
        #expect(spkId != nil, "CustomVoice config spk_id should be non-nil")
        #expect(spkId!.count == 9,
                "CustomVoice config should have 9 named speakers, got \(spkId!.count)")
        #expect(spkId!["serena"] == [3066],
                "Speaker 'serena' should have ID [3066]")
    }

    // MARK: - Codec Language ID Tests

    /// Parse codecLanguageId: 10 entries for Base
    @Test func testBaseConfigCodecLanguageId() throws {
        let config = try parseConfig(Self.baseConfigJSON)
        let langId = config.talkerConfig?.codecLanguageId
        #expect(langId != nil, "Base config codec_language_id should be non-nil")
        #expect(langId!.count == 10,
                "Base config should have 10 language entries, got \(langId!.count)")
        #expect(langId!["english"] == 2050)
        #expect(langId!["chinese"] == 2055)
    }

    /// Parse codecLanguageId: 10 entries for VoiceDesign
    @Test func testVoiceDesignConfigCodecLanguageId() throws {
        let config = try parseConfig(Self.voiceDesignConfigJSON)
        let langId = config.talkerConfig?.codecLanguageId
        #expect(langId != nil, "VoiceDesign config codec_language_id should be non-nil")
        #expect(langId!.count == 10,
                "VoiceDesign config should have 10 language entries, got \(langId!.count)")
    }

    /// Parse codecLanguageId: 12 entries for CustomVoice (includes dialects)
    @Test func testCustomVoiceConfigCodecLanguageId() throws {
        let config = try parseConfig(Self.customVoiceConfigJSON)
        let langId = config.talkerConfig?.codecLanguageId
        #expect(langId != nil, "CustomVoice config codec_language_id should be non-nil")
        #expect(langId!.count == 12,
                "CustomVoice config should have 12 language entries (including dialects), got \(langId!.count)")
        #expect(langId!["beijing_dialect"] == 2074,
                "CustomVoice should include beijing_dialect")
        #expect(langId!["sichuan_dialect"] == 2062,
                "CustomVoice should include sichuan_dialect")
    }
}


// MARK: - Qwen3-TTS Generation Path Routing Tests (no model download required)

// Run Qwen3TTSRoutingTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSRoutingTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSRoutingTests {

    // MARK: - Helper to create a Qwen3TTSModel from a minimal config JSON

    /// Creates a Qwen3TTSModel with the given tts_model_type for routing tests.
    /// No weights are loaded; only the config is parsed and a minimal speech tokenizer
    /// is attached so that the hasEncoder check works.
    private func makeModel(ttsModelType: String, hasEncoder: Bool = false) throws -> Qwen3TTSModel {
        let json = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "\(ttsModelType)",
            "tts_model_size": "0b6",
            "tokenizer_type": "qwen3_tts_tokenizer_12hz",
            "sample_rate": 24000,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let model = Qwen3TTSModel(config: config)

        // Attach a minimal speech tokenizer (no weights) so hasEncoder can be checked
        let tokenizerJson = "{}".data(using: .utf8)!
        let tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: tokenizerJson)
        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: tokenizerConfig)

        if hasEncoder {
            speechTokenizer.hasEncoder = true
        }

        model.speechTokenizer = speechTokenizer
        return model
    }

    // MARK: - Routing tests

    /// voice_design config routes to .voiceDesign
    @Test func voiceDesignRouting() throws {
        let model = try makeModel(ttsModelType: "voice_design")
        let path = try model.resolveGenerationPath(refAudio: nil, refText: nil)
        #expect(path == .voiceDesign, "voice_design config should route to .voiceDesign")
    }

    /// custom_voice config routes to .customVoice
    @Test func customVoiceRouting() throws {
        let model = try makeModel(ttsModelType: "custom_voice")
        let path = try model.resolveGenerationPath(refAudio: nil, refText: nil)
        #expect(path == .customVoice, "custom_voice config should route to .customVoice")
    }

    /// base config without refAudio routes to .base
    @Test func baseRoutingWithoutRefAudio() throws {
        let model = try makeModel(ttsModelType: "base")
        let path = try model.resolveGenerationPath(refAudio: nil, refText: nil)
        #expect(path == .base, "base config without refAudio should route to .base")
    }

    /// base config with refAudio but no refText routes to .base (not ICL)
    @Test func baseRoutingWithRefAudioButNoRefText() throws {
        let model = try makeModel(ttsModelType: "base")
        let refAudio = MLXArray.zeros([1, 1, 24000])
        let path = try model.resolveGenerationPath(refAudio: refAudio, refText: nil)
        #expect(path == .base, "base config with refAudio but no refText should route to .base")
    }

    /// base config with refAudio and refText but no encoder routes to .base (not ICL)
    @Test func baseRoutingWithRefAudioAndRefTextButNoEncoder() throws {
        let model = try makeModel(ttsModelType: "base", hasEncoder: false)
        let refAudio = MLXArray.zeros([1, 1, 24000])
        let path = try model.resolveGenerationPath(refAudio: refAudio, refText: "Hello")
        #expect(path == .base, "base config with refAudio + refText but no encoder should route to .base")
    }

    /// base config with refAudio + refText + hasEncoder routes to .icl
    @Test func baseRoutingWithRefAudioRefTextAndEncoder() throws {
        let model = try makeModel(ttsModelType: "base", hasEncoder: true)
        let refAudio = MLXArray.zeros([1, 1, 24000])
        let path = try model.resolveGenerationPath(refAudio: refAudio, refText: "Hello")
        #expect(path == .icl, "base config with refAudio + refText + hasEncoder should route to .icl")
    }

    /// Unknown model type throws an error
    @Test func unknownModelTypeThrows() throws {
        let model = try makeModel(ttsModelType: "unknown_type")
        #expect(throws: AudioGenerationError.self) {
            _ = try model.resolveGenerationPath(refAudio: nil, refText: nil)
        }
    }

    /// voice_design ignores refAudio/refText — always routes to .voiceDesign
    @Test func voiceDesignRoutingIgnoresRefAudio() throws {
        let model = try makeModel(ttsModelType: "voice_design")
        let refAudio = MLXArray.zeros([1, 1, 24000])
        let path = try model.resolveGenerationPath(refAudio: refAudio, refText: "Hello")
        #expect(path == .voiceDesign, "voice_design should always route to .voiceDesign regardless of refAudio/refText")
    }

    /// custom_voice ignores refAudio/refText — always routes to .customVoice
    @Test func customVoiceRoutingIgnoresRefAudio() throws {
        let model = try makeModel(ttsModelType: "custom_voice")
        let refAudio = MLXArray.zeros([1, 1, 24000])
        let path = try model.resolveGenerationPath(refAudio: refAudio, refText: "Hello")
        #expect(path == .customVoice, "custom_voice should always route to .customVoice regardless of refAudio/refText")
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
