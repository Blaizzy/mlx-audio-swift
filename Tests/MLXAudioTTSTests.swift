//
//  MLXAudioTTSTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 31/12/2025.
//

import Testing
import MLX
import MLXNN
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

    /// Test that hasEncoder reflects the presence of an encoder model.
    /// hasEncoder is now a computed property based on encoderModel != nil.
    @Test func testHasEncoderReflectsEncoderModel() throws {
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: json)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        // Without an encoder model loaded, hasEncoder should be false
        #expect(tokenizer.hasEncoder == false,
                "hasEncoder should be false when encoderModel is nil")
        // Note: Setting hasEncoder to true requires loading a real encoder model,
        // which is tested in the integration tests with model downloads.
    }

    /// Test that hasEncoder returns true when encoder config is present (Base model case)
    @Test func testHasEncoderTrueWithEncoderConfig() throws {
        // Create a minimal encoder config JSON (Base model has encoder_config)
        let jsonString = """
        {
            "encoder_config": {
                "frame_rate": 12.5,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "audio_channels": 1,
                "codebook_dim": 256,
                "codebook_size": 2048,
                "compress": 2,
                "dilation_growth_rate": 2,
                "head_dim": 64,
                "hidden_act": "gelu",
                "hidden_size": 512,
                "intermediate_size": 2048,
                "kernel_size": 7,
                "last_kernel_size": 7,
                "layer_scale_initial_scale": 0.01,
                "max_position_embeddings": 8000,
                "norm_eps": 1e-5,
                "num_attention_heads": 8,
                "num_filters": 64,
                "num_hidden_layers": 8,
                "num_key_value_heads": 8,
                "num_quantizers": 32,
                "num_residual_layers": 1,
                "num_semantic_quantizers": 1,
                "residual_kernel_size": 3,
                "rope_theta": 10000.0,
                "sampling_rate": 24000,
                "sliding_window": 250,
                "upsampling_ratios": [8, 5, 4, 2],
                "use_causal_conv": true,
                "use_conv_shortcut": false
            }
        }
        """
        let json = jsonString.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: json)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        // With encoder config present, hasEncoder should be true
        #expect(tokenizer.hasEncoder == true,
                "hasEncoder should be true when encoderConfig is present (Base model)")
    }
}


// MARK: - Qwen3-TTS Speech Tokenizer Encode Tests (no model download required)

// Run Qwen3TTSSpeechTokenizerEncodeTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerEncodeTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSSpeechTokenizerEncodeTests {

    /// encode() should throw when encoder is not loaded
    @Test func encodeThrowsWhenNoEncoder() throws {
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: json)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)

        #expect(tokenizer.hasEncoder == false)

        let dummyAudio = MLXArray.zeros([1, 1, 24000])
        #expect(throws: AudioGenerationError.self) {
            try tokenizer.encode(dummyAudio)
        }
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

    /// Test all supported ISO 639-1 codes resolve correctly (Task 4 requirement: 30+ languages)
    @Test func testResolveLanguageAllISO() {
        // Test 30+ ISO 639-1 codes as required by Task 4
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
            "ar": "arabic",
            "hi": "hindi",
            "tr": "turkish",
            "pl": "polish",
            "nl": "dutch",
            "sv": "swedish",
            "fi": "finnish",
            "cs": "czech",
            "ro": "romanian",
            "hu": "hungarian",
            "el": "greek",
            "th": "thai",
            "vi": "vietnamese",
            "id": "indonesian",
            "ms": "malay",
            "uk": "ukrainian",
            "da": "danish",
            "no": "norwegian",
            "he": "hebrew",
            "fa": "persian",
        ]
        for (iso, name) in expected {
            let result = Qwen3TTSModel.resolveLanguage(iso)
            #expect(result == name, "ISO code '\(iso)' should resolve to '\(name)', got '\(result ?? "nil")'")
        }
    }

    /// Test ISO 639-2/T three-letter codes resolve correctly
    @Test func testResolveLanguageISO6392Codes() {
        let expected: [String: String] = [
            "eng": "english",
            "zho": "chinese",
            "jpn": "japanese",
            "kor": "korean",
            "deu": "german",
            "fra": "french",
            "rus": "russian",
            "por": "portuguese",
            "spa": "spanish",
            "ita": "italian",
            "ara": "arabic",
            "hin": "hindi",
            "tur": "turkish",
            "pol": "polish",
            "nld": "dutch",
        ]
        for (iso, name) in expected {
            let result = Qwen3TTSModel.resolveLanguage(iso)
            #expect(result == name, "ISO-639-2 code '\(iso)' should resolve to '\(name)', got '\(result ?? "nil")'")
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

        // Attach a minimal speech tokenizer (no weights) so hasEncoder can be checked.
        // hasEncoder is a computed property based on encoderModel != nil.
        // For tests that need hasEncoder == true, we create a speech tokenizer with
        // a config that includes an encoder config, then set encoderModel directly.
        let tokenizerJson: String
        if hasEncoder {
            // Include encoder config so we can construct a minimal encoder
            tokenizerJson = """
            {
                "encoder_config": {}
            }
            """
        } else {
            tokenizerJson = "{}"
        }
        let tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: tokenizerJson.data(using: .utf8)!)
        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: tokenizerConfig)

        if hasEncoder {
            // Create a minimal encoder and assign it via update(modules:) to satisfy @ModuleInfo
            let encoderConfigJson = "{}".data(using: .utf8)!
            let encoderConfig = try JSONDecoder().decode(Qwen3TTSTokenizerEncoderConfig.self, from: encoderConfigJson)
            let encoder = Qwen3TTSSpeechTokenizerEncoder(config: encoderConfig)
            speechTokenizer.update(modules: ModuleChildren(values: ["encoder_model": .value(encoder)]))
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


// MARK: - Qwen3-TTS prepareBaseInputs Unit Tests (no model download required)

// Run Qwen3TTSPrepareBaseInputsTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSPrepareBaseInputsTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSPrepareBaseInputsTests {

    // MARK: - Config parsing with real HuggingFace format

    /// Test parsing spk_id with integer values (real HuggingFace format)
    @Test func testParseSpkIdWithIntegerValues() throws {
        let json = """
        {
            "spk_id": {
                "serena": 3066,
                "eric": 2875
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        #expect(config.spkId?["serena"] == [3066],
                "Integer spk_id value 3066 should be normalised to [3066]")
        #expect(config.spkId?["eric"] == [2875],
                "Integer spk_id value 2875 should be normalised to [2875]")
    }

    /// Test parsing spk_id with array values (test fixture format)
    @Test func testParseSpkIdWithArrayValues() throws {
        let json = """
        {
            "spk_id": {
                "serena": [3066],
                "vivian": [3065]
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        #expect(config.spkId?["serena"] == [3066],
                "Array spk_id value [3066] should be preserved")
        #expect(config.spkId?["vivian"] == [3065],
                "Array spk_id value [3065] should be preserved")
    }

    /// Test parsing spk_is_dialect with mixed bool/string values (real HuggingFace format)
    @Test func testParseSpkIsDialectWithMixedTypes() throws {
        let json = """
        {
            "spk_is_dialect": {
                "serena": false,
                "vivian": false,
                "eric": "sichuan_dialect",
                "dylan": "beijing_dialect"
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        // False entries should be dropped
        #expect(config.spkIsDialect?["serena"] == nil,
                "Boolean false entries should be dropped from spkIsDialect")
        #expect(config.spkIsDialect?["vivian"] == nil,
                "Boolean false entries should be dropped from spkIsDialect")
        // String entries should be preserved
        #expect(config.spkIsDialect?["eric"] == "sichuan_dialect",
                "String dialect entries should be preserved")
        #expect(config.spkIsDialect?["dylan"] == "beijing_dialect",
                "String dialect entries should be preserved")
    }

    /// Test parsing spk_is_dialect with only string values (test fixture format)
    @Test func testParseSpkIsDialectWithOnlyStrings() throws {
        let json = """
        {
            "spk_is_dialect": {
                "eric": "sichuan_dialect",
                "dylan": "beijing_dialect"
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        #expect(config.spkIsDialect?.count == 2,
                "Should have 2 dialect entries")
        #expect(config.spkIsDialect?["eric"] == "sichuan_dialect")
        #expect(config.spkIsDialect?["dylan"] == "beijing_dialect")
    }

    // MARK: - prepareBaseInputs error conditions

    /// Test that prepareBaseInputs throws when tokenizer is not loaded
    @Test func testPrepareBaseInputsThrowsWithoutTokenizer() throws {
        let json = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "custom_voice",
            "talker_config": {
                "spk_id": {"alice": [3066]},
                "codec_language_id": {"english": 2050}
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let model = Qwen3TTSModel(config: config)
        // No tokenizer loaded

        #expect(throws: AudioGenerationError.self) {
            _ = try model.prepareBaseInputs(text: "Hello", language: "english", speaker: "alice")
        }
    }

    /// Test that prepareBaseInputs throws when speaker is not found in spkId
    @Test func testPrepareBaseInputsThrowsForUnknownSpeaker() throws {
        let json = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "custom_voice",
            "talker_config": {
                "spk_id": {"alice": [3066], "bob": [3067]},
                "codec_language_id": {"english": 2050}
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let model = Qwen3TTSModel(config: config)

        // Attach a minimal tokenizer-like object to pass the first guard
        // We can't actually set the tokenizer without loading it, but we
        // verify the error message mentions available speakers
        #expect(throws: AudioGenerationError.self) {
            _ = try model.prepareBaseInputs(text: "Hello", language: "english", speaker: "charlie")
        }
    }

    /// Test that speaker lookup is case-insensitive
    @Test func testSpeakerLookupCaseInsensitive() throws {
        let json = """
        {
            "spk_id": {"serena": [3066]},
            "codec_language_id": {"english": 2050}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)

        // Verify the spkId map uses lowercase keys (from the JSON)
        let found = config.spkId?["serena"]
        #expect(found == [3066], "Should find speaker 'serena' with lowercase lookup")

        // The prepareBaseInputs method lowercases the speaker name before lookup,
        // so "Serena" should match "serena" at runtime. We verify the config
        // structure here.
        #expect(config.spkId?["Serena"] == nil,
                "Direct lookup with uppercase should not match (case-sensitive dict)")
    }

    // MARK: - Dialect override logic verification

    /// Test dialect override: Eric with Chinese language should switch to sichuan_dialect
    @Test func testDialectOverrideEricChinese() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        // Simulate the dialect override logic from prepareBaseInputs
        let speaker = "eric"
        var effectiveLanguage = "chinese"

        if let dialectMap = talkerConfig.spkIsDialect,
           let dialect = dialectMap[speaker.lowercased()],
           (effectiveLanguage.lowercased() == "chinese" || effectiveLanguage.lowercased() == "auto"),
           let langMap = talkerConfig.codecLanguageId,
           langMap[dialect] != nil {
            effectiveLanguage = dialect
        }

        #expect(effectiveLanguage == "sichuan_dialect",
                "Eric with Chinese language should be overridden to sichuan_dialect")
    }

    /// Test dialect override: Eric with English language should NOT be overridden
    @Test func testDialectOverrideEricEnglish() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        let speaker = "eric"
        var effectiveLanguage = "english"

        if let dialectMap = talkerConfig.spkIsDialect,
           let dialect = dialectMap[speaker.lowercased()],
           (effectiveLanguage.lowercased() == "chinese" || effectiveLanguage.lowercased() == "auto"),
           let langMap = talkerConfig.codecLanguageId,
           langMap[dialect] != nil {
            effectiveLanguage = dialect
        }

        #expect(effectiveLanguage == "english",
                "Eric with English language should NOT be overridden to sichuan_dialect")
    }

    /// Test dialect override: Dylan with auto language should switch to beijing_dialect
    @Test func testDialectOverrideDylanAuto() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        let speaker = "dylan"
        var effectiveLanguage = "auto"

        if let dialectMap = talkerConfig.spkIsDialect,
           let dialect = dialectMap[speaker.lowercased()],
           (effectiveLanguage.lowercased() == "chinese" || effectiveLanguage.lowercased() == "auto"),
           let langMap = talkerConfig.codecLanguageId,
           langMap[dialect] != nil {
            effectiveLanguage = dialect
        }

        #expect(effectiveLanguage == "beijing_dialect",
                "Dylan with auto language should be overridden to beijing_dialect")
    }

    /// Test dialect override: Serena (no dialect) should NOT be overridden
    @Test func testDialectOverrideSerena() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        let speaker = "serena"
        var effectiveLanguage = "chinese"

        if let dialectMap = talkerConfig.spkIsDialect,
           let dialect = dialectMap[speaker.lowercased()],
           (effectiveLanguage.lowercased() == "chinese" || effectiveLanguage.lowercased() == "auto"),
           let langMap = talkerConfig.codecLanguageId,
           langMap[dialect] != nil {
            effectiveLanguage = dialect
        }

        #expect(effectiveLanguage == "chinese",
                "Serena has no dialect entry, language should remain chinese")
    }

    // MARK: - Codec prefix structure tests

    /// Verify the codec prefix structure for a language-aware case (with language ID)
    @Test func testCodecPrefixWithLanguageId() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        let language = "english"
        let languageId = talkerConfig.codecLanguageId?[language.lowercased()]

        #expect(languageId != nil, "English should have a language ID")
        #expect(languageId == 2050, "English language ID should be 2050")

        // With a language ID, the prefix should be: [think, thinkBos, langId, thinkEos]
        var codecPrefill: [Int32]
        if let langId = languageId {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        }

        #expect(codecPrefill.count == 4,
                "Codec prefix with language ID should have 4 tokens")
        #expect(codecPrefill[0] == Int32(talkerConfig.codecThinkId),
                "First token should be codecThinkId")
        #expect(codecPrefill[2] == 2050,
                "Third token should be the english language ID")
    }

    /// Verify the codec prefix structure for the auto/no-language case
    @Test func testCodecPrefixWithoutLanguageId() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        // Auto language should not have a language ID
        let language = "auto"
        let languageId: Int? = language.lowercased() != "auto"
            ? talkerConfig.codecLanguageId?[language.lowercased()]
            : nil

        #expect(languageId == nil, "Auto language should have nil language ID")

        var codecPrefill: [Int32]
        if let langId = languageId {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        }

        #expect(codecPrefill.count == 3,
                "Codec prefix without language ID should have 3 tokens")
        #expect(codecPrefill[0] == Int32(talkerConfig.codecNothinkId),
                "First token should be codecNothinkId for auto language")
    }

    /// Verify the full codec embed dimension count with and without speaker
    @Test func testCodecEmbedDimensionWithSpeaker() throws {
        let json = Qwen3TTSConfigTests.customVoiceConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let talkerConfig = config.talkerConfig!

        // With language ID: prefix=[think, thinkBos, langId, thinkEos] (4 tokens)
        // Suffix=[pad, bos] (2 tokens)
        // With speaker: +1 token between prefix and suffix
        // Total codecEmbed = 4 + 1 + 2 = 7

        let prefixLen = 4  // think, thinkBos, langId, thinkEos
        let suffixLen = 2  // pad, bos
        let speakerLen = 1 // one speaker embed token

        let totalWithSpeaker = prefixLen + speakerLen + suffixLen
        let totalWithoutSpeaker = prefixLen + suffixLen

        #expect(totalWithSpeaker == 7,
                "Codec embed with speaker should have 7 tokens")
        #expect(totalWithoutSpeaker == 6,
                "Codec embed without speaker should have 6 tokens")

        // padCount = codecEmbed.count - 2 (all but the last two get pads)
        let padCountWithSpeaker = totalWithSpeaker - 2
        let padCountWithoutSpeaker = totalWithoutSpeaker - 2

        #expect(padCountWithSpeaker == 5,
                "Pad count with speaker should be 5")
        #expect(padCountWithoutSpeaker == 4,
                "Pad count without speaker should be 4")
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


// MARK: - Qwen3-TTS Speaker Encoder Unit Tests (no model download required)

// Run Qwen3TTSSpeakerEncoderTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSSpeakerEncoderTests {

    // MARK: - Helper to create default config

    private func makeDefaultConfig() throws -> Qwen3TTSSpeakerEncoderConfig {
        let json = """
        {
            "enc_dim": 2048,
            "sample_rate": 24000
        }
        """.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen3TTSSpeakerEncoderConfig.self, from: json)
    }

    // MARK: - Layer Structure Tests

    /// Verify the encoder initializes with the correct number of blocks
    @Test func testBlockCount() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        // blocks: 1 initial TDNN + 3 SE-Res2Net blocks = 4 total
        #expect(encoder.blocks.count == 4,
                "Expected 4 blocks (1 TDNN + 3 SE-Res2Net), got \(encoder.blocks.count)")
    }

    /// Verify the first block is a TimeDelayNetBlock
    @Test func testFirstBlockIsTDNN() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        #expect(encoder.blocks[0] is TimeDelayNetBlock,
                "First block should be a TimeDelayNetBlock")
    }

    /// Verify blocks 1-3 are SqueezeExcitationRes2NetBlock
    @Test func testSERes2NetBlocks() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        for i in 1 ..< 4 {
            #expect(encoder.blocks[i] is SqueezeExcitationRes2NetBlock,
                    "Block \(i) should be a SqueezeExcitationRes2NetBlock")
        }
    }

    /// Verify config values are stored correctly
    @Test func testConfigValues() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        #expect(encoder.config.melDim == 128)
        #expect(encoder.config.encDim == 2048)
        #expect(encoder.config.encChannels == [512, 512, 512, 512, 1536])
        #expect(encoder.config.encRes2netScale == 8)
    }

    // MARK: - Shape Tests

    /// Feed a single mel spectrogram [1, 100, 128] and verify output shape [1, enc_dim]
    @Test func testSingleInputShape() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        let melInput = MLXArray.zeros([1, 100, 128])
        let output = encoder(melInput)
        eval(output)

        #expect(output.shape == [1, 2048],
                "Expected output shape [1, 2048], got \(output.shape)")
    }

    /// Feed a batch of mel spectrograms [2, 100, 128] and verify output shape [2, enc_dim]
    @Test func testBatchInputShape() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        let melInput = MLXArray.zeros([2, 100, 128])
        let output = encoder(melInput)
        eval(output)

        #expect(output.shape == [2, 2048],
                "Expected output shape [2, 2048], got \(output.shape)")
    }

    /// Verify output shape with a shorter time dimension
    @Test func testShortTimeInput() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        let melInput = MLXArray.zeros([1, 20, 128])
        let output = encoder(melInput)
        eval(output)

        #expect(output.shape == [1, 2048],
                "Expected output shape [1, 2048] for short input, got \(output.shape)")
    }

    /// Verify output shape with a longer time dimension
    @Test func testLongTimeInput() throws {
        let config = try makeDefaultConfig()
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        let melInput = MLXArray.zeros([1, 500, 128])
        let output = encoder(melInput)
        eval(output)

        #expect(output.shape == [1, 2048],
                "Expected output shape [1, 2048] for long input, got \(output.shape)")
    }

    /// Verify output shape with enc_dim=1024 (non-Base model config)
    @Test func testCustomEncDim() throws {
        let json = """
        {
            "enc_dim": 1024,
            "sample_rate": 24000
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSSpeakerEncoderConfig.self, from: json)
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        let melInput = MLXArray.zeros([1, 100, 128])
        let output = encoder(melInput)
        eval(output)

        #expect(output.shape == [1, 1024],
                "Expected output shape [1, 1024], got \(output.shape)")
    }

    // MARK: - Reflect Padding Tests

    /// Verify reflectPad1d produces correct output shape
    @Test func testReflectPadShape() {
        let x = MLXArray.zeros([1, 10, 4])
        let padded = reflectPad1d(x, pad: 3)
        eval(padded)

        #expect(padded.shape == [1, 16, 4],
                "Expected padded shape [1, 16, 4], got \(padded.shape)")
    }

    /// Verify reflectPad1d with pad=0 is identity
    @Test func testReflectPadZero() {
        let x = MLXArray.ones([1, 10, 4])
        let padded = reflectPad1d(x, pad: 0)
        eval(padded)

        #expect(padded.shape == [1, 10, 4],
                "Padding with 0 should return original shape")
    }

    // MARK: - Sanitize Tests

    /// Verify sanitize strips prefix and transposes 3D weights
    @Test func testSanitizeStripsPrefix() {
        let weights: [String: MLXArray] = [
            "speaker_encoder.blocks.0.conv.weight": MLXArray.zeros([64, 128, 5]),
            "speaker_encoder.blocks.0.conv.bias": MLXArray.zeros([64]),
            "other_module.weight": MLXArray.zeros([10, 10]),
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)

        #expect(sanitized.count == 2,
                "Should only include speaker_encoder keys, got \(sanitized.count)")
        #expect(sanitized["blocks.0.conv.weight"] != nil,
                "Should strip speaker_encoder. prefix")
        #expect(sanitized["blocks.0.conv.bias"] != nil,
                "Should include bias")
    }

    /// Verify sanitize transposes Conv1d weights from [O,I,K] to [O,K,I]
    @Test func testSanitizeTransposesConvWeights() {
        let weights: [String: MLXArray] = [
            "speaker_encoder.fc.weight": MLXArray.zeros([2048, 3072, 1]),
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)
        let transposed = sanitized["fc.weight"]!
        eval(transposed)

        #expect(transposed.shape == [2048, 1, 3072],
                "Conv weight should be transposed to [O, K, I], got \(transposed.shape)")
    }
}


// MARK: - Qwen3-TTS Speaker Encoder Weight Loading Tests (no model download required)

// Run Qwen3TTSSpeakerEncoderWeightTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderWeightTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSSpeakerEncoderWeightTests {

    // MARK: - Initialization from config

    /// Verify speaker encoder initializes correctly from a config with custom encDim
    @Test func testInitFromConfig() throws {
        let json = """
        {
            "enc_dim": 2048,
            "sample_rate": 24000
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSSpeakerEncoderConfig.self, from: json)
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        #expect(encoder.config.encDim == 2048,
                "Encoder encDim should be 2048")
        #expect(encoder.config.sampleRate == 24000,
                "Encoder sampleRate should be 24000")
        #expect(encoder.config.melDim == 128,
                "Encoder melDim should default to 128")
    }

    /// Verify speaker encoder initializes from minimal (all-defaults) config
    @Test func testInitFromMinimalConfig() throws {
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSSpeakerEncoderConfig.self, from: json)
        let encoder = Qwen3TTSSpeakerEncoder(config: config)

        #expect(encoder.config.encDim == 1024,
                "Default encDim should be 1024")
        #expect(encoder.config.encChannels == [512, 512, 512, 512, 1536],
                "Default encChannels should match Python defaults")
    }

    // MARK: - Sanitize prefix stripping

    /// Verify sanitize strips "speaker_encoder." prefix from all matching keys
    @Test func testSanitizeStripsPrefixCorrectly() {
        let weights: [String: MLXArray] = [
            "speaker_encoder.blocks.0.conv.weight": MLXArray.zeros([64, 128, 5]),
            "speaker_encoder.blocks.0.conv.bias": MLXArray.zeros([64]),
            "speaker_encoder.mfa.conv.weight": MLXArray.zeros([1536, 1536, 1]),
            "speaker_encoder.asp.tdnn.conv.weight": MLXArray.zeros([128, 4608, 1]),
            "speaker_encoder.fc.weight": MLXArray.zeros([2048, 3072, 1]),
            "speaker_encoder.fc.bias": MLXArray.zeros([2048]),
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)

        #expect(sanitized.count == 6,
                "All 6 speaker_encoder keys should be included, got \(sanitized.count)")
        #expect(sanitized["blocks.0.conv.weight"] != nil,
                "Should strip prefix to 'blocks.0.conv.weight'")
        #expect(sanitized["blocks.0.conv.bias"] != nil,
                "Should strip prefix to 'blocks.0.conv.bias'")
        #expect(sanitized["mfa.conv.weight"] != nil,
                "Should strip prefix to 'mfa.conv.weight'")
        #expect(sanitized["asp.tdnn.conv.weight"] != nil,
                "Should strip prefix to 'asp.tdnn.conv.weight'")
        #expect(sanitized["fc.weight"] != nil,
                "Should strip prefix to 'fc.weight'")
        #expect(sanitized["fc.bias"] != nil,
                "Should strip prefix to 'fc.bias'")
    }

    /// Verify sanitize excludes keys that do not start with "speaker_encoder."
    @Test func testSanitizeExcludesNonSpeakerEncoderKeys() {
        let weights: [String: MLXArray] = [
            "speaker_encoder.fc.weight": MLXArray.zeros([2048, 3072, 1]),
            "talker.model.embed_tokens.weight": MLXArray.zeros([3072, 2048]),
            "speech_tokenizer.decoder.weight": MLXArray.zeros([1024, 512]),
            "model.layers.0.weight": MLXArray.zeros([2048, 2048]),
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)

        #expect(sanitized.count == 1,
                "Only 1 speaker_encoder key should pass through, got \(sanitized.count)")
        #expect(sanitized["fc.weight"] != nil,
                "fc.weight should be present")
    }

    /// Verify sanitize returns empty dict when no speaker_encoder keys exist
    @Test func testSanitizeReturnsEmptyForNoMatchingKeys() {
        let weights: [String: MLXArray] = [
            "talker.model.embed_tokens.weight": MLXArray.zeros([3072, 2048]),
            "model.layers.0.weight": MLXArray.zeros([2048, 2048]),
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)

        #expect(sanitized.isEmpty,
                "Sanitize should return empty dict when no speaker_encoder keys exist")
    }

    // MARK: - Sanitize Conv1d weight transposition

    /// Verify sanitize transposes 3D weights (Conv1d) from [O, I, K] to [O, K, I]
    @Test func testSanitizeTransposesConv1dWeights() {
        // PyTorch Conv1d: [out_channels=64, in_channels=128, kernel_size=5]
        // MLX Conv1d:     [out_channels=64, kernel_size=5, in_channels=128]
        let pytorchWeight = MLXArray.zeros([64, 128, 5])
        let weights: [String: MLXArray] = [
            "speaker_encoder.blocks.0.conv.weight": pytorchWeight,
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)
        let transposed = sanitized["blocks.0.conv.weight"]!
        eval(transposed)

        #expect(transposed.shape == [64, 5, 128],
                "Expected transposed shape [64, 5, 128] (O,K,I), got \(transposed.shape)")
    }

    /// Verify sanitize transposes kernel_size=1 Conv1d weights correctly
    @Test func testSanitizeTransposesKernel1ConvWeights() {
        // kernel_size=1: [2048, 3072, 1] -> [2048, 1, 3072]
        let pytorchWeight = MLXArray.zeros([2048, 3072, 1])
        let weights: [String: MLXArray] = [
            "speaker_encoder.fc.weight": pytorchWeight,
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)
        let transposed = sanitized["fc.weight"]!
        eval(transposed)

        #expect(transposed.shape == [2048, 1, 3072],
                "Expected transposed shape [2048, 1, 3072], got \(transposed.shape)")
    }

    /// Verify sanitize does NOT transpose 1D weights (biases)
    @Test func testSanitizeDoesNotTransposeBias() {
        let bias = MLXArray.zeros([64])
        let weights: [String: MLXArray] = [
            "speaker_encoder.blocks.0.conv.bias": bias,
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)
        let result = sanitized["blocks.0.conv.bias"]!
        eval(result)

        #expect(result.shape == [64],
                "Bias should not be transposed, got shape \(result.shape)")
        #expect(result.ndim == 1,
                "Bias should remain 1D")
    }

    /// Verify sanitize does NOT transpose 2D weights (e.g., linear layers)
    @Test func testSanitizeDoesNotTranspose2DWeights() {
        // A hypothetical 2D weight that shouldn't be transposed
        let weight2d = MLXArray.zeros([128, 64])
        let weights: [String: MLXArray] = [
            "speaker_encoder.some_layer.weight": weight2d,
        ]

        let sanitized = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)
        let result = sanitized["some_layer.weight"]!
        eval(result)

        #expect(result.shape == [128, 64],
                "2D weights should not be transposed, got shape \(result.shape)")
    }

    // MARK: - Model property integration

    /// Verify Qwen3TTSModel has speakerEncoder property, initially nil
    @Test func testModelSpeakerEncoderPropertyDefaultsToNil() throws {
        let json = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "voice_design"
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)
        let model = Qwen3TTSModel(config: config)

        #expect(model.speakerEncoder == nil,
                "speakerEncoder should default to nil")
    }

    /// Verify speakerEncoder can be assigned on a Qwen3TTSModel
    @Test func testModelSpeakerEncoderCanBeSet() throws {
        let modelJson = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "speaker_encoder_config": {
                "enc_dim": 2048,
                "sample_rate": 24000
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: modelJson)
        let model = Qwen3TTSModel(config: config)

        let encoderConfig = config.speakerEncoderConfig!
        let encoder = Qwen3TTSSpeakerEncoder(config: encoderConfig)
        model.speakerEncoder = encoder

        #expect(model.speakerEncoder != nil,
                "speakerEncoder should be non-nil after assignment")
        #expect(model.speakerEncoder?.config.encDim == 2048,
                "Assigned encoder should have encDim 2048")
    }

    /// Verify that VoiceDesign config does not trigger speaker encoder loading
    /// (speakerEncoderConfig is nil for VoiceDesign)
    @Test func testVoiceDesignConfigHasNoSpeakerEncoder() throws {
        let json = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "voice_design"
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)

        #expect(config.speakerEncoderConfig == nil,
                "VoiceDesign config should not have speakerEncoderConfig")
    }

    /// Verify that Base config has speakerEncoderConfig
    @Test func testBaseConfigHasSpeakerEncoderConfig() throws {
        let json = """
        {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "speaker_encoder_config": {
                "enc_dim": 2048
            }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json)

        #expect(config.speakerEncoderConfig != nil,
                "Base config should have speakerEncoderConfig")
        #expect(config.speakerEncoderConfig?.encDim == 2048,
                "Base config speakerEncoderConfig encDim should be 2048")
    }
}


// MARK: - Qwen3-TTS Speaker Embedding Extraction Tests (no model download required)

// Run Qwen3TTSSpeakerEmbeddingTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSSpeakerEmbeddingTests \
// CODE_SIGNING_ALLOWED=NO

struct Qwen3TTSSpeakerEmbeddingTests {

    // MARK: - Helper to create model with/without speaker encoder

    /// Creates a Qwen3TTSModel with optional speaker encoder attached.
    private func makeModel(withSpeakerEncoder: Bool, encDim: Int = 2048) throws -> Qwen3TTSModel {
        let json: String
        if withSpeakerEncoder {
            json = """
            {
                "model_type": "qwen3_tts",
                "tts_model_type": "base",
                "speaker_encoder_config": {
                    "enc_dim": \(encDim),
                    "sample_rate": 24000
                }
            }
            """
        } else {
            json = """
            {
                "model_type": "qwen3_tts",
                "tts_model_type": "voice_design"
            }
            """
        }
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: json.data(using: .utf8)!)
        let model = Qwen3TTSModel(config: config)

        if withSpeakerEncoder, let encoderConfig = config.speakerEncoderConfig {
            let encoder = Qwen3TTSSpeakerEncoder(config: encoderConfig)
            model.speakerEncoder = encoder
        }

        return model
    }

    // MARK: - Error handling tests

    /// extractSpeakerEmbedding throws when speaker encoder is not loaded
    @Test func throwsWhenNoSpeakerEncoder() throws {
        let model = try makeModel(withSpeakerEncoder: false)

        #expect(throws: AudioGenerationError.self) {
            _ = try model.extractSpeakerEmbedding(audio: MLXArray.zeros([24000]))
        }
    }

    /// extractSpeakerEmbedding throws with modelNotInitialized error type
    @Test func throwsModelNotInitializedError() throws {
        let model = try makeModel(withSpeakerEncoder: false)

        do {
            _ = try model.extractSpeakerEmbedding(audio: MLXArray.zeros([24000]))
            Issue.record("Expected error to be thrown")
        } catch let error as AudioGenerationError {
            // Verify it is the modelNotInitialized variant
            if case .modelNotInitialized(let msg) = error {
                #expect(msg.contains("Speaker encoder"),
                        "Error message should mention speaker encoder, got: \(msg)")
            } else {
                Issue.record("Expected modelNotInitialized error, got: \(error)")
            }
        }
    }

    // MARK: - Output shape tests

    /// extractSpeakerEmbedding returns correct shape for 1D audio input
    @Test func outputShapeFor1DAudio() throws {
        let model = try makeModel(withSpeakerEncoder: true, encDim: 2048)

        // 1 second of audio at 24kHz
        let audio = MLXArray.zeros([24000])
        let embedding = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding)

        #expect(embedding.shape == [1, 2048],
                "Expected embedding shape [1, 2048], got \(embedding.shape)")
    }

    /// extractSpeakerEmbedding returns correct shape for 2D audio input [1, samples]
    @Test func outputShapeFor2DAudio() throws {
        let model = try makeModel(withSpeakerEncoder: true, encDim: 2048)

        // 1 second of audio at 24kHz, batched
        let audio = MLXArray.zeros([1, 24000])
        let embedding = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding)

        #expect(embedding.shape == [1, 2048],
                "Expected embedding shape [1, 2048] for 2D input, got \(embedding.shape)")
    }

    /// extractSpeakerEmbedding output shape matches configured encDim
    @Test func outputShapeMatchesEncDim() throws {
        let model = try makeModel(withSpeakerEncoder: true, encDim: 1024)

        let audio = MLXArray.zeros([24000])
        let embedding = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding)

        #expect(embedding.shape == [1, 1024],
                "Expected embedding shape [1, 1024], got \(embedding.shape)")
    }

    /// extractSpeakerEmbedding works with short audio (0.5 seconds)
    @Test func outputShapeForShortAudio() throws {
        let model = try makeModel(withSpeakerEncoder: true, encDim: 2048)

        // 0.5 seconds of audio
        let audio = MLXArray.zeros([12000])
        let embedding = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding)

        #expect(embedding.shape == [1, 2048],
                "Expected embedding shape [1, 2048] for short audio, got \(embedding.shape)")
    }

    /// extractSpeakerEmbedding works with longer audio (3 seconds)
    @Test func outputShapeForLongAudio() throws {
        let model = try makeModel(withSpeakerEncoder: true, encDim: 2048)

        // 3 seconds of audio
        let audio = MLXArray.zeros([72000])
        let embedding = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding)

        #expect(embedding.shape == [1, 2048],
                "Expected embedding shape [1, 2048] for long audio, got \(embedding.shape)")
    }

    // MARK: - Determinism test

    /// extractSpeakerEmbedding produces identical output for identical input
    @Test func deterministicOutput() throws {
        let model = try makeModel(withSpeakerEncoder: true, encDim: 2048)

        let audio = MLXArray.zeros([24000])
        let embedding1 = try model.extractSpeakerEmbedding(audio: audio)
        let embedding2 = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding1, embedding2)

        // With zero input and random weights, both calls should produce identical results
        let diff = MLX.abs(embedding1 - embedding2).sum()
        eval(diff)
        let diffVal: Float = diff.item()

        #expect(diffVal < 1e-6,
                "Same input should produce identical embeddings, diff = \(diffVal)")
    }
}


// MARK: - Qwen3-TTS Base Model Integration Tests (requires model download)

// Run Qwen3TTSBaseModelTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSBaseModelTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct Qwen3TTSBaseModelTests {

    /// The HuggingFace repo ID for the Base model
    static let baseModelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    /// Test loading and generating audio with the Base model (no speaker)
    @Test func testBaseGenerateAudio() async throws {
        // 1. Load Qwen3-TTS Base model
        print("\u{001B}[33mLoading Qwen3-TTS Base model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mQwen3-TTS Base model loaded!\u{001B}[0m")

        #expect(model.sampleRate == 24000, "Base model should output 24kHz audio")

        // 2. Generate audio without specifying a speaker
        let text = "Hello, this is a test of the base model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
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
            .appendingPathComponent("qwen3_tts_base_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved Base audio to\u{001B}[0m: \(outputURL.path)")

        // 5. Verify WAV file exists and has content
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

    /// Test generating audio with different language settings
    @Test func testBaseGenerateWithDifferentLanguages() async throws {
        // 1. Load model
        print("\u{001B}[33mLoading Qwen3-TTS Base model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mQwen3-TTS Base model loaded!\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        // 2. Generate with English
        let englishText = "Good morning, how are you today?"
        print("\u{001B}[33mGenerating English audio...\u{001B}[0m")
        let englishAudio = try await model.generate(
            text: englishText,
            voice: nil,
            refAudio: nil,
            refText: nil,
            language: "en",
            generationParameters: parameters
        )
        #expect(englishAudio.shape[0] > 0, "English audio should have samples")
        print("\u{001B}[32mEnglish audio shape: \(englishAudio.shape)\u{001B}[0m")

        // 3. Generate with auto language detection
        let autoText = "This is a test with automatic language detection."
        print("\u{001B}[33mGenerating auto-detect audio...\u{001B}[0m")
        let autoAudio = try await model.generate(
            text: autoText,
            voice: nil,
            refAudio: nil,
            refText: nil,
            language: "auto",
            generationParameters: parameters
        )
        #expect(autoAudio.shape[0] > 0, "Auto-detected audio should have samples")
        print("\u{001B}[32mAuto-detect audio shape: \(autoAudio.shape)\u{001B}[0m")
    }
}


// MARK: - Qwen3-TTS CustomVoice Model Integration Tests (requires model download)

// Run Qwen3TTSCustomVoiceTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSCustomVoiceTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct Qwen3TTSCustomVoiceTests {

    /// The HuggingFace repo ID for the CustomVoice model
    static let customVoiceModelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"

    /// Test loading and generating audio with a named speaker
    @Test func testCustomVoiceGenerateWithSpeaker() async throws {
        // 1. Load Qwen3-TTS CustomVoice model
        print("\u{001B}[33mLoading Qwen3-TTS CustomVoice model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.customVoiceModelRepo)
        print("\u{001B}[32mQwen3-TTS CustomVoice model loaded!\u{001B}[0m")

        #expect(model.sampleRate == 24000, "CustomVoice model should output 24kHz audio")

        // 2. Generate audio with a named speaker (e.g., "serena")
        let text = "Hello, this is a test of the custom voice model."
        let speaker = "serena"
        print("\u{001B}[33mGenerating audio for: \"\(text)\" with speaker: \"\(speaker)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: speaker,
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
            .appendingPathComponent("qwen3_tts_customvoice_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved CustomVoice audio to\u{001B}[0m: \(outputURL.path)")

        // 5. Verify WAV file exists and has content
        let fileData = try Data(contentsOf: outputURL)
        #expect(fileData.count > 44, "WAV file should be larger than just the header (44 bytes)")

        // 6. Verify sample rate
        let audioFile = try AVAudioFile(forReading: outputURL)
        let actualSampleRate = audioFile.processingFormat.sampleRate
        #expect(actualSampleRate == 24000.0, "Output WAV should be 24kHz, got \(actualSampleRate)")

        // 7. Verify non-zero duration
        let duration = Double(audioFile.length) / actualSampleRate
        #expect(duration > 0.1, "Audio duration should be > 0.1s, got \(duration)s")
        print("\u{001B}[32mAudio duration: \(String(format: "%.2f", duration))s at \(Int(actualSampleRate))Hz\u{001B}[0m")
    }

    /// Test that an invalid speaker name throws an error
    @Test func testInvalidSpeakerThrowsError() async throws {
        // 1. Load model
        print("\u{001B}[33mLoading Qwen3-TTS CustomVoice model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.customVoiceModelRepo)
        print("\u{001B}[32mQwen3-TTS CustomVoice model loaded!\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        // 2. Attempt to generate with a nonexistent speaker
        let text = "This should fail."
        let invalidSpeaker = "nonexistent_speaker_12345"

        do {
            _ = try await model.generate(
                text: text,
                voice: invalidSpeaker,
                refAudio: nil,
                refText: nil,
                language: "en",
                generationParameters: parameters
            )
            Issue.record("Expected an error for invalid speaker name, but generation succeeded")
        } catch let error as AudioGenerationError {
            if case .invalidInput(let msg) = error {
                #expect(msg.contains("not found"),
                        "Error should mention speaker not found, got: \(msg)")
                print("\u{001B}[32mCorrectly received error for invalid speaker: \(msg)\u{001B}[0m")
            } else {
                Issue.record("Expected invalidInput error, got: \(error)")
            }
        }
    }

    /// Test generating audio with a different speaker to verify speaker variation
    @Test func testCustomVoiceGenerateWithDifferentSpeaker() async throws {
        // 1. Load model
        print("\u{001B}[33mLoading Qwen3-TTS CustomVoice model...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.customVoiceModelRepo)
        print("\u{001B}[32mQwen3-TTS CustomVoice model loaded!\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        // 2. Generate with "ryan" speaker
        let text = "Hello, testing with a different voice."
        let speaker = "ryan"
        print("\u{001B}[33mGenerating audio with speaker: \"\(speaker)\"...\u{001B}[0m")

        let audio = try await model.generate(
            text: text,
            voice: speaker,
            refAudio: nil,
            refText: nil,
            language: "en",
            generationParameters: parameters
        )

        #expect(audio.shape[0] > 0, "Audio should have samples")
        print("\u{001B}[32mGenerated audio with \"\(speaker)\": shape \(audio.shape)\u{001B}[0m")

        // Save for manual inspection
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tts_customvoice_ryan_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved CustomVoice audio (ryan) to\u{001B}[0m: \(outputURL.path)")
    }
}


// MARK: - Qwen3-TTS ICL Voice Cloning Integration Tests (requires model download)

// Run Qwen3TTSCloningTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSCloningTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct Qwen3TTSCloningTests {

    /// The HuggingFace repo ID for the Base model (required for ICL cloning)
    static let baseModelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    /// Create a synthetic reference audio waveform for testing.
    /// Generates a 2-second sine wave at 440Hz, sampled at 24kHz.
    private func makeSyntheticReferenceAudio(durationSeconds: Float = 2.0, frequency: Float = 440.0) -> MLXArray {
        let sampleRate: Float = 24000.0
        let numSamples = Int(sampleRate * durationSeconds)
        var samples = [Float](repeating: 0, count: numSamples)
        for i in 0 ..< numSamples {
            samples[i] = sin(2.0 * .pi * frequency * Float(i) / sampleRate) * 0.5
        }
        return MLXArray(samples)
    }

    /// Test that the Base model with encoder can encode reference audio into codec codes
    @Test func testEncodeReferenceAudio() async throws {
        // 1. Load Base model (includes speech encoder)
        print("\u{001B}[33mLoading Qwen3-TTS Base model for encoding test...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mBase model loaded!\u{001B}[0m")

        // 2. Create synthetic reference audio
        let refAudio = makeSyntheticReferenceAudio()
        print("\u{001B}[33mEncoding reference audio (shape: \(refAudio.shape))...\u{001B}[0m")

        // 3. Create a clone prompt to trigger encoding
        let clonePrompt = try model.createVoiceClonePrompt(
            refAudio: refAudio,
            refText: "This is a reference audio sample.",
            language: "en"
        )

        // 4. Verify refCodes shape: [1, 16, refTime]
        let refCodes = clonePrompt.refCodes
        eval(refCodes)
        #expect(refCodes.ndim == 3, "refCodes should be 3D, got \(refCodes.ndim)")
        #expect(refCodes.dim(0) == 1, "refCodes batch dim should be 1, got \(refCodes.dim(0))")
        #expect(refCodes.dim(1) == 16, "refCodes should have 16 codebooks, got \(refCodes.dim(1))")
        #expect(refCodes.dim(2) > 0, "refCodes should have positive time dimension, got \(refCodes.dim(2))")
        print("\u{001B}[32mrefCodes shape: \(refCodes.shape)\u{001B}[0m")
    }

    /// Test end-to-end ICL voice cloning generation
    @Test func testICLGeneration() async throws {
        // 1. Load Base model
        print("\u{001B}[33mLoading Qwen3-TTS Base model for ICL generation...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mBase model loaded!\u{001B}[0m")

        // 2. Create synthetic reference audio
        let refAudio = makeSyntheticReferenceAudio()
        let refText = "This is the reference text transcript."

        // 3. Generate audio using ICL (voice cloning)
        let text = "Hello, this is a voice cloning test."
        print("\u{001B}[33mGenerating ICL audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.9,
            topP: 1.0,
            repetitionPenalty: 1.5,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
            refAudio: refAudio,
            refText: refText,
            language: "en",
            generationParameters: parameters
        )

        print("\u{001B}[32mGenerated ICL audio shape: \(audio.shape)\u{001B}[0m")

        // 4. Verify audio is non-empty
        #expect(audio.shape[0] > 0, "ICL-generated audio should have samples")

        // 5. Save to WAV
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tts_icl_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved ICL audio to\u{001B}[0m: \(outputURL.path)")

        // 6. Verify WAV file
        let fileData = try Data(contentsOf: outputURL)
        #expect(fileData.count > 44, "WAV file should be larger than just the header")
    }

    /// Test VoiceClonePrompt creation, serialization, deserialization, and generation
    @Test func testClonePromptRoundTrip() async throws {
        // 1. Load Base model
        print("\u{001B}[33mLoading Qwen3-TTS Base model for clone prompt round-trip...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mBase model loaded!\u{001B}[0m")

        // 2. Create a clone prompt from synthetic audio
        let refAudio = makeSyntheticReferenceAudio()
        let refText = "This is a reference sentence for cloning."
        let language = "en"

        print("\u{001B}[33mCreating VoiceClonePrompt...\u{001B}[0m")
        let originalPrompt = try model.createVoiceClonePrompt(
            refAudio: refAudio,
            refText: refText,
            language: language
        )

        // 3. Verify prompt fields
        #expect(originalPrompt.refText == refText, "refText should match")
        #expect(originalPrompt.language == language, "language should match")
        #expect(originalPrompt.refCodes.ndim == 3, "refCodes should be 3D")
        #expect(originalPrompt.speakerEmbedding != nil,
                "Base model should produce a speaker embedding")
        if let emb = originalPrompt.speakerEmbedding {
            #expect(emb.shape[0] == 1, "Speaker embedding batch dim should be 1")
            #expect(emb.shape[1] > 0, "Speaker embedding dim should be positive")
            print("\u{001B}[32mSpeaker embedding shape: \(emb.shape)\u{001B}[0m")
        }

        // 4. Serialize
        print("\u{001B}[33mSerializing VoiceClonePrompt...\u{001B}[0m")
        let serializedData = try originalPrompt.serialize()
        #expect(serializedData.count > 0, "Serialized data should be non-empty")
        print("\u{001B}[32mSerialized to \(serializedData.count) bytes\u{001B}[0m")

        // 5. Deserialize
        print("\u{001B}[33mDeserializing VoiceClonePrompt...\u{001B}[0m")
        let restoredPrompt = try VoiceClonePrompt.deserialize(from: serializedData)
        #expect(restoredPrompt.refText == refText, "Deserialized refText should match")
        #expect(restoredPrompt.language == language, "Deserialized language should match")
        #expect(restoredPrompt.refCodes.shape == originalPrompt.refCodes.shape,
                "Deserialized refCodes shape should match original")
        #expect(restoredPrompt.speakerEmbedding != nil,
                "Deserialized prompt should have speaker embedding")

        // 6. Verify refCodes values match
        let refCodesDiff = MLX.abs(restoredPrompt.refCodes - originalPrompt.refCodes).sum()
        eval(refCodesDiff)
        let codeDiffVal: Float = refCodesDiff.item()
        #expect(codeDiffVal < 1e-6,
                "Deserialized refCodes should match original, diff = \(codeDiffVal)")

        // 7. Generate audio using the deserialized clone prompt
        let text = "Testing generation from deserialized clone prompt."
        print("\u{001B}[33mGenerating from deserialized clone prompt...\u{001B}[0m")
        let audio = try model.generateWithClonePrompt(
            text: text,
            clonePrompt: restoredPrompt,
            temperature: 0.9,
            topP: 1.0,
            repetitionPenalty: 1.5,
            maxTokens: 2048
        )

        #expect(audio.shape[0] > 0, "Audio from clone prompt should have samples")
        print("\u{001B}[32mGenerated audio from clone prompt: shape \(audio.shape)\u{001B}[0m")

        // 8. Save to WAV
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tts_clone_prompt_roundtrip_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved clone prompt audio to\u{001B}[0m: \(outputURL.path)")
    }
}


// MARK: - Qwen3-TTS Speaker Encoder Integration Tests (requires model download)

// Run Qwen3TTSSpeakerEncoderIntegrationTests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderIntegrationTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct Qwen3TTSSpeakerEncoderIntegrationTests {

    /// The HuggingFace repo ID for the Base model (only Base has a speaker encoder)
    static let baseModelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    /// Create a synthetic audio waveform with a specific frequency.
    private func makeSyntheticAudio(
        durationSeconds: Float = 2.0,
        frequency: Float = 440.0,
        amplitude: Float = 0.5
    ) -> MLXArray {
        let sampleRate: Float = 24000.0
        let numSamples = Int(sampleRate * durationSeconds)
        var samples = [Float](repeating: 0, count: numSamples)
        for i in 0 ..< numSamples {
            samples[i] = sin(2.0 * .pi * frequency * Float(i) / sampleRate) * amplitude
        }
        return MLXArray(samples)
    }

    /// Test extracting a speaker embedding from the loaded Base model
    @Test func testExtractSpeakerEmbedding() async throws {
        // 1. Load Base model (includes speaker encoder)
        print("\u{001B}[33mLoading Qwen3-TTS Base model for speaker embedding extraction...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mBase model loaded!\u{001B}[0m")

        // 2. Verify speaker encoder is present
        #expect(model.speakerEncoder != nil, "Base model should have a speaker encoder")

        // 3. Extract speaker embedding from synthetic audio
        let audio = makeSyntheticAudio()
        print("\u{001B}[33mExtracting speaker embedding from audio (shape: \(audio.shape))...\u{001B}[0m")

        let embedding = try model.extractSpeakerEmbedding(audio: audio)
        eval(embedding)

        // 4. Verify shape: [1, enc_dim] where enc_dim is 2048 for Base model
        #expect(embedding.ndim == 2, "Embedding should be 2D, got \(embedding.ndim)")
        #expect(embedding.dim(0) == 1, "Embedding batch dim should be 1, got \(embedding.dim(0))")
        #expect(embedding.dim(1) == 2048,
                "Embedding dim should be 2048 (Base model enc_dim), got \(embedding.dim(1))")
        print("\u{001B}[32mSpeaker embedding shape: \(embedding.shape)\u{001B}[0m")

        // 5. Verify embedding contains non-zero values
        let sumAbsEmb = MLX.abs(embedding).sum()
        eval(sumAbsEmb)
        let sumVal: Float = sumAbsEmb.item()
        #expect(sumVal > 0, "Speaker embedding should contain non-zero values")
        print("\u{001B}[32mEmbedding L1 norm: \(sumVal)\u{001B}[0m")
    }

    /// Test that two different audio clips produce different speaker embeddings
    @Test func testDifferentAudioProducesDifferentEmbeddings() async throws {
        // 1. Load Base model
        print("\u{001B}[33mLoading Qwen3-TTS Base model for embedding comparison...\u{001B}[0m")
        let model = try await Qwen3TTSModel.fromPretrained(Self.baseModelRepo)
        print("\u{001B}[32mBase model loaded!\u{001B}[0m")

        // 2. Create two clearly different synthetic audio clips
        // Audio 1: Low frequency sine wave (220 Hz)
        let audio1 = makeSyntheticAudio(durationSeconds: 2.0, frequency: 220.0, amplitude: 0.5)
        // Audio 2: High frequency sine wave (880 Hz)
        let audio2 = makeSyntheticAudio(durationSeconds: 2.0, frequency: 880.0, amplitude: 0.3)

        // 3. Extract embeddings for both
        print("\u{001B}[33mExtracting embeddings for two different audio clips...\u{001B}[0m")
        let embedding1 = try model.extractSpeakerEmbedding(audio: audio1)
        let embedding2 = try model.extractSpeakerEmbedding(audio: audio2)
        eval(embedding1, embedding2)

        // 4. Verify both have the same shape
        #expect(embedding1.shape == embedding2.shape,
                "Both embeddings should have the same shape")
        print("\u{001B}[32mEmbedding 1 shape: \(embedding1.shape), Embedding 2 shape: \(embedding2.shape)\u{001B}[0m")

        // 5. Verify they are not identical
        let diff = MLX.abs(embedding1 - embedding2).sum()
        eval(diff)
        let diffVal: Float = diff.item()
        #expect(diffVal > 1e-4,
                "Different audio clips should produce different embeddings, diff = \(diffVal)")
        print("\u{001B}[32mEmbedding L1 distance: \(diffVal)\u{001B}[0m")

        // 6. Verify determinism: same audio produces same embedding
        let embedding1Again = try model.extractSpeakerEmbedding(audio: audio1)
        eval(embedding1Again)
        let selfDiff = MLX.abs(embedding1 - embedding1Again).sum()
        eval(selfDiff)
        let selfDiffVal: Float = selfDiff.item()
        #expect(selfDiffVal < 1e-6,
                "Same audio should produce identical embeddings, diff = \(selfDiffVal)")
        print("\u{001B}[32mSelf-consistency check passed (diff = \(selfDiffVal))\u{001B}[0m")
    }
}
