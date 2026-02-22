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

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs


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


// MARK: - Kokoro Unit Tests

struct KokoroTokenizerTests {

    @Test func testTokenizeBasicPhonemes() {
        let vocab: [String: Int] = ["h": 1, "ɛ": 2, "l": 3, "oʊ": 4]
        // Single-char mapping: each character is looked up individually
        let tokens = KokoroTokenizer.tokenize(phonemizedText: "hɛl", vocab: vocab)
        #expect(tokens == [1, 2, 3], "Should map known phoneme characters to IDs")
    }

    @Test func testTokenizeUnknownCharsDropped() {
        let vocab: [String: Int] = ["a": 1, "b": 2]
        let tokens = KokoroTokenizer.tokenize(phonemizedText: "abc", vocab: vocab)
        #expect(tokens == [1, 2], "Unknown characters should be dropped via compactMap")
    }

    @Test func testTokenizeEmptyString() {
        let vocab: [String: Int] = ["a": 1]
        let tokens = KokoroTokenizer.tokenize(phonemizedText: "", vocab: vocab)
        #expect(tokens.isEmpty, "Empty input should produce empty output")
    }
}

struct KokoroConfigTests {

    @Test func testConfigDecodesFromJSON() throws {
        let json = """
        {
            "istftnet": {
                "upsample_kernel_sizes": [20, 12],
                "upsample_rates": [10, 6],
                "gen_istft_hop_size": 5,
                "gen_istft_n_fft": 16,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                "resblock_kernel_sizes": [3, 7],
                "upsample_initial_channel": 512
            },
            "dim_in": 64,
            "dropout": 0.2,
            "hidden_dim": 512,
            "max_conv_dim": 512,
            "max_dur": 50,
            "multispeaker": false,
            "n_layer": 3,
            "n_mels": 80,
            "n_token": 178,
            "style_dim": 128,
            "text_encoder_kernel_size": 5,
            "plbert": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_hidden_layers": 12,
                "dropout": 0.1
            },
            "vocab": {"a": 1, "b": 2, "c": 3}
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(KokoroConfig.self, from: data)

        #expect(config.hiddenDim == 512)
        #expect(config.styleDim == 128)
        #expect(config.nToken == 178)
        #expect(config.nMels == 80)
        #expect(config.nLayer == 3)
        #expect(config.vocab.count == 3)
        #expect(config.plbert.hiddenSize == 768)
        #expect(config.plbert.numHiddenLayers == 12)
        #expect(config.istftNet.genIstftHopSize == 5)
        #expect(config.istftNet.upsampleRates == [10, 6])
    }
}


// MARK: - TextProcessor Unit Tests

/// Mock G2P processor that returns a fixed IPA string for testing.
struct MockG2PProcessor: TextProcessor {
    let fixedOutput: String

    func process(text: String, language: String?) throws -> String {
        return fixedOutput
    }
}

/// Mock G2P processor that verifies it receives the expected input.
struct CapturingG2PProcessor: TextProcessor {
    let expectedInput: String
    let output: String

    func process(text: String, language: String?) throws -> String {
        // Verify the processor receives the original plain text
        assert(text == expectedInput, "Expected '\(expectedInput)' but got '\(text)'")
        return output
    }
}

struct TextProcessorProtocolTests {

    @Test func testTextProcessorProtocolConformance() {
        // Verify a basic implementation works
        let processor = MockG2PProcessor(fixedOutput: "hɛloʊ wˈɜɹld")
        let result = try! processor.process(text: "Hello world", language: nil)
        #expect(result == "hɛloʊ wˈɜɹld", "Mock processor should return fixed output")
    }

    @Test func testTextProcessorWithLanguage() {
        // Verify language parameter is passed through
        struct LanguageAwareProcessor: TextProcessor {
            func process(text: String, language: String?) throws -> String {
                if language == "en-gb" {
                    return "brɪtɪʃ"
                }
                return "əmɛɹɪkən"
            }
        }

        let processor = LanguageAwareProcessor()
        let usResult = try! processor.process(text: "test", language: "en-us")
        let gbResult = try! processor.process(text: "test", language: "en-gb")
        #expect(usResult == "əmɛɹɪkən")
        #expect(gbResult == "brɪtɪʃ")
    }

    @Test func testTextProcessorThrowsError() {
        struct FailingProcessor: TextProcessor {
            func process(text: String, language: String?) throws -> String {
                throw NSError(domain: "G2PError", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "Unsupported language"
                ])
            }
        }

        let processor = FailingProcessor()
        #expect(throws: (any Error).self) {
            try processor.process(text: "test", language: "unsupported")
        }
    }
}
