import Testing
import Foundation
@testable import MLXAudioG2P
import MLXAudioCore

// MARK: - Unit Tests (no model required)

struct ByT5TokenizerTests {
    let tokenizer = ByT5Tokenizer()

    @Test func encodeASCII() {
        let ids = tokenizer.encode("hi")
        // 'h'=104 +3=107, 'i'=105 +3=108, EOS=1
        #expect(ids == [107, 108, 1])
    }

    @Test func decodeRoundTrip() {
        let original = "hello"
        let decoded = tokenizer.decode(tokenizer.encode(original))
        #expect(decoded == original)
    }

    @Test func decodeStopsAtEOS() {
        let ids: [Int32] = [107, 108, 1, 109, 110]
        #expect(tokenizer.decode(ids) == "hi")
    }

    @Test func decodeSkipsPadAndUnk() {
        let ids: [Int32] = [0, 2, 107, 108, 1]
        #expect(tokenizer.decode(ids) == "hi")
    }

    @Test func formatInput() {
        #expect(tokenizer.formatInput("hello", language: "eng-us") == "<eng-us>: hello")
    }

    @Test func encodeMultibyteUTF8() {
        let ids = tokenizer.encode("ü")
        #expect(ids.count == 3) // ü = 2 UTF-8 bytes + EOS
        #expect(ids.last == ByT5Tokenizer.eosTokenId)
    }

    @Test func emptyStringEncodesEOSOnly() {
        #expect(tokenizer.encode("") == [1])
    }
}

struct WeightSanitizationTests {
    @Test func sharedToWTE() {
        #expect(WeightLoader.sanitizeKey("shared.weight") == "wte.weight")
    }

    @Test func encoderSelfAttention() {
        #expect(
            WeightLoader.sanitizeKey("encoder.block.0.layer.0.SelfAttention.q.weight")
                == "encoder.layers.0.attention.query_proj.weight"
        )
    }

    @Test func decoderSelfAttention() {
        #expect(
            WeightLoader.sanitizeKey("decoder.block.0.layer.0.SelfAttention.q.weight")
                == "decoder.layers.0.self_attention.query_proj.weight"
        )
    }

    @Test func decoderCrossAttention() {
        #expect(
            WeightLoader.sanitizeKey("decoder.block.0.layer.1.EncDecAttention.k.weight")
                == "decoder.layers.0.cross_attention.key_proj.weight"
        )
    }

    @Test func layerNormKeys() {
        #expect(
            WeightLoader.sanitizeKey("encoder.block.0.layer.0.layer_norm.weight")
                == "encoder.layers.0.ln1.weight"
        )
    }

    @Test func filtersIgnoredKeys() {
        #expect(
            WeightLoader.sanitizeKey(
                "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.embeddings.weight"
            ) == nil
        )
    }
}

struct T5ConfigTests {
    private static let sampleJSON = """
    {
        "vocab_size": 384,
        "d_model": 256,
        "d_ff": 512,
        "d_kv": 32,
        "num_heads": 4,
        "num_layers": 12,
        "num_decoder_layers": 4,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "layer_norm_epsilon": 1e-6,
        "feed_forward_proj": "gated-gelu",
        "tie_word_embeddings": true,
        "decoder_start_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 0
    }
    """

    @Test func decodesFromJSON() throws {
        let config = try JSONDecoder().decode(
            T5Config.self, from: Self.sampleJSON.data(using: .utf8)!
        )
        #expect(config.vocabSize == 384)
        #expect(config.dModel == 256)
        #expect(config.numHeads == 4)
        #expect(config.numLayers == 12)
        #expect(config.numDecoderLayers == 4)
        #expect(config.innerDim == 128)
        #expect(config.tieWordEmbeddings == true)
    }

    @Test func loadsFromDirectory() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try Self.sampleJSON.write(
            to: tmpDir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )

        let config = try T5Config.load(from: tmpDir)
        #expect(config.vocabSize == 384)
        #expect(config.innerDim == 128)
    }
}

// MARK: - Integration Tests (downloads model, requires Metal + network)

struct NeuralG2PIntegrationTests {
    private static let networkEnabled = ProcessInfo.processInfo.environment["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1"

    private static func modelDirectory() async throws -> URL {
        try await ModelUtils.resolveOrDownloadModel(
            repoID: "beshkenadze/g2p-multilingual-byT5-tiny-mlx",
            requiredExtension: "safetensors"
        )
    }

    @Test func convertEnglishWord() async throws {
        guard Self.networkEnabled else { print("Skipping network NeuralG2P test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable."); return }
        let g2p = try G2P(modelDirectory: try await Self.modelDirectory())
        let ipa = g2p.convert("hello", language: "eng-us")
        #expect(!ipa.isEmpty)
        #expect(ipa.contains("h") || ipa.contains("ɛ") || ipa.contains("l"))
    }

    @Test func batchConvert() async throws {
        guard Self.networkEnabled else { return }
        let g2p = try G2P(modelDirectory: try await Self.modelDirectory())
        let results = g2p.convert(["hello", "world"], language: "eng-us")
        #expect(results.count == 2)
        #expect(results.allSatisfy { !$0.isEmpty })
    }

    @Test func neuralPhonemizerProducesPhonemeUnits() async throws {
        guard Self.networkEnabled else { return }
        let phonemizer = try NeuralPhonemizer(
            modelDirectory: try await Self.modelDirectory()
        )
        let phonemes = try phonemizer.phonemize("test")
        #expect(!phonemes.isEmpty)
        #expect(phonemes.allSatisfy { !$0.symbol.isEmpty })
    }

    @Test func deterministicOutput() async throws {
        guard Self.networkEnabled else { return }
        let g2p = try G2P(modelDirectory: try await Self.modelDirectory())
        let first = g2p.convert("phone", language: "eng-us")
        let second = g2p.convert("phone", language: "eng-us")
        #expect(first == second)
    }

    @Test func spanishG2PVocabCoverage() async throws {
        guard Self.networkEnabled else { return }
        let g2p = try G2P(modelDirectory: try await Self.modelDirectory(), maxLength: 256)

        // Test individual Spanish words and a short sentence
        let words = ["hola", "mundo", "prueba", "español", "gracias"]
        let sentence = "Hola, esto es una prueba."

        var allIPA = ""
        for word in words {
            let ipa = g2p.convert(word, language: "es")
            fputs("[ES] \(word) → \(ipa)\n", stderr)
            allIPA += ipa
        }
        let sentenceIPA = g2p.convert(sentence, language: "es")
        fputs("[ES] sentence → \(sentenceIPA)\n", stderr)
        allIPA += sentenceIPA

        let kokoroVocabStr: [String] = [
            " ", "!", "\"", "(", ")", ",", ".", ":", ";", "?",
            "A", "I", "O", "Q", "S", "T", "W", "Y",
            "a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "æ", "ç", "ð", "ø", "ŋ", "œ", "ɐ", "ɑ", "ɒ", "ɔ", "ɕ", "ɖ",
            "ə", "ɚ", "ɛ", "ɜ", "ɟ", "ɡ", "ɣ", "ɤ", "ɥ", "ɨ", "ɪ", "ɯ",
            "ɰ", "ɲ", "ɳ", "ɴ", "ɸ", "ɹ", "ɻ", "ɽ", "ɾ", "ʁ", "ʂ", "ʃ",
            "ʈ", "ʊ", "ʋ", "ʌ", "ʎ", "ʒ", "ʔ", "ʝ", "ʣ", "ʤ", "ʥ", "ʦ",
            "ʧ", "ʨ", "ʰ", "ʲ", "ˈ", "ˌ", "ː", "\u{0303}", "β", "θ", "χ",
            "ᵊ", "ᵝ", "ᵻ", "\u{2014}", "\u{201C}", "\u{201D}", "\u{2026}",
            "\u{2192}", "\u{2193}", "\u{2197}", "\u{2198}", "ꭧ",
        ]
        let kokoroVocab = Set(kokoroVocabStr.compactMap(\.first))

        var mapped = 0, unmapped = 0
        var unmappedChars: Set<Character> = []
        for ch in allIPA {
            if kokoroVocab.contains(ch) {
                mapped += 1
            } else {
                unmapped += 1
                unmappedChars.insert(ch)
            }
        }

        let total = mapped + unmapped
        let coverage = total > 0 ? Double(mapped) / Double(total) : 0
        fputs("[ES] Coverage: \(Int(coverage * 100))% (\(mapped)/\(total))\n", stderr)
        fputs("[ES] Unmapped chars: \(unmappedChars.sorted(by: { String($0) < String($1) }))\n", stderr)

        #expect(!allIPA.isEmpty, "Spanish G2P should produce output")
        #expect(coverage > 0.7, "Kokoro vocab should cover most Spanish IPA (got \(Int(coverage * 100))%)")
    }
}
