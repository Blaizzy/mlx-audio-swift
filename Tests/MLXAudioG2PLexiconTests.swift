import Testing
@testable import MLXAudioG2P

struct MLXAudioG2PLexiconTests {
    @Test func findsWholeWordMatch() {
        let lexicon = InMemoryLexicon(entries: [
            LexiconEntry(grapheme: "hello", phonemes: ["h", "ɛ", "l", "oʊ"]),
        ])
        #expect(lexicon.lookup("hello")?.phonemes == ["h", "ɛ", "l", "oʊ"])
    }

    @Test func lookupIsCaseInsensitive() {
        let lexicon = InMemoryLexicon(entries: [
            LexiconEntry(grapheme: "hello", phonemes: ["h", "ɛ", "l", "oʊ"]),
        ])
        #expect(lexicon.lookup("HELLO") != nil)
        #expect(lexicon.lookup("Hello") != nil)
    }

    @Test func returnsNilForMissingWord() {
        let lexicon = InMemoryLexicon(entries: [])
        #expect(lexicon.lookup("missing") == nil)
    }

    @Test func pipelineUsesLexiconBeforeFallback() throws {
        let pipeline = G2PPipeline.preview()
        let output = try pipeline.convert("hello")
        #expect(output.phonemes.render() == "h ɛ l oʊ")
    }

    @Test func fallbackProducesNonEmptyForLetters() throws {
        let fallback = FallbackPhonemizer()
        let phonemes = try fallback.phonemize("xyz")
        #expect(!phonemes.isEmpty)
    }

    @Test func fallbackThrowsForDigitsOnly() {
        let fallback = FallbackPhonemizer()
        #expect(throws: G2PError.self) {
            try fallback.phonemize("123")
        }
    }
}
