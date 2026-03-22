import Testing
@testable import MLXAudioG2P

struct MLXAudioG2PTextNormalizerTests {
    @Test func collapsesRepeatedWhitespace() {
        let n = TextNormalizer.englishDefault
        #expect(n.normalize("Hello   world") == "Hello world")
    }

    @Test func normalizationIsIdempotent() {
        let n = TextNormalizer.englishDefault
        let once = n.normalize("Hello world")
        #expect(n.normalize(once) == once)
    }

    @Test func normalizesSmartQuotes() {
        let n = TextNormalizer.englishDefault
        #expect(n.normalize("\u{201C}Hi\u{201D}") == "\"Hi\"")
    }

    @Test func normalizesEmDash() {
        let n = TextNormalizer.englishDefault
        #expect(n.normalize("word\u{2014}word") == "word - word")
    }

    @Test func trimsLeadingTrailingWhitespace() {
        let n = TextNormalizer.englishDefault
        #expect(n.normalize("  hello  ") == "hello")
    }

    @Test func tokenizerSplitsPunctuation() {
        let t = TextTokenizer.englishDefault
        let surfaces = t.tokenize("Hi, Sam!").map(\.surface)
        #expect(surfaces == ["Hi", ",", " ", "Sam", "!"])
    }

    @Test func tokenRangesReconstructText() {
        let t = TextTokenizer.englishDefault
        let text = "Hello world"
        let tokens = t.tokenize(text)
        let reconstructed = tokens.map { String(text[$0.rangeInNormalizedText]) }.joined()
        #expect(reconstructed == text)
    }

    @Test func tokenizerAssignsCorrectKinds() {
        let t = TextTokenizer.englishDefault
        let tokens = t.tokenize("Hello, world!")
        #expect(tokens[0].kind == .word)
        #expect(tokens[1].kind == .punctuation)
        #expect(tokens[2].kind == .whitespace)
        #expect(tokens[3].kind == .word)
        #expect(tokens[4].kind == .punctuation)
    }
}
