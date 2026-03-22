import Testing
@testable import MLXAudioG2P

struct MLXAudioG2PAlignmentTests {
    @Test func alignerMapsTokensToPhonemeRanges() throws {
        let pipeline = G2PPipeline.englishWithAlignment()
        let output = try pipeline.convert("Hello world")
        let alignment = try #require(output.alignment)
        #expect(!alignment.isEmpty)
    }

    @Test func alignmentRangesAreMonotonic() throws {
        let pipeline = G2PPipeline.englishWithAlignment()
        let output = try pipeline.convert("Hello world")
        let alignment = try #require(output.alignment)
        for i in 1..<alignment.count {
            #expect(
                alignment[i].phonemeRange.lowerBound >= alignment[i - 1].phonemeRange.upperBound
            )
        }
    }

    @Test func alignmentRangesAreInBounds() throws {
        let pipeline = G2PPipeline.englishWithAlignment()
        let output = try pipeline.convert("Hello world")
        let alignment = try #require(output.alignment)
        for entry in alignment {
            #expect(entry.phonemeRange.lowerBound >= 0)
            #expect(entry.phonemeRange.upperBound <= output.phonemes.units.count)
        }
    }

    @Test func alignmentTokenIndicesMatchWordTokens() throws {
        let pipeline = G2PPipeline.englishWithAlignment()
        let output = try pipeline.convert("Hello world")
        let alignment = try #require(output.alignment)
        for entry in alignment {
            #expect(entry.tokenIndex < output.tokens.count)
            #expect(output.tokens[entry.tokenIndex].kind == .word)
        }
    }

    @Test func alignerThrowsForEmptyPhonemes() {
        let aligner = HeuristicTokenAligner()
        let text = "test"
        let token = G2PToken(
            surface: "test",
            kind: .word,
            rangeInNormalizedText: text.startIndex..<text.endIndex
        )
        #expect(throws: G2PError.self) {
            try aligner.align(
                tokens: [token],
                phonemes: PhonemeSequence(units: [])
            )
        }
    }
}
