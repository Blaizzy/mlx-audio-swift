import Foundation
import Testing
@testable import MLXAudioCore

@Suite("SentencePiece Unicode")
struct SentencePieceTokenizerTests {
    @Test func canonicallyEquivalentVocabularyPiecesKeepDistinctIDs() throws {
        // U+F97C canonically decomposes to U+826F, but SentencePiece assigns
        // different IDs and scores to these two UTF-8 byte sequences.
        let json = #"{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0],["▁",-1],["良",-2],["良",-3],["é",-2],["é",-3]]}}"#
        let tokenizer = try SentencePieceTokenizer(tokenizerJSONData: Data(json.utf8))
        #expect(tokenizer.tokenID(for: "良") == 2)
        #expect(tokenizer.tokenID(for: "良") == 3)
        #expect(tokenizer.encodeWithByteFallback("良") == [1, 2])
        #expect(tokenizer.encodeWithByteFallback("良") == [1, 3])
        #expect(tokenizer.encodeWithByteFallback("é") == [1, 4])
        #expect(tokenizer.encodeWithByteFallback("e\u{301}") == [1, 5])
    }

    @Test func unigramLatticeCanSplitWithinAGraphemeCluster() throws {
        let json = #"{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0],["▁",-1],["e",-1],["́",-1],["👩",-1],["🏽",-1]]}}"#
        let tokenizer = try SentencePieceTokenizer(tokenizerJSONData: Data(json.utf8))
        #expect(tokenizer.encodeWithByteFallback("e\u{301}") == [1, 2, 3])
        #expect(tokenizer.encodeWithByteFallback("👩🏽") == [1, 4, 5])
    }

    @Test func bpeAtomsUseUnicodeScalarsAndExactBytes() throws {
        let json = #"{"model":{"type":"BPE","unk_id":0,"vocab":[["<unk>",0],["▁",-1],["e",-1],["́",-1],["良",-1],["良",-2]]}}"#
        let tokenizer = try SentencePieceTokenizer(tokenizerJSONData: Data(json.utf8))
        #expect(tokenizer.encodeWithByteFallback("e\u{301}") == [1, 2, 3])
        #expect(tokenizer.encodeWithByteFallback("良") == [1, 4])
        #expect(tokenizer.encodeWithByteFallback("良") == [1, 5])
    }
}
