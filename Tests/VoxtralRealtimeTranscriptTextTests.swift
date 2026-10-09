//  The stream session's incremental transcript must read exactly like decoding every
//  token byte in one pass, and each step's delta must equal the one a comparison of
//  whole transcripts gives. No Metal needed.
//
//  Run:
//    xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
//      -only-testing:'MLXAudioTests/VoxtralRealtimeTranscriptTextTests' \
//      CODE_SIGNING_ALLOWED=NO

import Foundation
import Testing

@testable import MLXAudioSTT

struct VoxtralRealtimeTranscriptTextTests {
    /// Multi-byte characters cut at each position, a combining accent, regional
    /// indicators and a zero-width joiner, and bytes that are never valid UTF-8.
    private static let tokenPool: [[UInt8]] = [
        Array("hello".utf8), Array(" ".utf8), Array(" é".utf8), [0xC3], [0xA9],
        Array("e".utf8), Array("\u{301}".utf8), [0xCC], [0x81],
        Array("€".utf8), [0xE2], [0x82, 0xAC], [0xE2, 0x82], [0xAC],
        Array("😀".utf8), [0xF0, 0x9F], [0x98, 0x80], [0xF0], [0x9F, 0x98, 0x80],
        Array("🇫".utf8), Array("🇷".utf8), Array("👩".utf8), [0xE2, 0x80, 0x8D],
        [0xFF], [0x80], [0xC0, 0xAF], [0xED, 0xA0], [0xF4, 0x90], [0xE0, 0x80],
    ]

    /// `text` must equal a one-pass decode of all bytes so far, and each delta must
    /// equal the whole-transcript comparison.
    @Test func matchesOnePassDecodingAndWholeTranscriptDeltas() {
        for seed in 0..<64 {
            var random = SplitMix64(seed: UInt64(seed))
            var transcript = VoxtralRealtimeTranscriptText()
            var allBytes: [UInt8] = []
            var previousText = ""

            for _ in 0..<200 {
                let mark = transcript.mark
                // One decode step appends a few tokens.
                for _ in 0..<Int.random(in: 1...3, using: &random) {
                    let bytes = Self.tokenPool.randomElement(using: &random)!
                    transcript.append(bytes)
                    allBytes += bytes
                }

                let expectedText = String(decoding: allBytes, as: UTF8.self)
                let expectedDelta = expectedText.hasPrefix(previousText)
                    ? String(expectedText.dropFirst(previousText.count))
                    : expectedText
                #expect(
                    Array(transcript.text.unicodeScalars) == Array(expectedText.unicodeScalars),
                    "seed \(seed)"
                )
                #expect(
                    Array(transcript.delta(since: mark).unicodeScalars)
                        == Array(expectedDelta.unicodeScalars),
                    "seed \(seed)"
                )
                previousText = expectedText
            }
        }
    }

    @Test func aDeltaWithNoNewBytesIsEmpty() {
        var transcript = VoxtralRealtimeTranscriptText()
        transcript.append(Array("abc".utf8))
        transcript.append([0xE2, 0x82])
        let mark = transcript.mark
        #expect(transcript.delta(since: mark).isEmpty)
    }

    @Test func holdsBackOnlyAnUnfinishedTrailingSequence() {
        let start = VoxtralRealtimeTranscriptText.unsettledSuffixStart
        #expect(start([]) == 0)
        #expect(start(Array("ab".utf8)) == 2)
        #expect(start([0x61, 0xC3]) == 1)             // "a" + first byte of "é"
        #expect(start([0x61, 0xC3, 0xA9]) == 3)       // "aé" is finished
        #expect(start([0xE2, 0x82]) == 0)             // two of the three bytes of "€"
        #expect(start([0xF0, 0x9F, 0x98]) == 0)       // three of the four bytes of "😀"
        #expect(start([0xF0, 0x9F, 0x98, 0x80]) == 4)
        #expect(start([0x80, 0x80]) == 2)             // stray continuations never finish
        #expect(start([0xFF]) == 1)                   // never valid
        #expect(start([0xC0]) == 1)                   // overlong lead, never valid
        #expect(start([0xC3, 0xA9, 0x80]) == 3)       // a finished character, then a stray
    }

    @Test func textEndsInAReplacementCharacterWhileACharacterIsIncomplete() {
        var transcript = VoxtralRealtimeTranscriptText()
        transcript.append(Array("caf".utf8))
        transcript.append([0xC3])
        #expect(transcript.text == "caf\u{FFFD}")
        #expect(transcript.settled == "caf")

        let mark = transcript.mark
        transcript.append([0xA9])
        #expect(transcript.text == "café")
        // "caf\u{FFFD}" is not a prefix of "café", so the step reports the whole text.
        #expect(transcript.delta(since: mark) == "café")
    }
}

/// Small deterministic generator so every run sees the same token sequences.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
