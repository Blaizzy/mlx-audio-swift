import Foundation

/// The transcript of a stream session, built from each token's bytes as the token is
/// decoded, so a step's cost does not grow with the transcript.
///
/// `text` always equals `String(decoding: allBytes, as: UTF8.self)`, replacement
/// characters included: bytes only move into `settled` where that decoder has finished
/// a character.
struct VoxtralRealtimeTranscriptText {
    /// Text no later byte can change.
    private(set) var settled = ""
    /// The start of a UTF-8 sequence whose remaining bytes have not arrived yet.
    private var unsettledBytes: [UInt8] = []

    /// The whole transcript so far. While a character is incomplete it ends in U+FFFD,
    /// and reading it builds a new string as long as the transcript.
    var text: String {
        unsettledBytes.isEmpty ? settled : settled + unsettledText
    }

    private var unsettledText: String {
        String(decoding: unsettledBytes, as: UTF8.self)
    }

    mutating func append(_ bytes: [UInt8]) {
        guard !bytes.isEmpty else { return }
        let pending = unsettledBytes + bytes
        let cut = Self.unsettledSuffixStart(pending)
        settled += String(decoding: pending[..<cut], as: UTF8.self)
        unsettledBytes = Array(pending[cut...])
    }

    /// Where the trailing unfinished UTF-8 sequence in `bytes` starts, or `bytes.count`
    /// when there is none. A lead byte always starts a new character, so a one-pass
    /// decoder has finished everything before it.
    static func unsettledSuffixStart(_ bytes: [UInt8]) -> Int {
        var leadIndex = bytes.count
        var continuationCount = 0
        while leadIndex > 0, continuationCount < 3, bytes[leadIndex - 1] & 0xC0 == 0x80 {
            leadIndex -= 1
            continuationCount += 1
        }
        guard leadIndex > 0 else { return bytes.count }

        let expectedContinuations: Int
        switch bytes[leadIndex - 1] {
        case 0xC2...0xDF: expectedContinuations = 1
        case 0xE0...0xEF: expectedContinuations = 2
        case 0xF0...0xF4: expectedContinuations = 3
        default: return bytes.count  // ASCII, a stray continuation, or never valid
        }
        return continuationCount < expectedContinuations ? leadIndex - 1 : bytes.count
    }

    /// A position in the transcript to measure the next delta from.
    struct Mark {
        fileprivate let settledUTF8Count: Int
        fileprivate let lastSettledCharacter: String
        fileprivate let unsettledText: String
    }

    var mark: Mark {
        Mark(
            settledUTF8Count: settled.utf8.count,
            lastSettledCharacter: settled.last.map(String.init) ?? "",
            unsettledText: unsettledText
        )
    }

    /// The text added since `mark`, or the whole transcript when the text at `mark` is
    /// no longer a prefix: a trailing U+FFFD became a real character, or new text
    /// combined with the last character. Only that last character and what follows can
    /// differ, so only that tail is compared.
    func delta(since mark: Mark) -> String {
        // `settled` only grows and ends on a scalar boundary, so the old count is a
        // valid index.
        let start = settled.utf8.index(settled.utf8.startIndex, offsetBy: mark.settledUTF8Count)
        let newlySettled = String(settled.unicodeScalars[start...])
        let oldTail = mark.lastSettledCharacter + mark.unsettledText
        let newTail = mark.lastSettledCharacter + newlySettled + unsettledText
        guard newTail.hasPrefix(oldTail) else { return text }
        return String(newTail.dropFirst(oldTail.count))
    }
}
