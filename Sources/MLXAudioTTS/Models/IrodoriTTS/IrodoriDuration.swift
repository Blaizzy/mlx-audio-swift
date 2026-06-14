import Foundation
import MLX

// MARK: - Duration features (v3 duration predictor input)
// Mirrors mlx_audio/tts/models/irodori_tts/duration.py. The predictor NETWORK
// itself lives with the DiT modules (model.py) — this file builds its 14-dim
// per-utterance feature vector.

let irodoriAllowedAnnotationEmojis: [String] = [
    "⏩", "⏱️", "⏸️", "🌬️", "🍭", "🎛️", "🎭", "🎵", "🐢", "🐱", "👂", "👃", "👅",
    "👌", "👏", "💋", "💥", "💦", "💪", "📄", "📞", "📢", "📣", "😆", "😊", "😌",
    "😎", "😏", "😒", "😖", "😟", "😠", "😪", "😭", "😮", "😮\u{200D}💨", "😰",
    "😱", "😲", "😴", "🙄", "🙏", "🤐", "🤔", "🤢", "🤧", "🤭", "🥤", "🥱", "🥴",
    "🥵", "🥹", "🥺", "🫣", "🫶", "📖",
]

/// Longest-first matching, mirroring the Python regex alternation order.
private let irodoriEmojisLongestFirst =
    irodoriAllowedAnnotationEmojis.sorted { $0.count > $1.count }

func irodoriCountAnnotationEmojis(_ text: String) -> Int {
    var count = 0
    var rest = Substring(text)
    outer: while !rest.isEmpty {
        for e in irodoriEmojisLongestFirst where rest.hasPrefix(e) {
            count += 1
            rest = rest.dropFirst(e.count)
            continue outer
        }
        rest = rest.dropFirst()
    }
    return count
}

private func log1pCap(_ count: Int, _ cap: Int) -> Float {
    let v = Float(min(max(count, 0), cap))
    return log1p(v) / log1p(Float(cap))
}

private func log1pCapFloat(_ value: Float, _ cap: Float) -> Float {
    let v = min(max(value, 0), cap)
    return log1p(v) / log1p(cap)
}

private func isKana(_ s: UnicodeScalar) -> Bool {
    (0x3040...0x309F).contains(s.value) || (0x30A0...0x30FF).contains(s.value)
}

private func isKanji(_ s: UnicodeScalar) -> Bool {
    (0x3400...0x4DBF).contains(s.value) || (0x4E00...0x9FFF).contains(s.value)
        || (0xF900...0xFAFF).contains(s.value) || (0x20000...0x2FA1F).contains(s.value)
}

private func isAsciiAlnum(_ s: UnicodeScalar) -> Bool {
    s.isASCII && ((0x30...0x39).contains(s.value) || (0x41...0x5A).contains(s.value)
        || (0x61...0x7A).contains(s.value))
}

private func occurrences(_ text: String, _ needle: Character) -> Int {
    text.reduce(0) { $0 + ($1 == needle ? 1 : 0) }
}

/// Build the (B, 14) duration-feature matrix for the v3 duration predictor.
/// Python counts characters via `len(str)` (scalar-ish); we count unicode scalars
/// for kana/kanji/alnum and `String.count` for char_count — matching CPython's
/// code-point semantics closely enough for these capped/normalised features.
func irodoriBuildDurationFeatures(
    texts: [String],
    tokenCounts: [Int],
    maxTextLen: Int,
    hasSpeaker: [Bool]
) -> MLXArray {
    precondition(texts.count == tokenCounts.count && texts.count == hasSpeaker.count)

    var rows = [Float]()
    rows.reserveCapacity(texts.count * 14)

    for (i, text) in texts.enumerated() {
        let tokenCount = tokenCounts[i]
        let speakerAvailable = hasSpeaker[i]

        let charCount = max(text.unicodeScalars.count, 1)
        var kana = 0, kanji = 0, alnum = 0
        for s in text.unicodeScalars {
            if isKana(s) { kana += 1 }
            if isKanji(s) { kanji += 1 }
            if isAsciiAlnum(s) { alnum += 1 }
        }
        let emoji = irodoriCountAnnotationEmojis(text)

        let period = occurrences(text, "。") + occurrences(text, ".")
        let comma = occurrences(text, "、") + occurrences(text, ",")
        let longVowel = occurrences(text, "ー")
        let ellipsis = occurrences(text, "…")
        let exclamation = occurrences(text, "！") + occurrences(text, "!")
        let question = occurrences(text, "？") + occurrences(text, "?")

        rows.append(contentsOf: [
            min(max(Float(tokenCount), 0), Float(maxTextLen)) / Float(maxTextLen),
            log1pCapFloat(Float(charCount), 512),
            Float(tokenCount) / Float(charCount),
            log1pCap(period, 8),
            log1pCap(comma, 16),
            log1pCap(longVowel, 8),
            log1pCap(ellipsis, 8),
            log1pCap(exclamation, 8),
            log1pCap(question, 8),
            log1pCap(emoji, 8),
            Float(kana) / Float(charCount),
            Float(kanji) / Float(charCount),
            Float(alnum) / Float(charCount),
            speakerAvailable ? 1.0 : 0.0,
        ])
    }

    return MLXArray(rows).reshaped(texts.count, 14)
}
