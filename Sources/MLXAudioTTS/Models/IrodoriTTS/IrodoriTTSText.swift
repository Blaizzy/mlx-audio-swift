import Foundation
import MLX
import Tokenizers

// MARK: - Japanese text normalisation
// Mirrors mlx_audio/tts/models/irodori_tts/text.py (itself ported from
// Irodori-TTS/irodori_tts/text_normalization.py).

private let irodoriRegexReplacements: [(pattern: String, replacement: String)] = [
    ("\\t", ""),
    ("\\[n\\]", ""),
    ("\u{202F}", ""),               // narrow no-break space
    ("\u{3000}", ""),               // ideographic space
    ("[;▼♀♂《》≪≫①②③④⑤⑥]", ""),
    ("[\u{02d7}\u{2010}-\u{2015}\u{2043}\u{2212}\u{23af}\u{23e4}\u{2500}\u{2501}\u{2e3a}\u{2e3b}]", ""),
    ("[\u{ff5e}\u{301C}]", "ー"),
    ("？", "?"),
    ("！", "!"),
    ("[●◯〇]", "○"),
    ("♥", "♡"),
]

private func irodoriWidthFold(_ scalar: UnicodeScalar) -> UnicodeScalar {
    let v = scalar.value
    // Fullwidth A-Z a-z → halfwidth
    if (0xFF21...0xFF3A).contains(v) { return UnicodeScalar(v - 0xFF21 + 0x41)! }
    if (0xFF41...0xFF5A).contains(v) { return UnicodeScalar(v - 0xFF41 + 0x61)! }
    // Fullwidth 0-9 → halfwidth
    if (0xFF10...0xFF19).contains(v) { return UnicodeScalar(v - 0xFF10 + 0x30)! }
    return scalar
}

// Halfwidth katakana → fullwidth katakana (positional map, mirrors str.maketrans)
private let irodoriHWKana = Array("ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ")
private let irodoriFWKana = Array("ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン")
private let irodoriKanaMap: [Character: Character] = {
    var m = [Character: Character]()
    for (h, f) in zip(irodoriHWKana, irodoriFWKana) { m[h] = f }
    return m
}()

/// Normalise Japanese text for Irodori TTS input.
/// - Removes noise characters; folds fullwidth alnum → halfwidth and
///   halfwidth katakana → fullwidth; strips surrounding brackets and
///   trailing 。/、 punctuation; collapses 3+ ellipses to two.
func irodoriNormalizeText(_ input: String) -> String {
    var text = input

    for (pattern, replacement) in irodoriRegexReplacements {
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let range = NSRange(text.startIndex..., in: text)
            text = regex.stringByReplacingMatches(in: text, range: range, withTemplate: replacement)
        }
    }

    // Width folding + kana widening
    text = String(String.UnicodeScalarView(text.unicodeScalars.map(irodoriWidthFold)))
    text = String(text.map { irodoriKanaMap[$0] ?? $0 })

    // Collapse runs of 3+ ellipses to double
    if let regex = try? NSRegularExpression(pattern: "…{3,}") {
        let range = NSRange(text.startIndex..., in: text)
        text = regex.stringByReplacingMatches(in: text, range: range, withTemplate: "……")
    }

    // Strip surrounding bracket pairs
    for (open, close) in [("「", "」"), ("『", "』"), ("（", "）"), ("【", "】"), ("(", ")")] {
        if text.hasPrefix(open) && text.hasSuffix(close) && text.count >= 2 {
            text = String(text.dropFirst().dropLast())
        }
    }

    // Strip trailing Japanese sentence-ending punctuation
    while text.hasSuffix("。") || text.hasSuffix("、") {
        text = String(text.dropLast())
    }

    return text
}

// MARK: - Upstream text normalisation (v4)
// Exact port of the current mlx_audio text.py `normalize_text`, which mirrors
// Irodori-TTS/irodori_tts/text_normalization.py step for step (keeps 。/、, folds
// "..", NFKC, depth-aware outer-bracket stripping). v1–v3 keep `irodoriNormalizeText`
// above so their output does not change.

private let irodoriUpstreamSimpleReplacements: [(String, String)] = [
    ("\t", ""),
    ("[n]", ""),
    ("\\[n\\]", ""),
    ("\u{3000}", ""),  // ideographic space
    ("？", "?"),
    ("！", "!"),
    ("♥", "♡"),
    ("●", "○"),
    ("◯", "○"),
    ("〇", "○"),
]

private let irodoriUpstreamRegexReplacements: [(NSRegularExpression, String)] = [
    ("[;▼♀♂《》≪≫①②③④⑤⑥]", ""),
    ("[\u{02d7}\u{2010}-\u{2015}\u{2043}\u{2212}\u{23af}\u{23e4}\u{2500}\u{2501}\u{2e3a}\u{2e3b}]", ""),
    ("[\u{ff5e}\u{301C}]", "ー"),
    ("…{3,}", "……"),
].map { (try! NSRegularExpression(pattern: $0.0), $0.1) }

private let irodoriBracketPairs: [Unicode.Scalar: Unicode.Scalar] = [
    "「": "」", "『": "』", "（": "）", "【": "】", "(": ")",
]

/// Remove bracket pairs that enclose the whole string, repeatedly. Depth tracking
/// matters: in 「前半」と「後半」 the leading 「 closes before the end, so nothing is stripped.
/// Works on Unicode scalars, like Python's code-point string indexing.
func irodoriStripOuterBrackets(_ input: String) -> String {
    var text = Array(input.unicodeScalars)
    while text.count >= 2, let close = irodoriBracketPairs[text[0]], close == text[text.count - 1] {
        let open = text[0]
        var depth = 0
        var enclosesAll = true
        for (i, c) in text.enumerated() {
            if c == open { depth += 1 } else if c == close { depth -= 1 }
            if depth == 0 && i < text.count - 1 {
                enclosesAll = false
                break
            }
        }
        guard enclosesAll && depth == 0 else { break }
        text = Array(text[1 ..< text.count - 1])
    }
    var view = String.UnicodeScalarView()
    view.append(contentsOf: text)
    return String(view)
}

/// Normalise Japanese text for v4 TTS input (callers strip the result, as upstream does).
func irodoriNormalizeTextUpstream(_ input: String) -> String {
    var text = input
    for (old, new) in irodoriUpstreamSimpleReplacements {
        text = text.replacingOccurrences(of: old, with: new, options: .literal)
    }
    for (regex, replacement) in irodoriUpstreamRegexReplacements {
        let range = NSRange(text.startIndex..., in: text)
        text = regex.stringByReplacingMatches(in: text, range: range, withTemplate: replacement)
    }
    text = irodoriStripOuterBrackets(text)
    // NFKC as NFKD + NFC: Foundation's one-step NFKC leaves halfwidth voiced kana
    // (ｷﾞ) decomposed, while Python's unicodedata composes them (ギ).
    text = text.decomposedStringWithCompatibilityMapping.precomposedStringWithCanonicalMapping
    text = text.replacingOccurrences(of: "...", with: "…", options: .literal)
    text = text.replacingOccurrences(of: "..", with: "…", options: .literal)
    return text
}

/// Python's `str.isspace()` set (Foundation's whitespace set lacks U+001C–U+001F).
private let irodoriPythonWhitespace: Set<Unicode.Scalar> = {
    var set = Set<Unicode.Scalar>()
    for v: UInt32 in [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680,
                      0x2028, 0x2029, 0x202F, 0x205F, 0x3000] {
        set.insert(Unicode.Scalar(v)!)
    }
    for v: UInt32 in 0x2000 ... 0x200A { set.insert(Unicode.Scalar(v)!) }
    return set
}()

extension String {
    /// Python `str.strip()` with no arguments.
    func irodoriPythonStripped() -> String {
        let scalars = Array(unicodeScalars)
        guard let first = scalars.firstIndex(where: { !irodoriPythonWhitespace.contains($0) }),
              let last = scalars.lastIndex(where: { !irodoriPythonWhitespace.contains($0) })
        else { return "" }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars[first ... last])
        return String(view)
    }
}

// MARK: - Tokenisation

/// Tokenise a single string with an HF tokenizer, mirroring Irodori's
/// PretrainedTextTokenizer: no special tokens from the tokenizer, manual BOS,
/// right-padding to `maxLength` with the pad (or EOS) token id.
/// Returns `(ids: (1, maxLength) int32, mask: (1, maxLength) bool)`.
func irodoriEncodeText(
    _ text: String,
    tokenizer: Tokenizer,
    maxLength: Int,
    addBos: Bool = true
) throws -> (ids: MLXArray, mask: MLXArray) {
    var tokenIds = tokenizer.encode(text: text, addSpecialTokens: false)

    if addBos {
        // swift-transformers' bosTokenId only checks the (Unigram) model vocab; llm-jp's
        // <s> lives in added_tokens, so fall back to a token lookup that sees them.
        guard let bos = tokenizer.bosTokenId ?? tokenizer.convertTokenToId("<s>") else {
            throw IrodoriTTSError.tokenizer("Tokenizer has no bos_token_id but add_bos=true")
        }
        tokenIds.insert(bos, at: 0)
    }

    if tokenIds.count > maxLength {
        tokenIds = Array(tokenIds.prefix(maxLength))
    }
    let n = tokenIds.count

    // Padding is masked out, so the exact id is irrelevant — resolve </s>, else 0.
    let padId = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("</s>") ?? 0

    var padded = tokenIds
    padded.append(contentsOf: Array(repeating: padId, count: maxLength - n))

    let ids = MLXArray(padded.map { Int32($0) }).reshaped(1, maxLength)
    var maskVals = [Bool](repeating: false, count: maxLength)
    for i in 0..<n { maskVals[i] = true }
    let mask = MLXArray(maskVals).reshaped(1, maxLength)

    return (ids, mask)
}

// MARK: - Errors

public enum IrodoriTTSError: Error, LocalizedError {
    case tokenizer(String)
    case weights(String)
    case generation(String)

    public var errorDescription: String? {
        switch self {
        case .tokenizer(let m): return "IrodoriTTS tokenizer error: \(m)"
        case .weights(let m): return "IrodoriTTS weights error: \(m)"
        case .generation(let m): return "IrodoriTTS generation error: \(m)"
        }
    }
}
