public protocol Phonemizing: Sendable {
    func phonemize(_ grapheme: String) throws -> [PhonemeUnit]
}

public struct FallbackPhonemizer: Phonemizing, Sendable {
    private static let letterPhonemes: [Character: String] = [
        "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ",
        "f": "f", "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ",
        "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɑ",
        "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t",
        "u": "ʌ", "v": "v", "w": "w", "x": "ks", "y": "j",
        "z": "z",
    ]

    public init() {}

    public func phonemize(_ grapheme: String) throws -> [PhonemeUnit] {
        let lowered = grapheme.lowercased()
        var result: [PhonemeUnit] = []

        for char in lowered {
            if let phoneme = Self.letterPhonemes[char] {
                result.append(PhonemeUnit(symbol: phoneme))
            } else if char.isLetter {
                result.append(PhonemeUnit(symbol: String(char)))
            }
        }

        guard !result.isEmpty else {
            throw G2PError.phonemizationFailed(
                token: grapheme,
                reason: "No phonemes produced for token"
            )
        }

        return result
    }
}
