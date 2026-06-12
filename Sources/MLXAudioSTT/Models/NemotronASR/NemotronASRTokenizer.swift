import Foundation

enum NemotronASRTokenizer {
    static func isLanguageTag(_ piece: String) -> Bool {
        piece.hasPrefix("<") && piece.hasSuffix(">") && piece.contains("-")
    }

    static func isSpecialToken(_ tokenId: Int, vocabulary: [String]) -> Bool {
        guard tokenId >= 0, tokenId < vocabulary.count else { return false }
        let piece = vocabulary[tokenId]
        return isLanguageTag(piece)
            || (piece.hasPrefix("<|") && piece.hasSuffix("|>"))
            || piece == "<unk>"
            || piece == "<pad>"
    }

    static func decode(tokens: [Int], vocabulary: [String], stripLanguageTags: Bool = true) -> String {
        var parts: [String] = []
        parts.reserveCapacity(tokens.count)

        for token in tokens {
            guard token >= 0, token < vocabulary.count else { continue }
            let piece = vocabulary[token]
            if stripLanguageTags, isSpecialToken(token, vocabulary: vocabulary) {
                continue
            }
            parts.append(piece.replacingOccurrences(of: "▁", with: " "))
        }

        return parts.joined()
    }

    static func detectedLanguage(tokens: [Int], vocabulary: [String]) -> String? {
        for token in tokens {
            guard token >= 0, token < vocabulary.count else { continue }
            let piece = vocabulary[token]
            if isLanguageTag(piece) {
                return String(piece.dropFirst().dropLast())
            }
        }
        return nil
    }
}
