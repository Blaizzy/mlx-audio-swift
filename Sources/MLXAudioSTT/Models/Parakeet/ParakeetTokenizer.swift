import Foundation

enum ParakeetTokenizer {
    static func decode(tokens: [Int], vocabulary: [String]) -> String {
        var parts: [String] = []
        parts.reserveCapacity(tokens.count)

        for token in tokens {
            guard token >= 0, token < vocabulary.count else { continue }
            parts.append(vocabulary[token].replacingOccurrences(of: "▁", with: " "))
        }

        return parts.joined()
    }
}
