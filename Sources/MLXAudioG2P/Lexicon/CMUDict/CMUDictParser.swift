import Foundation

public struct CMUDictRawEntry: Sendable, Hashable {
    public let word: String
    public let arpabet: [String]
    public let variant: Int?

    public init(word: String, arpabet: [String], variant: Int?) {
        self.word = word
        self.arpabet = arpabet
        self.variant = variant
    }
}

public enum CMUDictParser {
    public static func parseLine(_ line: String) -> CMUDictRawEntry? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, !trimmed.hasPrefix(";;;") else { return nil }

        // Split on first whitespace: word is first token, pronunciation is the rest.
        // Handles both single-space (raw cmudict.dict) and double-space formats.
        guard let firstSpace = trimmed.firstIndex(of: " ") else { return nil }

        let wordPart = String(trimmed[trimmed.startIndex..<firstSpace])
        let pronPart = String(trimmed[trimmed.index(after: firstSpace)...])
            .trimmingCharacters(in: .whitespaces)

        guard !wordPart.isEmpty, !pronPart.isEmpty else { return nil }

        let word: String
        let variant: Int?

        if let parenStart = wordPart.firstIndex(of: "("),
           let parenEnd = wordPart.firstIndex(of: ")"),
           parenStart < parenEnd {
            word = String(wordPart[wordPart.startIndex..<parenStart]).lowercased()
            variant = Int(wordPart[wordPart.index(after: parenStart)..<parenEnd])
        } else {
            word = wordPart.lowercased()
            variant = nil
        }

        let arpabet = pronPart.split(separator: " ").map(String.init)
        guard !arpabet.isEmpty else { return nil }

        return CMUDictRawEntry(word: word, arpabet: arpabet, variant: variant)
    }

    public static func parse(text: String, primaryOnly: Bool = false) -> [CMUDictRawEntry] {
        text.split(separator: "\n", omittingEmptySubsequences: false)
            .compactMap { parseLine(String($0)) }
            .filter { !primaryOnly || $0.variant == nil }
    }
}
