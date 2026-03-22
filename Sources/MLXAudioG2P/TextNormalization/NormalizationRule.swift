import Foundation

public struct NormalizationRule: Sendable {
    public let name: String
    private let transform: @Sendable (String) -> String

    public init(name: String, transform: @escaping @Sendable (String) -> String) {
        self.name = name
        self.transform = transform
    }

    public func apply(to text: String) -> String {
        transform(text)
    }
}

extension NormalizationRule {
    static let collapseWhitespace = NormalizationRule(name: "collapseWhitespace") { text in
        text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    static let trimWhitespace = NormalizationRule(name: "trimWhitespace") { text in
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static let normalizeQuotes = NormalizationRule(name: "normalizeQuotes") { text in
        text.replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{2019}", with: "'")
    }

    static let normalizeDashes = NormalizationRule(name: "normalizeDashes") { text in
        text.replacingOccurrences(of: "\u{2014}", with: " - ")
            .replacingOccurrences(of: "\u{2013}", with: " - ")
    }
}
