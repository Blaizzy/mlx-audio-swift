public struct TextNormalizer: Sendable {
    public let rules: [NormalizationRule]

    public init(rules: [NormalizationRule]) {
        self.rules = rules
    }

    public func normalize(_ text: String) -> String {
        rules.reduce(text) { $1.apply(to: $0) }
    }

    public static let englishDefault = TextNormalizer(rules: [
        .collapseWhitespace,
        .normalizeQuotes,
        .normalizeDashes,
        .trimWhitespace,
    ])
}
