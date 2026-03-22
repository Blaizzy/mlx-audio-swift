public struct G2PToken: Sendable {
    public let surface: String
    public let kind: Kind
    public let rangeInNormalizedText: Range<String.Index>

    public enum Kind: String, Sendable, Hashable, CaseIterable {
        case word
        case punctuation
        case whitespace
    }

    public init(surface: String, kind: Kind, rangeInNormalizedText: Range<String.Index>) {
        self.surface = surface
        self.kind = kind
        self.rangeInNormalizedText = rangeInNormalizedText
    }
}
