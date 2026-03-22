public struct InMemoryLexicon: LexiconProviding, Sendable {
    private let entries: [String: LexiconEntry]

    public init(entries: [LexiconEntry]) {
        self.entries = Dictionary(
            uniqueKeysWithValues: entries.map { ($0.grapheme.lowercased(), $0) }
        )
    }

    public func lookup(_ grapheme: String) -> LexiconEntry? {
        entries[grapheme.lowercased()]
    }
}
