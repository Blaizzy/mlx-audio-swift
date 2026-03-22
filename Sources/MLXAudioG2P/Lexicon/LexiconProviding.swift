public protocol LexiconProviding: Sendable {
    func lookup(_ grapheme: String) -> LexiconEntry?
}
