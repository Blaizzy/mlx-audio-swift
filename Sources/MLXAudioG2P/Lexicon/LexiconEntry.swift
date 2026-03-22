public struct LexiconEntry: Sendable, Hashable {
    public let grapheme: String
    public let phonemes: [String]

    public init(grapheme: String, phonemes: [String]) {
        self.grapheme = grapheme
        self.phonemes = phonemes
    }
}
