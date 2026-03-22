public struct G2POutput: Sendable {
    public let normalizedText: String
    public let tokens: [G2PToken]
    public let phonemes: PhonemeSequence
    public let alignment: [TokenAlignment]?
}
