public protocol TokenAligning: Sendable {
    func align(tokens: [G2PToken], phonemes: PhonemeSequence) throws -> [TokenAlignment]
}
