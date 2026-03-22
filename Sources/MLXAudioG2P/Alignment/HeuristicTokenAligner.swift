public struct HeuristicTokenAligner: TokenAligning, Sendable {
    public init() {}

    public func align(
        tokens: [G2PToken],
        phonemes: PhonemeSequence
    ) throws -> [TokenAlignment] {
        let wordTokens = tokens.enumerated().filter { $0.element.kind == .word }

        guard !wordTokens.isEmpty else { return [] }

        guard !phonemes.isEmpty else {
            throw G2PError.alignmentFailed(
                reason: "Empty phoneme sequence for non-empty word tokens"
            )
        }

        let perToken = phonemes.units.count / wordTokens.count
        let remainder = phonemes.units.count % wordTokens.count
        var offset = 0
        var result: [TokenAlignment] = []

        for (i, (tokenIndex, _)) in wordTokens.enumerated() {
            let count = perToken + (i < remainder ? 1 : 0)
            result.append(TokenAlignment(tokenIndex: tokenIndex, phonemeRange: offset..<(offset + count)))
            offset += count
        }

        return result
    }
}
