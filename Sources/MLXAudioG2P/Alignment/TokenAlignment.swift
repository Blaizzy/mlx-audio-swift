public struct TokenAlignment: Sendable {
    public let tokenIndex: Int
    public let phonemeRange: Range<Int>

    public init(tokenIndex: Int, phonemeRange: Range<Int>) {
        self.tokenIndex = tokenIndex
        self.phonemeRange = phonemeRange
    }
}
