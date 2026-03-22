public struct EnglishLanguagePack: Sendable {
    public let normalizer: TextNormalizer
    public let tokenizer: TextTokenizer
    public let lexicon: InMemoryLexicon
    public let fallback: FallbackPhonemizer

    public init(
        normalizer: TextNormalizer,
        tokenizer: TextTokenizer,
        lexicon: InMemoryLexicon,
        fallback: FallbackPhonemizer
    ) {
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.fallback = fallback
    }

    public static let `default` = EnglishLanguagePack(
        normalizer: .englishDefault,
        tokenizer: .englishDefault,
        lexicon: InMemoryLexicon(entries: starterLexicon),
        fallback: FallbackPhonemizer()
    )

    /// Full CMUdict lexicon (~135K words). Requires bundle resources.
    public static func withCMUDict() throws -> EnglishLanguagePack {
        let lexicon = try CMUDictLoader.loadFromBundle()
        return EnglishLanguagePack(
            normalizer: .englishDefault,
            tokenizer: .englishDefault,
            lexicon: lexicon,
            fallback: FallbackPhonemizer()
        )
    }

    private static let starterLexicon: [LexiconEntry] = [
        LexiconEntry(grapheme: "hello", phonemes: ["h", "ɛ", "l", "oʊ"]),
        LexiconEntry(grapheme: "world", phonemes: ["w", "ɜː", "l", "d"]),
        LexiconEntry(grapheme: "the", phonemes: ["ð", "ə"]),
        LexiconEntry(grapheme: "a", phonemes: ["ə"]),
        LexiconEntry(grapheme: "is", phonemes: ["ɪ", "z"]),
    ]
}
