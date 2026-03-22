import Foundation

public struct G2PPipeline: Sendable {
    private let normalizer: TextNormalizer
    private let tokenizer: TextTokenizer
    private let lexicon: (any LexiconProviding)?
    private let fallback: any Phonemizing
    private let aligner: (any TokenAligning)?

    public init(
        normalizer: TextNormalizer,
        tokenizer: TextTokenizer,
        lexicon: (any LexiconProviding)? = nil,
        fallback: any Phonemizing,
        aligner: (any TokenAligning)? = nil
    ) {
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.fallback = fallback
        self.aligner = aligner
    }

    public func convert(_ text: String) throws -> G2POutput {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw G2PError.emptyInput
        }

        let normalized = normalizer.normalize(trimmed)
        let tokens = tokenizer.tokenize(normalized)

        var allPhonemes: [PhonemeUnit] = []
        for token in tokens where token.kind == .word {
            if let entry = lexicon?.lookup(token.surface) {
                allPhonemes.append(contentsOf: entry.phonemes.map { PhonemeUnit(symbol: $0) })
            } else {
                let phonemes = try fallback.phonemize(token.surface)
                allPhonemes.append(contentsOf: phonemes)
            }
        }

        let sequence = PhonemeSequence(units: allPhonemes)

        let alignment: [TokenAlignment]?
        if let aligner {
            alignment = try aligner.align(tokens: tokens, phonemes: sequence)
        } else {
            alignment = nil
        }

        return G2POutput(
            normalizedText: normalized,
            tokens: tokens,
            phonemes: sequence,
            alignment: alignment
        )
    }

    public static func preview() -> G2PPipeline {
        let pack = EnglishLanguagePack.default
        return G2PPipeline(
            normalizer: pack.normalizer,
            tokenizer: pack.tokenizer,
            lexicon: pack.lexicon,
            fallback: pack.fallback
        )
    }

    public static func englishWithAlignment() -> G2PPipeline {
        let pack = EnglishLanguagePack.default
        return G2PPipeline(
            normalizer: pack.normalizer,
            tokenizer: pack.tokenizer,
            lexicon: pack.lexicon,
            fallback: pack.fallback,
            aligner: HeuristicTokenAligner()
        )
    }

    /// Full English pipeline with CMUdict (~135K words).
    /// - Parameter cmuDictDirectory: Directory containing `cmudict.dict`.
    public static func english(cmuDictDirectory: URL) throws -> G2PPipeline {
        let pack = try EnglishLanguagePack.withCMUDict(directory: cmuDictDirectory)
        return G2PPipeline(
            normalizer: pack.normalizer,
            tokenizer: pack.tokenizer,
            lexicon: pack.lexicon,
            fallback: pack.fallback
        )
    }

    /// Full English pipeline with CMUdict + alignment.
    /// - Parameter cmuDictDirectory: Directory containing `cmudict.dict`.
    public static func englishFull(cmuDictDirectory: URL) throws -> G2PPipeline {
        let pack = try EnglishLanguagePack.withCMUDict(directory: cmuDictDirectory)
        return G2PPipeline(
            normalizer: pack.normalizer,
            tokenizer: pack.tokenizer,
            lexicon: pack.lexicon,
            fallback: pack.fallback,
            aligner: HeuristicTokenAligner()
        )
    }
}
