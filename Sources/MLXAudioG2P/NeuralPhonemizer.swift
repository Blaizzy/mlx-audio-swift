import Foundation

/// Neural G2P fallback using ByT5 model.
///
/// Usage:
/// ```swift
/// let neural = try NeuralPhonemizer(modelDirectory: modelURL, language: "eng-us")
/// let pack = EnglishLanguagePack(
///     normalizer: .englishDefault,
///     tokenizer: .englishDefault,
///     lexicon: try CMUDictLoader.loadFromBundle(),
///     fallback: neural
/// )
/// ```
public final class NeuralPhonemizer: Phonemizing, @unchecked Sendable {
    private let g2p: G2P
    private let language: String

    public init(modelDirectory: URL, language: String = "eng-us", maxLength: Int = 50) throws {
        self.g2p = try G2P(modelDirectory: modelDirectory, maxLength: maxLength)
        self.language = language
    }

    public func phonemize(_ grapheme: String) throws -> [PhonemeUnit] {
        let ipa = g2p.convert(grapheme, language: language)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !ipa.isEmpty else {
            throw G2PError.phonemizationFailed(
                token: grapheme,
                reason: "Neural model returned empty output"
            )
        }

        return ipa.filter { !$0.isWhitespace }
            .map { PhonemeUnit(symbol: String($0)) }
    }
}
