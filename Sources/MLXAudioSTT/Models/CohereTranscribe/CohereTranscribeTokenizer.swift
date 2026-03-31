import Foundation
import MLXAudioCore
import MLXLMCommon

public final class CohereTranscribeTokenizer {
    private let tokenizer: UnigramTokenizer
    private let specialTokenToID: [String: Int]
    private let specialIDs: Set<Int>

    public init(modelDir: URL, config _: CohereTranscribeConfig) throws {
        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.model")
        let tokenizerConfigURL = modelDir.appendingPathComponent("tokenizer_config.json")

        self.tokenizer = try UnigramTokenizer.from(sentencePieceModelURL: tokenizerURL)

        let configData = try Data(contentsOf: tokenizerConfigURL)
        let parsed = try JSONDecoder().decode(CohereTokenizerConfig.self, from: configData)
        let mapping = parsed.addedTokensDecoder.reduce(into: [String: Int]()) { partial, entry in
            partial[entry.value.content] = Int(entry.key) ?? -1
        }
        self.specialTokenToID = mapping
        self.specialIDs = Set(mapping.values)
    }

    public func encode(text: String) -> [Int] {
        if let id = specialTokenToID[text] {
            return [id]
        }
        return tokenizer.encodeWithByteFallback(text)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens.filter { !specialIDs.contains($0) })
    }

    public func buildPromptTokens(
        language: String,
        usePunctuation: Bool = true,
        useTimestamps: Bool = false
    ) -> [Int] {
        let langCode = mapLanguageCode(language)

        let promptTokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            langCode,
            langCode,
            usePunctuation ? "<|pnc|>" : "<|nopnc|>",
            "<|noitn|>",
            useTimestamps ? "<|timestamp|>" : "<|notimestamp|>",
            "<|nodiarize|>",
        ]

        return promptTokens.compactMap { specialTokenToID[$0] }
    }

    private func mapLanguageCode(_ language: String) -> String {
        let lang = language.lowercased()
        let languageMap: [String: String] = [
            "english": "<|en|>", "en": "<|en|>",
            "french": "<|fr|>", "fr": "<|fr|>",
            "german": "<|de|>", "de": "<|de|>",
            "spanish": "<|es|>", "es": "<|es|>",
            "italian": "<|it|>", "it": "<|it|>",
            "portuguese": "<|pt|>", "pt": "<|pt|>",
            "dutch": "<|nl|>", "nl": "<|nl|>",
            "polish": "<|pl|>", "pl": "<|pl|>",
            "greek": "<|el|>", "el": "<|el|>",
            "arabic": "<|ar|>", "ar": "<|ar|>",
            "japanese": "<|ja|>", "ja": "<|ja|>",
            "chinese": "<|zh|>", "zh": "<|zh|>",
            "vietnamese": "<|vi|>", "vi": "<|vi|>",
            "korean": "<|ko|>", "ko": "<|ko|>"
        ]
        return languageMap[lang] ?? "<|en|>"
    }
}

private struct CohereTokenizerConfig: Decodable {
    struct AddedToken: Decodable {
        let content: String
    }

    let addedTokensDecoder: [String: AddedToken]

    enum CodingKeys: String, CodingKey {
        case addedTokensDecoder = "added_tokens_decoder"
    }
}
