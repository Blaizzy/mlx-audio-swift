import Foundation
import Tokenizers

public final class WhisperTokenizer {
    let inner: any Tokenizer
    let isMultilingual: Bool

    let startOfTranscriptId: Int
    let endOfTextId: Int
    let noTimestampsId: Int
    let prevSotId: Int?
    let transcribeId: Int?
    let translateId: Int?
    let timestampBeginId: Int

    let languageToId: [String: Int]
    let specialTokenIds: Set<Int>

    public init(
        modelDirectory: URL,
        baseConfig: WhisperConfig,
        generationConfig: WhisperGenerationConfig?,
        tokenizerDirectory: URL? = nil
    ) async throws {
        let tokenizerDir = tokenizerDirectory ?? modelDirectory
        self.inner = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        let addedTokens = Self.loadAddedTokensDecoder(modelDirectory: tokenizerDir)

        if let id = generationConfig?.decoderStartTokenId {
            self.startOfTranscriptId = id
        } else if let id = Self.tokenId(in: addedTokens, name: "<|startoftranscript|>") {
            self.startOfTranscriptId = id
        } else {
            self.startOfTranscriptId = baseConfig.decoderStartTokenId
        }

        if let id = generationConfig?.eosTokenId {
            self.endOfTextId = id
        } else if let id = Self.tokenId(in: addedTokens, name: "<|endoftext|>") {
            self.endOfTextId = id
        } else {
            self.endOfTextId = baseConfig.eosTokenId
        }

        if let id = generationConfig?.noTimestampsTokenId {
            self.noTimestampsId = id
        } else if let id = Self.tokenId(in: addedTokens, name: "<|notimestamps|>") {
            self.noTimestampsId = id
        } else {
            self.noTimestampsId = baseConfig.decoderStartTokenId + (baseConfig.vocabSize > 51864 ? 105 : 104)
        }

        self.prevSotId = generationConfig?.prevSotTokenId
            ?? Self.tokenId(in: addedTokens, name: "<|startofprev|>")

        self.transcribeId = generationConfig?.taskToId?["transcribe"]
            ?? Self.tokenId(in: addedTokens, name: "<|transcribe|>")
        self.translateId = generationConfig?.taskToId?["translate"]
            ?? Self.tokenId(in: addedTokens, name: "<|translate|>")

        // Timestamp tokens span `<|0.00|>` … `<|30.00|>`, immediately following
        // `<|notimestamps|>`.
        self.timestampBeginId = noTimestampsId + 1

        var langMap: [String: Int] = [:]
        if let supplied = generationConfig?.langToId {
            for (key, id) in supplied {
                langMap[Self.normalizeLanguageKey(key)] = id
            }
        }
        for (id, name) in addedTokens {
            guard let code = Self.extractLanguageCode(name) else { continue }
            langMap[code] = id
        }
        self.languageToId = langMap

        if let flag = generationConfig?.isMultilingual {
            self.isMultilingual = flag
        } else {
            self.isMultilingual = !langMap.isEmpty
        }

        var allSpecials = Set<Int>(addedTokens.keys)
        allSpecials.insert(startOfTranscriptId)
        allSpecials.insert(endOfTextId)
        allSpecials.insert(noTimestampsId)
        if let id = prevSotId { allSpecials.insert(id) }
        if let id = transcribeId { allSpecials.insert(id) }
        if let id = translateId { allSpecials.insert(id) }
        for id in langMap.values { allSpecials.insert(id) }
        self.specialTokenIds = allSpecials
    }

    /// Build the decoder prefix for a transcription request. `language` may be
    /// an ISO code ("en") or a full name ("English"); pass `nil` to let
    /// multilingual variants auto-detect.
    public func buildPromptTokens(language: String?, task: String = "transcribe") -> [Int] {
        var tokens: [Int] = [startOfTranscriptId]
        if isMultilingual {
            if let resolved = resolveLanguage(language) {
                tokens.append(resolved)
            }
            switch task.lowercased() {
            case "translate":
                if let id = translateId { tokens.append(id) }
            default:
                if let id = transcribeId { tokens.append(id) }
            }
        }
        tokens.append(noTimestampsId)
        return tokens
    }

    public func resolveLanguage(_ language: String?) -> Int? {
        guard let raw = language?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else { return nil }
        let normalized = Self.normalizeLanguageKey(raw)
        if let direct = languageToId[normalized] {
            return direct
        }
        if let mapped = Self.fullNameToCode[normalized],
           let id = languageToId[mapped] {
            return id
        }
        return nil
    }

    /// Decode token IDs to text, stripping every Whisper special / timestamp token.
    public func decode(tokens: [Int]) -> String {
        let filtered = tokens.filter { id in
            id >= 0 && id < timestampBeginId && !specialTokenIds.contains(id)
        }
        return inner.decode(tokens: filtered, skipSpecialTokens: true)
    }

    private static func loadAddedTokensDecoder(modelDirectory: URL) -> [Int: String] {
        let configURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let added = json["added_tokens_decoder"] as? [String: Any]
        else {
            return [:]
        }
        var result: [Int: String] = [:]
        for (idString, entry) in added {
            guard let id = Int(idString) else { continue }
            if let dict = entry as? [String: Any], let content = dict["content"] as? String {
                result[id] = content
            } else if let str = entry as? String {
                result[id] = str
            }
        }
        return result
    }

    private static func tokenId(in added: [Int: String], name: String) -> Int? {
        for (id, content) in added where content == name { return id }
        return nil
    }

    private static func extractLanguageCode(_ token: String) -> String? {
        guard token.hasPrefix("<|"), token.hasSuffix("|>") else { return nil }
        let body = token.dropFirst(2).dropLast(2)
        guard !body.isEmpty, body.count <= 3 else { return nil }
        for scalar in body.unicodeScalars where !CharacterSet.lowercaseLetters.contains(scalar) {
            return nil
        }
        return String(body)
    }

    private static func normalizeLanguageKey(_ key: String) -> String {
        var k = key.lowercased()
        if k.hasPrefix("<|"), k.hasSuffix("|>") {
            k = String(k.dropFirst(2).dropLast(2))
        }
        return k
    }

    private static let fullNameToCode: [String: String] = [
        "english": "en", "chinese": "zh", "mandarin": "zh", "cantonese": "yue",
        "japanese": "ja", "korean": "ko", "french": "fr", "german": "de",
        "spanish": "es", "italian": "it", "portuguese": "pt", "russian": "ru",
        "polish": "pl", "turkish": "tr", "dutch": "nl", "arabic": "ar",
        "hindi": "hi", "indonesian": "id", "vietnamese": "vi", "thai": "th",
        "ukrainian": "uk", "swedish": "sv", "czech": "cs", "romanian": "ro",
        "hungarian": "hu", "danish": "da", "finnish": "fi", "norwegian": "no",
        "hebrew": "he", "greek": "el", "tagalog": "tl", "filipino": "tl",
        "malay": "ms", "tamil": "ta", "telugu": "te", "bengali": "bn",
        "urdu": "ur",
    ]
}
