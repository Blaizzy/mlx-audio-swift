import Foundation
import MLXAudioCore
import Tokenizers

let fishSpeechEOSToken = "<|endoftext|>"
let fishSpeechPadToken = "<|pad|>"
let fishSpeechIMStartToken = "<|im_start|>"
let fishSpeechIMEndToken = "<|im_end|>"
let fishSpeechTextModalityToken = "<|text|>"
let fishSpeechVoiceModalityToken = "<|voice|>"
let fishSpeechInterleaveModalityToken = "<|interleave|>"

let fishSpeechModalityTokens: [FishSpeechModality: String] = [
    .text: fishSpeechTextModalityToken,
    .voice: fishSpeechVoiceModalityToken,
    .interleave: fishSpeechInterleaveModalityToken,
]

func fishSpeechSemanticToken(_ index: Int) -> String {
    "<|semantic:\(index)|>"
}

protocol FishSpeechTokenizing: Sendable {
    var vocabSize: Int { get }
    var eosTokenID: Int { get }
    var padTokenID: Int { get }
    var semanticBeginID: Int { get }
    var semanticEndID: Int { get }

    func encode(_ text: String, addSpecialTokens: Bool) -> [Int]
    func decode(_ tokens: [Int], skipSpecialTokens: Bool) -> String
    func tokenID(for token: String) -> Int?
}

public final class FishSpeechTokenizer: FishSpeechTokenizing {
    public let tokenizer: any Tokenizer
    public let vocabSize: Int
    public let semanticBeginID: Int
    public let semanticEndID: Int

    public var eosTokenID: Int {
        tokenizer.eosTokenId ?? tokenID(for: fishSpeechEOSToken) ?? 0
    }

    public var padTokenID: Int {
        tokenizer.convertTokenToId(fishSpeechPadToken)
            ?? tokenizer.unknownTokenId
            ?? 0
    }

    init(tokenizer: any Tokenizer, vocabSizeHint: Int? = nil) throws {
        self.tokenizer = tokenizer

        var semanticIDs: [Int] = []
        semanticIDs.reserveCapacity(4_096)

        for codeIndex in 0 ..< 4_096 {
            guard let tokenID = tokenizer.convertTokenToId(fishSpeechSemanticToken(codeIndex)) else {
                throw AudioGenerationError.invalidInput(
                    "Fish tokenizer is missing semantic tokens; expected 4096 semantic IDs."
                )
            }
            semanticIDs.append(tokenID)
        }

        self.semanticBeginID = semanticIDs.min() ?? 0
        self.semanticEndID = semanticIDs.max() ?? 0
        let tokenizerUpperBound = semanticIDs.max().map { $0 + 1 } ?? 0
        self.vocabSize = max(vocabSizeHint ?? 0, tokenizerUpperBound)
    }

    static func fromModelDirectory(
        _ modelFolder: URL,
        vocabSizeHint: Int? = nil
    ) async throws -> FishSpeechTokenizer {
        try prepareTokenizerFilesIfNeeded(in: modelFolder)
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        let discoveredSize = try resolveVocabSize(in: modelFolder)
        return try FishSpeechTokenizer(
            tokenizer: tokenizer,
            vocabSizeHint: max(vocabSizeHint ?? 0, discoveredSize ?? 0)
        )
    }

    func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(_ tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    func tokenID(for token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}

private extension FishSpeechTokenizer {
    static func prepareTokenizerFilesIfNeeded(in modelFolder: URL) throws {
        try generateTokenizerJSONIfMissing(in: modelFolder)

        let tokenizerConfigPath = modelFolder.appendingPathComponent("tokenizer_config.json")
        guard !FileManager.default.fileExists(atPath: tokenizerConfigPath.path) else { return }

        let minimalConfig: [String: Any] = [
            "tokenizer_class": "Qwen3Tokenizer",
            "eos_token": fishSpeechEOSToken,
            "pad_token": fishSpeechPadToken,
        ]
        let data = try JSONSerialization.data(withJSONObject: minimalConfig, options: [.sortedKeys])
        try data.write(to: tokenizerConfigPath)
    }

    static func resolveVocabSize(in modelFolder: URL) throws -> Int? {
        let tokenizerJSONPath = modelFolder.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerJSONPath.path) {
            let data = try Data(contentsOf: tokenizerJSONPath)
            if let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let model = object["model"] as? [String: Any],
               let vocab = model["vocab"] as? [String: Any]
            {
                return (vocab.values.compactMap { $0 as? Int }.max() ?? -1) + 1
            }
        }

        let vocabJSONPath = modelFolder.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabJSONPath.path) {
            let data = try Data(contentsOf: vocabJSONPath)
            if let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                return (vocab.values.compactMap { $0 as? Int }.max() ?? -1) + 1
            }
        }

        return nil
    }

    static func generateTokenizerJSONIfMissing(in modelFolder: URL) throws {
        let tokenizerJSONPath = modelFolder.appendingPathComponent("tokenizer.json")
        guard !FileManager.default.fileExists(atPath: tokenizerJSONPath.path) else { return }

        let vocabURL = modelFolder.appendingPathComponent("vocab.json")
        let mergesURL = modelFolder.appendingPathComponent("merges.txt")
        let tokenizerConfigURL = modelFolder.appendingPathComponent("tokenizer_config.json")

        guard FileManager.default.fileExists(atPath: vocabURL.path),
              FileManager.default.fileExists(atPath: mergesURL.path) else {
            return
        }

        let vocabData = try Data(contentsOf: vocabURL)
        let vocabString = String(data: vocabData, encoding: .utf8) ?? "{}"

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let mergesJSON = mergesText
            .components(separatedBy: "\n")
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .map {
                "\"\($0.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\""))\""
            }
            .joined(separator: ",")

        var addedTokensJSON = "[]"
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            let configData = try Data(contentsOf: tokenizerConfigURL)
            if let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: Any]
            {
                var tokens: [(Int, [String: Any])] = []
                for (idString, value) in addedTokensDecoder {
                    guard let id = Int(idString), let tokenDict = value as? [String: Any] else { continue }
                    tokens.append((id, [
                        "id": id,
                        "content": tokenDict["content"] ?? "",
                        "single_word": tokenDict["single_word"] ?? false,
                        "lstrip": tokenDict["lstrip"] ?? false,
                        "rstrip": tokenDict["rstrip"] ?? false,
                        "normalized": tokenDict["normalized"] ?? false,
                        "special": tokenDict["special"] ?? false,
                    ]))
                }
                tokens.sort { $0.0 < $1.0 }
                let tokenData = try JSONSerialization.data(withJSONObject: tokens.map(\.1))
                addedTokensJSON = String(data: tokenData, encoding: .utf8) ?? "[]"
            }
        }

        let preTokenizerPattern =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        let escapedPattern = preTokenizerPattern
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")

        let tokenizerJSON = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": \(addedTokensJSON),
          "normalizer": {"type": "NFC"},
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {
                "type": "Split",
                "pattern": {"Regex": "\(escapedPattern)"},
                "behavior": "Isolated",
                "invert": false
              },
              {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false
              }
            ]
          },
          "post_processor": null,
          "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
          },
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": \(vocabString),
            "merges": [\(mergesJSON)]
          }
        }
        """

        try tokenizerJSON.write(to: tokenizerJSONPath, atomically: true, encoding: .utf8)
    }
}
