import Foundation
import Hub
import Tokenizers

func loadQwen2AudioTokenizer(from directory: URL) async throws -> Tokenizer {
    let configuration = LanguageModelConfigurationFromHub(modelFolder: directory)
    guard let config = try await configuration.tokenizerConfig else {
        throw TokenizerError.missingConfig
    }
    let data = try await configuration.tokenizerData
    guard var dictionary = data.dictionary() else {
        return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
    }
    // Some published checkpoints keep audio and timestamp tokens only in the
    // Transformers configuration. Add them in memory without changing the cache.
    var tokens: [Int: Config] = [:]
    for token in data["added_tokens"].array(or: []) {
        if let id = token["id"].integer() { tokens[id] = token }
    }
    for (key, token) in config["added_tokens_decoder"].dictionary() ?? [:] {
        guard let id = Int(key.string), tokens[id] == nil, var fields = token.dictionary() else {
            continue
        }
        fields["id"] = Config(id)
        tokens[id] = Config(fields)
    }
    dictionary["added_tokens"] = Config(tokens.keys.sorted().compactMap { tokens[$0] })
    return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: Config(dictionary))
}
