import Foundation
import Hub
import Tokenizers

func loadSTTTokenizer(from directory: URL, tokenizerClass: String? = nil) async throws -> Tokenizer
{
    let configuration = LanguageModelConfigurationFromHub(modelFolder: directory)
    guard let originalConfig = try await configuration.tokenizerConfig else {
        throw TokenizerError.missingConfig
    }
    var config = originalConfig
    if let tokenizerClass, var fields = config.dictionary() {
        fields["tokenizer_class"] = Config(tokenizerClass)
        config = Config(fields)
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

func transcriptDelta(from previous: String, to current: String, complete: Bool = false) -> String? {
    let sent = previous.unicodeScalars, text = current.unicodeScalars
    guard !text.elementsEqual(sent) else { return nil }
    guard complete || (text.last != "\u{FFFD}" && !sent.starts(with: text)) else { return nil }
    guard text.starts(with: sent) else { return current }
    return String(String.UnicodeScalarView(text.dropFirst(sent.count)))
}
