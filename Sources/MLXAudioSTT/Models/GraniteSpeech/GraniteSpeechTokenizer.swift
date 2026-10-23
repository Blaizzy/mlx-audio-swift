import Foundation
import Hub
import Tokenizers

func loadGraniteSpeechTokenizer(from directory: URL) async throws -> Tokenizer {
    let configuration = LanguageModelConfigurationFromHub(modelFolder: directory)
    guard let config = try await configuration.tokenizerConfig else {
        throw TokenizerError.missingConfig
    }
    let data = try await configuration.tokenizerData
    // Python's GPT2Tokenizer selects the GPT-2 byte-level pre-tokenizer even when
    // tokenizer.json contains a newer Split expression. Preserve that behavior.
    if config["tokenizer_class"].string() == "GPT2Tokenizer", var dictionary = data.dictionary() {
        dictionary["pre_tokenizer"] = [
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
        ]
        return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: Config(dictionary))
    }
    return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
}
