import Foundation

public struct MiMoMLXManifest: Decodable, Sendable {
    public let format: String?
    public let model: String?
    public let precision: String?
    public let weightFile: String?
    public let indexFile: String?
    public let configFile: String?
    public let audioTokenizerDir: String?
    public let audioTokenizerRepo: String?
    public let bf16FallbackDir: String?
    public let pythonLoader: String?

    enum CodingKeys: String, CodingKey {
        case format
        case model
        case precision
        case weightFile = "weight_file"
        case indexFile = "index_file"
        case configFile = "config_file"
        case audioTokenizerDir = "audio_tokenizer_dir"
        case audioTokenizerRepo = "audio_tokenizer_repo"
        case bf16FallbackDir = "bf16_fallback_dir"
        case pythonLoader = "python_loader"
    }
}
