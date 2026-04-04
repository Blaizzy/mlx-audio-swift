import Tokenizers

// Both MLXLMCommon and swift-transformers expose a `Tokenizer` protocol.
// TTS model loading uses the swift-transformers tokenizer API explicitly.
public typealias TTSModelTokenizer = any Tokenizers.Tokenizer
