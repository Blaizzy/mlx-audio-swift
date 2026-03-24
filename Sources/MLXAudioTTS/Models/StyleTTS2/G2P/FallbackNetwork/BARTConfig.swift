import Foundation

// MARK: - Configuration
struct BARTConfig: Codable {
  let activationDropout: Double
  let activationFunction: String
  let architectures: [String]
  let attentionDropout: Double
  let bosTokenId: Int
  let classifierDropout: Double
  let dModel: Int
  let decoderAttentionHeads: Int
  let decoderFFNDim: Int
  let decoderLayerDrop: Double
  let decoderLayers: Int
  let decoderStartTokenId: Int
  let dropout: Double
  let encoderAttentionHeads: Int
  let encoderFFNDim: Int
  let encoderLayerDrop: Double
  let encoderLayers: Int
  let eosTokenId: Int
  let forcedEosTokenId: Int
  let graphemeChars: String
  let id2label: [String: String]
  let initStd: Double
  let isEncoderDecoder: Bool
  let label2id: [String: Int]
  let maxPositionEmbeddings: Int
  let modelType: String
  let numHiddenLayers: Int
  let padTokenId: Int
  let phonemeChars: String
  let scaleEmbedding: Bool
  let torchDType: String
  let transformersVersion: String
  let useCache: Bool
  let vocabSize: Int
    
  enum CodingKeys: String, CodingKey {
    case activationDropout = "activation_dropout"
    case activationFunction = "activation_function"
    case architectures = "architectures"
    case attentionDropout = "attention_dropout"
    case bosTokenId = "bos_token_id"
    case classifierDropout = "classifier_dropout"
    case dModel = "d_model"
    case decoderAttentionHeads = "decoder_attention_heads"
    case decoderFFNDim = "decoder_ffn_dim"
    case decoderLayerDrop = "decoder_layerdrop"
    case decoderLayers = "decoder_layers"
    case decoderStartTokenId = "decoder_start_token_id"
    case dropout = "dropout"
    case encoderAttentionHeads = "encoder_attention_heads"
    case encoderFFNDim = "encoder_ffn_dim"
    case encoderLayerDrop = "encoder_layerdrop"
    case encoderLayers = "encoder_layers"
    case eosTokenId = "eos_token_id"
    case forcedEosTokenId = "forced_eos_token_id"
    case graphemeChars = "grapheme_chars"
    case id2label = "id2label"
    case initStd = "init_std"
    case isEncoderDecoder = "is_encoder_decoder"
    case label2id = "label2id"
    case maxPositionEmbeddings = "max_position_embeddings"
    case modelType = "model_type"
    case numHiddenLayers = "num_hidden_layers"
    case padTokenId = "pad_token_id"
    case phonemeChars = "phoneme_chars"
    case scaleEmbedding = "scale_embedding"
    case torchDType = "torch_dtype"
    case transformersVersion = "transformers_version"
    case useCache = "use_cache"
    case vocabSize = "vocab_size"
  }
}
