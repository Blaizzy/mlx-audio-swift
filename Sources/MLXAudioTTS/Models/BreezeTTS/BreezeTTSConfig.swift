import Foundation
import MLXLMCommon

public struct BreezeBackboneConfig: Codable, Sendable {
    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_936
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6_144
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 40_960
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

public struct BreezeDepthDecoderConfig: Codable, Sendable {
    var vocabSize: Int
    var numCodebooks: Int
    var audioEmbedSize: Int
    var backboneHiddenSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case numCodebooks = "num_codebooks"
        case audioEmbedSize = "audio_embed_size"
        case backboneHiddenSize = "backbone_hidden_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 2_051
        numCodebooks = try c.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? 16
        audioEmbedSize = try c.decodeIfPresent(Int.self, forKey: .audioEmbedSize) ?? 2_048
        backboneHiddenSize = try c.decodeIfPresent(Int.self, forKey: .backboneHiddenSize) ?? 2_048
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1_024
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8_192
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 33
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

public struct BreezeTextEncoderConfig: Decodable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var queryPreAttentionScalar: Float
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var slidingWindowPattern: Int
    var eoiTokenIndex: Int
    var layerTypes: [String]

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case queryPreAttentionScalar = "query_pre_attn_scalar"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case privateSlidingWindowPattern = "_sliding_window_pattern"
        case eoiTokenIndex = "eoi_token_index"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262_208
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_304
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9_216
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 26
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        queryPreAttentionScalar = try c.decodeIfPresent(Float.self, forKey: .queryPreAttentionScalar) ?? 256
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4_096
        slidingWindowPattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern)
            ?? c.decodeIfPresent(Int.self, forKey: .privateSlidingWindowPattern)
            ?? 6
        eoiTokenIndex = try c.decodeIfPresent(Int.self, forKey: .eoiTokenIndex) ?? 256_000
        if let decodedLayerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = decodedLayerTypes
        } else {
            let layerCount = numHiddenLayers
            let pattern = slidingWindowPattern
            layerTypes = (0..<layerCount).map {
                ($0 + 1).isMultiple(of: pattern) ? "full_attention" : "sliding_attention"
            }
        }
    }
}

public struct BreezeCodecConfig: Decodable, Sendable {
    var samplingRate: Int
    var codebookSize: Int

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case outputSamplingRate = "output_sample_rate"
        case codebookSize = "codebook_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate)
            ?? c.decodeIfPresent(Int.self, forKey: .outputSamplingRate)
            ?? 24_000
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2_048
    }
}

public struct BreezeTTSConfig: Decodable, Sendable {
    var modelType: String
    var audioNumCodebooks: Int
    var audioVocabSize: Int
    var audioEmbedSize: Int
    var textVocabSize: Int
    var audioTokenID: Int
    var audioEOSTokenID: Int
    var codebookPadTokenID: Int
    var codebookEOSTokenID: Int
    var tieCodebooksEmbeddings: Bool
    var backboneConfig: BreezeBackboneConfig
    var codecConfig: BreezeCodecConfig
    var depthDecoderConfig: BreezeDepthDecoderConfig
    var textEncoderConfig: BreezeTextEncoderConfig
    var quantization: BaseConfiguration.Quantization?
    var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    var numCodebooks: Int { audioNumCodebooks }
    var codecVocabSize: Int { codecConfig.codebookSize }
    var sampleRate: Int { codecConfig.samplingRate }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioNumCodebooks = "audio_num_codebooks"
        case numCodebooks = "num_codebooks"
        case audioVocabSize = "audio_vocab_size"
        case vocabSize = "vocab_size"
        case audioEmbedSize = "audio_embed_size"
        case textVocabSize = "text_vocab_size"
        case audioTokenID = "audio_token_id"
        case audioEOSTokenID = "audio_eos_token_id"
        case codebookPadTokenID = "codebook_pad_token_id"
        case codebookEOSTokenID = "codebook_eos_token_id"
        case tieCodebooksEmbeddings = "tie_codebooks_embeddings"
        case backboneConfig = "backbone_config"
        case codecConfig = "codec_config"
        case depthDecoderConfig = "depth_decoder_config"
        case textEncoderConfig = "text_encoder_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "breeze"
        audioNumCodebooks = try c.decodeIfPresent(Int.self, forKey: .audioNumCodebooks)
            ?? c.decodeIfPresent(Int.self, forKey: .numCodebooks)
            ?? 16
        audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize)
            ?? c.decodeIfPresent(Int.self, forKey: .vocabSize)
            ?? 2_051
        backboneConfig = try c.decode(BreezeBackboneConfig.self, forKey: .backboneConfig)
        codecConfig = try c.decodeIfPresent(BreezeCodecConfig.self, forKey: .codecConfig)
            ?? JSONDecoder().decode(BreezeCodecConfig.self, from: Data("{}".utf8))
        depthDecoderConfig = try c.decodeIfPresent(BreezeDepthDecoderConfig.self, forKey: .depthDecoderConfig)
            ?? JSONDecoder().decode(BreezeDepthDecoderConfig.self, from: Data("{}".utf8))
        textEncoderConfig = try c.decodeIfPresent(BreezeTextEncoderConfig.self, forKey: .textEncoderConfig)
            ?? JSONDecoder().decode(BreezeTextEncoderConfig.self, from: Data("{}".utf8))
        audioEmbedSize = try c.decodeIfPresent(Int.self, forKey: .audioEmbedSize) ?? backboneConfig.hiddenSize
        textVocabSize = try c.decodeIfPresent(Int.self, forKey: .textVocabSize) ?? 262_158
        audioTokenID = try c.decodeIfPresent(Int.self, forKey: .audioTokenID) ?? 262_144
        audioEOSTokenID = try c.decodeIfPresent(Int.self, forKey: .audioEOSTokenID) ?? 262_145
        codebookPadTokenID = try c.decodeIfPresent(Int.self, forKey: .codebookPadTokenID) ?? 2_050
        codebookEOSTokenID = try c.decodeIfPresent(Int.self, forKey: .codebookEOSTokenID) ?? 0
        tieCodebooksEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieCodebooksEmbeddings) ?? true

        let base = try? BaseConfiguration(from: decoder)
        quantization = base?.quantization
        perLayerQuantization = base?.perLayerQuantization
    }
}
