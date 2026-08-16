import Foundation

public struct Wav2Vec2STTConfig: Codable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let hiddenAct: String
    public let hiddenDropout: Float
    public let activationDropout: Float
    public let attentionDropout: Float
    public let featProjDropout: Float
    public let finalDropout: Float
    public let layerNormEps: Float
    public let featExtractNorm: String
    public let featExtractActivation: String
    public let convDim: [Int]
    public let convStride: [Int]
    public let convKernel: [Int]
    public let convBias: Bool
    public let numConvPosEmbeddings: Int
    public let numConvPosEmbeddingGroups: Int
    public let numFeatExtractLayers: Int
    public let doStableLayerNorm: Bool
    public let padTokenId: Int
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let adapterAttnDim: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case hiddenDropout = "hidden_dropout"
        case activationDropout = "activation_dropout"
        case attentionDropout = "attention_dropout"
        case featProjDropout = "feat_proj_dropout"
        case finalDropout = "final_dropout"
        case layerNormEps = "layer_norm_eps"
        case featExtractNorm = "feat_extract_norm"
        case featExtractActivation = "feat_extract_activation"
        case convDim = "conv_dim"
        case convStride = "conv_stride"
        case convKernel = "conv_kernel"
        case convBias = "conv_bias"
        case numConvPosEmbeddings = "num_conv_pos_embeddings"
        case numConvPosEmbeddingGroups = "num_conv_pos_embedding_groups"
        case numFeatExtractLayers = "num_feat_extract_layers"
        case doStableLayerNorm = "do_stable_layer_norm"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case adapterAttnDim = "adapter_attn_dim"
    }

    public init(
        modelType: String = "wav2vec2",
        vocabSize: Int = 32,
        hiddenSize: Int = 768,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        intermediateSize: Int = 3_072,
        hiddenAct: String = "gelu",
        hiddenDropout: Float = 0.1,
        activationDropout: Float = 0.1,
        attentionDropout: Float = 0.1,
        featProjDropout: Float = 0.0,
        finalDropout: Float = 0.1,
        layerNormEps: Float = 1e-5,
        featExtractNorm: String = "group",
        featExtractActivation: String = "gelu",
        convDim: [Int] = [512, 512, 512, 512, 512, 512, 512],
        convStride: [Int] = [5, 2, 2, 2, 2, 2, 2],
        convKernel: [Int] = [10, 3, 3, 3, 3, 2, 2],
        convBias: Bool = false,
        numConvPosEmbeddings: Int = 128,
        numConvPosEmbeddingGroups: Int = 16,
        numFeatExtractLayers: Int? = nil,
        doStableLayerNorm: Bool = false,
        padTokenId: Int = 0,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        adapterAttnDim: Int? = nil
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.hiddenAct = hiddenAct
        self.hiddenDropout = hiddenDropout
        self.activationDropout = activationDropout
        self.attentionDropout = attentionDropout
        self.featProjDropout = featProjDropout
        self.finalDropout = finalDropout
        self.layerNormEps = layerNormEps
        self.featExtractNorm = featExtractNorm
        self.featExtractActivation = featExtractActivation
        self.convDim = convDim
        self.convStride = convStride
        self.convKernel = convKernel
        self.convBias = convBias
        self.numConvPosEmbeddings = numConvPosEmbeddings
        self.numConvPosEmbeddingGroups = numConvPosEmbeddingGroups
        self.numFeatExtractLayers = numFeatExtractLayers ?? convDim.count
        self.doStableLayerNorm = doStableLayerNorm
        self.padTokenId = padTokenId
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.adapterAttnDim = adapterAttnDim
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let convDim = try c.decodeIfPresent([Int].self, forKey: .convDim) ?? [512, 512, 512, 512, 512, 512, 512]
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "wav2vec2"
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3_072
        self.hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        self.hiddenDropout = try c.decodeIfPresent(Float.self, forKey: .hiddenDropout) ?? 0.1
        self.activationDropout = try c.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0.1
        self.attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.1
        self.featProjDropout = try c.decodeIfPresent(Float.self, forKey: .featProjDropout) ?? 0.0
        self.finalDropout = try c.decodeIfPresent(Float.self, forKey: .finalDropout) ?? 0.1
        self.layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        self.featExtractNorm = try c.decodeIfPresent(String.self, forKey: .featExtractNorm) ?? "group"
        self.featExtractActivation = try c.decodeIfPresent(String.self, forKey: .featExtractActivation) ?? "gelu"
        self.convDim = convDim
        self.convStride = try c.decodeIfPresent([Int].self, forKey: .convStride) ?? [5, 2, 2, 2, 2, 2, 2]
        self.convKernel = try c.decodeIfPresent([Int].self, forKey: .convKernel) ?? [10, 3, 3, 3, 3, 2, 2]
        self.convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.numConvPosEmbeddings = try c.decodeIfPresent(Int.self, forKey: .numConvPosEmbeddings) ?? 128
        self.numConvPosEmbeddingGroups = try c.decodeIfPresent(Int.self, forKey: .numConvPosEmbeddingGroups) ?? 16
        self.numFeatExtractLayers = try c.decodeIfPresent(Int.self, forKey: .numFeatExtractLayers) ?? convDim.count
        self.doStableLayerNorm = try c.decodeIfPresent(Bool.self, forKey: .doStableLayerNorm) ?? false
        self.padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        self.bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.adapterAttnDim = try c.decodeIfPresent(Int.self, forKey: .adapterAttnDim)
    }
}
