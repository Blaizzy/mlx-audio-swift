import Foundation
import MLXLMCommon

// MARK: - Preprocessor Config

public struct PreprocessorConfig: Codable, Sendable {
    public var sampleRate: Int
    public var normalize: String
    public var windowSize: Float
    public var windowStride: Float
    public var window: String
    public var features: Int
    public var nFft: Int
    public var log: Bool
    public var frameSplicing: Int
    public var dither: Float
    public var padTo: Int
    public var padValue: Float
    public var preemph: Float

    public var hopLength: Int { Int(Float(sampleRate) * windowStride) }
    public var winLength: Int { Int(Float(sampleRate) * windowSize) }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case features
        case nFft = "n_fft"
        case log
        case frameSplicing = "frame_splicing"
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        normalize = try c.decodeIfPresent(String.self, forKey: .normalize) ?? "per_feature"
        windowSize = try c.decodeIfPresent(Float.self, forKey: .windowSize) ?? 0.025
        windowStride = try c.decodeIfPresent(Float.self, forKey: .windowStride) ?? 0.01
        window = try c.decodeIfPresent(String.self, forKey: .window) ?? "hann"
        features = try c.decodeIfPresent(Int.self, forKey: .features) ?? 128
        nFft = try c.decodeIfPresent(Int.self, forKey: .nFft) ?? 512
        log = try c.decodeIfPresent(Bool.self, forKey: .log) ?? true
        frameSplicing = try c.decodeIfPresent(Int.self, forKey: .frameSplicing) ?? 1
        dither = try c.decodeIfPresent(Float.self, forKey: .dither) ?? 1e-05
        padTo = try c.decodeIfPresent(Int.self, forKey: .padTo) ?? 0
        padValue = try c.decodeIfPresent(Float.self, forKey: .padValue) ?? 0.0
        preemph = try c.decodeIfPresent(Float.self, forKey: .preemph) ?? 0.97
    }

    public init() {
        sampleRate = 16000; normalize = "per_feature"; windowSize = 0.025
        windowStride = 0.01; window = "hann"; features = 128; nFft = 512
        log = true; frameSplicing = 1; dither = 1e-05; padTo = 0
        padValue = 0.0; preemph = 0.97
    }
}

// MARK: - Conformer Encoder Config

public struct ConformerEncoderConfig: Codable, Sendable {
    public var featIn: Int
    public var featOut: Int
    public var nLayers: Int
    public var dModel: Int
    public var subsampling: String
    public var subsamplingFactor: Int
    public var subsamplingConvChannels: Int
    public var causalDownsampling: Bool
    public var ffExpansionFactor: Int
    public var selfAttentionModel: String
    public var nHeads: Int
    public var attContextSize: [Int]
    public var xscaling: Bool
    public var untieBiases: Bool
    public var posEmbMaxLen: Int
    public var convKernelSize: Int
    public var convNormType: String
    public var dropout: Float
    public var dropoutPreEncoder: Float
    public var dropoutEmb: Float
    public var dropoutAtt: Float

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case featOut = "feat_out"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case subsampling
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case causalDownsampling = "causal_downsampling"
        case ffExpansionFactor = "ff_expansion_factor"
        case selfAttentionModel = "self_attention_model"
        case nHeads = "n_heads"
        case attContextSize = "att_context_size"
        case xscaling
        case untieBiases = "untie_biases"
        case posEmbMaxLen = "pos_emb_max_len"
        case convKernelSize = "conv_kernel_size"
        case convNormType = "conv_norm_type"
        case dropout
        case dropoutPreEncoder = "dropout_pre_encoder"
        case dropoutEmb = "dropout_emb"
        case dropoutAtt = "dropout_att"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try c.decodeIfPresent(Int.self, forKey: .featIn) ?? 128
        featOut = try c.decodeIfPresent(Int.self, forKey: .featOut) ?? -1
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 17
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 512
        subsampling = try c.decodeIfPresent(String.self, forKey: .subsampling) ?? "dw_striding"
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        subsamplingConvChannels = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        causalDownsampling = try c.decodeIfPresent(Bool.self, forKey: .causalDownsampling) ?? false
        ffExpansionFactor = try c.decodeIfPresent(Int.self, forKey: .ffExpansionFactor) ?? 4
        selfAttentionModel = try c.decodeIfPresent(String.self, forKey: .selfAttentionModel) ?? "rel_pos"
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 8
        attContextSize = try c.decodeIfPresent([Int].self, forKey: .attContextSize) ?? [-1, -1]
        xscaling = try c.decodeIfPresent(Bool.self, forKey: .xscaling) ?? false
        untieBiases = try c.decodeIfPresent(Bool.self, forKey: .untieBiases) ?? true
        posEmbMaxLen = try c.decodeIfPresent(Int.self, forKey: .posEmbMaxLen) ?? 5000
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
        convNormType = try c.decodeIfPresent(String.self, forKey: .convNormType) ?? "batch_norm"
        dropout = try c.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.1
        dropoutPreEncoder = try c.decodeIfPresent(Float.self, forKey: .dropoutPreEncoder) ?? 0.1
        dropoutEmb = try c.decodeIfPresent(Float.self, forKey: .dropoutEmb) ?? 0.0
        dropoutAtt = try c.decodeIfPresent(Float.self, forKey: .dropoutAtt) ?? 0.1
    }

    public init() {
        featIn = 128; featOut = -1; nLayers = 17; dModel = 512
        subsampling = "dw_striding"; subsamplingFactor = 8
        subsamplingConvChannels = 256; causalDownsampling = false
        ffExpansionFactor = 4; selfAttentionModel = "rel_pos"
        nHeads = 8; attContextSize = [-1, -1]; xscaling = false
        untieBiases = true; posEmbMaxLen = 5000; convKernelSize = 9
        convNormType = "batch_norm"; dropout = 0.1
        dropoutPreEncoder = 0.1; dropoutEmb = 0.0; dropoutAtt = 0.1
    }
}

// MARK: - Depthformer Config

public struct DepthformerConfig: Codable, Sendable {
    public var layers: Int
    public var dim: Int
    public var numHeads: Int
    public var numKvHeads: Int
    public var tie: Bool

    enum CodingKeys: String, CodingKey {
        case layers, dim
        case numHeads = "num_heads"
        case numKvHeads = "num_kv_heads"
        case tie
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 6
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 1024
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 32
        numKvHeads = try c.decodeIfPresent(Int.self, forKey: .numKvHeads) ?? 8
        tie = try c.decodeIfPresent(Bool.self, forKey: .tie) ?? true
    }

    public init() {
        layers = 6; dim = 1024; numHeads = 32; numKvHeads = 8; tie = true
    }
}

// MARK: - Detokenizer Config

public struct DetokenizerConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var layerTypes: [String]
    public var slidingWindow: Int
    public var intermediateSize: Int
    public var normEps: Float
    public var ropeTheta: Float
    public var outputSize: Int
    public var numCodebooks: Int
    public var vocabSize: Int
    public var nFft: Int
    public var hopLength: Int
    public var upsampleFactor: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case outputSize = "output_size"
        case numCodebooks = "num_codebooks"
        case vocabSize = "vocab_size"
        case nFft = "n_fft"
        case hopLength = "hop_length"
        case upsampleFactor = "upsample_factor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? [
            "conv", "conv", "sliding_attention", "conv",
            "sliding_attention", "conv", "sliding_attention", "conv",
        ]
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 30
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2304
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        outputSize = try c.decodeIfPresent(Int.self, forKey: .outputSize) ?? 1282
        numCodebooks = 8
        vocabSize = 2048
        nFft = try c.decodeIfPresent(Int.self, forKey: .nFft) ?? 1280
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 320
        upsampleFactor = try c.decodeIfPresent(Int.self, forKey: .upsampleFactor) ?? 6
    }

    public init() {
        hiddenSize = 512; numHiddenLayers = 8; numAttentionHeads = 16
        numKeyValueHeads = 8
        layerTypes = ["conv", "conv", "sliding_attention", "conv",
                      "sliding_attention", "conv", "sliding_attention", "conv"]
        slidingWindow = 30; intermediateSize = 2304; normEps = 1e-5
        ropeTheta = 1000000.0; outputSize = 1282; numCodebooks = 8
        vocabSize = 2048; nFft = 1280; hopLength = 320; upsampleFactor = 6
    }
}

// MARK: - LFM2 Backbone Config

public struct LFM2BackboneConfig: Codable, Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var maxPositionEmbeddings: Int?
    public var normEps: Float
    public var convBias: Bool
    public var convLCache: Int
    public var blockDim: Int?
    public var blockFFDim: Int?
    public var blockMultipleOf: Int
    public var blockFFNDimMultiplier: Float
    public var blockAutoAdjustFFDim: Bool
    public var fullAttnIdxs: [Int]?
    public var layerTypes: [String]?
    public var ropeTheta: Float

    public var effectiveBlockDim: Int { blockDim ?? hiddenSize }
    public var effectiveBlockFFDim: Int { blockFFDim ?? hiddenSize }

    public var resolvedFullAttnIdxs: [Int] {
        if let idxs = fullAttnIdxs { return idxs }
        if let types = layerTypes {
            return types.enumerated().compactMap { i, t in t == "full_attention" ? i : nil }
        }
        return Array(0..<numHiddenLayers)
    }

    public var headDimensions: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case blockDim = "block_dim"
        case blockFFDim = "block_ff_dim"
        case blockMultipleOf = "block_multiple_of"
        case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
        case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
        case fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 65536
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        blockDim = try c.decodeIfPresent(Int.self, forKey: .blockDim)
        blockFFDim = try c.decodeIfPresent(Int.self, forKey: .blockFFDim)
        blockMultipleOf = try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        blockFFNDimMultiplier = try c.decodeIfPresent(Float.self, forKey: .blockFFNDimMultiplier) ?? 1.0
        blockAutoAdjustFFDim = try c.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
        fullAttnIdxs = try c.decodeIfPresent([Int].self, forKey: .fullAttnIdxs)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
    }
}

// MARK: - Top-Level LFM2 Audio Config

public struct LFM2AudioConfig: Codable, Sendable {
    public var modelType: String
    public var sampleRate: Int
    public var codebooks: Int
    public var tieAudioEmbeddings: Bool
    public var semanticCodebookFactor: Int
    public var codebookWeight: String
    public var audioVocabSize: Int
    public var interleavedNText: Int
    public var interleavedNAudio: Int
    public var adapterHiddenDims: [Int]
    public var adapterDropout: Float
    public var adapterUseLayerNorm: Bool
    public var preprocessor: PreprocessorConfig
    public var encoder: ConformerEncoderConfig
    public var lfm: LFM2BackboneConfig
    public var depthformer: DepthformerConfig

    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case codebooks
        case tieAudioEmbeddings = "tie_audio_embeddings"
        case semanticCodebookFactor = "semantic_codebook_factor"
        case codebookWeight = "codebook_weight"
        case audioVocabSize = "audio_vocab_size"
        case interleavedNText = "interleaved_n_text"
        case interleavedNAudio = "interleaved_n_audio"
        case adapterHiddenDims = "adapter_hidden_dims"
        case adapterDropout = "adapter_dropout"
        case adapterUseLayerNorm = "adapter_use_layer_norm"
        case preprocessor, encoder, lfm, depthformer
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm_audio"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        codebooks = try c.decodeIfPresent(Int.self, forKey: .codebooks) ?? 8
        tieAudioEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieAudioEmbeddings) ?? false
        semanticCodebookFactor = try c.decodeIfPresent(Int.self, forKey: .semanticCodebookFactor) ?? 100
        codebookWeight = try c.decodeIfPresent(String.self, forKey: .codebookWeight) ?? "log"
        audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize) ?? 2049
        interleavedNText = try c.decodeIfPresent(Int.self, forKey: .interleavedNText) ?? 6
        interleavedNAudio = try c.decodeIfPresent(Int.self, forKey: .interleavedNAudio) ?? 12
        adapterHiddenDims = try c.decodeIfPresent([Int].self, forKey: .adapterHiddenDims) ?? [2048]
        adapterDropout = try c.decodeIfPresent(Float.self, forKey: .adapterDropout) ?? 0.0
        adapterUseLayerNorm = try c.decodeIfPresent(Bool.self, forKey: .adapterUseLayerNorm) ?? true
        preprocessor = try c.decodeIfPresent(PreprocessorConfig.self, forKey: .preprocessor) ?? PreprocessorConfig()
        encoder = try c.decodeIfPresent(ConformerEncoderConfig.self, forKey: .encoder) ?? ConformerEncoderConfig()
        lfm = try c.decode(LFM2BackboneConfig.self, forKey: .lfm)
        depthformer = try c.decodeIfPresent(DepthformerConfig.self, forKey: .depthformer) ?? DepthformerConfig()

        let baseConfig = try? BaseConfiguration(from: decoder)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public func encode(to coder: Swift.Encoder) throws {
        var container = coder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(codebooks, forKey: .codebooks)
        try container.encode(tieAudioEmbeddings, forKey: .tieAudioEmbeddings)
        try container.encode(semanticCodebookFactor, forKey: .semanticCodebookFactor)
        try container.encode(codebookWeight, forKey: .codebookWeight)
        try container.encode(audioVocabSize, forKey: .audioVocabSize)
        try container.encode(interleavedNText, forKey: .interleavedNText)
        try container.encode(interleavedNAudio, forKey: .interleavedNAudio)
        try container.encode(adapterHiddenDims, forKey: .adapterHiddenDims)
        try container.encode(adapterDropout, forKey: .adapterDropout)
        try container.encode(adapterUseLayerNorm, forKey: .adapterUseLayerNorm)
        try container.encode(preprocessor, forKey: .preprocessor)
        try container.encode(encoder, forKey: .encoder)
        try container.encode(lfm, forKey: .lfm)
        try container.encode(depthformer, forKey: .depthformer)
    }
}
