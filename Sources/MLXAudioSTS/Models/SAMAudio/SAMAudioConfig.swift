import Foundation
import MLXAudioCodecs

/// Configuration for the T5 text encoder used by SAMAudio.
public struct T5EncoderConfig: Codable {
    public var name: String
    public var maxLength: Int?
    public var padMode: String
    public var dim: Int

    enum CodingKeys: String, CodingKey {
        case name
        case maxLength = "max_length"
        case padMode = "pad_mode"
        case dim
    }

    public init(
        name: String = "t5-base",
        maxLength: Int? = 512,
        padMode: String = "longest",
        dim: Int = 768
    ) {
        self.name = name
        self.maxLength = maxLength
        self.padMode = padMode
        self.dim = dim
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "t5-base"
        maxLength = try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? 512
        padMode = try c.decodeIfPresent(String.self, forKey: .padMode) ?? "longest"
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 768
    }
}

/// Diffusion transformer configuration for SAMAudio.
public struct TransformerConfig: Codable {
    public var dim: Int
    public var nHeads: Int
    public var nLayers: Int
    public var dropout: Float
    public var normEps: Float
    public var qkNorm: Bool
    public var fcBias: Bool
    public var ffnExp: Int
    public var ffnDimMultiplier: Int
    public var multipleOf: Int
    public var nonLinearity: String
    public var useRope: Bool
    public var maxPositions: Int
    public var frequencyEmbeddingDim: Int
    public var timestepNonLinearity: String
    public var tBlockNonLinearity: String
    public var tBlockBias: Bool
    public var contextDim: Int
    public var contextNonLinearity: String
    public var contextEmbedderDropout: Float
    public var contextNorm: Bool
    public var outChannels: Int
    public var inChannels: Int?

    enum CodingKeys: String, CodingKey {
        case dim
        case nHeads = "n_heads"
        case nLayers = "n_layers"
        case dropout
        case normEps = "norm_eps"
        case qkNorm = "qk_norm"
        case fcBias = "fc_bias"
        case ffnExp = "ffn_exp"
        case ffnDimMultiplier = "ffn_dim_multiplier"
        case multipleOf = "multiple_of"
        case nonLinearity = "non_linearity"
        case useRope = "use_rope"
        case maxPositions = "max_positions"
        case frequencyEmbeddingDim = "frequency_embedding_dim"
        case timestepNonLinearity = "timestep_non_linearity"
        case tBlockNonLinearity = "t_block_non_linearity"
        case tBlockBias = "t_block_bias"
        case contextDim = "context_dim"
        case contextNonLinearity = "context_non_linearity"
        case contextEmbedderDropout = "context_embedder_dropout"
        case contextNorm = "context_norm"
        case outChannels = "out_channels"
        case inChannels = "in_channels"
    }

    public init(
        dim: Int = 2816,
        nHeads: Int = 22,
        nLayers: Int = 22,
        dropout: Float = 0.1,
        normEps: Float = 1.0e-5,
        qkNorm: Bool = true,
        fcBias: Bool = false,
        ffnExp: Int = 4,
        ffnDimMultiplier: Int = 1,
        multipleOf: Int = 64,
        nonLinearity: String = "swiglu",
        useRope: Bool = true,
        maxPositions: Int = 10000,
        frequencyEmbeddingDim: Int = 256,
        timestepNonLinearity: String = "swiglu",
        tBlockNonLinearity: String = "silu",
        tBlockBias: Bool = true,
        contextDim: Int = 2816,
        contextNonLinearity: String = "swiglu",
        contextEmbedderDropout: Float = 0,
        contextNorm: Bool = false,
        outChannels: Int = 256,
        inChannels: Int? = nil
    ) {
        self.dim = dim
        self.nHeads = nHeads
        self.nLayers = nLayers
        self.dropout = dropout
        self.normEps = normEps
        self.qkNorm = qkNorm
        self.fcBias = fcBias
        self.ffnExp = ffnExp
        self.ffnDimMultiplier = ffnDimMultiplier
        self.multipleOf = multipleOf
        self.nonLinearity = nonLinearity
        self.useRope = useRope
        self.maxPositions = maxPositions
        self.frequencyEmbeddingDim = frequencyEmbeddingDim
        self.timestepNonLinearity = timestepNonLinearity
        self.tBlockNonLinearity = tBlockNonLinearity
        self.tBlockBias = tBlockBias
        self.contextDim = contextDim
        self.contextNonLinearity = contextNonLinearity
        self.contextEmbedderDropout = contextEmbedderDropout
        self.contextNorm = contextNorm
        self.outChannels = outChannels
        self.inChannels = inChannels
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 2816
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 22
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 22
        dropout = try c.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.1
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1.0e-5
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? true
        fcBias = try c.decodeIfPresent(Bool.self, forKey: .fcBias) ?? false
        ffnExp = try c.decodeIfPresent(Int.self, forKey: .ffnExp) ?? 4
        ffnDimMultiplier = try c.decodeIfPresent(Int.self, forKey: .ffnDimMultiplier) ?? 1
        multipleOf = try c.decodeIfPresent(Int.self, forKey: .multipleOf) ?? 64
        nonLinearity = try c.decodeIfPresent(String.self, forKey: .nonLinearity) ?? "swiglu"
        useRope = try c.decodeIfPresent(Bool.self, forKey: .useRope) ?? true
        maxPositions = try c.decodeIfPresent(Int.self, forKey: .maxPositions) ?? 10000
        frequencyEmbeddingDim = try c.decodeIfPresent(Int.self, forKey: .frequencyEmbeddingDim) ?? 256
        timestepNonLinearity = try c.decodeIfPresent(String.self, forKey: .timestepNonLinearity) ?? "swiglu"
        tBlockNonLinearity = try c.decodeIfPresent(String.self, forKey: .tBlockNonLinearity) ?? "silu"
        tBlockBias = try c.decodeIfPresent(Bool.self, forKey: .tBlockBias) ?? true
        contextDim = try c.decodeIfPresent(Int.self, forKey: .contextDim) ?? dim
        contextNonLinearity = try c.decodeIfPresent(String.self, forKey: .contextNonLinearity) ?? "swiglu"
        contextEmbedderDropout = try c.decodeIfPresent(Float.self, forKey: .contextEmbedderDropout) ?? 0
        contextNorm = try c.decodeIfPresent(Bool.self, forKey: .contextNorm) ?? false
        outChannels = try c.decodeIfPresent(Int.self, forKey: .outChannels) ?? 256
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels)
    }
}

/// Main SAMAudio model configuration.
public struct SAMAudioConfig: Codable {
    public var inChannels: Int
    public var audioCodec: DACVAEConfig
    public var textEncoder: T5EncoderConfig
    public var transformer: TransformerConfig
    public var numAnchors: Int
    public var anchorEmbeddingDim: Int

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case audioCodec = "audio_codec"
        case textEncoder = "text_encoder"
        case transformer
        case numAnchors = "num_anchors"
        case anchorEmbeddingDim = "anchor_embedding_dim"
    }

    public init(
        inChannels: Int = 768,
        audioCodec: DACVAEConfig = DACVAEConfig(),
        textEncoder: T5EncoderConfig = T5EncoderConfig(),
        transformer: TransformerConfig = TransformerConfig(),
        numAnchors: Int = 3,
        anchorEmbeddingDim: Int = 128
    ) {
        self.inChannels = inChannels
        self.audioCodec = audioCodec
        self.textEncoder = textEncoder
        self.transformer = transformer
        self.numAnchors = numAnchors
        self.anchorEmbeddingDim = anchorEmbeddingDim
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        audioCodec = try c.decodeIfPresent(DACVAEConfig.self, forKey: .audioCodec) ?? DACVAEConfig()
        textEncoder = try c.decodeIfPresent(T5EncoderConfig.self, forKey: .textEncoder) ?? T5EncoderConfig()
        transformer = try c.decodeIfPresent(TransformerConfig.self, forKey: .transformer) ?? TransformerConfig()
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? (6 * audioCodec.codebookDim)
        numAnchors = try c.decodeIfPresent(Int.self, forKey: .numAnchors) ?? 3
        anchorEmbeddingDim = try c.decodeIfPresent(Int.self, forKey: .anchorEmbeddingDim) ?? 128
    }
}

public extension SAMAudioConfig {
    static var small: SAMAudioConfig {
        SAMAudioConfig(
            inChannels: 768,
            audioCodec: DACVAEConfig(),
            textEncoder: T5EncoderConfig(name: "t5-base", maxLength: 512, padMode: "longest", dim: 768),
            transformer: TransformerConfig(dim: 1024, nHeads: 8, nLayers: 12, contextDim: 1024, outChannels: 256),
            numAnchors: 3,
            anchorEmbeddingDim: 128
        )
    }

    static var base: SAMAudioConfig {
        SAMAudioConfig(
            inChannels: 768,
            audioCodec: DACVAEConfig(),
            textEncoder: T5EncoderConfig(name: "t5-base", maxLength: 512, padMode: "longest", dim: 768),
            transformer: TransformerConfig(dim: 1536, nHeads: 12, nLayers: 16, contextDim: 1536, outChannels: 256),
            numAnchors: 3,
            anchorEmbeddingDim: 128
        )
    }

    static var large: SAMAudioConfig {
        SAMAudioConfig(
            inChannels: 768,
            audioCodec: DACVAEConfig(),
            textEncoder: T5EncoderConfig(name: "t5-base", maxLength: 512, padMode: "longest", dim: 768),
            transformer: TransformerConfig(dim: 2816, nHeads: 22, nLayers: 22, contextDim: 2816, outChannels: 256),
            numAnchors: 3,
            anchorEmbeddingDim: 128
        )
    }
}
