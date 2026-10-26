import Foundation
import MLXAudioCore

/// Architecture and audio vocabulary of rumik-oss 1.
public struct RumikConfiguration: Decodable, Sendable {
    public var hiddenSize = 2048
    public var headDim = 128
    public var hiddenLayers = 36
    public var intermediateSize = 11008
    public var attentionHeads = 16
    public var kvHeads = 4
    public var ropeTheta: Float = 50000
    public var vocabularySize = 277404
    public var layerNormEps: Float = 1e-5
    public var logitScale: Float = 1
    public var attentionBias = false
    public var layerNormBias = false
    public var slidingWindow = 4096
    public var slidingWindowPattern = 4
    public var numQuantizers = 8
    public var codebookSize = 2048
    public var firstUnitID = 261008
    public var lastUnitID = 277391
    public var audioEndTokenID = 277394
    public var frameRate: Double = 12.5
    public var sampleRate = 24000
    public var speakers = ["Ira", "Aisha", "Siya", "Zoya"]
    public var mimiRepo = "kyutai/moshiko-pytorch-bf16"
    public var quantization: BaseConfiguration.Quantization?

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", headDim = "head_dim", hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size", attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads", ropeTheta = "rope_theta", vocabularySize = "vocab_size"
        case layerNormEps = "layer_norm_eps", logitScale = "logit_scale", attentionBias = "attention_bias"
        case layerNormBias = "layer_norm_bias", slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern", numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size", firstUnitID = "first_unit_id", lastUnitID = "last_unit_id"
        case audioEndTokenID = "audio_end_token_id", frameRate = "frame_rate_hz"
        case sampleRate = "sample_rate", speakers, mimiRepo = "mimi_repo", quantization
    }

    public init(from decoder: Swift.Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? headDim
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? kvHeads
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
        logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale) ?? logitScale
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
        layerNormBias = try c.decodeIfPresent(Bool.self, forKey: .layerNormBias) ?? layerNormBias
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? slidingWindow
        slidingWindowPattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? slidingWindowPattern
        numQuantizers = try c.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? numQuantizers
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? codebookSize
        firstUnitID = try c.decodeIfPresent(Int.self, forKey: .firstUnitID) ?? firstUnitID
        audioEndTokenID = try c.decodeIfPresent(Int.self, forKey: .audioEndTokenID) ?? audioEndTokenID
        frameRate = try c.decodeIfPresent(Double.self, forKey: .frameRate) ?? frameRate
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        speakers = try c.decodeIfPresent([String].self, forKey: .speakers) ?? speakers
        mimiRepo = try c.decodeIfPresent(String.self, forKey: .mimiRepo) ?? mimiRepo
        lastUnitID = try c.decodeIfPresent(Int.self, forKey: .lastUnitID)
            ?? (firstUnitID + codebookSize * numQuantizers - 1)
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
    }

    func validate() throws {
        guard hiddenSize > 0, headDim > 0, headDim % 2 == 0,
              attentionHeads * headDim == hiddenSize, kvHeads > 0,
              attentionHeads % kvHeads == 0, hiddenLayers > 0, intermediateSize > 0,
              slidingWindow > 0, slidingWindowPattern > 0,
              numQuantizers == 8, codebookSize == 2048,
              firstUnitID >= 0, lastUnitID == firstUnitID + numQuantizers * codebookSize - 1,
              lastUnitID < vocabularySize, audioEndTokenID >= 0, audioEndTokenID < vocabularySize,
              !(firstUnitID...lastUnitID).contains(audioEndTokenID), !speakers.isEmpty,
              sampleRate == 24000, frameRate == 12.5 else {
            throw AudioGenerationError.invalidInput("Unsupported Rumik architecture or Mimi vocabulary")
        }
    }
}

/// Delivery controls; text may also include <laugh>, <chuckle>, and <sigh> tags.
public struct RumikGenerationOptions: Sendable {
    public var description: String?
    public var instruct: String?
    public var minTokens: Int
    public init(description: String? = nil, instruct: String? = nil, minTokens: Int = 8) {
        self.description = description
        self.instruct = instruct
        self.minTokens = minTokens
    }
}
