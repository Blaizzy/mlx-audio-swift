// Copyright © 2023-2024 Apple Inc.
// Bailing MoE architecture adapted from mlx-lm and mlx-audio.
import Foundation
import MLXAudioCore

public struct MingOmniTextConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 2048
    public var intermediateSize: Int = 5632
    public var numHiddenLayers: Int = 28
    public var vocabSize: Int = 126464
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 4
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 600000
    public var numExperts: Int = 64
    public var numExpertsPerToken: Int = 6
    public var numSharedExperts: Int = 2
    public var moeIntermediateSize: Int = 1408
    public var firstKDenseReplace: Int = 0
    public var normTopKProbability: Bool = true
    public var routedScalingFactor: Float = 1
    public var scoreFunction: String = "softmax"
    public var numGroups: Int = 1
    public var topKGroups: Int = 4
    public var useQKNormalization: Bool = false
    public var useBias: Bool = false
    public var useQKVBias: Bool = false
    public var tieWordEmbeddings: Bool = false
    public var expertBiasEnabled: Bool = false
    public var sharedExpertEnabled: Bool = true
    public var partialRotaryFactor: Float = 1
    public var rotaryDim: Int?
    public var sharedExpertIntermediateSize: Int?
    public var ropeScaling: RotaryConfiguration?
    public init() {}
    public struct RotaryConfiguration: Codable, Sendable {
        public var type: String
        public var sections: [Int]?
        enum CodingKeys: String, CodingKey { case type, sections = "mrope_section" }
    }
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case normTopKProbability = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case numGroups = "n_group"
        case topKGroups = "topk_group"
        case useQKNormalization = "use_qk_norm"
        case useBias = "use_bias"
        case useQKVBias = "use_qkv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case expertBiasEnabled = "moe_router_enable_expert_bias"
        case sharedExpertEnabled = "moe_router_enable_shared_expert"
        case partialRotaryFactor = "partial_rotary_factor"
        case rotaryDim = "rotary_dim", sharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case ropeScaling = "rope_scaling"
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5632
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 126464
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 4
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 600000
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 64
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 6
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 2
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1408
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        normTopKProbability = try c.decodeIfPresent(Bool.self, forKey: .normTopKProbability) ?? true
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        scoreFunction = try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "softmax"
        numGroups = try c.decodeIfPresent(Int.self, forKey: .numGroups) ?? 1
        topKGroups = try c.decodeIfPresent(Int.self, forKey: .topKGroups) ?? 4
        useQKNormalization = try c.decodeIfPresent(Bool.self, forKey: .useQKNormalization) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useQKVBias = try c.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        expertBiasEnabled = try c.decodeIfPresent(Bool.self, forKey: .expertBiasEnabled) ?? false
        sharedExpertEnabled = try c.decodeIfPresent(Bool.self, forKey: .sharedExpertEnabled) ?? true
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1
        rotaryDim = try c.decodeIfPresent(Int.self, forKey: .rotaryDim)
        sharedExpertIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize)
        ropeScaling = try c.decodeIfPresent(RotaryConfiguration.self, forKey: .ropeScaling)
    }
    var headDim: Int { hiddenSize / numAttentionHeads }
    var ropeDimensions: Int { rotaryDim ?? Int(Float(headDim) * partialRotaryFactor) }
    func validate() throws {
        guard hiddenSize > 0, numAttentionHeads > 0, numKeyValueHeads > 0,
              hiddenSize % numAttentionHeads == 0, numAttentionHeads % numKeyValueHeads == 0,
              numHiddenLayers > 0, vocabSize > 0, intermediateSize > 0, moeIntermediateSize > 0,
              numExperts > 0, (1...numExperts).contains(numExpertsPerToken), numSharedExperts >= 0,
              numGroups > 0, numExperts % numGroups == 0,
              numGroups == 1 || (1...numGroups).contains(topKGroups),
              (0...numHiddenLayers).contains(firstKDenseReplace),
              partialRotaryFactor.isFinite, partialRotaryFactor > 0, partialRotaryFactor <= 1,
              sharedExpertIntermediateSize.map({ $0 > 0 }) ?? true,
              ropeScaling?.sections?.allSatisfy({ $0 > 0 }) ?? true,
              (2...headDim).contains(ropeDimensions), ropeDimensions % 2 == 0,
              ropeTheta.isFinite, ropeTheta > 0, rmsNormEps.isFinite, rmsNormEps > 0,
              routedScalingFactor.isFinite, ["softmax", "sigmoid"].contains(scoreFunction), !useBias else {
            throw AudioGenerationError.invalidInput("Invalid or unsupported Ming Omni Bailing MoE configuration")
        }
    }
}

public struct MingOmniConfiguration: Decodable, Sendable {
    public var text: MingOmniTextConfiguration
    public var audio: MingAudioVAEConfiguration
    public var flow: MingTransformerConfiguration
    public var aggregator: MingTransformerConfiguration
    public var quantization: BaseConfiguration.Quantization?
    enum CodingKeys: String, CodingKey {
        case text = "llm_config", audio = "audio_tokenizer_config", flow = "ditar_config"
        case aggregator = "aggregator_config", quantization
    }
    public init(text: MingOmniTextConfiguration, audio: MingAudioVAEConfiguration,
                flow: MingTransformerConfiguration, aggregator: MingTransformerConfiguration) {
        self.text = text; self.audio = audio; self.flow = flow; self.aggregator = aggregator
    }
}
