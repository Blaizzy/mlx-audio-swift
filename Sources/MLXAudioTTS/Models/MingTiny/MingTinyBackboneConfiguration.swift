// Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
// Adapted to Swift from inclusionAI's Ming V3 ASR and mlx-audio.

import Foundation
import MLXAudioCore

public struct MingTinyTextConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1536
    public var intermediateSize: Int = 4608
    public var numHiddenLayers: Int = 24
    public var vocabSize: Int = 157184
    public var numAttentionHeads: Int = 16
    public var headDim: Int = 128
    public var layerGroupSize: Int = 4
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 10000
    public var ropeInterleave: Bool = true
    public var qLoraRank: Int? = 256
    public var kvLoraRank: Int = 512
    public var qkNopeHeadDim: Int = 128
    public var qkRopeHeadDim: Int = 64
    public var vHeadDim: Int = 128
    public var shortConvKernelSize: Int = 4
    public var noKdaLora: Bool = true
    public var kdaSafeGate: Bool = true
    public var kdaLowerBound: Float = -5
    public var numExperts: Int = 128
    public var numSharedExperts: Int = 1
    public var numExpertsPerToken: Int = 8
    public var moeIntermediateSize: Int = 512
    public var moeSharedExpertIntermediateSize: Int = 512
    public var firstKDenseReplace: Int = 1
    public var nGroup: Int = 8
    public var topKGroup: Int = 4
    public var routedScalingFactor: Float = 2.5
    public var attentionGate: String? = "head_wise"
    public var expertSwiGLULimits: [Float]? = nil
    public var sharedExpertSwiGLULimits: [Float]? = nil
    public var maxPositionEmbeddings: Int = 8192
    public var tieWordEmbeddings: Bool = false
    public var eosTokenIds: [Int] = [156892]
    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case layerGroupSize = "layer_group_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeInterleave = "rope_interleave"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
        case noKdaLora = "no_kda_lora"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
        case numExperts = "num_experts"
        case numSharedExperts = "num_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case nGroup = "n_group"
        case topKGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case attentionGate = "gated_attention_proj_granularity_type"
        case expertSwiGLULimits = "expert_swiglu_limit_list"
        case sharedExpertSwiGLULimits = "share_expert_swiglu_limit_list"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case eosTokenIds = "eos_token_id"
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4608
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 157184
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        layerGroupSize = try c.decodeIfPresent(Int.self, forKey: .layerGroupSize) ?? 4
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? true
        if c.contains(.qLoraRank) {
            qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        }
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        shortConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .shortConvKernelSize) ?? 4
        noKdaLora = try c.decodeIfPresent(Bool.self, forKey: .noKdaLora) ?? true
        kdaSafeGate = try c.decodeIfPresent(Bool.self, forKey: .kdaSafeGate) ?? true
        kdaLowerBound = try c.decodeIfPresent(Float.self, forKey: .kdaLowerBound) ?? -5
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        moeSharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize) ?? 512
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 8
        topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? 4
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        if c.contains(.attentionGate) {
            attentionGate = try c.decodeIfPresent(String.self, forKey: .attentionGate)
        }
        if c.contains(.expertSwiGLULimits) {
            expertSwiGLULimits = try c.decodeIfPresent([Float].self, forKey: .expertSwiGLULimits)
        }
        if c.contains(.sharedExpertSwiGLULimits) {
            sharedExpertSwiGLULimits = try c.decodeIfPresent(
                [Float].self, forKey: .sharedExpertSwiGLULimits)
        }
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        eosTokenIds =
            try (try? c.decode([Int].self, forKey: .eosTokenIds)) ?? [
                try c.decodeIfPresent(Int.self, forKey: .eosTokenIds) ?? 156892
            ]
    }
    func usesMLA(_ index: Int) -> Bool {
        (index + 1) % layerGroupSize == 0
            || index >= numHiddenLayers / layerGroupSize * layerGroupSize
    }
}
