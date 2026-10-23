// Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted to Swift from inclusionAI's Ming V3 ASR and mlx-audio.

import Foundation
import MLXAudioCore

public struct MingTextConfig: Codable, Sendable {
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
    public init(from decoder: Decoder) throws {
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

public struct MingAudioConfig: Codable, Sendable {
    public var numMelBins: Int = 128
    public var maxSourcePositions: Int = 3000
    public var hiddenSize: Int = 1280
    public var numAttentionHeads: Int = 20
    public var numHiddenLayers: Int = 32
    public var downsampleKernels: [Int] = [3]
    public var downsampleStrides: [Int] = [2]
    public var downsamplePaddings: [Int]? = nil
    public var chunkFrames: Int = 3000
    public init() {}
    enum CodingKeys: String, CodingKey {
        case numMelBins = "n_mels"
        case maxSourcePositions = "n_audio_ctx"
        case hiddenSize = "n_audio_state"
        case numAttentionHeads = "n_audio_head"
        case numHiddenLayers = "n_audio_layer"
        case downsampleKernels = "downsampler_kernel_size_list"
        case downsampleStrides = "downsampler_stride_list"
        case downsamplePaddings = "downsampler_padding_list"
        case chunkFrames = "whisper_chunk_size"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 3000
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 20
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        downsampleKernels = try c.decodeIfPresent([Int].self, forKey: .downsampleKernels) ?? [3]
        downsampleStrides = try c.decodeIfPresent([Int].self, forKey: .downsampleStrides) ?? [2]
        if c.contains(.downsamplePaddings) {
            downsamplePaddings = try c.decodeIfPresent([Int].self, forKey: .downsamplePaddings)
        }
        chunkFrames = try c.decodeIfPresent(Int.self, forKey: .chunkFrames) ?? 3000
    }
    var paddings: [Int] { downsamplePaddings ?? downsampleKernels.map { $0 / 2 } }
    func embeddingLength(_ frames: Int) -> Int {
        var length = (frames + 1) / 2
        for i in downsampleKernels.indices {
            length = (length + 2 * paddings[i] - downsampleKernels[i]) / downsampleStrides[i] + 1
        }
        return length
    }
    func tokenCount(_ frames: Int, chunkFrames: Int? = nil) -> Int {
        let chunk = chunkFrames ?? self.chunkFrames
        return stride(from: 0, to: frames, by: chunk).reduce(0) {
            $0 + embeddingLength(min(chunk, frames - $1))
        }
    }
    var whisperConfig: WhisperConfig {
        var c = WhisperConfig()
        c.numMelBins = numMelBins
        c.dModel = hiddenSize
        c.encoderLayers = numHiddenLayers
        c.encoderAttentionHeads = numAttentionHeads
        c.encoderFfnDim = 4 * hiddenSize
        c.maxSourcePositions = maxSourcePositions
        return c
    }
}

public struct MingASRConfig: Codable {
    public var modelType = "ming_v3_asr"
    public var textConfig = MingTextConfig()
    public var audioConfig = MingAudioConfig()
    public var audioTokenId = 157168
    public var normalizeAudioEmbeddings = true
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    public init() {}
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case audioConfig = "audio_config"
        case audioTokenId = "audio_token_id"
        case normalizeAudioEmbeddings = "norm_query_embeds"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ming_v3_asr"
        textConfig = try c.decodeIfPresent(MingTextConfig.self, forKey: .textConfig) ?? .init()
        audioConfig = try c.decodeIfPresent(MingAudioConfig.self, forKey: .audioConfig) ?? .init()
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 157168
        normalizeAudioEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .normalizeAudioEmbeddings) ?? true
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }
    func validate() throws {
        let t = textConfig
        let a = audioConfig
        guard t.hiddenSize > 0, t.intermediateSize > 0, t.numHiddenLayers > 0,
            t.vocabSize > 0, t.numAttentionHeads > 0, t.headDim > 0, t.layerGroupSize > 0,
            t.kvLoraRank > 0, t.qLoraRank == nil || t.qLoraRank! > 0,
            t.qkNopeHeadDim > 0, t.qkRopeHeadDim > 0, t.qkRopeHeadDim % 2 == 0,
            t.vHeadDim > 0, t.shortConvKernelSize > 0, t.rmsNormEps > 0,
            t.numExperts > 0, t.nGroup > 0, t.numExperts % t.nGroup == 0,
            (1 ... t.nGroup).contains(t.topKGroup),
            (1 ... t.numExperts).contains(t.numExpertsPerToken),
            t.numExpertsPerToken <= t.topKGroup * (t.numExperts / t.nGroup),
            t.numSharedExperts >= 0, t.moeIntermediateSize > 0,
            t.moeSharedExpertIntermediateSize > 0,
            t.maxPositionEmbeddings > 0,
            [nil, "head_wise", "element_wise"].contains(t.attentionGate),
            (0 ..< t.vocabSize).contains(audioTokenId),
            a.hiddenSize > 0, a.numHiddenLayers > 0, a.numAttentionHeads > 0,
            a.hiddenSize % a.numAttentionHeads == 0, a.numMelBins > 0,
            a.maxSourcePositions > 0, a.chunkFrames > 0, a.chunkFrames <= 2 * a.maxSourcePositions,
            !a.downsampleKernels.isEmpty, a.downsampleKernels.count == a.downsampleStrides.count,
            a.paddings.count == a.downsampleKernels.count,
            a.downsampleKernels.allSatisfy({ $0 > 0 }), a.downsampleStrides.allSatisfy({ $0 > 0 }),
            a.paddings.allSatisfy({ $0 >= 0 })
        else { throw STTError.modelNotInitialized("Inconsistent Ming ASR configuration") }
    }
}
