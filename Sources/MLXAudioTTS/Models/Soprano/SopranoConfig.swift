//
//  SopranoConfig.swift
//  MLXAudio
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLXLMCommon

// MARK: - Decoder Configuration

public struct SopranoDecoderConfig: Codable, Sendable {
    public var decoderNumLayers: Int
    public var decoderDim: Int
    public var decoderIntermediateDim: Int
    public var hopLength: Int
    public var nFft: Int
    public var upscale: Int
    public var dwKernel: Int
    public var tokenSize: Int
    public var receptiveField: Int

    enum CodingKeys: String, CodingKey {
        case decoderNumLayers = "decoder_num_layers"
        case decoderDim = "decoder_dim"
        case decoderIntermediateDim = "decoder_intermediate_dim"
        case hopLength = "hop_length"
        case nFft = "n_fft"
        case upscale
        case dwKernel = "dw_kernel"
        case tokenSize = "token_size"
        case receptiveField = "receptive_field"
    }

    public init(
        decoderNumLayers: Int = 8,
        decoderDim: Int = 512,
        decoderIntermediateDim: Int = 1536,
        hopLength: Int = 512,
        nFft: Int = 2048,
        upscale: Int = 4,
        dwKernel: Int = 3,
        tokenSize: Int = 2048,
        receptiveField: Int = 4
    ) {
        self.decoderNumLayers = decoderNumLayers
        self.decoderDim = decoderDim
        self.decoderIntermediateDim = decoderIntermediateDim
        self.hopLength = hopLength
        self.nFft = nFft
        self.upscale = upscale
        self.dwKernel = dwKernel
        self.tokenSize = tokenSize
        self.receptiveField = receptiveField
    }
}

// MARK: - Main Configuration

public struct SopranoConfiguration: Codable, Sendable {
    // Transformer config (Qwen3-based)
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var vocabularySize: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool

    // Token IDs
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var padTokenId: Int

    // Audio config
    public var sampleRate: Int

    // Decoder config (embedded)
    public var decoderNumLayers: Int
    public var decoderDim: Int
    public var decoderIntermediateDim: Int
    public var hopLength: Int
    public var nFft: Int
    public var upscale: Int
    public var dwKernel: Int
    public var tokenSize: Int
    public var receptiveField: Int

    // Quantization
    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case sampleRate = "sample_rate"
        case decoderNumLayers = "decoder_num_layers"
        case decoderDim = "decoder_dim"
        case decoderIntermediateDim = "decoder_intermediate_dim"
        case hopLength = "hop_length"
        case nFft = "n_fft"
        case upscale
        case dwKernel = "dw_kernel"
        case tokenSize = "token_size"
        case receptiveField = "receptive_field"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Transformer config
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 512
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // Token IDs
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0

        // Audio config
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 32000

        // Decoder config
        self.decoderNumLayers = try container.decodeIfPresent(Int.self, forKey: .decoderNumLayers) ?? 8
        self.decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 512
        self.decoderIntermediateDim = try container.decodeIfPresent(Int.self, forKey: .decoderIntermediateDim) ?? 1536
        self.hopLength = try container.decodeIfPresent(Int.self, forKey: .hopLength) ?? 512
        self.nFft = try container.decodeIfPresent(Int.self, forKey: .nFft) ?? 2048
        self.upscale = try container.decodeIfPresent(Int.self, forKey: .upscale) ?? 4
        self.dwKernel = try container.decodeIfPresent(Int.self, forKey: .dwKernel) ?? 3
        self.tokenSize = try container.decodeIfPresent(Int.self, forKey: .tokenSize) ?? 2048
        self.receptiveField = try container.decodeIfPresent(Int.self, forKey: .receptiveField) ?? 4

        // Quantization
        let baseConfig = try? BaseConfiguration(from: decoder)
        self.perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(bosTokenId, forKey: .bosTokenId)
        try container.encode(eosTokenId, forKey: .eosTokenId)
        try container.encode(padTokenId, forKey: .padTokenId)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(decoderNumLayers, forKey: .decoderNumLayers)
        try container.encode(decoderDim, forKey: .decoderDim)
        try container.encode(decoderIntermediateDim, forKey: .decoderIntermediateDim)
        try container.encode(hopLength, forKey: .hopLength)
        try container.encode(nFft, forKey: .nFft)
        try container.encode(upscale, forKey: .upscale)
        try container.encode(dwKernel, forKey: .dwKernel)
        try container.encode(tokenSize, forKey: .tokenSize)
        try container.encode(receptiveField, forKey: .receptiveField)
    }

    /// Get decoder configuration as a separate struct
    public var decoderConfig: SopranoDecoderConfig {
        SopranoDecoderConfig(
            decoderNumLayers: decoderNumLayers,
            decoderDim: decoderDim,
            decoderIntermediateDim: decoderIntermediateDim,
            hopLength: hopLength,
            nFft: nFft,
            upscale: upscale,
            dwKernel: dwKernel,
            tokenSize: tokenSize,
            receptiveField: receptiveField
        )
    }
}
