//
//  Config.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/25.
//

import Foundation
import MLXLMCommon

// MARK: - Configuration

public struct Qwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var eosTokenId: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var quantization: BaseConfiguration.Quantization?
    var perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768
    var sampleRate: Int = 24_000
    var tokenizer_name: String? = nil

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case eosTokenId = "eos_token_id"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case sampleRate = "sample_rate"
        case tokenizer_name = "tokenizer_name"
        
    }

    public init(from decoder: Swift.Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
        self.eosTokenId = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.eosTokenId)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
        
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
        self.tokenizer_name = try container.decodeIfPresent(String.self, forKey: .tokenizer_name) ?? nil
        
        // MARK: - Decode Quantization
        let baseConfig = try? BaseConfiguration(from: decoder)
        self.perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        vocabularySize: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        sampleRate: Int = 24_000,
        eosTokenId: Int = 151645
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.vocabularySize = vocabularySize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.sampleRate = sampleRate
        self.eosTokenId = eosTokenId
        self.maxPositionEmbeddings = 40960
        self.quantization = nil
        self.perLayerQuantization = nil
        self.tokenizer_name = nil
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(eosTokenId, forKey: .eosTokenId)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(headDim, forKey: .headDim)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encodeIfPresent(tokenizer_name, forKey: .tokenizer_name)
        
    }
}
