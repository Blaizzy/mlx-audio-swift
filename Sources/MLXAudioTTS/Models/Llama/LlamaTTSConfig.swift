//
//  LlamaTTSConfig.swift
//  MLXAudio
//
//  Created by Prince Canuma on 02/01/2026.
//

import Foundation
import MLXAudioCore

// MARK: - Configuration

/// Configuration for Llama-based TTS models (e.g., Orpheus-TTS).
///
public struct LlamaTTSConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var headDimensions: Int?
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var kvHeads: Int
    public var maxPositionEmbeddings: Int?
    public var ropeTheta: Float = 10000
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]?
    public var tieWordEmbeddings: Bool = true
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false

    // TTS-specific
    public var sampleRate: Int = 24_000
    public var tokenizerName: String? = nil

    // Quantization
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case sampleRate = "sample_rate"
        case tokenizerName = "tokenizer_name"
    }

    public init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimensions: Int? = nil,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 10000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        sampleRate: Int = 24_000,
        tokenizerName: String? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.sampleRate = sampleRate
        self.tokenizerName = tokenizerName
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)

        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) {
            self.ropeTraditional = ropeTraditional
        }
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)

        if let tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }

        // TTS-specific
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
        tokenizerName = try container.decodeIfPresent(String.self, forKey: .tokenizerName)

        // Quantization
        let baseConfig = try? BaseConfiguration(from: decoder)
        self.perLayerQuantization = baseConfig?.perLayerQuantization

        // Validate rope_scaling if present
        if let ropeScaling {
            if ropeScaling["factor"] == nil {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain 'factor'")
            }
            if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
                if case .string = ropeType {
                    let options: [StringOrNumber] = [
                        .string("linear"), .string("dynamic"), .string("llama3")
                    ]
                    if !options.contains(ropeType) {
                        throw DecodingError.dataCorruptedError(
                            forKey: .ropeScaling, in: container,
                            debugDescription:
                            "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'")
                    }
                }
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain either 'type' or 'rope_type'")
            }
        }
    }
}
