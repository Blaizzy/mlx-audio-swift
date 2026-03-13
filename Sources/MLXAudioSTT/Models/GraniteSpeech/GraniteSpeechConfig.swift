//
//  GraniteSpeechConfig.swift
//  MLXAudioSTT
//

import Foundation
import MLXLMCommon

// MARK: - Encoder Config (CTC Conformer)

public struct GraniteSpeechEncoderConfig: Codable {
    public var inputDim: Int
    public var numLayers: Int
    public var hiddenDim: Int
    public var feedforwardMult: Int
    public var numHeads: Int
    public var dimHead: Int
    public var outputDim: Int
    public var contextSize: Int
    public var maxPosEmb: Int
    public var dropout: Float
    public var convKernelSize: Int
    public var convExpansionFactor: Int

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case numLayers = "num_layers"
        case hiddenDim = "hidden_dim"
        case feedforwardMult = "feedforward_mult"
        case numHeads = "num_heads"
        case dimHead = "dim_head"
        case outputDim = "output_dim"
        case contextSize = "context_size"
        case maxPosEmb = "max_pos_emb"
        case dropout
        case convKernelSize = "conv_kernel_size"
        case convExpansionFactor = "conv_expansion_factor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? 160
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 10
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 1024
        feedforwardMult = try c.decodeIfPresent(Int.self, forKey: .feedforwardMult) ?? 4
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 8
        dimHead = try c.decodeIfPresent(Int.self, forKey: .dimHead) ?? 128
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 42
        contextSize = try c.decodeIfPresent(Int.self, forKey: .contextSize) ?? 200
        maxPosEmb = try c.decodeIfPresent(Int.self, forKey: .maxPosEmb) ?? 512
        dropout = try c.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.1
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 15
        convExpansionFactor = try c.decodeIfPresent(Int.self, forKey: .convExpansionFactor) ?? 2
    }
}

// MARK: - Projector Config (QFormer / BLIP-2)

public struct GraniteSpeechProjectorConfig: Codable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var hiddenAct: String
    public var layerNormEps: Float
    public var encoderHiddenSize: Int
    public var crossAttentionFrequency: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
        case encoderHiddenSize = "encoder_hidden_size"
        case crossAttentionFrequency = "cross_attention_frequency"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 2
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-12
        encoderHiddenSize = try c.decodeIfPresent(Int.self, forKey: .encoderHiddenSize) ?? 1024
        crossAttentionFrequency = try c.decodeIfPresent(Int.self, forKey: .crossAttentionFrequency) ?? 1
    }
}

// MARK: - Text Config (Granite LLM)

public struct GraniteSpeechTextConfig: Codable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var attentionBias: Bool
    public var mlpBias: Bool
    public var attentionMultiplier: Float
    public var embeddingMultiplier: Float
    public var residualMultiplier: Float
    public var logitsScaling: Float
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case logitsScaling = "logits_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granite"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 100353
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 40
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 4
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        attentionMultiplier = try c.decodeIfPresent(Float.self, forKey: .attentionMultiplier) ?? 0.0078125
        embeddingMultiplier = try c.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 12.0
        residualMultiplier = try c.decodeIfPresent(Float.self, forKey: .residualMultiplier) ?? 0.22
        logitsScaling = try c.decodeIfPresent(Float.self, forKey: .logitsScaling) ?? 8.0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - Top-Level Model Config

public struct GraniteSpeechModelConfig: Codable {
    public var modelType: String
    public var encoderConfig: GraniteSpeechEncoderConfig
    public var projectorConfig: GraniteSpeechProjectorConfig
    public var textConfig: GraniteSpeechTextConfig
    public var audioTokenIndex: Int
    public var downsampleRate: Int
    public var windowSize: Int
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoderConfig = "encoder_config"
        case projectorConfig = "projector_config"
        case textConfig = "text_config"
        case audioTokenIndex = "audio_token_index"
        case downsampleRate = "downsample_rate"
        case windowSize = "window_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granite_speech"
        encoderConfig = try c.decodeIfPresent(GraniteSpeechEncoderConfig.self, forKey: .encoderConfig)
            ?? GraniteSpeechEncoderConfig(from: decoder)
        projectorConfig = try c.decodeIfPresent(GraniteSpeechProjectorConfig.self, forKey: .projectorConfig)
            ?? GraniteSpeechProjectorConfig(from: decoder)
        textConfig = try c.decodeIfPresent(GraniteSpeechTextConfig.self, forKey: .textConfig)
            ?? GraniteSpeechTextConfig(from: decoder)
        audioTokenIndex = try c.decodeIfPresent(Int.self, forKey: .audioTokenIndex) ?? 100352
        downsampleRate = try c.decodeIfPresent(Int.self, forKey: .downsampleRate) ?? 5
        windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 15

        let baseConfig = try? BaseConfiguration(from: decoder)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }
}
