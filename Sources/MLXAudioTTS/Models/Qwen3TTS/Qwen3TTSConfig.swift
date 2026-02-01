//
//  Qwen3TTSConfig.swift
//  MLXAudio
//
//  Configuration structures for Qwen3-TTS model.
//  Ported from mlx_audio/tts/models/qwen3_tts/config.py
//

import Foundation

// MARK: - JSONValue for flexible dictionary values

public enum JSONValue: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case intArray([Int])
    case object([String: JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([Int].self) {
            self = .intArray(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if container.decodeNil() {
            self = .null
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .intArray(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    /// Extract [Int] from JSONValue if possible
    public var intArrayValue: [Int]? {
        switch self {
        case .intArray(let arr): return arr
        case .array(let arr): return arr.compactMap {
            if case .int(let v) = $0 { return v }
            return nil
        }
        default: return nil
        }
    }

    /// Extract Bool from JSONValue if possible
    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    /// Extract String from JSONValue if possible
    public var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }
}

// MARK: - Speaker Encoder Config

/// Configuration for ECAPA-TDNN speaker encoder.
public struct Qwen3TTSSpeakerEncoderConfig: Codable, Sendable {
    public let melDim: Int
    public let encDim: Int
    public let encChannels: [Int]
    public let encKernelSizes: [Int]
    public let encDilations: [Int]
    public let encAttentionChannels: Int
    public let encRes2netScale: Int
    public let encSeChannels: Int
    public let sampleRate: Int

    public init(
        melDim: Int = 128,
        encDim: Int = 1024,
        encChannels: [Int] = [512, 512, 512, 512, 1536],
        encKernelSizes: [Int] = [5, 3, 3, 3, 1],
        encDilations: [Int] = [1, 2, 3, 4, 1],
        encAttentionChannels: Int = 128,
        encRes2netScale: Int = 8,
        encSeChannels: Int = 128,
        sampleRate: Int = 24000
    ) {
        self.melDim = melDim
        self.encDim = encDim
        self.encChannels = encChannels
        self.encKernelSizes = encKernelSizes
        self.encDilations = encDilations
        self.encAttentionChannels = encAttentionChannels
        self.encRes2netScale = encRes2netScale
        self.encSeChannels = encSeChannels
        self.sampleRate = sampleRate
    }

    private enum CodingKeys: String, CodingKey {
        case melDim = "mel_dim"
        case encDim = "enc_dim"
        case encChannels = "enc_channels"
        case encKernelSizes = "enc_kernel_sizes"
        case encDilations = "enc_dilations"
        case encAttentionChannels = "enc_attention_channels"
        case encRes2netScale = "enc_res2net_scale"
        case encSeChannels = "enc_se_channels"
        case sampleRate = "sample_rate"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.melDim = try container.decodeIfPresent(Int.self, forKey: .melDim) ?? 128
        self.encDim = try container.decodeIfPresent(Int.self, forKey: .encDim) ?? 1024
        self.encChannels = try container.decodeIfPresent([Int].self, forKey: .encChannels) ?? [512, 512, 512, 512, 1536]
        self.encKernelSizes = try container.decodeIfPresent([Int].self, forKey: .encKernelSizes) ?? [5, 3, 3, 3, 1]
        self.encDilations = try container.decodeIfPresent([Int].self, forKey: .encDilations) ?? [1, 2, 3, 4, 1]
        self.encAttentionChannels = try container.decodeIfPresent(Int.self, forKey: .encAttentionChannels) ?? 128
        self.encRes2netScale = try container.decodeIfPresent(Int.self, forKey: .encRes2netScale) ?? 8
        self.encSeChannels = try container.decodeIfPresent(Int.self, forKey: .encSeChannels) ?? 128
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
    }
}

// MARK: - Talker Code Predictor Config

/// Configuration for the code predictor sub-model.
public struct Qwen3TTSTalkerCodePredictorConfig: Codable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: JSONValue]?
    public let attentionBias: Bool
    public let slidingWindow: Int?
    public let layerTypes: [String]?
    public let attentionDropout: Float
    public let numCodeGroups: Int

    public init(
        vocabSize: Int = 2048,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 5,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 65536,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1000000.0,
        ropeScaling: [String: JSONValue]? = nil,
        attentionBias: Bool = false,
        slidingWindow: Int? = nil,
        layerTypes: [String]? = nil,
        attentionDropout: Float = 0.0,
        numCodeGroups: Int = 16
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: numHiddenLayers)
        self.attentionDropout = attentionDropout
        self.numCodeGroups = numCodeGroups
    }

    private enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionDropout = "attention_dropout"
        case numCodeGroups = "num_code_groups"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let numLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 5
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 2048
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        self.numHiddenLayers = numLayers
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        self.ropeScaling = try container.decodeIfPresent([String: JSONValue].self, forKey: .ropeScaling)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? Array(repeating: "full_attention", count: numLayers)
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.numCodeGroups = try container.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
    }
}

// MARK: - Talker Config

/// Configuration for the main talker model.
public struct Qwen3TTSTalkerConfig: Codable, Sendable {
    public let codePredictorConfig: Qwen3TTSTalkerCodePredictorConfig?
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: JSONValue]?
    public let attentionBias: Bool
    public let slidingWindow: Int?
    public let attentionDropout: Float
    public let numCodeGroups: Int
    public let textHiddenSize: Int
    public let textVocabSize: Int
    public let codecEosTokenId: Int
    public let codecThinkId: Int
    public let codecNothinkId: Int
    public let codecThinkBosId: Int
    public let codecThinkEosId: Int
    public let codecPadId: Int
    public let codecBosId: Int
    public let codecLanguageId: [String: Int]?
    public let spkId: [String: Int]?
    public let spkIsDialect: [String: JSONValue]?
    public let ttsPadTokenId: Int
    public let ttsBosTokenId: Int
    public let ttsEosTokenId: Int

    /// Extract mrope_section from ropeScaling if present
    public var mropeSection: [Int] {
        if let scaling = ropeScaling,
           let section = scaling["mrope_section"]?.intArrayValue {
            return section
        }
        return [24, 20, 20]  // Default
    }

    /// Check if interleaved MRoPE is enabled
    public var isInterleaved: Bool {
        if let scaling = ropeScaling,
           let interleaved = scaling["interleaved"]?.boolValue {
            return interleaved
        }
        return true  // Default
    }

    public init(
        codePredictorConfig: Qwen3TTSTalkerCodePredictorConfig? = nil,
        vocabSize: Int = 3072,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 32768,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1000000.0,
        ropeScaling: [String: JSONValue]? = [
            "interleaved": .bool(true),
            "mrope_section": .intArray([24, 20, 20]),
            "rope_type": .string("default")
        ],
        attentionBias: Bool = false,
        slidingWindow: Int? = nil,
        attentionDropout: Float = 0.0,
        numCodeGroups: Int = 16,
        textHiddenSize: Int = 2048,
        textVocabSize: Int = 151936,
        codecEosTokenId: Int = 2150,
        codecThinkId: Int = 2154,
        codecNothinkId: Int = 2155,
        codecThinkBosId: Int = 2156,
        codecThinkEosId: Int = 2157,
        codecPadId: Int = 2148,
        codecBosId: Int = 2149,
        codecLanguageId: [String: Int]? = nil,
        spkId: [String: Int]? = nil,
        spkIsDialect: [String: JSONValue]? = nil,
        ttsPadTokenId: Int = 151671,
        ttsBosTokenId: Int = 151672,
        ttsEosTokenId: Int = 151673
    ) {
        self.codePredictorConfig = codePredictorConfig ?? Qwen3TTSTalkerCodePredictorConfig()
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.slidingWindow = slidingWindow
        self.attentionDropout = attentionDropout
        self.numCodeGroups = numCodeGroups
        self.textHiddenSize = textHiddenSize
        self.textVocabSize = textVocabSize
        self.codecEosTokenId = codecEosTokenId
        self.codecThinkId = codecThinkId
        self.codecNothinkId = codecNothinkId
        self.codecThinkBosId = codecThinkBosId
        self.codecThinkEosId = codecThinkEosId
        self.codecPadId = codecPadId
        self.codecBosId = codecBosId
        self.codecLanguageId = codecLanguageId
        self.spkId = spkId
        self.spkIsDialect = spkIsDialect
        self.ttsPadTokenId = ttsPadTokenId
        self.ttsBosTokenId = ttsBosTokenId
        self.ttsEosTokenId = ttsEosTokenId
    }

    private enum CodingKeys: String, CodingKey {
        case codePredictorConfig = "code_predictor_config"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case attentionDropout = "attention_dropout"
        case numCodeGroups = "num_code_groups"
        case textHiddenSize = "text_hidden_size"
        case textVocabSize = "text_vocab_size"
        case codecEosTokenId = "codec_eos_token_id"
        case codecThinkId = "codec_think_id"
        case codecNothinkId = "codec_nothink_id"
        case codecThinkBosId = "codec_think_bos_id"
        case codecThinkEosId = "codec_think_eos_id"
        case codecPadId = "codec_pad_id"
        case codecBosId = "codec_bos_id"
        case codecLanguageId = "codec_language_id"
        case spkId = "spk_id"
        case spkIsDialect = "spk_is_dialect"
        case ttsPadTokenId = "tts_pad_token_id"
        case ttsBosTokenId = "tts_bos_token_id"
        case ttsEosTokenId = "tts_eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.codePredictorConfig = try container.decodeIfPresent(Qwen3TTSTalkerCodePredictorConfig.self, forKey: .codePredictorConfig)
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 3072
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        self.ropeScaling = try container.decodeIfPresent([String: JSONValue].self, forKey: .ropeScaling)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.numCodeGroups = try container.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
        self.textHiddenSize = try container.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? 2048
        self.textVocabSize = try container.decodeIfPresent(Int.self, forKey: .textVocabSize) ?? 151936
        self.codecEosTokenId = try container.decodeIfPresent(Int.self, forKey: .codecEosTokenId) ?? 2150
        self.codecThinkId = try container.decodeIfPresent(Int.self, forKey: .codecThinkId) ?? 2154
        self.codecNothinkId = try container.decodeIfPresent(Int.self, forKey: .codecNothinkId) ?? 2155
        self.codecThinkBosId = try container.decodeIfPresent(Int.self, forKey: .codecThinkBosId) ?? 2156
        self.codecThinkEosId = try container.decodeIfPresent(Int.self, forKey: .codecThinkEosId) ?? 2157
        self.codecPadId = try container.decodeIfPresent(Int.self, forKey: .codecPadId) ?? 2148
        self.codecBosId = try container.decodeIfPresent(Int.self, forKey: .codecBosId) ?? 2149
        self.codecLanguageId = try container.decodeIfPresent([String: Int].self, forKey: .codecLanguageId)
        self.spkId = try container.decodeIfPresent([String: Int].self, forKey: .spkId)
        self.spkIsDialect = try container.decodeIfPresent([String: JSONValue].self, forKey: .spkIsDialect)
        self.ttsPadTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsPadTokenId) ?? 151671
        self.ttsBosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsBosTokenId) ?? 151672
        self.ttsEosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsEosTokenId) ?? 151673
    }
}

// MARK: - Tokenizer Decoder Config

/// Configuration for the speech tokenizer decoder.
public struct Qwen3TTSTokenizerDecoderConfig: Codable, Sendable {
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let latentDim: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let decoderDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerScaleInitialScale: Float
    public let maxPositionEmbeddings: Int
    public let headDim: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let numQuantizers: Int
    public let numSemanticQuantizers: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let semanticCodebookSize: Int
    public let slidingWindow: Int
    public let upsampleRates: [Int]
    public let upsamplingRatios: [Int]
    public let vectorQuantizationHiddenDimension: Int

    public init(
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        latentDim: Int = 1024,
        codebookDim: Int = 512,
        codebookSize: Int = 2048,
        decoderDim: Int = 1536,
        hiddenAct: String = "silu",
        hiddenSize: Int = 512,
        intermediateSize: Int = 1024,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        headDim: Int = 64,
        numAttentionHeads: Int = 16,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 16,
        numQuantizers: Int = 16,
        numSemanticQuantizers: Int = 1,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10000.0,
        semanticCodebookSize: Int = 4096,
        slidingWindow: Int = 72,
        upsampleRates: [Int] = [8, 5, 4, 3],
        upsamplingRatios: [Int] = [2, 2],
        vectorQuantizationHiddenDimension: Int = 512
    ) {
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.latentDim = latentDim
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.decoderDim = decoderDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.headDim = headDim
        self.numAttentionHeads = numAttentionHeads
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numSemanticQuantizers = numSemanticQuantizers
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.semanticCodebookSize = semanticCodebookSize
        self.slidingWindow = slidingWindow
        self.upsampleRates = upsampleRates
        self.upsamplingRatios = upsamplingRatios
        self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
    }

    private enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case latentDim = "latent_dim"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case decoderDim = "decoder_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case semanticCodebookSize = "semantic_codebook_size"
        case slidingWindow = "sliding_window"
        case upsampleRates = "upsample_rates"
        case upsamplingRatios = "upsampling_ratios"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.latentDim = try container.decodeIfPresent(Int.self, forKey: .latentDim) ?? 1024
        self.codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 512
        self.codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        self.decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1536
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1024
        self.layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 16
        self.numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 16
        self.numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.semanticCodebookSize = try container.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 4096
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 72
        self.upsampleRates = try container.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 5, 4, 3]
        self.upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [2, 2]
        self.vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 512
    }
}

// MARK: - Tokenizer Encoder Config

/// Configuration for the speech tokenizer encoder.
public struct Qwen3TTSTokenizerEncoderConfig: Codable, Sendable {
    public let frameRate: Float
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let audioChannels: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let compress: Int
    public let dilationGrowthRate: Int
    public let headDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let kernelSize: Int
    public let lastKernelSize: Int
    public let layerScaleInitialScale: Float
    public let maxPositionEmbeddings: Int
    public let normEps: Float
    public let numAttentionHeads: Int
    public let numFilters: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let numQuantizers: Int
    public let numResidualLayers: Int
    public let numSemanticQuantizers: Int
    public let residualKernelSize: Int
    public let ropeTheta: Float
    public let samplingRate: Int
    public let slidingWindow: Int
    public let upsamplingRatios: [Int]
    public let useCausalConv: Bool
    public let useConvShortcut: Bool
    public let vectorQuantizationHiddenDimension: Int

    public init(
        frameRate: Float = 12.5,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        audioChannels: Int = 1,
        codebookDim: Int = 256,
        codebookSize: Int = 2048,
        compress: Int = 2,
        dilationGrowthRate: Int = 2,
        headDim: Int = 64,
        hiddenAct: String = "gelu",
        hiddenSize: Int = 512,
        intermediateSize: Int = 2048,
        kernelSize: Int = 7,
        lastKernelSize: Int = 3,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        normEps: Float = 1e-5,
        numAttentionHeads: Int = 8,
        numFilters: Int = 64,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 8,
        numQuantizers: Int = 32,
        numResidualLayers: Int = 1,
        numSemanticQuantizers: Int = 1,
        residualKernelSize: Int = 3,
        ropeTheta: Float = 10000.0,
        samplingRate: Int = 24000,
        slidingWindow: Int = 250,
        upsamplingRatios: [Int] = [8, 6, 5, 4],
        useCausalConv: Bool = true,
        useConvShortcut: Bool = false,
        vectorQuantizationHiddenDimension: Int = 256
    ) {
        self.frameRate = frameRate
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.audioChannels = audioChannels
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.compress = compress
        self.dilationGrowthRate = dilationGrowthRate
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.kernelSize = kernelSize
        self.lastKernelSize = lastKernelSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normEps = normEps
        self.numAttentionHeads = numAttentionHeads
        self.numFilters = numFilters
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numResidualLayers = numResidualLayers
        self.numSemanticQuantizers = numSemanticQuantizers
        self.residualKernelSize = residualKernelSize
        self.ropeTheta = ropeTheta
        self.samplingRate = samplingRate
        self.slidingWindow = slidingWindow
        self.upsamplingRatios = upsamplingRatios
        self.useCausalConv = useCausalConv
        self.useConvShortcut = useConvShortcut
        self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
    }

    private enum CodingKeys: String, CodingKey {
        case frameRate = "frame_rate"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case audioChannels = "audio_channels"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case compress
        case dilationGrowthRate = "dilation_growth_rate"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case numAttentionHeads = "num_attention_heads"
        case numFilters = "num_filters"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numResidualLayers = "num_residual_layers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case residualKernelSize = "residual_kernel_size"
        case ropeTheta = "rope_theta"
        case samplingRate = "sampling_rate"
        case slidingWindow = "sliding_window"
        case upsamplingRatios = "upsampling_ratios"
        case useCausalConv = "use_causal_conv"
        case useConvShortcut = "use_conv_shortcut"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.frameRate = try container.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.audioChannels = try container.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 1
        self.codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 256
        self.codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        self.compress = try container.decodeIfPresent(Int.self, forKey: .compress) ?? 2
        self.dilationGrowthRate = try container.decodeIfPresent(Int.self, forKey: .dilationGrowthRate) ?? 2
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        self.kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        self.lastKernelSize = try container.decodeIfPresent(Int.self, forKey: .lastKernelSize) ?? 3
        self.layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        self.normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.numFilters = try container.decodeIfPresent(Int.self, forKey: .numFilters) ?? 64
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 32
        self.numResidualLayers = try container.decodeIfPresent(Int.self, forKey: .numResidualLayers) ?? 1
        self.numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        self.residualKernelSize = try container.decodeIfPresent(Int.self, forKey: .residualKernelSize) ?? 3
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 250
        self.upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [8, 6, 5, 4]
        self.useCausalConv = try container.decodeIfPresent(Bool.self, forKey: .useCausalConv) ?? true
        self.useConvShortcut = try container.decodeIfPresent(Bool.self, forKey: .useConvShortcut) ?? false
        self.vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 256
    }
}

// MARK: - Tokenizer Config

/// Configuration for the speech tokenizer.
public struct Qwen3TTSTokenizerConfig: Codable, Sendable {
    public let encoderConfig: Qwen3TTSTokenizerEncoderConfig?
    public let decoderConfig: Qwen3TTSTokenizerDecoderConfig?
    public let encoderValidNumQuantizers: Int
    public let inputSampleRate: Int
    public let outputSampleRate: Int
    public let decodeUpsampleRate: Int
    public let encodeDownsampleRate: Int

    public init(
        encoderConfig: Qwen3TTSTokenizerEncoderConfig? = nil,
        decoderConfig: Qwen3TTSTokenizerDecoderConfig? = nil,
        encoderValidNumQuantizers: Int = 16,
        inputSampleRate: Int = 24000,
        outputSampleRate: Int = 24000,
        decodeUpsampleRate: Int = 1920,
        encodeDownsampleRate: Int = 1920
    ) {
        self.encoderConfig = encoderConfig ?? Qwen3TTSTokenizerEncoderConfig()
        self.decoderConfig = decoderConfig ?? Qwen3TTSTokenizerDecoderConfig()
        self.encoderValidNumQuantizers = encoderValidNumQuantizers
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
        self.decodeUpsampleRate = decodeUpsampleRate
        self.encodeDownsampleRate = encodeDownsampleRate
    }

    private enum CodingKeys: String, CodingKey {
        case encoderConfig = "encoder_config"
        case decoderConfig = "decoder_config"
        case encoderValidNumQuantizers = "encoder_valid_num_quantizers"
        case inputSampleRate = "input_sample_rate"
        case outputSampleRate = "output_sample_rate"
        case decodeUpsampleRate = "decode_upsample_rate"
        case encodeDownsampleRate = "encode_downsample_rate"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.encoderConfig = try container.decodeIfPresent(Qwen3TTSTokenizerEncoderConfig.self, forKey: .encoderConfig)
        self.decoderConfig = try container.decodeIfPresent(Qwen3TTSTokenizerDecoderConfig.self, forKey: .decoderConfig)
        self.encoderValidNumQuantizers = try container.decodeIfPresent(Int.self, forKey: .encoderValidNumQuantizers) ?? 16
        self.inputSampleRate = try container.decodeIfPresent(Int.self, forKey: .inputSampleRate) ?? 24000
        self.outputSampleRate = try container.decodeIfPresent(Int.self, forKey: .outputSampleRate) ?? 24000
        self.decodeUpsampleRate = try container.decodeIfPresent(Int.self, forKey: .decodeUpsampleRate) ?? 1920
        self.encodeDownsampleRate = try container.decodeIfPresent(Int.self, forKey: .encodeDownsampleRate) ?? 1920
    }
}

// MARK: - Quantization Config

/// Configuration for quantized models.
public struct Qwen3TTSQuantization: Codable, Sendable, Equatable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String?

    public init(groupSize: Int = 64, bits: Int = 4, mode: String? = "affine") {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }
}

// MARK: - Main Model Config

/// Main configuration for Qwen3-TTS model.
public struct Qwen3TTSModelConfig: Codable, Sendable {
    public let modelType: String
    public let talkerConfig: Qwen3TTSTalkerConfig?
    public let speakerEncoderConfig: Qwen3TTSSpeakerEncoderConfig?
    public let tokenizerConfig: Qwen3TTSTokenizerConfig?
    public let tokenizerType: String
    public let ttsModelSize: String
    public let ttsModelType: String
    public let imStartTokenId: Int
    public let imEndTokenId: Int
    public let ttsPadTokenId: Int
    public let ttsBosTokenId: Int
    public let ttsEosTokenId: Int
    public let sampleRate: Int

    /// Quantization config (present for quantized models like 4-bit)
    public let quantization: Qwen3TTSQuantization?
    /// Alternative key for quantization config
    public let quantizationConfig: Qwen3TTSQuantization?

    /// Returns the effective quantization config (checks both keys)
    public var effectiveQuantization: Qwen3TTSQuantization? {
        quantization ?? quantizationConfig
    }

    public init(
        modelType: String = "qwen3_tts",
        talkerConfig: Qwen3TTSTalkerConfig? = nil,
        speakerEncoderConfig: Qwen3TTSSpeakerEncoderConfig? = nil,
        tokenizerConfig: Qwen3TTSTokenizerConfig? = nil,
        tokenizerType: String = "qwen3_tts_tokenizer_12hz",
        ttsModelSize: String = "0b6",
        ttsModelType: String = "base",
        imStartTokenId: Int = 151644,
        imEndTokenId: Int = 151645,
        ttsPadTokenId: Int = 151671,
        ttsBosTokenId: Int = 151672,
        ttsEosTokenId: Int = 151673,
        sampleRate: Int = 24000,
        quantization: Qwen3TTSQuantization? = nil,
        quantizationConfig: Qwen3TTSQuantization? = nil
    ) {
        self.modelType = modelType
        self.talkerConfig = talkerConfig ?? Qwen3TTSTalkerConfig()
        self.speakerEncoderConfig = speakerEncoderConfig ?? Qwen3TTSSpeakerEncoderConfig()
        self.tokenizerConfig = tokenizerConfig
        self.tokenizerType = tokenizerType
        self.ttsModelSize = ttsModelSize
        self.ttsModelType = ttsModelType
        self.imStartTokenId = imStartTokenId
        self.imEndTokenId = imEndTokenId
        self.ttsPadTokenId = ttsPadTokenId
        self.ttsBosTokenId = ttsBosTokenId
        self.ttsEosTokenId = ttsEosTokenId
        self.sampleRate = sampleRate
        self.quantization = quantization
        self.quantizationConfig = quantizationConfig
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case talkerConfig = "talker_config"
        case speakerEncoderConfig = "speaker_encoder_config"
        case tokenizerConfig = "tokenizer_config"
        case tokenizerType = "tokenizer_type"
        case ttsModelSize = "tts_model_size"
        case ttsModelType = "tts_model_type"
        case imStartTokenId = "im_start_token_id"
        case imEndTokenId = "im_end_token_id"
        case ttsPadTokenId = "tts_pad_token_id"
        case ttsBosTokenId = "tts_bos_token_id"
        case ttsEosTokenId = "tts_eos_token_id"
        case sampleRate = "sample_rate"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_tts"
        self.talkerConfig = try container.decodeIfPresent(Qwen3TTSTalkerConfig.self, forKey: .talkerConfig)
        self.speakerEncoderConfig = try container.decodeIfPresent(Qwen3TTSSpeakerEncoderConfig.self, forKey: .speakerEncoderConfig)
        self.tokenizerConfig = try container.decodeIfPresent(Qwen3TTSTokenizerConfig.self, forKey: .tokenizerConfig)
        self.tokenizerType = try container.decodeIfPresent(String.self, forKey: .tokenizerType) ?? "qwen3_tts_tokenizer_12hz"
        self.ttsModelSize = try container.decodeIfPresent(String.self, forKey: .ttsModelSize) ?? "0b6"
        self.ttsModelType = try container.decodeIfPresent(String.self, forKey: .ttsModelType) ?? "base"
        self.imStartTokenId = try container.decodeIfPresent(Int.self, forKey: .imStartTokenId) ?? 151644
        self.imEndTokenId = try container.decodeIfPresent(Int.self, forKey: .imEndTokenId) ?? 151645
        self.ttsPadTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsPadTokenId) ?? 151671
        self.ttsBosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsBosTokenId) ?? 151672
        self.ttsEosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsEosTokenId) ?? 151673
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        self.quantization = try container.decodeIfPresent(Qwen3TTSQuantization.self, forKey: .quantization)
        self.quantizationConfig = try container.decodeIfPresent(Qwen3TTSQuantization.self, forKey: .quantizationConfig)
    }

    /// Load configuration from a JSON file
    public static func load(from url: URL) throws -> Qwen3TTSModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(Qwen3TTSModelConfig.self, from: data)
    }
}
