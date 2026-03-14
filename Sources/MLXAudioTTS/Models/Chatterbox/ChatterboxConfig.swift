//
//  ChatterboxConfig.swift
//  MLXAudio
//
//  Chatterbox TTS configuration.
//  Ported from mlx-audio Python: chatterbox/config.py
//

import Foundation
import MLXLMCommon

// MARK: - LLaMA 520M Configuration (T3 backbone)

/// Static LLaMA configuration used by T3.
/// Maps to Python's LLAMA_520M_CONFIG dict.
public struct LlamaBackboneConfig: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: RopeScaling?
    public var attentionBias: Bool
    public var mlpBias: Bool
    public var tieWordEmbeddings: Bool

    public struct RopeScaling: Codable, Sendable {
        public var factor: Float
        public var highFreqFactor: Float
        public var lowFreqFactor: Float
        public var originalMaxPositionEmbeddings: Int
        public var ropeType: String

        enum CodingKeys: String, CodingKey {
            case factor
            case highFreqFactor = "high_freq_factor"
            case lowFreqFactor = "low_freq_factor"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case ropeType = "rope_type"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Default LLaMA 520M config matching Python's LLAMA_520M_CONFIG.
    public static let llama520M = LlamaBackboneConfig(
        modelType: "llama",
        vocabSize: 4000,
        hiddenSize: 1024,
        numHiddenLayers: 30,
        intermediateSize: 4096,
        numAttentionHeads: 16,
        numKeyValueHeads: 16,
        headDim: 64,
        maxPositionEmbeddings: 131072,
        rmsNormEps: 1e-05,
        ropeTheta: 500000.0,
        ropeScaling: RopeScaling(
            factor: 8.0,
            highFreqFactor: 4.0,
            lowFreqFactor: 1.0,
            originalMaxPositionEmbeddings: 8192,
            ropeType: "llama3"
        ),
        attentionBias: false,
        mlpBias: false,
        tieWordEmbeddings: false
    )
}

// MARK: - GPT-2 Medium Configuration (T3 Turbo backbone)

/// GPT-2 configuration used by Chatterbox Turbo T3.
/// Maps to Python's GPT2_MEDIUM_CONFIG dict.
public struct GPT2BackboneConfig: Codable, Sendable {
    public var activationFunction: String
    public var nCtx: Int
    public var hiddenSize: Int
    public var nHead: Int
    public var nLayer: Int
    public var vocabSize: Int
    public var layerNormEpsilon: Float

    enum CodingKeys: String, CodingKey {
        case activationFunction = "activation_function"
        case nCtx = "n_ctx"
        case hiddenSize = "hidden_size"
        case nHead = "n_head"
        case nLayer = "n_layer"
        case vocabSize = "vocab_size"
        case layerNormEpsilon = "layer_norm_epsilon"
    }

    /// Default GPT-2 Medium config matching Python's GPT2_MEDIUM_CONFIG.
    public static let medium = GPT2BackboneConfig(
        activationFunction: "gelu_new",
        nCtx: 8196,
        hiddenSize: 1024,
        nHead: 16,
        nLayer: 24,
        vocabSize: 50276,
        layerNormEpsilon: 1e-05
    )

    /// Head dimension (hiddenSize / nHead).
    public var headDim: Int { hiddenSize / nHead }

    /// Intermediate (feed-forward) size — GPT-2 uses 4x hidden size.
    public var intermediateSize: Int { hiddenSize * 4 }
}

// MARK: - T3 Configuration

/// Configuration for the T3 (Token-To-Token) model.
/// Maps to Python's T3Config dataclass.
public struct T3Configuration: Codable, Sendable {
    public var textTokensDictSize: Int
    public var startTextToken: Int
    public var stopTextToken: Int
    public var maxTextTokens: Int
    public var speechTokensDictSize: Int
    public var startSpeechToken: Int
    public var stopSpeechToken: Int
    public var maxSpeechTokens: Int
    public var llamaConfigName: String
    public var inputPosEmb: String?
    public var speechCondPromptLen: Int
    public var encoderType: String
    public var speakerEmbedSize: Int
    public var usePerceiverResampler: Bool
    public var emotionAdv: Bool

    enum CodingKeys: String, CodingKey {
        case textTokensDictSize = "text_tokens_dict_size"
        case startTextToken = "start_text_token"
        case stopTextToken = "stop_text_token"
        case maxTextTokens = "max_text_tokens"
        case speechTokensDictSize = "speech_tokens_dict_size"
        case startSpeechToken = "start_speech_token"
        case stopSpeechToken = "stop_speech_token"
        case maxSpeechTokens = "max_speech_tokens"
        case llamaConfigName = "llama_config_name"
        case inputPosEmb = "input_pos_emb"
        case speechCondPromptLen = "speech_cond_prompt_len"
        case encoderType = "encoder_type"
        case speakerEmbedSize = "speaker_embed_size"
        case usePerceiverResampler = "use_perceiver_resampler"
        case emotionAdv = "emotion_adv"
    }

    /// Whether this config uses a GPT-2 backbone (Turbo) vs LLaMA.
    public var isGPT: Bool {
        llamaConfigName.contains("GPT2")
    }

    /// Number of channels (hidden size) — 1024 for both LLaMA 520M and GPT-2 Medium.
    public var nChannels: Int {
        isGPT ? GPT2BackboneConfig.medium.hiddenSize : LlamaBackboneConfig.llama520M.hiddenSize
    }

    /// Number of transformer layers.
    public var numLayers: Int {
        isGPT ? GPT2BackboneConfig.medium.nLayer : LlamaBackboneConfig.llama520M.numHiddenLayers
    }

    /// Whether this is a multilingual model.
    public var isMultilingual: Bool {
        textTokensDictSize == 2454
    }

    /// Default English-only configuration (LLaMA backbone).
    public static let englishOnly = T3Configuration(
        textTokensDictSize: 704,
        startTextToken: 255,
        stopTextToken: 0,
        maxTextTokens: 2048,
        speechTokensDictSize: 8194,
        startSpeechToken: 6561,
        stopSpeechToken: 6562,
        maxSpeechTokens: 4096,
        llamaConfigName: "Llama_520M",
        inputPosEmb: "learned",
        speechCondPromptLen: 150,
        encoderType: "voice_encoder",
        speakerEmbedSize: 256,
        usePerceiverResampler: true,
        emotionAdv: true
    )

    /// Default multilingual configuration (LLaMA backbone).
    public static let multilingual = T3Configuration(
        textTokensDictSize: 2454,
        startTextToken: 255,
        stopTextToken: 0,
        maxTextTokens: 2048,
        speechTokensDictSize: 8194,
        startSpeechToken: 6561,
        stopSpeechToken: 6562,
        maxSpeechTokens: 4096,
        llamaConfigName: "Llama_520M",
        inputPosEmb: "learned",
        speechCondPromptLen: 150,
        encoderType: "voice_encoder",
        speakerEmbedSize: 256,
        usePerceiverResampler: true,
        emotionAdv: true
    )

    /// Turbo configuration (GPT-2 backbone).
    public static let turbo = T3Configuration(
        textTokensDictSize: 50276,
        startTextToken: 255,
        stopTextToken: 0,
        maxTextTokens: 2048,
        speechTokensDictSize: 6563,
        startSpeechToken: 6561,
        stopSpeechToken: 6562,
        maxSpeechTokens: 4096,
        llamaConfigName: "GPT2_medium",
        inputPosEmb: nil,
        speechCondPromptLen: 375,
        encoderType: "voice_encoder",
        speakerEmbedSize: 256,
        usePerceiverResampler: false,
        emotionAdv: false
    )
}

// MARK: - Voice Encoder Configuration

/// Configuration for the VoiceEncoder (LSTM-based speaker embedding).
/// Maps to Python's VoiceEncConfig dataclass.
public struct VoiceEncoderConfiguration: Codable, Sendable {
    public var numMels: Int
    public var sampleRate: Int
    public var speakerEmbedSize: Int
    public var veHiddenSize: Int
    public var nFft: Int
    public var hopSize: Int
    public var winSize: Int
    public var fmax: Int
    public var fmin: Int
    public var preemphasis: Float
    public var melPower: Float
    public var melType: String
    public var normalizedMels: Bool
    public var vePartialFrames: Int
    public var veFinalRelu: Bool
    public var stftMagnitudeMin: Float

    enum CodingKeys: String, CodingKey {
        case numMels = "num_mels"
        case sampleRate = "sample_rate"
        case speakerEmbedSize = "speaker_embed_size"
        case veHiddenSize = "ve_hidden_size"
        case nFft = "n_fft"
        case hopSize = "hop_size"
        case winSize = "win_size"
        case fmax, fmin
        case preemphasis
        case melPower = "mel_power"
        case melType = "mel_type"
        case normalizedMels = "normalized_mels"
        case vePartialFrames = "ve_partial_frames"
        case veFinalRelu = "ve_final_relu"
        case stftMagnitudeMin = "stft_magnitude_min"
    }

    /// Default configuration.
    public static let `default` = VoiceEncoderConfiguration(
        numMels: 40,
        sampleRate: 16000,
        speakerEmbedSize: 256,
        veHiddenSize: 256,
        nFft: 400,
        hopSize: 160,
        winSize: 400,
        fmax: 8000,
        fmin: 0,
        preemphasis: 0.0,
        melPower: 2.0,
        melType: "amp",
        normalizedMels: false,
        vePartialFrames: 160,
        veFinalRelu: true,
        stftMagnitudeMin: 1e-4
    )
}

// MARK: - Top-Level Model Configuration

/// Top-level Chatterbox model configuration.
/// Maps to Python's ModelConfig dataclass.
///
/// Supports two config formats:
/// - Regular: minimal `{"model_type": "chatterbox"}` with defaults
/// - Turbo: full config with `t3`, `gpt2`, `voice_encoder`, `s3gen` sections
public struct ChatterboxConfiguration: Codable, Sendable {
    public var modelType: String
    public var t3Config: T3Configuration
    public var gpt2Config: GPT2BackboneConfig?
    public var s3Sr: Int
    public var s3genSr: Int
    public var sampleRate: Int
    public var encCondLen: Int
    public var decCondLen: Int
    public var modelPath: String?

    // S3Gen decoder config
    public var meanflow: Bool
    public var decoderInChannels: Int
    public var decoderOutChannels: Int
    public var decoderChannels: [Int]
    public var decoderNBlocks: Int
    public var decoderNumMidBlocks: Int
    public var decoderNumHeads: Int
    public var decoderAttentionHeadDim: Int

    // Quantization
    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    /// Whether this is a Turbo (GPT-2) model.
    public var isTurbo: Bool {
        modelType == "chatterbox_turbo" || t3Config.isGPT
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case t3Config = "t3_config"
        case t3 = "t3"
        case gpt2 = "gpt2"
        case s3Sr = "s3_sr"
        case s3genSr = "s3gen_sr"
        case sampleRate = "sample_rate"
        case encCondLen = "enc_cond_len"
        case encCondLenSeconds = "enc_cond_len_seconds"
        case decCondLen = "dec_cond_len"
        case decCondLenSeconds = "dec_cond_len_seconds"
        case modelPath = "model_path"
        case meanflow
        case decoderInChannels = "decoder_in_channels"
        case decoderOutChannels = "decoder_out_channels"
        case decoderChannels = "decoder_channels"
        case decoderNBlocks = "decoder_n_blocks"
        case decoderNumMidBlocks = "decoder_num_mid_blocks"
        case decoderNumHeads = "decoder_num_heads"
        case decoderAttentionHeadDim = "decoder_attention_head_dim"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(
        modelType: String,
        t3Config: T3Configuration,
        gpt2Config: GPT2BackboneConfig? = nil,
        s3Sr: Int = 16000,
        s3genSr: Int = 24000,
        sampleRate: Int = 24000,
        encCondLen: Int = 6 * 16000,
        decCondLen: Int = 10 * 24000,
        modelPath: String? = nil,
        meanflow: Bool = true,
        decoderInChannels: Int = 320,
        decoderOutChannels: Int = 80,
        decoderChannels: [Int] = [256],
        decoderNBlocks: Int = 4,
        decoderNumMidBlocks: Int = 12,
        decoderNumHeads: Int = 8,
        decoderAttentionHeadDim: Int = 64,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelType = modelType
        self.t3Config = t3Config
        self.gpt2Config = gpt2Config
        self.s3Sr = s3Sr
        self.s3genSr = s3genSr
        self.sampleRate = sampleRate
        self.encCondLen = encCondLen
        self.decCondLen = decCondLen
        self.modelPath = modelPath
        self.meanflow = meanflow
        self.decoderInChannels = decoderInChannels
        self.decoderOutChannels = decoderOutChannels
        self.decoderChannels = decoderChannels
        self.decoderNBlocks = decoderNBlocks
        self.decoderNumMidBlocks = decoderNumMidBlocks
        self.decoderNumHeads = decoderNumHeads
        self.decoderAttentionHeadDim = decoderAttentionHeadDim
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "chatterbox"

        // T3 config: try "t3_config" first (our format), then "t3" (HF Turbo format)
        if let t3 = try container.decodeIfPresent(T3Configuration.self, forKey: .t3Config) {
            self.t3Config = t3
        } else if let t3 = try container.decodeIfPresent(T3Configuration.self, forKey: .t3) {
            self.t3Config = t3
        } else if modelType == "chatterbox_turbo" {
            self.t3Config = .turbo
        } else {
            self.t3Config = .englishOnly
        }

        // GPT-2 config (Turbo only)
        self.gpt2Config = try container.decodeIfPresent(GPT2BackboneConfig.self, forKey: .gpt2)

        self.s3Sr = try container.decodeIfPresent(Int.self, forKey: .s3Sr) ?? 16000
        self.s3genSr = try container.decodeIfPresent(Int.self, forKey: .s3genSr) ?? 24000
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000

        // Support both absolute lengths and seconds-based lengths
        if let encLen = try container.decodeIfPresent(Int.self, forKey: .encCondLen) {
            self.encCondLen = encLen
        } else if let encSec = try container.decodeIfPresent(Int.self, forKey: .encCondLenSeconds) {
            self.encCondLen = encSec * self.s3Sr
        } else {
            self.encCondLen = 6 * self.s3Sr
        }

        if let decLen = try container.decodeIfPresent(Int.self, forKey: .decCondLen) {
            self.decCondLen = decLen
        } else if let decSec = try container.decodeIfPresent(Int.self, forKey: .decCondLenSeconds) {
            self.decCondLen = decSec * self.s3genSr
        } else {
            self.decCondLen = 10 * self.s3genSr
        }

        self.modelPath = try container.decodeIfPresent(String.self, forKey: .modelPath)

        // S3Gen decoder config
        // meanflow=true for Turbo (distilled flow matching), false for Regular (ODE solver with CFG)
        let isTurboModel = self.modelType == "chatterbox_turbo" || (self.t3Config.isGPT)
        self.meanflow = try container.decodeIfPresent(Bool.self, forKey: .meanflow) ?? isTurboModel
        self.decoderInChannels = try container.decodeIfPresent(Int.self, forKey: .decoderInChannels) ?? 320
        self.decoderOutChannels = try container.decodeIfPresent(Int.self, forKey: .decoderOutChannels) ?? 80
        self.decoderChannels = try container.decodeIfPresent([Int].self, forKey: .decoderChannels) ?? [256]
        self.decoderNBlocks = try container.decodeIfPresent(Int.self, forKey: .decoderNBlocks) ?? 4
        self.decoderNumMidBlocks = try container.decodeIfPresent(Int.self, forKey: .decoderNumMidBlocks) ?? 12
        self.decoderNumHeads = try container.decodeIfPresent(Int.self, forKey: .decoderNumHeads) ?? 8
        self.decoderAttentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .decoderAttentionHeadDim) ?? 64

        // Quantization
        let baseConfig = try? BaseConfiguration(from: decoder)
        let globalQuant = try container.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization
        )
        let altGlobalQuant = try container.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantizationConfig
        )
        self.quantization = globalQuant ?? altGlobalQuant ?? baseConfig?.quantization
        self.perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(t3Config, forKey: .t3Config)
        try container.encodeIfPresent(gpt2Config, forKey: .gpt2)
        try container.encode(s3Sr, forKey: .s3Sr)
        try container.encode(s3genSr, forKey: .s3genSr)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(encCondLen, forKey: .encCondLen)
        try container.encode(decCondLen, forKey: .decCondLen)
        try container.encodeIfPresent(modelPath, forKey: .modelPath)
        try container.encode(meanflow, forKey: .meanflow)
        try container.encode(decoderInChannels, forKey: .decoderInChannels)
        try container.encode(decoderOutChannels, forKey: .decoderOutChannels)
        try container.encode(decoderChannels, forKey: .decoderChannels)
        try container.encode(decoderNBlocks, forKey: .decoderNBlocks)
        try container.encode(decoderNumMidBlocks, forKey: .decoderNumMidBlocks)
        try container.encode(decoderNumHeads, forKey: .decoderNumHeads)
        try container.encode(decoderAttentionHeadDim, forKey: .decoderAttentionHeadDim)
    }

    /// Default configuration (regular Chatterbox).
    public static let `default` = ChatterboxConfiguration(
        modelType: "chatterbox",
        t3Config: .englishOnly
    )

    /// Turbo configuration.
    public static let turbo = ChatterboxConfiguration(
        modelType: "chatterbox_turbo",
        t3Config: .turbo,
        gpt2Config: .medium,
        encCondLen: 15 * 16000,
        decCondLen: 10 * 24000
    )
}

// MARK: - Constants

/// Global constants for Chatterbox.
public enum ChatterboxConstants {
    /// S3 tokenizer sample rate (16kHz).
    public static let s3SampleRate = 16000
    /// S3Gen output sample rate (24kHz).
    public static let s3genSampleRate = 24000
    /// Speech vocabulary size (before special tokens).
    public static let speechVocabSize = 6561
    /// Encoder conditioning length: 6 seconds at 16kHz.
    public static let encCondLen = 6 * s3SampleRate
    /// Decoder conditioning length: 10 seconds at 24kHz.
    public static let decCondLen = 10 * s3genSampleRate
}
