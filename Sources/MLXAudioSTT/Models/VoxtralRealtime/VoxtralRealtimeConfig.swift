//
//  VoxtralRealtimeConfig.swift
//  MLXAudioSTT
//

import Foundation
import MLXLMCommon

// MARK: - Audio Encoding Config

public struct AudioEncodingConfig: Codable {
    public var samplingRate: Int
    public var frameRate: Float
    public var numMelBins: Int
    public var hopLength: Int
    public var windowSize: Int
    public var globalLogMelMax: Float

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case frameRate = "frame_rate"
        case numMelBins = "num_mel_bins"
        case hopLength = "hop_length"
        case windowSize = "window_size"
        case globalLogMelMax = "global_log_mel_max"
    }

    public init(
        samplingRate: Int = 16000,
        frameRate: Float = 12.5,
        numMelBins: Int = 128,
        hopLength: Int = 160,
        windowSize: Int = 400,
        globalLogMelMax: Float = 1.5
    ) {
        self.samplingRate = samplingRate
        self.frameRate = frameRate
        self.numMelBins = numMelBins
        self.hopLength = hopLength
        self.windowSize = windowSize
        self.globalLogMelMax = globalLogMelMax
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
        frameRate = try c.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
        windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 400
        globalLogMelMax = try c.decodeIfPresent(Float.self, forKey: .globalLogMelMax) ?? 1.5
    }
}

// MARK: - Encoder Config

public struct VoxtralEncoderConfig: Codable {
    public var dim: Int
    public var nLayers: Int
    public var nHeads: Int
    public var headDim: Int
    public var hiddenDim: Int
    public var nKvHeads: Int
    public var normEps: Float
    public var ropeTheta: Float
    public var slidingWindow: Int
    public var causal: Bool
    public var useBiases: Bool
    public var downsampleFactor: Int

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nKvHeads = "n_kv_heads"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case causal
        case useBiases = "use_biases"
        case downsampleFactor = "downsample_factor"
    }

    public init(
        dim: Int = 1280,
        nLayers: Int = 32,
        nHeads: Int = 32,
        headDim: Int = 64,
        hiddenDim: Int = 5120,
        nKvHeads: Int = 32,
        normEps: Float = 1e-5,
        ropeTheta: Float = 1_000_000.0,
        slidingWindow: Int = 750,
        causal: Bool = true,
        useBiases: Bool = true,
        downsampleFactor: Int = 4
    ) {
        self.dim = dim
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.nKvHeads = nKvHeads
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.slidingWindow = slidingWindow
        self.causal = causal
        self.useBiases = useBiases
        self.downsampleFactor = downsampleFactor
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 1280
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 32
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 32
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 5120
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 32
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 750
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        useBiases = try c.decodeIfPresent(Bool.self, forKey: .useBiases) ?? true
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 4
    }
}

// MARK: - Decoder Config

public struct VoxtralDecoderConfig: Codable {
    public var dim: Int
    public var nLayers: Int
    public var nHeads: Int
    public var nKvHeads: Int
    public var headDim: Int
    public var hiddenDim: Int
    public var vocabSize: Int
    public var normEps: Float
    public var ropeTheta: Float
    public var slidingWindow: Int
    public var tiedEmbeddings: Bool
    public var adaRmsNormTCond: Bool
    public var adaRmsNormTCondDim: Int

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case vocabSize = "vocab_size"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case tiedEmbeddings = "tied_embeddings"
        case adaRmsNormTCond = "ada_rms_norm_t_cond"
        case adaRmsNormTCondDim = "ada_rms_norm_t_cond_dim"
    }

    public init(
        dim: Int = 3072,
        nLayers: Int = 26,
        nHeads: Int = 32,
        nKvHeads: Int = 8,
        headDim: Int = 128,
        hiddenDim: Int = 9216,
        vocabSize: Int = 131072,
        normEps: Float = 1e-5,
        ropeTheta: Float = 1_000_000.0,
        slidingWindow: Int = 8192,
        tiedEmbeddings: Bool = true,
        adaRmsNormTCond: Bool = true,
        adaRmsNormTCondDim: Int = 32
    ) {
        self.dim = dim
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.vocabSize = vocabSize
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.slidingWindow = slidingWindow
        self.tiedEmbeddings = tiedEmbeddings
        self.adaRmsNormTCond = adaRmsNormTCond
        self.adaRmsNormTCondDim = adaRmsNormTCondDim
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 3072
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 26
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 32
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 9216
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 131072
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 8192
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? true
        adaRmsNormTCond = try c.decodeIfPresent(Bool.self, forKey: .adaRmsNormTCond) ?? true
        adaRmsNormTCondDim = try c.decodeIfPresent(Int.self, forKey: .adaRmsNormTCondDim) ?? 32
    }
}

// MARK: - Model Config

public struct VoxtralRealtimeModelConfig: Codable {
    public var modelType: String
    public var encoderArgs: VoxtralEncoderConfig
    public var decoder: VoxtralDecoderConfig
    public var audioEncodingArgs: AudioEncodingConfig

    public var vocabSize: Int
    public var hiddenSize: Int

    public var transcriptionDelayMs: Int
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var streamingPadTokenId: Int
    public var nLeftPadTokens: Int

    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoderArgs = "encoder_args"
        case decoder
        case audioEncodingArgs = "audio_encoding_args"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case transcriptionDelayMs = "transcription_delay_ms"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case streamingPadTokenId = "streaming_pad_token_id"
        case nLeftPadTokens = "n_left_pad_tokens"
    }

    public init(
        modelType: String = "voxtral_realtime",
        encoderArgs: VoxtralEncoderConfig = VoxtralEncoderConfig(),
        decoder: VoxtralDecoderConfig = VoxtralDecoderConfig(),
        audioEncodingArgs: AudioEncodingConfig = AudioEncodingConfig(),
        transcriptionDelayMs: Int = 480,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        streamingPadTokenId: Int = 32,
        nLeftPadTokens: Int = 32
    ) {
        self.modelType = modelType
        self.encoderArgs = encoderArgs
        self.decoder = decoder
        self.audioEncodingArgs = audioEncodingArgs
        self.vocabSize = decoder.vocabSize
        self.hiddenSize = decoder.dim
        self.transcriptionDelayMs = transcriptionDelayMs
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.streamingPadTokenId = streamingPadTokenId
        self.nLeftPadTokens = nLeftPadTokens
        self.perLayerQuantization = nil
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxtral_realtime"

        // Handle nested audio_encoding_args inside encoder_args
        if var encDict = try? c.decode([String: AnyCodable].self, forKey: .encoderArgs) {
            if let audioEncVal = encDict.removeValue(forKey: "audio_encoding_args") {
                // Re-encode the audio_encoding_args portion
                let audioEncData = try JSONSerialization.data(
                    withJSONObject: audioEncVal.value)
                audioEncodingArgs = try JSONDecoder().decode(
                    AudioEncodingConfig.self, from: audioEncData)
            } else {
                audioEncodingArgs =
                    try c.decodeIfPresent(AudioEncodingConfig.self, forKey: .audioEncodingArgs)
                    ?? AudioEncodingConfig()
            }
            // Re-encode remaining encoder dict
            let encData = try JSONSerialization.data(
                withJSONObject: encDict.mapValues { $0.value })
            encoderArgs = try JSONDecoder().decode(VoxtralEncoderConfig.self, from: encData)
        } else {
            encoderArgs =
                try c.decodeIfPresent(VoxtralEncoderConfig.self, forKey: .encoderArgs)
                ?? VoxtralEncoderConfig()
            audioEncodingArgs =
                try c.decodeIfPresent(AudioEncodingConfig.self, forKey: .audioEncodingArgs)
                ?? AudioEncodingConfig()
        }

        let dec =
            try c.decodeIfPresent(VoxtralDecoderConfig.self, forKey: .decoder)
            ?? VoxtralDecoderConfig()
        self.decoder = dec

        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? dec.vocabSize
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? dec.dim
        transcriptionDelayMs = try c.decodeIfPresent(Int.self, forKey: .transcriptionDelayMs) ?? 480
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        streamingPadTokenId = try c.decodeIfPresent(Int.self, forKey: .streamingPadTokenId) ?? 32
        nLeftPadTokens = try c.decodeIfPresent(Int.self, forKey: .nLeftPadTokens) ?? 32

        let baseConfig = try? BaseConfiguration(from: decoder)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }
}
