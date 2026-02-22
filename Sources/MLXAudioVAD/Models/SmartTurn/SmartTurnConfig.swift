import Foundation

public struct SmartTurnEncoderConfig: Codable, Sendable {
    public var modelType: String
    public var numMelBins: Int
    public var maxSourcePositions: Int
    public var dModel: Int
    public var encoderAttentionHeads: Int
    public var encoderLayers: Int
    public var encoderFfnDim: Int
    public var kProjBias: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case dModel = "d_model"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderLayers = "encoder_layers"
        case encoderFfnDim = "encoder_ffn_dim"
        case kProjBias = "k_proj_bias"
    }

    public init(
        modelType: String = "smart_turn_encoder",
        numMelBins: Int = 80,
        maxSourcePositions: Int = 400,
        dModel: Int = 384,
        encoderAttentionHeads: Int = 6,
        encoderLayers: Int = 4,
        encoderFfnDim: Int = 1536,
        kProjBias: Bool = false
    ) {
        self.modelType = modelType
        self.numMelBins = numMelBins
        self.maxSourcePositions = maxSourcePositions
        self.dModel = dModel
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderLayers = encoderLayers
        self.encoderFfnDim = encoderFfnDim
        self.kProjBias = kProjBias
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "smart_turn_encoder"
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 80
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 400
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 384
        encoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 6
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 4
        encoderFfnDim = try c.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 1536
        kProjBias = try c.decodeIfPresent(Bool.self, forKey: .kProjBias) ?? false
    }
}

public struct SmartTurnProcessorConfig: Codable, Sendable {
    public var samplingRate: Int
    public var maxAudioSeconds: Int
    public var nFft: Int
    public var hopLength: Int
    public var nMels: Int
    public var normalizeAudio: Bool
    public var threshold: Float

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case maxAudioSeconds = "max_audio_seconds"
        case nFft = "n_fft"
        case hopLength = "hop_length"
        case nMels = "n_mels"
        case normalizeAudio = "normalize_audio"
        case threshold
    }

    public init(
        samplingRate: Int = 16000,
        maxAudioSeconds: Int = 8,
        nFft: Int = 400,
        hopLength: Int = 160,
        nMels: Int = 80,
        normalizeAudio: Bool = true,
        threshold: Float = 0.5
    ) {
        self.samplingRate = samplingRate
        self.maxAudioSeconds = maxAudioSeconds
        self.nFft = nFft
        self.hopLength = hopLength
        self.nMels = nMels
        self.normalizeAudio = normalizeAudio
        self.threshold = threshold
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
        maxAudioSeconds = try c.decodeIfPresent(Int.self, forKey: .maxAudioSeconds) ?? 8
        nFft = try c.decodeIfPresent(Int.self, forKey: .nFft) ?? 400
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
        nMels = try c.decodeIfPresent(Int.self, forKey: .nMels) ?? 80
        normalizeAudio = try c.decodeIfPresent(Bool.self, forKey: .normalizeAudio) ?? true
        threshold = try c.decodeIfPresent(Float.self, forKey: .threshold) ?? 0.5
    }
}

public struct SmartTurnConfig: Codable, Sendable {
    public var modelType: String
    public var architecture: String
    public var dtype: String
    public var encoderConfig: SmartTurnEncoderConfig
    public var processorConfig: SmartTurnProcessorConfig

    // Compatibility keys from conversion scripts.
    public var sampleRate: Int
    public var maxAudioSeconds: Int
    public var threshold: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture
        case dtype
        case encoderConfig = "encoder_config"
        case processorConfig = "processor_config"
        case sampleRate = "sample_rate"
        case maxAudioSeconds = "max_audio_seconds"
        case threshold
    }

    public init(
        modelType: String = "smart_turn",
        architecture: String = "smart_turn",
        dtype: String = "float32",
        encoderConfig: SmartTurnEncoderConfig = SmartTurnEncoderConfig(),
        processorConfig: SmartTurnProcessorConfig? = nil,
        sampleRate: Int = 16000,
        maxAudioSeconds: Int = 8,
        threshold: Float = 0.5
    ) {
        self.modelType = modelType
        self.architecture = architecture
        self.dtype = dtype
        self.encoderConfig = encoderConfig
        self.sampleRate = sampleRate
        self.maxAudioSeconds = maxAudioSeconds
        self.threshold = threshold
        self.processorConfig = processorConfig ?? SmartTurnProcessorConfig(
            samplingRate: sampleRate,
            maxAudioSeconds: maxAudioSeconds,
            nMels: encoderConfig.numMelBins,
            threshold: threshold
        )
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "smart_turn"
        architecture = try c.decodeIfPresent(String.self, forKey: .architecture) ?? "smart_turn"
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype) ?? "float32"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        maxAudioSeconds = try c.decodeIfPresent(Int.self, forKey: .maxAudioSeconds) ?? 8
        threshold = try c.decodeIfPresent(Float.self, forKey: .threshold) ?? 0.5

        encoderConfig = try c.decodeIfPresent(SmartTurnEncoderConfig.self, forKey: .encoderConfig)
            ?? SmartTurnEncoderConfig()

        if let decodedProcessor = try c.decodeIfPresent(SmartTurnProcessorConfig.self, forKey: .processorConfig) {
            processorConfig = decodedProcessor
        } else {
            processorConfig = SmartTurnProcessorConfig(
                samplingRate: sampleRate,
                maxAudioSeconds: maxAudioSeconds,
                nMels: encoderConfig.numMelBins,
                threshold: threshold
            )
        }
    }
}
