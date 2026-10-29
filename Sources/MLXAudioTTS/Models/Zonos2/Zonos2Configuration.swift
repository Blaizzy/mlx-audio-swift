import Foundation

/// Configuration for published ZONOS2 checkpoints.
public struct Zonos2Configuration: Codable, Sendable {
    public var modelType: String = "zonos2"
    public var nLayers: Int = 28
    public var dim: Int = 2048
    public var headDim: Int = 128
    public var nHeads: Int? = nil
    public var nKVHeads: Int? = 4
    public var ffnDimMultiplier: Float = 1.5
    public var multipleOf: Int = 256
    public var normEps: Float = 1e-5
    public var ropeTheta: Float = 10000
    public var maxSequenceLength: Int = 6144
    public var nCodebooks: Int = 9
    public var codebookSize: Int = 1024
    public var eoaID: Int = 1024
    public var audioPadID: Int = 1025
    public var textVocab: Int = 519
    public var lossSoftcap: Float = 15
    public var sampleRate: Int = 44100
    public var dacModelID: String = "mlx-community/descript-audio-codec-44khz"
    public var speakerEnabled: Bool = true
    public var speakerEmbeddingDim: Int = 2048
    public var speakerLDADim: Int? = 1024
    public var speakerEncoderModelID: String = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
    public var speakerEncoderPath: String? = "speaker_encoder"
    public var speakerEncoderSampleRate: Int = 24000
    public var speakerBackgroundTokenEnabled: Bool = true
    public var accurateModeTokenEnabled: Bool = true
    public var speakingRateNumBuckets: Int = 8
    public var qualityFeatures: [String] = [
        "lufs", "estimated_snr", "max_pause", "estimated_bandlimit_hz", "leading_silence_s",
        "trailing_silence_s",
    ]
    public var qualityBuckets: [String: [String]] = [:]
    public var moeNExperts: Int = 16
    public var moeRouterTopK: Int = 1
    public var specialTopKLayers: [String: Int] = ["26": 2]
    public var moeRouterDim: Int = 128
    public var moeStartFromLayer: Int = 3
    public var moeEndFromLayer: Int = 1
    public var normTopKProbability: Bool = false
    public var moeBalancingStrategy: String = "legacy"
    public init() {}
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case nLayers = "n_layers"
        case dim = "dim"
        case headDim = "head_dim"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case ffnDimMultiplier = "ffn_dim_multiplier"
        case multipleOf = "multiple_of"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case maxSequenceLength = "max_seqlen"
        case nCodebooks = "n_codebooks"
        case codebookSize = "codebook_size"
        case eoaID = "eoa_id"
        case audioPadID = "audio_pad_id"
        case textVocab = "text_vocab"
        case lossSoftcap = "loss_softcap"
        case sampleRate = "sample_rate"
        case dacModelID = "dac_model_id"
        case speakerEnabled = "speaker_enabled"
        case speakerEmbeddingDim = "speaker_embedding_dim"
        case speakerLDADim = "speaker_lda_dim"
        case speakerEncoderModelID = "speaker_encoder_model_id"
        case speakerEncoderPath = "speaker_encoder_path"
        case speakerEncoderSampleRate = "speaker_encoder_sample_rate"
        case speakerBackgroundTokenEnabled = "speaker_background_token_enabled"
        case accurateModeTokenEnabled = "accurate_mode_token_enabled"
        case speakingRateNumBuckets = "speaking_rate_num_buckets"
        case qualityFeatures = "quality_features"
        case qualityBuckets = "quality_buckets"
        case moeNExperts = "moe_n_experts"
        case moeRouterTopK = "moe_router_topk"
        case specialTopKLayers = "special_topk_layers"
        case moeRouterDim = "moe_router_dim"
        case moeStartFromLayer = "moe_start_from_layer"
        case moeEndFromLayer = "moe_end_from_layer"
        case normTopKProbability = "norm_topk_prob"
        case moeBalancingStrategy = "moe_balancing_strategy"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "zonos2"
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 28
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 2048
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? nil
        nKVHeads = c.contains(.nKVHeads) ? try c.decodeIfPresent(Int.self, forKey: .nKVHeads) : 4
        ffnDimMultiplier = try c.decodeIfPresent(Float.self, forKey: .ffnDimMultiplier) ?? 1.5
        multipleOf = try c.decodeIfPresent(Int.self, forKey: .multipleOf) ?? 256
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        maxSequenceLength = try c.decodeIfPresent(Int.self, forKey: .maxSequenceLength) ?? 6144
        nCodebooks = try c.decodeIfPresent(Int.self, forKey: .nCodebooks) ?? 9
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 1024
        eoaID = try c.decodeIfPresent(Int.self, forKey: .eoaID) ?? 1024
        audioPadID = try c.decodeIfPresent(Int.self, forKey: .audioPadID) ?? 1025
        textVocab = try c.decodeIfPresent(Int.self, forKey: .textVocab) ?? 519
        lossSoftcap = try c.decodeIfPresent(Float.self, forKey: .lossSoftcap) ?? 15
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 44100
        dacModelID =
            try c.decodeIfPresent(String.self, forKey: .dacModelID)
            ?? "mlx-community/descript-audio-codec-44khz"
        speakerEnabled = try c.decodeIfPresent(Bool.self, forKey: .speakerEnabled) ?? true
        speakerEmbeddingDim = try c.decodeIfPresent(Int.self, forKey: .speakerEmbeddingDim) ?? 2048
        speakerLDADim =
            c.contains(.speakerLDADim) ? try c.decodeIfPresent(Int.self, forKey: .speakerLDADim) : 1024
        speakerEncoderModelID =
            try c.decodeIfPresent(String.self, forKey: .speakerEncoderModelID)
            ?? "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
        speakerEncoderPath =
            c.contains(.speakerEncoderPath)
            ? try c.decodeIfPresent(String.self, forKey: .speakerEncoderPath) : "speaker_encoder"
        speakerEncoderSampleRate =
            try c.decodeIfPresent(Int.self, forKey: .speakerEncoderSampleRate) ?? 24000
        speakerBackgroundTokenEnabled =
            try c.decodeIfPresent(Bool.self, forKey: .speakerBackgroundTokenEnabled) ?? true
        accurateModeTokenEnabled =
            try c.decodeIfPresent(Bool.self, forKey: .accurateModeTokenEnabled) ?? true
        speakingRateNumBuckets = try c.decodeIfPresent(Int.self, forKey: .speakingRateNumBuckets) ?? 8
        qualityFeatures =
            try c.decodeIfPresent([String].self, forKey: .qualityFeatures) ?? [
                "lufs", "estimated_snr", "max_pause", "estimated_bandlimit_hz", "leading_silence_s",
                "trailing_silence_s",
            ]
        qualityBuckets = try c.decodeIfPresent([String: [String]].self, forKey: .qualityBuckets) ?? [:]
        moeNExperts = try c.decodeIfPresent(Int.self, forKey: .moeNExperts) ?? 16
        moeRouterTopK = try c.decodeIfPresent(Int.self, forKey: .moeRouterTopK) ?? 1
        specialTopKLayers =
            try c.decodeIfPresent([String: Int].self, forKey: .specialTopKLayers) ?? ["26": 2]
        moeRouterDim = try c.decodeIfPresent(Int.self, forKey: .moeRouterDim) ?? 128
        moeStartFromLayer = try c.decodeIfPresent(Int.self, forKey: .moeStartFromLayer) ?? 3
        moeEndFromLayer = try c.decodeIfPresent(Int.self, forKey: .moeEndFromLayer) ?? 1
        normTopKProbability = try c.decodeIfPresent(Bool.self, forKey: .normTopKProbability) ?? false
        moeBalancingStrategy =
            try c.decodeIfPresent(String.self, forKey: .moeBalancingStrategy) ?? "legacy"
    }
    public var numHeads: Int { nHeads ?? dim / headDim }
    public var numKVHeads: Int { nKVHeads ?? numHeads }
    public var intermediateSize: Int {
        multipleOf * ((Int(ffnDimMultiplier * Float(dim)) + multipleOf - 1) / multipleOf)
    }
    public var audioVocabSize: Int { codebookSize + 2 }
    public var qualityBucketCounts: [Int] {
        if qualityBuckets.isEmpty {
            return qualityFeatures.map {
                [
                    "lufs": 12, "estimated_snr": 12, "max_pause": 12, "estimated_bandlimit_hz": 8,
                    "leading_silence_s": 8, "trailing_silence_s": 8,
                ][$0] ?? 0
            }
        }
        return qualityFeatures.map { qualityBuckets[$0]?.count ?? 0 }
    }
    public var speakerBackgroundCount: Int { speakerBackgroundTokenEnabled ? 2 : 0 }
    public var accurateModeCount: Int {
        accurateModeTokenEnabled && speakerBackgroundTokenEnabled ? 1 : 0
    }
    func isMoE(_ index: Int) -> Bool {
        moeNExperts > 1 && index >= moeStartFromLayer && nLayers - index > moeEndFromLayer
    }
    func topK(_ index: Int) -> Int { specialTopKLayers[String(index)] ?? moeRouterTopK }
    func validate() throws {
        guard nLayers > 0, dim > 0, headDim > 0, headDim % 2 == 0, numHeads > 0, numKVHeads > 0,
            numHeads % numKVHeads == 0, multipleOf > 0, ffnDimMultiplier > 0, normEps > 0,
            (1...9).contains(nCodebooks), codebookSize > 0, eoaID == codebookSize,
            audioPadID == codebookSize + 1,
            textVocab >= 448 + speakingRateNumBuckets + qualityBucketCounts.reduce(0, +)
                + speakerBackgroundCount + accurateModeCount,
            speakingRateNumBuckets >= 0, moeRouterDim > 0, moeNExperts > 0, sampleRate > 0,
            speakerEmbeddingDim > 0, speakerEncoderSampleRate > 0,
            (speakerLDADim ?? 1) > 0, maxSequenceLength > 0
        else {
            throw Zonos2Error.invalidConfiguration(
                "Invalid transformer, codebook or conditioning dimensions")
        }
        for i in 0..<nLayers where isMoE(i) {
            guard (1...moeNExperts).contains(topK(i)) else {
                throw Zonos2Error.invalidConfiguration("Invalid router top-k at layer \(i)")
            }
        }
    }
}

public enum Zonos2Error: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case missingComponent(String)
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let s), .invalidInput(let s), .missingComponent(let s): return s
        }
    }
}

/// ZONOS2 controls used by both waveform and incremental generation.
public struct Zonos2Options: Sendable {
    public var speakingRateBucket: Int?
    /// Feature names follow `config.qualityFeatures`. An empty dictionary disables quality markers.
    public var qualityBuckets: [String: Int] = ["trailing_silence_s": 3]
    public var cleanSpeakerBackground = false
    public var accurateMode = true
    public var textNormalization = true
    public var repetitionCodebooks = 8
    public var ignoreEOS = false
    public var referenceSampleRate: Int?
    public init() {}
}
