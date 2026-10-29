import Foundation
import MLXAudioCore
import MLXAudioVAD

private struct NemotronConfigKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(_ value: String) { stringValue = value }
    init?(stringValue: String) { self.init(stringValue) }
    init?(intValue: Int) { return nil }
}

extension KeyedDecodingContainer where Key == NemotronConfigKey {
    fileprivate func value<T: Decodable>(_ key: String, _ fallback: T) throws -> T {
        try decodeIfPresent(T.self, forKey: .init(key)) ?? fallback
    }
}

public struct NemotronSpeechEncoderConfiguration: Decodable, Sendable {
    public var features = 128
    public var width = 1280
    public var layers = 32
    public var heads = 10
    public var factor = 8
    public var expansion: Float = 4
    public var qkvBias = false
    public var qkNorm = true
    public var preBlockNorm = true
    public var scaleInput = false
    public var rotaryBase: Float = 10000
    public var rotaryFraction: Float = 0.5
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: NemotronConfigKey.self)
        features = try c.value("feat_in", features)
        width = try c.value("d_model", width)
        layers = try c.value("n_layers", layers)
        heads = try c.value("n_heads", heads)
        factor = try c.value("subsampling_factor", factor)
        expansion = try c.value("ff_expansion", expansion)
        qkvBias = try c.value("qkv_bias", qkvBias)
        qkNorm = try c.value("qk_norm", false)
        preBlockNorm = try c.value("pre_block_norm", preBlockNorm)
        scaleInput = try c.value("xscaling", scaleInput)
        rotaryBase = try c.value("rope_base", rotaryBase)
        rotaryFraction = try c.value("rotary_fraction", 1)
        guard try c.value("subsampling", "feature_stacking") == "feature_stacking",
            try c.value("self_attention_model", "rope") == "rope",
            try c.value("attn_mode", "full") == "full", try c.value("feat_out", -1) == -1
        else {
            throw STTError.modelNotInitialized("Unsupported Nemotron speech encoder")
        }
    }
    func validate() throws {
        guard features > 0, width > 0, width % 2 == 0, layers > 0, heads > 0, width % heads == 0,
            factor > 0, expansion > 0, rotaryBase > 0,
            rotaryFraction > 0, rotaryFraction <= 1,
            Int(Float(width / heads) * rotaryFraction) >= 2,
            Int(Float(width / heads) * rotaryFraction) % 2 == 0
        else {
            throw STTError.modelNotInitialized("Invalid Nemotron speech encoder dimensions")
        }
    }
}

public struct NemotronMelConfiguration: Decodable, Sendable {
    public var sampleRate = 16000
    public var features = 128
    public var fftSize = 512
    public var hopLength = 160
    public var windowLength = 400
    public var preemphasis: Float = 0.97
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: NemotronConfigKey.self)
        sampleRate = try c.value("sample_rate", sampleRate)
        features = try c.value("features", features)
        fftSize = try c.value("n_fft", fftSize)
        hopLength = Int(Float(sampleRate) * (try c.value("window_stride", Float(0.01))))
        windowLength = Int(Float(sampleRate) * (try c.value("window_size", Float(0.025))))
        preemphasis = try c.value("preemph", preemphasis)
        let normalization = try c.decodeIfPresent(String.self, forKey: .init("normalize"))
        guard normalization == nil || normalization == "NA",
            try c.value("window", "hann") == "hann", try c.value("log", true),
            try c.value("frame_splicing", 1) == 1, try !c.value("exact_pad", false),
            try c.value("mag_power", Float(2)) == 2,
            try c.value("log_zero_guard_type", "add") == "add",
            try c.value("log_zero_guard_value", Float(pow(2.0, -24))) == Float(pow(2.0, -24))
        else {
            throw STTError.modelNotInitialized("Unsupported Nemotron log-mel preprocessing")
        }
    }
    func validate() throws {
        guard sampleRate > 0, features > 0, fftSize > 0, hopLength > 0,
            windowLength > 1, windowLength <= fftSize, preemphasis.isFinite
        else {
            throw STTError.modelNotInitialized("Invalid Nemotron log-mel configuration")
        }
    }
}

public struct NemotronSpeakerConfiguration: Decodable, Sendable {
    public var encoder: NemotronSpeechEncoderConfiguration
    public var preprocessor = NemotronMelConfiguration()
    public var modules: ModulesConfig
    public init() {
        var encoder = NemotronSpeechEncoderConfiguration()
        encoder.width = 512
        encoder.layers = 31
        encoder.heads = 8
        encoder.qkNorm = false
        encoder.rotaryFraction = 1
        self.encoder = encoder
        var modules = ModulesConfig(numSpeakers: 8)
        modules.chunkLen = 500
        modules.fifoLen = 40
        modules.spkcacheLen = 188
        modules.spkcacheUpdatePeriod = 300
        modules.chunkLeftContext = 0
        modules.chunkRightContext = 0
        modules.spkcacheSilFramesPerSpk = 1
        modules.predScoreThreshold = 0.25
        modules.maxIndex = 99999
        modules.scoresBoostLatest = 0.05
        modules.silThreshold = 0.2
        modules.strongBoostRate = 0.75
        modules.weakBoostRate = 1.5
        modules.minPosScoresRate = 0.5
        modules.useAosc = true
        self.modules = modules
    }
    public init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: NemotronConfigKey.self)
        guard try c.value("high_resolution", false), !c.contains(.init("transformer_encoder"))
        else {
            throw STTError.modelNotInitialized(
                "Expected Nemotron's high-resolution single speaker encoder")
        }
        encoder = try c.decode(NemotronSpeechEncoderConfiguration.self, forKey: .init("encoder"))
        preprocessor = try c.decode(NemotronMelConfiguration.self, forKey: .init("preprocessor"))
        let m = try c.nestedContainer(
            keyedBy: NemotronConfigKey.self, forKey: .init("sortformer_modules"))
        modules.numSpeakers = try m.value("num_spks", modules.numSpeakers)
        modules.fcDModel = try m.value("fc_d_model", modules.fcDModel)
        modules.tfDModel = try m.value("tf_d_model", modules.tfDModel)
        modules.subsamplingFactor = encoder.factor
        modules.spkcacheSilFramesPerSpk = try m.value(
            "spkcache_sil_frames_per_spk", modules.spkcacheSilFramesPerSpk)
        modules.predScoreThreshold = try m.value("pred_score_threshold", modules.predScoreThreshold)
        modules.maxIndex = try m.value("max_index", modules.maxIndex)
        modules.scoresBoostLatest = try m.value("scores_boost_latest", modules.scoresBoostLatest)
        modules.strongBoostRate = try m.value("strong_boost_rate", modules.strongBoostRate)
        modules.weakBoostRate = try m.value("weak_boost_rate", modules.weakBoostRate)
        modules.minPosScoresRate = try m.value("min_pos_scores_rate", modules.minPosScoresRate)
        guard try m.value("use_learnable_sil_emb", true) else {
            throw STTError.modelNotInitialized("Expected a learned speaker-cache silence embedding")
        }
    }
}

public struct NemotronTranscribeConfiguration: Decodable, Sendable {
    public var text = NemotronHConfiguration()
    public var speech = NemotronSpeechEncoderConfiguration()
    public var speaker = NemotronSpeakerConfiguration()
    public var preprocessor = NemotronMelConfiguration()
    public var outputDimension = 2688
    public var audioToken = "<|audio|>"
    public var eosTokenIDs = [11]
    public var padTokenID = 0
    public var windowLength = 500
    public var leftContext = 50
    public var rightContext = 50
    public var speakerMode = "thresholded"
    public var speakerThreshold: Float? = 0.5
    public var speakerScale: Float = 1
    public init() {}
    public init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: NemotronConfigKey.self)
        text = try c.decode(NemotronHConfiguration.self, forKey: .init("llm_config"))
        let lm = try c.nestedContainer(keyedBy: NemotronConfigKey.self, forKey: .init("llm_config"))
        guard try lm.value("model_type", "") == "nemotron_h",
            try c.value("prompt_format", "nemotron3p5") == "nemotron3p5"
        else {
            throw STTError.modelNotInitialized("Expected the Nemotron-H speech language backbone")
        }
        audioToken = try c.value("audio_locator_tag", audioToken)
        padTokenID = try c.value("pad_token_id", padTokenID)
        if let ids = try? c.decode([Int].self, forKey: .init("eos_token_id")) {
            eosTokenIDs = ids
        } else {
            eosTokenIDs = [try c.value("eos_token_id", 11)]
        }
        let perception = try c.nestedContainer(
            keyedBy: NemotronConfigKey.self, forKey: .init("perception"))
        preprocessor = try perception.decode(
            NemotronMelConfiguration.self, forKey: .init("preprocessor"))
        outputDimension = try perception.decode(Int.self, forKey: .init("output_dim"))
        let adapter = try perception.nestedContainer(
            keyedBy: NemotronConfigKey.self, forKey: .init("modality_adapter"))
        let target = try adapter.value("_target_", "")
        guard target.split(separator: ".").last == "IdentityConnector",
            !perception.contains(.init("rote"))
                || (try? perception.decodeNil(forKey: .init("rote"))) == true
        else {
            throw STTError.modelNotInitialized("Unsupported Nemotron perception adapter")
        }
        let pe = try c.nestedContainer(
            keyedBy: NemotronConfigKey.self, forKey: .init("pe_encoder_config"))
        speech = try pe.decode(
            NemotronSpeechEncoderConfiguration.self, forKey: .init("asr_encoder_cfg"))
        speaker = try pe.decode(
            NemotronSpeakerConfiguration.self, forKey: .init("diarization_model_cfg"))
        windowLength = try pe.value("online_inference_length", windowLength)
        leftContext = max(0, try pe.value("chunk_left_context", leftContext))
        rightContext = max(0, try pe.value("chunk_right_context", rightContext))
        speakerMode = try pe.value("speaker_feature_mode", speakerMode)
        speakerThreshold = try pe.decodeIfPresent(
            Float.self, forKey: .init("speaker_activity_threshold"))
        speakerScale = try pe.value("spk_kernel_scale", speakerScale)
        speaker.modules.chunkLen = windowLength
        speaker.modules.fifoLen = try pe.value("diar_fifo_len", speaker.modules.fifoLen)
        speaker.modules.spkcacheLen = try pe.value("diar_spkcache_len", speaker.modules.spkcacheLen)
        speaker.modules.spkcacheUpdatePeriod = try pe.value(
            "diar_spkcache_update_period", speaker.modules.spkcacheUpdatePeriod)
        let diarNormalization = try pe.decodeIfPresent(
            String.self, forKey: .init("diar_normalize_type"))
        guard try pe.value("asr_encoder_type", "transformer") == "transformer",
            try pe.value("speaker_feature_config_version", 0) == 1,
            try pe.value("asr_normalize_type", "per_feature") == "per_feature",
            diarNormalization == nil || diarNormalization == "NA"
        else {
            throw STTError.modelNotInitialized("Unsupported Nemotron parallel expert configuration")
        }
        try validate()
    }
    public func validate() throws {
        try text.validate()
        try speech.validate()
        try speaker.encoder.validate()
        try preprocessor.validate()
        try speaker.preprocessor.validate()
        let m = speaker.modules
        guard outputDimension == text.hiddenSize, speech.features == preprocessor.features,
            speaker.encoder.features == preprocessor.features,
            speaker.preprocessor.sampleRate == preprocessor.sampleRate,
            speech.factor == speaker.encoder.factor,
            m.subsamplingFactor == speech.factor, m.fcDModel == speaker.encoder.width,
            m.numSpeakers > 0, m.tfDModel > 0, windowLength > 0,
            m.fifoLen >= 0, m.spkcacheUpdatePeriod > 0, m.spkcacheSilFramesPerSpk >= 0,
            m.spkcacheLen >= (1 + m.spkcacheSilFramesPerSpk) * m.numSpeakers,
            speakerScale.isFinite, padTokenID >= 0, padTokenID < text.vocabularySize,
            !eosTokenIDs.isEmpty, eosTokenIDs.allSatisfy({ $0 >= 0 && $0 < text.vocabularySize }),
            !audioToken.isEmpty,
            (speakerMode == "thresholded" && speakerThreshold != nil
                && (0 ... 1).contains(speakerThreshold!))
                || (speakerMode == "continuous" && speakerThreshold == nil)
        else {
            throw STTError.modelNotInitialized(
                "Invalid Nemotron parallel expert dimensions or speaker features")
        }
    }
}
