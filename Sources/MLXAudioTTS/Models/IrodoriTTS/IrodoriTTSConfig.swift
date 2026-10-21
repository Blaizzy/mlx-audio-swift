import Foundation

/// Configuration for Irodori TTS — a flow-matching Japanese TTS (Echo-TTS family)
/// over continuous Semantic-DACVAE latents (48 kHz).
/// Mirrors `mlx_audio/tts/models/irodori_tts/config.py`.
public struct IrodoriDiTConfig: Codable {
    // Flow parameterization: "rf_velocity" (standard RF, 40-step default) or
    // "meanflow" (MeanFlow distillation target, 4-step default). MeanFlow
    // checkpoints add a delta_cond_module conditioned on the interval length (t - r).
    public var flowParameterization: String = "rf_velocity"

    // Audio latent dimensions (v2/v3: 32-dim Semantic-DACVAE, v1: 128-dim DACVAE)
    public var latentDim: Int = 32
    public var latentPatchSize: Int = 1

    // DiT backbone
    public var modelDim: Int = 1280
    public var numLayers: Int = 12
    public var numHeads: Int = 20
    public var mlpRatio: Float = 2.875
    public var textMlpRatio: Float? = 2.6
    public var speakerMlpRatio: Float? = 2.6

    // Text encoder
    public var textVocabSize: Int = 99_574
    public var textTokenizerRepo: String = "llm-jp/llm-jp-3-150m"
    public var textAddBos: Bool = true
    public var textDim: Int = 512
    public var textLayers: Int = 10
    public var textHeads: Int = 8

    // Pretrained text encoder (v4): a shared ModernBERT backbone feeds separate
    // text/caption projectors instead of the two scratch-trained encoders.
    public var textEncoderType: String = "scratch"
    public var textEncoderRevision: String? = nil
    public var textEncoderConfig: IrodoriModernBertConfig? = nil
    public var pretrainedProjectorType: String = "linear"
    public var pretrainedProjectorHiddenRatio: Float = 2.0
    public var pretrainedProjectorDropout: Float = 0.0

    // Speaker (reference latent) encoder
    public var speakerDim: Int = 768
    public var speakerLayers: Int = 8
    public var speakerHeads: Int = 12
    public var speakerPatchSize: Int = 1

    // Conditioning
    public var timestepEmbedDim: Int = 512
    public var adalnRank: Int = 192
    public var normEps: Float = 1e-5

    // Caption (Voice Design) conditioning — can coexist with speaker (v3 VoiceDesign dual mode)
    public var useCaptionCondition: Bool = false
    public var useSpeakerCondition: Bool? = nil
    public var captionVocabSize: Int? = nil
    public var captionTokenizerRepo: String? = nil
    public var captionAddBos: Bool? = nil
    public var captionDim: Int? = nil
    public var captionLayers: Int? = nil
    public var captionHeads: Int? = nil
    public var captionMlpRatio: Float? = nil

    // Duration predictor (v3)
    public var useDurationPredictor: Bool = false
    public var durationAuxDim: Int = 14
    public var durationHiddenDim: Int = 1024
    public var durationLayers: Int = 3
    public var durationDropout: Float = 0.1
    public var durationAttentionHeads: Int = 8
    public var durationArchitecture: String = "token_sum_adarn_zero_no_aux"
    public var durationTokenInitFrames: Float = 9.0
    public var durationSpeakerFusion: String = "adarn_zero"
    public var durationCaptionFusion: String = "adarn_zero"
    public var durationCaptionPooling: String = "masked_mean"

    public init() {}

    // MARK: - Resolved properties (mirror config.py)

    public var usePretrainedTextEncoder: Bool {
        textEncoderType.trimmingCharacters(in: .whitespaces).lowercased() == "pretrained"
    }
    public var useMeanflow: Bool {
        flowParameterization.trimmingCharacters(in: .whitespaces).lowercased() == "meanflow"
    }

    public var useSpeakerConditionResolved: Bool {
        useSpeakerCondition ?? !useCaptionCondition
    }
    public var captionVocabSizeResolved: Int { captionVocabSize ?? textVocabSize }
    public var captionTokenizerRepoResolved: String { captionTokenizerRepo ?? textTokenizerRepo }
    public var captionAddBosResolved: Bool { captionAddBos ?? textAddBos }
    public var captionDimResolved: Int { captionDim ?? textDim }
    public var captionLayersResolved: Int { captionLayers ?? textLayers }
    public var captionHeadsResolved: Int { captionHeads ?? textHeads }
    public var captionMlpRatioResolved: Float { captionMlpRatio ?? textMlpRatioResolved }
    public var patchedLatentDim: Int { latentDim * latentPatchSize }
    public var speakerPatchedLatentDim: Int { patchedLatentDim * speakerPatchSize }
    public var textMlpRatioResolved: Float { textMlpRatio ?? mlpRatio }
    public var speakerMlpRatioResolved: Float { speakerMlpRatio ?? mlpRatio }

    enum CodingKeys: String, CodingKey {
        case flowParameterization = "flow_parameterization"
        case latentDim = "latent_dim"
        case latentPatchSize = "latent_patch_size"
        case modelDim = "model_dim"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case mlpRatio = "mlp_ratio"
        case textMlpRatio = "text_mlp_ratio"
        case speakerMlpRatio = "speaker_mlp_ratio"
        case textVocabSize = "text_vocab_size"
        case textTokenizerRepo = "text_tokenizer_repo"
        case textAddBos = "text_add_bos"
        case textDim = "text_dim"
        case textLayers = "text_layers"
        case textHeads = "text_heads"
        case textEncoderType = "text_encoder_type"
        case textEncoderRevision = "text_encoder_revision"
        case textEncoderConfig = "text_encoder_config"
        case pretrainedProjectorType = "pretrained_projector_type"
        case pretrainedProjectorHiddenRatio = "pretrained_projector_hidden_ratio"
        case pretrainedProjectorDropout = "pretrained_projector_dropout"
        case speakerDim = "speaker_dim"
        case speakerLayers = "speaker_layers"
        case speakerHeads = "speaker_heads"
        case speakerPatchSize = "speaker_patch_size"
        case timestepEmbedDim = "timestep_embed_dim"
        case adalnRank = "adaln_rank"
        case normEps = "norm_eps"
        case useCaptionCondition = "use_caption_condition"
        case useSpeakerCondition = "use_speaker_condition"
        case captionVocabSize = "caption_vocab_size"
        case captionTokenizerRepo = "caption_tokenizer_repo"
        case captionAddBos = "caption_add_bos"
        case captionDim = "caption_dim"
        case captionLayers = "caption_layers"
        case captionHeads = "caption_heads"
        case captionMlpRatio = "caption_mlp_ratio"
        case useDurationPredictor = "use_duration_predictor"
        case durationAuxDim = "duration_aux_dim"
        case durationHiddenDim = "duration_hidden_dim"
        case durationLayers = "duration_layers"
        case durationDropout = "duration_dropout"
        case durationAttentionHeads = "duration_attention_heads"
        case durationArchitecture = "duration_architecture"
        case durationTokenInitFrames = "duration_token_init_frames"
        case durationSpeakerFusion = "duration_speaker_fusion"
        case durationCaptionFusion = "duration_caption_fusion"
        case durationCaptionPooling = "duration_caption_pooling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        flowParameterization = try c.decodeIfPresent(String.self, forKey: .flowParameterization) ?? "rf_velocity"
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 32
        latentPatchSize = try c.decodeIfPresent(Int.self, forKey: .latentPatchSize) ?? 1
        modelDim = try c.decodeIfPresent(Int.self, forKey: .modelDim) ?? 1280
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 20
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 2.875
        textMlpRatio = try c.decodeIfPresent(Float.self, forKey: .textMlpRatio) ?? 2.6
        speakerMlpRatio = try c.decodeIfPresent(Float.self, forKey: .speakerMlpRatio) ?? 2.6
        textVocabSize = try c.decodeIfPresent(Int.self, forKey: .textVocabSize) ?? 99_574
        textTokenizerRepo = try c.decodeIfPresent(String.self, forKey: .textTokenizerRepo) ?? "llm-jp/llm-jp-3-150m"
        textAddBos = try c.decodeIfPresent(Bool.self, forKey: .textAddBos) ?? true
        textDim = try c.decodeIfPresent(Int.self, forKey: .textDim) ?? 512
        textLayers = try c.decodeIfPresent(Int.self, forKey: .textLayers) ?? 10
        textHeads = try c.decodeIfPresent(Int.self, forKey: .textHeads) ?? 8
        textEncoderType = try c.decodeIfPresent(String.self, forKey: .textEncoderType) ?? "scratch"
        textEncoderRevision = try c.decodeIfPresent(String.self, forKey: .textEncoderRevision)
        textEncoderConfig = try c.decodeIfPresent(IrodoriModernBertConfig.self, forKey: .textEncoderConfig)
        pretrainedProjectorType = try c.decodeIfPresent(String.self, forKey: .pretrainedProjectorType) ?? "linear"
        pretrainedProjectorHiddenRatio = try c.decodeIfPresent(Float.self, forKey: .pretrainedProjectorHiddenRatio) ?? 2.0
        pretrainedProjectorDropout = try c.decodeIfPresent(Float.self, forKey: .pretrainedProjectorDropout) ?? 0.0
        speakerDim = try c.decodeIfPresent(Int.self, forKey: .speakerDim) ?? 768
        speakerLayers = try c.decodeIfPresent(Int.self, forKey: .speakerLayers) ?? 8
        speakerHeads = try c.decodeIfPresent(Int.self, forKey: .speakerHeads) ?? 12
        speakerPatchSize = try c.decodeIfPresent(Int.self, forKey: .speakerPatchSize) ?? 1
        timestepEmbedDim = try c.decodeIfPresent(Int.self, forKey: .timestepEmbedDim) ?? 512
        adalnRank = try c.decodeIfPresent(Int.self, forKey: .adalnRank) ?? 192
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        useCaptionCondition = try c.decodeIfPresent(Bool.self, forKey: .useCaptionCondition) ?? false
        useSpeakerCondition = try c.decodeIfPresent(Bool.self, forKey: .useSpeakerCondition)
        captionVocabSize = try c.decodeIfPresent(Int.self, forKey: .captionVocabSize)
        captionTokenizerRepo = try c.decodeIfPresent(String.self, forKey: .captionTokenizerRepo)
        captionAddBos = try c.decodeIfPresent(Bool.self, forKey: .captionAddBos)
        captionDim = try c.decodeIfPresent(Int.self, forKey: .captionDim)
        captionLayers = try c.decodeIfPresent(Int.self, forKey: .captionLayers)
        captionHeads = try c.decodeIfPresent(Int.self, forKey: .captionHeads)
        captionMlpRatio = try c.decodeIfPresent(Float.self, forKey: .captionMlpRatio)
        useDurationPredictor = try c.decodeIfPresent(Bool.self, forKey: .useDurationPredictor) ?? false
        durationAuxDim = try c.decodeIfPresent(Int.self, forKey: .durationAuxDim) ?? 14
        durationHiddenDim = try c.decodeIfPresent(Int.self, forKey: .durationHiddenDim) ?? 1024
        durationLayers = try c.decodeIfPresent(Int.self, forKey: .durationLayers) ?? 3
        durationDropout = try c.decodeIfPresent(Float.self, forKey: .durationDropout) ?? 0.1
        durationAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .durationAttentionHeads) ?? 8
        durationArchitecture = try c.decodeIfPresent(String.self, forKey: .durationArchitecture) ?? "token_sum_adarn_zero_no_aux"
        durationTokenInitFrames = try c.decodeIfPresent(Float.self, forKey: .durationTokenInitFrames) ?? 9.0
        durationSpeakerFusion = try c.decodeIfPresent(String.self, forKey: .durationSpeakerFusion) ?? "adarn_zero"
        durationCaptionFusion = try c.decodeIfPresent(String.self, forKey: .durationCaptionFusion) ?? "adarn_zero"
        durationCaptionPooling = try c.decodeIfPresent(String.self, forKey: .durationCaptionPooling) ?? "masked_mean"
    }
}

public struct IrodoriSamplerConfig: Codable {
    public var numSteps: Int = 40
    public var cfgScaleText: Float = 3.0
    public var cfgScaleSpeaker: Float = 5.0
    public var cfgScaleCaption: Float = 3.0
    /// "independent" (Python default, ~3x memory) or "alternating" (mobile-friendly).
    /// On-device default is overridden to "alternating" by IrodoriTTSModel.
    public var cfgGuidanceMode: String = "independent"
    public var cfgMinT: Float = 0.5
    public var cfgMaxT: Float = 1.0
    public var truncationFactor: Float? = nil
    public var rescaleK: Float? = nil
    public var rescaleSigma: Float? = nil
    public var contextKvCache: Bool = true
    public var speakerKvScale: Float? = nil
    public var speakerKvMinT: Float? = 0.9
    public var speakerKvMaxLayers: Int? = nil
    /// Python default 750 needs ~24 GB; on-device default is overridden to 300 (~2 GB, ~12 s).
    public var sequenceLength: Int = 750
    // Sway Sampling (v3)
    public var tScheduleMode: String = "linear"
    public var swayCoeff: Float = -1.0
    // Duration prediction (v3)
    public var durationScale: Float = 1.0
    public var minSeconds: Float = 0.5
    public var maxSeconds: Float = 30.0

    public init() {}

    enum CodingKeys: String, CodingKey {
        case numSteps = "num_steps"
        case cfgScaleText = "cfg_scale_text"
        case cfgScaleSpeaker = "cfg_scale_speaker"
        case cfgScaleCaption = "cfg_scale_caption"
        case cfgGuidanceMode = "cfg_guidance_mode"
        case cfgMinT = "cfg_min_t"
        case cfgMaxT = "cfg_max_t"
        case truncationFactor = "truncation_factor"
        case rescaleK = "rescale_k"
        case rescaleSigma = "rescale_sigma"
        case contextKvCache = "context_kv_cache"
        case speakerKvScale = "speaker_kv_scale"
        case speakerKvMinT = "speaker_kv_min_t"
        case speakerKvMaxLayers = "speaker_kv_max_layers"
        case sequenceLength = "sequence_length"
        case tScheduleMode = "t_schedule_mode"
        case swayCoeff = "sway_coeff"
        case durationScale = "duration_scale"
        case minSeconds = "min_seconds"
        case maxSeconds = "max_seconds"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numSteps = try c.decodeIfPresent(Int.self, forKey: .numSteps) ?? 40
        cfgScaleText = try c.decodeIfPresent(Float.self, forKey: .cfgScaleText) ?? 3.0
        cfgScaleSpeaker = try c.decodeIfPresent(Float.self, forKey: .cfgScaleSpeaker) ?? 5.0
        cfgScaleCaption = try c.decodeIfPresent(Float.self, forKey: .cfgScaleCaption) ?? 3.0
        cfgGuidanceMode = try c.decodeIfPresent(String.self, forKey: .cfgGuidanceMode) ?? "independent"
        cfgMinT = try c.decodeIfPresent(Float.self, forKey: .cfgMinT) ?? 0.5
        cfgMaxT = try c.decodeIfPresent(Float.self, forKey: .cfgMaxT) ?? 1.0
        truncationFactor = try c.decodeIfPresent(Float.self, forKey: .truncationFactor)
        rescaleK = try c.decodeIfPresent(Float.self, forKey: .rescaleK)
        rescaleSigma = try c.decodeIfPresent(Float.self, forKey: .rescaleSigma)
        contextKvCache = try c.decodeIfPresent(Bool.self, forKey: .contextKvCache) ?? true
        speakerKvScale = try c.decodeIfPresent(Float.self, forKey: .speakerKvScale)
        speakerKvMinT = try c.decodeIfPresent(Float.self, forKey: .speakerKvMinT) ?? 0.9
        speakerKvMaxLayers = try c.decodeIfPresent(Int.self, forKey: .speakerKvMaxLayers)
        sequenceLength = try c.decodeIfPresent(Int.self, forKey: .sequenceLength) ?? 750
        tScheduleMode = try c.decodeIfPresent(String.self, forKey: .tScheduleMode) ?? "linear"
        swayCoeff = try c.decodeIfPresent(Float.self, forKey: .swayCoeff) ?? -1.0
        durationScale = try c.decodeIfPresent(Float.self, forKey: .durationScale) ?? 1.0
        minSeconds = try c.decodeIfPresent(Float.self, forKey: .minSeconds) ?? 0.5
        maxSeconds = try c.decodeIfPresent(Float.self, forKey: .maxSeconds) ?? 30.0
    }
}

public struct IrodoriTTSConfig: Codable {
    public var modelType: String = "irodori_tts"
    public var sampleRate: Int = 48_000
    public var maxTextLength: Int = 256
    public var maxCaptionLength: Int = 512
    public var maxSpeakerLatentLength: Int = 6_400
    /// Cap on total reference audio duration. v4 checkpoints set 120 s (trained with up
    /// to 120 s of reference). When absent (v1–v3) only `maxSpeakerLatentLength` applies,
    /// as before; upstream Python would default to 30 s there.
    public var refMaxSeconds: Float? = nil
    /// DACVAE hop_length = 2*8*10*12 = 1920 (48 kHz)
    public var audioDownsampleFactor: Int = 1920
    public var dacvaeRepo: String = "Aratako/Semantic-DACVAE-Japanese-32dim"
    public var modelPath: String? = nil
    public var dit: IrodoriDiTConfig = IrodoriDiTConfig()
    public var sampler: IrodoriSamplerConfig = IrodoriSamplerConfig()

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case maxTextLength = "max_text_length"
        case maxCaptionLength = "max_caption_length"
        case maxSpeakerLatentLength = "max_speaker_latent_length"
        case refMaxSeconds = "ref_max_seconds"
        case audioDownsampleFactor = "audio_downsample_factor"
        case dacvaeRepo = "dacvae_repo"
        case modelPath = "model_path"
        case dit
        case sampler
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "irodori_tts"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 48_000
        maxTextLength = try c.decodeIfPresent(Int.self, forKey: .maxTextLength) ?? 256
        maxCaptionLength = try c.decodeIfPresent(Int.self, forKey: .maxCaptionLength) ?? 512
        maxSpeakerLatentLength = try c.decodeIfPresent(Int.self, forKey: .maxSpeakerLatentLength) ?? 6_400
        refMaxSeconds = try c.decodeIfPresent(Float.self, forKey: .refMaxSeconds)
        audioDownsampleFactor = try c.decodeIfPresent(Int.self, forKey: .audioDownsampleFactor) ?? 1920
        dacvaeRepo = try c.decodeIfPresent(String.self, forKey: .dacvaeRepo) ?? "Aratako/Semantic-DACVAE-Japanese-32dim"
        modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath)
        dit = try c.decodeIfPresent(IrodoriDiTConfig.self, forKey: .dit) ?? IrodoriDiTConfig()
        sampler = try c.decodeIfPresent(IrodoriSamplerConfig.self, forKey: .sampler) ?? IrodoriSamplerConfig()
    }
}

/// Configuration of the pretrained ModernBERT text encoder embedded in v4
/// checkpoints (`dit.text_encoder_config`, a HuggingFace `ModernBertConfig`).
/// Mirrors `mlx_audio/tts/models/irodori_tts/modernbert.py` `ModernBertConfig`.
public struct IrodoriModernBertConfig: Codable {
    public var modelType: String = "modernbert"
    public var vocabSize: Int = 102_400
    public var hiddenSize: Int = 768
    public var intermediateSize: Int = 3072
    public var numHiddenLayers: Int = 25
    public var numAttentionHeads: Int = 12
    public var hiddenActivation: String = "gelu"
    public var normEps: Float = 1e-5
    public var normBias: Bool = false
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false
    public var globalAttnEveryNLayers: Int = 3
    public var localAttention: Int = 128
    public var globalRopeTheta: Float = 160_000
    public var localRopeTheta: Float = 10_000
    public var maxPositionEmbeddings: Int = 8192
    public var padTokenId: Int = 3
    public var bosTokenId: Int = 1
    public var eosTokenId: Int = 2

    public init() {}

    public var headDim: Int { hiddenSize / numAttentionHeads }

    public func isGlobalLayer(_ layerIdx: Int) -> Bool {
        globalAttnEveryNLayers == 0 || layerIdx % globalAttnEveryNLayers == 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenActivation = "hidden_activation"
        case normEps = "norm_eps"
        case normBias = "norm_bias"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case globalAttnEveryNLayers = "global_attn_every_n_layers"
        case localAttention = "local_attention"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
    }

    private struct RopeParameters: Codable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "modernbert"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 102_400
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 25
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        hiddenActivation = try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu"
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        normBias = try c.decodeIfPresent(Bool.self, forKey: .normBias) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        globalAttnEveryNLayers = try c.decodeIfPresent(Int.self, forKey: .globalAttnEveryNLayers) ?? 3
        localAttention = try c.decodeIfPresent(Int.self, forKey: .localAttention) ?? 128
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 3
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        // transformers >= 5 nests the per-layer-type RoPE settings; earlier
        // releases exposed flat global_/local_rope_theta keys.
        let rope = try? c.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters)
        globalRopeTheta = try c.decodeIfPresent(Float.self, forKey: .globalRopeTheta)
            ?? rope?["full_attention"]?.ropeTheta ?? 160_000
        localRopeTheta = try c.decodeIfPresent(Float.self, forKey: .localRopeTheta)
            ?? rope?["sliding_attention"]?.ropeTheta ?? 10_000
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(hiddenActivation, forKey: .hiddenActivation)
        try c.encode(normEps, forKey: .normEps)
        try c.encode(normBias, forKey: .normBias)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(mlpBias, forKey: .mlpBias)
        try c.encode(globalAttnEveryNLayers, forKey: .globalAttnEveryNLayers)
        try c.encode(localAttention, forKey: .localAttention)
        try c.encode(globalRopeTheta, forKey: .globalRopeTheta)
        try c.encode(localRopeTheta, forKey: .localRopeTheta)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(padTokenId, forKey: .padTokenId)
        try c.encode(bosTokenId, forKey: .bosTokenId)
        try c.encode(eosTokenId, forKey: .eosTokenId)
    }
}
