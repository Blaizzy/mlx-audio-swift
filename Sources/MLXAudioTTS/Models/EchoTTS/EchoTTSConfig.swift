import Foundation

public struct EchoDiTConfig: Codable {
    public let latentSize: Int
    public let modelSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let intermediateSize: Int
    public let normEps: Float
    public let textVocabSize: Int
    public let textModelSize: Int
    public let textNumLayers: Int
    public let textNumHeads: Int
    public let textIntermediateSize: Int
    public let speakerPatchSize: Int
    public let speakerModelSize: Int
    public let speakerNumLayers: Int
    public let speakerNumHeads: Int
    public let speakerIntermediateSize: Int
    public let timestepEmbedSize: Int
    public let adalnRank: Int

    public init(
        latentSize: Int = 80,
        modelSize: Int = 2048,
        numLayers: Int = 24,
        numHeads: Int = 16,
        intermediateSize: Int = 5888,
        normEps: Float = 1e-5,
        textVocabSize: Int = 256,
        textModelSize: Int = 1280,
        textNumLayers: Int = 14,
        textNumHeads: Int = 10,
        textIntermediateSize: Int = 3328,
        speakerPatchSize: Int = 4,
        speakerModelSize: Int = 1280,
        speakerNumLayers: Int = 14,
        speakerNumHeads: Int = 10,
        speakerIntermediateSize: Int = 3328,
        timestepEmbedSize: Int = 512,
        adalnRank: Int = 256
    ) {
        self.latentSize = latentSize
        self.modelSize = modelSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.intermediateSize = intermediateSize
        self.normEps = normEps
        self.textVocabSize = textVocabSize
        self.textModelSize = textModelSize
        self.textNumLayers = textNumLayers
        self.textNumHeads = textNumHeads
        self.textIntermediateSize = textIntermediateSize
        self.speakerPatchSize = speakerPatchSize
        self.speakerModelSize = speakerModelSize
        self.speakerNumLayers = speakerNumLayers
        self.speakerNumHeads = speakerNumHeads
        self.speakerIntermediateSize = speakerIntermediateSize
        self.timestepEmbedSize = timestepEmbedSize
        self.adalnRank = adalnRank
    }

    enum CodingKeys: String, CodingKey {
        case latentSize = "latent_size"
        case modelSize = "model_size"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case textVocabSize = "text_vocab_size"
        case textModelSize = "text_model_size"
        case textNumLayers = "text_num_layers"
        case textNumHeads = "text_num_heads"
        case textIntermediateSize = "text_intermediate_size"
        case speakerPatchSize = "speaker_patch_size"
        case speakerModelSize = "speaker_model_size"
        case speakerNumLayers = "speaker_num_layers"
        case speakerNumHeads = "speaker_num_heads"
        case speakerIntermediateSize = "speaker_intermediate_size"
        case timestepEmbedSize = "timestep_embed_size"
        case adalnRank = "adaln_rank"
    }
}

public struct EchoTTSSamplerConfig: Codable {
    public let numSteps: Int
    public let cfgScaleText: Float
    public let cfgScaleSpeaker: Float
    public let cfgMinT: Float
    public let cfgMaxT: Float
    public let truncationFactor: Float?
    public let rescaleK: Float?
    public let rescaleSigma: Float?
    public let speakerKVScale: Float?
    public let speakerKVMaxLayers: Int?
    public let speakerKVMinT: Float?
    public let sequenceLength: Int

    public init(
        numSteps: Int = 40,
        cfgScaleText: Float = 3.0,
        cfgScaleSpeaker: Float = 8.0,
        cfgMinT: Float = 0.5,
        cfgMaxT: Float = 1.0,
        truncationFactor: Float? = nil,
        rescaleK: Float? = nil,
        rescaleSigma: Float? = nil,
        speakerKVScale: Float? = nil,
        speakerKVMaxLayers: Int? = nil,
        speakerKVMinT: Float? = nil,
        sequenceLength: Int = 640
    ) {
        self.numSteps = numSteps
        self.cfgScaleText = cfgScaleText
        self.cfgScaleSpeaker = cfgScaleSpeaker
        self.cfgMinT = cfgMinT
        self.cfgMaxT = cfgMaxT
        self.truncationFactor = truncationFactor
        self.rescaleK = rescaleK
        self.rescaleSigma = rescaleSigma
        self.speakerKVScale = speakerKVScale
        self.speakerKVMaxLayers = speakerKVMaxLayers
        self.speakerKVMinT = speakerKVMinT
        self.sequenceLength = sequenceLength
    }

    enum CodingKeys: String, CodingKey {
        case numSteps = "num_steps"
        case cfgScaleText = "cfg_scale_text"
        case cfgScaleSpeaker = "cfg_scale_speaker"
        case cfgMinT = "cfg_min_t"
        case cfgMaxT = "cfg_max_t"
        case truncationFactor = "truncation_factor"
        case rescaleK = "rescale_k"
        case rescaleSigma = "rescale_sigma"
        case speakerKVScale = "speaker_kv_scale"
        case speakerKVMaxLayers = "speaker_kv_max_layers"
        case speakerKVMinT = "speaker_kv_min_t"
        case sequenceLength = "sequence_length"
    }
}

public struct EchoTTSConfig: Codable {
    public let modelType: String
    public let sampleRate: Int
    public let maxTextLength: Int
    public let maxSpeakerLatentLength: Int
    public let audioDownsampleFactor: Int
    public let normalizeText: Bool
    public let deleteBlockwiseModules: Bool
    public let pcaFilename: String
    public let fishCodecRepo: String
    public let modelPath: String?
    public let dit: EchoDiTConfig
    public let sampler: EchoTTSSamplerConfig

    public init(
        modelType: String = "echo_tts",
        sampleRate: Int = 44_100,
        maxTextLength: Int = 768,
        maxSpeakerLatentLength: Int = 6_400,
        audioDownsampleFactor: Int = 2_048,
        normalizeText: Bool = true,
        deleteBlockwiseModules: Bool = false,
        pcaFilename: String = "pca_state.safetensors",
        fishCodecRepo: String = "jordand/fish-s1-dac-min",
        modelPath: String? = nil,
        dit: EchoDiTConfig = EchoDiTConfig(),
        sampler: EchoTTSSamplerConfig = EchoTTSSamplerConfig()
    ) {
        self.modelType = modelType
        self.sampleRate = sampleRate
        self.maxTextLength = maxTextLength
        self.maxSpeakerLatentLength = maxSpeakerLatentLength
        self.audioDownsampleFactor = audioDownsampleFactor
        self.normalizeText = normalizeText
        self.deleteBlockwiseModules = deleteBlockwiseModules
        self.pcaFilename = pcaFilename
        self.fishCodecRepo = fishCodecRepo
        self.modelPath = modelPath
        self.dit = dit
        self.sampler = sampler
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case maxTextLength = "max_text_length"
        case maxSpeakerLatentLength = "max_speaker_latent_length"
        case audioDownsampleFactor = "audio_downsample_factor"
        case normalizeText = "normalize_text"
        case deleteBlockwiseModules = "delete_blockwise_modules"
        case pcaFilename = "pca_filename"
        case fishCodecRepo = "fish_codec_repo"
        case modelPath = "model_path"
        case dit
        case sampler
    }
}
