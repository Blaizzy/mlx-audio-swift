import Foundation

// MARK: - EchoDiT Configuration

public struct EchoDiTConfig: Decodable {
    public var latentSize: Int
    public var modelSize: Int
    public var numLayers: Int
    public var numHeads: Int
    public var intermediateSize: Int

    // Text encoder
    public var textVocabSize: Int
    public var textModelSize: Int
    public var textNumLayers: Int
    public var textNumHeads: Int
    public var textIntermediateSize: Int

    // Speaker encoder
    public var speakerPatchSize: Int
    public var speakerModelSize: Int
    public var speakerNumLayers: Int
    public var speakerNumHeads: Int
    public var speakerIntermediateSize: Int

    // AdaLN
    public var timestepEmbedSize: Int
    public var adalnRank: Int

    enum CodingKeys: String, CodingKey {
        case latentSize = "latent_size"
        case modelSize = "model_size"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case intermediateSize = "intermediate_size"
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

    public init(from jsonDecoder: any Swift.Decoder) throws {
        let c = try jsonDecoder.container(keyedBy: CodingKeys.self)
        latentSize = (try? c.decode(Int.self, forKey: CodingKeys.latentSize)) ?? 80
        modelSize = (try? c.decode(Int.self, forKey: CodingKeys.modelSize)) ?? 2048
        numLayers = (try? c.decode(Int.self, forKey: CodingKeys.numLayers)) ?? 24
        numHeads = (try? c.decode(Int.self, forKey: CodingKeys.numHeads)) ?? 16
        intermediateSize = (try? c.decode(Int.self, forKey: CodingKeys.intermediateSize)) ?? 5888
        textVocabSize = (try? c.decode(Int.self, forKey: CodingKeys.textVocabSize)) ?? 256
        textModelSize = (try? c.decode(Int.self, forKey: CodingKeys.textModelSize)) ?? 1280
        textNumLayers = (try? c.decode(Int.self, forKey: CodingKeys.textNumLayers)) ?? 14
        textNumHeads = (try? c.decode(Int.self, forKey: CodingKeys.textNumHeads)) ?? 10
        textIntermediateSize = (try? c.decode(Int.self, forKey: CodingKeys.textIntermediateSize)) ?? 3328
        speakerPatchSize = (try? c.decode(Int.self, forKey: CodingKeys.speakerPatchSize)) ?? 4
        speakerModelSize = (try? c.decode(Int.self, forKey: CodingKeys.speakerModelSize)) ?? 1280
        speakerNumLayers = (try? c.decode(Int.self, forKey: CodingKeys.speakerNumLayers)) ?? 14
        speakerNumHeads = (try? c.decode(Int.self, forKey: CodingKeys.speakerNumHeads)) ?? 10
        speakerIntermediateSize = (try? c.decode(Int.self, forKey: CodingKeys.speakerIntermediateSize)) ?? 3328
        timestepEmbedSize = (try? c.decode(Int.self, forKey: CodingKeys.timestepEmbedSize)) ?? 512
        adalnRank = (try? c.decode(Int.self, forKey: CodingKeys.adalnRank)) ?? 256
    }

    public init() {
        latentSize = 80
        modelSize = 2048
        numLayers = 24
        numHeads = 16
        intermediateSize = 5888
        textVocabSize = 256
        textModelSize = 1280
        textNumLayers = 14
        textNumHeads = 10
        textIntermediateSize = 3328
        speakerPatchSize = 4
        speakerModelSize = 1280
        speakerNumLayers = 14
        speakerNumHeads = 10
        speakerIntermediateSize = 3328
        timestepEmbedSize = 512
        adalnRank = 256
    }
}

// MARK: - Sampler Configuration

public struct EchoSamplerConfig: Decodable {
    public var numSteps: Int
    public var cfgScaleText: Float
    public var cfgScaleSpeaker: Float
    public var cfgMinT: Float
    public var cfgMaxT: Float
    public var sequenceLength: Int
    public var truncationFactor: Float
    public var rescaleK: Float?
    public var rescaleSigma: Float?
    public var speakerKvScale: Float?
    public var speakerKvMaxLayers: Int?
    public var speakerKvMinT: Float?

    enum CodingKeys: String, CodingKey {
        case numSteps = "num_steps"
        case cfgScaleText = "cfg_scale_text"
        case cfgScaleSpeaker = "cfg_scale_speaker"
        case cfgMinT = "cfg_min_t"
        case cfgMaxT = "cfg_max_t"
        case sequenceLength = "sequence_length"
        case truncationFactor = "truncation_factor"
        case rescaleK = "rescale_k"
        case rescaleSigma = "rescale_sigma"
        case speakerKvScale = "speaker_kv_scale"
        case speakerKvMaxLayers = "speaker_kv_max_layers"
        case speakerKvMinT = "speaker_kv_min_t"
    }

    public init(from jsonDecoder: any Swift.Decoder) throws {
        let c = try jsonDecoder.container(keyedBy: CodingKeys.self)
        numSteps = (try? c.decode(Int.self, forKey: CodingKeys.numSteps)) ?? 40
        cfgScaleText = (try? c.decode(Float.self, forKey: CodingKeys.cfgScaleText)) ?? 3.0
        cfgScaleSpeaker = (try? c.decode(Float.self, forKey: CodingKeys.cfgScaleSpeaker)) ?? 8.0
        cfgMinT = (try? c.decode(Float.self, forKey: CodingKeys.cfgMinT)) ?? 0.5
        cfgMaxT = (try? c.decode(Float.self, forKey: CodingKeys.cfgMaxT)) ?? 1.0
        sequenceLength = (try? c.decode(Int.self, forKey: CodingKeys.sequenceLength)) ?? 640
        truncationFactor = (try? c.decode(Float.self, forKey: CodingKeys.truncationFactor)) ?? 0.96
        rescaleK = try? c.decode(Float.self, forKey: CodingKeys.rescaleK)
        rescaleSigma = try? c.decode(Float.self, forKey: CodingKeys.rescaleSigma)
        speakerKvScale = try? c.decode(Float.self, forKey: CodingKeys.speakerKvScale)
        speakerKvMaxLayers = try? c.decode(Int.self, forKey: CodingKeys.speakerKvMaxLayers)
        speakerKvMinT = try? c.decode(Float.self, forKey: CodingKeys.speakerKvMinT)
    }

    public init() {
        numSteps = 40
        cfgScaleText = 3.0
        cfgScaleSpeaker = 8.0
        cfgMinT = 0.5
        cfgMaxT = 1.0
        sequenceLength = 640
        truncationFactor = 0.96
        rescaleK = nil
        rescaleSigma = nil
        speakerKvScale = nil
        speakerKvMaxLayers = nil
        speakerKvMinT = nil
    }
}

// MARK: - Model Configuration

public struct EchoTTSConfig: Decodable {
    public var modelType: String
    public var sampleRate: Int
    public var maxTextLength: Int
    public var maxSpeakerLatentLength: Int
    public var audioDownsampleFactor: Int
    public var normalizeText: Bool
    public var deleteBlockwiseModules: Bool
    public var pcaFilename: String
    public var fishCodecRepo: String
    public var dit: EchoDiTConfig
    public var sampler: EchoSamplerConfig

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
        case dit
        case sampler
    }

    public init(from jsonDecoder: any Swift.Decoder) throws {
        let c = try jsonDecoder.container(keyedBy: CodingKeys.self)
        modelType = (try? c.decode(String.self, forKey: CodingKeys.modelType)) ?? "echo_tts"
        sampleRate = (try? c.decode(Int.self, forKey: CodingKeys.sampleRate)) ?? 44100
        maxTextLength = (try? c.decode(Int.self, forKey: CodingKeys.maxTextLength)) ?? 768
        maxSpeakerLatentLength = (try? c.decode(Int.self, forKey: CodingKeys.maxSpeakerLatentLength)) ?? 6400
        audioDownsampleFactor = (try? c.decode(Int.self, forKey: CodingKeys.audioDownsampleFactor)) ?? 2048
        normalizeText = (try? c.decode(Bool.self, forKey: CodingKeys.normalizeText)) ?? true
        deleteBlockwiseModules = (try? c.decode(Bool.self, forKey: CodingKeys.deleteBlockwiseModules)) ?? false
        pcaFilename = (try? c.decode(String.self, forKey: CodingKeys.pcaFilename)) ?? "pca_state.safetensors"
        fishCodecRepo = (try? c.decode(String.self, forKey: CodingKeys.fishCodecRepo)) ?? "jordand/fish-s1-dac-min"
        dit = (try? c.decode(EchoDiTConfig.self, forKey: CodingKeys.dit)) ?? EchoDiTConfig()
        sampler = (try? c.decode(EchoSamplerConfig.self, forKey: CodingKeys.sampler)) ?? EchoSamplerConfig()
    }

    public init() {
        modelType = "echo_tts"
        sampleRate = 44100
        maxTextLength = 768
        maxSpeakerLatentLength = 6400
        audioDownsampleFactor = 2048
        normalizeText = true
        deleteBlockwiseModules = false
        pcaFilename = "pca_state.safetensors"
        fishCodecRepo = "jordand/fish-s1-dac-min"
        dit = EchoDiTConfig()
        sampler = EchoSamplerConfig()
    }
}
