import Foundation
import MLXAudioCore

public struct MossMusicAudioConfig: Codable, Sendable {
    public var hiddenSize: Int = 1280
    public var outputDim: Int = 1280
    public var numMelBins: Int = 128
    public var numHiddenLayers: Int = 32
    public var numAttentionHeads: Int = 20
    public var intermediateSize: Int = 5120
    public var downsampleHiddenSize: Int = 480
    public var downsampleRate: Int = 8
    public var maxSourcePositions: Int = 1500
    public var layerNormEps: Float = 1e-5
    public var windowSize: Int = 200
    public var convBatchSize: Int = 64
    public var deepstackLayerIndices: [Int] = [8, 16, 24]
    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "d_model"
        case outputDim = "output_dim"
        case numMelBins = "num_mel_bins"
        case numHiddenLayers = "encoder_layers"
        case numAttentionHeads = "encoder_attention_heads"
        case intermediateSize = "encoder_ffn_dim"
        case downsampleHiddenSize = "downsample_hidden_size"
        case downsampleRate = "downsample_rate"
        case maxSourcePositions = "max_source_positions"
        case layerNormEps = "layer_norm_eps"
        case windowSize = "n_window"
        case convBatchSize = "conv_chunksize"
        case deepstackLayerIndices = "deepstack_encoder_layer_indexes"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 1280
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 20
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        downsampleHiddenSize = try c.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 480
        downsampleRate = try c.decodeIfPresent(Int.self, forKey: .downsampleRate) ?? 8
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 200
        convBatchSize = try c.decodeIfPresent(Int.self, forKey: .convBatchSize) ?? 64
        deepstackLayerIndices =
            try c.decodeIfPresent([Int].self, forKey: .deepstackLayerIndices) ?? [8, 16, 24]
    }
    static func outputLength(_ frames: Int) -> Int { (frames + 7) / 8 }
}

public struct MossMusicConfig: Codable {
    public var modelType = "moss_music"
    public var audioConfig = MossMusicAudioConfig()
    public var languageConfig = Qwen3TextConfig(
        hiddenSize: 4096, intermediateSize: 12288,
        numHiddenLayers: 36, numAttentionHeads: 32, numKeyValueHeads: 8,
        maxPositionEmbeddings: 40960, tieWordEmbeddings: false)
    public var adapterHiddenSize = 8192
    public var deepstackNumInjectLayers: Int? = 3
    public var sampleRate = 16000
    public var audioTokenId = 151654
    public var audioStartId = 151669
    public var audioEndId = 151670
    public var eosTokenId = 151645
    public var enableTimeMarkers = true
    public var stripThinking = true
    public var defaultPrompt = "Please give a detailed musical description of this clip."
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    public init() {}
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
        case languageConfig = "language_config"
        case adapterHiddenSize = "adapter_hidden_size"
        case deepstackNumInjectLayers = "deepstack_num_inject_layers"
        case sampleRate = "sample_rate"
        case audioTokenId = "audio_token_id"
        case audioStartId = "audio_start_id"
        case audioEndId = "audio_end_id"
        case eosTokenId = "eos_token_id"
        case enableTimeMarkers = "enable_time_marker"
        case stripThinking = "strip_thinking"
        case defaultPrompt = "default_prompt"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        audioConfig =
            try c.decodeIfPresent(MossMusicAudioConfig.self, forKey: .audioConfig) ?? audioConfig
        languageConfig =
            try c.decodeIfPresent(Qwen3TextConfig.self, forKey: .languageConfig) ?? languageConfig
        adapterHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .adapterHiddenSize) ?? adapterHiddenSize
        if c.contains(.deepstackNumInjectLayers) {
            deepstackNumInjectLayers = try c.decodeIfPresent(
                Int.self, forKey: .deepstackNumInjectLayers)
        }
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? audioTokenId
        audioStartId = try c.decodeIfPresent(Int.self, forKey: .audioStartId) ?? audioStartId
        audioEndId = try c.decodeIfPresent(Int.self, forKey: .audioEndId) ?? audioEndId
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? eosTokenId
        enableTimeMarkers = try c.decodeIfPresent(Bool.self, forKey: .enableTimeMarkers) ?? true
        stripThinking = try c.decodeIfPresent(Bool.self, forKey: .stripThinking) ?? true
        defaultPrompt = try c.decodeIfPresent(String.self, forKey: .defaultPrompt) ?? defaultPrompt
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }
    var injectionCount: Int {
        min(
            audioConfig.deepstackLayerIndices.count,
            deepstackNumInjectLayers ?? audioConfig.deepstackLayerIndices.count)
    }
    func validate() throws {
        let a = audioConfig
        let t = languageConfig
        guard sampleRate == 16000, a.hiddenSize > 2, a.hiddenSize % 2 == 0,
            a.numMelBins > 0, a.numHiddenLayers > 0, a.numAttentionHeads > 0,
            a.hiddenSize % a.numAttentionHeads == 0, a.intermediateSize > 0,
            a.outputDim > 0, a.downsampleHiddenSize > 0, a.downsampleRate == 8,
            a.maxSourcePositions > 0, a.windowSize > 0, a.windowSize % 4 == 0,
            MossMusicAudioConfig.outputLength(a.windowSize * 2) <= a.maxSourcePositions,
            a.convBatchSize > 0, a.layerNormEps > 0,
            Set(a.deepstackLayerIndices).count == a.deepstackLayerIndices.count,
            a.deepstackLayerIndices.allSatisfy({ (0 ..< a.numHiddenLayers).contains($0) }),
            (deepstackNumInjectLayers ?? 0) >= 0, injectionCount <= t.numHiddenLayers,
            adapterHiddenSize > 0, t.hiddenSize > 0, t.intermediateSize > 0,
            t.numHiddenLayers > 0, t.numAttentionHeads > 0, t.numKeyValueHeads > 0,
            t.numAttentionHeads % t.numKeyValueHeads == 0, t.headDim > 0, t.headDim % 2 == 0,
            t.rmsNormEps > 0, t.maxPositionEmbeddings > 0, !t.attentionBias,
            t.hiddenAct == "silu", t.ropeScaling == nil || t.ropeScaling!.isEmpty,
            [audioTokenId, audioStartId, audioEndId, eosTokenId].allSatisfy({
                (0 ..< t.vocabSize).contains($0)
            }),
            Set([audioTokenId, audioStartId, audioEndId]).count == 3
        else { throw STTError.modelNotInitialized("Unsupported MOSS-Music configuration") }
    }
}
