import Foundation
import MLXAudioCore

public enum MingTinyError: Error, LocalizedError {
    case invalidConfiguration(String), invalidInput(String), missingComponent(String)
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message), .missingComponent(let message): message
        }
    }
}

public struct MingTinySpeechConfiguration: Decodable, Sendable {
    public var latentDim = 64
    public var patchSize = 1
    public var historySize = 1
    public var speakerEmbeddingDim = 192
    public var audioPatchTokenId = 157164
    public var audioStartTokenId = 157165
    public var audioEndTokenId = 157166
    public var speakerStartTokenId = 157175
    public var speakerEndTokenId = 157176
    public var promptEosTokenId = 156895
    public var minimumDecodeSteps = 13
    public var maximumDecodeSteps = 250
    public var stopThreshold: Float = 0.5
    public var promptFormat = "legacy_antmmf"
    public var samplingContract = "native_fp32"
    public var decodeAlignment = "old_shift"
    public var withRML = true
    enum CodingKeys: String, CodingKey {
        case latentDim = "latent_dim", patchSize = "patch_size", historySize = "history_size"
        case speakerEmbeddingDim = "speaker_embedding_dim", audioPatchTokenId = "audio_patch_token_id"
        case audioStartTokenId = "audio_start_token_id", audioEndTokenId = "audio_end_token_id"
        case speakerStartTokenId = "speaker_start_token_id", speakerEndTokenId = "speaker_end_token_id"
        case promptEosTokenId = "prompt_eos_token_id", minimumDecodeSteps = "minimum_decode_steps"
        case maximumDecodeSteps = "maximum_decode_steps", stopThreshold = "stop_threshold"
        case promptFormat = "prompt_format", samplingContract = "sampling_contract"
        case decodeAlignment = "decode_alignment", withRML = "with_rml"
    }
    public init() {}
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 1
        historySize = try c.decodeIfPresent(Int.self, forKey: .historySize) ?? 1
        speakerEmbeddingDim = try c.decodeIfPresent(Int.self, forKey: .speakerEmbeddingDim) ?? 192
        audioPatchTokenId = try c.decodeIfPresent(Int.self, forKey: .audioPatchTokenId) ?? 157164
        audioStartTokenId = try c.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 157165
        audioEndTokenId = try c.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? 157166
        speakerStartTokenId = try c.decodeIfPresent(Int.self, forKey: .speakerStartTokenId) ?? 157175
        speakerEndTokenId = try c.decodeIfPresent(Int.self, forKey: .speakerEndTokenId) ?? 157176
        promptEosTokenId = try c.decodeIfPresent(Int.self, forKey: .promptEosTokenId) ?? 156895
        minimumDecodeSteps = try c.decodeIfPresent(Int.self, forKey: .minimumDecodeSteps) ?? 13
        maximumDecodeSteps = try c.decodeIfPresent(Int.self, forKey: .maximumDecodeSteps) ?? 250
        stopThreshold = try c.decodeIfPresent(Float.self, forKey: .stopThreshold) ?? 0.5
        promptFormat = try c.decodeIfPresent(String.self, forKey: .promptFormat) ?? "legacy_antmmf"
        samplingContract = try c.decodeIfPresent(String.self, forKey: .samplingContract) ?? "native_fp32"
        decodeAlignment = try c.decodeIfPresent(String.self, forKey: .decodeAlignment) ?? "old_shift"
        withRML = try c.decodeIfPresent(Bool.self, forKey: .withRML) ?? true
    }
}

public struct MingTinyConfiguration: Decodable, Sendable {
    public var text: MingTinyTextConfiguration
    public var audio: MingAudioVAEConfiguration
    public var speech: MingTinySpeechConfiguration
    public var flow: MingTransformerConfiguration
    public var aggregator: MingTransformerConfiguration
    public var quantization: BaseConfiguration.Quantization?
    enum CodingKeys: String, CodingKey {
        case text = "llm_config", audio = "audio_config", speech = "tts_config"
        case flow = "flow_config", aggregator = "aggregator_config", quantization
    }
    public init(text: MingTinyTextConfiguration, audio: MingAudioVAEConfiguration,
                speech: MingTinySpeechConfiguration = .init(), flow: MingTransformerConfiguration,
                aggregator: MingTransformerConfiguration) {
        self.text = text; self.audio = audio; self.speech = speech
        self.flow = flow; self.aggregator = aggregator
    }
    func validate() throws {
        let t = text, s = speech
        guard t.hiddenSize > 0, t.numHiddenLayers > 0, t.vocabSize > 0,
              t.layerGroupSize > 0, t.numAttentionHeads > 0, t.headDim > 0,
              t.qkRopeHeadDim > 0, t.qkRopeHeadDim % 2 == 0,
              t.nGroup > 0, t.numExperts > 0, t.numExperts % t.nGroup == 0,
              (1...t.nGroup).contains(t.topKGroup),
              (1...t.numExperts).contains(t.numExpertsPerToken),
              t.numExpertsPerToken <= t.topKGroup * (t.numExperts / t.nGroup),
              (0...t.numHiddenLayers).contains(t.firstKDenseReplace),
              t.intermediateSize > 0, t.moeIntermediateSize > 0, t.moeSharedExpertIntermediateSize > 0,
              t.numSharedExperts >= 0, t.maxPositionEmbeddings > 0,
              t.shortConvKernelSize > 0, t.kvLoraRank > 0,
              t.qLoraRank == nil || t.qLoraRank! > 0,
              [nil, "head_wise", "element_wise"].contains(t.attentionGate),
              s.latentDim > 0, s.patchSize > 0, s.historySize >= s.patchSize,
              s.speakerEmbeddingDim == 192, s.withRML,
              ["legacy_antmmf", "base_v1"].contains(s.promptFormat),
              s.decodeAlignment == "old_shift", s.samplingContract == "native_fp32",
              flow.hiddenSize > 0, flow.numHeads > 0, flow.hiddenSize % flow.numHeads == 0,
              aggregator.hiddenSize > 0, aggregator.numHeads > 0,
              aggregator.hiddenSize % aggregator.numHeads == 0,
              audio.sampleRate > 0, audio.patchSize > 0, audio.encoder.hopSize > 0,
              audio.encoder.inputDim > 0, audio.decoder.outputDim > 0,
              audio.encoder.latentDim == s.latentDim, audio.decoder.latentDim == s.latentDim
        else { throw MingTinyError.invalidConfiguration("Unsupported Ming Tiny configuration or prompt/sampling contract") }
    }
}
