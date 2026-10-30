// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLXAudioCore

public struct VibeVoiceTTSConfiguration: Decodable, Sendable {
    public var modelType = "vibevoice"
    public var architectures: [String]?
    public var sampleRate = 24000
    public var acousticVAEDim = 64
    public var semanticVAEDim = 128
    public var maxPositionEmbeddings = 65536
    public var acousticTokenizer: VibeVoiceAcousticConfig
    public var semanticTokenizer: VibeVoiceAcousticConfig
    public var decoder: Qwen2Configuration
    public var diffusion: VibeVoiceDiffusionConfig
    public var referenceDistribution = "gaussian"
    public var quantization: BaseConfiguration.PerLayerQuantization?

    public init(decoder: Qwen2Configuration, acousticTokenizer: VibeVoiceAcousticConfig,
                semanticTokenizer: VibeVoiceAcousticConfig, diffusion: VibeVoiceDiffusionConfig) {
        self.decoder = decoder; self.acousticTokenizer = acousticTokenizer
        self.semanticTokenizer = semanticTokenizer; self.diffusion = diffusion
        acousticVAEDim = acousticTokenizer.vaeDim; semanticVAEDim = semanticTokenizer.vaeDim
    }
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", architectures
        case sampleRate = "sample_rate", samplingRate = "sampling_rate"
        case acousticVAEDim = "acoustic_vae_dim", misspelledVAEDim = "acostic_vae_dim"
        case semanticVAEDim = "semantic_vae_dim", acousticTokenizer = "acoustic_tokenizer_config"
        case semanticTokenizer = "semantic_tokenizer_config", decoder = "decoder_config"
        case diffusion = "diffusion_head_config"
    }
    private struct DecoderDetails: Decodable { let max_position_embeddings: Int? }
    private struct AcousticDetails: Decodable { let std_dist_type: String? }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures)
        self.decoder = try c.decode(Qwen2Configuration.self, forKey: .decoder)
        maxPositionEmbeddings = try c.decode(DecoderDetails.self, forKey: .decoder).max_position_embeddings ?? 65536
        acousticTokenizer = try c.decode(VibeVoiceAcousticConfig.self, forKey: .acousticTokenizer)
        referenceDistribution = try c.decode(AcousticDetails.self, forKey: .acousticTokenizer).std_dist_type ?? "gaussian"
        semanticTokenizer = try c.decode(VibeVoiceAcousticConfig.self, forKey: .semanticTokenizer)
        diffusion = try c.decode(VibeVoiceDiffusionConfig.self, forKey: .diffusion)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        acousticVAEDim = try c.decodeIfPresent(Int.self, forKey: .acousticVAEDim) ?? c.decodeIfPresent(Int.self, forKey: .misspelledVAEDim) ?? 64
        semanticVAEDim = try c.decodeIfPresent(Int.self, forKey: .semanticVAEDim) ?? 128
        quantization = try BaseConfiguration(from: decoder).perLayerQuantization
        if let q = quantization {
            quantization = .init(quantization: q.quantization, perLayerQuantization: Dictionary(uniqueKeysWithValues:
                q.perLayerQuantization.map { (canonicalVibeVoiceSynthesisPath($0.key), $0.value) }))
        }
    }
    func validate() throws {
        func validEncoder(_ c: VibeVoiceAcousticConfig) -> Bool {
            c.causal && c.channels == 1 && c.encoderNFilters > 0 && c.encoderRatios.allSatisfy { $0 > 0 }
                && c.encoderDepths.count == c.encoderRatios.count + 1 && c.encoderDepths.allSatisfy { $0 > 0 }
        }
        guard modelType == "vibevoice",
              architectures == nil || architectures!.contains("VibeVoiceForConditionalGeneration"),
              sampleRate > 0, maxPositionEmbeddings > 0, decoder.hiddenLayers > 0,
              decoder.attentionHeads > 0, decoder.kvHeads > 0,
              decoder.hiddenSize % decoder.attentionHeads == 0,
              decoder.attentionHeads % decoder.kvHeads == 0,
              decoder.vocabularySize > VibeVoiceTTSModel.speechDiffusionID,
              acousticVAEDim == acousticTokenizer.vaeDim, semanticVAEDim == semanticTokenizer.vaeDim,
              diffusion.hiddenSize == decoder.hiddenSize, diffusion.latentSize == acousticVAEDim,
              validEncoder(acousticTokenizer), validEncoder(semanticTokenizer),
              acousticTokenizer.encoderRatios.reduce(1, *) == semanticTokenizer.encoderRatios.reduce(1, *),
              acousticTokenizer.hopLength == acousticTokenizer.encoderRatios.reduce(1, *),
              (acousticTokenizer.decoderDepths ?? acousticTokenizer.encoderDepths).count == (acousticTokenizer.decoderRatios ?? acousticTokenizer.encoderRatios).count + 1,
              diffusion.ddpmNumInferenceSteps > 0, diffusion.ddpmNumSteps > 1,
              diffusion.ddpmNumInferenceSteps <= diffusion.ddpmNumSteps,
              ["cosine", "linear"].contains(diffusion.ddpmBetaSchedule),
              ["v_prediction", "epsilon", "sample"].contains(diffusion.predictionType),
              ["gaussian", "fix", "none"].contains(referenceDistribution) else {
            throw AudioGenerationError.invalidInput("Incompatible full VibeVoice synthesis configuration")
        }
    }
}

public struct VibeVoiceTTSGenerationOptions: Sendable {
    public var guidanceScale: Float
    public var diffusionSteps: Int?
    /// Greedy speech controls are the reference default, independently of diffusion noise.
    public var sampleControlTokens: Bool
    public init(guidanceScale: Float = 1.5, diffusionSteps: Int? = nil, sampleControlTokens: Bool = false) {
        self.guidanceScale = guidanceScale; self.diffusionSteps = diffusionSteps
        self.sampleControlTokens = sampleControlTokens
    }
}
