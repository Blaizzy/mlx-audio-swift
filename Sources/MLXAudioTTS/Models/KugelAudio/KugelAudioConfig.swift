// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLXAudioCore

public struct KugelAudioConfiguration: Decodable, Sendable {
    public var sampleRate = 24000
    public var acousticVAEDim = 64
    public var acousticTokenizer: VibeVoiceAcousticConfig
    public var decoder: Qwen2Configuration
    public var diffusion: VibeVoiceDiffusionConfig
    public var tieWordEmbeddings = false
    public var quantization: BaseConfiguration.PerLayerQuantization?

    public init(decoder: Qwen2Configuration, acousticTokenizer: VibeVoiceAcousticConfig = .init(),
                diffusion: VibeVoiceDiffusionConfig = .init()) {
        self.decoder = decoder
        self.acousticTokenizer = acousticTokenizer
        self.diffusion = diffusion
        acousticVAEDim = acousticTokenizer.vaeDim
    }
    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate", acousticVAEDim = "acoustic_vae_dim", misspelledVAEDim = "acostic_vae_dim"
        case acousticTokenizer = "acoustic_tokenizer_config", decoder = "decoder_config"
        case diffusion = "diffusion_head_config", tieWordEmbeddings = "tie_word_embeddings"
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.decoder = try c.decode(Qwen2Configuration.self, forKey: .decoder)
        acousticTokenizer = try c.decode(VibeVoiceAcousticConfig.self, forKey: .acousticTokenizer)
        diffusion = try c.decode(VibeVoiceDiffusionConfig.self, forKey: .diffusion)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        acousticVAEDim = try c.decodeIfPresent(Int.self, forKey: .acousticVAEDim)
            ?? c.decodeIfPresent(Int.self, forKey: .misspelledVAEDim) ?? 64
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        quantization = try BaseConfiguration(from: decoder).perLayerQuantization
        if let q = quantization {
            quantization = .init(quantization: q.quantization, perLayerQuantization:
                Dictionary(uniqueKeysWithValues: q.perLayerQuantization.map {
                    (canonicalVibeVoiceSynthesisPath($0.key), $0.value)
                }))
        }
    }
    func validate() throws {
        guard !tieWordEmbeddings, !decoder.tieWordEmbeddings else {
            throw AudioGenerationError.invalidInput("KugelAudio requires the published untied LM head")
        }
        guard sampleRate > 0, acousticVAEDim == acousticTokenizer.vaeDim,
              diffusion.latentSize == acousticVAEDim, diffusion.hiddenSize == decoder.hiddenSize,
              decoder.vocabularySize > KugelAudioModel.speechDiffusionID,
              decoder.hiddenLayers > 0, decoder.attentionHeads > 0, decoder.kvHeads > 0,
              decoder.hiddenSize % decoder.attentionHeads == 0,
              decoder.attentionHeads % decoder.kvHeads == 0,
              diffusion.ddpmNumInferenceSteps > 0, diffusion.ddpmNumSteps > 1,
              diffusion.ddpmNumInferenceSteps <= diffusion.ddpmNumSteps,
              ["v_prediction", "epsilon", "sample"].contains(diffusion.predictionType),
              ["cosine", "linear"].contains(diffusion.ddpmBetaSchedule), acousticTokenizer.causal else {
            throw AudioGenerationError.invalidInput("Incompatible KugelAudio synthesis configuration")
        }
    }
}

/// The reference uses greedy speech-token selection; these settings control latent diffusion.
public struct KugelAudioGenerationOptions: Sendable {
    public var guidanceScale: Float
    public var diffusionSteps: Int?
    public init(guidanceScale: Float = 3, diffusionSteps: Int? = nil) {
        self.guidanceScale = guidanceScale
        self.diffusionSteps = diffusionSteps
    }
}
