import Foundation
import MLX
import MLXNN
import HuggingFace
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon

// MARK: - Echo TTS Model

public final class EchoTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public let config: EchoTTSConfig
    let model: EchoDiT
    var fishAE: FishS1DAC?
    var pcaState: EchoPCAState?

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: config.sampler.sequenceLength,
            temperature: 1.0,
            topP: 1.0,
            repetitionPenalty: 1.0,
            repetitionContextSize: 0
        )
    }

    init(config: EchoTTSConfig) {
        self.config = config
        self.model = EchoDiT(config.dit, hasLatentAttention: !config.deleteBlockwiseModules)
    }

    // MARK: - Weight Sanitization

    static func sanitize(_ weights: [String: MLXArray], config: EchoTTSConfig) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        let pcaKeys = Set(["pca_components", "pca_mean", "latent_scale"])
        let blockwiseKeys = ["wk_latent", "wv_latent", "latent_encoder", "latent_norm"]

        for (key, value) in weights {
            // Skip PCA state keys
            if pcaKeys.contains(key) { continue }

            // Skip blockwise module keys if configured
            if config.deleteBlockwiseModules {
                if blockwiseKeys.contains(where: { key.contains($0) }) { continue }
            }

            var newKey = key

            // Remap Sequential naming: cond_module.0.weight -> cond_module.layers.0.weight
            if newKey.contains("cond_module.") && !newKey.contains("cond_module.layers.") {
                newKey = newKey.replacingOccurrences(
                    of: "cond_module.",
                    with: "cond_module.layers."
                )
            }

            // Prefix with model. if not already present
            if !newKey.hasPrefix("model.") {
                newKey = "model." + newKey
            }

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Load from Pretrained

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> EchoTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )

        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(EchoTTSConfig.self, from: configData)

        // Create model
        let model = EchoTTSModel(config: config)

        // Load weights
        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: weightsURL)
        let sanitizedWeights = sanitize(weights, config: config)

        try model.model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights),
            verify: .none
        )
        eval(model.model)

        // Load PCA state
        let pcaURL = modelDir.appendingPathComponent(config.pcaFilename)
        model.pcaState = try echoLoadPCAState(from: pcaURL)

        // Load Fish S1 DAC codec
        model.fishAE = try await FishS1DAC.fromPretrained(config.fishCodecRepo, cache: cache)

        return model
    }

    // MARK: - Generate

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard let fishAE = fishAE else {
            throw AudioGenerationError.modelNotInitialized("Fish S1 DAC codec not loaded")
        }
        guard let pcaState = pcaState else {
            throw AudioGenerationError.modelNotInitialized("PCA state not loaded")
        }

        // Prepare text
        let (textTokens, textMask, _) = echoGetTextInputIdsAndMask(
            [text], maxLength: config.maxTextLength, normalize: config.normalizeText
        )

        // Prepare speaker latent
        let speakerLatent: MLXArray
        let speakerMask: MLXArray

        if let ref = refAudio {
            // ref: expected shape [samples] or [1, samples] or [1, 1, samples]
            var refNCL = ref
            if refNCL.ndim == 1 {
                refNCL = refNCL.reshaped([1, 1, -1])
            } else if refNCL.ndim == 2 {
                refNCL = refNCL.expandedDimensions(axis: 0)
            }
            (speakerLatent, speakerMask) = echoGetSpeakerLatentAndMask(
                fishAE: fishAE, pcaState: pcaState, audio: refNCL, config: config
            )
        } else {
            // Default: zero speaker latent
            let patchSize = config.dit.speakerPatchSize
            let defaultLen = patchSize * 4  // Small default length
            speakerLatent = MLXArray.zeros([1, defaultLen, config.dit.latentSize])
            speakerMask = MLXArray.zeros([1, defaultLen])
        }

        // Generate latents via diffusion sampling
        let latents = echoSampleEulerCFG(
            model: model,
            textTokens: textTokens,
            textMask: textMask,
            speakerLatent: speakerLatent,
            speakerMask: speakerMask,
            config: config
        )

        // Detect silence/flattening
        let flatPoint = echoFindFlatteningPoint(latents)

        // Decode latents to audio
        var audio = echoAEDecode(fishAE: fishAE, pcaState: pcaState, zQ: latents)

        // Crop at flattening point
        audio = echoCropAudioToFlatteningPoint(
            audio: audio, flatteningPoint: flatPoint,
            downsampleFactor: config.audioDownsampleFactor
        )

        // Return as 1D audio: [samples]
        return audio.reshaped([-1])
    }

    // MARK: - Generate Stream

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        nonisolated(unsafe) let refAudioCopy = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    let startTime = Date()

                    let audio = try await self.generate(
                        text: text,
                        voice: voice,
                        refAudio: refAudioCopy,
                        refText: refText,
                        language: language,
                        generationParameters: generationParameters
                    )

                    let generateTime = Date().timeIntervalSince(startTime)
                    let peakMemory = Double(Memory.snapshot().peakMemory) / 1e9

                    let info = AudioGenerationInfo(
                        promptTokenCount: text.utf8.count,
                        generationTokenCount: audio.dim(0),
                        prefillTime: 0,
                        generateTime: generateTime,
                        tokensPerSecond: Double(audio.dim(0)) / max(generateTime, 0.001),
                        peakMemoryUsage: peakMemory
                    )

                    continuation.yield(.info(info))
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
