import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public final class EchoTTSModel: Module, @unchecked Sendable {
    public let config: EchoTTSConfig
    public let sampleRate: Int
    public let defaultGenerationParameters: GenerateParameters

    @ModuleInfo(key: "model") var model: EchoDiT

    var fishAE: EchoTTSAudioCodec?
    var pcaState: EchoTTSPCAState?

    init(
        config: EchoTTSConfig,
        fishAE: EchoTTSAudioCodec? = nil,
        pcaState: EchoTTSPCAState? = nil
    ) {
        self.config = config
        self.sampleRate = config.sampleRate
        self.defaultGenerationParameters = GenerateParameters(
            maxTokens: config.sampler.sequenceLength,
            temperature: 0,
            topP: 1
        )
        self._model = ModuleInfo(wrappedValue: EchoDiT(
            latentSize: config.dit.latentSize,
            modelSize: config.dit.modelSize,
            numLayers: config.dit.numLayers,
            numHeads: config.dit.numHeads,
            intermediateSize: config.dit.intermediateSize,
            normEps: config.dit.normEps,
            textVocabSize: config.dit.textVocabSize,
            textModelSize: config.dit.textModelSize,
            textNumLayers: config.dit.textNumLayers,
            textNumHeads: config.dit.textNumHeads,
            textIntermediateSize: config.dit.textIntermediateSize,
            speakerPatchSize: config.dit.speakerPatchSize,
            speakerModelSize: config.dit.speakerModelSize,
            speakerNumLayers: config.dit.speakerNumLayers,
            speakerNumHeads: config.dit.speakerNumHeads,
            speakerIntermediateSize: config.dit.speakerIntermediateSize,
            timestepEmbedSize: config.dit.timestepEmbedSize,
            adalnRank: config.dit.adalnRank,
            enableBlockwiseModules: !config.deleteBlockwiseModules
        ))
        self.fishAE = fishAE
        self.pcaState = pcaState
    }

    func callAsFunction(
        x: MLXArray,
        t: MLXArray,
        textMask: MLXArray,
        speakerMask: MLXArray,
        kvCacheText: [EchoTTSKVCache],
        kvCacheSpeaker: [EchoTTSKVCache],
        startPos: Int? = nil,
        kvCacheLatent: [EchoTTSKVCache]? = nil
    ) -> MLXArray {
        model(
            x: x,
            t: t,
            textMask: textMask,
            speakerMask: speakerMask,
            kvCacheText: kvCacheText,
            kvCacheSpeaker: kvCacheSpeaker,
            startPos: startPos,
            kvCacheLatent: kvCacheLatent
        )
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let skipped = Set(["pca_components", "pca_mean", "latent_scale"])

        func isBlockwiseKey(_ key: String) -> Bool {
            key.hasPrefix("latent_encoder.")
                || key.hasPrefix("latent_norm.")
                || key.contains(".wk_latent.")
                || key.contains(".wv_latent.")
        }

        func normalizedComponent(_ component: String) -> String {
            switch component {
            case "attention_adaln":
                return "attentionAdaLN"
            case "mlp_adaln":
                return "mlpAdaLN"
            default:
                guard component.contains("_") else { return component }
                let parts = component.split(separator: "_")
                guard let head = parts.first else { return component }
                return String(head) + parts.dropFirst().map {
                    $0.prefix(1).uppercased() + $0.dropFirst()
                }.joined()
            }
        }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            if skipped.contains(rawKey) {
                continue
            }

            var key = rawKey.hasPrefix("model.") ? rawKey : "model." + rawKey
            let bareKey = String(key.dropFirst("model.".count))
            if config.deleteBlockwiseModules && isBlockwiseKey(bareKey) {
                continue
            }

            if bareKey.hasPrefix("cond_module.") {
                var parts = bareKey.split(separator: ".").map(String.init)
                if parts.count > 1, Int(parts[1]) != nil {
                    parts.insert("layers", at: 1)
                    key = "model." + parts.joined(separator: ".")
                }
            }

            let normalizedPath = key.split(separator: ".").map { part -> String in
                let component = String(part)
                return Int(component) == nil ? normalizedComponent(component) : component
            }
            key = normalizedPath.joined(separator: ".")

            sanitized[key] = value
        }

        return sanitized
    }

    func prepareText(_ text: String, maxLength: Int? = nil) -> (inputIDs: MLXArray, mask: MLXArray, normalizedTexts: [String]) {
        echoTtsTextInputIDsAndMask(
            [text],
            maxLength: maxLength ?? config.maxTextLength,
            normalize: config.normalizeText,
            padToMax: false
        )
    }

    func prepareReferenceAudio(_ refAudio: MLXArray?) -> MLXArray? {
        guard let refAudio else { return nil }

        switch refAudio.ndim {
        case 1:
            return refAudio.expandedDimensions(axis: 0)
        case 2:
            if refAudio.shape[0] == 1 {
                return refAudio
            }
            if refAudio.shape[1] == 1 {
                return refAudio.transposed(1, 0)
            }
            if refAudio.shape[0] <= 8 {
                return mean(refAudio, axis: 0, keepDims: true)
            }
            return mean(refAudio, axis: 1, keepDims: true).transposed(1, 0)
        case 3:
            if refAudio.shape[0] != 1 {
                return prepareReferenceAudio(refAudio[0, 0..., 0...])
            }
            if refAudio.shape[1] == 1 {
                return refAudio[0, 0, 0...].expandedDimensions(axis: 0)
            }
            if refAudio.shape[2] == 1 {
                return refAudio[0, 0..., 0].expandedDimensions(axis: 0)
            }
            return prepareReferenceAudio(refAudio[0, 0..., 0...])
        default:
            return nil
        }
    }

    func generateLatents(
        text: String,
        speakerLatent: MLXArray? = nil,
        speakerMask: MLXArray? = nil,
        rngSeed: Int = 0,
        blockSizes: [Int]? = nil,
        numSteps: Int? = nil,
        sequenceLength: Int? = nil
    ) throws -> MLXArray {
        let textInputs = prepareText(text)
        let conditionedSpeakerLatent = speakerLatent
            ?? MLXArray.zeros([1, config.dit.speakerPatchSize, config.dit.latentSize], dtype: .float32)
        let conditionedSpeakerMask = speakerMask
            ?? MLXArray.zeros([1, conditionedSpeakerLatent.shape[1]], dtype: .bool)

        let steps = numSteps ?? config.sampler.numSteps
        let targetLength = sequenceLength ?? config.sampler.sequenceLength

        if let blockSizes {
            guard !config.deleteBlockwiseModules else {
                throw AudioGenerationError.invalidInput(
                    "Blockwise generation requires latent-prefix modules. Set deleteBlockwiseModules=false in the model config."
                )
            }
            return try echoTtsSampleBlockwiseEulerCFGIndependentGuidances(
                model: model,
                speakerLatent: conditionedSpeakerLatent,
                speakerMask: conditionedSpeakerMask,
                textInputIDs: textInputs.inputIDs,
                textMask: textInputs.mask,
                rngSeed: rngSeed,
                blockSizes: blockSizes,
                numSteps: steps,
                cfgScaleText: config.sampler.cfgScaleText,
                cfgScaleSpeaker: config.sampler.cfgScaleSpeaker,
                cfgMinT: config.sampler.cfgMinT,
                cfgMaxT: config.sampler.cfgMaxT,
                truncationFactor: config.sampler.truncationFactor,
                rescaleK: config.sampler.rescaleK,
                rescaleSigma: config.sampler.rescaleSigma,
                speakerKVScale: config.sampler.speakerKVScale,
                speakerKVMaxLayers: config.sampler.speakerKVMaxLayers,
                speakerKVMinT: config.sampler.speakerKVMinT
            )
        }

        return echoTtsSampleEulerCFGIndependentGuidances(
            model: model,
            speakerLatent: conditionedSpeakerLatent,
            speakerMask: conditionedSpeakerMask,
            textInputIDs: textInputs.inputIDs,
            textMask: textInputs.mask,
            rngSeed: rngSeed,
            numSteps: steps,
            cfgScaleText: config.sampler.cfgScaleText,
            cfgScaleSpeaker: config.sampler.cfgScaleSpeaker,
            cfgMinT: config.sampler.cfgMinT,
            cfgMaxT: config.sampler.cfgMaxT,
            truncationFactor: config.sampler.truncationFactor,
            rescaleK: config.sampler.rescaleK,
            rescaleSigma: config.sampler.rescaleSigma,
            speakerKVScale: config.sampler.speakerKVScale,
            speakerKVMaxLayers: config.sampler.speakerKVMaxLayers,
            speakerKVMinT: config.sampler.speakerKVMinT,
            sequenceLength: targetLength
        )
    }

    func decodeLatents(_ latents: MLXArray) throws -> MLXArray {
        guard let fishAE else {
            throw AudioGenerationError.modelNotInitialized("Fish S1 DAC is not loaded")
        }
        guard let pcaState else {
            throw AudioGenerationError.modelNotInitialized("Echo PCA state is not loaded")
        }
        return echoTtsAEDecode(codec: fishAE, pcaState: pcaState, latent: latents)
    }

    func generateDetailed(
        text: String,
        refAudio: MLXArray?,
        rngSeed: Int = 0,
        numSteps: Int? = nil,
        sequenceLength: Int? = nil,
        blockSizes: [Int]? = nil
    ) throws -> (audio: MLXArray, info: AudioGenerationInfo) {
        let started = CFAbsoluteTimeGetCurrent()
        let preparedText = prepareText(text)

        var speakerLatent: MLXArray?
        var speakerMask: MLXArray?
        if let preparedAudio = prepareReferenceAudio(refAudio) {
            guard let fishAE else {
                throw AudioGenerationError.modelNotInitialized("Fish S1 DAC is not loaded")
            }
            guard let pcaState else {
                throw AudioGenerationError.modelNotInitialized("Echo PCA state is not loaded")
            }
            let conditioned = echoTtsGetSpeakerLatentAndMask(
                codec: fishAE,
                pcaState: pcaState,
                audio: preparedAudio,
                maxSpeakerLatentLength: config.maxSpeakerLatentLength,
                audioChunkSize: 640 * config.audioDownsampleFactor,
                audioDownsampleFactor: config.audioDownsampleFactor,
                divisByPatchSize: config.dit.speakerPatchSize
            )
            if conditioned.speakerLatent.shape[1] > 0 {
                speakerLatent = conditioned.speakerLatent
                speakerMask = conditioned.speakerMask
            }
        }

        let latents = try generateLatents(
            text: text,
            speakerLatent: speakerLatent,
            speakerMask: speakerMask,
            rngSeed: rngSeed,
            blockSizes: blockSizes,
            numSteps: numSteps,
            sequenceLength: sequenceLength
        )
        let decoded = try decodeLatents(latents)
        let cropped = echoTtsCropAudioToFlatteningPoint(
            audio: decoded,
            latent: latents[0, 0..., 0...],
            audioDownsampleFactor: config.audioDownsampleFactor
        )
        let waveform = cropped[0, 0, 0...]
        eval(waveform)

        let elapsed = max(CFAbsoluteTimeGetCurrent() - started, 1e-6)
        let info = AudioGenerationInfo(
            promptTokenCount: preparedText.inputIDs.shape[1],
            generationTokenCount: latents.shape[1],
            prefillTime: 0,
            generateTime: elapsed,
            tokensPerSecond: Double(latents.shape[1]) / elapsed,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        return (waveform, info)
    }

    public static func fromPretrained(
        _ modelRepo: String = "mlx-community/echo-tts-base",
        cache: HubCache = .default
    ) async throws -> EchoTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(EchoTTSConfig.self, from: configData)
        let model = EchoTTSModel(config: config)

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let weightFiles = files
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent != config.pcaFilename }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !weightFiles.isEmpty else {
            throw AudioGenerationError.modelNotInitialized("No model safetensors found in \(modelDir.path)")
        }

        var weights: [String: MLXArray] = [:]
        for file in weightFiles {
            for (key, value) in try loadArrays(url: file) {
                guard weights[key] == nil else {
                    throw AudioGenerationError.invalidInput("Duplicate Echo weight key: \(key)")
                }
                weights[key] = value
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)),
            verify: .noUnusedKeys
        )

        let pcaURL = modelDir.appendingPathComponent(config.pcaFilename)
        guard FileManager.default.fileExists(atPath: pcaURL.path) else {
            throw AudioGenerationError.modelNotInitialized("Missing PCA state file: \(config.pcaFilename)")
        }
        model.pcaState = try loadEchoTTSPCAState(from: pcaURL)
        model.fishAE = try await FishS1DAC.fromPretrained(config.fishCodecRepo, cache: cache)

        eval(model.parameters())
        return model
    }
}

extension EchoTTSModel: SpeechGenerationModel {
    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = voice
        _ = refText
        _ = language
        return try generateDetailed(
            text: text,
            refAudio: refAudio,
            rngSeed: 0,
            sequenceLength: generationParameters.maxTokens
        ).audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        AsyncThrowingStream { continuation in
            do {
                _ = voice
                _ = refText
                _ = language
                let result = try generateDetailed(
                    text: text,
                    refAudio: refAudio,
                    rngSeed: 0,
                    sequenceLength: generationParameters.maxTokens
                )
                continuation.yield(.info(result.info))
                continuation.yield(.audio(result.audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}
