import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public final class DramaBoxModel: SpeechGenerationModel, @unchecked Sendable {
    public let sampleRate = 48_000
    public var generateConfig: DramaBoxGenerateConfig

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters()
    }

    let promptEncoder: DramaBoxPromptEncoder
    let dit: DramaBoxLTXModel
    let audioVAE: DramaBoxAudioVAE
    let vocoder: DramaBoxVocoderWithBWE
    var negativeACtx: MLXArray?

    init(
        promptEncoder: DramaBoxPromptEncoder,
        dit: DramaBoxLTXModel,
        audioVAE: DramaBoxAudioVAE,
        vocoder: DramaBoxVocoderWithBWE,
        generateConfig: DramaBoxGenerateConfig = .default
    ) {
        self.promptEncoder = promptEncoder
        self.dit = dit
        self.audioVAE = audioVAE
        self.vocoder = vocoder
        self.generateConfig = generateConfig
    }

    public static func fromPretrained(
        _ modelRepo: String = dramaBoxDefaultAudioRepository,
        gemmaRepo: String = dramaBoxDefaultGemmaRepository,
        cache: HubCache = .default,
        hfToken: String? = nil
    ) async throws -> DramaBoxModel {
        let token = hfToken
            ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        let gemmaSpec = DramaBoxWeights.resolveGemmaRepository(gemmaRepo)
        let audioDir = try await DramaBoxWeights.resolveDirectory(
            spec: modelRepo,
            requiredFiles: DramaBoxWeights.audioRequiredFiles,
            cache: cache,
            hfToken: token
        )
        let gemmaDir = try await DramaBoxWeights.resolveGemmaDirectory(
            spec: gemmaSpec,
            cache: cache,
            hfToken: token
        )
        return try await fromModelDirectory(audioDir, gemmaDir: gemmaDir)
    }

    public static func fromModelDirectory(
        _ audioDir: URL,
        gemmaDir: URL? = nil,
        cache: HubCache = .default,
        hfToken: String? = nil
    ) async throws -> DramaBoxModel {
        let resolvedGemmaDir: URL
        if let gemmaDir {
            resolvedGemmaDir = gemmaDir
        } else {
            let token = hfToken
                ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
            resolvedGemmaDir = try await DramaBoxWeights.resolveGemmaDirectory(
                spec: dramaBoxDefaultGemmaRepository,
                cache: cache,
                hfToken: token
            )
        }
        print("Loading Gemma 3 backbone from \(resolvedGemmaDir.path)")
        let (gemma, _) = try DramaBoxGemma3TextBackbone.load(from: resolvedGemmaDir)
        let tokenizer = try await LTXVGemmaTokenizer.fromDirectory(resolvedGemmaDir)

        let components = try DramaBoxWeights.loadSafetensors(
            from: audioDir,
            files: [dramaBoxAudioComponentsFile]
        )
        let featureExtractor = DramaBoxFeatureExtractor()
        try loadDramaBoxFeatureExtractorWeights(featureExtractor, state: components)
        let connector = DramaBoxEmbeddings1DConnector()
        try loadDramaBoxConnectorWeights(connector, state: components)
        let processor = DramaBoxEmbeddingsProcessor(
            featureExtractor: featureExtractor,
            audioConnector: connector
        )
        let promptEncoder = DramaBoxPromptEncoder(
            gemma: gemma,
            tokenizer: tokenizer,
            processor: processor
        )

        let dit = DramaBoxLTXModel()
        let ditState = try DramaBoxWeights.loadSafetensors(
            from: audioDir,
            files: [dramaBoxDiTWeightFile]
        )
        try loadDramaBoxDiTWeights(dit, state: ditState)

        let vae = DramaBoxAudioVAE()
        try loadDramaBoxAudioVAEWeights(vae, state: components)

        let vocoder = DramaBoxVocoderWithBWE()
        try loadDramaBoxVocoderWeights(vocoder, state: components)

        eval(dit)
        eval(vae)
        eval(vocoder)
        eval(featureExtractor)
        eval(connector)
        return DramaBoxModel(
            promptEncoder: promptEncoder,
            dit: dit,
            audioVAE: vae,
            vocoder: vocoder
        )
    }

    public func generate(
        prompt: String,
        referenceAudio: MLXArray? = nil,
        referenceSampleRate: Int? = nil,
        config: DramaBoxGenerateConfig? = nil
    ) async throws -> DramaBoxResult {
        let config = config ?? generateConfig
        if config.denoiseRef {
            throw DramaBoxError.denoiseRefUnsupported
        }
        let rescale = config.resolvedRescaleScale
        let params = DramaBoxGuiderParams(
            cfgScale: config.cfgScale,
            stgScale: config.stgScale,
            rescaleScale: rescale,
            modalityScale: config.modalityScale
        )

        let aCtx = promptEncoder.encode(prompt, maxLength: config.maxPromptLength)
        eval(aCtx)
        var aCtxNeg: MLXArray?
        if params.needsUncond {
            if negativeACtx == nil {
                negativeACtx = promptEncoder.encode(
                    dramaBoxDefaultNegativePrompt,
                    maxLength: config.maxPromptLength
                )
                eval(negativeACtx!)
            }
            aCtxNeg = negativeACtx
        }

        let target = dramaBoxTargetShapeFromDuration(config.durationSeconds)
        let shape = DramaBoxAudioLatentShape(target)
        let patchifier = DramaBoxAudioPatchifier()
        let tools = DramaBoxAudioLatentTools(patchifier: patchifier, targetShape: shape)
        var state = tools.createInitialState(dtype: .bfloat16)

        var loopDenoiseMask: MLXArray?
        if let referenceAudio {
            let sr = referenceSampleRate ?? config.referenceSampleRate ?? sampleRate
            let prepared = try prepareDramaBoxReferenceAudio(
                referenceAudio,
                sampleRate: sr,
                refDurationS: config.referenceDurationSeconds
            )
            let processor = DramaBoxAudioProcessor()
            let refMel = try processor.waveformToMel(prepared.waveform, sampleRate: prepared.sampleRate)
            let refLatent = audioVAE.encode(refMel)
            state = dramaBoxApplyReferenceLatent(state, refLatent: refLatent, patchifier: patchifier)
            loopDenoiseMask = state.denoiseMask
        }

        let noiser = DramaBoxGaussianNoiser(seed: config.seed)
        state = noiser(state, noiseScale: 1.0)

        let sigmas = DramaBoxLTX2Scheduler().execute(
            steps: config.steps,
            tokens: state.latent.dim(-1)
        )
        let x0 = DramaBoxX0Model(dit: dit)
        state = dramaBoxEulerDenoisingLoop(
            state,
            sigmas: sigmas,
            x0Model: x0,
            aCtx: aCtx,
            aCtxNeg: aCtxNeg,
            params: params,
            positions: state.positions,
            denoiseMask: loopDenoiseMask
        )

        state = tools.clearConditioning(state)
        state = tools.unpatchifyState(state)
        let latent4d = dramaBoxSilencePriorFix(state.latent)
        let mel = audioVAE.decode(latent4d)
        let waveform = vocoder(mel)
        eval(waveform)
        let stereo = waveform[0].asType(.float32)
        guard MLX.all(MLX.isFinite(stereo)).item(Bool.self) else {
            throw DramaBoxError.generationFailed(
                "DramaBox produced non-finite samples (NaN/Inf). Prompt encoding or the vocoder path exploded."
            )
        }
        return DramaBoxResult(
            waveform: stereo,
            sampleRate: sampleRate,
            durationSeconds: config.durationSeconds,
            settings: config
        )
    }

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
        _ = generationParameters
        let result = try await generate(
            prompt: text,
            referenceAudio: refAudio,
            config: generateConfig
        )
        return result.waveform
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}
