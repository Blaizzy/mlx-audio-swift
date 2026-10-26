import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXAudioCore
import Tokenizers

/// Flow matching controls shared by VoxCPM 1 and 1.5.
public struct VoxCPMGenerationParameters: Sendable {
    public var maxTokens: Int
    public var inferenceTimesteps: Int
    public var cfgValue: Float

    public init(maxTokens: Int = 4096, inferenceTimesteps: Int = 10, cfgValue: Float = 2) {
        self.maxTokens = maxTokens
        self.inferenceTimesteps = inferenceTimesteps
        self.cfgValue = cfgValue
    }
}

/// VoxCPM 1 / 1.5: MiniCPM, scalar quantization, local flow matching and a 44.1 kHz VAE.
public final class VoxCPMModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: VoxCPMConfiguration
    public let tokenizer: Tokenizers.Tokenizer?
    let patchSize: Int
    let featDim: Int

    @ModuleInfo(key: "base_lm") var baseLM: VoxMiniCPMModel
    @ModuleInfo(key: "residual_lm") var residualLM: VoxMiniCPMModel
    @ModuleInfo(key: "feat_encoder") var featEncoder: VoxCPMLocEnc
    @ModuleInfo(key: "feat_decoder") var featDecoder: VoxUnifiedCFM
    @ModuleInfo(key: "fsq_layer") var fsqLayer: ScalarQuantizationLayer
    @ModuleInfo(key: "enc_to_lm_proj") var encToLmProj: Linear
    @ModuleInfo(key: "lm_to_dit_proj") var lmToDitProj: Linear
    @ModuleInfo(key: "res_to_dit_proj") var resToDitProj: Linear
    @ModuleInfo(key: "stop_proj") var stopProj: Linear
    @ModuleInfo(key: "stop_head") var stopHead: Linear
    @ModuleInfo(key: "audio_vae") var audioVAE: VoxAudioVAE

    public var sampleRate: Int { config.audioVaeConfig.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 4096, temperature: 1)
    }

    public init(_ config: VoxCPMConfiguration, tokenizer: Tokenizers.Tokenizer? = nil) {
        self.config = config
        self.tokenizer = tokenizer
        self.patchSize = config.patchSize
        self.featDim = config.featDim

        func withDefaultRope(_ config: VoxCPM2LMConfig) -> VoxCPM2LMConfig {
            var c = config
            if c.ropeShortFactor.isEmpty { c.ropeShortFactor = Array(repeating: 1, count: c.headDim / 2) }
            if c.ropeLongFactor.isEmpty { c.ropeLongFactor = Array(repeating: 1, count: c.headDim / 2) }
            return c
        }
        let lm = withDefaultRope(config.lmConfig)
        self._baseLM.wrappedValue = VoxMiniCPMModel(lm)
        var residual = lm
        residual.numHiddenLayers = config.residualLmNumLayers
        residual.vocabSize = 0
        self._residualLM.wrappedValue = VoxMiniCPMModel(residual)
        var encoder = lm
        encoder.hiddenSize = config.encoderConfig.hiddenDim
        encoder.intermediateSize = config.encoderConfig.ffnDim
        encoder.numAttentionHeads = config.encoderConfig.numHeads
        encoder.numHiddenLayers = config.encoderConfig.numLayers
        encoder.vocabSize = 0
        self._featEncoder.wrappedValue = VoxCPMLocEnc(withDefaultRope(encoder), inputDim: config.featDim)
        var dit = lm
        dit.hiddenSize = config.ditConfig.hiddenDim
        dit.intermediateSize = config.ditConfig.ffnDim
        dit.numAttentionHeads = config.ditConfig.numHeads
        dit.numHiddenLayers = config.ditConfig.numLayers
        dit.vocabSize = 0
        self._featDecoder.wrappedValue = VoxUnifiedCFM(
            inChannels: config.featDim, cfmParams: config.ditConfig.cfmConfig,
            estimator: VoxCPMLocDiT(withDefaultRope(dit), inChannels: config.featDim, conditioning: .summed))
        self._fsqLayer.wrappedValue = ScalarQuantizationLayer(
            inDim: lm.hiddenSize, outDim: lm.hiddenSize,
            latentDim: config.scalarQuantizationLatentDim, scale: config.scalarQuantizationScale)
        self._encToLmProj.wrappedValue = Linear(config.encoderConfig.hiddenDim, lm.hiddenSize)
        self._lmToDitProj.wrappedValue = Linear(lm.hiddenSize, config.ditConfig.hiddenDim)
        self._resToDitProj.wrappedValue = Linear(lm.hiddenSize, config.ditConfig.hiddenDim)
        self._stopProj.wrappedValue = Linear(lm.hiddenSize, lm.hiddenSize)
        self._stopHead.wrappedValue = Linear(lm.hiddenSize, 2, bias: false)
        // Python's causal transposed convolution applies odd-stride output padding
        // before trimming. Preserve the existing VoxCPM2 decoder default separately.
        self._audioVAE.wrappedValue = VoxAudioVAE(config.audioVaeConfig.sharedConfiguration,
            applyTransposeOutputPadding: true)
        super.init()
    }

    /// Reference audio is mono at `sampleRate`; left padding matches VoxCPM's prompt alignment.
    func encodePromptAudio(_ audio: MLXArray) throws -> MLXArray {
        guard audio.ndim == 1, audio.size > 0 else {
            throw AudioGenerationError.invalidInput("VoxCPM reference audio must be a nonempty mono waveform")
        }
        let alignment = patchSize * audioVAE.hopLength
        let padding = (alignment - audio.size % alignment) % alignment
        let padded = MLX.padded(audio, widths: [IntOrPair((padding, 0))])
        let encoded = audioVAE.encode(padded.reshaped(1, -1, 1), sampleRate: sampleRate).squeezed(axis: 0)
        let patches = encoded.dim(0) / patchSize
        return encoded[..<(patches * patchSize)].reshaped(patches, patchSize, featDim)
    }

    func tokenize(_ text: String) throws -> [Int] {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("VoxCPM tokenizer not loaded")
        }
        // Unlike VoxCPM2, v1 uses tokenizer.encode() including the initial BOS.
        return tokenizer.encode(text: text)
    }

    public func generate(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        try await generate(text: text, refAudio: refAudio, refText: refText,
            parameters: VoxCPMGenerationParameters(maxTokens: generationParameters.maxTokens ?? 4096))
    }

    public func generate(
        text: String, refAudio: sending MLXArray? = nil, refText: String? = nil,
        parameters: VoxCPMGenerationParameters = VoxCPMGenerationParameters()
    ) async throws -> sending MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              parameters.maxTokens > 0, parameters.inferenceTimesteps > 0,
              parameters.cfgValue.isFinite else {
            throw AudioGenerationError.invalidInput("VoxCPM requires text, positive token and timestep limits, and finite guidance")
        }
        guard (refAudio == nil) == (refText == nil) else {
            throw AudioGenerationError.invalidInput("VoxCPM voice cloning requires both reference audio and its transcript")
        }
        let scale = Float(config.lmConfig.useMup ? 1 : config.lmConfig.scaleEmb)
        let ids = try tokenize((refText ?? "") + text) + [101]
        guard ids.count < config.maxLength else {
            throw AudioGenerationError.invalidInput("VoxCPM text exceeds the model context length")
        }
        let combined: MLXArray
        var prefix = MLXArray.zeros([1, patchSize, featDim])
        var audioMask: MLXArray?
        var textMask: MLXArray?
        var featureEmbedding: MLXArray?
        if let refAudio {
            let features = try encodePromptAudio(refAudio)
            let n = features.dim(0)
            guard ids.count + n < config.maxLength else {
                throw AudioGenerationError.invalidInput("VoxCPM reference and text exceed the model context length")
            }
            let tokens = MLXArray((ids + Array(repeating: 0, count: n)).map(Int32.init)).expandedDimensions(axis: 0)
            let allFeatures = MLX.concatenated([
                MLXArray.zeros([ids.count, patchSize, featDim]), features
            ], axis: 0).expandedDimensions(axis: 0)
            let tm = MLX.concatenated([MLXArray.ones([ids.count]), MLXArray.zeros([n])]).reshaped(1, -1, 1)
            let am = 1 - tm
            let feat = encToLmProj(featEncoder(allFeatures))
            combined = tm * baseLM.embedTokens!(tokens) * scale + am * feat
            prefix = allFeatures[0..., -1, 0..., 0...]
            textMask = tm
            audioMask = am
            featureEmbedding = feat
        } else {
            combined = baseLM.embedTokens!(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0)) * scale
        }
        var (encoded, lmCache) = baseLM(inputsEmbeds: combined)
        if let audioMask, let textMask {
            encoded = fsqLayer(encoded) * audioMask + encoded * textMask
        }
        var hidden = encoded[0..., -1, 0...]
        if audioMask == nil { hidden = fsqLayer(hidden) }
        let residualInput: MLXArray
        if let audioMask, let featureEmbedding {
            residualInput = encoded + audioMask * featureEmbedding
        } else {
            residualInput = encoded
        }
        let (residual, initialResidualCache) = residualLM(inputsEmbeds: residualInput)
        var residualCache = initialResidualCache
        var residualHidden = residual[0..., -1, 0...]
        var patches: [MLXArray] = []
        let limit = min(parameters.maxTokens, config.maxLength - combined.dim(1))
        for i in 0..<limit {
            try Task.checkCancellation()
            let mu = lmToDitProj(hidden) + resToDitProj(residualHidden)
            let prediction = featDecoder.sample(mu: mu, nTimesteps: parameters.inferenceTimesteps,
                patchSize: patchSize, cond: prefix.transposed(0, 2, 1), cfgValue: parameters.cfgValue)
                .transposed(0, 2, 1)
            patches.append(prediction)
            let current = encToLmProj(featEncoder(prediction.expandedDimensions(axis: 1)))
            let stop = MLX.argMax(stopHead(silu(stopProj(hidden))), axis: -1).item(Int.self)
            if i > 5 && stop == 1 { break }
            let next: MLXArray
            (next, lmCache) = baseLM(inputsEmbeds: current, cache: lmCache)
            hidden = fsqLayer(next[0..., -1, 0...])
            let nextResidual: MLXArray
            (nextResidual, residualCache) = residualLM(
                inputsEmbeds: hidden.expandedDimensions(axis: 1) + current, cache: residualCache)
            residualHidden = nextResidual[0..., -1, 0...]
            prefix = prediction
            eval(hidden, residualHidden, prediction)
        }
        let audio = audioVAE.decode(MLX.concatenated(patches, axis: 1)).flattened()
        eval(audio)
        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let refAudio = SendingBox(refAudio)
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let startTime = Date()
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio.take(),
                    refText: refText, language: language,
                    generationParameters: generationParameters
                )
                let generateTime = Date().timeIntervalSince(startTime)

                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: audio.dim(audio.ndim - 1),
                    prefillTime: 0,
                    generateTime: generateTime,
                    tokensPerSecond: Double(audio.dim(audio.ndim - 1)) / max(generateTime, 0.001),
                    peakMemoryUsage: 0
                )
                continuation.yield(.audio(audio))
                continuation.yield(.info(info))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { @Sendable _ in
            task.cancel()
        }

        return stream
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let prefix = "audio_vae."
        let vae = Dictionary(uniqueKeysWithValues: weights.compactMap { key, value in
            key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
        })
        var result: [String: MLXArray] = [:]
        for (key, value) in weights where !key.hasPrefix(prefix) {
            let mapped = key.replacingOccurrences(of: ".inv_freq", with: ".invFreq")
                .replacingOccurrences(of: ".long_factor", with: ".longFactor")
                .replacingOccurrences(of: ".short_factor", with: ".shortFactor")
            result[mapped] = value
        }
        for (key, value) in audioVAE.sanitize(weights: vae) {
            result[prefix + key] = value
        }
        if let special = result.removeValue(forKey: "feat_encoder.special_token") {
            result["feat_encoder.specialToken"] = special
        }
        // These analytic buffers are constructed by MLX Swift, not learned checkpoint tensors.
        for (key, value) in parameters().flattened() {
            if result[key] == nil && (key.contains(".rope.") || key.hasSuffix("time_embeddings.freqs")) {
                result[key] = value
            }
        }
        return result
    }

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> VoxCPMModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )

        return try await fromModelDirectory(modelDir, hfToken: hfToken)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        hfToken: String? = nil
    ) async throws -> VoxCPMModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(VoxCPMConfiguration.self, from: Data(contentsOf: configURL))
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        let model = VoxCPMModel(config, tokenizer: tokenizer)

        // Load weights (single or sharded)
        let weights = try loadWeights(modelDir: modelDir)

        // Sanitize
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize only modules whose published checkpoint carries learned scales.
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitizedWeights["\(path).scales"] != nil else { return nil }
                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path)
                {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }

        // Update model parameters
        guard sanitizedWeights.keys.contains(where: { $0.hasPrefix("audio_vae.") }) else {
            throw AudioGenerationError.modelNotInitialized("VoxCPM requires AudioVAE safetensors weights")
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .all)
        eval(model)

        return model
    }

    private static func loadWeights(modelDir: URL) throws -> [String: MLXArray] {
        let singleURL = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: singleURL.path) {
            return try MLX.loadArrays(url: singleURL)
        }

        // Sharded weights
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !safetensorFiles.isEmpty else {
            throw AudioGenerationError.modelNotInitialized(
                "No .safetensors files found in \(modelDir.path)"
            )
        }

        var allWeights = [String: MLXArray]()
        for file in safetensorFiles {
            let shard = try MLX.loadArrays(url: file)
            for (key, value) in shard {
                allWeights[key] = value
            }
        }
        return allWeights
    }
}
