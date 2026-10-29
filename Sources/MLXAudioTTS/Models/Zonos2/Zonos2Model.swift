import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
@preconcurrency import MLXAudioCore
import MLXFFT
import MLXFast
import MLXNN

/// Native ZONOS2 text-to-speech. Serialize calls on each model instance.
public final class Zonos2Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: Zonos2Configuration
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        .init(
            maxTokens: 1024, temperature: 1.15, topP: 0, topK: 106, minP: 0.18,
            repetitionPenalty: 1.2, repetitionContextSize: 50)
    }
    @ModuleInfo(key: "multi_embedder") var embedder: Zonos2MultiEmbedding
    @ModuleInfo(key: "speaker_lda_projection") var speakerLDA: Linear?
    @ModuleInfo(key: "speaker_projection") var speakerProjection: Linear?
    let layers: [Zonos2Block]
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm
    @ModuleInfo(key: "multi_output") var output: Linear
    private var dac: DescriptDAC?
    private var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var directory: URL?
    var hubCache: HubCache = .default

    public init(config: Zonos2Configuration) throws {
        try config.validate()
        self.config = config
        _embedder.wrappedValue = Zonos2MultiEmbedding(config)
        if config.speakerEnabled {
            if let width = config.speakerLDADim {
                _speakerLDA.wrappedValue = Linear(config.speakerEmbeddingDim, width)
            }
            _speakerProjection.wrappedValue = Linear(
                config.speakerLDADim ?? config.speakerEmbeddingDim, config.dim)
        }
        layers = (0..<config.nLayers).map { Zonos2Block(config, index: $0) }
        _outNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _output.wrappedValue = Linear(
            config.dim, config.nCodebooks * config.audioVocabSize, bias: false)
    }

    func hidden(
        _ ids: MLXArray, cache: [KVCache]? = nil, speakerEmbedding: MLXArray? = nil,
        speakerPosition: Int? = nil
    ) -> MLXArray {
        var h = embedder(ids)
        if let embedding = speakerEmbedding, let position = speakerPosition,
            let projection = speakerProjection
        {
            var speaker = embedding
            if let lda = speakerLDA { speaker = lda(speaker.asType(lda.weight.dtype)) }
            speaker = projection(speaker.asType(projection.weight.dtype)).asType(h.dtype)
            h = concatenated(
                [
                    h[0..., 0..<position, 0...], speaker.expandedDimensions(axis: 1),
                    h[0..., (position + 1)..., 0...],
                ], axis: 1)
        }
        h = MLXFast.rmsNorm(h, weight: MLXArray.ones([h.dim(-1)], dtype: h.dtype), eps: config.normEps)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = createAttentionMask(
            h: h, cache: cache?.first)
        var residual: MLXArray?
        var state: MLXArray?
        for (i, layer) in layers.enumerated() {
            (h, residual, state) = layer(
                h, residual: residual, routerState: state, mask: mask, cache: cache?[i])
        }
        return outNorm(residual.map { $0 + h } ?? h)
    }
    func callAsFunction(
        _ ids: MLXArray, cache: [KVCache]? = nil, speakerEmbedding: MLXArray? = nil,
        speakerPosition: Int? = nil
    ) -> MLXArray {
        let h = hidden(
            ids, cache: cache, speakerEmbedding: speakerEmbedding, speakerPosition: speakerPosition)
        var logits = output(h).reshaped(h.dim(0), h.dim(1), config.nCodebooks, config.audioVocabSize)
        if config.lossSoftcap > 0 { logits = config.lossSoftcap * tanh(logits / config.lossSoftcap) }
        return logits
    }

    static func speakerMel(_ audio: MLXArray, sampleRate: Int) throws -> MLXArray {
        guard audio.ndim == 1, audio.size > 384 else {
            throw Zonos2Error.invalidInput("Speaker audio needs more than 384 mono samples")
        }
        let left = audio[1..<385][.stride(by: -1)]
        let right = audio[(audio.size - 385)..<(audio.size - 1)][.stride(by: -1)]
        let padded = concatenated([left, audio, right])
        let count = 1 + (padded.size - 1024) / 256
        let frames = asStrided(padded, [count, 1024], strides: [256, 1])
        let magnitude = abs(MLXFFT.rfft(frames * hanningWindow(size: 1024, periodic: true), axis: -1))
        let filters = melFilters(
            sampleRate: sampleRate, nFft: 1024, nMels: 128, fMin: 0, fMax: Float(sampleRate) / 2,
            norm: "slaney", melScale: .slaney)
        return log(maximum(matmul(magnitude, filters), 1e-5)).expandedDimensions(axis: 0)
    }

    private func loadSpeakerEncoder() async throws -> Qwen3TTSSpeakerEncoder {
        if let speakerEncoder { return speakerEncoder }
        var location: URL?
        if let path = config.speakerEncoderPath {
            let candidate =
                path.hasPrefix("/") ? URL(fileURLWithPath: path) : directory?.appendingPathComponent(path)
            if let candidate,
                FileManager.default.fileExists(
                    atPath: candidate.appendingPathComponent("model.safetensors").path)
            {
                location = candidate
            }
        }
        if location == nil {
            guard let id = Repo.ID(rawValue: config.speakerEncoderModelID) else {
                throw TTSModelError.invalidRepositoryID(config.speakerEncoderModelID)
            }
            location = try await ModelUtils.resolveOrDownloadModel(
                repoID: id, requiredExtension: ".safetensors", cache: hubCache)
        }
        let folder = location!
        let c = try JSONDecoder().decode(
            Qwen3TTSSpeakerEncoderConfig.self,
            from: Data(contentsOf: folder.appendingPathComponent("config.json")))
        guard c.encDim == config.speakerEmbeddingDim, c.sampleRate == config.speakerEncoderSampleRate
        else {
            throw Zonos2Error.invalidConfiguration(
                "Speaker encoder dimensions/sample rate do not match ZONOS2")
        }
        let encoder = Qwen3TTSSpeakerEncoder(config: c)
        let weights = try MLX.loadArrays(url: folder.appendingPathComponent("model.safetensors"))
        let prefixed = Dictionary(
            uniqueKeysWithValues: weights.map {
                ($0.key.hasPrefix("speaker_encoder.") ? $0.key : "speaker_encoder." + $0.key, $0.value)
            })
        try encoder.update(
            parameters: ModuleParameters.unflattened(Qwen3TTSSpeakerEncoder.sanitize(weights: prefixed)),
            verify: .all)
        encoder.train(false)
        eval(encoder)
        speakerEncoder = encoder
        return encoder
    }

    /// Extract the 2048-dimensional speaker vector. Input can be mono or a two-dimensional channel/sample array.
    public func extractSpeakerEmbedding(_ audio: sending MLXArray, sampleRate sourceRate: Int = 24000)
        async throws -> sending MLXArray
    {
        try await speakerEmbedding(audio, sampleRate: sourceRate)
    }
    private func speakerEmbedding(_ audio: MLXArray, sampleRate sourceRate: Int) async throws
        -> MLXArray
    {
        guard config.speakerEnabled, sourceRate > 0, audio.ndim == 1 || audio.ndim == 2, audio.size > 0,
            all(isFinite(audio)).item(Bool.self)
        else { throw Zonos2Error.invalidInput("Speaker reference must contain finite audio") }
        var waveform = audio.asType(.float32)
        if waveform.ndim == 2 {
            waveform = mean(waveform, axis: waveform.dim(0) <= waveform.dim(1) ? 0 : 1)
        }
        if sourceRate != config.speakerEncoderSampleRate {
            waveform = MLXArray(
                try resampleAudio(
                    waveform.asArray(Float.self), from: sourceRate, to: config.speakerEncoderSampleRate))
        }
        let encoder = try await loadSpeakerEncoder()
        let embedding = encoder(
            try Self.speakerMel(waveform, sampleRate: config.speakerEncoderSampleRate)
        ).asType(.float32)
        eval(embedding)
        return embedding
    }

    private func loadDAC() async throws -> DescriptDAC {
        if let dac { return dac }
        let local = directory?.appendingPathComponent("dac")
        let codec: DescriptDAC
        if let local,
            FileManager.default.fileExists(atPath: local.appendingPathComponent("model.safetensors").path)
        {
            codec = try DescriptDAC.fromModelDirectory(local)
        } else {
            codec = try await DescriptDAC.fromPretrained(config.dacModelID, cache: hubCache)
        }
        guard codec.sampleRate == sampleRate, codec.config.nCodebooks == config.nCodebooks,
            codec.config.codebookSize == config.codebookSize
        else {
            throw Zonos2Error.invalidConfiguration("DAC codebooks/sample rate do not match ZONOS2")
        }
        codec.train(false)
        dac = codec
        return codec
    }
    func decode(_ state: Zonos2GenerationState, frameLimit: Int? = nil) async throws
        -> sending MLXArray
    {
        let codes = Zonos2Prompt.codes(
            state.generated, config: config, eosFrame: state.eosFrame, frameLimit: frameLimit)
        guard codes.size > 0 else { return MLXArray.zeros([0]) }
        let codec = try await loadDAC()
        let audio = codec.decodeFromCodes(codes.T.expandedDimensions(axis: 0)).asType(.float32)
            .reshaped(-1)
        let waveform = audio[0..<min(audio.size, codes.dim(0) * 512)]
        eval(waveform)
        return waveform
    }

    private func run(
        text: String, reference: MLXArray?, speakerEmbedding: MLXArray?, language: String?,
        options o: Zonos2Options,
        parameters p: GenerateParameters, interval: Double?, emit: ((sending MLXArray) -> Void)?,
        seedStride: Int = 1, seedOffset: Int = 0
    ) async throws -> MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            p.temperature.isFinite, p.temperature >= 0, p.topP.isFinite, p.minP.isFinite,
            p.topP >= 0, p.topP <= 1, p.minP >= 0, p.minP <= 1, p.topK >= 0,
            (p.repetitionPenalty ?? 1.2).isFinite, (p.repetitionPenalty ?? 1.2) > 0,
            (p.maxTokens ?? 1024) > 0
        else { throw Zonos2Error.invalidInput("Invalid text or sampling parameters") }
        if let interval, !interval.isFinite || interval <= 0 {
            throw Zonos2Error.invalidInput("Streaming interval must be positive and finite")
        }
        guard reference == nil || speakerEmbedding == nil else {
            throw Zonos2Error.invalidInput("Provide reference audio or a speaker embedding, not both")
        }
        var speaker = speakerEmbedding
        if let reference {
            speaker = try await self.speakerEmbedding(
                reference, sampleRate: o.referenceSampleRate ?? sampleRate)
        }
        if let value = speaker {
            guard value.size == config.speakerEmbeddingDim,
                value.ndim == 1 || value.shape == [1, config.speakerEmbeddingDim],
                all(isFinite(value)).item(Bool.self)
            else {
                throw Zonos2Error.invalidInput("Speaker embedding has the wrong shape or nonfinite values")
            }
            speaker = value.reshaped(1, config.speakerEmbeddingDim)
        }
        let normalized =
            o.textNormalization ? Zonos2Text.normalize(text, language: language ?? "en_us") : text
        let rows = try Zonos2Prompt.build(
            text: normalized, config: config, options: o, speaker: speaker != nil)
        let limit = p.maxTokens ?? 1024
        guard rows.count + limit <= config.maxSequenceLength else {
            throw Zonos2Error.invalidInput("Prompt plus maximum audio frames exceeds max_seqlen")
        }
        let cache: [KVCache] = (0..<config.nLayers).map { _ in KVCacheSimple() }
        let prompt = MLXArray(rows.flatMap { $0.map(Int32.init) }).reshaped(
            1, rows.count, config.nCodebooks + 1)
        var logits = callAsFunction(
            prompt, cache: cache, speakerEmbedding: speaker, speakerPosition: speaker == nil ? nil : 0)[
                0, -1]
        var state = Zonos2GenerationState(codebooks: config.nCodebooks, eoaID: config.eoaID)
        let chunkFrames = max(1, Int((interval ?? 2) * Double(sampleRate) / 512))
        var emitted = 0
        for step in 0..<limit {
            try Task.checkCancellation()
            let frame = Zonos2Sampling.sample(
                logits, state: state, parameters: p, options: o, textVocab: config.textVocab,
                step: step * seedStride + seedOffset)
            state.append(frame, ignoreEOS: o.ignoreEOS)
            if state.finished { break }
            if let emit, interval != nil {
                let complete = max(0, state.generated.count - (config.nCodebooks - 1))
                if complete - emitted / 512 >= chunkFrames {
                    // This matches the reference prefix decoder: only fully delayed frames are emitted.
                    var prefixState = state
                    prefixState.eosFrame = nil
                    let prefix = try await decode(prefixState, frameLimit: complete)
                    if prefix.size > emitted {
                        let chunk = prefix[emitted...]
                        eval(chunk)
                        emitted = prefix.size
                        emit(chunk)
                    }
                }
            }
            let next = MLXArray(frame.map(Int32.init)).reshaped(1, 1, -1)
            logits = callAsFunction(next, cache: cache)[0, -1]
        }
        if let emit {
            let audio = try await decode(state)
            if audio.size > emitted {
                let chunk = audio[emitted...]
                eval(chunk)
                emit(chunk)
            }
            return MLXArray.zeros([0])
        }
        return try await decode(state)
    }

    public func generate(
        text: String, refAudio: sending MLXArray? = nil, speakerEmbedding: sending MLXArray? = nil,
        language: String? = nil, options: Zonos2Options = .init(),
        generationParameters: GenerateParameters? = nil
    ) async throws -> sending MLXArray {
        try await run(
            text: text, reference: refAudio, speakerEmbedding: speakerEmbedding, language: language,
            options: options,
            parameters: generationParameters ?? defaultGenerationParameters, interval: nil, emit: nil)
    }
    /// Serial batch convenience, preserving input order and per-sequence conditioning.
    /// Seed keys match the reference: `seed + step * texts.count + sequenceIndex`.
    public func generateBatch(
        texts: [String], refAudio: sending MLXArray? = nil,
        refAudios: sending [MLXArray]? = nil, speakerEmbedding sharedSpeaker: sending MLXArray? = nil,
        speakerEmbeddings: sending MLXArray? = nil, referenceSampleRates: [Int]? = nil,
        language: String? = nil, options: Zonos2Options = .init(),
        generationParameters: GenerateParameters? = nil
    ) async throws -> sending [MLXArray] {
        let supplied = [
            refAudio != nil, refAudios != nil, sharedSpeaker != nil, speakerEmbeddings != nil,
        ].filter { $0 }.count
        guard supplied <= 1 else {
            throw Zonos2Error.invalidInput("Provide one shared or per-sequence speaker input")
        }
        if let refAudios, refAudios.count != texts.count {
            throw Zonos2Error.invalidInput("One reference is required per sequence")
        }
        if let referenceSampleRates, referenceSampleRates.count != texts.count || refAudios == nil {
            throw Zonos2Error.invalidInput(
                "Reference sample rates require one per-sequence reference each")
        }
        if let speakerEmbeddings, speakerEmbeddings.shape != [texts.count, config.speakerEmbeddingDim] {
            throw Zonos2Error.invalidInput(
                "Speaker embeddings must have shape [texts.count, speakerEmbeddingDim]")
        }
        guard !texts.isEmpty else { return [] }
        var shared = sharedSpeaker
        if let refAudio {
            shared = try await self.speakerEmbedding(
                refAudio, sampleRate: options.referenceSampleRate ?? sampleRate)
        }
        var result: [MLXArray] = []
        for i in texts.indices {
            var speaker = shared
            if let refAudios {
                speaker = try await self.speakerEmbedding(
                    refAudios[i],
                    sampleRate: referenceSampleRates?[i] ?? options.referenceSampleRate ?? sampleRate)
            } else if let speakerEmbeddings {
                speaker = speakerEmbeddings[i]
            }
            result.append(
                try await run(
                    text: texts[i], reference: nil, speakerEmbedding: speaker, language: language,
                    options: options,
                    parameters: generationParameters ?? defaultGenerationParameters, interval: nil, emit: nil,
                    seedStride: texts.count, seedOffset: i))
        }
        return result
    }
    public func generate(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        try await generate(
            text: text, refAudio: refAudio, language: language, generationParameters: generationParameters
        )
    }
    public func generateStream(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, refAudio: refAudio, language: language, generationParameters: generationParameters
        )
    }
    public func generateStream(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters, streamingInterval: Double
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, refAudio: refAudio, language: language,
            generationParameters: generationParameters, streamingInterval: streamingInterval)
    }
    public func generateStream(
        text: String, refAudio: sending MLXArray? = nil, speakerEmbedding: sending MLXArray? = nil,
        language: String? = nil, options: Zonos2Options = .init(),
        generationParameters: GenerateParameters? = nil,
        streamingInterval: Double = 2
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let referenceBox = SendingBox(refAudio)
        let speakerBox = SendingBox(speakerEmbedding)
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    _ = try await run(
                        text: text, reference: referenceBox.take(), speakerEmbedding: speakerBox.take(),
                        language: language, options: options,
                        parameters: generationParameters ?? defaultGenerationParameters,
                        interval: streamingInterval,
                        emit: { continuation.yield(.audio($0)) })
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
