import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXFast
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Tokenizers

public final class HiggsV2Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: HiggsV2Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "audio_codebook_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo var layers: [HiggsV2DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "audio_decoder_proj") var projector: HiggsV2DecoderProjector
    private var _tokenizer: (any Tokenizer)?
    private var _codec: HiggsAudioTokenizer?

    public var sampleRate: Int { 24_000 }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 1200, temperature: 0.7, topP: 0.95)
    }

    public init(_ configuration: HiggsV2Configuration) {
        self.configuration = configuration
        let text = configuration.textConfig
        _embedTokens.wrappedValue = Embedding(embeddingCount: text.vocabularySize, dimensions: text.hiddenSize)
        _audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.audioNumCodebooks * (configuration.audioCodebookSize + 2),
            dimensions: text.hiddenSize)
        _layers.wrappedValue = (0..<text.hiddenLayers).map { _ in HiggsV2DecoderLayer(text) }
        _norm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _projector.wrappedValue = HiggsV2DecoderProjector(configuration)
    }

    func hiddenStates(_ embeddings: MLXArray, audioMask: MLXArray,
                      cache: [KVCache]? = nil, audioOnly: Bool = false) -> MLXArray {
        var h = embeddings
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = h.dim(1) > 1
            ? .array(createCausalMask(n: h.dim(1), offset: cache?.first?.offset ?? 0)) : .none
        for (index, layer) in layers.enumerated() {
            h = layer(h, audioMask: audioMask, mask: mask, cache: cache?[index], audioOnly: audioOnly)
        }
        return norm(h)
    }

    public func callAsFunction(inputsEmbeds: MLXArray, audioOutMask: MLXArray,
                               cache: [KVCache]? = nil) -> (MLXArray, MLXArray) {
        let h = hiddenStates(inputsEmbeds, audioMask: audioOutMask, cache: cache)
        return (projector.textHead(h), audioLogits(h))
    }

    private func audioLogits(_ h: MLXArray) -> MLXArray {
        projector.audioHead(h).reshaped([h.dim(0), h.dim(1), configuration.audioNumCodebooks,
                                        configuration.audioCodebookSize + 2])
    }

    func audioEmbedding(_ codes: MLXArray) -> MLXArray {
        let shifts = MLXArray(0..<configuration.audioNumCodebooks).asType(.int32)
            * (configuration.audioCodebookSize + 2)
        return sum(audioEmbeddings(codes + shifts.expandedDimensions(axis: -1)), axis: 0)
    }

    func preparePrompt(text: String, refAudio: MLXArray?, refText: String?,
                       options: HiggsV2GenerationOptions) throws -> (MLXArray, MLXArray) {
        guard let tokenizer = _tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Higgs text tokenizer is unavailable")
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("Higgs synthesis requires nonempty text")
        }
        var embeddings: [MLXArray] = []
        var masks: [MLXArray] = []
        func appendText(_ text: String) {
            let ids: [Int32] = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
            guard !ids.isEmpty else { return }
            embeddings.append(embedTokens(MLXArray(ids)))
            masks.append(MLXArray.zeros([ids.count], dtype: .bool))
        }
        func appendAudio(_ codes: MLXArray) {
            let k = configuration.audioNumCodebooks
            let wrapped = concatenated([
                MLXArray.full([k, 1], values: MLXArray(Int32(configuration.audioStreamBOS)), dtype: .int32),
                codes,
                MLXArray.full([k, 1], values: MLXArray(Int32(configuration.audioStreamEOS)), dtype: .int32)
            ], axis: 1)
            let delayed = higgsV2ApplyDelay(wrapped, bos: configuration.audioStreamBOS,
                                            eos: configuration.audioStreamEOS)
            embeddings.append(audioEmbedding(delayed))
            masks.append(MLXArray.ones([delayed.dim(1)], dtype: .bool))
        }
        appendText("<|begin_of_text|>")
        if let system = options.systemPrompt, !system.isEmpty {
            appendText("<|start_header_id|>system<|end_header_id|>\n\n\(system)<|eot_id|>")
        }
        for turn in options.conversation {
            guard ["system", "user", "assistant"].contains(turn.role) else {
                throw AudioGenerationError.invalidInput("Invalid Higgs conversation role: \(turn.role)")
            }
            appendText("<|start_header_id|>\(turn.role)<|end_header_id|>\n\n\(turn.text)")
            if let rows = turn.audioCodes {
                guard rows.count == configuration.audioNumCodebooks, let count = rows.first?.count,
                      count > 0, rows.allSatisfy({ $0.count == count && $0.allSatisfy { $0 >= 0 && $0 < configuration.audioCodebookSize } }) else {
                    throw AudioGenerationError.invalidInput("Invalid Higgs conversation audio codebooks")
                }
                appendText("<|audio_out_bos|>")
                appendAudio(MLXArray(rows.flatMap { $0 }).reshaped([rows.count, count]))
                appendText("<|audio_eos|>")
            }
            appendText("<|eot_id|>")
        }
        if let refAudio {
            guard let codec = _codec else {
                throw AudioGenerationError.modelNotInitialized("Higgs audio codec is unavailable")
            }
            guard let refText, !refText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AudioGenerationError.invalidInput("Reference audio requires its transcript")
            }
            appendText("<|start_header_id|>user<|end_header_id|>\n\n\(refText)<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>")
            guard refAudio.ndim == 1 || (refAudio.ndim == 2 && (refAudio.dim(0) == 1 || refAudio.dim(1) == 1)) else {
                throw AudioGenerationError.invalidInput("Higgs reference audio must be a mono waveform")
            }
            let waveform = refAudio.reshaped([1, -1, 1]).asType(.float32)
            let codes = try codec.encode(waveform)[0].transposed(1, 0)
            eval(codes)
            appendAudio(codes)
            appendText("<|audio_eos|><|eot_id|>")
        }
        appendText("<|start_header_id|>user<|end_header_id|>\n\n\(text)<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>")
        return (concatenated(embeddings, axis: 0).expandedDimensions(axis: 0),
                concatenated(masks, axis: 0).expandedDimensions(axis: 0))
    }

    private func generateCodes(
        prompt: MLXArray, audioMask: MLXArray, parameters: GenerateParameters,
        options: HiggsV2GenerationOptions,
        onFrame: (([MLXArray], Bool) throws -> Void)? = nil
    ) throws -> (MLXArray, AudioGenerationInfo) {
        let maxFrames = parameters.maxTokens ?? 1200
        guard maxFrames > configuration.audioNumCodebooks + 1,
              parameters.temperature.isFinite, parameters.temperature >= 0,
              options.maximumRepeat > 0, options.samplingWarmupFrames >= 0 else {
            throw AudioGenerationError.invalidInput("Invalid Higgs frame limit or sampling options")
        }
        let start = Date()
        let cache: [KVCache] = layers.map { _ in KVCacheSimple() }
        let prefill = hiddenStates(prompt, audioMask: audioMask, cache: cache)
        eval(prefill)
        let prefillTime = Date().timeIntervalSince(start)
        let generationStart = Date()
        let k = configuration.audioNumCodebooks
        let bos = Int32(configuration.audioStreamBOS), eos = Int32(configuration.audioStreamEOS)
        var frames = [MLXArray([Int32](repeating: bos, count: k))]
        var history = [[Int32]](repeating: [bos], count: k)
        var remaining: Int? = nil
        var delay = 0
        var key = MLXRandom.key(parameters.seed ?? UInt64.random(in: 0...UInt64.max))
        let stepMask = MLXArray.ones([1, 1], dtype: .bool)
        for step in 0..<maxFrames {
            try Task.checkCancellation()
            let frame: [Int32] = autoreleasepool {
                let embedding = audioEmbedding(frames.last!.reshaped([k, 1])).expandedDimensions(axis: 0)
                let h = hiddenStates(embedding, audioMask: stepMask, cache: cache, audioOnly: true)
                let logits = audioLogits(h)[0, 0]
                let (nextKey, sampleKey) = MLXRandom.split(key: key)
                key = nextKey
                let sampled = higgsV2Sample(logits, temperature: step < options.samplingWarmupFrames ? 0 : parameters.temperature,
                    topP: parameters.topP, topK: parameters.topK, key: sampleKey)
                var tokens = sampled.asArray(Int32.self)
                if options.repetitionWindow > 0 {
                    let greedy = argMax(logits, axis: -1).asArray(Int32.self)
                    for i in 0..<k where history[i].suffix(options.repetitionWindow).filter({ $0 == tokens[i] }).count >= options.maximumRepeat {
                        tokens[i] = greedy[i]
                    }
                }
                if delay + 1 < k {
                    for i in (delay + 1)..<k { tokens[i] = bos }
                    delay += 1
                }
                if let value = remaining {
                    for i in 0..<(k - value) { tokens[i] = eos }
                    remaining = value - 1
                } else if let lastEOS = tokens.lastIndex(of: eos) {
                    for i in 0..<lastEOS { tokens[i] = eos }
                    remaining = k - lastEOS - 1
                }
                return tokens
            }
            for i in 0..<k {
                history[i].append(frame[i])
                if history[i].count > max(1, options.repetitionWindow) {
                    history[i].removeFirst(history[i].count - max(1, options.repetitionWindow))
                }
            }
            frames.append(MLXArray(frame))
            let stopped = remaining.map { $0 <= 0 } ?? false
            try onFrame?(frames, stopped || step == maxFrames - 1)
            if stopped { break }
        }
        let aligned = try alignedCodes(frames)
        let elapsed = Date().timeIntervalSince(generationStart)
        return (aligned, AudioGenerationInfo(promptTokenCount: prompt.dim(1), generationTokenCount: frames.count - 1,
            prefillTime: prefillTime, generateTime: elapsed,
            tokensPerSecond: Double(frames.count - 1) / max(elapsed, 0.001),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9))
    }

    private func alignedCodes(_ frames: [MLXArray]) throws -> MLXArray {
        let aligned = higgsV2RevertDelay(stacked(frames, axis: 1))
        guard aligned.dim(1) > 2 else {
            throw AudioGenerationError.generationFailed("Higgs did not generate any audio frames")
        }
        return clip(aligned[0..., 1..<(aligned.dim(1) - 1)], min: 0, max: configuration.audioCodebookSize - 1)
    }

    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try await generate(text: text, refAudio: refAudio, refText: refText,
                           parameters: generationParameters, options: .init())
    }

    public func generate(text: String, refAudio: MaterializedArray? = nil, refText: String? = nil,
                         parameters: GenerateParameters? = nil,
                         options: HiggsV2GenerationOptions = .init()) async throws -> MaterializedArray {
        guard let codec = _codec else {
            throw AudioGenerationError.modelNotInitialized("Higgs codec is unavailable")
        }
        let (prompt, mask) = try preparePrompt(text: text, refAudio: refAudio, refText: refText, options: options)
        let (codes, _) = try generateCodes(prompt: prompt, audioMask: mask,
            parameters: parameters ?? defaultGenerationParameters, options: options)
        var pcm = codec.decode(codes.transposed(1, 0)).asType(.float32).asArray(Float.self)
        higgsV2Fade(&pcm, milliseconds: options.fadeInMilliseconds, sampleRate: sampleRate, fadeIn: true)
        higgsV2Fade(&pcm, milliseconds: options.fadeOutMilliseconds, sampleRate: sampleRate, fadeIn: false)
        return (MLXArray(pcm)).materialized()
    }

    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, refAudio: refAudio, refText: refText, parameters: generationParameters)
    }

    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters,
                               streamingInterval: Double) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, refAudio: refAudio, refText: refText, parameters: generationParameters,
                       streamingInterval: streamingInterval)
    }

    public func generateStream(text: String, refAudio: MaterializedArray? = nil, refText: String? = nil,
                               parameters: GenerateParameters? = nil, options: HiggsV2GenerationOptions = .init(),
                               streamingInterval: Double = 0.64) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { continuation.finish(throwing: CancellationError()); return }
            do {
                guard streamingInterval.isFinite, streamingInterval > 0,
                      options.overlapMilliseconds.isFinite, options.overlapMilliseconds >= 0 else {
                    throw AudioGenerationError.invalidInput("Invalid Higgs streaming interval or overlap")
                }
                guard let codec = self._codec else {
                    throw AudioGenerationError.modelNotInitialized("Higgs codec is unavailable")
                }
                let (prompt, mask) = try self.preparePrompt(text: text, refAudio: reference, refText: refText, options: options)
                let interval = max(self.configuration.audioNumCodebooks + 2, Int(streamingInterval / 0.04))
                var emittedAt = 0
                var streamer = HiggsV2PCMStreamer(overlap: Int(options.overlapMilliseconds * Float(self.sampleRate) / 1000))
                let (_, info) = try self.generateCodes(prompt: prompt, audioMask: mask,
                    parameters: parameters ?? self.defaultGenerationParameters, options: options) { frames, final in
                    guard frames.count > self.configuration.audioNumCodebooks + 1,
                          final || frames.count - emittedAt >= interval else { return }
                    let codes = try self.alignedCodes(frames)
                    var pcm = codec.decode(codes.transposed(1, 0)).asType(.float32).asArray(Float.self)
                    if streamer.emitted == 0 {
                        higgsV2Fade(&pcm, milliseconds: options.fadeInMilliseconds, sampleRate: self.sampleRate, fadeIn: true)
                    }
                    if final {
                        higgsV2Fade(&pcm, milliseconds: options.fadeOutMilliseconds, sampleRate: self.sampleRate, fadeIn: false)
                    }
                    let chunk = streamer.consume(pcm, final: final)
                    if !chunk.isEmpty { continuation.yield(.audio(MLXArray(chunk))) }
                    emittedAt = frames.count
                }
                continuation.yield(.info(info))
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    public static func fromPretrained(_ repo: String, cache: HubCache = .default,
                                      codecSource: String = "mlx-community/higgs-audio-v2-tokenizer") async throws -> HiggsV2Model {
        guard let id = Repo.ID(rawValue: repo) else { throw TTSModelError.invalidRepositoryID(repo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: id,
            requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory, cache: cache, codecSource: codecSource)
    }

    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default,
                                          codecSource: String = "mlx-community/higgs-audio-v2-tokenizer") async throws -> HiggsV2Model {
        let config = try JSONDecoder().decode(HiggsV2Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = HiggsV2Model(config)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            where file.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: file) {
                guard weights[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate Higgs weight: \(key)") }
                weights[key] = value
            }
        }
        guard !weights.isEmpty else { throw AudioGenerationError.modelNotInitialized("No Higgs safetensors found") }
        if let quantization = config.quantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil ? nil : quantization.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        model._tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let bundledCodec = directory.appendingPathComponent("audio_tokenizer/config.json")
        let resolvedCodecSource = FileManager.default.fileExists(atPath: bundledCodec.path)
            ? directory.path : codecSource
        model._codec = try await HiggsAudioTokenizer.fromPretrained(resolvedCodecSource, cache: cache)
        guard model._codec?.config.dacNumCodebooks == config.audioNumCodebooks,
              model._codec?.config.sampleRate == model.sampleRate else {
            throw AudioGenerationError.invalidInput("Higgs model and codec codebooks/sample rates differ")
        }
        return model
    }
}
