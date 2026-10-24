import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

/// Prompt-driven music understanding and lyrics transcription.
public final class MossMusicModel: Module, STTGenerationModel {
    public let config: MossMusicConfig
    @ModuleInfo(key: "audio_encoder") var audioEncoder: MossMusicEncoder
    @ModuleInfo(key: "language_model") var languageModel: Qwen3ASRTextModel
    @ModuleInfo(key: "audio_adapter") var adapter: MossMusicMerger
    @ModuleInfo(key: "deepstack_audio_merger_list") var deepstackMergers: [MossMusicMerger]
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    var tokenizer: Tokenizer?
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 1024, topP: 1, repetitionContextSize: 100)
    }
    public init(_ config: MossMusicConfig) {
        self.config = config
        _audioEncoder.wrappedValue = MossMusicEncoder(config.audioConfig)
        _languageModel.wrappedValue = Qwen3ASRTextModel(config.languageConfig)
        _adapter.wrappedValue = MossMusicMerger(config)
        _deepstackMergers.wrappedValue = (0 ..< config.injectionCount).map { _ in
            MossMusicMerger(config)
        }
        _lmHead.wrappedValue = Linear(
            config.languageConfig.hiddenSize, config.languageConfig.vocabSize, bias: false)
    }

    func extractFeatures(_ audio: MLXArray) -> MLXArray {
        let mono: MLXArray
        if audio.ndim == 2 {
            mono =
                audio.dim(0) <= 8 && audio.dim(1) > audio.dim(0)
                ? audio.mean(axis: 0) : audio.mean(axis: 1)
        } else {
            mono = audio
        }
        precondition(mono.ndim == 1 && mono.size > 200, "MOSS-Music expects a recording at 16 kHz")
        return WhisperAudio.logMelSpectrogram(mono, nMels: config.audioConfig.numMelBins)
    }

    func encodeAudio(_ features: MLXArray) -> (MLXArray, [MLXArray]) {
        let c = config.audioConfig
        let chunkFrames = config.audioConfig.windowSize * 2
        let length = features.dim(1)
        let maxFrames = min(length, chunkFrames)
        var chunks: [MLXArray] = []
        var lengths: [Int] = []
        for start in stride(from: 0, to: length, by: chunkFrames) {
            let frames = min(chunkFrames, length - start)
            var chunk = features[0..., start ..< (start + frames)]
            if frames < maxFrames {
                chunk = padded(chunk, widths: [.init((0, 0)), .init((0, maxFrames - frames))])
            }
            chunks.append(chunk)
            lengths.append(frames)
        }
        var encoded: [MLXArray] = []
        var deepParts = Array(repeating: [MLXArray](), count: deepstackMergers.count)
        for start in stride(from: 0, to: chunks.count, by: c.convBatchSize) {
            let end = min(start + c.convBatchSize, chunks.count)
            let batch = stacked(Array(chunks[start ..< end])).asType(
                audioEncoder.conv1.weight.dtype)
            let (out, deep) = audioEncoder(batch, lengths: Array(lengths[start ..< end]))
            eval([out] + deep)
            for i in start ..< end {
                let frames = MossMusicAudioConfig.outputLength(lengths[i])
                encoded.append(out[i - start, 0 ..< frames, 0...])
                for j in deepParts.indices {
                    deepParts[j].append(deep[j][i - start, 0 ..< frames, 0...])
                }
            }
        }
        return (concatenated(encoded, axis: 0), deepParts.map { concatenated($0, axis: 0) })
    }

    func prepareInputs(
        audio: MLXArray, prompt: String?, timeMarkers: Bool, enableThinking: Bool? = nil
    ) -> (
        ids: MLXArray, embeddings: MLXArray, deepstack: [MLXArray]
    ) {
        guard let tokenizer else { preconditionFailure("MOSS-Music tokenizer is not loaded") }
        let features = extractFeatures(audio)
        let ids = MossMusicPrompt(config: config, tokenizer: tokenizer).inputIds(
            audioTokens: MossMusicAudioConfig.outputLength(features.dim(1)), prompt: prompt,
            timeMarkers: timeMarkers, enableThinking: enableThinking)
        let arrays = encodeAudio(features)
        return injectAudio(ids: ids, features: arrays.0, deepstack: arrays.1)
    }

    func injectAudio(ids: [Int], features: MLXArray, deepstack: [MLXArray]) -> (
        ids: MLXArray, embeddings: MLXArray, deepstack: [MLXArray]
    ) {
        let input = MLXArray(ids.map(Int32.init))
        let mask = MLXArray(ids.map { $0 == config.audioTokenId })
        precondition(ids.filter { $0 == config.audioTokenId }.count == features.dim(0))
        let text = languageModel.embedTokens(
            MLX.where(mask, MLXArray(Int32(0)), input).expandedDimensions(axis: 0))
        let indices = clip(cumsum(mask.asType(.int32)) - 1, min: 0, max: features.dim(0) - 1)
        let merged = adapter(features).asType(text.dtype)[indices].expandedDimensions(axis: 0)
        let expandedMask = mask.expandedDimensions(axes: [0, 2])
        let embeddings = MLX.where(expandedMask, merged, text)
        let deep = zip(deepstackMergers, deepstack).map { merger, features in
            let values = merger(features).asType(text.dtype)[indices].expandedDimensions(axis: 0)
            return MLX.where(
                expandedMask, values, MLXArray.zeros(values.shape, dtype: values.dtype))
        }
        return (input, embeddings, deep)
    }

    func forward(_ embeddings: MLXArray, cache: [KVCache], deepstack: [MLXArray] = []) -> MLXArray {
        let mask = Qwen3ASRTextModel.attentionMask(h: embeddings, cache: cache.first)
        var h = embeddings
        for (i, layer) in languageModel.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache[i])
            if i < deepstack.count { h = h + deepstack[i] }
        }
        return lmHead(languageModel.norm(h))
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generateImpl(
            audio: audio, parameters: generationParameters, prompt: nil,
            stripThinking: config.stripThinking, timeMarkers: config.enableTimeMarkers,
            enableThinking: nil,
            prefillStepSize: 2048, minP: 0, emit: nil)
    }
    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, stripThinking: Bool? = nil, enableTimeMarkers: Bool? = nil,
        enableThinking: Bool? = nil,
        prefillStepSize: Int = 2048, minP: Float = 0
    ) -> STTOutput {
        generateImpl(
            audio: audio, parameters: generationParameters ?? defaultGenerationParameters,
            prompt: prompt, stripThinking: stripThinking ?? config.stripThinking,
            timeMarkers: enableTimeMarkers ?? config.enableTimeMarkers,
            enableThinking: enableThinking,
            prefillStepSize: prefillStepSize, minP: minP, emit: nil)
    }

    private func generateImpl(
        audio: MLXArray, parameters p: STTGenerateParameters, prompt: String?,
        stripThinking: Bool, timeMarkers: Bool, enableThinking: Bool?, prefillStepSize: Int,
        minP: Float,
        emit: ((String) -> Void)?
    ) -> STTOutput {
        guard let tokenizer else { preconditionFailure("MOSS-Music tokenizer is not loaded") }
        precondition(
            prefillStepSize > 0 && p.maxTokens >= 0 && p.temperature >= 0 && p.repetitionPenalty > 0
        )
        let start = Date()
        let prepared = prepareInputs(
            audio: audio, prompt: prompt, timeMarkers: timeMarkers, enableThinking: enableThinking)
        let ids = prepared.ids
        let embeddings = prepared.embeddings
        let available = config.languageConfig.maxPositionEmbeddings - ids.size
        precondition(available > 0, "Audio prompt exceeds the MOSS-Music context budget")
        var cache: [KVCache] = languageModel.layers.map { _ in KVCacheSimple() }
        let cacheConfiguration = p.kvBits.map { bits -> KVCacheConfiguration in
            guard
                let affine = try? AffineKVCacheConfiguration(
                    bits: bits, groupSize: p.kvGroupSize,
                    compressionStart: p.quantizedKVStart)
            else { preconditionFailure("Invalid MOSS-Music KV-cache settings") }
            return .init(strategy: .affine(affine))
        }
        func finishStep(_ logits: MLXArray?) {
            if let logits { eval(logits, cache) } else { eval(cache) }
            if let cacheConfiguration {
                do {
                    _ = try applyKVCacheConfiguration(
                        cache: &cache, configuration: cacheConfiguration)
                } catch {
                    preconditionFailure("Unsupported MOSS-Music KV-cache settings: \(error)")
                }
            }
        }
        var offset = 0
        while ids.size - offset > 1 {
            if Task.isCancelled { break }
            let count = min(prefillStepSize, ids.size - offset - 1)
            let slice = offset ..< (offset + count)
            _ = forward(
                embeddings[0..., slice, 0...], cache: cache,
                deepstack: prepared.deepstack.map { $0[0..., slice, 0...] })
            finishStep(nil)
            offset += count
        }
        var logits = forward(
            embeddings[0..., offset..., 0...], cache: cache,
            deepstack: prepared.deepstack.map { $0[0..., offset..., 0...] })
        finishStep(logits)
        let prefillTime = Date().timeIntervalSince(start)
        let decodeStart = Date()
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK, minP: minP)
        var tokens: [Int] = []
        var emitted = ""
        // Python's repetition processor begins with the final prefill token.
        var history = [ids[(-1)...].item(Int.self)]
        func emitText(_ raw: String, final: Bool) {
            let text = MossMusicText.response(raw, stripThinking: stripThinking, final: final)
            guard !text.contains("\u{FFFD}"), text.hasPrefix(emitted) else { return }
            let delta = String(text.dropFirst(emitted.count))
            if !delta.isEmpty {
                emit?(delta)
                if p.verbose { print(delta, terminator: "") }
            }
            emitted = text
        }
        for step in 0 ..< min(p.maxTokens, available) {
            if Task.isCancelled { break }
            let scores = logits[0, -1, 0...]
            if p.repetitionPenalty != 1 {
                let recent = Array(Set(history.suffix(max(1, p.repetitionContextSize))))
                let indices = MLXArray(recent.map(Int32.init))
                let values = scores[MLXArray(recent.map(Int32.init))]
                scores[indices] = MLX.where(
                    values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
            }
            let token = sampler.sample(logits: scores).item(Int.self)
            if token == config.eosTokenId { break }
            tokens.append(token)
            history.append(token)
            if emit != nil || p.verbose {
                emitText(tokenizer.decode(tokens: tokens, skipSpecialTokens: true), final: false)
            }
            if step + 1 < min(p.maxTokens, available) {
                logits = forward(
                    languageModel.embedTokens(MLXArray([Int32(token)]).reshaped(1, 1)), cache: cache
                )
                finishStep(logits)
            }
        }
        let raw = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
        let text = MossMusicText.response(raw, stripThinking: stripThinking)
        emitText(raw, final: true)
        if p.verbose { print() }
        let decodeTime = Date().timeIntervalSince(decodeStart)
        let samples = audio.ndim == 1 ? audio.size : max(audio.dim(0), audio.dim(1))
        let duration = Double(samples) / Double(sampleRate)
        return STTOutput(
            text: text, segments: MossMusicText.segments(text, duration: duration),
            promptTokens: ids.size, generationTokens: tokens.count,
            totalTokens: ids.size + tokens.count,
            promptTps: Double(ids.size) / max(prefillTime, 1e-6),
            generationTps: Double(tokens.count) / max(decodeTime, 1e-6),
            totalTime: Date().timeIntervalSince(start),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        generateStream(audio: audio, generationParameters: Optional(generationParameters))
    }
    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, stripThinking: Bool? = nil, enableTimeMarkers: Bool? = nil,
        enableThinking: Bool? = nil,
        prefillStepSize: Int = 2048, minP: Float = 0
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = model.value.generateImpl(
                        audio: input.value,
                        parameters: generationParameters ?? model.value.defaultGenerationParameters,
                        prompt: prompt,
                        stripThinking: stripThinking ?? model.value.config.stripThinking,
                        timeMarkers: enableTimeMarkers ?? model.value.config.enableTimeMarkers,
                        enableThinking: enableThinking,
                        prefillStepSize: prefillStepSize, minP: minP
                    ) { continuation.yield(.token($0)) }
                    try Task.checkCancellation()
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
