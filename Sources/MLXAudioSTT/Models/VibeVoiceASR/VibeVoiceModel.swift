// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

final class VibeVoiceConnector: Module {
    @ModuleInfo var fc1: Linear
    let norm: RMSNorm
    @ModuleInfo var fc2: Linear
    init(_ input: Int, _ output: Int) {
        _fc1.wrappedValue = Linear(input, output)
        norm = RMSNorm(dimensions: output, eps: 1e-6)
        _fc2.wrappedValue = Linear(output, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(norm(fc1(x))) }
}

/// Persistent text and speaker context for one VibeVoice streaming session.
public final class VibeVoiceStreamingState {
    var cache: [KVCache]
    let speechStart: MLXArray
    let speechEnd: MLXArray
    public internal(set) var promptTokens: Int
    public internal(set) var generationTokens = 0
    init(cache: [KVCache], speechStart: MLXArray, speechEnd: MLXArray, promptTokens: Int) {
        self.cache = cache
        self.speechStart = speechStart
        self.speechEnd = speechEnd
        self.promptTokens = promptTokens
    }
}

/// VibeVoice long-form and streaming speech recognition.
public final class VibeVoiceASRModel: Module, STTGenerationModel {
    public let config: VibeVoiceConfig
    @ModuleInfo(key: "acoustic_tokenizer") var acousticTokenizer: VibeVoiceTokenizerEncoder
    @ModuleInfo(key: "semantic_tokenizer") var semanticTokenizer: VibeVoiceTokenizerEncoder
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceConnector
    @ModuleInfo(key: "semantic_connector") var semanticConnector: VibeVoiceConnector
    @ModuleInfo(key: "language_model") public var languageModel: Qwen2Model
    var tokenizer: Tokenizer?
    var speechStartId = 151646
    var speechEndId = 151647
    var speechPadId = 151648
    var textChunkEndId: Int?
    public var sampleRate: Int { config.sampleRate }
    public var isStreamingModel: Bool {
        config.chunkFrames != nil && config.lookaheadFrames != nil && textChunkEndId != nil
    }
    public var streamingChunkSamples: Int {
        (config.chunkFrames ?? 0) * config.speechTokenCompressRatio
    }
    public var streamingWindowSamples: Int {
        ((config.chunkFrames ?? 0) + (config.lookaheadFrames ?? 0))
            * config.speechTokenCompressRatio
    }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 8192, temperature: 0, topP: 1)
    }

    public init(_ config: VibeVoiceConfig) {
        self.config = config
        _acousticTokenizer.wrappedValue = VibeVoiceTokenizerEncoder(config.acousticTokenizerConfig)
        _semanticTokenizer.wrappedValue = VibeVoiceTokenizerEncoder(config.semanticTokenizerConfig)
        _acousticConnector.wrappedValue = VibeVoiceConnector(
            config.acousticVaeDim, config.decoderConfig.hiddenSize)
        _semanticConnector.wrappedValue = VibeVoiceConnector(
            config.semanticVaeDim, config.decoderConfig.hiddenSize)
        _languageModel.wrappedValue = Qwen2Model(config.decoderConfig)
    }
    func prepareAudio(_ audio: MLXArray) -> MLXArray {
        precondition(
            (audio.ndim == 1 || audio.ndim == 2) && audio.size > 0,
            "Expected nonempty mono or time-major audio")
        var mono = (audio.ndim == 2 ? audio.mean(axis: -1) : audio).asType(.float32)
        if config.normalizeAudio {
            mono = mono * (pow(Float(10), -25 / 20) / (sqrt(mean(mono * mono)) + 1e-6))
            mono = mono / maximum(abs(mono).max() + 1e-6, MLXArray(Float(1)))
        }
        return mono[0 ..< min(mono.size, 59 * 60 * sampleRate)].expandedDimensions(axis: 0)
    }
    /// Encode mono 24 kHz audio, shaped [samples] or [1, samples], without loudness normalization.
    public func encodeSpeech(_ audio: MLXArray) -> MLXArray {
        let waveform = (audio.ndim == 1 ? audio.expandedDimensions(axis: 0) : audio).asType(
            .float32)
        precondition(waveform.ndim == 2 && waveform.dim(0) == 1 && waveform.dim(1) > 0)
        let acoustic = acousticConnector(acousticTokenizer(waveform))
        eval(acoustic)
        let semantic = semanticConnector(semanticTokenizer(waveform))
        let features = acoustic + semantic
        eval(features)
        return features
    }
    static func context(_ context: String?, hotwords: [String]) -> String? {
        let terms = hotwords.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter {
            !$0.isEmpty
        }
        guard !terms.isEmpty else { return context }
        let joined = terms.joined(separator: ", ")
        return context.flatMap { $0.isEmpty ? nil : $0 }.map { $0 + "\n" + joined } ?? joined
    }
    func buildPrompt(frames: Int, duration: Double, context: String?) -> MLXArray {
        guard let tokenizer else { preconditionFailure("VibeVoice tokenizer is not loaded") }
        let system =
            "You are a helpful assistant that transcribes audio input into text output in JSON format."
        let keys = "Start time, End time, Speaker ID, Content"
        let length = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), duration)
        let info = context?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let suffix =
            info.isEmpty
            ? "This is a \(length) seconds audio, please transcribe it with these keys: " + keys
            : "This is a \(length) seconds audio, with extra info: \(info)\n\nPlease transcribe it with these keys: "
                + keys
        let prefix =
            "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n<|object_ref_start|>"
        let end = "<|object_ref_end|>\n" + suffix + "<|im_end|>\n<|im_start|>assistant\n"
        let tokens =
            tokenizer.encode(text: prefix, addSpecialTokens: false)
            + Array(repeating: speechPadId, count: frames)
            + tokenizer.encode(text: end, addSpecialTokens: false)
        return MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)
    }
    func mergeEmbeddings(_ ids: MLXArray, features: MLXArray) -> MLXArray {
        let text = languageModel.embedTokens(ids)
        let mask = ids .== speechPadId
        let indices = clip(
            cumsum(mask.asType(.int32), axis: 1) - 1, min: 0, max: features.dim(1) - 1)
        let speech = features.asType(text.dtype)[0][indices]
        return MLX.where(mask.expandedDimensions(axis: -1), speech, text)
    }
    func compress(_ cache: inout [KVCache], _ p: STTGenerateParameters) {
        guard let bits = p.kvBits else { return }
        do {
            let affine = try AffineKVCacheConfiguration(
                bits: bits, groupSize: p.kvGroupSize, compressionStart: p.quantizedKVStart)
            _ = try applyKVCacheConfiguration(
                cache: &cache, configuration: .init(strategy: .affine(affine)))
        } catch { preconditionFailure("Invalid VibeVoice KV-cache settings: \(error)") }
    }
    @discardableResult
    func decode(
        _ embeddings: MLXArray, cache: inout [KVCache], parameters p: STTGenerateParameters,
        stopIds: Set<Int>, prefillStepSize: Int, minP: Float, repetitionTokens: [Int] = [],
        emit: ((String) -> Void)? = nil
    ) -> [Int] {
        guard let tokenizer else { preconditionFailure("VibeVoice tokenizer is not loaded") }
        precondition(
            prefillStepSize > 0 && p.maxTokens >= 0 && p.temperature >= 0 && p.repetitionPenalty > 0
        )
        if Task.isCancelled { return [] }
        var offset = 0
        while embeddings.dim(1) - offset > prefillStepSize {
            let x = embeddings[0..., offset ..< (offset + prefillStepSize), 0...]
            eval(languageModel(inputEmbeddings: x, cache: cache), cache)
            compress(&cache, p)
            offset += prefillStepSize
            if Task.isCancelled { return [] }
        }
        var logits = languageModel(inputEmbeddings: embeddings[0..., offset..., 0...], cache: cache)
        eval(logits, cache)
        compress(&cache, p)
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK, minP: minP)
        var tokens: [Int] = []
        var emitted = ""
        for _ in 0 ..< p.maxTokens {
            if Task.isCancelled { break }
            var scores = logits[0, -1, 0...]
            let history = p.repetitionPenalty == 1 ? [] : repetitionTokens + tokens
            if p.repetitionPenalty != 1 && !history.isEmpty {
                // Streaming checkpoints apply their penalty to log probabilities.
                if isStreamingModel { scores = scores - logSumExp(scores) }
                let indices = MLXArray(
                    Array(Set(history.suffix(max(1, p.repetitionContextSize)))).map(Int32.init))
                let values = scores[indices]
                scores[indices] = MLX.where(
                    values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
            }
            let token = sampler.sample(logits: scores).item(Int.self)
            if stopIds.contains(token) { break }
            tokens.append(token)
            if emit != nil || p.verbose {
                let text = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                if !text.contains("\u{FFFD}"), text.hasPrefix(emitted) {
                    let delta = String(text.dropFirst(emitted.count))
                    if !delta.isEmpty {
                        emit?(delta)
                        if p.verbose { print(delta, terminator: "") }
                    }
                    emitted = text
                }
            }
            logits = languageModel(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
            eval(logits, cache)
            compress(&cache, p)
        }
        let text = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
        if text.hasPrefix(emitted) {
            let tail = String(text.dropFirst(emitted.count))
            if !tail.isEmpty {
                emit?(tail)
                if p.verbose { print(tail, terminator: "") }
            }
        }
        return tokens
    }
    public func makeStreamingState(context: String? = nil, hotwords: [String] = [])
        -> VibeVoiceStreamingState
    {
        precondition(
            isStreamingModel, "This checkpoint does not support the VibeVoice streaming protocol")
        guard let tokenizer else { preconditionFailure("VibeVoice tokenizer is not loaded") }
        var prompt =
            "You are a helpful assistant that transcribes audio input into text output. Please transcribe the following audios streamingly with these keys: speaker, content"
        if let info = Self.context(context, hotwords: hotwords), !info.isEmpty {
            prompt += " and extra info: " + info
        }
        prompt += "\n"
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let cache: [KVCache] = (0 ..< config.decoderConfig.hiddenLayers).map { _ in KVCacheSimple()
        }
        eval(
            languageModel(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0), cache: cache),
            cache)
        let start = languageModel.embedTokens(MLXArray([Int32(speechStartId)]).reshaped(1, 1))
        let end = languageModel.embedTokens(MLXArray([Int32(speechEndId)]).reshaped(1, 1))
        eval(start, end)
        return VibeVoiceStreamingState(
            cache: cache, speechStart: start, speechEnd: end, promptTokens: ids.count)
    }
    /// Advance one live session using features from a complete audio window, including lookahead.
    public func streamingGenerateStep(
        audioFeatures: MLXArray, state: VibeVoiceStreamingState,
        generationParameters: STTGenerateParameters = .init(maxTokens: 256), minP: Float = 0
    ) -> String {
        precondition(isStreamingModel)
        let audio =
            audioFeatures.ndim == 2 ? audioFeatures.expandedDimensions(axis: 0) : audioFeatures
        let embeddings = concatenated(
            [state.speechStart, audio, state.speechEnd], axis: 1)
        var stop: Set<Int> = [textChunkEndId!]
        if let eos = tokenizer?.eosTokenId { stop.insert(eos) }
        let tokens = decode(
            embeddings, cache: &state.cache, parameters: generationParameters, stopIds: stop,
            prefillStepSize: 2048, minP: minP)
        // Every window ends with this delimiter, even at EOS or the token limit.
        eval(
            languageModel(MLXArray([Int32(textChunkEndId!)]).reshaped(1, 1), cache: state.cache),
            state.cache)
        compress(&state.cache, generationParameters)
        state.generationTokens += tokens.count
        var text = tokenizer!.decode(tokens: tokens, skipSpecialTokens: true)
        for marker in [
            "<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>",
        ] {
            text = text.replacingOccurrences(of: marker, with: "")
        }
        return text
    }
    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generateImpl(
            audio, generationParameters, context: nil, hotwords: [], maxTokensPerChunk: nil,
            prefillStepSize: 2048, minP: 0, emit: nil)
    }
    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        context: String? = nil, hotwords: [String] = [], maxTokensPerChunk: Int? = nil,
        prefillStepSize: Int = 2048, minP: Float = 0
    ) -> STTOutput {
        generateImpl(
            audio, generationParameters ?? defaultGenerationParameters, context: context,
            hotwords: hotwords, maxTokensPerChunk: maxTokensPerChunk,
            prefillStepSize: prefillStepSize, minP: minP, emit: nil)
    }
    private func generateImpl(
        _ audio: MLXArray, _ p: STTGenerateParameters, context: String?, hotwords: [String],
        maxTokensPerChunk: Int?, prefillStepSize: Int, minP: Float, emit: ((String) -> Void)?
    ) -> STTOutput {
        let started = Date()
        let waveform = prepareAudio(audio)
        var promptTokens = 0
        var generationTokens = 0
        var text = ""
        var segments: [[String: Any]] = []
        if isStreamingModel {
            let state = makeStreamingState(context: context, hotwords: hotwords)
            let parameters = STTGenerateParameters(
                maxTokens: maxTokensPerChunk ?? min(p.maxTokens, 256), temperature: p.temperature,
                topP: p.topP, topK: p.topK, repetitionPenalty: p.repetitionPenalty,
                repetitionContextSize: p.repetitionContextSize, kvBits: p.kvBits,
                kvGroupSize: p.kvGroupSize, quantizedKVStart: p.quantizedKVStart)
            for start in Swift.stride(from: 0, to: waveform.dim(1), by: streamingChunkSamples) {
                if Task.isCancelled { break }
                let end = min(start + streamingWindowSamples, waveform.dim(1))
                let chunk = MLX.padded(
                    waveform[0..., start ..< end],
                    widths: [.init((0, 0)), .init((0, streamingWindowSamples - (end - start)))])
                let part = streamingGenerateStep(
                    audioFeatures: encodeSpeech(chunk), state: state,
                    generationParameters: parameters, minP: minP)
                text += part
                emit?(part)
                if p.verbose { print(part, terminator: "") }
                segments.append([
                    "text": part, "start": Double(start) / Double(sampleRate),
                    "end": Double(min(start + streamingChunkSamples, waveform.dim(1)))
                        / Double(sampleRate),
                ])
            }
            promptTokens = state.promptTokens
            generationTokens = state.generationTokens
        } else {
            let features = encodeSpeech(waveform)
            let ids = buildPrompt(
                frames: features.dim(1), duration: Double(waveform.dim(1)) / Double(sampleRate),
                context: Self.context(context, hotwords: hotwords))
            var cache: [KVCache] = (0 ..< config.decoderConfig.hiddenLayers).map { _ in
                KVCacheSimple()
            }
            var stop: Set<Int> = [151643, 151645]
            if let eos = tokenizer?.eosTokenId { stop.insert(eos) }
            let tokens = decode(
                mergeEmbeddings(ids, features: features), cache: &cache, parameters: p,
                stopIds: stop, prefillStepSize: prefillStepSize, minP: minP,
                repetitionTokens: ids.asArray(Int.self), emit: emit)
            text = tokenizer!.decode(tokens: tokens, skipSpecialTokens: true)
            promptTokens = ids.size
            generationTokens = tokens.count
            segments = Self.parseTranscription(text)
        }
        if p.verbose { print() }
        let elapsed = Date().timeIntervalSince(started)
        return STTOutput(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines), segments: segments,
            promptTokens: promptTokens, generationTokens: generationTokens,
            totalTokens: promptTokens + generationTokens,
            promptTps: Double(promptTokens) / max(elapsed, 0.001),
            generationTps: Double(generationTokens) / max(elapsed, 0.001),
            totalTime: elapsed, peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }
    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        makeStream(
            audio, generationParameters, context: nil, hotwords: [], maxTokensPerChunk: nil,
            prefillStepSize: 2048, minP: 0)
    }
    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        context: String? = nil, hotwords: [String] = [], maxTokensPerChunk: Int? = nil,
        prefillStepSize: Int = 2048, minP: Float = 0
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        makeStream(
            audio, generationParameters ?? defaultGenerationParameters, context: context,
            hotwords: hotwords, maxTokensPerChunk: maxTokensPerChunk,
            prefillStepSize: prefillStepSize, minP: minP)
    }
    private func makeStream(
        _ audio: MLXArray, _ parameters: STTGenerateParameters, context: String?,
        hotwords: [String], maxTokensPerChunk: Int?, prefillStepSize: Int, minP: Float
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = model.value.generateImpl(
                        input.value, parameters, context: context, hotwords: hotwords,
                        maxTokensPerChunk: maxTokensPerChunk, prefillStepSize: prefillStepSize,
                        minP: minP
                    ) { continuation.yield(.token($0)) }
                    try Task.checkCancellation()
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
    public static func parseTranscription(_ text: String) -> [[String: Any]] {
        var value: Any?
        // Try each JSON opener; JSONSerialization handles brackets inside quoted content.
        for i in text.indices where text[i] == "[" || text[i] == "{" {
            for j in text.indices.reversed() where j >= i && (text[j] == "]" || text[j] == "}") {
                if let json = try? JSONSerialization.jsonObject(with: Data(text[i ... j].utf8)) {
                    value = json
                    break
                }
            }
            if value != nil { break }
        }
        let items = value as? [[String: Any]] ?? (value as? [String: Any]).map { [$0] } ?? []
        let mapping = [
            "Start time": "start", "Start": "start", "End time": "end", "End": "end",
            "Speaker ID": "speaker_id", "Speaker": "speaker_id", "Content": "text",
        ]
        return items.compactMap { item in
            var segment: [String: Any] = [:]
            for (key, name) in mapping { if let value = item[key] { segment[name] = value } }
            return segment.isEmpty ? nil : segment
        }
    }
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let original = weights.keys.contains { $0.hasPrefix("model.") }
        var result: [String: MLXArray] = [:]
        for (key, var value) in weights {
            let path = canonicalVibeVoicePath(key)
            if path.contains("acoustic_tokenizer.decoder") || path.contains("position_ids")
                || path.contains("fix_std") || path.contains("rotary_emb.inv_freq")
            {
                continue
            }
            if original && path.contains("conv") && path.hasSuffix("weight") && value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
            result[path] = value
        }
        return result
    }
    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> VibeVoiceASRModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }
    public static func fromModelDirectory(_ directory: URL, tokenizerDirectory: URL? = nil)
        async throws -> VibeVoiceASRModel
    {
        var config = try JSONDecoder().decode(
            VibeVoiceConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.loadProcessor(directory)
        try config.validate()
        let model = VibeVoiceASRModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No VibeVoice safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights)
        if config.decoderConfig.tieWordEmbeddings {
            weights = weights.filter { !$0.key.hasPrefix("language_model.lm_head.") }
        }
        if let q = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights[path + ".scales"] != nil ? q.quantization(layer: path)?.asTuple : nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        if let tokenizerDirectory {
            model.tokenizer = try await loadSTTTokenizer(from: tokenizerDirectory)
        } else if FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path)
        {
            model.tokenizer = try await loadSTTTokenizer(from: directory)
        } else {
            model.tokenizer = try await AutoTokenizer.from(pretrained: "Qwen/Qwen2.5-7B")
        }
        try model.prepareTokenizer()
        model.train(false)
        eval(model)
        return model
    }
    func prepareTokenizer() throws {
        guard let tokenizer,
            let start = tokenizer.convertTokenToId("<|object_ref_start|>"),
            let end = tokenizer.convertTokenToId("<|object_ref_end|>"),
            let pad = tokenizer.convertTokenToId("<|box_start|>")
        else { throw STTError.modelNotInitialized("VibeVoice speech tokens are missing") }
        speechStartId = start
        speechEndId = end
        speechPadId = pad
        textChunkEndId = tokenizer.convertTokenToId("<|text_chunk_end|>")
        if textChunkEndId == tokenizer.unknownTokenId { textChunkEndId = nil }
        if config.chunkFrames != nil && textChunkEndId == nil {
            throw STTError.modelNotInitialized(
                "VibeVoice streaming tokenizer is missing <|text_chunk_end|>")
        }
    }
}
