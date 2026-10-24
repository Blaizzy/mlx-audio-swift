import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

final class MingAudioProjector: Module {
    @ModuleInfo var convolutions: [Conv1d]
    @ModuleInfo var projection: Linear
    init(_ config: MingASRConfig) {
        let a = config.audioConfig
        let hidden = config.textConfig.hiddenSize
        var width = a.hiddenSize
        var convs: [Conv1d] = []
        for i in a.downsampleKernels.indices {
            convs.append(
                Conv1d(
                    inputChannels: width, outputChannels: hidden,
                    kernelSize: a.downsampleKernels[i], stride: a.downsampleStrides[i],
                    padding: a.paddings[i]))
            width = hidden
        }
        _convolutions.wrappedValue = convs
        _projection.wrappedValue = Linear(hidden, hidden)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var h = input
        for conv in convolutions { h = gelu(conv(h)) }
        return projection(h)
    }
}

/// Chinese and English speech recognition with Ming-3.0 ASR Tiny.
public final class MingASRModel: Module, STTGenerationModel {
    public let config: MingASRConfig
    public static let transcriptionInstruction =
        "Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe."
    @ModuleInfo var model: MingBackbone
    @ModuleInfo(key: "linear_proj_audio") var audioProjector: MingAudioProjector
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    var tokenizer: Tokenizer?
    var audioMarkers = (start: "<audio>", end: "</audio>")
    public var defaultGenerationParameters: STTGenerateParameters { .init(maxTokens: 256, topP: 1) }

    public init(_ config: MingASRConfig) {
        self.config = config
        _model.wrappedValue = MingBackbone(config)
        let hidden = config.textConfig.hiddenSize
        _audioProjector.wrappedValue = MingAudioProjector(config)
        if !config.textConfig.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(hidden, config.textConfig.vocabSize, bias: false)
        }
    }

    func extractFeatures(_ audio: MLXArray) -> MLXArray {
        let mono = audio.ndim == 2 ? audio.mean(axis: -1) : audio
        precondition(
            mono.ndim == 1 && mono.size > 200, "Ming expects more than 200 samples at 16 kHz")
        return WhisperAudio.logMelSpectrogram(mono, nMels: config.audioConfig.numMelBins)
            .T.asType(.bfloat16)
    }

    func encodeAudio(_ features: MLXArray, chunkFrames: Int? = nil) -> MLXArray {
        let a = config.audioConfig
        let encoder = model.audio
        let chunk = chunkFrames ?? a.chunkFrames
        precondition(chunk > 0 && chunk <= 2 * a.maxSourcePositions)
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: features.dim(0), by: chunk) {
            let input = features[start ..< min(start + chunk, features.dim(0))].expandedDimensions(
                axis: 0
            )
            .asType(encoder.conv1.weight.dtype)
            var h = gelu(encoder.conv2(gelu(encoder.conv1(input))))
            h = (h + encoder.embedPositions.weight[0 ..< h.dim(1)]).asType(h.dtype)
            for layer in encoder.layers { h = layer(h) }
            h = encoder.layerNorm(h)
            h = audioProjector(h)
            if config.normalizeAudioEmbeddings {
                let norm = sqrt(h.asType(.float32).square().sum(axis: -1, keepDims: true))
                h = (h / maximum(norm, 1e-12)).asType(encoder.conv1.weight.dtype)
            }
            eval(h)
            parts.append(h)
        }
        return concatenated(parts, axis: 1)
    }

    static func languageLabel(_ language: String?) -> String? {
        guard let language, !language.isEmpty else { return nil }
        switch language.lowercased() {
        case "en", "english": return "English"
        case "zh", "chinese": return "Chinese"
        default: preconditionFailure("Ming language hints support en or zh")
        }
    }

    func buildPrompt(audioTokens: Int, language: String?, prompt: String?, hotwords: [String])
        -> [Int]
    {
        guard let tokenizer else { preconditionFailure("Ming tokenizer is not loaded") }
        let instruction = prompt ?? Self.transcriptionInstruction
        precondition(
            instruction.hasPrefix(Self.transcriptionInstruction)
                && !instruction.contains("\n") && !instruction.contains("\r"),
            "Ming prompts must begin with the transcription instruction and contain no newlines")
        let text =
            instruction
            + (hotwords.isEmpty ? "" : " Hotwords: " + hotwords.joined(separator: ", ") + ".")
        var ids = tokenizer.encode(text: text + audioMarkers.start, addSpecialTokens: false)
        ids += Array(repeating: config.audioTokenId, count: audioTokens)
        ids += tokenizer.encode(text: audioMarkers.end, addSpecialTokens: false)
        if let label = Self.languageLabel(language) {
            ids += tokenizer.encode(text: label + "\t", addSpecialTokens: false)
        }
        return ids
    }

    func injectAudio(_ ids: [Int], audio: MLXArray) -> MLXArray {
        let input = MLXArray(ids.map(Int32.init))
        let isAudio = input .== config.audioTokenId
        let text = model.embeddings(input.expandedDimensions(axis: 0))
        let indices = clip(cumsum(isAudio.asType(.int32)) - 1, min: 0, max: audio.dim(1) - 1)
        let values = audio[0, indices, 0...].asType(text.dtype).expandedDimensions(axis: 0)
        return MLX.where(isAudio.expandedDimensions(axes: [0, 2]), values, text)
    }

    static func decodeTranscript(_ raw: String) -> (text: String, language: String?) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let tab = text.firstIndex(of: "\t") {
            return (
                String(text[text.index(after: tab)...]).trimmingCharacters(
                    in: .whitespacesAndNewlines),
                String(text[..<tab]).trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        for language in ["Chinese", "English", "Canton", "Japanese", "Korean"]
        where text.hasPrefix(language) {
            return (
                String(text.dropFirst(language.count)).trimmingCharacters(
                    in: .whitespacesAndNewlines), language
            )
        }
        return (text, nil)
    }

    static func bannedTokens(_ tokens: [Int], ngramSize n: Int) -> [Int] {
        guard n > 0, tokens.count >= n else { return [] }
        let prefix = n > 1 ? Array(tokens.suffix(n - 1)) : []
        return Array(
            Set(
                (0 ... (tokens.count - n)).compactMap { i in
                    Array(tokens[i ..< (i + n - 1)]) == prefix ? tokens[i + n - 1] : nil
                })
        ).sorted()
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generateImpl(
            audio: audio, parameters: generationParameters, prompt: nil, hotwords: [],
            audioChunkFrames: nil, noRepeatNgramSize: 10, emit: nil)
    }

    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil, prompt: String? = nil,
        hotwords: [String] = [], audioChunkFrames: Int? = nil, noRepeatNgramSize: Int = 10
    ) -> STTOutput {
        generateImpl(
            audio: audio, parameters: generationParameters ?? defaultGenerationParameters,
            prompt: prompt, hotwords: hotwords, audioChunkFrames: audioChunkFrames,
            noRepeatNgramSize: noRepeatNgramSize, emit: nil)
    }

    private func generateImpl(
        audio: MLXArray, parameters p: STTGenerateParameters, prompt: String?, hotwords: [String],
        audioChunkFrames: Int?, noRepeatNgramSize: Int, emit: ((String) -> Void)?
    ) -> STTOutput {
        guard let tokenizer else { preconditionFailure("Ming tokenizer is not loaded") }
        precondition(p.maxTokens >= 0 && p.temperature >= 0 && p.repetitionPenalty > 0)
        let started = Date()
        let encoded = encodeAudio(extractFeatures(audio), chunkFrames: audioChunkFrames)
        let ids = buildPrompt(
            audioTokens: encoded.dim(1), language: p.language, prompt: prompt, hotwords: hotwords)
        let available = config.textConfig.maxPositionEmbeddings - ids.count
        precondition(available > 0, "Audio prompt exceeds the Ming context budget")
        let embeddings = injectAudio(ids, audio: encoded)
        let cache = model.makeCache()
        let mlaIndices =
            config.textConfig.numHiddenLayers > 0
            ? (0 ..< config.textConfig.numHiddenLayers).filter { config.textConfig.usesMLA($0) }
            : []
        var attentionCaches = mlaIndices.map { cache[$0].attention }
        let cacheConfiguration = p.kvBits.map { bits -> KVCacheConfiguration in
            guard
                let affine = try? AffineKVCacheConfiguration(
                    bits: bits, groupSize: p.kvGroupSize,
                    compressionStart: p.quantizedKVStart)
            else { preconditionFailure("Invalid Ming KV-cache settings") }
            return .init(strategy: .affine(affine))
        }
        func finishStep(_ hidden: MLXArray) {
            eval([hidden] + cache.flatMap { $0.arrays })
            if let cacheConfiguration {
                do {
                    _ = try applyKVCacheConfiguration(
                        cache: &attentionCaches, configuration: cacheConfiguration)
                } catch { preconditionFailure("Unsupported Ming KV-cache settings: \(error)") }
                for (i, index) in mlaIndices.enumerated() {
                    cache[index].attention = attentionCaches[i]
                }
            }
        }
        var hidden: MLXArray!
        for start in stride(from: 0, to: ids.count, by: 256) {
            if Task.isCancelled { break }
            hidden = model(
                embeddings[0..., start ..< min(start + 256, ids.count), 0...], cache: cache)
            finishStep(hidden)
        }
        let prefillTime = Date().timeIntervalSince(started)
        let decodeStart = Date()
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK)
        var eos = Set(config.textConfig.eosTokenIds)
        if let id = tokenizer.eosTokenId { eos.insert(id) }
        if let id = tokenizer.convertTokenToId("<|role_end|>") { eos.insert(id) }
        var tokens: [Int] = []
        var emitted = ""
        func emitText(_ text: String) {
            guard !text.contains("\u{FFFD}"), text.hasPrefix(emitted) else { return }
            let delta = String(text.dropFirst(emitted.count))
            if !delta.isEmpty {
                emit?(delta)
                if p.verbose { print(delta, terminator: "") }
            }
            emitted = text
        }
        if let initial = hidden {
            hidden = initial
            for _ in 0 ..< min(p.maxTokens, available) {
                if Task.isCancelled { break }
                let scores =
                    (config.textConfig.tieWordEmbeddings
                    ? model.embeddings.asLinear(hidden[0..., -1, 0...])
                    : lmHead!(hidden[0..., -1, 0...]))
                    .asType(.float32).squeezed(axis: 0)
                let banned = Self.bannedTokens(tokens, ngramSize: noRepeatNgramSize)
                if !banned.isEmpty {
                    scores[MLXArray(banned.map(Int32.init))] = MLXArray(-Float.infinity)
                }
                if p.repetitionPenalty != 1, !tokens.isEmpty {
                    let indices = MLXArray(
                        Array(Set(tokens.suffix(max(1, p.repetitionContextSize)))).map(Int32.init))
                    let values = scores[indices]
                    scores[indices] = MLX.where(
                        values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
                }
                let token = sampler.sample(logits: scores).item(Int.self)
                if eos.contains(token) { break }
                tokens.append(token)
                if emit != nil || p.verbose {
                    let raw = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                    // Hold back an incomplete language label during automatic detection.
                    if p.language?.isEmpty == false || raw.contains("\t") {
                        emitText(Self.decodeTranscript(raw).text)
                    }
                }
                hidden = model(
                    model.embeddings(MLXArray([Int32(token)]).reshaped(1, 1)), cache: cache)
                finishStep(hidden)
            }
        }
        let decoded = Self.decodeTranscript(
            tokenizer.decode(tokens: tokens, skipSpecialTokens: true))
        emitText(decoded.text)
        if p.verbose { print() }
        let decodeTime = Date().timeIntervalSince(decodeStart)
        return STTOutput(
            text: decoded.text, language: decoded.language ?? p.language,
            promptTokens: ids.count, generationTokens: tokens.count,
            totalTokens: ids.count + tokens.count,
            promptTps: Double(ids.count) / max(prefillTime, 1e-6),
            generationTps: Double(tokens.count) / max(decodeTime, 1e-6),
            totalTime: Date().timeIntervalSince(started),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        generateStream(audio: audio, generationParameters: Optional(generationParameters))
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil, prompt: String? = nil,
        hotwords: [String] = [], audioChunkFrames: Int? = nil, noRepeatNgramSize: Int = 10
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let output = model.value.generateImpl(
                        audio: input.value,
                        parameters: generationParameters ?? model.value.defaultGenerationParameters,
                        prompt: prompt, hotwords: hotwords, audioChunkFrames: audioChunkFrames,
                        noRepeatNgramSize: noRepeatNgramSize
                    ) { continuation.yield(.token($0)) }
                    try Task.checkCancellation()
                    continuation.yield(.result(output))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
