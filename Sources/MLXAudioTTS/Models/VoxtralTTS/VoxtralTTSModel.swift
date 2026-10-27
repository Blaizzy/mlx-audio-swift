// Copyright (c) 2026, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXRandom
import MLXAudioCore

private final class VoxtralTTSVoices {
    var files: [String: URL] = [:]
    var embeddings: [String: MLXArray] = [:]
}

/// Multilingual Voxtral speech synthesis with preset voices and incremental codec decoding.
/// Use one model instance serially. Semantic tokens use argmax; acoustic sampling uses flow noise.
public final class VoxtralTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: VoxtralTTSConfiguration
    public var generationOptions = VoxtralTTSGenerationOptions()
    public var sampleRate: Int { configuration.sampleRate }
    public var defaultGenerationParameters: GenerateParameters { .init(maxTokens: 4096, temperature: 0) }
    public var availableVoices: [String] { voices.files.keys.sorted() }
    let backbone: VoxtralTTSBackbone
    @ModuleInfo(key: "acoustic_transformer") var acousticHead: VoxtralTTSAcousticHead
    @ModuleInfo(key: "audio_tokenizer") var audioDecoder: VoxtralTTSAudioDecoder
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    private let voices = VoxtralTTSVoices()
    var textProcessor: VoxtralTTSTextProcessor?

    public init(_ configuration: VoxtralTTSConfiguration) {
        self.configuration = configuration
        backbone = VoxtralTTSBackbone(configuration)
        _acousticHead.wrappedValue = VoxtralTTSAcousticHead(configuration)
        _audioDecoder.wrappedValue = VoxtralTTSAudioDecoder(configuration)
        _audioEmbeddings.wrappedValue = Embedding(embeddingCount: configuration.audioEmbeddingCount, dimensions: configuration.dim)
    }
    func registerVoice(_ name: String, file: URL) { voices.files[name] = file }
    func voiceEmbedding(_ name: String) throws -> MLXArray {
        if let cached = voices.embeddings[name] { return cached }
        guard let file = voices.files[name] else { throw AudioGenerationError.invalidInput("Missing Voxtral voice embedding: \(name)") }
        let arrays = try MLX.loadArrays(url: file)
        guard let value = arrays["embedding"] ?? (arrays.count == 1 ? arrays.values.first : nil),
              let expected = textProcessor?.voiceLengths[name], value.shape == [expected,configuration.dim] else {
            throw AudioGenerationError.invalidInput("Invalid Voxtral voice embedding shape for \(name)")
        }
        eval(value)
        guard isFinite(value).all().item(Bool.self) else { throw AudioGenerationError.invalidInput("Nonfinite Voxtral voice embedding: \(name)") }
        voices.embeddings[name] = value
        return value
    }
    func promptTokens(_ text: String, voice: String, options: VoxtralTTSGenerationOptions) throws -> [Int] {
        guard let textProcessor else { throw AudioGenerationError.modelNotInitialized("Voxtral speech tokenizer") }
        return try textProcessor.encode(text, voice: voice, sanitize: options.sanitizeText)
    }
    func inputEmbeddings(_ ids: [Int], voice: String) throws -> MLXArray {
        let tokens = MLXArray(ids).reshaped(1,-1)
        let base = backbone.embedTokens(tokens)
        let voice = try voiceEmbedding(voice).asType(base.dtype)
        let mask = tokens[0] .== configuration.audioID
        let indices = clip(cumsum(mask.asType(.int32)) - 1, min: 0, max: voice.dim(0) - 1)
        // Preserve the reference's BF16 subtraction/addition rounding when replacing placeholders.
        let delta = mask.reshaped(-1,1).asType(base.dtype) * (voice[indices] - base[0])
        return (base[0] + delta).expandedDimensions(axis: 0)
    }
    func globalIndices(_ codes: MLXArray) -> MLXArray {
        let c = configuration
        let offsets = [0] + (0..<c.acousticCodebooks).map { c.semanticSize + 2 + $0 * (c.acousticSize + 2) }
        return codes + MLXArray(offsets).asType(codes.dtype)
    }
    private func validate(_ parameters: GenerateParameters, options: VoxtralTTSGenerationOptions,
                          refAudio: MLXArray?, refText: String?, interval: Double?) throws {
        try configuration.validate()
        guard (parameters.maxTokens ?? 4096) > 0, options.denoisingSteps >= 2,
              options.guidanceScale.isFinite, options.guidanceScale >= 0,
              interval.map({ $0.isFinite && $0 > 0 }) ?? true else {
            throw AudioGenerationError.invalidInput("Voxtral requires positive maxTokens/streamingInterval, at least two denoising steps and finite nonnegative guidance")
        }
        guard refAudio == nil && refText == nil else { throw AudioGenerationError.invalidInput("Voxtral TTS uses preset voices; this checkpoint does not expose reference-audio conditioning") }
    }
    private func synthesize(text: String, voice: String, parameters: GenerateParameters,
                            options: VoxtralTTSGenerationOptions, interval: Double? = nil,
                            emit: ((sending MLXArray) -> Void)? = nil) throws -> sending MLXArray {
        try Task.checkCancellation()
        try validate(parameters, options: options, refAudio: nil, refText: nil, interval: interval)
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let ids = try promptTokens(text, voice: voice, options: options)
        guard ids.count < configuration.maxPositionEmbeddings else { throw AudioGenerationError.invalidInput("Voxtral prompt exceeds the model context") }
        let cacheConfiguration = try parameters.resolvedCacheConfiguration()
        var cache = try (0..<configuration.layers).map { _ in try makeAttentionKVCache(configuration: cacheConfiguration) }
        var hidden = backbone(try inputEmbeddings(ids, voice: voice), cache: cache)
        eval(hidden, cache)
        var frames: [MLXArray] = [], yielded = 0
        let requested = parameters.maxTokens ?? 4096
        let maximum = min(requested, configuration.maxPositionEmbeddings - ids.count)
        let framesPerChunk = interval.map { max(1, Int(min($0 * Double(configuration.frameRate), Double(Int.max / 2)))) } ?? Int.max
        for _ in 0..<maximum {
            try Task.checkCancellation()
            let codes = acousticHead.decodeFrame(hidden[0...,-1,0...], options: options)
            eval(codes)
            if codes[0,0].item(Int.self) <= 1 { break }
            frames.append(codes.expandedDimensions(axis: 1))
            let next = audioEmbeddings(globalIndices(codes)).sum(axis: 1, keepDims: true)
            hidden = backbone(next, cache: cache)
            eval(hidden, cache)
            if let cacheConfiguration { _ = try applyKVCacheConfiguration(cache: &cache, configuration: cacheConfiguration) }
            if interval != nil && frames.count - yielded >= framesPerChunk {
                let audio = decodeChunk(frames, from: yielded)
                yielded = frames.count
                emit?(audio)
            }
        }
        guard !frames.isEmpty else { throw AudioGenerationError.generationFailed("Voxtral generated no audio frames") }
        if yielded == frames.count { return .zeros([0]) }
        return decodeChunk(frames, from: yielded)
    }
    func decodeChunk(_ frames: [MLXArray], from start: Int) -> sending MLXArray {
        let contextStart = max(0, start - 16)
        let codes = concatenated(Array(frames[contextStart...]), axis: 1)
        let audio = audioDecoder.decode(codes).reshaped(-1)
        let trim = (start - contextStart) * configuration.samplesPerFrame
        let result = audio[trim...].asType(.float32)
        eval(result)
        // The evaluated waveform owns fresh output storage; it aliases neither model parameters nor codes.
        nonisolated(unsafe) let output = result
        return output
    }
    public func generate(text: String, voice: String = "casual_male", options: VoxtralTTSGenerationOptions = .init(),
                         generationParameters: GenerateParameters = .init(maxTokens: 4096, temperature: 0)) async throws -> sending MLXArray {
        try synthesize(text: text, voice: voice, parameters: generationParameters, options: options)
    }
    public func generate(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> sending MLXArray {
        try validate(generationParameters, options: generationOptions, refAudio: refAudio, refText: refText, interval: nil)
        return try synthesize(text: text, voice: voice ?? "casual_male", parameters: generationParameters, options: generationOptions)
    }
    public func generateStream(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
                       generationParameters: generationParameters, streamingInterval: 2)
    }
    public func generateStream(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters, streamingInterval: Double) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let reference = SendingBox(refAudio), options = generationOptions
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.validate(generationParameters, options: options, refAudio: reference.take(), refText: refText, interval: streamingInterval)
                    let audio = try self.synthesize(text: text, voice: voice ?? "casual_male", parameters: generationParameters,
                        options: options, interval: streamingInterval, emit: { continuation.yield(.audio($0)) })
                    if audio.size > 0 { continuation.yield(.audio(audio)) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
