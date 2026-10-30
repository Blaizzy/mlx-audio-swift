import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioCodecs
import MLXNN
import Tokenizers

/// OuteTTS 1.0 Qwen3 speech synthesis with portable speaker profiles.
public final class OuteTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    @ModuleInfo(key: "model") var backbone: Qwen3Model
    public let configuration: Qwen3Configuration
    public var sampleRate: Int { configuration.sampleRate }
    public var speaker: OuteTTSSpeaker?
    /// Maximum words per sentence group, matching the upstream default.
    public var maxChunkWords = 30

    var tokenizer: (any Tokenizer)?
    var tokenMap: OuteTTSTokenMap?
    // Keep the separately loaded codec outside the language-model parameter tree.
    private var codecBox: OuteTTSCodecBox?
    var codec: DescriptDAC? {
        get { codecBox?.model }
        set { codecBox = newValue.map(OuteTTSCodecBox.init) }
    }

    public init(_ configuration: Qwen3Configuration) {
        self.configuration = configuration
        _backbone.wrappedValue = Qwen3Model(configuration)
    }

    public var defaultGenerationParameters: GenerateParameters {
        .init(maxTokens: 1200, temperature: 0.4, topP: 0.9, topK: 40, minP: 0.05,
              repetitionPenalty: 1.1, repetitionContextSize: 64)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights where !key.contains("rotary_emb.inv_freq") {
            let name = key.hasPrefix("model.model.") || key.hasPrefix("model.lm_head.") ? key : "model." + key
            if configuration.tieWordEmbeddings && name.hasPrefix("model.lm_head.") { continue }
            result[name] = value
        }
        return result
    }

    /// Build a voice from mono 24 kHz audio and explicit word alignment.
    /// Transcription/alignment can be supplied by the caller's preferred system.
    public func createSpeaker(audio: MLXArray, text: String, words: [OuteTTSAlignedWord]) throws -> OuteTTSSpeaker {
        guard let codec else { throw AudioGenerationError.modelNotInitialized("OuteTTS DAC codec") }
        guard audio.ndim == 1, audio.size > 0, !words.isEmpty else {
            throw AudioGenerationError.invalidInput("Supply mono 24 kHz audio and word timestamps")
        }
        var previousEnd = 0.0
        for word in words {
            guard word.start.isFinite, word.end.isFinite, word.start >= previousEnd,
                  word.end > word.start, word.end <= Double(audio.size) / Double(sampleRate),
                  !word.word.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AudioGenerationError.invalidInput("OuteTTS word timestamps must be ordered and lie within the reference audio")
            }
            previousEnd = word.end
        }
        let normalized = try OuteTTSAudio.normalize(audio, sampleRate: sampleRate)
        // The reference interface encodes successive five-second windows.
        let window = Int(ceil(5 * Double(sampleRate) / Double(codec.hopLength))) * codec.hopLength
        var encoded = [MLXArray]()
        for start in stride(from: 0, to: normalized.size, by: window) {
            let input = normalized[start..<min(start + window, normalized.size)].reshaped(1, -1, 1)
            let codes = codec.encode(codec.preprocess(input), nQuantizers: 2).1
            eval(codes)
            encoded.append(codes)
        }
        let codes = concatenated(encoded, axis: -1)
        let c1 = codes[0, 0].asArray(Int32.self).map(Int.init)
        let c2 = codes[0, 1].asArray(Int32.self).map(Int.init)
        var start = max(0, Int(words[0].start * 75) - 20)
        var profileWords = [OuteTTSSpeaker.Word]()
        for (index, word) in words.enumerated() {
            let end = min(c1.count, Int(word.end * 75) + (index == words.count - 1 ? 20 : 0))
            guard end > start else {
                throw AudioGenerationError.invalidInput("OuteTTS aligned word has no codec frames")
            }
            let wordAudio = audio[Int(word.start * Double(sampleRate))..<Int(word.end * Double(sampleRate))]
            profileWords.append(.init(
                word: word.word.trimmingCharacters(in: .whitespacesAndNewlines),
                duration: (Double(end - start) / 75 * 100).rounded(.toNearestOrEven) / 100,
                c1: Array(c1[start..<end]), c2: Array(c2[start..<end]),
                features: OuteTTSAudio.features(wordAudio, sampleRate: sampleRate)))
            start = end
        }
        let profile = OuteTTSSpeaker(text: OuteTTSPrompt.normalize(text), words: profileWords,
                                   globalFeatures: OuteTTSAudio.features(audio, sampleRate: sampleRate))
        try profile.validate()
        return profile
    }

    private func resolveSpeaker(voice: String?, refAudio: MLXArray?, refText: String?) throws -> OuteTTSSpeaker? {
        if refAudio != nil || refText != nil {
            throw AudioGenerationError.invalidInput(
                "OuteTTS reference audio requires word alignment. Use createSpeaker(audio:text:words:), assign model.speaker, or pass a speaker JSON path as voice.")
        }
        if let voice {
            return try OuteTTSSpeaker.load(from: URL(fileURLWithPath: (voice as NSString).expandingTildeInPath))
        }
        return speaker
    }

    func decode(_ tokens: [Int], map: OuteTTSTokenMap, codec: DescriptDAC) throws -> MaterializedArray? {
        let codes = map.extract(tokens)
        guard !codes[0].isEmpty else { return nil }
        var chunks = [MLXArray]()
        for start in stride(from: 0, to: codes[0].count, by: 4096) {
            let end = min(start + 4096, codes[0].count)
            let tensor = MLXArray((Array(codes[0][start..<end]) + Array(codes[1][start..<end])).map(Int32.init))
                .reshaped(1, 2, end - start)
            let audio = codec.decodeFromCodes(tensor).flattened()
            eval(audio)
            chunks.append(audio)
        }
        return (try OuteTTSAudio.normalize(concatenated(chunks), sampleRate: sampleRate)).materialized()
    }

    private func synthesize(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        parameters: GenerateParameters, streamingInterval: Double?,
        emit: (AudioGeneration) -> Void
    ) throws {
        guard let tokenizer, let map = tokenMap, let codec else {
            throw AudioGenerationError.modelNotInitialized("OuteTTS tokenizer and DAC codec")
        }
        guard maxChunkWords > 0, (parameters.maxTokens ?? 1200) > 0,
              streamingInterval.map({ $0.isFinite && $0 > 0 && $0 <= Double(Int.max / 138) }) ?? true else {
            throw AudioGenerationError.invalidInput("OuteTTS token limit, word limit and streaming interval must be positive")
        }
        let chunks = OuteTTSPrompt.chunks(text, maxWords: maxChunkWords)
        guard !chunks.isEmpty else { throw AudioGenerationError.invalidInput("OuteTTS text is empty") }
        let speaker = try resolveSpeaker(voice: voice, refAudio: refAudio, refText: refText)
        let started = Date()
        var promptCount = 0, generatedCount = 0
        var prefillTime = 0.0
        let interval = streamingInterval.map { max(1, Int($0 * 137.5)) }
        let maskValues: [Float] = (0..<configuration.vocabularySize).map { map.forbidden.contains($0) ? -.infinity : 0 }
        let mask = MLXArray(maskValues)
        for text in chunks {
            try Task.checkCancellation()
            let prompt = try OuteTTSPrompt.completion(text: text, speaker: speaker)
            let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
            guard ids.count <= configuration.maxPositionEmbeddings,
                  (parameters.maxTokens ?? 1200) <= configuration.maxPositionEmbeddings - ids.count else {
                throw AudioGenerationError.invalidInput("OuteTTS prompt and token limit exceed the model context")
            }
            promptCount += ids.count
            let cacheConfiguration = try parameters.resolvedCacheConfiguration()
            var cache = try (0..<configuration.hiddenLayers).map { _ in
                try makeAttentionKVCache(configuration: cacheConfiguration)
            }
            let sampler = parameters.sampler()
            var processor = parameters.processor()
            processor?.prompt(MLXArray(ids.map(Int32.init)))
            var input = MLXArray(ids.map(Int32.init)).reshaped(1, -1)
            var output = [Int]()
            var yieldedSamples = 0
            for index in 0..<(parameters.maxTokens ?? 1200) {
                try Task.checkCancellation()
                let prefillStarted = Date()
                let logits = backbone(input, cache: cache)[0..., -1, 0...]
                let processed = processor?.process(logits: logits) ?? logits
                let token = sampler.sample(logits: processed + mask)
                eval(token)
                if index == 0 { prefillTime += Date().timeIntervalSince(prefillStarted) }
                let id = token.item(Int.self)
                if map.stop.contains(id) { break }
                output.append(id)
                generatedCount += 1
                processor?.didSample(token: token)
                emit(.token(id))
                input = token.reshaped(1, 1)
                if let cacheConfiguration {
                    _ = try applyKVCacheConfiguration(cache: &cache, configuration: cacheConfiguration)
                }
                if let interval, output.count.isMultiple(of: interval),
                   let audio = try decode(output, map: map, codec: codec), audio.size > yieldedSamples {
                    let suffix = audio[yieldedSamples...]
                    yieldedSamples = audio.size
                    eval(suffix)
                    emit(.audio(suffix))
                }
            }
            guard let audio = try decode(output, map: map, codec: codec) else {
                throw AudioGenerationError.generationFailed("OuteTTS produced no complete DAC frames; increase maxTokens")
            }
            if audio.size > yieldedSamples {
                let suffix = audio[yieldedSamples...]
                eval(suffix)
                emit(.audio(suffix))
            }
        }
        let elapsed = Date().timeIntervalSince(started)
        emit(.info(.init(promptTokenCount: promptCount, generationTokenCount: generatedCount,
                         prefillTime: prefillTime, generateTime: max(0, elapsed - prefillTime),
                         tokensPerSecond: Double(generatedCount) / max(elapsed - prefillTime, 0.001),
                         peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
    }

    public func generate(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        var samples = [MLXArray]()
        try synthesize(text: text, voice: voice, refAudio: refAudio, refText: refText,
                       parameters: generationParameters, streamingInterval: nil) {
            if case .audio(let audio) = $0 { samples.append(audio) }
        }
        return (concatenated(samples)).materialized()
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText,
                       language: language, generationParameters: generationParameters, streamingInterval: 2)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters, streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    try self.synthesize(text: text, voice: voice, refAudio: reference, refText: refText,
                                        parameters: generationParameters, streamingInterval: streamingInterval) {
                        continuation.yield($0)
                    }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public static func fromPretrained(_ modelRepo: String, hfToken: String? = nil, cache: HubCache = .default) async throws -> OuteTTSModel {
        guard let id = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: id, requiredExtension: "safetensors", hfToken: hfToken, cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }

    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default) async throws -> OuteTTSModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "qwen3" || base.modelType == "outetts" else {
            throw AudioGenerationError.invalidInput("This OuteTTS loader supports Qwen3-based OuteTTS 1.0 checkpoints")
        }
        let configuration = try JSONDecoder().decode(Qwen3Configuration.self, from: data)
        guard configuration.sampleRate == 24_000 else {
            throw AudioGenerationError.invalidInput("OuteTTS speech DAC requires a 24 kHz checkpoint")
        }
        let model = OuteTTSModel(configuration)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        var weights = [String: MLXArray]()
        for file in files.filter({ $0.pathExtension == "safetensors" }) {
            weights.merge(try loadArrays(url: file)) { _, new in new }
        }
        let sanitized = model.sanitize(weights: weights)
        if let quantization = base.perLayerQuantization {
            quantize(model: model) { path, _ in
                guard sanitized[path + ".scales"] != nil else { return nil }
                return quantization.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        model.tokenMap = try OuteTTSTokenMap(tokenizer: model.tokenizer!, vocabularySize: configuration.vocabularySize)
        let localCodec = directory.appendingPathComponent("audio_codec")
        if FileManager.default.fileExists(atPath: localCodec.appendingPathComponent("config.json").path) {
            model.codec = try DescriptDAC.fromModelDirectory(localCodec)
        } else {
            model.codec = try await DescriptDAC.fromPretrained("mlx-community/dac-speech-24khz-1.5kbps", cache: cache)
        }
        guard model.codec?.sampleRate == 24_000, model.codec?.config.nCodebooks == 2,
              model.codec?.config.codebookSize == 1024, model.codec?.hopLength == 320 else {
            throw AudioGenerationError.invalidInput("OuteTTS requires the two-codebook 24 kHz speech DAC")
        }
        let localSpeaker = directory.appendingPathComponent("default_speaker.json")
        let speakerURL = FileManager.default.fileExists(atPath: localSpeaker.path)
            ? localSpeaker : Bundle.module.url(forResource: "outetts_default_speaker", withExtension: "json")!
        model.speaker = try OuteTTSSpeaker.load(from: speakerURL)
        model.train(false)
        return model
    }
}

private final class OuteTTSCodecBox {
    let model: DescriptDAC
    init(_ model: DescriptDAC) { self.model = model }
}
