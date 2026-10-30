// Copyright 2026 MLX Audio contributors.
// Ming architecture adapted from inclusionAI and mlx-audio.
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
@preconcurrency import MLXAudioCore
import MLXNN
import Tokenizers

final class MingTinyLanguageModel: Module {
    let model: MingTinyBackbone
    @ModuleInfo(key: "lm_head") var head: Linear
    init(_ config: MingTinyTextConfiguration) {
        model = MingTinyBackbone(config)
        _head.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }
}

/// Ming-3.0 Tiny's MoE speech generator. Model instances require serialized use.
public final class MingTinyModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: MingTinyConfiguration
    public var sampleRate: Int { config.audio.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: config.speech.maximumDecodeSteps, temperature: 0)
    }
    let model: MingTinyLanguageModel
    let audio: MingAudioVAE
    @ModuleInfo(key: "linear_proj_audio") var aggregator: MingLatentAggregator
    @ModuleInfo(key: "flowloss") var flow: MingTinyFlow
    @ModuleInfo(key: "spk_head") var speakerHead: Linear
    @ModuleInfo(key: "stop_head") var stopHead: Linear
    var tokenizer: (any Tokenizers.Tokenizer)?
    private var speakerEncoder: MingCAMPPlus?
    var directory: URL?
    public private(set) var lastDecodeSteps = 0

    public init(config: MingTinyConfiguration) throws {
        try config.validate()
        self.config = config
        model = MingTinyLanguageModel(config.text)
        audio = MingAudioVAE(config.audio)
        _aggregator.wrappedValue = MingLatentAggregator(latentDim: config.speech.latentDim,
            outputDim: config.text.hiddenSize, config: config.aggregator)
        _flow.wrappedValue = MingTinyFlow(latentDim: config.speech.latentDim,
            conditionDim: config.text.hiddenSize, config: config.flow)
        _speakerHead.wrappedValue = Linear(192, config.text.hiddenSize)
        _stopHead.wrappedValue = Linear(config.text.hiddenSize, 2)
    }

    func speakerEmbeddings(references: [MLXArray], embeddings: MLXArray?) throws -> MLXArray {
        let embedding: MLXArray
        if let embeddings {
            guard embeddings.size > 0, embeddings.size % 192 == 0 else {
                throw MingTinyError.invalidInput("Speaker embeddings require 192 values per speaker")
            }
            embedding = embeddings.asType(.float32).reshaped(-1, 192)
        } else {
            guard !references.isEmpty else { throw MingTinyError.invalidInput("A speaker reference is required") }
            if speakerEncoder == nil {
                guard let directory else { throw MingTinyError.missingComponent("Speaker weights are unavailable") }
                let file = directory.appendingPathComponent("speaker_encoder/campplus.safetensors")
                guard FileManager.default.fileExists(atPath: file.path) else {
                    throw MingTinyError.missingComponent("Missing speaker_encoder/campplus.safetensors")
                }
                speakerEncoder = try MingCAMPPlus.fromWeights(MLX.loadArrays(url: file))
            }
            embedding = try concatenated(references.map { reference in
                let samples = try resampleAudio(reference.asArray(Float.self), from: sampleRate, to: 16000)
                guard samples.count >= 720 else { throw MingTinyError.invalidInput("Speaker references require at least 45 ms") }
                let features = MingCAMPPlus.features(MLXArray(samples))
                return speakerEncoder!(features.expandedDimensions(axis: 0))
            }, axis: 0)
        }
        guard all(isFinite(embedding)).item(Bool.self) else {
            throw MingTinyError.invalidInput("Speaker embeddings must be finite")
        }
        // Keep CAMPPlus FP32 precision until the checkpoint's speaker projection.
        return speakerHead(embedding.asType(speakerHead.weight.dtype))
    }

    func prepareInputs(text: String, task: MingTinyTask, instruction: String?, language: String?,
                       references: [MLXArray], transcripts: [String], speakerEmbeddings: MLXArray?) throws
        -> (ids: [Int], embeddings: MLXArray, latents: MLXArray?) {
        guard let tokenizer else { throw MingTinyError.missingComponent("Text tokenizer is not loaded") }
        if task.usesContinuation {
            guard !references.isEmpty, transcripts.count == references.count,
                  transcripts.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
                throw MingTinyError.invalidInput("Continuation requires one transcript per reference waveform")
            }
        } else if !transcripts.isEmpty { throw MingTinyError.invalidInput("This task does not accept reference transcripts") }
        if [.voiceDesign, .ip].contains(task), !references.isEmpty || speakerEmbeddings != nil {
            throw MingTinyError.invalidInput("This task uses an empty speaker slot")
        }
        if task != .podcast, references.count > 1 {
            throw MingTinyError.invalidInput("Only podcast accepts multiple references")
        }
        if task.requiresSpeaker, references.isEmpty && speakerEmbeddings == nil {
            throw MingTinyError.invalidInput("This task requires a speaker reference or speaker embeddings")
        }
        if task == .podcast { try MingTinyPrompt.validatePodcast(text, count: references.count) }
        for reference in references {
            guard reference.ndim == 1, reference.size > 0, all(isFinite(reference)).item(Bool.self) else {
                throw MingTinyError.invalidInput("References must be finite nonempty mono audio at \(sampleRate) Hz")
            }
        }
        let count = task == .podcast ? references.count : 1
        let masked = [.voiceDesign, .ip, .freeform].contains(task) && references.isEmpty && speakerEmbeddings == nil
        let speakers = try masked ? MLXArray.zeros([1, config.text.hiddenSize], dtype: speakerHead.weight.dtype)
            : self.speakerEmbeddings(references: references, embeddings: speakerEmbeddings)
        guard speakers.dim(0) == count else { throw MingTinyError.invalidInput("Speaker embedding count does not match the references") }
        var promptLatents: MLXArray?, promptHidden: MLXArray?
        if task.usesContinuation {
            var waveform = concatenated(references).expandedDimensions(axis: 0)
            let alignment = config.audio.encoder.hopSize * config.audio.patchSize * config.speech.patchSize
            let padding = (alignment - waveform.dim(1) % alignment) % alignment
            waveform = MLX.padded(waveform, widths: [.init(0), .init((0, padding))])
            let latents = audio.encode(waveform)
            promptLatents = latents
            promptHidden = aggregator(latents.reshaped(-1, config.speech.patchSize, config.speech.latentDim))
                .reshaped(1, -1, config.text.hiddenSize)
        }
        let transcript = task == .podcast ? transcripts.enumerated().map { " speaker_\($0.offset + 1):\($0.element)\n" }.joined() : transcripts.first
        let prompt = MingTinyPrompt.build(encode: { tokenizer.encode(text: $0, addSpecialTokens: false) },
            speech: config.speech, text: text, instruction: instruction, language: language,
            speakers: count, promptText: transcript, promptSteps: promptHidden?.dim(1) ?? 0)
        let embeddings = model.model.embeddings(MLXArray(prompt.ids).reshaped(1, -1))
        for (index, position) in prompt.positions.enumerated() {
            embeddings[0, position] = speakers[index].asType(embeddings.dtype)
        }
        if let promptHidden {
            embeddings[0..., (embeddings.dim(1) - promptHidden.dim(1))..., 0...] = promptHidden.asType(embeddings.dtype)
        }
        return (prompt.ids, embeddings, promptLatents)
    }

    func generateWaveform(text: String, voice: String?, references: [MLXArray], transcripts: [String],
                          language: String?, speakerEmbeddings: MLXArray?, options o: MingTinyOptions,
                          parameters: GenerateParameters) throws -> MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw MingTinyError.invalidInput("Text must be nonempty") }
        guard o.cfgScale.isFinite, o.sigma.isFinite, parameters.temperature.isFinite,
              o.cfgScale >= 0, o.sigma >= 0, parameters.temperature >= 0, o.diffusionSteps > 0 else {
            throw MingTinyError.invalidInput("Flow parameters must be finite and nonnegative, and diffusion steps positive")
        }
        let task = o.task ?? (!references.isEmpty && !transcripts.isEmpty
            ? (["en", "english", "zh", "chinese"].contains((language ?? "en").lowercased()) ? .seed : .multilingual)
            : o.style != nil ? .voiceDesign : o.instruction != nil ? .freeform
            : o.emotion != nil ? .emotion : o.dialect != nil ? .dialect : .ip)
        let instruction = try MingTinyPrompt.instruction(task: task, voice: voice ?? "US_James", options: o)
        var targetLanguage: String?
        if task.usesContinuation {
            targetLanguage = language.flatMap { MingTinyPrompt.languages[$0.lowercased()] }
                ?? (task == .podcast && language == nil ? (text.range(of: "[一-鿿]", options: .regularExpression) != nil ? "Chinese" : "English") : nil)
            if language == nil && task != .podcast { targetLanguage = "English" }
            let allowed = task == .multilingual ? ["German", "Spanish", "French", "Korean", "Japanese"] : ["Chinese", "English"]
            guard task == .podcast || targetLanguage.map({ allowed.contains($0) }) == true else {
                throw MingTinyError.invalidInput("This task requires a supported target language")
            }
        }
        let limit = parameters.maxTokens ?? config.speech.maximumDecodeSteps
        let minimum = o.minimumDecodeSteps ?? config.speech.minimumDecodeSteps
        let threshold = o.stopThreshold ?? config.speech.stopThreshold
        guard limit > 0, minimum >= 0, minimum <= limit, threshold.isFinite, (0...1).contains(threshold) else {
            throw MingTinyError.invalidInput("Invalid decode limits or stop threshold")
        }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let prepared = try prepareInputs(text: text, task: task, instruction: instruction,
            language: targetLanguage, references: references, transcripts: transcripts, speakerEmbeddings: speakerEmbeddings)
        guard prepared.ids.count + limit <= config.text.maxPositionEmbeddings else {
            throw MingTinyError.invalidInput("Prompt and generation exceed the Ming Tiny context budget")
        }
        var embeddings = prepared.embeddings
        let cache = model.model.makeCache(), s = config.speech
        var history = MLXArray.zeros([1, s.historySize, s.latentDim], dtype: .float32)
        if let latents = prepared.latents {
            let available = min(s.historySize, latents.dim(1))
            history[0..., (s.historySize - available)..., 0...] = latents[0..., (latents.dim(1) - available)..., 0...].asType(.float32)
        }
        var latents: [MLXArray] = []
        for _ in 0..<limit {
            try Task.checkCancellation()
            let output = model.model(embeddings, cache: cache)
            let hidden = output[0..., (output.dim(1) - 1)..., 0...]
            let sampled = flow.sample(hidden, history: history, patchSize: s.patchSize, steps: o.diffusionSteps,
                cfg: o.cfgScale, sigma: o.sigma, temperature: parameters.temperature)
            latents.append(sampled)
            let probability = softmax(stopHead(hidden).asType(.float32), axis: -1)[0, 0, 1]
            if latents.count >= minimum, probability.item(Float.self) > threshold { break }
            embeddings = aggregator(sampled.asType(hidden.dtype))
            history = concatenated([history[0..., s.patchSize..., 0...], sampled], axis: 1)
            eval(embeddings, history, cache.flatMap(\.arrays))
        }
        lastDecodeSteps = latents.count
        let waveform = audio.decode(concatenated(latents, axis: 1).asType(audio.decoder.fc1.weight.dtype))[0].asType(.float32)
        eval(waveform)
        guard waveform.size > 0, all(isFinite(waveform)).item(Bool.self) else {
            throw MingTinyError.invalidInput("Ming Tiny produced invalid audio")
        }
        return waveform
    }

    public func generate(text: String, voice: String? = nil, refAudio: MaterializedArray? = nil,
                         refText: String? = nil, language: String? = nil,
                         generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try (generateWaveform(text: text, voice: voice, references: refAudio.map { [$0] } ?? [],
            transcripts: refText.map { [$0] } ?? [], language: language, speakerEmbeddings: nil,
            options: .init(), parameters: generationParameters)).materialized()
    }

    /// References contain mono waveforms at `sampleRate`. Podcast uses one transcript per speaker.
    public func generate(text: String, voice: String? = nil, references: [MaterializedArray] = [],
                         transcripts: [String] = [], language: String? = nil,
                         speakerEmbeddings: MaterializedArray? = nil, options: MingTinyOptions,
                         generationParameters: GenerateParameters? = nil) async throws -> MaterializedArray {
        try generateWaveform(text: text, voice: voice, references: references, transcripts: transcripts,
            language: language, speakerEmbeddings: speakerEmbeddings, options: options,
            parameters: generationParameters ?? defaultGenerationParameters).materialized()
    }

    /// Ming Tiny synthesizes offline; this stream yields one completed audio event.
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters)
        -> AsyncThrowingStream<AudioGeneration, Error> {
        let box = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    let waveform = try await generate(text: text, voice: voice, refAudio: box,
                        refText: refText, language: language, generationParameters: generationParameters)
                    continuation.yield(.audio(waveform)); continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
