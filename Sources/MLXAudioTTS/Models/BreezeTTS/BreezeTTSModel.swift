import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

public final class BreezeTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: BreezeTTSConfig
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "embed_text_tokens") var embedTextTokens: Embedding
    @ModuleInfo(key: "backbone_model") var backboneModel: BreezeBackbone
    @ModuleInfo(key: "depth_decoder") var depthDecoder: BreezeDepthDecoder
    @ModuleInfo(key: "text_encoder") var textEncoder: BreezeTTSTextEncoder
    @ModuleInfo(key: "text_encoder_proj") var textEncoderProjection: Linear

    var tokenizer: Tokenizers.Tokenizer?
    var audioTokenizer: Qwen3TTSSpeechTokenizer?

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 750,
            temperature: 0.9,
            topP: 1,
            topK: 50,
            repetitionPenalty: 1
        )
    }

    init(config: BreezeTTSConfig) {
        self.config = config
        _lmHead.wrappedValue = Linear(
            config.backboneConfig.hiddenSize,
            config.audioVocabSize + 1,
            bias: false
        )
        _embedTextTokens.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize,
            dimensions: config.backboneConfig.hiddenSize
        )
        _backboneModel.wrappedValue = BreezeBackbone(config: config)
        _depthDecoder.wrappedValue = BreezeDepthDecoder(config: config.depthDecoderConfig)
        _textEncoder.wrappedValue = BreezeTTSTextEncoder(config: config.textEncoderConfig)
        _textEncoderProjection.wrappedValue = Linear(
            config.textEncoderConfig.hiddenSize,
            config.backboneConfig.hiddenSize,
            bias: false
        )
    }

    static func promptText(text: String, instruction: String?) -> String {
        guard let instruction, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "[S0]\(text)"
        }
        return "[S0]<ins_bos>\(instruction)<ins_eos>\(text)"
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { key, _ in
            !key.hasPrefix("codec_model.")
                && !key.hasSuffix(".initialized")
                && !key.contains("rotary_emb.inv_freq")
        }
        let depthKey = "depth_decoder.model.embed_tokens.weight"
        let backboneKey = "backbone_model.embed_tokens.embed_audio_tokens.weight"
        if let tied = result[depthKey] {
            result[backboneKey] = tied
        }
        return result
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = language
        return try generateAudio(
            text: text,
            instruction: voice,
            refAudio: refAudio,
            refText: refText,
            parameters: generationParameters
        ).audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = language
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    let result = try generateAudio(
                        text: text,
                        instruction: voice,
                        refAudio: refAudio,
                        refText: refText,
                        parameters: generationParameters,
                        onToken: { continuation.yield(.token($0)) }
                    )
                    continuation.yield(.info(result.info))
                    continuation.yield(.audio(result.audio))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private struct GenerationResult {
        let audio: MLXArray
        let info: AudioGenerationInfo
    }

    private func generateAudio(
        text: String,
        instruction: String?,
        refAudio: MLXArray?,
        refText: String?,
        parameters: GenerateParameters,
        onToken: ((Int) -> Void)? = nil
    ) throws -> GenerationResult {
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Breeze text tokenizer is not loaded")
        }
        guard let audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Breeze audio tokenizer is not loaded")
        }
        if refAudio != nil && (refText?.isEmpty != false) {
            throw AudioGenerationError.invalidInput("Breeze voice cloning needs a reference transcript")
        }
        let maxTokens = parameters.maxTokens ?? 750
        guard maxTokens > 0 else {
            throw AudioGenerationError.invalidInput("maxTokens must be positive")
        }

        let started = Date()
        let conditionalPrompt = try promptEmbeddings(
            text: text,
            instruction: instruction,
            refAudio: refAudio,
            refText: refText
        )
        let usesGuidance = instruction?.isEmpty == false
        let unconditionalPrompt = usesGuidance
            ? try promptEmbeddings(text: text, instruction: nil, refAudio: refAudio, refText: refText)
            : nil

        let conditionalCache = backboneModel.makeCache()
        var conditionalHidden = backboneModel(
            inputEmbeddings: conditionalPrompt,
            cache: conditionalCache
        )[0..., -1, 0...]
        let unconditionalCache = unconditionalPrompt.map { _ in backboneModel.makeCache() }
        var unconditionalHidden = zipOptional(unconditionalPrompt, unconditionalCache).map {
            backboneModel(inputEmbeddings: $0.0, cache: $0.1)[0..., -1, 0...]
        }
        eval(conditionalHidden)
        if let unconditionalHidden { eval(unconditionalHidden) }
        let prefillTime = Date().timeIntervalSince(started)

        let sampler = parameters.temperature > 0
            ? TopPSampler(
                temperature: parameters.temperature,
                topP: parameters.topP,
                topK: min(parameters.topK, config.codecVocabSize),
                minP: parameters.minP,
                seed: parameters.seed
            )
            : nil
        var frames = [[Int32]]()

        for _ in 0..<maxTokens {
            if Task.isCancelled { throw CancellationError() }
            var logits = lmHead(conditionalHidden)
            if let unconditionalHidden {
                let unconditioned = lmHead(unconditionalHidden)
                logits = unconditioned + 4 * (logits - unconditioned)
            }
            logits = applyRepetitionPenalty(
                logits,
                generatedTokens: frames.map { Int($0[0]) },
                penalty: parameters.repetitionPenalty ?? 1
            )
            logits = maskReservedTokens(logits, allowsEOS: true)
            let first = sample(logits, sampler: sampler)
            if first == config.audioVocabSize { break }
            onToken?(first)

            var frame = [Int32(first)]
            var depthInputs = [Int32(0), Int32(first)]
            while frame.count < config.numCodebooks {
                let ids = MLXArray(depthInputs).reshaped([1, depthInputs.count])
                var depthLogits = depthDecoder.nextLogits(
                    tokenIDs: ids,
                    backboneHiddenState: conditionalHidden
                )
                if let unconditionalHidden {
                    let unconditioned = depthDecoder.nextLogits(
                        tokenIDs: ids,
                        backboneHiddenState: unconditionalHidden
                    )
                    depthLogits = unconditioned + 4 * (depthLogits - unconditioned)
                }
                depthLogits = maskReservedTokens(depthLogits, allowsEOS: false)
                let next = sample(depthLogits, sampler: sampler)
                frame.append(Int32(next))
                depthInputs.append(Int32(next))
            }
            frames.append(frame)

            let codebooks = MLXArray(frame).reshaped([1, 1, config.numCodebooks])
            conditionalHidden = backboneModel(inputIDs: codebooks, cache: conditionalCache)[0..., -1, 0...]
            if let cache = unconditionalCache {
                unconditionalHidden = backboneModel(inputIDs: codebooks, cache: cache)[0..., -1, 0...]
            }
            eval(conditionalHidden)
            if let unconditionalHidden { eval(unconditionalHidden) }
        }

        let audio: MLXArray
        if frames.isEmpty {
            audio = MLXArray.zeros([0])
        } else {
            let flat = frames.flatMap { $0 }
            let codes = MLXArray(flat).reshaped([1, frames.count, config.numCodebooks])
            let decoded = audioTokenizer.decode(codes)
            var waveform = decoded.0.squeezed()
            let validSamples = decoded.1.asArray(Int32.self).first.map(Int.init) ?? waveform.dim(0)
            if validSamples >= 0 && validSamples < waveform.dim(0) {
                waveform = waveform[..<validSamples]
            }
            audio = waveform
        }
        eval(audio)

        let totalTime = Date().timeIntervalSince(started)
        let generationTime = max(totalTime - prefillTime, 0)
        return GenerationResult(
            audio: audio,
            info: AudioGenerationInfo(
                promptTokenCount: conditionalPrompt.dim(1),
                generationTokenCount: frames.count,
                prefillTime: prefillTime,
                generateTime: generationTime,
                tokensPerSecond: Double(frames.count) / max(generationTime, 0.001),
                peakMemoryUsage: Double(Memory.peakMemory) / 1_000_000_000
            )
        )
    }

    private func promptEmbeddings(
        text: String,
        instruction: String?,
        refAudio: MLXArray?,
        refText: String?
    ) throws -> MLXArray {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Breeze text tokenizer is not loaded")
        }
        var parts = [MLXArray]()
        if let refAudio, let refText {
            parts.append(textEmbeddings(for: "[S0]\(refText)", tokenizer: tokenizer))
            parts.append(try referenceAudioEmbeddings(refAudio))
            let eos = MLXArray(
                [Int32](repeating: Int32(config.codebookEOSTokenID), count: config.numCodebooks)
            ).reshaped([1, 1, config.numCodebooks])
            parts.append(backboneModel.embedTokens(eos))
        }
        parts.append(textEmbeddings(for: Self.promptText(text: text, instruction: instruction), tokenizer: tokenizer))
        return concatenated(parts, axis: 1)
    }

    private func textEmbeddings(for text: String, tokenizer: Tokenizers.Tokenizer) -> MLXArray {
        let ids = MLXArray(tokenizer.encode(text: text).map { Int32($0) }).reshaped([1, -1])
        return textEncoderProjection(textEncoder(ids))
    }

    private func referenceAudioEmbeddings(_ reference: MLXArray) throws -> MLXArray {
        guard let audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Breeze audio tokenizer is not loaded")
        }
        let input: MLXArray
        switch reference.ndim {
        case 1: input = reference.reshaped([1, 1, reference.dim(0)])
        case 2: input = reference.expandedDimensions(axis: 1)
        case 3: input = reference
        default:
            throw AudioGenerationError.invalidInput("Reference audio must have one to three dimensions")
        }
        guard input.dim(0) == 1 else {
            throw AudioGenerationError.invalidInput("Breeze accepts one reference clip per request")
        }
        return backboneModel.embedTokens(audioTokenizer.encode(input).transposed(0, 2, 1))
    }

    private func maskReservedTokens(_ logits: MLXArray, allowsEOS: Bool) -> MLXArray {
        var result = logits
        if config.codecVocabSize < config.audioVocabSize {
            let count = config.audioVocabSize - config.codecVocabSize
            let ids = MLXArray((config.codecVocabSize..<config.audioVocabSize).map(Int32.init)).reshaped([1, count])
            result = putAlong(
                result,
                ids,
                values: MLXArray.full([1, count], values: MLXArray(-Float.infinity), dtype: result.dtype),
                axis: -1
            )
        }
        let width = allowsEOS ? config.audioVocabSize + 1 : config.audioVocabSize
        return result[0..., ..<width]
    }

    private func sample(_ logits: MLXArray, sampler: TopPSampler?) -> Int {
        let token = sampler?.sample(logits: logits) ?? argMax(logits, axis: -1)
        eval(token)
        return token.item(Int.self)
    }

    private func applyRepetitionPenalty(
        _ logits: MLXArray,
        generatedTokens: [Int],
        penalty: Float
    ) -> MLXArray {
        let unique = Array(Set(generatedTokens)).filter { $0 >= 0 && $0 < logits.dim(-1) }
        guard penalty != 1, !unique.isEmpty else { return logits }
        let ids = MLXArray(unique.map(Int32.init)).reshaped([1, -1])
        let selected = takeAlong(logits, ids, axis: -1)
        let penalized = MLX.where(selected .< 0, selected * penalty, selected / penalty)
        return putAlong(logits, ids, values: penalized, axis: -1)
    }

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> BreezeTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            cache: cache
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> BreezeTTSModel {
        let data = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(BreezeTTSConfig.self, from: data)
        let model = BreezeTTSModel(config: config)
        let weights = try loadSafetensors(in: modelDir)
        let sanitized = sanitize(weights: weights)

        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitized["\(path).scales"] != nil else { return nil }
                if let layer = config.perLayerQuantization?.quantization(layer: path) {
                    return layer.asTuple
                }
                return config.quantization?.asTuple
            }
        }
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }),
            verify: .all
        )
        eval(model.parameters())
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        model.audioTokenizer = try loadAudioTokenizer(
            from: modelDir.appendingPathComponent("audio_tokenizer")
        )
        return model
    }

    private static func loadSafetensors(in directory: URL) throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        for file in files where file.pathExtension == "safetensors" {
            result.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        return result
    }

    private static func loadAudioTokenizer(from directory: URL) throws -> Qwen3TTSSpeechTokenizer {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw AudioGenerationError.modelNotInitialized("Breeze checkpoint has no audio_tokenizer directory")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: data)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)
        let weights = try loadSafetensors(in: directory)
        let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: weights)
        try tokenizer.update(
            parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }),
            verify: .all
        )
        eval(tokenizer.parameters())
        return tokenizer
    }
}

private func zipOptional<A, B>(_ lhs: A?, _ rhs: B?) -> (A, B)? {
    guard let lhs, let rhs else { return nil }
    return (lhs, rhs)
}
