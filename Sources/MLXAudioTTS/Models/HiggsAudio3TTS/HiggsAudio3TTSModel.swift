import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioCodecs
import MLXNN
import Tokenizers

/// Delayed reference codes can be cached and reused across requests.
public struct HiggsV3Reference {
    public var codes: MLXArray
    public var text: String?
    public init(codes: MLXArray, text: String? = nil) {
        self.codes = codes
        self.text = text
    }
}

public final class HiggsAudio3TTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: HiggsAudio3TTSConfig
    @ModuleInfo(key: "backbone") var backbone: MossQwen3Model
    @ModuleInfo(key: "multimodal_embedding") var multimodalEmbedding: Embedding
    public var tokenizer: (any Tokenizer)?
    private var codec: HiggsAudioTokenizer?

    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        .init(maxTokens: 2048, temperature: 0.7, topP: 0.95)
    }
    public var fadeInMilliseconds: Float = 30
    public var fadeOutMilliseconds: Float = 15

    public init(_ config: HiggsAudio3TTSConfig) {
        self.config = config
        _backbone.wrappedValue = MossQwen3Model(config.textConfig)
        _multimodalEmbedding.wrappedValue = Embedding(
            embeddingCount: config.audioNumCodebooks * config.audioCodebookSize,
            dimensions: config.textConfig.hiddenSize)
    }

    func embedAudio(_ codes: MLXArray) -> MLXArray {
        let rows = codes.ndim == 1 ? codes.expandedDimensions(axis: 0) : codes
        let offsets = MLXArray(0..<config.audioNumCodebooks).asType(.int32) * config.audioCodebookSize
        return multimodalEmbedding(rows.asType(.int32) + offsets).sum(axis: -2)
    }
    func audioLogits(_ hidden: MLXArray) -> MLXArray {
        multimodalEmbedding.asLinear(hidden).reshaped(
            hidden.shape.dropLast() + [config.audioNumCodebooks, config.audioCodebookSize])
    }

    /// Encodes mono samples at `sampleRate` with the semantic and acoustic towers.
    public func encodeReferenceAudio(_ audio: MLXArray) throws -> MLXArray {
        guard let codec else { throw AudioGenerationError.modelNotInitialized("Higgs codec is missing") }
        var mono = audio.asType(.float32)
        if mono.ndim == 2 {
            mono = mono.dim(0) <= 2 ? mono.mean(axis: 0) : mono.mean(axis: -1)
        }
        guard mono.ndim == 1, mono.size > 0 else {
            throw AudioGenerationError.invalidInput("Reference audio must contain mono or stereo samples")
        }
        if mono.size < sampleRate {
            mono = concatenated([mono, MLXArray.zeros([sampleRate - mono.size])])
        }
        let codes = try codec.encode(mono.reshaped([1, -1, 1]))[0]
        let delayed = higgsV3Delay(codes, boc: config.audioBOCTokenID, eoc: config.audioEOCTokenID)
        eval(delayed)
        return delayed
    }

    func promptEmbeddings(text: String, references: [HiggsV3Reference]) throws -> (MLXArray, [Int]) {
        guard let tokenizer else { throw AudioGenerationError.modelNotInitialized("Higgs tokenizer is missing") }
        func special(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw AudioGenerationError.invalidInput("Tokenizer is missing \(token)")
            }
            return id
        }
        var ids = [try special("<|tts|>")]
        var pieces: [MLXArray] = []
        func flush() {
            if !ids.isEmpty { pieces.append(backbone.embedTokens(MLXArray(ids).expandedDimensions(axis: 0))[0]) }
        }
        var allIDs = ids
        for reference in references {
            guard reference.codes.ndim == 2, reference.codes.dim(1) == config.audioNumCodebooks,
                  reference.codes.dim(0) > 0 else {
                throw AudioGenerationError.invalidInput("Reference codes must have shape [frames, \(config.audioNumCodebooks)]")
            }
            let values = reference.codes.asArray(Int32.self)
            guard values.allSatisfy({ $0 >= 0 && $0 < config.audioCodebookSize }) else {
                throw AudioGenerationError.invalidInput("Reference codes are outside the audio vocabulary")
            }
            if let text = reference.text, !text.isEmpty,
               let id = tokenizer.convertTokenToId("<|ref_text|>") {
                ids.append(id)
                ids += tokenizer.encode(text: text, addSpecialTokens: false)
            }
            ids.append(try special("<|ref_audio|>"))
            allIDs += Array(ids.dropFirst(pieces.isEmpty ? 1 : 0))
            flush()
            pieces.append(embedAudio(reference.codes))
            allIDs += [Int](repeating: -100, count: reference.codes.dim(0))
            ids = []
        }
        ids += [try special("<|text|>")]
        ids += tokenizer.encode(text: text, addSpecialTokens: false)
        ids += [try special("<|audio|>")]
        if references.isEmpty { allIDs = ids } else { allIDs += ids }
        flush()
        return (concatenated(pieces, axis: 0).expandedDimensions(axis: 0), allIDs)
    }

    /// Inline `<|emotion:...|>`, `<|style:...|>` and `<|prosody:...|>` controls
    /// are passed through the model's tokenizer without text normalization.
    public func generate(
        text: String, references: [HiggsV3Reference],
        generationParameters: GenerateParameters? = nil
    ) async throws -> MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("Text must not be empty")
        }
        guard config.useDelayPattern else {
            throw AudioGenerationError.invalidInput("This Higgs v3 checkpoint must use delayed audio codebooks")
        }
        guard let codec else { throw AudioGenerationError.modelNotInitialized("Higgs codec is missing") }
        let parameters = generationParameters ?? defaultGenerationParameters
        let limit = parameters.maxTokens ?? 2048
        guard limit >= config.audioNumCodebooks else {
            throw AudioGenerationError.invalidInput("maxTokens must allow at least one complete audio frame")
        }
        let (prompt, _) = try promptEmbeddings(text: text, references: references)
        let cache = backbone.makeCache()
        var hidden = try backbone(inputEmbeddings: prompt, cache: cache)[0, -1, 0...]
        let sampler = HiggsV3Sampler(parameters)
        var state = HiggsV3SamplerState()
        var rows: [MLXArray] = []
        for _ in 0..<limit {
            try Task.checkCancellation()
            let codes = state.step(sampler.sample(audioLogits(hidden)),
                codebooks: config.audioNumCodebooks, boc: config.audioBOCTokenID, eoc: config.audioEOCTokenID)
            eval(codes)
            rows.append(codes)
            if state.done { break }
            hidden = try backbone(inputEmbeddings: embedAudio(codes).expandedDimensions(axis: 0), cache: cache)[0, -1, 0...]
        }
        let raw = higgsV3Undelay(stacked(rows, axis: 0))
        var audio = codec.decode(raw).asType(.float32).reshaped(-1)
        let nIn = Int(fadeInMilliseconds * Float(sampleRate) / 1000)
        let nOut = Int(fadeOutMilliseconds * Float(sampleRate) / 1000)
        if nIn > 0 && audio.size > nIn { audio[0..<nIn] = audio[0..<nIn] * MLXArray.linspace(0, 1, count: nIn) }
        if nOut > 0 && audio.size > nOut { audio[(audio.size - nOut)...] = audio[(audio.size - nOut)...] * MLXArray.linspace(1, 0, count: nOut) }
        eval(audio)
        return audio
    }

    public func generate(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        let references = try refAudio.map { [HiggsV3Reference(codes: try encodeReferenceAudio($0), text: refText)] } ?? []
        return try await generate(text: text, references: references, generationParameters: generationParameters)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let refAudio = SendingBox(refAudio)
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    let audio = try await self.generate(text: text, voice: voice, refAudio: refAudio.take(),
                        refText: refText, language: language, generationParameters: generationParameters)
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("tied.embedding.modality_embeddings.0.model.") || key.hasPrefix("tied.head.") { continue }
            let mapped = key
                .replacingOccurrences(of: "tied.embedding.text_embedding.", with: "backbone.embed_tokens.")
                .replacingOccurrences(of: "body.layers.", with: "backbone.layers.")
                .replacingOccurrences(of: "body.norm.", with: "backbone.norm.")
                .replacingOccurrences(of: "tied.embedding.modality_embeddings.0.embedding.", with: "multimodal_embedding.")
            result[mapped] = value
        }
        return result
    }

    public static func fromPretrained(_ source: String, cache: HubCache = .default) async throws -> HiggsAudio3TTSModel {
        let local = URL(fileURLWithPath: NSString(string: source).expandingTildeInPath)
        var directory: ObjCBool = false
        if FileManager.default.fileExists(atPath: local.path, isDirectory: &directory), directory.boolValue {
            return try await fromModelDirectory(local)
        }
        guard let repo = Repo.ID(rawValue: source) else { throw TTSModelError.invalidRepositoryID(source) }
        let path = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["audio_tokenizer/*.json", "audio_tokenizer/*.safetensors"], cache: cache)
        return try await fromModelDirectory(path)
    }

    public static func fromModelDirectory(_ path: URL) async throws -> HiggsAudio3TTSModel {
        let data = try Data(contentsOf: path.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(HiggsAudio3TTSConfig.self, from: data)
        let model = HiggsAudio3TTSModel(config)
        let raw = try loadWeights(from: path)
        let weights = model.sanitize(weights: raw)
        let quantization = try JSONDecoder().decode(BaseConfiguration.self, from: data).perLayerQuantization
        if let quantization {
            quantize(model: model) { name, _ in
                guard name.hasPrefix("backbone."), weights[name + ".scales"] != nil else { return nil }
                return quantization.quantization(layer: name)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: path)
        if FileManager.default.fileExists(atPath: path.appendingPathComponent("audio_tokenizer/config.json").path) {
            model.codec = try HiggsAudioTokenizer.fromModelDirectory(path)
        } else {
            var codecConfig = HiggsAudioTokenizerConfig(dacNumCodebooks: config.audioNumCodebooks)
            codecConfig.semanticModelConfig = HiggsAudioSemanticConfig()
            let codec = HiggsAudioTokenizer(config: codecConfig)
            // The reference bundled-codec reader promotes BF16 to Float32.
            let codecWeights = HiggsAudioTokenizer.sanitize(weights: raw,
                prefix: "tied.embedding.modality_embeddings.0.model.", includeSemantic: true)
                .mapValues { $0.asType(.float32) }
            try codec.update(parameters: ModuleParameters.unflattened(codecWeights), verify: .all)
            eval(codec)
            model.codec = codec
        }
        return model
    }
}
