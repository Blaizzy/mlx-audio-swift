import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXLLM
@preconcurrency import MLXLMCommon
import MLXAudioCore
import MLXNN
import Tokenizers

public enum SparkTTSError: Error {
    case invalidRepo(String)
    case noAudioTokens
}

/// BiCodec configuration for the published `Spark-TTS-0.5B` checkpoint (the
/// subset used by the synthesis path).
private let sparkBiCodecConfigJSON = """
{
 "mel_params":{"sample_rate":16000},
 "decoder":{"input_channel":1024,"channels":1536,"rates":[8,5,4,2],"kernel_sizes":[16,11,8,4]},
 "quantizer":{"input_dim":1024,"codebook_size":8192,"codebook_dim":8},
 "speaker_encoder":{"out_dim":1024,"latent_dim":128,"token_num":32,"fsq_levels":[4,4,4,4,4,4]},
 "prenet":{"input_channels":1024,"vocos_dim":384,"vocos_intermediate_dim":2048,"vocos_num_layers":12,"out_channels":1024,"condition_dim":1024,"sample_ratios":[1,1],"use_tanh_at_final":false}
}
"""

public final class SparkModel: SpeechGenerationModel, @unchecked Sendable {
    private let backbone: Qwen2Model
    private let bicodec: SparkBiCodec
    private let tokenizer: Tokenizers.Tokenizer
    private let modelDir: URL
    private var cachedWav2Vec2: SparkWav2Vec2?

    public let sampleRate: Int

    private static let stopTokens: Set<Int> = [128258, 151645]

    init(
        backbone: Qwen2Model, bicodec: SparkBiCodec, tokenizer: Tokenizers.Tokenizer,
        modelDir: URL, sampleRate: Int
    ) {
        self.backbone = backbone
        self.bicodec = bicodec
        self.tokenizer = tokenizer
        self.modelDir = modelDir
        self.sampleRate = sampleRate
    }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 3000, temperature: 0.8, topP: 0.95,
            repetitionPenalty: 1.3, repetitionContextSize: 20)
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> SparkModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw SparkTTSError.invalidRepo(modelRepo)
        }
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: [
                "BiCodec/*", "wav2vec2-large-xlsr-53/*", "*.json",
                "tokenizer*", "vocab*", "merges*", "special_tokens*",
            ],
            cache: cache)

        let lmConfig = try JSONDecoder().decode(
            Qwen2Configuration.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let backbone = Qwen2Model(lmConfig)
        let lmWeights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        try backbone.update(
            parameters: ModuleParameters.unflattened(backbone.sanitize(weights: lmWeights)),
            verify: .none)

        let bcConfig = try JSONDecoder().decode(
            BiCodecConfiguration.self, from: Data(sparkBiCodecConfigJSON.utf8))
        let bicodec = SparkBiCodec(bcConfig)
        let bcWeights = try MLX.loadArrays(url: dir.appendingPathComponent("BiCodec/model.safetensors"))
        try bicodec.update(
            parameters: ModuleParameters.unflattened(bicodec.sanitize(bcWeights)), verify: .none)
        bicodec.train(false)

        let tokenizer = try await AutoTokenizer.from(modelFolder: dir)
        eval(backbone, bicodec)
        return SparkModel(
            backbone: backbone, bicodec: bicodec, tokenizer: tokenizer,
            modelDir: dir, sampleRate: bcConfig.melParams.sampleRate)
    }

    private func wav2vec2() throws -> SparkWav2Vec2 {
        if let cachedWav2Vec2 { return cachedWav2Vec2 }
        let w2v = SparkWav2Vec2()
        let weights = try MLX.loadArrays(
            url: modelDir.appendingPathComponent("wav2vec2-large-xlsr-53/model.safetensors"))
        try w2v.update(parameters: ModuleParameters.unflattened(w2v.sanitize(weights)), verify: .none)
        w2v.train(false)
        eval(w2v)
        cachedWav2Vec2 = w2v
        return w2v
    }

    private func referenceClip(_ refAudio: MLXArray) -> MLXArray {
        let mono = refAudio.ndim > 1 ? refAudio.reshaped([-1]) : refAudio
        let refLen = (sampleRate * 6) / 320 * 320
        let n = mono.shape[0]
        if refLen > n {
            let reps = refLen / n + 1
            return MLX.tiled(mono, repetitions: [reps])[0 ..< refLen]
        }
        return mono[0 ..< refLen]
    }

    private func tokenizeReference(_ refAudio: MLXArray, refText: String?) throws -> ([Int], [Int]?) {
        let mel = SparkMel.melSpectrogram(referenceClip(refAudio))
        let global = bicodec.tokenizeGlobal(mel)
        eval(global)
        let globalIds = global.reshaped([-1]).asArray(Int32.self).map { Int($0) }

        guard refText != nil else { return (globalIds, nil) }
        let feat = try wav2vec2().features(refAudio)
        let semantic = bicodec.tokenizeSemantic(feat)
        eval(semantic)
        let semanticIds = semantic.reshaped([-1]).asArray(Int32.self).map { Int($0) }
        return (globalIds, semanticIds)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        var refGlobalIds: [Int]? = nil
        let prompt: String
        if let refAudio {
            let (globalIds, semanticIds) = try tokenizeReference(refAudio, refText: refText)
            refGlobalIds = globalIds
            prompt = SparkPrompt.clone(
                text: text, refText: refText, globalTokenIds: globalIds, semanticTokenIds: semanticIds)
        } else {
            let gender: SparkGender = (voice?.lowercased() == "male") ? .male : .female
            prompt = SparkPrompt.control(gender: gender, pitch: .moderate, speed: .moderate, text: text)
        }
        let promptIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let inputIds = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])

        let cache = try backbone.newCache(parameters: generationParameters)
        let sampler = generationParameters.sampler()
        var processor = generationParameters.processor()
        processor?.prompt(MLXArray(promptIds.map { Int32($0) }))

        var logits = backbone(inputIds, cache: cache)
        var generated: [Int] = []
        let maxTokens = generationParameters.maxTokens ?? 3000

        for step in 0..<maxTokens {
            try Task.checkCancellation()
            let tokenValue: Int = autoreleasepool {
                var last = logits[0..., -1, 0...]
                last = processor?.process(logits: last) ?? last
                let next = sampler.sample(logits: last)
                let value = next.item(Int.self)
                if !SparkModel.stopTokens.contains(value) {
                    processor?.didSample(token: next)
                    logits = backbone(next.reshaped([1, 1]), cache: cache)
                    eval(logits)
                }
                return value
            }
            if SparkModel.stopTokens.contains(tokenValue) { break }
            generated.append(tokenValue)
            if step % 50 == 0 { Memory.clearCache() }
        }
        Memory.clearCache()

        let decoded = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
        let semantic = SparkPrompt.extractTokenIds(decoded, kind: "semantic")
        let global = refGlobalIds ?? SparkPrompt.extractTokenIds(decoded, kind: "global")
        guard !semantic.isEmpty, !global.isEmpty else { throw SparkTTSError.noAudioTokens }

        let s = MLXArray(semantic.map { Int32($0) }).reshaped([1, semantic.count])
        let g = MLXArray(global.map { Int32($0) }).reshaped([1, global.count])
        let audio = bicodec.detokenize(semanticTokens: s, globalTokens: g)
        eval(audio)
        Memory.clearCache()
        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language, generationParameters: generationParameters, streamingInterval: 2.0)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { continuation.finish(); return }
            do {
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio, refText: refText,
                    language: language, generationParameters: generationParameters)
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }
}
