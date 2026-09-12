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

    public let sampleRate: Int

    private static let stopTokens: Set<Int> = [128258, 151645]

    init(backbone: Qwen2Model, bicodec: SparkBiCodec, tokenizer: Tokenizers.Tokenizer, sampleRate: Int) {
        self.backbone = backbone
        self.bicodec = bicodec
        self.tokenizer = tokenizer
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
                "BiCodec/*", "*.json", "tokenizer*", "vocab*", "merges*", "special_tokens*",
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

        let tokenizer = try await AutoTokenizer.from(modelFolder: dir)
        eval(backbone, bicodec)
        return SparkModel(
            backbone: backbone, bicodec: bicodec, tokenizer: tokenizer,
            sampleRate: bcConfig.melParams.sampleRate)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let gender: SparkGender = (voice?.lowercased() == "male") ? .male : .female
        let prompt = SparkPrompt.control(gender: gender, pitch: .moderate, speed: .moderate, text: text)
        let promptIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let inputIds = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])

        let cache = backbone.newCache(parameters: generationParameters)
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
        let global = SparkPrompt.extractTokenIds(decoded, kind: "global")
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
