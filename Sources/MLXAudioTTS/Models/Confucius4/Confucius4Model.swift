// Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import HuggingFace
import Hub
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXAudioCodecs
import MLXNN
import MLXRandom
import Tokenizers

public struct Confucius4Config: Decodable, Sendable {
    public var sampleRate = 22050
    public var quantBits = 8
    public var quantGroupSize = 64
    enum CodingKeys: String, CodingKey { case sampleRate = "sample_rate", quantBits = "quant_bits", quantGroupSize = "quant_group_size" }
    public init() {}
    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 22050
        quantBits = try c.decodeIfPresent(Int.self, forKey: .quantBits) ?? 8
        quantGroupSize = try c.decodeIfPresent(Int.self, forKey: .quantGroupSize) ?? 64
    }
}

public struct Confucius4GenerationOptions: Sendable {
    /// Protocol/CLI reference tensors use the model's sample rate (22,050 Hz).
    /// Set this for a reference tensor at another rate, such as 16,000 Hz.
    public var referenceSampleRate: Int?
    public var diffusionSteps: Int
    public var guidance: Float
    public init(referenceSampleRate: Int? = nil, diffusionSteps: Int = 25, guidance: Float = 0.7) {
        self.referenceSampleRate = referenceSampleRate; self.diffusionSteps = diffusionSteps; self.guidance = guidance
    }
}

/// Multilingual, cross-lingual voice cloning with the published Confucius4
/// checkpoint. A finite mono reference waveform is required for every request.
public final class Confucius4Model: SpeechGenerationModel, @unchecked Sendable {
    public static let defaultRepositoryID = "mlx-community/Confucius4-TTS-mlx-int8"
    public let config: Confucius4Config
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        .init(maxTokens: 512, temperature: 0.8, topP: 0.8, topK: 30, repetitionPenalty: 10, seed: 0)
    }
    let semanticEncoder: Confucius4W2VBert
    let prefix: Confucius4Prefix
    let t2s: Confucius4T2S
    let acoustic: Confucius4Acoustic
    let speaker: CAMPPlus
    let vocoder: BigVGAN
    let mean: MLXArray
    let std: MLXArray
    let melFilters: MLXArray
    let window: MLXArray
    let tokenizer: any Tokenizer

    init(config: Confucius4Config, w2v: [String: MLXArray], t2s: [String: MLXArray],
         s2a: [String: MLXArray], speaker: CAMPPlus, vocoder: BigVGAN,
         stats: [String: MLXArray], filters: [String: MLXArray], tokenizer: any Tokenizer) {
        self.config = config
        semanticEncoder = Confucius4W2VBert(w2v, groupSize: config.quantGroupSize, bits: config.quantBits)
        prefix = Confucius4Prefix(t2s, groupSize: config.quantGroupSize, bits: config.quantBits)
        self.t2s = Confucius4T2S(t2s, groupSize: config.quantGroupSize, bits: config.quantBits)
        acoustic = Confucius4Acoustic(s2a); self.speaker = speaker; self.vocoder = vocoder
        mean = stats["mean"]!; std = stats["std"]!; melFilters = filters["mel"]!; window = filters["window"]!
        self.tokenizer = tokenizer
    }
    static let languageTokens = ["zh": "请用中文朗读接下来的文字", "en": "请用英文朗读接下来的文字",
        "vi": "请用越南语朗读接下来的文字", "ja": "请用日语朗读接下来的文字",
        "ko": "请用韩语朗读接下来的文字", "th": "请用泰语朗读接下来的文字"]
    static func prompt(text: String, language: String?) -> String {
        let token = languageTokens[language ?? "vi"] ?? languageTokens["en"]!
        return "You are a helpful assistant. \(token):\(text)"
    }
    func run(text: String, reference: MLXArray?, language: String?, parameters p: GenerateParameters,
             options: Confucius4GenerationOptions) throws -> (MLXArray, Int, Int) {
        try Task.checkCancellation()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AudioGenerationError.invalidInput("Text must not be empty") }
        guard let reference, reference.size > 0,
              reference.ndim == 1 || (reference.ndim == 2 && reference.shape.contains(1)),
              all(isFinite(reference)).item(Bool.self) else {
            throw AudioGenerationError.invalidInput("Confucius4 requires a finite mono reference waveform")
        }
        let rate = options.referenceSampleRate ?? sampleRate
        guard rate > 0, options.diffusionSteps > 0, options.diffusionSteps <= 1000, options.guidance.isFinite,
              options.guidance >= 0, p.temperature.isFinite, p.temperature >= 0,
              p.topP.isFinite, p.topP > 0, p.topP <= 1, p.topK >= 0,
              (p.maxTokens ?? 512) >= 2, (p.repetitionPenalty ?? 10).isFinite, (p.repetitionPenalty ?? 10) > 0 else {
            throw AudioGenerationError.invalidInput("Invalid Confucius4 generation parameters")
        }
        let values = Confucius4Features.resample(reference.reshaped([-1]).asArray(Float.self), from: rate, to: 16000)
        guard values.count >= 2400 else { throw AudioGenerationError.invalidInput("Confucius4 reference must contain at least 0.15 seconds of audio") }
        let audio = MLXArray(values)
        let features = Confucius4Features.fbank(audio, mel: melFilters, window: window)
        // Reflect-padded ECAPA dilated kernels need at least 5 semantic frames.
        guard features.shape[1] > 4 else { throw AudioGenerationError.invalidInput("Reference audio is too short for semantic conditioning") }
        let semantic = try semanticEncoder(features)
        let condition = prefix.condition((semantic - mean) / std)
        let style = speaker.inference([audio]).reshaped([1, 192])
        let refMel = try Confucius4Features.referenceMel(audio)
        let ids = tokenizer.encode(text: Self.prompt(text: text, language: language), addSpecialTokens: true)
        guard !ids.isEmpty, ids.count <= prefix["text_position_embedding.embedding.weight"].shape[0],
              ids.allSatisfy({ $0 >= 0 && $0 < prefix["text_projector.embed.weight"].shape[0] }) else {
            throw AudioGenerationError.invalidInput("Confucius4 text exceeds its learned text positions or vocabulary")
        }
        let textEmbedding = prefix.text(MLXArray(ids.map(Int32.init)).reshaped([1, -1]))
        eval(condition, style, refMel, textEmbedding)
        let (codes, latent) = try t2s.generate(condition: condition, text: textEmbedding, parameters: p)
        let mu = acoustic.buildMu(codes: codes, latent: latent, referenceFrames: refMel.shape[1])
        let noise = MLXRandom.normal([1, 80, mu.shape[1]], key: MLXRandom.split(key: MLXRandom.key(p.seed ?? 0)).1)
        let fullMel = try acoustic.solve(noise: noise, prompt: refMel.transposed(0, 2, 1), mu: mu, speaker: style,
                                        steps: options.diffusionSteps, guidance: options.guidance)
        let mel = fullMel[0..., 0..., refMel.shape[1]...]
        try Task.checkCancellation()
        let wav = vocoder(mel).reshaped([-1]); eval(wav)
        guard wav.size > 0, all(isFinite(wav)).item(Bool.self) else { throw AudioGenerationError.generationFailed("Confucius4 produced invalid audio") }
        return (wav, ids.count, codes.shape[1])
    }
    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
                         generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try await generate(text: text, refAudio: refAudio, language: language, generationParameters: generationParameters)
    }
    public func generate(text: String, refAudio: MaterializedArray?, language: String? = nil,
                         generationParameters: GenerateParameters? = nil,
                         options: Confucius4GenerationOptions = .init()) async throws -> MaterializedArray {
        let (audio, _, _) = try run(text: text, reference: refAudio, language: language,
                                    parameters: generationParameters ?? defaultGenerationParameters, options: options)
        return audio.materialized()
    }
    /// Matches the reference model's full-utterance generation: one audio event.
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
                               generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, refAudio: refAudio, language: language, generationParameters: generationParameters)
    }
    public func generateStream(text: String, refAudio: MaterializedArray?, language: String? = nil,
                               generationParameters: GenerateParameters? = nil,
                               options: Confucius4GenerationOptions = .init()) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio, parameters = generationParameters ?? defaultGenerationParameters
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    let start = Date()
                    let (audio, prompt, tokens) = try run(text: text, reference: reference, language: language, parameters: parameters, options: options)
                    if case .terminated = continuation.yield(.audio(audio)) { throw CancellationError() }
                    let elapsed = Date().timeIntervalSince(start)
                    continuation.yield(.info(.init(promptTokenCount: prompt, generationTokenCount: tokens, prefillTime: 0,
                        generateTime: elapsed, tokensPerSecond: Double(tokens) / max(elapsed, 1e-6), peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public static func fromPretrained(_ repository: String = defaultRepositoryID, cache: HubCache = .default) async throws -> Confucius4Model {
        guard let id = Repo.ID(rawValue: repository) else { throw TTSModelError.invalidRepositoryID(repository) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: id, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.npz", "checkpoints/tokenizer.json"], cache: cache)
        return try await fromModelDirectory(directory)
    }
    public static func fromModelDirectory(_ directory: URL) async throws -> Confucius4Model {
        let config = try JSONDecoder().decode(Confucius4Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard config.sampleRate == 22050, [4, 8].contains(config.quantBits), [32, 64, 128].contains(config.quantGroupSize) else {
            throw AudioGenerationError.invalidInput("Unsupported Confucius4 configuration")
        }
        func read(_ name: String) throws -> [String: MLXArray] { try loadArrays(url: directory.appendingPathComponent(name + ".safetensors")) }
        let w2v = try read("w2vbert_mlx"), t2s = try read("t2s_model"), s2a = try read("s2a_mlx")
        try Confucius4Checkpoint.validate(w2v: w2v, t2s: t2s, s2a: s2a, config: config)
        let stats = try NumpyArchive.load(url: directory.appendingPathComponent("w2v_stats.npz"))
        let filters = try NumpyArchive.load(url: directory.appendingPathComponent("fbank_filters.npz"))
        guard stats["mean"]?.size == 1024, stats["std"]?.size == 1024,
              filters["mel"]?.shape == [257, 80], filters["window"]?.shape == [400],
              all(isFinite(stats["mean"]!)).item(Bool.self), all(isFinite(stats["std"]!)).item(Bool.self),
              all(stats["std"]! .> 0).item(Bool.self) else {
            throw AudioGenerationError.invalidInput("Invalid Confucius4 feature/statistics sidecars")
        }
        let camp = CAMPPlus(referenceCompatible: true)
        try camp.update(parameters: ModuleParameters.unflattened(CAMPPlus.sanitize(weights: try read("campplus"), model: camp)), verify: .all)
        camp.train(false)
        let vocoder = BigVGAN(config: .init(numMels: 80, upsampleRates: [4, 4, 2, 2, 2, 2], upsampleKernelSizes: [8, 8, 4, 4, 4, 4],
            upsampleInitialChannel: 1536, resblock: .one, resblockKernelSizes: [3, 7, 11],
            resblockDilationSizes: Array(repeating: [1, 3, 5], count: 3), activation: .snakebeta, snakeLogscale: true,
            useBiasAtFinal: false, useTanhAtFinal: false))
        let mapped = try Confucius4Checkpoint.vocoder(try read("bigvgan_mlx"), model: vocoder)
        try vocoder.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        vocoder.train(false)
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("checkpoints/tokenizer.json"))) as? [NSString: Any] else {
            throw AudioGenerationError.invalidInput("Invalid Confucius4 tokenizer JSON")
        }
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: ["tokenizer_class": "LlamaTokenizer", "bos_token": "<s>",
            "eos_token": "</s>", "unk_token": "<unk>", "add_bos_token": true, "add_eos_token": false], tokenizerData: Config(object))
        eval(camp, vocoder)
        return Confucius4Model(config: config, w2v: w2v, t2s: t2s, s2a: s2a, speaker: camp, vocoder: vocoder,
                               stats: stats, filters: filters, tokenizer: tokenizer)
    }
}
