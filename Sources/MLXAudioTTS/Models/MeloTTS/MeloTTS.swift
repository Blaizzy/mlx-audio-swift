// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXRandom

public struct MeloTTSConfiguration: Decodable, Sendable {
    public var samplingRate = 44100
    public var addBlank = true
    public var nSpeakers = 256
    public var spk2id: [String: Int] = [:]
    public var interChannels = 192
    public var hiddenChannels = 192
    public var filterChannels = 768
    public var heads = 2
    public var layers = 6
    public var flowLayers = 3
    public var kernelSize = 3
    public var ginChannels = 256
    public var numTones = 16
    public var numLanguages = 10
    public var vocabularySize = 219
    public var useSpeakerEncoder = true
    public var resblockKernels = [3, 7, 11]
    public var resblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var upsampleRates = [8, 8, 2, 2, 2]
    public var upsampleChannels = 512
    public var upsampleKernels = [16, 16, 8, 2, 2]
    public var symbols: [String] = []
    public init() {}
    private struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        func value<T: Decodable>(_ key: String, _ fallback: T) throws -> T { try c.decodeIfPresent(T.self, forKey: Key(key)) ?? fallback }
        samplingRate = try value("sampling_rate", samplingRate); addBlank = try value("add_blank", addBlank)
        nSpeakers = try value("n_speakers", nSpeakers); spk2id = try value("spk2id", spk2id)
        interChannels = try value("inter_channels", interChannels); hiddenChannels = try value("hidden_channels", hiddenChannels)
        filterChannels = try value("filter_channels", filterChannels); heads = try value("n_heads", heads)
        layers = try value("n_layers", layers); flowLayers = try value("n_layers_trans_flow", flowLayers)
        kernelSize = try value("kernel_size", kernelSize); ginChannels = try value("gin_channels", ginChannels)
        numTones = try value("num_tones", numTones); numLanguages = try value("num_languages", numLanguages)
        vocabularySize = try value("n_vocab", vocabularySize); useSpeakerEncoder = try value("use_spk_conditioned_encoder", useSpeakerEncoder)
        resblockKernels = try value("resblock_kernel_sizes", resblockKernels); resblockDilations = try value("resblock_dilation_sizes", resblockDilations)
        upsampleRates = try value("upsample_rates", upsampleRates); upsampleChannels = try value("upsample_initial_channel", upsampleChannels)
        upsampleKernels = try value("upsample_kernel_sizes", upsampleKernels); symbols = try value("symbols", symbols)
        guard try value("resblock", "1") == "1", try value("use_transformer_flow", true) else {
            throw AudioGenerationError.invalidInput("MeloTTS supports transformer flows and type-1 HiFi-GAN residual blocks")
        }
        try validate()
    }
    func validate() throws {
        guard samplingRate > 0, interChannels > 0, interChannels % 2 == 0, hiddenChannels > 0,
              filterChannels > 0, heads > 0, hiddenChannels % heads == 0, interChannels % heads == 0,
              layers > 0, flowLayers > 0, ginChannels > 0, nSpeakers > 0, numTones >= 11, numLanguages > 2,
              kernelSize > 0, kernelSize % 2 == 1, upsampleChannels > 0,
              !upsampleRates.isEmpty, upsampleRates.count < 12, upsampleRates.count == upsampleKernels.count,
              upsampleChannels / (1 << upsampleRates.count) > 0,
              zip(upsampleRates, upsampleKernels).allSatisfy({ $0 > 0 && $1 >= $0 && ($1 - $0) % 2 == 0 }),
              !resblockKernels.isEmpty, resblockKernels.count == resblockDilations.count,
              resblockKernels.allSatisfy({ $0 > 0 && $0 % 2 == 1 }),
              resblockDilations.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0 > 0 } }),
              symbols.count == vocabularySize, Set(symbols).count == symbols.count, symbols.first == "_", symbols.contains("UNK"),
              spk2id.values.allSatisfy({ $0 >= 0 && $0 < nSpeakers }) else {
            throw AudioGenerationError.invalidInput("Invalid MeloTTS configuration")
        }
    }
}

public struct MeloTTSGenerationOptions: Sendable {
    public var speed: Float
    public var noiseScale: Float
    public var durationNoiseScale: Float
    public var stochasticDurationRatio: Float
    public init(speed: Float = 1, noiseScale: Float = 0.667, durationNoiseScale: Float = 0.8, stochasticDurationRatio: Float = 0) {
        self.speed = speed; self.noiseScale = noiseScale; self.durationNoiseScale = durationNoiseScale
        self.stochasticDurationRatio = stochasticDurationRatio
    }
    func validate() throws {
        guard speed.isFinite, speed > 0, noiseScale.isFinite, noiseScale >= 0,
              durationNoiseScale.isFinite, durationNoiseScale >= 0,
              stochasticDurationRatio.isFinite, (0...1).contains(stochasticDurationRatio) else {
            throw AudioGenerationError.invalidInput("Invalid MeloTTS speed or noise settings")
        }
    }
}

public final class MeloTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public let configuration: MeloTTSConfiguration
    public var generationOptions = MeloTTSGenerationOptions()
    public var sampleRate: Int { configuration.samplingRate }
    public var defaultGenerationParameters: GenerateParameters { .init(temperature: 0) }
    public var voices: [String] { configuration.spk2id.keys.sorted() }
    let network: MeloLayers
    let bert: MeloLayers
    let frontend: MeloEnglishFrontend

    init(configuration: MeloTTSConfiguration, network: MeloLayers, bert: MeloLayers, frontend: MeloEnglishFrontend) {
        self.configuration = configuration; self.network = network; self.bert = bert; self.frontend = frontend
    }
    func encode(_ result: MeloTextResult, speaker: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let c = configuration, n = result.phoneIDs.count
        let ids = MLXArray(result.phoneIDs).reshaped(1, n), tones = MLXArray(result.tones).reshaped(1, n)
        let features = bert.bert(MLXArray(result.tokenIDs).reshaped(1, -1))
            .take(MLXArray(result.featureIndices), axis: 1).swappedAxes(1, 2)
        let g = network.w("emb_g.weight")[speaker].reshaped(1, c.ginChannels, 1)
        let x = network.w("enc_p.emb.weight").take(ids, axis: 0).swappedAxes(1, 2)
            + network.w("enc_p.tone_emb.weight").take(tones, axis: 0).swappedAxes(1, 2)
            + network.w("enc_p.language_emb.weight")[2].reshaped(1, -1, 1)
            + network.conv(.zeros([1, 1024, n]), "enc_p.bert_proj")
            + network.conv(features, "enc_p.ja_bert_proj")
        let encoded = network.encoder(x, "enc_p.encoder", layers: c.layers, heads: c.heads, kernel: c.kernelSize, g: c.useSpeakerEncoder ? g : nil)
        let stats = network.conv(encoded, "enc_p.proj")
        return (encoded, stats[0..., ..<c.interChannels, 0...], stats[0..., c.interChannels..., 0...], g)
    }
    static func path(_ durations: MLXArray, frames: Int) -> MLXArray {
        let cumulative = cumsum(durations, axis: -1)
        let starts = cumulative - durations
        let positions = MLXArray(0..<frames).reshaped(1, 1, frames)
        return ((positions .>= starts.swappedAxes(1, 2)) .&& (positions .< cumulative.swappedAxes(1, 2))).asType(.float32)
    }
    func prepare(text: String, voice: String?, language: String?, options: MeloTTSGenerationOptions, parameters: GenerateParameters) throws -> (MLXArray, MLXArray) {
        try options.validate()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AudioGenerationError.invalidInput("MeloTTS text is empty") }
        if let language, language.uppercased() != "EN", !language.uppercased().hasPrefix("EN-") && language.uppercased() != "EN_INDIA" {
            throw AudioGenerationError.invalidInput("MeloTTS supports English text")
        }
        if let voice, configuration.spk2id[voice] == nil { throw AudioGenerationError.invalidInput("Unknown MeloTTS voice: \(voice)") }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let speaker = configuration.spk2id[voice ?? ""] ?? configuration.spk2id[language ?? "EN-US"] ?? configuration.spk2id["EN-Default"] ?? 0
        let result = try frontend.process(text, symbols: configuration.symbols, addBlank: configuration.addBlank)
        let (x, mean, logs, g) = encode(result, speaker: speaker)
        var logDuration = network.duration(x, g: g)
        if options.stochasticDurationRatio > 0 {
            logDuration = options.stochasticDurationRatio * network.stochasticDuration(x, g: g, noise: options.durationNoiseScale)
                + (1 - options.stochasticDurationRatio) * logDuration
        }
        let durations = ceil(exp(logDuration) / options.speed)
        guard isFinite(durations).all().item(Bool.self) else { throw AudioGenerationError.invalidInput("Non-finite MeloTTS durations") }
        let count = durations.sum().item(Float.self)
        guard count >= 1, count <= Float(parameters.maxTokens ?? 65536) else {
            throw AudioGenerationError.invalidInput("MeloTTS duration exceeds the latent frame budget")
        }
        let alignment = Self.path(durations, frames: Int(count))
        let m = matmul(mean, alignment), s = matmul(logs, alignment)
        let latent = m + MLXRandom.normal(m.shape) * exp(s) * options.noiseScale
        let z = network.flow(latent, g: g, config: configuration)
        eval(z, g)
        return (z, g)
    }
    public func generate(text: String, voice: String? = nil, language: String? = "EN-US", options: MeloTTSGenerationOptions,
                         generationParameters: GenerateParameters = .init(temperature: 0)) async throws -> MaterializedArray {
        let (z, g) = try prepare(text: text, voice: voice, language: language, options: options, parameters: generationParameters)
        let audio = network.decode(z, g: g, config: configuration).reshaped(-1)
        eval(audio)
        return audio.materialized()
    }
    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        guard refAudio == nil, refText == nil else { throw AudioGenerationError.invalidInput("MeloTTS uses preset voices and does not accept reference audio") }
        return (try await generate(text: text, voice: voice, language: language, options: generationOptions, generationParameters: generationParameters)).materialized()
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, generationParameters: generationParameters, streamingInterval: 1)
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?, generationParameters: GenerateParameters, streamingInterval: Double) -> AsyncThrowingStream<AudioGeneration, Error> {
        let hasReference = refAudio != nil || refText != nil, options = generationOptions
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard !hasReference else { throw AudioGenerationError.invalidInput("MeloTTS uses preset voices and does not accept reference audio") }
                    guard streamingInterval.isFinite, streamingInterval > 0 else { throw AudioGenerationError.invalidInput("Streaming interval must be positive") }
                    let (z, g) = try self.prepare(text: text, voice: voice, language: language, options: options, parameters: generationParameters)
                    let hop = self.configuration.upsampleRates.reduce(1, *)
                    let frames = max(1, Int(min(Double(z.dim(2)), Double(self.sampleRate) * streamingInterval / Double(hop))))
                    for start in stride(from: 0, to: z.dim(2), by: frames) {
                        try Task.checkCancellation()
                        let end = min(start + frames, z.dim(2)), context = max(0, start - 16)
                        let audio = self.network.decode(z[0..., 0..., context..<end], g: g, config: self.configuration).reshaped(-1)[((start - context) * hop)...]
                        eval(audio)
                        continuation.yield(.audio(audio))
                    }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
