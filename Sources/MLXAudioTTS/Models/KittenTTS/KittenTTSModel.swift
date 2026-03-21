import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public final class KittenTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: KittenTTSConfig
    public let sampleRate: Int
    public let defaultGenerationParameters: GenerateParameters

    @ModuleInfo var bert: KittenAlbert
    @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
    @ModuleInfo var predictor: KittenProsodyPredictor
    @ModuleInfo(key: "text_encoder") var textEncoder: KittenTextEncoder
    @ModuleInfo var decoder: KittenDecoder

    private var voices: [String: MLXArray] = [:]
    private let contextLength: Int
    private let g2p = EnglishG2P(british: false)

    private init(config: KittenTTSConfig) {
        self.config = config
        self.sampleRate = config.sampleRate
        self.defaultGenerationParameters = GenerateParameters(temperature: 0, topP: 1)
        self.contextLength = config.plbert.maxPositionEmbeddings

        _bert = ModuleInfo(wrappedValue: KittenAlbert(config: config.plbert, vocabSize: config.nToken))
        _bertEncoder = ModuleInfo(wrappedValue: Linear(config.plbert.hiddenSize, config.hiddenDim), key: "bert_encoder")
        _predictor = ModuleInfo(wrappedValue: KittenProsodyPredictor(
            styleDim: config.styleDim, dHid: config.hiddenDim,
            nLayers: config.nLayer, maxDur: config.maxDur, dropout: 0.0))
        _textEncoder = ModuleInfo(wrappedValue: KittenTextEncoder(
            channels: config.hiddenDim, kernelSize: config.textEncoderKernelSize,
            depth: config.nLayer, nSymbols: config.nToken), key: "text_encoder")
        _decoder = ModuleInfo(wrappedValue: KittenDecoder(config: config))
    }

    /// For testing: create model from config without loading weights
    static func testInit(config: KittenTTSConfig) -> KittenTTSModel {
        KittenTTSModel(config: config)
    }

    // MARK: - Forward Pass

    func callAsFunction(
        inputIds: MLXArray, refS: MLXArray, speed: Float = 1.0
    ) -> (audio: MLXArray, predDur: MLXArray) {
        let seqLen = inputIds.shape[inputIds.ndim - 1]
        let inputLengths = MLXArray([Int32(seqLen)])
        var textMask = MLXArray(Array(0..<Int32(seqLen))).reshaped([1, -1])
        textMask = (textMask + 1) .> inputLengths.reshaped([-1, 1])

        let attMask = MLXArray(1) - textMask.asType(.int32)
        let (bertOut, _) = bert(inputIds, attentionMask: attMask)
        let dEn = bertEncoder(bertOut).transposed(0, 2, 1)

        let s = refS[0..., 128...]
        let d = predictor.textEncoder(dEn, style: s, textLengths: inputLengths, mask: textMask)
        let (x, _) = predictor.lstm(d)
        let duration = predictor.durationProj(x)
        let durSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
        let predDur = MLX.clip(MLX.round(durSigmoid), min: 1).asType(.int32)[0]

        let durArray: [Int32] = predDur.asArray(Int32.self)
        var indices = [MLXArray]()
        for (i, n) in durArray.enumerated() {
            if n > 0 {
                indices.append(MLXArray(Array(repeating: Int32(i), count: Int(n))))
            }
        }
        let allIndices = MLX.concatenated(indices, axis: 0)

        var predAlnTrg = MLXArray.zeros([inputIds.shape[1], allIndices.shape[0]])
        predAlnTrg[allIndices, MLXArray(Array(0..<Int32(allIndices.shape[0])))] = MLXArray(Float(1))
        let predAln = predAlnTrg.expandedDimensions(axis: 0)

        let en = MLX.matmul(d.transposed(0, 2, 1), predAln)
        let (f0Pred, nPred) = predictor.f0Ntrain(en, s)

        let tEn = textEncoder(inputIds, inputLengths: inputLengths, mask: textMask)
        let asr = MLX.matmul(tEn, predAln)

        let audio = decoder(asr, f0Pred, nPred, refS[0..., ..<128])
        return (audio[0], predDur)
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let (inputIds, refS, speed) = try prepareInputs(text: text, voice: voice)
        let (audio, _) = self.callAsFunction(inputIds: inputIds, refS: refS, speed: speed)
        return audio.reshaped([-1])
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        AsyncThrowingStream { continuation in
            do {
                let (inputIds, refS, speed) = try self.prepareInputs(text: text, voice: voice)
                let (audio, _) = self.callAsFunction(inputIds: inputIds, refS: refS, speed: speed)
                continuation.yield(.audio(audio.reshaped([-1])))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    // MARK: - Text Processing

    private func prepareInputs(text: String, voice: String?, speed: Float = 1.0) throws -> (MLXArray, MLXArray, Float) {
        var voiceKey = voice ?? "expr-voice-5-m"
        if let alias = config.voiceAliases?[voiceKey] {
            voiceKey = alias
        }
        guard let voiceEmb = voices[voiceKey] else {
            let available = voices.keys.sorted().joined(separator: ", ")
            throw AudioGenerationError.invalidInput("Voice '\(voiceKey)' not available. Choose from: \(available)")
        }

        var adjustedSpeed = speed
        if let prior = config.speedPriors?[voiceKey] {
            adjustedSpeed *= prior
        }

        let tokens = phonemize(text)
        var tokenArray = [Int32(0)]
        tokenArray.append(contentsOf: tokens.map { Int32($0) })
        tokenArray.append(0)

        let inputIds = MLXArray(tokenArray).reshaped([1, -1])
        let refId = min(text.count, voiceEmb.shape[0] - 1)
        let refS = voiceEmb[refId..<(refId + 1)]

        return (inputIds, refS, adjustedSpeed)
    }

    private func phonemize(_ text: String) -> [Int] {
        let (phonemes, _) = g2p.phonemize(text: text)
        return KittenTTSTextCleaner.cleanText(phonemes)
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> KittenTTSModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )

        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(KittenTTSConfig.self, from: configData)
        let model = KittenTTSModel(config: config)

        let weights = try loadWeights(modelDir: modelDir)
        let sanitized = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)

        try model.loadVoices(modelDir: modelDir)

        model.train(false)
        MLX.eval(model.parameters())
        return model
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (k, v) in weights {
            let nk = k.replacingOccurrences(of: ".alpha1.", with: ".alpha1_")
                      .replacingOccurrences(of: ".alpha2.", with: ".alpha2_")
            result[nk] = v
        }
        return result
    }

    private func loadVoices(modelDir: URL) throws {
        let safetensorsURL = modelDir.appendingPathComponent("voices.safetensors")
        let npzURL = modelDir.appendingPathComponent(config.voicesPath)

        if FileManager.default.fileExists(atPath: safetensorsURL.path) {
            let arrays = try MLX.loadArrays(url: safetensorsURL)
            for (k, v) in arrays {
                voices[k] = v.asType(.float32)
            }
        } else if FileManager.default.fileExists(atPath: npzURL.path) {
            throw AudioGenerationError.invalidInput(
                "voices.npz format not supported. Convert to safetensors using scripts/convert_voices_npz.py")
        } else {
            throw AudioGenerationError.invalidInput("No voices file found in model directory")
        }
    }

    private static func loadWeights(modelDir: URL) throws -> [String: MLXArray] {
        let url = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: url.path) {
            return try MLX.loadArrays(url: url)
        }
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent != "voices.safetensors" }
        for file in files {
            for (k, v) in try MLX.loadArrays(url: file) {
                allWeights[k] = v
            }
        }
        return allWeights
    }
}
