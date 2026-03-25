import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public final class KokoroModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: KokoroConfig
    public let sampleRate: Int
    public let defaultGenerationParameters: GenerateParameters

    @ModuleInfo var bert: Albert
    @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
    @ModuleInfo var predictor: KokoroProsodyPredictor
    @ModuleInfo(key: "text_encoder") var textEncoder: KokoroTextEncoder
    @ModuleInfo var decoder: KokoroDecoder

    private let modelDirectory: URL?
    private var voiceCache: [String: MLXArray] = [:]
    private let voiceCacheLock = NSLock()
    private let maxTokenCount = 510
    public private(set) var textProcessor: TextProcessor?

    /// Speed multiplier for speech rate. Higher = faster.
    public var speed: Float = 1.0

    private init(config: KokoroConfig, modelDirectory: URL? = nil, textProcessor: TextProcessor? = nil) {
        self.config = config
        self.modelDirectory = modelDirectory
        self.textProcessor = textProcessor
        self.sampleRate = config.sampleRate
        self.defaultGenerationParameters = GenerateParameters(temperature: 0, topP: 1)

        _bert = ModuleInfo(wrappedValue: Albert(config: config.plbert, vocabSize: config.nToken))
        _bertEncoder = ModuleInfo(
            wrappedValue: Linear(config.plbert.hiddenSize, config.hiddenDim),
            key: "bert_encoder"
        )
        _predictor = ModuleInfo(wrappedValue: KokoroProsodyPredictor(
            styleDim: config.styleDim, dHid: config.hiddenDim,
            nLayers: config.nLayer, maxDur: config.maxDur, dropout: 0.0
        ))
        _textEncoder = ModuleInfo(wrappedValue: KokoroTextEncoder(
            channels: config.hiddenDim, kernelSize: config.textEncoderKernelSize,
            depth: config.nLayer, nSymbols: config.nToken
        ), key: "text_encoder")
        _decoder = ModuleInfo(wrappedValue: KokoroDecoder(config: config))
    }

    /// For testing: create model from config without loading weights
    static func testInit(config: KokoroConfig) -> KokoroModel {
        KokoroModel(config: config)
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

        let globalStyle = refS[0..., 128...]
        let acousticStyle = refS[0..., ..<128]

        let d = predictor.textEncoder(dEn, style: globalStyle, textLengths: inputLengths, mask: textMask)
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

        let predAlnTrg = MLXArray.zeros([inputIds.shape[1], allIndices.shape[0]])
        predAlnTrg[allIndices, MLXArray(Array(0..<Int32(allIndices.shape[0])))] = MLXArray(Float(1))
        let predAln = predAlnTrg.expandedDimensions(axis: 0)

        let en = MLX.matmul(d.transposed(0, 2, 1), predAln)
        let (f0Pred, nPred) = predictor.predict(en, globalStyle)

        let tEn = textEncoder(inputIds, inputLengths: inputLengths, mask: textMask)
        let asr = MLX.matmul(tEn, predAln)

        let audio = decoder(asr, f0Pred, nPred, acousticStyle)
        return (audio[0], predDur)
    }

    // MARK: - Tokenizer

    func tokenize(_ text: String) -> [Int] {
        text.compactMap { config.vocab[String($0)] }
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
        let voiceName = voice ?? "af_heart"
        let voiceEmb: MLXArray
        if let refAudio {
            voiceEmb = refAudio
        } else {
            voiceEmb = try loadVoice(named: voiceName)
        }

        let inferredLang = language ?? KokoroMultilingualProcessor.languageForVoice(voiceName)

        if let multilingual = textProcessor as? KokoroMultilingualProcessor,
            let lang = inferredLang
        {
            try await multilingual.prepare(for: lang)
        }

        let phonemized: String
        if let textProcessor {
            phonemized = try textProcessor.process(text: text, language: inferredLang)
        } else {
            phonemized = text
        }

        let tokens = tokenize(phonemized)
        guard tokens.count <= maxTokenCount else {
            throw AudioGenerationError.invalidInput(
                "Input too long: \(tokens.count) tokens exceeds max \(maxTokenCount)")
        }

        var tokenArray = [Int32(0)]
        tokenArray.append(contentsOf: tokens.map { Int32($0) })
        tokenArray.append(0)

        let inputIds = MLXArray(tokenArray).reshaped([1, -1])
        let refIdx = min(tokens.count, voiceEmb.shape[0] - 1)
        let refS = voiceEmb[refIdx..<(refIdx + 1)]

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
            Task { @Sendable [weak self] in
                guard let self else {
                    continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                    return
                }
                do {
                    let audio = try await self.generate(
                        text: text, voice: voice, refAudio: refAudio,
                        refText: refText, language: language,
                        generationParameters: generationParameters
                    )
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Voice Loading

    public func loadVoice(named name: String) throws -> MLXArray {
        if let cached = voiceCacheLock.withLock({ voiceCache[name] }) { return cached }

        guard let dir = modelDirectory else {
            throw AudioGenerationError.invalidInput("Voice '\(name)' not available: no model directory")
        }

        let voicePath = dir.appendingPathComponent("voices/\(name).safetensors")
        guard FileManager.default.fileExists(atPath: voicePath.path) else {
            let available = availableVoices().joined(separator: ", ")
            throw AudioGenerationError.invalidInput(
                "Voice '\(name)' not found. Available: \(available)")
        }

        let voiceWeights = try MLX.loadArrays(url: voicePath)
        guard let voiceArray = voiceWeights["voice"] ?? voiceWeights.values.first else {
            throw AudioGenerationError.invalidInput("Voice '\(name)' file has no voice data")
        }

        var voice = voiceArray.asType(.float32)
        if voice.ndim == 3 { voice = voice.squeezed(axis: 1) }
        voiceCacheLock.withLock { voiceCache[name] = voice }
        return voice
    }

    public func availableVoices() -> [String] {
        guard let dir = modelDirectory else { return [] }
        let voicesDir = dir.appendingPathComponent("voices")
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: voicesDir.path) else {
            return []
        }
        return files
            .filter { $0.hasSuffix(".safetensors") }
            .map { String($0.dropLast(".safetensors".count)) }
            .sorted()
    }

    public func setTextProcessor(_ processor: TextProcessor?) {
        self.textProcessor = processor
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        textProcessor: TextProcessor? = nil,
        cache: HubCache = .default
    ) async throws -> KokoroModel {
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
        let config = try JSONDecoder().decode(KokoroConfig.self, from: configData)
        let model = KokoroModel(config: config, modelDirectory: modelDir, textProcessor: textProcessor)

        let weights = try loadWeights(modelDir: modelDir)
        let sanitized = model.sanitize(weights: weights)

        if let quant = config.quantization?.asTuple {
            quantizeTree(
                module: model, weights: sanitized,
                quantization: quant
            )
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)

        try await model.textProcessor?.prepare()

        model.train(false)
        MLX.eval(model.parameters())
        return model
    }

    // MARK: - Weight Sanitization

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (k, v) in weights {
            if k.contains("position_ids") { continue }

            var nk = k

            // LSTM: remap reverse FIRST to avoid partial match with forward keys
            nk = nk.replacingOccurrences(of: ".weight_ih_l0_reverse", with: ".Wx_backward")
            nk = nk.replacingOccurrences(of: ".weight_hh_l0_reverse", with: ".Wh_backward")
            nk = nk.replacingOccurrences(of: ".bias_ih_l0_reverse", with: ".bias_ih_backward")
            nk = nk.replacingOccurrences(of: ".bias_hh_l0_reverse", with: ".bias_hh_backward")
            nk = nk.replacingOccurrences(of: ".weight_ih_l0", with: ".Wx_forward")
            nk = nk.replacingOccurrences(of: ".weight_hh_l0", with: ".Wh_forward")
            nk = nk.replacingOccurrences(of: ".bias_ih_l0", with: ".bias_ih_forward")
            nk = nk.replacingOccurrences(of: ".bias_hh_l0", with: ".bias_hh_forward")

            nk = nk.replacingOccurrences(of: ".gamma", with: ".weight")
            nk = nk.replacingOccurrences(of: ".beta", with: ".bias")
            nk = nk.replacingOccurrences(of: ".alpha1.", with: ".alpha1_")
            nk = nk.replacingOccurrences(of: ".alpha2.", with: ".alpha2_")

            var value = v
            if (nk.contains("F0_proj.weight") || nk.contains("N_proj.weight")) && v.ndim == 3 {
                value = v.transposed(0, 2, 1)
            } else if nk.contains("noise_convs") && nk.hasSuffix(".weight") && v.ndim == 3 {
                value = v.transposed(0, 2, 1)
            } else if nk.hasSuffix("weight_v") && v.ndim == 3 {
                // Transpose non-canonical [outCh, kH, kW] layout
                let (o, h, w) = (v.shape[0], v.shape[1], v.shape[2])
                if !(o >= h && o >= w && h == w) {
                    value = v.transposed(0, 2, 1)
                }
            }

            result[nk] = value
        }
        return result
    }

    // MARK: - Weight Loading

    private static func loadWeights(modelDir: URL) throws -> [String: MLXArray] {
        let url = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: url.path) {
            return try MLX.loadArrays(url: url)
        }
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors"
                && !$0.lastPathComponent.contains("voices") }
        for file in files {
            for (k, v) in try MLX.loadArrays(url: file) {
                allWeights[k] = v
            }
        }
        return allWeights
    }

    // MARK: - Tree-Walking Quantization

    private static func quantizeTree(
        module: Module,
        weights: [String: MLXArray],
        quantization: (groupSize: Int, bits: Int, mode: QuantizationMode),
        path: String = ""
    ) {
        var replacements = [(String, Module)]()

        for (key, child) in module.items() {
            let childPath = path.isEmpty ? key : "\(path).\(key)"

            switch child {
            case .value(.module(let childModule)):
                if childModule is Quantizable,
                   weights["\(childPath).scales"] != nil,
                   let quantized = quantizeSingle(
                       layer: childModule,
                       groupSize: quantization.groupSize,
                       bits: quantization.bits,
                       mode: quantization.mode
                   )
                {
                    replacements.append((key, quantized))
                } else {
                    quantizeTree(
                        module: childModule, weights: weights,
                        quantization: quantization, path: childPath
                    )
                }

            case .array(let items):
                for (index, item) in items.enumerated() {
                    let indexPath = "\(childPath).\(index)"
                    switch item {
                    case .value(.module(let childModule)):
                        quantizeTree(
                            module: childModule, weights: weights,
                            quantization: quantization, path: indexPath
                        )
                    case .array(let nestedItems):
                        for (nestedIndex, nestedItem) in nestedItems.enumerated() {
                            if case .value(.module(let nestedModule)) = nestedItem {
                                quantizeTree(
                                    module: nestedModule, weights: weights,
                                    quantization: quantization,
                                    path: "\(indexPath).\(nestedIndex)"
                                )
                            }
                        }
                    default:
                        break
                    }
                }

            default:
                break
            }
        }

        if !replacements.isEmpty {
            module.update(modules: ModuleChildren.unflattened(replacements))
        }
    }
}
