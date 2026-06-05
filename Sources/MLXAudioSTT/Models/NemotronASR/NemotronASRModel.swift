import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace

public final class NemotronASRModel: Module, STTGenerationModel {
    public let config: NemotronASRConfig
    public let preprocessConfig: NemotronASRPreprocessConfig
    public let encoderConfig: NemotronASRConformerConfig
    public let vocabulary: [String]
    public let promptDictionary: [String: Int]
    public let numPrompts: Int
    public let blankTokenID: Int
    public let defaultLanguage: String
    public let defaultAttContextSize: [Int]
    public let maxSymbols: Int?

    public var computeDType: DType = .bfloat16

    @ModuleInfo(key: "encoder") var encoder: NemotronASRConformer
    @ModuleInfo(key: "prompt_kernel") var promptKernel: NemotronASRPromptKernel
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 8192,
            temperature: 0.0,
            topP: 0.95,
            topK: 0,
            verbose: false,
            language: defaultLanguage,
            chunkDuration: 1200.0,
            minChunkDuration: 1.0
        )
    }

    public init(_ config: NemotronASRConfig) {
        self.config = config
        self.preprocessConfig = config.preprocessor
        self.encoderConfig = config.encoder
        self.vocabulary = config.vocabulary
        self.promptDictionary = config.prompt.promptDictionary
        self.numPrompts = config.prompt.numPrompts
        self.blankTokenID = config.decoder.vocabSize
        self.defaultLanguage = config.defaultLanguage
        self.defaultAttContextSize = config.defaultAttContextSize
        self.maxSymbols = config.maxSymbols

        self._encoder.wrappedValue = NemotronASRConformer(args: config.encoder)
        self._promptKernel.wrappedValue = NemotronASRPromptKernel(
            dModel: config.encoder.dModel,
            numPrompts: config.prompt.numPrompts,
            promptHidden: config.prompt.promptHidden
        )
        self._decoder.wrappedValue = ParakeetPredictNetwork(
            args: ParakeetPredictConfig(
                blankAsPad: config.decoder.blankAsPad,
                vocabSize: config.decoder.vocabSize,
                prednet: ParakeetPredictNetworkConfig(
                    predHidden: config.decoder.predHidden,
                    predRnnLayers: config.decoder.predRnnLayers
                )
            )
        )
        self._joint.wrappedValue = ParakeetJointNetwork(
            args: ParakeetJointConfig(
                numClasses: config.joint.numClasses,
                vocabulary: config.vocabulary,
                jointnet: ParakeetJointNetworkConfig(
                    jointHidden: config.joint.jointHidden,
                    activation: config.joint.activation,
                    encoderHidden: config.joint.encoderHidden,
                    predHidden: config.joint.predHidden
                )
            )
        )
    }

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        let startTime = CFAbsoluteTimeGetCurrent()
        let audio1D = normalizeAudioToMono(audio).asType(.float32)
        let sampleRate = preprocessConfig.sampleRate
        let totalSamples = audio1D.shape[0]
        let audioDuration = Double(totalSamples) / Double(sampleRate)
        let chunkDuration = Double(generationParameters.chunkDuration)
        let result: ParakeetAlignedResult

        if chunkDuration <= 0 || audioDuration <= chunkDuration {
            result = decodeChunk(audio1D, language: generationParameters.language)
        } else {
            let chunkSamples = max(1, Int(chunkDuration * Double(sampleRate)))
            let overlapDuration = 2.0
            let overlapSamples = max(0, min(chunkSamples - 1, Int(overlapDuration * Double(sampleRate))))
            let stepSamples = max(1, chunkSamples - overlapSamples)

            var allTokens: [ParakeetAlignedToken] = []
            var start = 0
            while start < totalSamples {
                let end = min(start + chunkSamples, totalSamples)
                let chunkResult = decodeChunk(audio1D[start..<end], language: generationParameters.language)
                var chunkTokens = flattenTokens(from: chunkResult)
                let chunkOffset = Double(start) / Double(sampleRate)
                for i in chunkTokens.indices {
                    chunkTokens[i].start += chunkOffset
                }

                allTokens = mergeTokenSequences(
                    existing: allTokens,
                    incoming: chunkTokens,
                    overlapDuration: overlapDuration
                )
                start += stepSamples
            }
            result = ParakeetAlignment.sentencesToResult(ParakeetAlignment.tokensToSentences(allTokens))
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return STTOutput(
            text: result.text,
            segments: result.segments,
            language: generationParameters.language,
            totalTime: elapsed
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let audio1D = self.normalizeAudioToMono(audio).asType(.float32)
            let sampleRate = self.preprocessConfig.sampleRate
            let totalSamples = audio1D.shape[0]
            let audioDuration = Double(totalSamples) / Double(sampleRate)
            let requestedChunk = Double(generationParameters.chunkDuration)
            let chunkDuration = requestedChunk >= 1199 ? 5.0 : max(0.5, requestedChunk)
            let overlapDuration = 1.0

            let chunkSamples = max(1, Int(chunkDuration * Double(sampleRate)))
            let overlapSamples = max(0, min(chunkSamples - 1, Int(overlapDuration * Double(sampleRate))))
            let stepSamples = max(1, chunkSamples - overlapSamples)

            var allTokens: [ParakeetAlignedToken] = []
            var previousText = ""
            var start = 0

            while start < totalSamples {
                let end = min(start + chunkSamples, totalSamples)
                let isLast = end >= totalSamples
                let chunkResult = self.decodeChunk(audio1D[start..<end], language: generationParameters.language)
                var chunkTokens = self.flattenTokens(from: chunkResult)
                let chunkOffset = Double(start) / Double(sampleRate)
                for i in chunkTokens.indices {
                    chunkTokens[i].start += chunkOffset
                }

                allTokens = self.mergeTokenSequences(
                    existing: allTokens,
                    incoming: chunkTokens,
                    overlapDuration: overlapDuration
                )

                let currentResult = ParakeetAlignment.sentencesToResult(
                    ParakeetAlignment.tokensToSentences(allTokens)
                )
                let fullText = currentResult.text
                let nextText = fullText.hasPrefix(previousText)
                    ? String(fullText.dropFirst(previousText.count))
                    : fullText
                previousText = fullText

                if !nextText.isEmpty {
                    continuation.yield(.token(nextText))
                }

                if isLast {
                    continuation.yield(
                        .result(
                            STTOutput(
                                text: currentResult.text,
                                segments: currentResult.segments,
                                language: generationParameters.language,
                                totalTime: audioDuration
                            )
                        )
                    )
                    continuation.finish()
                    return
                }

                start += stepSamples
            }

            continuation.yield(
                .result(
                    STTOutput(
                        text: previousText,
                        language: generationParameters.language,
                        totalTime: audioDuration
                    )
                )
            )
            continuation.finish()
        }
    }

    func decode(
        mel: MLXArray,
        language: String? = nil,
        attContextSize: [Int]? = nil
    ) -> ParakeetAlignedResult {
        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        assert(
            features.ndim == 3 && features.shape[2] == preprocessConfig.features,
            "Nemotron ASR input feature shape mismatch: expected [B, T, \(preprocessConfig.features)], got \(features.shape)"
        )

        features = features.asType(computeDType)
        let encoded = encoder(features, attContextSize: attContextSize ?? defaultAttContextSize)
        let prompted = applyPrompt(encoded.0, language: language)
        eval(prompted, encoded.1)

        let frameSeconds = Double(encoderConfig.subsamplingFactor * preprocessConfig.hopLength)
            / Double(preprocessConfig.sampleRate)
        var results: [ParakeetAlignedToken] = []
        let maxLength = Int(encoded.1[0].item(Int32.self))
        var lastToken = blankTokenID
        var decoderState: ParakeetLSTMState?
        var time = 0
        var newSymbols = 0

        while time < maxLength {
            let frame = prompted[0..., time..<(time + 1), 0...]
            let currentToken: MLXArray? = lastToken == blankTokenID
                ? nil
                : MLXArray(Int32(lastToken)).reshaped([1, 1]).asType(.int32)

            let decoderOutput = decoder(currentToken, state: decoderState)
            let pred = decoderOutput.0.asType(frame.dtype)
            let proposedState: ParakeetLSTMState = (
                hidden: decoderOutput.1.hidden?.asType(frame.dtype),
                cell: decoderOutput.1.cell?.asType(frame.dtype)
            )

            let jointOutput = joint(frame, pred)
            eval(jointOutput)
            let token = jointOutput.argMax(axis: -1).item(Int.self)
            let step = ParakeetDecodingLogic.rnntStep(
                predictedToken: token,
                blankToken: blankTokenID,
                time: time,
                newSymbols: newSymbols,
                maxSymbols: maxSymbols
            )

            if step.emittedToken {
                lastToken = token
                decoderState = proposedState
                if !NemotronASRTokenizer.isSpecialToken(token, vocabulary: vocabulary) {
                    results.append(
                        ParakeetAlignedToken(
                            id: token,
                            text: NemotronASRTokenizer.decode(tokens: [token], vocabulary: vocabulary),
                            start: Double(time) * frameSeconds,
                            duration: frameSeconds
                        )
                    )
                }
            }

            time = step.nextTime
            newSymbols = step.nextNewSymbols
        }

        let aligned = ParakeetAlignment.sentencesToResult(ParakeetAlignment.tokensToSentences(results))
        if aligned.text.isEmpty {
            return aligned
        }
        return aligned
    }

    func applyPrompt(_ encoded: MLXArray, language: String? = nil) -> MLXArray {
        let promptIndex = resolvePromptIndex(language)
        let batch = encoded.shape[0]
        let time = encoded.shape[1]
        let promptIDs = MLXArray(Array(repeating: Int32(promptIndex), count: batch * time))
            .reshaped([batch, time])
            .expandedDimensions(axis: 2)
        let promptRange = MLX.arange(numPrompts, dtype: .int32).reshaped([1, 1, numPrompts])
        let oneHot = MLX.where(promptRange .== promptIDs, MLXArray(Float(1)), MLXArray(Float(0))).asType(encoded.dtype)
        let conditioned = MLX.concatenated([encoded, oneHot], axis: 2)
        return promptKernel(conditioned)
    }

    func resolvePromptIndex(_ language: String?) -> Int {
        let resolvedLanguage = language ?? defaultLanguage
        if let index = promptDictionary[resolvedLanguage] {
            return index
        }
        if let index = promptDictionary[defaultLanguage] {
            return index
        }
        return 0
    }

    private func normalizeAudioToMono(_ audio: MLXArray) -> MLXArray {
        audio.ndim > 1 ? audio.mean(axis: -1) : audio
    }

    private func decodeChunk(_ chunkAudio: MLXArray, language: String?) -> ParakeetAlignedResult {
        let mel = NemotronASRAudio.logMelSpectrogram(chunkAudio, config: preprocessConfig)
        return decode(mel: mel, language: language)
    }

    private func flattenTokens(from result: ParakeetAlignedResult) -> [ParakeetAlignedToken] {
        result.sentences.flatMap { $0.tokens }
    }

    private func mergeTokenSequences(
        existing: [ParakeetAlignedToken],
        incoming: [ParakeetAlignedToken],
        overlapDuration: Double
    ) -> [ParakeetAlignedToken] {
        if existing.isEmpty { return incoming }
        if incoming.isEmpty { return existing }

        do {
            return try ParakeetAlignment.mergeLongestContiguous(existing, incoming, overlapDuration: overlapDuration)
        } catch {
            return ParakeetAlignment.mergeLongestCommonSubsequence(existing, incoming, overlapDuration: overlapDuration)
        }
    }
}

final class NemotronASRPromptKernel: Module {
    @ModuleInfo(key: "0") var linear0: Linear
    @ModuleInfo(key: "2") var linear2: Linear

    init(dModel: Int, numPrompts: Int, promptHidden: Int) {
        self._linear0.wrappedValue = Linear(dModel + numPrompts, promptHidden)
        self._linear2.wrappedValue = Linear(promptHidden, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(relu(linear0(x)))
    }
}

public extension NemotronASRModel {
    private static func normalizedConfigData(_ rawData: Data) -> Data {
        guard var text = String(data: rawData, encoding: .utf8) else {
            return rawData
        }

        text = text.replacingOccurrences(of: "-Infinity", with: "null")
        text = text.replacingOccurrences(of: "Infinity", with: "null")
        text = text.replacingOccurrences(of: "NaN", with: "null")
        return Data(text.utf8)
    }

    static func fromDirectory(
        _ modelDir: URL,
        computeDType: DType = .bfloat16
    ) throws -> NemotronASRModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let rawConfigData = try Data(contentsOf: configURL)
        let configData = normalizedConfigData(rawConfigData)
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: configData)
        let quantConfig = try JSONDecoder().decode(NemotronASRQuantizationConfig.self, from: configData)

        let model = NemotronASRModel(config)
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensors = files.filter { $0.pathExtension == "safetensors" }
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let sanitized = sanitize(weights: weights)

        if let perLayerQuant = quantConfig.perLayerQuantization {
            quantize(model: model) { path, _ in
                if sanitized["\(path).scales"] != nil {
                    return perLayerQuant.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        model.computeDType = computeDType

        let casted = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value -> (String, MLXArray) in
                guard value.dtype.isFloatingPoint, value.dtype != computeDType else {
                    return (key, value)
                }
                return (key, value.asType(computeDType))
            }
        )
        try model.update(parameters: ModuleParameters.unflattened(casted), verify: .noUnusedKeys)

        model.train(false)
        eval(model)
        return model
    }

    static func fromPretrained(
        _ modelPath: String = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
        computeDType: DType = .bfloat16,
        cache: HubCache = .default
    ) async throws -> NemotronASRModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "NemotronASRModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try fromDirectory(modelDir, computeDType: computeDType)
    }
}

private extension NemotronASRModel {
    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            guard let remapped = remapKey(key) else { continue }
            sanitized[remapped] = value
        }

        return sanitized
    }

    static func remapKey(_ key: String) -> String? {
        var newKey = key
        newKey = newKey.replacingOccurrences(of: "joint.joint_net.2.", with: "joint.joint_net.")
        newKey = newKey.replacingOccurrences(of: ".pos_bias_u", with: ".posBiasU")
        newKey = newKey.replacingOccurrences(of: ".pos_bias_v", with: ".posBiasV")

        if let converted = remapPreEncodeConvListKey(newKey) {
            newKey = converted
        } else if shouldSkipPreEncodeConvListKey(newKey) {
            return nil
        }

        return newKey
    }

    static func remapPreEncodeConvListKey(_ key: String) -> String? {
        let pieces = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard pieces.count >= 5 else { return nil }
        guard pieces[0] == "encoder", pieces[1] == "pre_encode", pieces[2] == "conv" else { return nil }
        guard let rawIndex = Int(pieces[3]) else { return nil }

        let suffix = pieces.dropFirst(4).joined(separator: ".")

        if rawIndex == 0 {
            return "encoder.pre_encode.conv0.\(suffix)"
        }
        if rawIndex < 2 {
            return nil
        }

        let shifted = rawIndex - 2
        let block = shifted / 3
        let mod = shifted % 3

        if mod == 0 {
            return "encoder.pre_encode.depthwise_layers.\(block).\(suffix)"
        }
        if mod == 1 {
            return "encoder.pre_encode.pointwise_layers.\(block).\(suffix)"
        }

        return nil
    }

    static func shouldSkipPreEncodeConvListKey(_ key: String) -> Bool {
        let pieces = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard pieces.count >= 5 else { return false }
        guard pieces[0] == "encoder", pieces[1] == "pre_encode", pieces[2] == "conv" else { return false }
        guard let rawIndex = Int(pieces[3]), rawIndex >= 2 else { return false }

        let shifted = rawIndex - 2
        return shifted % 3 == 2
    }
}

private struct NemotronASRQuantizationConfig: Decodable {
    let perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    init(from decoder: Decoder) throws {
        if let base = try? BaseConfiguration(from: decoder) {
            self.perLayerQuantization = base.perLayerQuantization
            return
        }

        struct FlatQuantization: Decodable {
            let groupSize: Int
            let bits: Int
            enum CodingKeys: String, CodingKey {
                case groupSize = "group_size"
                case bits
            }
        }
        enum Keys: String, CodingKey { case quantization }
        let container = try decoder.container(keyedBy: Keys.self)
        if let q = try? container.decode(FlatQuantization.self, forKey: .quantization) {
            self.perLayerQuantization = BaseConfiguration.PerLayerQuantization(
                quantization: BaseConfiguration.Quantization(groupSize: q.groupSize, bits: q.bits),
                perLayerQuantization: [:]
            )
        } else {
            self.perLayerQuantization = nil
        }
    }
}
