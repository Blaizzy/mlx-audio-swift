import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace

public final class ParakeetModel: Module, STTGenerationModel {
    public enum Variant: Sendable {
        case tdt
        case tdtCtc
        case rnnt
        case ctc
    }

    public let variant: Variant
    public let preprocessConfig: ParakeetPreprocessConfig
    public let encoderConfig: ParakeetConformerConfig

    public let vocabulary: [String]
    public let durations: [Int]
    public let maxSymbols: Int?

    @ModuleInfo(key: "encoder") var encoder: ParakeetConformer
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork?
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork?
    @ModuleInfo(key: "ctc_decoder") var ctcDecoder: ParakeetConvASRDecoder?

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 8192,
            temperature: 0.0,
            topP: 0.95,
            topK: 0,
            verbose: false,
            language: "en",
            chunkDuration: 1200.0,
            minChunkDuration: 1.0
        )
    }

    private init(
        variant: Variant,
        preprocessConfig: ParakeetPreprocessConfig,
        encoderConfig: ParakeetConformerConfig,
        vocabulary: [String],
        durations: [Int],
        maxSymbols: Int?,
        decoderConfig: ParakeetPredictConfig?,
        jointConfig: ParakeetJointConfig?,
        ctcConfig: ParakeetConvASRDecoderConfig?
    ) {
        self.variant = variant
        self.preprocessConfig = preprocessConfig
        self.encoderConfig = encoderConfig
        self.vocabulary = vocabulary
        self.durations = durations
        self.maxSymbols = maxSymbols

        self._encoder.wrappedValue = ParakeetConformer(args: encoderConfig)
        if let decoderConfig {
            self._decoder.wrappedValue = ParakeetPredictNetwork(args: decoderConfig)
        } else {
            self._decoder.wrappedValue = nil
        }
        if let jointConfig {
            self._joint.wrappedValue = ParakeetJointNetwork(args: jointConfig)
        } else {
            self._joint.wrappedValue = nil
        }
        if let ctcConfig {
            self._ctcDecoder.wrappedValue = ParakeetConvASRDecoder(args: ctcConfig)
        } else {
            self._ctcDecoder.wrappedValue = nil
        }
    }

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let sampleRate = preprocessConfig.sampleRate
        let totalSamples = audio1D.shape[0]
        let audioDuration = Double(totalSamples) / Double(sampleRate)
        let chunkDuration = Double(generationParameters.chunkDuration)
        let overlapDuration = 15.0

        let result: ParakeetAlignedResult
        if chunkDuration <= 0 || audioDuration <= chunkDuration {
            result = decodeChunk(audio1D)
        } else {
            let chunkSamples = max(1, Int(chunkDuration * Double(sampleRate)))
            let overlapSamples = max(0, min(chunkSamples - 1, Int(overlapDuration * Double(sampleRate))))
            let stepSamples = max(1, chunkSamples - overlapSamples)

            var allTokens: [ParakeetAlignedToken] = []
            var start = 0
            while start < totalSamples {
                let end = min(start + chunkSamples, totalSamples)
                let chunkAudio = audio1D[start..<end]
                let chunkResult = decodeChunk(chunkAudio)

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

        return STTOutput(
            text: result.text,
            segments: result.segments,
            language: generationParameters.language
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
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
                let chunkAudio = audio1D[start..<end]
                let chunkResult = self.decodeChunk(chunkAudio)

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
                let nextText: String
                if fullText.hasPrefix(previousText) {
                    nextText = String(fullText.dropFirst(previousText.count))
                } else {
                    nextText = fullText
                }
                previousText = fullText

                if !nextText.isEmpty {
                    continuation.yield(.token(nextText))
                }

                if isLast {
                    let output = STTOutput(
                        text: currentResult.text,
                        segments: currentResult.segments,
                        language: generationParameters.language,
                        totalTime: audioDuration
                    )
                    continuation.yield(.result(output))
                    continuation.finish()
                    return
                }

                start += stepSamples
            }

            let finalOutput = STTOutput(
                text: previousText,
                segments: nil,
                language: generationParameters.language,
                totalTime: audioDuration
            )
            continuation.yield(.result(finalOutput))
            continuation.finish()
        }
    }

    func decode(mel: MLXArray) -> [ParakeetAlignedResult] {
        switch variant {
        case .tdt, .tdtCtc:
            return decodeTDT(mel: mel)
        case .rnnt:
            return decodeRNNT(mel: mel)
        case .ctc:
            return decodeCTC(mel: mel)
        }
    }

    private func decodeTDT(mel: MLXArray) -> [ParakeetAlignedResult] {
        guard let decoder, let joint else { return [] }

        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        let encoded = encoder(features)
        let batchFeatures = encoded.0
        let lengths = encoded.1
        eval(batchFeatures, lengths)

        var results: [ParakeetAlignedResult] = []
        let batchSize = batchFeatures.shape[0]
        let blankToken = vocabulary.count

        for b in 0..<batchSize {
            let featureSeq = batchFeatures[b..<(b + 1)]
            let maxLength = Int(lengths[b].item(Int32.self))

            var lastToken = blankToken
            var hypothesis: [ParakeetAlignedToken] = []

            var t = 0
            var newSymbols = 0
            var state: ParakeetLSTMState?

            while t < maxLength {
                let frame = featureSeq[0..., t..<(t + 1), 0...]
                let currentToken: MLXArray? = lastToken == blankToken ? nil : MLXArray(lastToken).reshaped([1, 1]).asType(.int32)

                let decoderOut = decoder(currentToken, state: state)
                let pred = decoderOut.0.asType(frame.dtype)
                let proposedState: ParakeetLSTMState = (
                    hidden: decoderOut.1.hidden?.asType(frame.dtype),
                    cell: decoderOut.1.cell?.asType(frame.dtype)
                )

                let jointOut = joint(frame, pred)
                let tokenLogits = jointOut[0, 0, 0, ..<(blankToken + 1)]
                let durationLogits = jointOut[0, 0, 0, (blankToken + 1)...]
                let token = tokenLogits.argMax(axis: -1).item(Int.self)
                let decision = durationLogits.argMax(axis: -1).item(Int.self)
                let step = ParakeetDecodingLogic.tdtStep(
                    predictedToken: token,
                    blankToken: blankToken,
                    decisionIndex: decision,
                    durations: durations,
                    time: t,
                    newSymbols: newSymbols,
                    maxSymbols: maxSymbols
                )

                if token != blankToken {
                    let start = frameTimeSeconds(frameIndex: t)
                    let duration = frameTimeSeconds(frameIndex: step.jump)
                    hypothesis.append(
                        ParakeetAlignedToken(
                            id: token,
                            text: ParakeetTokenizer.decode(tokens: [token], vocabulary: vocabulary),
                            start: start,
                            duration: duration
                        )
                    )
                    lastToken = token
                    state = proposedState
                }

                t = step.nextTime
                newSymbols = step.nextNewSymbols
            }

            results.append(
                ParakeetAlignment.sentencesToResult(
                    ParakeetAlignment.tokensToSentences(hypothesis)
                )
            )
        }

        return results
    }

    private func decodeRNNT(mel: MLXArray) -> [ParakeetAlignedResult] {
        guard let decoder, let joint else { return [] }

        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        let encoded = encoder(features)
        let batchFeatures = encoded.0
        let lengths = encoded.1
        eval(batchFeatures, lengths)

        var results: [ParakeetAlignedResult] = []
        let batchSize = batchFeatures.shape[0]
        let blankToken = vocabulary.count

        for b in 0..<batchSize {
            let featureSeq = batchFeatures[b..<(b + 1)]
            let maxLength = Int(lengths[b].item(Int32.self))

            var lastToken = blankToken
            var hypothesis: [ParakeetAlignedToken] = []

            var t = 0
            var newSymbols = 0
            var state: ParakeetLSTMState?

            while t < maxLength {
                let frame = featureSeq[0..., t..<(t + 1), 0...]
                let currentToken: MLXArray? = lastToken == blankToken ? nil : MLXArray(lastToken).reshaped([1, 1]).asType(.int32)

                let decoderOut = decoder(currentToken, state: state)
                let pred = decoderOut.0.asType(frame.dtype)
                let proposedState: ParakeetLSTMState = (
                    hidden: decoderOut.1.hidden?.asType(frame.dtype),
                    cell: decoderOut.1.cell?.asType(frame.dtype)
                )

                let jointOut = joint(frame, pred)
                let token = jointOut.argMax(axis: -1).item(Int.self)
                let step = ParakeetDecodingLogic.rnntStep(
                    predictedToken: token,
                    blankToken: blankToken,
                    time: t,
                    newSymbols: newSymbols,
                    maxSymbols: maxSymbols
                )

                if step.emittedToken {
                    let start = frameTimeSeconds(frameIndex: t)
                    let duration = frameTimeSeconds(frameIndex: 1)
                    hypothesis.append(
                        ParakeetAlignedToken(
                            id: token,
                            text: ParakeetTokenizer.decode(tokens: [token], vocabulary: vocabulary),
                            start: start,
                            duration: duration
                        )
                    )

                    lastToken = token
                    state = proposedState
                }

                t = step.nextTime
                newSymbols = step.nextNewSymbols
            }

            results.append(
                ParakeetAlignment.sentencesToResult(
                    ParakeetAlignment.tokensToSentences(hypothesis)
                )
            )
        }

        return results
    }

    private func decodeCTC(mel: MLXArray) -> [ParakeetAlignedResult] {
        guard let ctcDecoder else { return [] }

        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        let encoded = encoder(features)
        let batchFeatures = encoded.0
        let lengths = encoded.1
        let logits = ctcDecoder(batchFeatures)
        eval(logits, lengths)

        var results: [ParakeetAlignedResult] = []
        let blankToken = vocabulary.count

        for b in 0..<logits.shape[0] {
            let featLen = Int(lengths[b].item(Int32.self))
            let pred = logits[b, ..<featLen, 0...]
            let bestTokens = pred.argMax(axis: 1)

            let ids: [Int] = (0..<featLen).map { bestTokens[$0].item(Int.self) }
            let spans = ParakeetDecodingLogic.ctcSpans(bestTokens: ids, blankToken: blankToken)
            let hypothesis: [ParakeetAlignedToken] = spans.map { span in
                let start = frameTimeSeconds(frameIndex: span.startFrame)
                let end = frameTimeSeconds(frameIndex: span.endFrame)
                return ParakeetAlignedToken(
                    id: span.token,
                    text: ParakeetTokenizer.decode(tokens: [span.token], vocabulary: vocabulary),
                    start: start,
                    duration: end - start
                )
            }

            results.append(
                ParakeetAlignment.sentencesToResult(
                    ParakeetAlignment.tokensToSentences(hypothesis)
                )
            )
        }

        return results
    }

    private func frameTimeSeconds(frameIndex: Int) -> Double {
        Double(frameIndex * encoderConfig.subsamplingFactor * preprocessConfig.hopLength) / Double(preprocessConfig.sampleRate)
    }

    private func decodeChunk(_ chunkAudio: MLXArray) -> ParakeetAlignedResult {
        let mel = ParakeetAudio.logMelSpectrogram(chunkAudio, config: preprocessConfig)
        return decode(mel: mel)[0]
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

public extension ParakeetModel {
    private static func normalizedConfigData(_ rawData: Data) -> Data {
        guard var text = String(data: rawData, encoding: .utf8) else {
            return rawData
        }

        // Some exported NeMo configs use non-standard JSON float tokens.
        text = text.replacingOccurrences(of: "-Infinity", with: "null")
        text = text.replacingOccurrences(of: "Infinity", with: "null")
        text = text.replacingOccurrences(of: "NaN", with: "null")
        return Data(text.utf8)
    }

    static func fromDirectory(_ modelDir: URL) throws -> ParakeetModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let rawConfigData = try Data(contentsOf: configURL)
        let configData = normalizedConfigData(rawConfigData)
        let rawConfig = try JSONDecoder().decode(ParakeetRawConfig.self, from: configData)
        let quantConfig = try JSONDecoder().decode(ParakeetQuantizationConfig.self, from: configData)
        let variant = try ParakeetVariantResolver.resolve(rawConfig)

        let model: ParakeetModel
        switch variant {
        case .tdt:
            let cfg = try ParakeetConfigParser.parseTDT(rawConfig)
            model = ParakeetModel(
                variant: .tdt,
                preprocessConfig: cfg.preprocessor,
                encoderConfig: cfg.encoder,
                vocabulary: cfg.joint.vocabulary,
                durations: cfg.decoding.durations,
                maxSymbols: cfg.decoding.greedy?.maxSymbols,
                decoderConfig: cfg.decoder,
                jointConfig: cfg.joint,
                ctcConfig: nil
            )
        case .tdtCtc:
            let cfg = try ParakeetConfigParser.parseTDTCTC(rawConfig)
            model = ParakeetModel(
                variant: .tdtCtc,
                preprocessConfig: cfg.preprocessor,
                encoderConfig: cfg.encoder,
                vocabulary: cfg.joint.vocabulary,
                durations: cfg.decoding.durations,
                maxSymbols: cfg.decoding.greedy?.maxSymbols,
                decoderConfig: cfg.decoder,
                jointConfig: cfg.joint,
                ctcConfig: cfg.auxCTC.decoder
            )
        case .rnnt:
            let cfg = try ParakeetConfigParser.parseRNNT(rawConfig)
            model = ParakeetModel(
                variant: .rnnt,
                preprocessConfig: cfg.preprocessor,
                encoderConfig: cfg.encoder,
                vocabulary: cfg.joint.vocabulary,
                durations: [1],
                maxSymbols: cfg.decoding.greedy?.maxSymbols,
                decoderConfig: cfg.decoder,
                jointConfig: cfg.joint,
                ctcConfig: nil
            )
        case .ctc:
            let cfg = try ParakeetConfigParser.parseCTC(rawConfig)
            model = ParakeetModel(
                variant: .ctc,
                preprocessConfig: cfg.preprocessor,
                encoderConfig: cfg.encoder,
                vocabulary: cfg.decoder.vocabulary,
                durations: [1],
                maxSymbols: nil,
                decoderConfig: nil,
                jointConfig: nil,
                ctcConfig: cfg.decoder
            )
        }

        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensors = files.filter { $0.pathExtension == "safetensors" }
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let sanitized = sanitize(weights: weights, variant: model.variant)

        if let perLayerQuant = quantConfig.perLayerQuantization {
            quantize(model: model) { path, _ in
                if sanitized["\(path).scales"] != nil {
                    return perLayerQuant.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)
        return model
    }

    static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> ParakeetModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "ParakeetModel",
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
        return try fromDirectory(modelDir)
    }
}

private extension ParakeetModel {
    static func sanitize(weights: [String: MLXArray], variant: Variant) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            guard let remapped = remapKey(key, variant: variant) else { continue }
            sanitized[remapped] = value
        }

        return sanitized
    }

    static func remapKey(_ key: String, variant: Variant) -> String? {
        var newKey = key

        // CTC-only checkpoints keep decoder at top level; Swift model uses ctc_decoder.
        if variant == .ctc, newKey.hasPrefix("decoder.") {
            newKey = "ctc_decoder." + newKey.dropFirst("decoder.".count)
        }

        // ConvASRDecoder list index -> single module path.
        newKey = newKey.replacingOccurrences(of: ".decoder_layers.0.", with: ".decoder_layers.")

        // Joint net linear is index 2 in the source list.
        newKey = newKey.replacingOccurrences(of: "joint.joint_net.2.", with: "joint.joint_net.")
        newKey = newKey.replacingOccurrences(of: ".pos_bias_u", with: ".posBiasU")
        newKey = newKey.replacingOccurrences(of: ".pos_bias_v", with: ".posBiasV")

        // DwStridingSubsampling list remap:
        // conv.0 -> conv0
        // conv.(2 + 3n) -> depthwise_layers.n
        // conv.(3 + 3n) -> pointwise_layers.n
        // conv.(4 + 3n) are ReLU placeholders (no params), skip if encountered.
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

private struct ParakeetQuantizationConfig: Decodable {
    let perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    init(from decoder: Decoder) throws {
        let base = try? BaseConfiguration(from: decoder)
        self.perLayerQuantization = base?.perLayerQuantization
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}
