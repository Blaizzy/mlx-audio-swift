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

    /// Compute dtype applied to encoder features and decoder LSTM state.
    /// Defaults to `.bfloat16` (measured ~8% wall-clock speedup on batched decode
    /// with ~0.2% word drift). Set to `.float32` via the factory method to fall back.
    public var computeDType: DType = .bfloat16

    enum TDTDecoderImplementation: Sendable {
        case serial
        case hybrid
    }

    enum EncoderExecutionImplementation: Sendable {
        case plain
        case compiled
    }

    struct TDTTraceStep: Sendable, Equatable {
        let row: Int
        let time: Int
        let newSymbols: Int
        let token: Int
        let decisionIndex: Int
        let committedState: Bool
    }

    @ModuleInfo(key: "encoder") var encoder: ParakeetConformer
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork?
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork?
    @ModuleInfo(key: "ctc_decoder") var ctcDecoder: ParakeetConvASRDecoder?

    var tdtDecoderImplementation: TDTDecoderImplementation?
    var encoderExecutionImplementation: EncoderExecutionImplementation?
    var tdtTraceEmitter: (@Sendable (TDTTraceStep) -> Void)?
    private var compiledEncoderFeaturesByShape: [String: @Sendable (MLXArray) -> MLXArray] = [:]

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

    private var blankTokenId: Int {
        vocabulary.count
    }

    private lazy var compiledTDTStep = makeCompiledTDTStep(
        decoder: self.decoder,
        joint: self.joint,
        blankTokenId: self.blankTokenId
    )

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
        let audio1D = normalizeAudioToMono(audio)
        let sampleRate = preprocessConfig.sampleRate
        let totalSamples = audio1D.shape[0]
        let audioDuration = Double(totalSamples) / Double(sampleRate)
        let chunkDuration = Double(generationParameters.chunkDuration)
        let overlapDuration = 2.0

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

    public func generateBatch(
        audios: [MLXArray],
        generationParameters: STTGenerateParameters = STTGenerateParameters()
    ) throws -> [STTOutput] {
        guard !audios.isEmpty else {
            throw STTError.invalidInput("Parakeet generateBatch requires at least one chunk-sized audio input.")
        }

        let previousTDTDecoderImplementation = tdtDecoderImplementation
        let previousEncoderExecutionImplementation = encoderExecutionImplementation
        if previousTDTDecoderImplementation == nil {
            tdtDecoderImplementation = audios.count > 1 ? .hybrid : .serial
        }
        if previousEncoderExecutionImplementation == nil {
            encoderExecutionImplementation = .compiled
        }
        defer {
            tdtDecoderImplementation = previousTDTDecoderImplementation
            encoderExecutionImplementation = previousEncoderExecutionImplementation
        }

        let batchFeatures = makeBatchFeatures(audios)
        let results = decode(mel: batchFeatures.features, lengths: batchFeatures.lengths)
        return results.map {
            STTOutput(
                text: $0.text,
                segments: $0.segments,
                language: generationParameters.language
            )
        }
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let audio1D = self.normalizeAudioToMono(audio)
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

    func decode(mel: MLXArray, lengths: MLXArray? = nil) -> [ParakeetAlignedResult] {
        switch variant {
        case .tdt, .tdtCtc:
            return decodeTDT(mel: mel, lengths: lengths)
        case .rnnt:
            return decodeRNNT(mel: mel, lengths: lengths)
        case .ctc:
            return decodeCTC(mel: mel, lengths: lengths)
        }
    }

    func predictTDTToken(_ token: MLXArray?, state: ParakeetLSTMState? = nil) -> (MLXArray, ParakeetLSTMState)? {
        guard let decoder else { return nil }
        return decoder(token, state: state)
    }

    func predictTDTBatch(
        _ tokenIds: MLXArray,
        state: ParakeetLSTMState? = nil,
        blankToken: Int32
    ) -> (MLXArray, ParakeetLSTMState)? {
        guard let decoder else { return nil }
        return decoder.predictBatched(tokenIds, state: state, blankToken: blankToken)
    }

    func encodeBatchFeatures(_ features: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let resolvedLengths = lengths ?? MLXArray(Array(repeating: Int32(features.shape[1]), count: features.shape[0])).asType(.int32)
        switch encoderExecutionImplementation ?? .plain {
        case .plain:
            return encoder(features, lengths: resolvedLengths)
        case .compiled:
            let encodedFeatures = compiledEncoderFeatures(for: features)(features)
            let encodedLengths = computeEncodedLengths(from: resolvedLengths)
            return (encodedFeatures, encodedLengths)
        }
    }

    func compiledEncoderFeatures(for features: MLXArray) -> @Sendable (MLXArray) -> MLXArray {
        let key = "\(features.shape)-\(features.dtype)"
        if let compiled = compiledEncoderFeaturesByShape[key] {
            return compiled
        }

        let compiled: @Sendable (MLXArray) -> MLXArray = compile { [self] features in
            self.encoder(features).0
        }
        compiledEncoderFeaturesByShape[key] = compiled
        return compiled
    }

    func computeEncodedLengths(from lengths: MLXArray) -> MLXArray {
        guard encoder.preEncodeDw != nil else {
            return lengths.asType(.int32)
        }

        let samplingNum = Int(log2(Double(encoderConfig.subsamplingFactor)))
        var outLengths = lengths.asType(.float32)
        for _ in 0..<samplingNum {
            outLengths = floor((outLengths + Float(-1)) / Float(2)) + 1
        }
        return outLengths.asType(.int32)
    }

    func decodeEncoded(batchFeatures: MLXArray, lengths: MLXArray) -> [ParakeetAlignedResult] {
        switch variant {
        case .tdt, .tdtCtc:
            return decodeTDTEncoded(batchFeatures: batchFeatures, lengths: lengths)
        case .rnnt:
            return decodeRNNTEncoded(batchFeatures: batchFeatures, lengths: lengths)
        case .ctc:
            return decodeCTCEncoded(batchFeatures: batchFeatures, lengths: lengths)
        }
    }

    private func decodeTDT(mel: MLXArray, lengths: MLXArray? = nil) -> [ParakeetAlignedResult] {
        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        assert(
            features.ndim == 3 && features.shape[2] == preprocessConfig.features,
            "Parakeet TDT input feature shape mismatch: expected [B, T, \(preprocessConfig.features)], got \(features.shape)"
        )

        features = features.asType(computeDType)
        let encoded = encodeBatchFeatures(features, lengths: lengths)
        return decodeTDTEncoded(batchFeatures: encoded.0, lengths: encoded.1)
    }

    private func decodeTDTEncoded(batchFeatures: MLXArray, lengths: MLXArray) -> [ParakeetAlignedResult] {
        guard let decoder, let joint else { return [] }

        assert(
            batchFeatures.ndim == 3 && batchFeatures.shape[2] == encoderConfig.dModel,
            "Parakeet TDT encoder output shape mismatch: expected last dim \(encoderConfig.dModel), got \(batchFeatures.shape)"
        )
        eval(batchFeatures, lengths)

        switch tdtDecoderImplementation ?? .serial {
        case .serial:
            return decodeTDTSerial(batchFeatures: batchFeatures, lengths: lengths, decoder: decoder, joint: joint)
        case .hybrid:
            return decodeTDTHybrid(batchFeatures: batchFeatures, lengths: lengths, decoder: decoder, joint: joint)
        }
    }

    private func decodeTDTSerial(
        batchFeatures: MLXArray,
        lengths: MLXArray,
        decoder: ParakeetPredictNetwork,
        joint: ParakeetJointNetwork
    ) -> [ParakeetAlignedResult] {

        var results: [ParakeetAlignedResult] = []
        let batchSize = batchFeatures.shape[0]
        let blankToken = blankTokenId

        for b in 0..<batchSize {
            let featureSeq = batchFeatures[b..<(b + 1)]
            let maxLength = Int(lengths[b].item(Int32.self))

            var lastToken = blankToken
            var hypothesis: [ParakeetAlignedToken] = []

            var t = 0
            var newSymbols = 0
            var state = makeInitialDecoderState(batchSize: 1, dtype: computeDType)
            var currentToken = MLXArray(Int32(lastToken)).reshaped([1, 1]).asType(.int32)

            while t < maxLength {
                let frame = featureSeq[0..., t..<(t + 1), 0...]

                let stepOutputs = compiledTDTStep([
                    frame,
                    currentToken,
                    state.hidden!,
                    state.cell!
                ])
                let decisions = stepOutputs[0]
                let hidden = stepOutputs[1]
                let cell = stepOutputs[2]
                MLX.eval(decisions, hidden, cell)
                let decisionPair = decisions.asArray(Int32.self)
                let token = Int(decisionPair[0])
                let decisionIndex = Int(decisionPair[1])
                let step = ParakeetDecodingLogic.tdtStep(
                    predictedToken: token,
                    blankToken: blankToken,
                    decisionIndex: decisionIndex,
                    durations: durations,
                    time: t,
                    newSymbols: newSymbols,
                    maxSymbols: maxSymbols
                )

                tdtTraceEmitter?(
                    TDTTraceStep(
                        row: b,
                        time: t,
                        newSymbols: newSymbols,
                        token: token,
                        decisionIndex: decisionIndex,
                        committedState: token != blankToken
                    )
                )

                if token != blankToken {
                    lastToken = token
                    state = (hidden: hidden, cell: cell)
                    currentToken = MLXArray(Int32(lastToken)).reshaped([1, 1]).asType(.int32)
                    if !ParakeetTokenizer.isSpecialToken(token, vocabulary: vocabulary) {
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
                    }
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

    private func decodeTDTHybrid(
        batchFeatures: MLXArray,
        lengths: MLXArray,
        decoder: ParakeetPredictNetwork,
        joint: ParakeetJointNetwork
    ) -> [ParakeetAlignedResult] {
        let batchSize = batchFeatures.shape[0]
        let blankToken = vocabulary.count
        let maxLengthByRow = lengths.asArray(Int32.self).map(Int.init)
        let hiddenSize = decoder.prediction.decRnn.layers.first?.hiddenSize ?? decoder.predHidden
        let numLayers = decoder.prediction.decRnn.numLayers
        let stateShape = [numLayers, batchSize, hiddenSize]

        let stateDType: DType = computeDType
        var fullState: ParakeetLSTMState = (
            hidden: MLXArray.zeros(stateShape, dtype: stateDType),
            cell: MLXArray.zeros(stateShape, dtype: stateDType)
        )

        var timeByRow = Array(repeating: 0, count: batchSize)
        var newSymbolsByRow = Array(repeating: 0, count: batchSize)
        var lastTokenByRow = Array(repeating: blankToken, count: batchSize)
        var doneByRow = Array(repeating: false, count: batchSize)
        var hypothesisByRow = Array(repeating: [ParakeetAlignedToken](), count: batchSize)

        while true {
            let activeRows = (0..<batchSize).filter { row in
                let isActive = timeByRow[row] < maxLengthByRow[row]
                doneByRow[row] = !isActive
                return isActive
            }

            if activeRows.isEmpty {
                break
            }

            let activeFrames = gatherActiveFrames(batchFeatures: batchFeatures, activeRows: activeRows, timeByRow: timeByRow)
            let activeState = gatherActiveState(fullState, activeRows: activeRows)
            let tokenIds = MLXArray(activeRows.map { Int32(lastTokenByRow[$0]) }).reshaped([activeRows.count, 1]).asType(.int32)

            let decoderOut = decoder.predictBatched(tokenIds, state: activeState, blankToken: Int32(blankToken))
            let pred = decoderOut.0.asType(activeFrames.dtype)
            let proposedState: ParakeetLSTMState = (
                hidden: decoderOut.1.hidden?.asType(activeFrames.dtype),
                cell: decoderOut.1.cell?.asType(activeFrames.dtype)
            )

            let jointOut = joint(activeFrames, pred)
            let tokenLogits = jointOut[0..., 0, 0, ..<(blankToken + 1)]
            let durationLogits = jointOut[0..., 0, 0, (blankToken + 1)...]
            let tokenArgMax = tokenLogits.argMax(axis: -1).asType(.int32)
            let durationArgMax = durationLogits.argMax(axis: -1).asType(.int32)
            let decisions = MLX.stacked([tokenArgMax, durationArgMax], axis: 0)
            eval(decisions)
            let decisionPairs = decisions.asArray(Int32.self)
            let activeCount = activeRows.count
            let predictedTokens = (0..<activeCount).map { Int(decisionPairs[$0]) }
            let decisionIndices = (0..<activeCount).map { Int(decisionPairs[activeCount + $0]) }

            var committedRows = Array(repeating: false, count: activeRows.count)

            for (activeIndex, row) in activeRows.enumerated() {
                let token = predictedTokens[activeIndex]
                let decisionIndex = decisionIndices[activeIndex]
                let currentTime = timeByRow[row]
                let currentNewSymbols = newSymbolsByRow[row]
                let step = ParakeetDecodingLogic.tdtStep(
                    predictedToken: token,
                    blankToken: blankToken,
                    decisionIndex: decisionIndex,
                    durations: durations,
                    time: currentTime,
                    newSymbols: currentNewSymbols,
                    maxSymbols: maxSymbols
                )

                tdtTraceEmitter?(
                    TDTTraceStep(
                        row: row,
                        time: currentTime,
                        newSymbols: currentNewSymbols,
                        token: token,
                        decisionIndex: decisionIndex,
                        committedState: token != blankToken
                    )
                )

                if token != blankToken {
                    committedRows[activeIndex] = true
                    lastTokenByRow[row] = token

                    if !ParakeetTokenizer.isSpecialToken(token, vocabulary: vocabulary) {
                        hypothesisByRow[row].append(
                            ParakeetAlignedToken(
                                id: token,
                                text: ParakeetTokenizer.decode(tokens: [token], vocabulary: vocabulary),
                                start: frameTimeSeconds(frameIndex: currentTime),
                                duration: frameTimeSeconds(frameIndex: step.jump)
                            )
                        )
                    }
                }

                timeByRow[row] = step.nextTime
                newSymbolsByRow[row] = step.nextNewSymbols
                doneByRow[row] = step.nextTime >= maxLengthByRow[row]
            }

            fullState = mergeUpdatedState(fullState, activeRows: activeRows, updatedState: proposedState, committedRows: committedRows)
        }

        return hypothesisByRow.map { hypothesis in
            ParakeetAlignment.sentencesToResult(
                ParakeetAlignment.tokensToSentences(hypothesis)
            )
        }
    }

    private func gatherActiveFrames(
        batchFeatures: MLXArray,
        activeRows: [Int],
        timeByRow: [Int]
    ) -> MLXArray {
        let gathered = activeRows.map { row in
            let time = timeByRow[row]
            return batchFeatures[row..<(row + 1), time..<(time + 1), 0...]
        }

        if gathered.count == 1 {
            return gathered[0]
        }
        return MLX.concatenated(gathered, axis: 0)
    }

    private func gatherActiveState(_ state: ParakeetLSTMState, activeRows: [Int]) -> ParakeetLSTMState {
        func gather(_ array: MLXArray?) -> MLXArray? {
            guard let array else { return nil }
            let slices = activeRows.map { row in
                array[0..., row..<(row + 1), 0...]
            }

            if slices.count == 1 {
                return slices[0]
            }
            return MLX.concatenated(slices, axis: 1)
        }

        return (hidden: gather(state.hidden), cell: gather(state.cell))
    }

    private func mergeUpdatedState(
        _ state: ParakeetLSTMState,
        activeRows: [Int],
        updatedState: ParakeetLSTMState,
        committedRows: [Bool]
    ) -> ParakeetLSTMState {
        func merge(_ original: MLXArray?, _ updated: MLXArray?) -> MLXArray? {
            guard let original, let updated else { return original }
            guard !activeRows.isEmpty else { return original }

            var rowSlices = (0..<original.shape[1]).map { row in
                original[0..., row..<(row + 1), 0...]
            }
            let updatedSlices = updated.split(parts: activeRows.count, axis: 1)

            for (index, row) in activeRows.enumerated() where committedRows[index] {
                rowSlices[row] = updatedSlices[index]
            }

            if rowSlices.count == 1 {
                return rowSlices[0]
            }
            return MLX.concatenated(rowSlices, axis: 1)
        }

        return (
            hidden: merge(state.hidden, updatedState.hidden),
            cell: merge(state.cell, updatedState.cell)
        )
    }

    private func decodeRNNT(mel: MLXArray, lengths: MLXArray? = nil) -> [ParakeetAlignedResult] {
        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        assert(
            features.ndim == 3 && features.shape[2] == preprocessConfig.features,
            "Parakeet RNNT input feature shape mismatch: expected [B, T, \(preprocessConfig.features)], got \(features.shape)"
        )

        let encoded = encodeBatchFeatures(features, lengths: lengths)
        return decodeRNNTEncoded(batchFeatures: encoded.0, lengths: encoded.1)
    }

    private func decodeRNNTEncoded(batchFeatures: MLXArray, lengths: MLXArray) -> [ParakeetAlignedResult] {
        guard let decoder, let joint else { return [] }

        assert(
            batchFeatures.ndim == 3 && batchFeatures.shape[2] == encoderConfig.dModel,
            "Parakeet RNNT encoder output shape mismatch: expected last dim \(encoderConfig.dModel), got \(batchFeatures.shape)"
        )
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
                eval(jointOut)
                let token = jointOut.argMax(axis: -1).item(Int.self)
                let step = ParakeetDecodingLogic.rnntStep(
                    predictedToken: token,
                    blankToken: blankToken,
                    time: t,
                    newSymbols: newSymbols,
                    maxSymbols: maxSymbols
                )

                if step.emittedToken {
                    lastToken = token
                    state = proposedState
                    if !ParakeetTokenizer.isSpecialToken(token, vocabulary: vocabulary) {
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
                    }
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

    private func decodeCTC(mel: MLXArray, lengths: MLXArray? = nil) -> [ParakeetAlignedResult] {
        var features = mel
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        assert(
            features.ndim == 3 && features.shape[2] == preprocessConfig.features,
            "Parakeet CTC input feature shape mismatch: expected [B, T, \(preprocessConfig.features)], got \(features.shape)"
        )

        let encoded = encodeBatchFeatures(features, lengths: lengths)
        return decodeCTCEncoded(batchFeatures: encoded.0, lengths: encoded.1)
    }

    private func decodeCTCEncoded(batchFeatures: MLXArray, lengths: MLXArray) -> [ParakeetAlignedResult] {
        guard let ctcDecoder else { return [] }

        assert(
            batchFeatures.ndim == 3 && batchFeatures.shape[2] == encoderConfig.dModel,
            "Parakeet CTC encoder output shape mismatch: expected last dim \(encoderConfig.dModel), got \(batchFeatures.shape)"
        )
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
            let hypothesis: [ParakeetAlignedToken] = spans.compactMap { span in
                if ParakeetTokenizer.isSpecialToken(span.token, vocabulary: vocabulary) { return nil }
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

    private func normalizeAudioToMono(_ audio: MLXArray) -> MLXArray {
        audio.ndim > 1 ? audio.mean(axis: -1) : audio
    }

    func makeBatchFeatures(_ audios: [MLXArray]) -> (features: MLXArray, lengths: MLXArray) {
        let melFeatures = audios.map { makeMelFeatures(from: normalizeAudioToMono($0)) }
        assert(
            melFeatures.allSatisfy { $0.ndim == 2 && $0.shape[1] == preprocessConfig.features },
            "Parakeet batch mel feature shape mismatch before stacking; expected trailing dim \(preprocessConfig.features)"
        )
        let frameLengths = melFeatures.map { Int32($0.shape[0]) }
        let maxFrameLength = melFeatures.map { $0.shape[0] }.max() ?? 0
        let padded = melFeatures.map { padMelFeatures($0, targetFrameLength: maxFrameLength) }

        let stacked = MLX.stacked(padded, axis: 0)
        assert(
            stacked.ndim == 3 && stacked.shape[2] == preprocessConfig.features,
            "Parakeet batch feature stack mismatch: expected [B, T, \(preprocessConfig.features)], got \(stacked.shape)"
        )

        return (
            features: stacked,
            lengths: MLXArray(frameLengths).asType(.int32)
        )
    }

    private func makeMelFeatures(from audio: MLXArray) -> MLXArray {
        ParakeetAudio.logMelSpectrogram(audio, config: preprocessConfig).squeezed(axis: 0)
    }

    private func padMelFeatures(_ mel: MLXArray, targetFrameLength: Int) -> MLXArray {
        let currentFrameLength = mel.shape[0]
        guard currentFrameLength < targetFrameLength else {
            return mel
        }

        let featureCount = mel.shape[1]
        let padding = MLXArray.zeros([targetFrameLength - currentFrameLength, featureCount], type: Float.self)
            .asType(mel.dtype)
        return MLX.concatenated([mel, padding], axis: 0)
    }

    private func makeInitialDecoderState(batchSize: Int, dtype: DType) -> ParakeetLSTMState {
        guard let decoder else {
            return (hidden: nil, cell: nil)
        }

        let decRnn = decoder.prediction.decRnn
        let hiddenSize = decRnn.layers.first?.hiddenSize ?? decoder.predHidden
        let shape = [decRnn.numLayers, batchSize, hiddenSize]
        let zeros = MLXArray.zeros(shape, type: Float.self).asType(dtype)
        return (hidden: zeros, cell: zeros)
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

private func makeCompiledTDTStep(
    decoder: ParakeetPredictNetwork?,
    joint: ParakeetJointNetwork?,
    blankTokenId: Int
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    guard let decoder, let joint else {
        return { arrays in
            [MLXArray([Int32(0), Int32(0)]), arrays[2], arrays[3]]
        }
    }

    let blankTokenArray = MLXArray(Int32(blankTokenId)).reshaped([1, 1])

    return compile { arrays in
        let feature = arrays[0]
        let currentToken = arrays[1]
        let hidden = arrays[2]
        let cell = arrays[3]

        let embedded = decoder.prediction.embed(currentToken)
        let blankMask = (currentToken .== blankTokenArray).expandedDimensions(axis: 2)
        let zeroEmbedded = MLXArray.zeros(like: embedded)
        let maskedEmbedded = MLX.where(blankMask, zeroEmbedded, embedded)

        let decoderOut = decoder.prediction.decRnn(maskedEmbedded, state: (hidden: hidden, cell: cell))
        let pred = decoderOut.0.asType(feature.dtype)
        let hiddenOut = decoderOut.1.hidden!.asType(feature.dtype)
        let cellOut = decoderOut.1.cell!.asType(feature.dtype)

        let jointOut = joint(feature, pred)
        let tokenLogits = jointOut[0, 0, 0, ..<(blankTokenId + 1)]
        let durationLogits = jointOut[0, 0, 0, (blankTokenId + 1)...]
        let predToken = tokenLogits.argMax(axis: -1).asType(.int32)
        let decision = durationLogits.argMax(axis: -1).asType(.int32)
        let decisions = MLX.stacked([predToken, decision], axis: 0)
        return [decisions, hiddenOut, cellOut]
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

    static func fromDirectory(
        _ modelDir: URL,
        computeDType: DType = .bfloat16
    ) throws -> ParakeetModel {
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

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        model.computeDType = computeDType

        // Cast all floating-point params to computeDType after load.
        // Skips params already matching target dtype and leaves non-float (e.g. uint32
        // packed quantized) weights untouched.
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
        _ modelPath: String,
        computeDType: DType = .bfloat16,
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
        return try fromDirectory(modelDir, computeDType: computeDType)
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
        // BaseConfiguration requires model_type, but Parakeet configs use 'target'
        // instead, so BaseConfiguration decoding fails. Try it first for future
        // compatibility, then fall back to reading 'quantization' directly.
        if let base = try? BaseConfiguration(from: decoder) {
            self.perLayerQuantization = base.perLayerQuantization
            return
        }

        // Parakeet config has: "quantization": { "group_size": N, "bits": N, "mode": "..." }
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

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}

