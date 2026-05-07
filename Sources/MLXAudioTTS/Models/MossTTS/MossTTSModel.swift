import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public let mossTTSDefaultAudioTokenizerRepo = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

final class MosiTTSModel: Module {
    let config: MossTTSConfig

    @ModuleInfo(key: "embedding_list") var embeddingList: [Embedding]
    @ModuleInfo(key: "language_model") var languageModel: MossQwen3Model

    init(config: MossTTSConfig) {
        self.config = config
        var embeddings = [Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)]
        embeddings.append(
            contentsOf: (0 ..< config.nVQ).map { _ in
                Embedding(embeddingCount: config.audioVocabSize + 1, dimensions: config.hiddenSize)
            }
        )
        _embeddingList.wrappedValue = embeddings
        _languageModel.wrappedValue = MossQwen3Model(config.languageConfig)
    }

    func prepareMultiModalInputs(
        _ inputIDs: MLXArray,
        nVQForInference: Int? = nil
    ) throws -> MLXArray {
        guard inputIDs.ndim == 3, inputIDs.dim(2) == config.nVQ + 1 else {
            throw AudioGenerationError.invalidInput(
                "Expected input_ids shape [batch, seq, \(config.nVQ + 1)], got \(inputIDs.shape)"
            )
        }
        let channels = min(inputIDs.dim(2), 1 + (nVQForInference ?? config.nVQ))
        var inputsEmbeds = MLXArray.zeros(
            [inputIDs.dim(0), inputIDs.dim(1), config.hiddenSize],
            dtype: embeddingList[0].weight.dtype
        )
        for channel in 0 ..< channels {
            inputsEmbeds = inputsEmbeds + embeddingList[channel](inputIDs[0..., 0..., channel])
        }
        return inputsEmbeds
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil,
        nVQForInference: Int? = nil
    ) throws -> MLXArray {
        let embeddings: MLXArray
        if let inputEmbeddings {
            embeddings = inputEmbeddings
        } else if let inputIDs {
            embeddings = try prepareMultiModalInputs(inputIDs, nVQForInference: nVQForInference)
        } else {
            throw AudioGenerationError.invalidInput("inputIDs or inputEmbeddings are required")
        }
        return try languageModel(inputEmbeddings: embeddings, cache: cache)
    }
}

public final class MossTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: MossTTSConfig
    public private(set) var generationConfig: MossTTSGenerationConfig = .init()

    @ModuleInfo(key: "language_model") var languageModel: MossQwen3Model?
    @ModuleInfo(key: "emb_ext") var embExt: [Embedding]
    @ModuleInfo(key: "model") var localModel: MosiTTSModel?
    @ModuleInfo(key: "local_transformer") var localTransformer: MossTTSLocalTransformer?
    @ModuleInfo(key: "speech_embedding_to_local_mlp") var speechEmbeddingToLocalMLP: MossTTSMLP?
    @ModuleInfo(key: "local_to_speech_embedding_mlps") var localToSpeechEmbeddingMLPs: [MossTTSMLP]
    @ModuleInfo(key: "layer_norm_before_lm_heads") var layerNormBeforeLMHeads: [RMSNorm]
    @ModuleInfo(key: "lm_heads") var lmHeads: [Linear]

    public var tokenizer: MossTTSTextTokenizing?
    public var audioTokenizer: MossAudioTokenizing?

    public var sampleRate: Int { config.samplingRate }

    public var defaultGenerationParameters: GenerateParameters {
        if config.isLocalTransformer {
            return GenerateParameters(
                maxTokens: 4_096,
                temperature: 1.0,
                topP: 0.95,
                topK: 50,
                repetitionPenalty: 1.1
            )
        }
        return GenerateParameters(
            maxTokens: generationConfig.maxNewTokens ?? 4_096,
            temperature: generationConfig.temperature ?? 1.7,
            topP: generationConfig.topP ?? 0.8,
            topK: generationConfig.topK ?? 25,
            repetitionPenalty: generationConfig.repetitionPenalty ?? 1.0
        )
    }

    public init(config: MossTTSConfig) throws {
        self.config = config
        let channels = config.nVQ + 1
        if config.isLocalTransformer {
            _languageModel.wrappedValue = nil
            _embExt.wrappedValue = []
            _localModel.wrappedValue = MosiTTSModel(config: config)
            let localConfig = try config.localTransformerConfig()
            _localTransformer.wrappedValue = MossTTSLocalTransformer(localConfig)
            _speechEmbeddingToLocalMLP.wrappedValue = MossTTSMLP(
                inputSize: config.hiddenSize,
                ffnHiddenSize: config.additionalMLPFFNHiddenSize!,
                outputSize: config.localHiddenSize!
            )
            _localToSpeechEmbeddingMLPs.wrappedValue = (0 ..< channels).map { _ in
                MossTTSMLP(
                    inputSize: config.localHiddenSize!,
                    ffnHiddenSize: config.additionalMLPFFNHiddenSize!,
                    outputSize: config.hiddenSize
                )
            }
            _layerNormBeforeLMHeads.wrappedValue = (0 ..< channels).map { _ in
                RMSNorm(dimensions: config.hiddenSize)
            }
        } else {
            _languageModel.wrappedValue = MossQwen3Model(config.languageConfig)
            _embExt.wrappedValue = (0 ..< config.nVQ).map { _ in
                Embedding(embeddingCount: config.audioVocabSize + 1, dimensions: config.hiddenSize)
            }
            _localModel.wrappedValue = nil
            _localTransformer.wrappedValue = nil
            _speechEmbeddingToLocalMLP.wrappedValue = nil
            _localToSpeechEmbeddingMLPs.wrappedValue = []
            _layerNormBeforeLMHeads.wrappedValue = []
        }

        var heads = [Linear(config.hiddenSize, config.vocabSize, bias: false)]
        heads.append(
            contentsOf: (0 ..< config.nVQ).map { _ in
                Linear(config.hiddenSize, config.audioVocabSize + 1, bias: false)
            }
        )
        _lmHeads.wrappedValue = heads
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard !config.isLocalTransformer else { return weights }
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.hasPrefix("model.") {
                sanitized[String(key.dropFirst("model.".count))] = value
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }

    public func buildInputsEmbeds(_ inputIDs: MLXArray) throws -> MLXArray {
        guard inputIDs.ndim == 3, inputIDs.dim(2) == config.nVQ + 1 else {
            throw AudioGenerationError.invalidInput(
                "Expected input_ids shape [batch, seq, \(config.nVQ + 1)], got \(inputIDs.shape)"
            )
        }
        if let localModel {
            return try localModel.prepareMultiModalInputs(inputIDs)
        }
        guard let languageModel else {
            throw AudioGenerationError.modelNotInitialized("MOSS language model is not initialized")
        }
        var inputsEmbeds = languageModel.embedTokens(inputIDs[0..., 0..., 0])
        for (index, embedding) in embExt.enumerated() {
            inputsEmbeds = inputsEmbeds + embedding(inputIDs[0..., 0..., index + 1])
        }
        return inputsEmbeds
    }

    func headLogits(_ hiddenStates: MLXArray, headIndex: Int) -> MLXArray {
        let logits = lmHeads[headIndex](hiddenStates)
        guard headIndex != 0 else { return logits }
        let last = logits.dim(logits.ndim - 1)
        let invalidPad = MLXArray.full(
            Array(logits.shape.dropLast()) + [1],
            values: MLXArray(-Float.infinity),
            dtype: logits.dtype
        )
        if logits.ndim == 3 {
            return MLX.concatenated([logits[0..., 0..., 0..<(last - 1)], invalidPad], axis: -1)
        }
        return MLX.concatenated([logits[0..., 0..<(last - 1)], invalidPad], axis: -1)
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil,
        headIndices: [Int]? = nil,
        labels: MLXArray? = nil,
        nVQForInference: Int? = nil
    ) throws -> [MLXArray] {
        if config.isLocalTransformer {
            return try localForward(
                inputIDs: inputIDs,
                inputEmbeddings: inputEmbeddings,
                cache: cache,
                labels: labels,
                headIndices: headIndices,
                nVQForInference: nVQForInference
            )
        }

        let embeddings: MLXArray
        if let inputEmbeddings {
            embeddings = inputEmbeddings
        } else if let inputIDs {
            embeddings = try buildInputsEmbeds(inputIDs)
        } else {
            throw AudioGenerationError.invalidInput("inputIDs or inputEmbeddings are required")
        }
        guard let languageModel else {
            throw AudioGenerationError.modelNotInitialized("MOSS language model is not initialized")
        }
        let hiddenStates = try languageModel(inputEmbeddings: embeddings, cache: cache)
        return (headIndices ?? Array(0 ..< (config.nVQ + 1))).map {
            headLogits(hiddenStates, headIndex: $0)
        }
    }

    func makeCache() -> [KVCache] {
        if let localModel {
            return localModel.languageModel.makeCache()
        }
        return languageModel?.makeCache() ?? []
    }

    private func maskedEmbedding(_ embedding: Embedding, inputIDs: MLXArray) -> MLXArray {
        let mask = inputIDs .!= MLXArray(Int32(-100))
        let safeIDs = MLX.where(mask, inputIDs, MLXArray(Int32(0))).asType(.int32)
        return MLX.where(
            mask.expandedDimensions(axis: -1),
            embedding(safeIDs),
            MLXArray(0.0)
        )
    }

    private func localForward(
        inputIDs: MLXArray?,
        inputEmbeddings: MLXArray?,
        cache: [KVCache]?,
        labels: MLXArray?,
        headIndices: [Int]?,
        nVQForInference: Int?
    ) throws -> [MLXArray] {
        guard let localModel,
              let localTransformer,
              let speechEmbeddingToLocalMLP
        else {
            throw AudioGenerationError.modelNotInitialized("MOSS local-transformer modules are not initialized")
        }
        let globalHiddenStates = try localModel(
            inputIDs: inputIDs,
            inputEmbeddings: inputEmbeddings,
            cache: cache,
            nVQForInference: nVQForInference
        )
        let effectiveLabels: MLXArray
        if let labels {
            effectiveLabels = labels
        } else if let inputIDs {
            effectiveLabels = inputIDs
        } else {
            throw AudioGenerationError.invalidInput("labels are required when inputEmbeddings are provided")
        }
        let channels = config.nVQ + 1
        guard effectiveLabels.ndim == 3, effectiveLabels.dim(2) == channels else {
            throw AudioGenerationError.invalidInput(
                "Expected labels shape [batch, seq, \(channels)], got \(effectiveLabels.shape)"
            )
        }

        var localInputs = [globalHiddenStates]
        for channel in 0 ..< (channels - 1) {
            localInputs.append(maskedEmbedding(localModel.embeddingList[channel], inputIDs: effectiveLabels[0..., 0..., channel]))
        }
        var stackedInputs = MLX.stacked(localInputs, axis: 0)
        stackedInputs = speechEmbeddingToLocalMLP(stackedInputs)
        let channelCount = stackedInputs.dim(0)
        let batch = stackedInputs.dim(1)
        let seqLen = stackedInputs.dim(2)
        let localDim = stackedInputs.dim(3)
        stackedInputs = stackedInputs.transposed(1, 2, 0, 3).reshaped(batch * seqLen, channelCount, localDim)
        let localOutputs = localTransformer(stackedInputs)

        return (headIndices ?? Array(0 ..< channels)).map { headIndex in
            var headHidden = localOutputs[0..., headIndex, 0...]
            headHidden = localToSpeechEmbeddingMLPs[headIndex](headHidden)
            headHidden = layerNormBeforeLMHeads[headIndex](headHidden)
            headHidden = headHidden.reshaped(batch, seqLen, config.hiddenSize)
            return lmHeads[headIndex](headHidden)
        }
    }

    public func encodeReferenceAudio(
        _ refAudio: MLXArray,
        numQuantizers: Int? = nil
    ) throws -> MLXArray {
        guard let audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("MOSS audio tokenizer is not initialized")
        }
        return try audioTokenizer.encodeAudio(refAudio, numQuantizers: numQuantizers ?? config.nVQ)
    }

    public func decodeAudioTokenIDs(
        _ audioTokenIDs: MLXArray,
        numQuantizers: Int? = nil
    ) throws -> MLXArray {
        guard let audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("MOSS audio tokenizer is not initialized")
        }
        return try audioTokenizer.decodeAudioCodes(audioTokenIDs, numQuantizers: numQuantizers ?? config.nVQ)
    }

    private func ensureAudioTokenizer() async throws {
        if audioTokenizer != nil { return }
        if let modelPath = config.modelPath {
            let local = URL(fileURLWithPath: modelPath).appendingPathComponent("audio_tokenizer", isDirectory: true)
            if FileManager.default.fileExists(atPath: local.appendingPathComponent("config.json").path) {
                audioTokenizer = try MLXMossAudioTokenizer.fromModelDirectory(local)
                return
            }
        }
        let source = config.audioTokenizerPretrainedNameOrPath?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let source, !source.isEmpty {
            let localSource = URL(fileURLWithPath: (source as NSString).expandingTildeInPath)
            if FileManager.default.fileExists(atPath: localSource.appendingPathComponent("config.json").path) {
                audioTokenizer = try MLXMossAudioTokenizer.fromModelDirectory(localSource)
                return
            }
        }
        if let cached = Self.findCachedHubSnapshot(
            repo: source?.isEmpty == false ? source! : mossTTSDefaultAudioTokenizerRepo
        ) {
            audioTokenizer = try MLXMossAudioTokenizer.fromModelDirectory(cached)
            return
        }
        audioTokenizer = try await MLXMossAudioTokenizer.fromPretrained(
            (source?.isEmpty == false ? source! : mossTTSDefaultAudioTokenizerRepo)
        )
    }

    private static func findCachedHubSnapshot(repo: String) -> URL? {
        let components = repo.split(separator: "/", maxSplits: 1).map(String.init)
        guard components.count == 2 else { return nil }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let snapshotsDir = home
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
            .appendingPathComponent("models--\(components[0])--\(components[1])", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: nil
        ) else {
            return nil
        }
        return snapshots.sorted { $0.lastPathComponent > $1.lastPathComponent }.first {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path)
        }
    }

    private static func findLastEqual(_ values: MLXArray, target: Int) -> Int {
        let items = values.asType(.int32).asArray(Int32.self)
        for index in stride(from: items.count - 1, through: 0, by: -1) {
            if Int(items[index]) == target { return index }
        }
        return -1
    }

    private static func setLogitsToNegInf(_ logits: MLXArray, tokenIDs: [Int]) -> MLXArray {
        guard !tokenIDs.isEmpty else { return logits }
        let ids = MLXArray(tokenIDs.map(Int32.init)).reshaped([1, -1])
        let values = MLXArray.full(ids.shape, values: MLXArray(-Float.infinity), dtype: logits.dtype)
        return putAlong(logits, ids, values: values, axis: -1)
    }

    private static func keepOnlyLogits(_ logits: MLXArray, tokenIDs: [Int]) -> MLXArray {
        guard !tokenIDs.isEmpty else { return logits }
        let ids = MLXArray(tokenIDs.map(Int32.init)).reshaped([1, -1])
        let selected = takeAlong(logits, ids, axis: -1)
        let base = MLXArray.full(logits.shape, values: MLXArray(-Float.infinity), dtype: logits.dtype)
        return putAlong(base, ids, values: selected, axis: -1)
    }

    public func generateDelayPatternIDs(
        inputIDs: MLXArray,
        maxNewTokens: Int = 4_096,
        textTemperature: Float = 1.5,
        textTopP: Float = 1.0,
        textTopK: Int = 50,
        audioTemperature: Float = 1.7,
        audioTopP: Float = 0.8,
        audioTopK: Int = 25,
        audioRepetitionPenalty: Float = 1.0
    ) throws -> [(startLength: Int, generationIDs: MLXArray)] {
        guard inputIDs.ndim == 3 else {
            throw AudioGenerationError.invalidInput("Expected input_ids rank 3, got \(inputIDs.shape)")
        }
        guard inputIDs.dim(0) == 1 else {
            throw AudioGenerationError.generationFailed("MOSS-TTS batch generation is not implemented.")
        }

        var textTemperature = textTemperature
        let textDoSample = textTemperature > 0
        if !textDoSample { textTemperature = 1 }
        var audioTemperature = audioTemperature
        let audioDoSample = audioTemperature > 0
        if !audioDoSample { audioTemperature = 1 }

        let batchSize = inputIDs.dim(0)
        let seqLen = inputIDs.dim(1)
        let width = inputIDs.dim(2)
        let nVQ = width - 1
        guard nVQ == config.nVQ else {
            throw AudioGenerationError.invalidInput("Expected \(config.nVQ) VQ channels, got \(nVQ)")
        }

        let cache = makeCache()
        var currentInputIDs = inputIDs
        var generationIDs = inputIDs
        var isStopping = false
        var audioLengths = 0
        var delayedLengths = Int.max

        let lastTextToken = inputIDs[0, -1, 0].item(Int.self)
        let isContinuation = lastTextToken == config.audioStartTokenID
            || lastTextToken == config.audioAssistantGenSlotTokenID
        let audioStartIndex = Self.findLastEqual(inputIDs[0, 0..., 0], target: config.audioStartTokenID)
        var isAudio = isContinuation && audioStartIndex != -1
        if isAudio {
            audioLengths = seqLen - audioStartIndex
        }

        let textExcludeOutsideAudio = [
            config.padTokenID,
            config.audioAssistantGenSlotTokenID,
            config.audioAssistantDelaySlotTokenID,
            config.audioEndTokenID,
        ]
        let textKeepInsideAudio = [
            config.audioAssistantGenSlotTokenID,
            config.audioAssistantDelaySlotTokenID,
        ]

        for timeStep in 0 ..< maxNewTokens {
            let outputs = try self(inputIDs: currentInputIDs, cache: cache)
            let nextTokenLogits = outputs.enumerated().map { index, logits in
                logits[0..., -1, 0...] / MLXArray(index == 0 ? textTemperature : audioTemperature)
            }

            var nextTextTokenValue = config.padTokenID
            if !isStopping && delayedLengths < nVQ {
                nextTextTokenValue = config.audioAssistantDelaySlotTokenID
            } else if !isStopping && delayedLengths == nVQ {
                nextTextTokenValue = config.audioEndTokenID
                isAudio = false
            } else if !isStopping && delayedLengths > nVQ {
                var textLogits = nextTokenLogits[0]
                if isAudio {
                    textLogits = Self.keepOnlyLogits(textLogits, tokenIDs: textKeepInsideAudio)
                } else {
                    textLogits = Self.setLogitsToNegInf(textLogits, tokenIDs: textExcludeOutsideAudio)
                }
                if timeStep == 0 {
                    textLogits = Self.setLogitsToNegInf(textLogits, tokenIDs: [config.audioAssistantDelaySlotTokenID])
                }
                if timeStep <= nVQ {
                    textLogits = Self.setLogitsToNegInf(textLogits, tokenIDs: [config.imEndTokenID])
                }
                let token = mossTTSSampleToken(
                    logits: textLogits,
                    topP: textTopP,
                    topK: textTopK,
                    doSample: textDoSample
                )
                eval(token)
                nextTextTokenValue = token.item(Int.self)
            }

            if nextTextTokenValue == config.audioStartTokenID {
                isAudio = true
            }
            if nextTextTokenValue == config.imEndTokenID {
                isStopping = true
            }

            var nextAudioValues = Array(repeating: Int32(config.audioPadCode), count: batchSize * nVQ)
            for codebookIndex in 0 ..< nVQ {
                let preAudio = audioLengths > codebookIndex
                let postAudio = delayedLengths == Int.max ? true : codebookIndex > delayedLengths - 1
                guard preAudio && postAudio else { continue }

                var channelLogits = nextTokenLogits[codebookIndex + 1]
                channelLogits = Self.setLogitsToNegInf(channelLogits, tokenIDs: [config.audioPadCode])
                let channelToken = mossTTSSampleToken(
                    logits: channelLogits,
                    previousTokens: generationIDs[0..., 0..., codebookIndex + 1],
                    repetitionPenalty: audioRepetitionPenalty,
                    topP: audioTopP,
                    topK: audioTopK,
                    doSample: audioDoSample
                )
                eval(channelToken)
                nextAudioValues[codebookIndex] = Int32(channelToken.item(Int.self))
            }

            if [
                config.audioStartTokenID,
                config.audioAssistantGenSlotTokenID,
                config.audioAssistantDelaySlotTokenID,
            ].contains(nextTextTokenValue) {
                audioLengths += 1
            }
            if nextTextTokenValue == config.audioEndTokenID {
                audioLengths = 0
            }
            if delayedLengths == Int.max && nextTextTokenValue == config.audioAssistantDelaySlotTokenID {
                delayedLengths = 0
            }
            if delayedLengths != Int.max {
                delayedLengths += 1
            }
            if delayedLengths > nVQ {
                delayedLengths = Int.max
            }

            let nextTextToken = MLXArray([Int32(nextTextTokenValue)], [batchSize, 1, 1]).asType(.int32)
            let nextAudioTokens = MLXArray(nextAudioValues, [batchSize, 1, nVQ]).asType(.int32)
            currentInputIDs = MLX.concatenated([nextTextToken, nextAudioTokens], axis: 2)
            generationIDs = MLX.concatenated([generationIDs, currentInputIDs], axis: 1)
            eval(currentInputIDs)

            if isStopping { break }
        }

        var startIndex = Self.findLastEqual(inputIDs[0, 0..., 0], target: config.imStartTokenID)
        startIndex = startIndex != -1 ? startIndex + 3 : seqLen
        let startLength = seqLen - startIndex
        return [(startLength, generationIDs[0, startIndex..., 0...])]
    }

    public func generateLocalIDs(
        inputIDs: MLXArray,
        maxNewTokens: Int = 4_096,
        textTemperature: Float = 1.5,
        textTopP: Float = 1.0,
        textTopK: Int = 50,
        textRepetitionPenalty: Float = 1.0,
        audioTemperature: Float = 1.0,
        audioTopP: Float = 0.95,
        audioTopK: Int = 50,
        audioRepetitionPenalty: Float = 1.1,
        nVQForInference: Int? = nil
    ) throws -> [(startLength: Int, generationIDs: MLXArray)] {
        guard inputIDs.ndim == 3 else {
            throw AudioGenerationError.invalidInput("Expected input_ids rank 3, got \(inputIDs.shape)")
        }
        guard inputIDs.dim(0) == 1 else {
            throw AudioGenerationError.generationFailed("MOSS-TTS batch generation is not implemented.")
        }
        guard let localModel,
              let localTransformer,
              let speechEmbeddingToLocalMLP
        else {
            throw AudioGenerationError.modelNotInitialized("MOSS local-transformer modules are not initialized")
        }

        var textTemperature = textTemperature
        let textDoSample = textTemperature > 0
        if !textDoSample { textTemperature = 1 }
        var audioTemperature = audioTemperature
        let audioDoSample = audioTemperature > 0
        if !audioDoSample { audioTemperature = 1 }

        let batchSize = inputIDs.dim(0)
        let seqLen = inputIDs.dim(1)
        let channels = inputIDs.dim(2)
        guard channels == config.nVQ + 1 else {
            throw AudioGenerationError.invalidInput("Expected \(config.nVQ + 1) channels, got \(channels)")
        }
        let nVQ = max(1, min(channels - 1, nVQForInference ?? (channels - 1)))
        let activeChannels = 1 + nVQ

        let cache = makeCache()
        var currentInputIDs = inputIDs
        var generationIDs = inputIDs

        for _ in 0 ..< maxNewTokens {
            let globalHiddenStates = try localModel(
                inputIDs: currentInputIDs,
                cache: cache,
                nVQForInference: nVQ
            )
            var currentLocalInput = speechEmbeddingToLocalMLP(globalHiddenStates[0..., -1, 0...])
            var localInputs = MLXArray.zeros(
                [batchSize, 0, config.localHiddenSize!],
                dtype: currentLocalInput.dtype
            )
            var nextValues: [Int32] = []
            nextValues.reserveCapacity(channels)

            for channelIndex in 0 ..< activeChannels {
                localInputs = MLX.concatenated([localInputs, currentLocalInput.expandedDimensions(axis: 1)], axis: 1)
                let localOutputs = localTransformer(localInputs)
                var headHidden = localOutputs[0..., -1, 0...]
                headHidden = localToSpeechEmbeddingMLPs[channelIndex](headHidden)
                headHidden = layerNormBeforeLMHeads[channelIndex](headHidden)
                var logits = lmHeads[channelIndex](headHidden)
                if channelIndex != 0 {
                    logits = Self.setLogitsToNegInf(logits, tokenIDs: [config.audioPadCode])
                }

                let isText = channelIndex == 0
                let doSample = isText ? textDoSample : audioDoSample
                let repetitionPenalty = doSample
                    ? (isText ? textRepetitionPenalty : audioRepetitionPenalty)
                    : 1.0
                let token = mossTTSSampleToken(
                    logits: logits / MLXArray(isText ? textTemperature : audioTemperature),
                    previousTokens: generationIDs[0..., 0..., channelIndex],
                    repetitionPenalty: repetitionPenalty,
                    topP: isText ? textTopP : audioTopP,
                    topK: isText ? textTopK : audioTopK,
                    doSample: doSample
                )
                eval(token)
                nextValues.append(Int32(token.item(Int.self)))

                currentLocalInput = localModel.embeddingList[channelIndex](token)
                currentLocalInput = speechEmbeddingToLocalMLP(currentLocalInput)
            }

            while nextValues.count < channels {
                nextValues.append(0)
            }
            let nextTokens = MLXArray(nextValues, [batchSize, channels]).asType(.int32)
            currentInputIDs = nextTokens.expandedDimensions(axis: 1)
            generationIDs = MLX.concatenated([generationIDs, currentInputIDs], axis: 1)
            eval(currentInputIDs)

            if Int(nextValues[0]) == config.audioEndTokenID {
                break
            }
        }

        let audioStartIndex = Self.findLastEqual(inputIDs[0, 0..., 0], target: config.audioStartTokenID)
        let startIndex = audioStartIndex != -1 ? audioStartIndex : seqLen
        let startLength = audioStartIndex != -1 ? seqLen - startIndex - 1 : 0
        return [(startLength, generationIDs[0, startIndex..., 0...])]
    }

    private func decodeGeneratedAudio(
        _ outputs: [(startLength: Int, generationIDs: MLXArray)]
    ) throws -> (audio: MLXArray, tokenCount: Int) {
        var segments: [MLXArray] = []
        var tokenCount = 0
        for output in outputs {
            var audioCodes = output.generationIDs[0..., 1...].asType(.int32)
            if !config.isLocalTransformer {
                audioCodes = try mossTTSApplyDeDelayPattern(audioCodes)
            }
            let rows = audioCodes.dim(0)
            let cols = audioCodes.dim(1)
            let values = audioCodes.asArray(Int32.self)
            var nonPad: [Int] = []
            for row in 0 ..< rows {
                var allPad = true
                for col in 0 ..< cols where values[row * cols + col] != Int32(config.audioPadCode) {
                    allPad = false
                    break
                }
                if !allPad { nonPad.append(row) }
            }
            guard !nonPad.isEmpty else { continue }

            var breakStart = 0
            while breakStart < nonPad.count {
                var breakEnd = breakStart + 1
                while breakEnd < nonPad.count && nonPad[breakEnd] == nonPad[breakEnd - 1] + 1 {
                    breakEnd += 1
                }
                let segmentRows = nonPad[breakStart ..< breakEnd]
                let codes = audioCodes[segmentRows.first! ... segmentRows.last!, 0...]
                tokenCount += codes.dim(0)
                var audio = try decodeAudioTokenIDs(codes, numQuantizers: config.nVQ)
                if output.startLength > 0 && segments.isEmpty {
                    let firstCodesLength = codes.dim(0)
                    if firstCodesLength > 0 {
                        let trimRatio = max(0, min(Float(output.startLength) / Float(firstCodesLength), 1))
                        let trimSamples = Int(Float(audio.dim(0)) * trimRatio)
                        if trimSamples < audio.dim(0) {
                            audio = audio[trimSamples..., 0...]
                        } else {
                            audio = MLXArray.zeros([0, audio.dim(1)], dtype: .float32)
                        }
                    }
                }
                segments.append(audio)
                breakStart = breakEnd
            }
        }

        guard !segments.isEmpty else {
            return (MLXArray.zeros([0, 1], dtype: .float32), 0)
        }
        return (MLX.concatenated(segments, axis: 0), tokenCount)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = voice
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("MOSS tokenizer is not initialized")
        }
        try await ensureAudioTokenizer()

        let promptAudioCodes = try refAudio.map {
            try encodeReferenceAudio($0, numQuantizers: config.nVQ)
        }
        let mode = (refText != nil && promptAudioCodes != nil) ? "continuation" : "generation"
        let processor: MossTTSDelayProcessor = config.isLocalTransformer
            ? try MossTTSLocalProcessor(tokenizer: tokenizer, config: config)
            : try MossTTSDelayProcessor(tokenizer: tokenizer, config: config)

        let userMessage = processor.buildUserMessage(
            text: mode == "generation" ? text : (refText ?? "") + text,
            reference: mode == "generation" ? promptAudioCodes.map { [$0] } : nil,
            language: language
        )
        let conversations: [[MossTTSConversationMessage]]
        if mode == "generation" {
            conversations = [[userMessage]]
        } else {
            guard let promptAudioCodes else {
                throw AudioGenerationError.invalidInput("continuation mode requires refAudio")
            }
            conversations = [[
                userMessage,
                processor.buildAssistantMessage(audioCodesList: [promptAudioCodes]),
            ]]
        }

        let batch = try processor(conversations, mode: mode)
        let outputs: [(startLength: Int, generationIDs: MLXArray)]
        if config.isLocalTransformer {
            outputs = try generateLocalIDs(
                inputIDs: batch.inputIDs,
                maxNewTokens: generationParameters.maxTokens ?? 4_096,
                audioTemperature: generationParameters.temperature,
                audioTopP: generationParameters.topP,
                audioTopK: generationParameters.topK,
                audioRepetitionPenalty: generationParameters.repetitionPenalty ?? 1.1
            )
        } else {
            outputs = try generateDelayPatternIDs(
                inputIDs: batch.inputIDs,
                maxNewTokens: generationParameters.maxTokens ?? generationConfig.maxNewTokens ?? 4_096,
                textTemperature: generationConfig.temperature ?? 1.5,
                textTopP: generationConfig.topP ?? 1.0,
                textTopK: 50,
                audioTemperature: generationParameters.temperature,
                audioTopP: generationParameters.topP,
                audioTopK: generationParameters.topK,
                audioRepetitionPenalty: generationParameters.repetitionPenalty ?? generationConfig.repetitionPenalty ?? 1.0
            )
        }
        return try decodeGeneratedAudio(outputs).audio
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
            let task = Task { @Sendable in
                do {
                    let audio = try await self.generate(
                        text: text,
                        voice: voice,
                        refAudio: refAudio,
                        refText: refText,
                        language: language,
                        generationParameters: generationParameters
                    )
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public static func fromPretrained(
        _ modelRepo: String,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> MossTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.json", "*.txt", "*.jinja", "*.py", "*.md", "*.index.json"],
            hfToken: hfToken,
            cache: cache
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> MossTTSModel {
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(MossTTSConfig.self, from: configData)
        config.modelPath = modelDir.path
        let model = try MossTTSModel(config: config)
        model.generationConfig = MossTTSGenerationConfig.fromFileIfPresent(
            modelDir.appendingPathComponent("generation_config.json")
        )

        let weights = try loadWeights(from: modelDir)
        try model.update(parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
        eval(model)

        model.tokenizer = try await MossTTSTokenizerAdapter.fromModelDirectory(modelDir)

        let audioTokenizerDir = modelDir.appendingPathComponent("audio_tokenizer", isDirectory: true)
        if FileManager.default.fileExists(atPath: audioTokenizerDir.appendingPathComponent("config.json").path) {
            model.audioTokenizer = try MLXMossAudioTokenizer.fromModelDirectory(audioTokenizerDir)
        }
        return model
    }
}
