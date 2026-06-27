import Foundation
import HuggingFace
import MLX
import MLXLMCommon
import MLXNN
import MLXAudioCore
import Tokenizers

private struct MiMoUncheckedSendableBox<T>: @unchecked Sendable {
    let value: T
}

private struct MiMoTranscriptionResult {
    let text: String
    let promptTokens: Int
    let generationTokens: Int
    let promptTime: TimeInterval
    let generationTime: TimeInterval
    let totalTime: TimeInterval
    let peakMemoryUsage: Double
}

public struct MiMoV2ASRAssets: Sendable {
    public let modelDirectory: URL
    public let tokenizerDirectory: URL
    public let config: MiMoV2ASRConfig
    public let tokenizerConfig: MiMoAudioTokenizerConfig
    public let manifest: MiMoMLXManifest?
}

public final class MiMoV2ASRModel: STTGenerationModel {
    public let assets: MiMoV2ASRAssets
    let coreModel: MiMoV2ASRCore
    let audioTokenizerModel: MiMoAudioTokenizerModel
    public var tokenizer: Tokenizers.Tokenizer?
    var specialTokens: MiMoSpecialTokens?

    init(
        assets: MiMoV2ASRAssets,
        coreModel: MiMoV2ASRCore,
        audioTokenizerModel: MiMoAudioTokenizerModel
    ) {
        self.assets = assets
        self.coreModel = coreModel
        self.audioTokenizerModel = audioTokenizerModel
    }

    public var config: MiMoV2ASRConfig { assets.config }
    public var tokenizerConfig: MiMoAudioTokenizerConfig { assets.tokenizerConfig }

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 4096,
            temperature: 0.0,
            topP: 0.95,
            topK: 0,
            verbose: false,
            language: nil,
            chunkDuration: 30.0,
            minChunkDuration: 1.0
        )
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        do {
            let result = try transcribe(audio: audio, generationParameters: generationParameters)
            return STTOutput(
                text: result.text,
                language: generationParameters.language,
                promptTokens: result.promptTokens,
                generationTokens: result.generationTokens,
                totalTokens: result.promptTokens + result.generationTokens,
                promptTps: Double(result.promptTokens) / max(result.promptTime, 0.001),
                generationTps: Double(result.generationTokens) / max(result.generationTime, 0.001),
                totalTime: result.totalTime,
                peakMemoryUsage: result.peakMemoryUsage
            )
        } catch {
            if generationParameters.verbose {
                fputs("MiMo generation failed: \(error)\n", stderr)
            }
            return STTOutput(
                text: "",
                language: generationParameters.language,
                totalTime: 0,
                peakMemoryUsage: Double(Memory.peakMemory) / 1e9
            )
        }
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let sendableModel = MiMoUncheckedSendableBox(value: self)
        let sendableAudio = MiMoUncheckedSendableBox(value: audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    let result = try sendableModel.value.transcribe(
                        audio: sendableAudio.value,
                        generationParameters: generationParameters
                    ) { tokenText in
                        guard !tokenText.isEmpty else { return }
                        continuation.yield(STTGeneration.token(tokenText))
                    }

                    let info = STTGenerationInfo(
                        promptTokenCount: result.promptTokens,
                        generationTokenCount: result.generationTokens,
                        prefillTime: result.promptTime,
                        generateTime: result.generationTime,
                        tokensPerSecond: Double(result.generationTokens) / max(result.generationTime, 0.001),
                        peakMemoryUsage: result.peakMemoryUsage
                    )
                    continuation.yield(.info(info))
                    continuation.yield(
                        .result(
                            STTOutput(
                                text: result.text,
                                language: generationParameters.language,
                                promptTokens: result.promptTokens,
                                generationTokens: result.generationTokens,
                                totalTokens: result.promptTokens + result.generationTokens,
                                promptTps: Double(result.promptTokens) / max(result.promptTime, 0.001),
                                generationTps: Double(result.generationTokens) / max(result.generationTime, 0.001),
                                totalTime: result.totalTime,
                                peakMemoryUsage: result.peakMemoryUsage
                            )
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public static func loadAssets(
        modelDirectory: URL,
        tokenizerDirectory: URL? = nil,
        cache: HubCache = .default
    ) async throws -> MiMoV2ASRAssets {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MiMoV2ASRConfig.self, from: configData)

        let manifest = try loadManifestIfPresent(in: modelDirectory)
        let resolvedTokenizerDirectory = try await resolveTokenizerDirectory(
            modelDirectory: modelDirectory,
            explicitTokenizerDirectory: tokenizerDirectory,
            manifest: manifest,
            cache: cache
        )

        let tokenizerConfigURL = resolvedTokenizerDirectory.appendingPathComponent("config.json")
        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfig = try JSONDecoder().decode(MiMoAudioTokenizerConfig.self, from: tokenizerConfigData)

        return MiMoV2ASRAssets(
            modelDirectory: modelDirectory,
            tokenizerDirectory: resolvedTokenizerDirectory,
            config: config,
            tokenizerConfig: tokenizerConfig,
            manifest: manifest
        )
    }

    public static func fromModelDirectory(
        _ modelDirectory: URL,
        tokenizerDirectory: URL? = nil,
        cache: HubCache = .default
    ) async throws -> MiMoV2ASRModel {
        let assets = try await loadAssets(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            cache: cache
        )
        let coreModel = try MiMoV2ASRCore.load(
            modelDirectory: assets.modelDirectory,
            config: assets.config
        )
        let audioTokenizerModel = try MiMoAudioTokenizerModel.load(
            modelDirectory: assets.tokenizerDirectory,
            config: assets.tokenizerConfig,
            activeQuantizers: assets.config.audioChannels
        )
        let model = MiMoV2ASRModel(
            assets: assets,
            coreModel: coreModel,
            audioTokenizerModel: audioTokenizerModel
        )
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        if let tokenizer = model.tokenizer {
            model.specialTokens = try MiMoPromptBuilder.resolveSpecialTokens(using: tokenizer)
        }
        return model
    }

    public static func fromPretrained(
        _ modelPath: String,
        tokenizerPath: String? = nil,
        cache: HubCache = .default
    ) async throws -> MiMoV2ASRModel {
        let fileManager = FileManager.default
        let expandedModelPath = (modelPath as NSString).expandingTildeInPath
        let modelDirectory: URL
        if fileManager.fileExists(atPath: expandedModelPath) {
            modelDirectory = URL(fileURLWithPath: expandedModelPath)
        } else {
            guard let repoID = Repo.ID(rawValue: modelPath) else {
                throw NSError(
                    domain: "MiMoV2ASRModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID or local path: \(modelPath)"]
                )
            }
            modelDirectory = try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: "safetensors",
                cache: cache
            )
        }

        let explicitTokenizerDirectory: URL?
        if let tokenizerPath {
            let expandedTokenizerPath = (tokenizerPath as NSString).expandingTildeInPath
            if fileManager.fileExists(atPath: expandedTokenizerPath) {
                explicitTokenizerDirectory = URL(fileURLWithPath: expandedTokenizerPath)
            } else if let tokenizerRepoID = Repo.ID(rawValue: tokenizerPath) {
                explicitTokenizerDirectory = try await ModelUtils.resolveOrDownloadModel(
                    repoID: tokenizerRepoID,
                    requiredExtension: "safetensors",
                    cache: cache
                )
            } else {
                explicitTokenizerDirectory = URL(fileURLWithPath: expandedTokenizerPath)
            }
        } else {
            explicitTokenizerDirectory = nil
        }

        return try await fromModelDirectory(
            modelDirectory,
            tokenizerDirectory: explicitTokenizerDirectory,
            cache: cache
        )
    }

    static func loadManifestIfPresent(in modelDirectory: URL) throws -> MiMoMLXManifest? {
        let manifestURL = modelDirectory.appendingPathComponent("mlx_manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(MiMoMLXManifest.self, from: data)
    }

    static func resolveTokenizerDirectory(
        modelDirectory: URL,
        explicitTokenizerDirectory: URL?,
        manifest: MiMoMLXManifest?,
        cache: HubCache = .default
    ) async throws -> URL {
        if let explicitTokenizerDirectory {
            return explicitTokenizerDirectory.standardizedFileURL
        }

        if let manifestTokenizerDir = manifest?.audioTokenizerDir {
            let candidate = modelDirectory
                .appendingPathComponent(manifestTokenizerDir)
                .standardizedFileURL
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        if let tokenizerRepo = manifest?.audioTokenizerRepo,
           let tokenizerRepoID = Repo.ID(rawValue: tokenizerRepo) {
            return try await ModelUtils.resolveOrDownloadModel(
                repoID: tokenizerRepoID,
                requiredExtension: "safetensors",
                cache: cache
            )
        }

        let sibling = modelDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("MiMo-Audio-Tokenizer")
            .standardizedFileURL
        if FileManager.default.fileExists(atPath: sibling.path) {
            return sibling
        }

        throw NSError(
            domain: "MiMoV2ASRModel",
            code: 2,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Unable to resolve MiMo audio tokenizer directory from explicit path, manifest, tokenizer repo, or sibling directory."
            ]
        )
    }

    private func encodeAudioChunks(_ chunks: [MLXArray]) throws -> [Int32] {
        var allCodes: [Int32] = []

        for chunk in chunks {
            let mel = MiMoAudioPreprocessing.computeLogMel(chunk)
            let codes = audioTokenizerModel.encodeMel(mel, nQuantizers: config.audioChannels)
            eval(codes)
            allCodes.append(contentsOf: codes.asArray(Int32.self))
        }

        guard !allCodes.isEmpty else { return [] }

        let frameCount = allCodes.count / config.audioChannels
        let remainder = frameCount % config.groupSize
        guard remainder != 0 else { return allCodes }

        let padFrames = config.groupSize - remainder
        let lastFrameStart = max(0, allCodes.count - config.audioChannels)
        let lastFrame = Array(allCodes[lastFrameStart..<allCodes.count])
        for _ in 0..<padFrames {
            allCodes.append(contentsOf: lastFrame)
        }
        return allCodes
    }

    private func transcribePrompt(
        _ prompt: MLXArray,
        generationParameters: STTGenerateParameters,
        specialTokens: MiMoSpecialTokens,
        onPartialText: ((String) throws -> Void)? = nil
    ) throws -> ([Int32], TimeInterval, TimeInterval) {
        let prompt3D = prompt.expandedDimensions(axis: 0)
        let globalCache = coreModel.makeGlobalCache()
        let promptStart = Date()
        let prefill = globalForward(prompt3D, cache: globalCache, specialTokens: specialTokens)
        let promptTime = Date().timeIntervalSince(promptStart)

        let generationStart = Date()
        var nextTextToken = sampleTextToken(
            logits: prefill.textLogits,
            temperature: generationParameters.temperature
        )
        var nextLocalHidden = prefill.localHiddenStates
        var generatedTextTokens: [Int32] = []
        let stopTokens = Set([Int32(tokenizer?.eosTokenId ?? 0), specialTokens.eot])
        var lastEmittedText = ""

        for _ in 0..<generationParameters.maxTokens {
            try Task.checkCancellation()

            if stopTokens.contains(nextTextToken) {
                break
            }

            generatedTextTokens.append(nextTextToken)
            if let onPartialText, let tokenizer {
                let decoded = cleanDecodedText(tokenizer.decode(tokens: generatedTextTokens.map(Int.init)))
                let delta = incrementalTextDelta(previous: lastEmittedText, current: decoded)
                if !delta.isEmpty {
                    try onPartialText(delta)
                }
                lastEmittedText = decoded
            }

            let speechTokens: MLXArray
            if nextTextToken == specialTokens.empty {
                speechTokens = localForward(nextLocalHidden)
            } else {
                var perChannel: [Int32] = []
                perChannel.reserveCapacity(config.groupSize * config.audioChannels)
                for _ in 0..<config.groupSize {
                    perChannel.append(contentsOf: config.parsedSpeechEmptyIDs.map(Int32.init))
                }
                speechTokens = MLXArray(perChannel).reshaped([config.groupSize, config.audioChannels])
            }

            let incremental = makeIncrementalInput(textToken: nextTextToken, speechTokens: speechTokens)
            let step = globalForward(incremental, cache: globalCache, specialTokens: specialTokens)
            nextTextToken = sampleTextToken(
                logits: step.textLogits,
                temperature: generationParameters.temperature
            )
            nextLocalHidden = step.localHiddenStates
        }

        return (generatedTextTokens, promptTime, Date().timeIntervalSince(generationStart))
    }

    private func globalForward(
        _ inputIDs: MLXArray,
        cache: [KVCache],
        specialTokens: MiMoSpecialTokens
    ) -> (textLogits: MLXArray, localHiddenStates: MLXArray) {
        let inputEmbeddings = prepareInputEmbeddings(inputIDs, specialTokens: specialTokens)
        let hiddenStates = coreModel.model(inputEmbeddings: inputEmbeddings, cache: cache)
        let lastHiddenState = hiddenStates[0..., (hiddenStates.shape[1] - 1) ..< hiddenStates.shape[1], 0...]
        let textLogits = coreModel.lmHead(lastHiddenState)
        let localHiddenStates = coreModel.hiddenStatesDowncast(lastHiddenState)
        eval(textLogits, localHiddenStates)
        return (textLogits, localHiddenStates)
    }

    private func prepareInputEmbeddings(
        _ inputIDs: MLXArray,
        specialTokens: MiMoSpecialTokens
    ) -> MLXArray {
        let batch = inputIDs.shape[0]
        let audioChannels = config.audioChannels
        let groupSize = config.groupSize
        let time = inputIDs.shape[2]
        let groups = time / groupSize

        let textInputIDs = inputIDs[0..., 0, .stride(from: 0, by: groupSize)]
        let speechInputIDs = inputIDs[0..., 1..., 0...]
            .reshaped([batch, audioChannels, groups, groupSize])
            .transposed(0, 2, 1, 3)

        let isSpeech = textInputIDs .== specialTokens.empty
        var speechEmbeddings = MLXArray.zeros([batch, groups, groupSize, config.inputLocalDim], type: Float.self)

        for channel in 0..<audioChannels {
            let currentSpeechIDs = speechInputIDs[0..., 0..., channel, 0...]
            var currentEmbeddings = coreModel.speechEmbeddings[channel](currentSpeechIDs)
            let mask = currentSpeechIDs .== Int32(config.parsedSpeechEmptyIDs[channel])
            currentEmbeddings = MLX.where(
                mask.expandedDimensions(axis: -1),
                MLXArray.zeros(currentEmbeddings.shape, dtype: currentEmbeddings.dtype),
                currentEmbeddings
            )
            speechEmbeddings = speechEmbeddings + currentEmbeddings
        }

        speechEmbeddings = speechEmbeddings * isSpeech.expandedDimensions(axis: -1).expandedDimensions(axis: -1).asType(.float32)

        let localInput = speechEmbeddings.reshaped([batch * groups, groupSize, config.inputLocalDim])
        var encodedSpeech = coreModel.inputLocalTransformer(inputEmbeddings: localInput)
        encodedSpeech = encodedSpeech.reshaped([batch, groups, groupSize, config.inputLocalDim])
        encodedSpeech = encodedSpeech * isSpeech.expandedDimensions(axis: -1).expandedDimensions(axis: -1).asType(.float32)

        let groupedSpeech = coreModel.speechGroupDowncast(
            encodedSpeech.reshaped([batch, groups, config.inputLocalDim * groupSize])
        )

        var textEmbeddings = coreModel.textEmbedding(textInputIDs)
        let textMask = textInputIDs .== specialTokens.empty
        textEmbeddings = MLX.where(
            textMask.expandedDimensions(axis: -1),
            MLXArray.zeros(textEmbeddings.shape, dtype: textEmbeddings.dtype),
            textEmbeddings
        )

        let combined = textEmbeddings + groupedSpeech
        eval(combined)
        return combined
    }

    private func localForward(_ localEmbeddings: MLXArray) -> MLXArray {
        let delayIterations = config.groupSize + (config.parsedDelayPattern.max() ?? 0)
        let localCache = coreModel.makeLocalCache()
        var currentEmbeddings = localEmbeddings
        var tokens = [Int32](
            repeating: 0,
            count: config.groupSize * config.audioChannels
        )

        for step in 0..<delayIterations {
            let hiddenStates = coreModel.localTransformer(inputEmbeddings: currentEmbeddings, cache: localCache)
            let lastHiddenState = hiddenStates[0..., (hiddenStates.shape[1] - 1) ..< hiddenStates.shape[1], 0...]
            var nextEmbeddings = MLXArray.zeros(currentEmbeddings.shape, dtype: currentEmbeddings.dtype)

            for channel in 0..<config.audioChannels {
                let start = config.parsedDelayPattern[channel]
                let end = start + config.groupSize
                guard start <= step, step < end else { continue }

                let logits = coreModel.localLogits(channel: channel, hiddenStates: lastHiddenState)
                let token = sampleSpeechToken(
                    logits: logits[0..., -1, 0...],
                    emptyTokenID: Int32(config.parsedSpeechEmptyIDs[channel])
                )
                let offset = (step - start) * config.audioChannels + channel
                tokens[offset] = token

                let tokenArray = MLXArray([token]).reshaped([1, 1])
                nextEmbeddings = nextEmbeddings + coreModel.speechEmbeddings[channel](tokenArray)
            }

            currentEmbeddings = nextEmbeddings
        }

        return MLXArray(tokens).reshaped([config.groupSize, config.audioChannels])
    }

    private func makeIncrementalInput(textToken: Int32, speechTokens: MLXArray) -> MLXArray {
        let textRow = [Int32](repeating: textToken, count: config.groupSize)
        let speech = speechTokens.asArray(Int32.self)

        var flattened: [Int32] = []
        flattened.reserveCapacity((config.audioChannels + 1) * config.groupSize)
        flattened.append(contentsOf: textRow)
        for channel in 0..<config.audioChannels {
            for step in 0..<config.groupSize {
                flattened.append(speech[step * config.audioChannels + channel])
            }
        }

        return MLXArray(flattened).reshaped([1, config.audioChannels + 1, config.groupSize])
    }

    private func sampleTextToken(logits: MLXArray, temperature: Float) -> Int32 {
        let lastLogits = logits[0..., -1, 0...]
        if temperature <= 0 {
            return lastLogits.argMax(axis: -1).item(Int32.self)
        }
        let sampled = categorical((lastLogits / temperature).expandedDimensions(axis: 0))
        return sampled.item(Int32.self)
    }

    private func sampleSpeechToken(logits: MLXArray, emptyTokenID: Int32) -> Int32 {
        let filtered = suppressToken(logits, tokenID: emptyTokenID)
        let scaled = filtered / MLXArray(0.9)
        let nucleus = applyTopP(scaled, topP: 0.95)
        let sampled = categorical(nucleus)
        return sampled.item(Int32.self)
    }

    private func suppressToken(_ logits: MLXArray, tokenID: Int32) -> MLXArray {
        let indices = MLXArray([tokenID]).reshaped([1, 1])
        let values = MLXArray.full([1, 1], values: MLXArray(-Float.infinity), dtype: logits.dtype)
        return putAlong(logits, indices, values: values, axis: -1)
    }

    private func applyTopP(_ logits: MLXArray, topP: Float) -> MLXArray {
        guard topP > 0, topP < 1 else { return logits }

        let vocabularySize = logits.dim(logits.ndim - 1)
        guard vocabularySize > 1 else { return logits }

        let logProbabilities = logSoftmax(logits, axis: -1)
        let sortedIndices = argSort(logProbabilities, axis: -1)
        let sortedProbabilities = exp(takeAlong(logProbabilities, sortedIndices, axis: -1))
        let cumulativeProbabilities = MLX.cumsum(sortedProbabilities, axis: -1)

        let positions = MLXArray(0 ..< vocabularySize).reshaped([1, -1]).asType(.int32)
        let inverseIndices = putAlong(
            MLXArray.zeros(sortedIndices.shape, type: Int32.self),
            sortedIndices.asType(Int32.self),
            values: positions,
            axis: -1
        )
        let cumulativeOriginalOrder = takeAlong(cumulativeProbabilities, inverseIndices, axis: -1)
        return MLX.where(
            cumulativeOriginalOrder .> MLXArray(1 - topP),
            logits,
            MLXArray(-Float.infinity).asType(logits.dtype)
        )
    }

    private func promptLanguage(from language: String?) -> MiMoPromptLanguage {
        guard let language else { return .auto }
        switch language.lowercased() {
        case "zh", "zh-cn", "chinese", "mandarin":
            return .chinese
        case "en", "en-us", "english":
            return .english
        default:
            return .auto
        }
    }

    private func cleanDecodedText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<|empty|>", with: "")
            .replacingOccurrences(of: "<|eot|>", with: "")
            .replacingOccurrences(of: "<|eostm|>", with: "")
            .replacingOccurrences(of: "<chinese>", with: "")
            .replacingOccurrences(of: "<english>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func transcribe(
        audio: MLXArray,
        generationParameters: STTGenerateParameters,
        onPartialText: ((String) throws -> Void)? = nil
    ) throws -> MiMoTranscriptionResult {
        guard let tokenizer, let specialTokens else {
            throw STTError.modelNotInitialized("Tokenizer or special tokens not loaded")
        }

        let startTime = Date()
        let waveform = MiMoAudioPreprocessing.normalizeWaveform(audio)
        let chunks = MiMoAudioPreprocessing.splitIntoChunks(
            waveform,
            sampleRate: MiMoAudioPreprocessing.targetSampleRate
        )
        let audioTokens = try encodeAudioChunks(chunks)
        let promptLanguage = promptLanguage(from: generationParameters.language)
        let prompt = MiMoPromptBuilder.buildASRPrompt(
            tokenizer: tokenizer,
            specialTokens: specialTokens,
            audioTokens: audioTokens,
            groupSize: config.groupSize,
            audioChannels: config.audioChannels,
            speechEmptyIDs: config.parsedSpeechEmptyIDs.map(Int32.init),
            language: promptLanguage
        )

        let (generated, promptTime, generationTime) = try transcribePrompt(
            prompt,
            generationParameters: generationParameters,
            specialTokens: specialTokens,
            onPartialText: onPartialText
        )
        let totalTime = Date().timeIntervalSince(startTime)

        Memory.clearCache()

        return MiMoTranscriptionResult(
            text: cleanDecodedText(tokenizer.decode(tokens: generated.map(Int.init))),
            promptTokens: prompt.shape[1],
            generationTokens: generated.count,
            promptTime: promptTime,
            generationTime: generationTime,
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    private func incrementalTextDelta(previous: String, current: String) -> String {
        guard !current.isEmpty else { return "" }
        guard current.hasPrefix(previous) else {
            return current == previous ? "" : current
        }
        return String(current.dropFirst(previous.count))
    }
}
