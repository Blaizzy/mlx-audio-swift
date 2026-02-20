//
//  Qwen3ForcedAligner.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 07/02/2026.
//

import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace
import Tokenizers

// MARK: - Force Align Result Types

/// One aligned item span with word/character text and timing.
public struct ForcedAlignItem: Sendable {
    public let text: String
    public let startTime: Double
    public let endTime: Double

    public init(text: String, startTime: Double, endTime: Double) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }
}

/// Forced alignment output for one audio sample.
public struct ForcedAlignResult: Sendable {
    public let items: [ForcedAlignItem]

    /// Number of tokens in the prompt.
    public let promptTokens: Int

    /// Total processing time in seconds.
    public let totalTime: Double

    /// Peak memory usage in GB.
    public let peakMemoryUsage: Double

    public init(
        items: [ForcedAlignItem],
        promptTokens: Int = 0,
        totalTime: Double = 0.0,
        peakMemoryUsage: Double = 0.0
    ) {
        self.items = items
        self.promptTokens = promptTokens
        self.totalTime = totalTime
        self.peakMemoryUsage = peakMemoryUsage
    }

    /// Full text from all aligned items.
    public var text: String {
        items.map { $0.text }.joined(separator: " ")
    }

    /// Segments in STTOutput-compatible format.
    public var segments: [[String: Any]] {
        items.map { item in
            [
                "text": item.text,
                "start": item.startTime,
                "end": item.endTime,
            ] as [String: Any]
        }
    }
}

// MARK: - Force Align Processor

public class ForceAlignProcessor {

    public init() {}

    // MARK: - Character Utilities
    public func isKeptChar(_ ch: Character) -> Bool {
        if ch == "'" { return true }
        // Letters and numbers
        return ch.isLetter || ch.isNumber
    }

    /// Remove non-kept characters from token.
    public func cleanToken(_ token: String) -> String {
        String(token.filter { isKeptChar($0) })
    }

    /// Check if character is a CJK ideograph.
    public func isCJKChar(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first else { return false }
        let code = scalar.value
        return (0x4E00 <= code && code <= 0x9FFF)    // CJK Unified Ideographs
            || (0x3400 <= code && code <= 0x4DBF)     // Extension A
            || (0x20000 <= code && code <= 0x2A6DF)   // Extension B
            || (0x2A700 <= code && code <= 0x2B73F)   // Extension C
            || (0x2B740 <= code && code <= 0x2B81F)   // Extension D
            || (0x2B820 <= code && code <= 0x2CEAF)   // Extension E
            || (0xF900 <= code && code <= 0xFAFF)     // Compatibility Ideographs
    }

    // MARK: - Tokenization

    /// Tokenize text with Chinese characters (each CJK char is its own token).
    public func tokenizeChineseMixed(_ text: String) -> [String] {
        var tokens: [String] = []
        var currentLatin: [Character] = []

        func flushLatin() {
            if !currentLatin.isEmpty {
                let token = String(currentLatin)
                let cleaned = cleanToken(token)
                if !cleaned.isEmpty {
                    tokens.append(cleaned)
                }
                currentLatin = []
            }
        }

        for ch in text {
            if isCJKChar(ch) {
                flushLatin()
                tokens.append(String(ch))
            } else if isKeptChar(ch) {
                currentLatin.append(ch)
            } else {
                flushLatin()
            }
        }

        flushLatin()
        return tokens
    }

    /// Split a segment that may contain CJK characters mixed with non-CJK.
    private func splitSegmentWithChinese(_ seg: String) -> [String] {
        var tokens: [String] = []
        var buf: [Character] = []

        func flushBuf() {
            if !buf.isEmpty {
                tokens.append(String(buf))
                buf = []
            }
        }

        for ch in seg {
            if isCJKChar(ch) {
                flushBuf()
                tokens.append(String(ch))
            } else {
                buf.append(ch)
            }
        }

        flushBuf()
        return tokens
    }

    /// Tokenize space-separated languages (English, etc.).
    public func tokenizeSpaceLang(_ text: String) -> [String] {
        var tokens: [String] = []
        for seg in text.split(separator: " ") {
            let cleaned = cleanToken(String(seg))
            if !cleaned.isEmpty {
                tokens.append(contentsOf: splitSegmentWithChinese(cleaned))
            }
        }
        return tokens
    }

    // MARK: - Timestamp Fixing (LIS)


    public func fixTimestamp(_ data: [Double]) -> [Int] {
        let n = data.count
        if n == 0 { return [] }

        let intData = data.map { Int($0) }

        var dp = [Int](repeating: 1, count: n)
        var parent = [Int](repeating: -1, count: n)

        for i in 1..<n {
            for j in 0..<i {
                if intData[j] <= intData[i] && dp[j] + 1 > dp[i] {
                    dp[i] = dp[j] + 1
                    parent[i] = j
                }
            }
        }

        let maxLength = dp.max() ?? 0
        let maxIdx = dp.firstIndex(of: maxLength) ?? 0

        // Reconstruct LIS indices
        var lisIndices: [Int] = []
        var idx = maxIdx
        while idx != -1 {
            lisIndices.append(idx)
            idx = parent[idx]
        }
        lisIndices.reverse()

        var isNormal = [Bool](repeating: false, count: n)
        for idx in lisIndices {
            isNormal[idx] = true
        }

        var result = intData

        var i = 0
        while i < n {
            if !isNormal[i] {
                var j = i
                while j < n && !isNormal[j] {
                    j += 1
                }

                let anomalyCount = j - i

                if anomalyCount <= 2 {
                    // For small anomalies, use nearest valid neighbor
                    var leftVal: Int? = nil
                    for k in stride(from: i - 1, through: 0, by: -1) {
                        if isNormal[k] {
                            leftVal = result[k]
                            break
                        }
                    }

                    var rightVal: Int? = nil
                    for k in j..<n {
                        if isNormal[k] {
                            rightVal = result[k]
                            break
                        }
                    }

                    for k in i..<j {
                        if leftVal == nil {
                            result[k] = rightVal ?? 0
                        } else if rightVal == nil {
                            result[k] = leftVal!
                        } else {
                            result[k] = (k - (i - 1)) <= (j - k) ? leftVal! : rightVal!
                        }
                    }
                } else {
                    // For large anomalies, interpolate linearly
                    var leftVal: Int? = nil
                    for k in stride(from: i - 1, through: 0, by: -1) {
                        if isNormal[k] {
                            leftVal = result[k]
                            break
                        }
                    }

                    var rightVal: Int? = nil
                    for k in j..<n {
                        if isNormal[k] {
                            rightVal = result[k]
                            break
                        }
                    }

                    if let lv = leftVal, let rv = rightVal {
                        let step = Double(rv - lv) / Double(anomalyCount + 1)
                        for k in i..<j {
                            result[k] = lv + Int(step * Double(k - i + 1))
                        }
                    } else if let lv = leftVal {
                        for k in i..<j { result[k] = lv }
                    } else if let rv = rightVal {
                        for k in i..<j { result[k] = rv }
                    }
                }

                i = j
            } else {
                i += 1
            }
        }

        return result
    }

    // MARK: - Timestamp Encoding/Parsing


    public func encodeTimestamp(text: String, language: String) -> ([String], String) {
        let lang = language.lowercased()

        let wordList: [String]
        if lang == "chinese" {
            wordList = tokenizeChineseMixed(text)
        } else {
            // Default: space-separated languages (i.e, English)
            // JP and Ko require external tokenizers not available in Swift
            wordList = tokenizeSpaceLang(text)
        }

        let inputText = "<|audio_start|><|audio_pad|><|audio_end|>"
            + wordList.joined(separator: "<timestamp><timestamp>")
            + "<timestamp><timestamp>"

        return (wordList, inputText)
    }

    /// Parse timestamps into word-level alignments.
    public func parseTimestamp(
        wordList: [String],
        timestamp: [Double]
    ) -> [ForcedAlignItem] {
        let timestampFixed = fixTimestamp(timestamp)

        var items: [ForcedAlignItem] = []
        for (i, word) in wordList.enumerated() {
            let startTimeMs = timestampFixed[i * 2]
            let endTimeMs = timestampFixed[i * 2 + 1]
            items.append(ForcedAlignItem(
                text: word,
                startTime: Double(startTimeMs) / 1000.0,
                endTime: Double(endTimeMs) / 1000.0
            ))
        }

        return items
    }
}

// MARK: - Qwen3 Forced Aligner Model

public class Qwen3ForcedAlignerModel: Module {
    public let config: Qwen3ASRConfig

    @ModuleInfo(key: "audio_tower") var audioTower: Qwen3ASRAudioEncoder
    @ModuleInfo(key: "model") var model: Qwen3ASRTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public var tokenizer: Tokenizer?
    let alignerProcessor = ForceAlignProcessor()

    public init(_ config: Qwen3ASRConfig) {
        self.config = config

        self._audioTower.wrappedValue = Qwen3ASRAudioEncoder(config.audioConfig)
        self._model.wrappedValue = Qwen3ASRTextModel(config.textConfig)

        let classifyNum = config.classifyNum ?? 5000
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            classifyNum,
            bias: false
        )
    }

    public func callAsFunction(
        inputIds: MLXArray,
        inputFeatures: MLXArray? = nil,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        var inputsEmbeds = model.embedTokens(inputIds)

        if let features = inputFeatures {
            let audioFeatures = audioTower(features, featureAttentionMask: featureAttentionMask)
                .asType(inputsEmbeds.dtype)

            let flatMask = (inputIds .== MLXArray(Int32(config.audioTokenId))).reshaped(-1)
            let batchSize = inputsEmbeds.dim(0)
            let seqLen = inputsEmbeds.dim(1)
            let hiddenDim = inputsEmbeds.dim(2)

            let numAudioTokens = Int(flatMask.asType(.int32).sum().item(Int32.self))
            if numAudioTokens > 0 && audioFeatures.dim(0) > 0 {
                let numToReplace = min(numAudioTokens, audioFeatures.dim(0))
                let flatEmbeds = inputsEmbeds.reshaped(-1, hiddenDim)
                let totalLen = flatEmbeds.dim(0)

                // Audio tokens are contiguous — find start and splice directly
                let maskValues = flatMask.asType(.int32).asArray(Int32.self)
                var firstAudioPos = -1
                for (i, v) in maskValues.enumerated() {
                    if v != 0 { firstAudioPos = i; break }
                }
                if firstAudioPos >= 0 {
                    let endAudioPos = firstAudioPos + numToReplace
                    var parts: [MLXArray] = []
                    if firstAudioPos > 0 {
                        parts.append(flatEmbeds[0..<firstAudioPos])
                    }
                    parts.append(audioFeatures[0..<numToReplace])
                    if endAudioPos < totalLen {
                        parts.append(flatEmbeds[endAudioPos..<totalLen])
                    }
                    inputsEmbeds = MLX.concatenated(parts, axis: 0).reshaped(batchSize, seqLen, hiddenDim)
                }
            }
        }

        let hiddenStates = model(inputsEmbeds: inputsEmbeds, cache: nil)
        return lmHead(hiddenStates)
    }

    // MARK: - Audio Preprocessing

    public func preprocessAudio(_ audio: MLXArray) -> (MLXArray, MLXArray, Int) {
        let melSpec = MLXAudioCore.computeMelSpectrogram(
            audio: audio,
            sampleRate: 16000,
            nFft: 400,
            hopLength: 160,
            nMels: config.audioConfig.numMelBins
        )

        let transposed = melSpec.transposed(1, 0)
        let inputFeatures = transposed.expandedDimensions(axis: 0)

        let numFrames = melSpec.dim(0)
        let featureAttentionMask = MLX.ones([1, numFrames]).asType(.int32)

        let audioLengths = featureAttentionMask.sum(axis: -1).asType(.int32)
        let aftercnnLens = getFeatExtractOutputLengths(audioLengths)
        let numAudioTokens = Int(aftercnnLens[0].item(Int32.self))

        return (inputFeatures, featureAttentionMask, numAudioTokens)
    }

    // MARK: - Generate Alignment

    public func generate(
        audio: MLXArray,
        text: String,
        language: String = "English"
    ) -> ForcedAlignResult {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded")
        }

        let startTime = Date()

        let (inputFeatures, featureAttentionMask, numAudioTokens) = preprocessAudio(audio)

        let (wordList, alignerInputText) = alignerProcessor.encodeTimestamp(text: text, language: language)

        // Replace single audio_pad with correct number
        let expandedText = alignerInputText.replacingOccurrences(
            of: "<|audio_pad|>",
            with: String(repeating: "<|audio_pad|>", count: numAudioTokens)
        )

        let inputIdsList = tokenizer.encode(text: expandedText)
        let inputIds = MLXArray(inputIdsList.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptTokenCount = inputIds.dim(1)

        let logits = callAsFunction(
            inputIds: inputIds,
            inputFeatures: inputFeatures,
            featureAttentionMask: featureAttentionMask
        )
        eval(logits)

        let outputIds = logits.argMax(axis: -1)

        // Extract timestamps at timestamp token positions
        let timestampTokenId = config.timestampTokenId ?? 151705
        let timestampSegmentTime = Double(config.timestampSegmentTime ?? 80.0)

        let inputIdsFlat = inputIds[0]
        let outputIdsFlat = outputIds[0]

        let totalTokens = inputIdsFlat.dim(0)
        var timestampValues: [Double] = []

        for i in 0..<totalTokens {
            let tokenId = Int(inputIdsFlat[i].item(Int32.self))
            if tokenId == timestampTokenId {
                let predictedClass = Int(outputIdsFlat[i].item(Int32.self))
                timestampValues.append(Double(predictedClass) * timestampSegmentTime)
            }
        }

        // Parse timestamps into word-level alignments
        let items = alignerProcessor.parseTimestamp(
            wordList: wordList,
            timestamp: timestampValues
        )

        let endTime = Date()
        let peakMemory = Double(Memory.peakMemory) / 1e9
        Memory.clearCache()

        return ForcedAlignResult(
            items: items,
            promptTokens: promptTokenCount,
            totalTime: endTime.timeIntervalSince(startTime),
            peakMemoryUsage: peakMemory
        )
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        let isFormatted = !weights.keys.contains { $0.hasPrefix("thinker.") }

        for (key, var value) in weights {
            var newKey = key

            if newKey.hasPrefix("thinker.") {
                newKey = String(newKey.dropFirst("thinker.".count))
            }

            // ForcedAligner uses lm_head, don't skip it

            // Transpose Conv2d weights from PyTorch format
            if !isFormatted && newKey.contains("conv2d") && newKey.contains("weight") && value.ndim == 4 {
                value = value.transposed(0, 2, 3, 1)
            }

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> Qwen3ForcedAlignerModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "Qwen3ForcedAlignerModel",
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

        // Load config
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: configData)

        let perLayerQuantization = config.perLayerQuantization

        let model = Qwen3ForcedAlignerModel(config)


        try Qwen3ASRModel.generateTokenizerJSONIfMissing(in: modelDir)


        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitizedWeights = Qwen3ForcedAlignerModel.sanitize(weights: weights)

        if perLayerQuantization != nil {
            quantize(model: model) { path, module in
                if path.hasPrefix("audio_tower") {
                    return nil
                }
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])
        eval(model)

        return model
    }
}
