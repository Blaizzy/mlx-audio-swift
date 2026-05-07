import Foundation
@preconcurrency import MLX
import MLXAudioCore
import Tokenizers

public let mossTTSAudioPlaceholder = "<|audio|>"

public protocol MossTTSTextTokenizing {
    func encode(_ text: String) -> [Int]
    func decode(_ tokenIDs: [Int]) -> String
    func tokenString(for tokenID: Int) -> String?
}

public final class MossTTSTokenizerAdapter: MossTTSTextTokenizing {
    private let tokenizer: any Tokenizer
    private let tokenStringsByID: [Int: String]

    public init(tokenizer: any Tokenizer, tokenStringsByID: [Int: String]) {
        self.tokenizer = tokenizer
        self.tokenStringsByID = tokenStringsByID
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ tokenIDs: [Int]) -> String {
        tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)
    }

    public func tokenString(for tokenID: Int) -> String? {
        tokenStringsByID[tokenID]
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> MossTTSTokenizerAdapter {
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        return MossTTSTokenizerAdapter(
            tokenizer: tokenizer,
            tokenStringsByID: try loadAddedTokenStrings(modelDir: modelDir)
        )
    }

    public static func loadAddedTokenStrings(modelDir: URL) throws -> [Int: String] {
        let tokenizerConfig = modelDir.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: tokenizerConfig),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let decoder = object["added_tokens_decoder"] as? [String: [String: Any]] {
            var result: [Int: String] = [:]
            for (key, value) in decoder {
                guard let id = Int(key), let content = value["content"] as? String else { continue }
                result[id] = content
            }
            if !result.isEmpty { return result }
        }

        let addedTokens = modelDir.appendingPathComponent("added_tokens.json")
        if let data = try? Data(contentsOf: addedTokens),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Int] {
            return Dictionary(uniqueKeysWithValues: object.map { ($0.value, $0.key) })
        }
        return [:]
    }
}

public struct MossTTSConversationMessage {
    public var role: String
    public var content: String
    public var audioCodesList: [MLXArray]
}

public struct MossTTSProcessorBatch {
    public var inputIDs: MLXArray
    public var attentionMask: MLXArray
}

public func mossTTSApplyDelayPattern(_ codes: MLXArray, padCode: Int) throws -> MLXArray {
    guard codes.ndim == 2 else {
        throw AudioGenerationError.invalidInput("Expected codes shape [frames, n_vq], got \(codes.shape)")
    }
    let frames = codes.dim(0)
    let nVQ = codes.dim(1)
    var values = Array(repeating: Int32(padCode), count: (frames + nVQ - 1) * nVQ)
    let source = codes.asType(.int32).asArray(Int32.self)
    for codebook in 0 ..< nVQ {
        for frame in 0 ..< frames {
            values[(codebook + frame) * nVQ + codebook] = source[frame * nVQ + codebook]
        }
    }
    return MLXArray(values, [frames + nVQ - 1, nVQ]).asType(.int32)
}

public func mossTTSApplyDeDelayPattern(_ delayedCodes: MLXArray) throws -> MLXArray {
    guard delayedCodes.ndim == 2 else {
        throw AudioGenerationError.invalidInput("Expected delay_codes shape [frames, n_vq], got \(delayedCodes.shape)")
    }
    let delayedFrames = delayedCodes.dim(0)
    let nVQ = delayedCodes.dim(1)
    let outputLength = delayedFrames - nVQ + 1
    guard outputLength > 0 else {
        return MLXArray.zeros([0, nVQ], type: Int32.self)
    }
    let source = delayedCodes.asType(.int32).asArray(Int32.self)
    var values = Array(repeating: Int32(0), count: outputLength * nVQ)
    for codebook in 0 ..< nVQ {
        for frame in 0 ..< outputLength {
            values[frame * nVQ + codebook] = source[(codebook + frame) * nVQ + codebook]
        }
    }
    return MLXArray(values, [outputLength, nVQ]).asType(.int32)
}

public class MossTTSDelayProcessor {
    public let tokenizer: MossTTSTextTokenizing
    public let config: MossTTSConfig
    public let useDelayPattern: Bool
    public let appendAudioStartForGeneration: Bool

    private let audioUserSlotToken: String
    private let audioAssistantGenSlotToken: String
    private let audioAssistantDelaySlotToken: String
    private let audioStartToken: String
    private let audioEndToken: String

    public init(
        tokenizer: MossTTSTextTokenizing,
        config: MossTTSConfig,
        useDelayPattern: Bool = true,
        appendAudioStartForGeneration: Bool = false
    ) throws {
        self.tokenizer = tokenizer
        self.config = config
        self.useDelayPattern = useDelayPattern
        self.appendAudioStartForGeneration = appendAudioStartForGeneration
        self.audioUserSlotToken = try Self.tokenString(tokenizer, id: config.audioUserSlotTokenID)
        self.audioAssistantGenSlotToken = try Self.tokenString(tokenizer, id: config.audioAssistantGenSlotTokenID)
        self.audioAssistantDelaySlotToken = try Self.tokenString(tokenizer, id: config.audioAssistantDelaySlotTokenID)
        self.audioStartToken = try Self.tokenString(tokenizer, id: config.audioStartTokenID)
        self.audioEndToken = try Self.tokenString(tokenizer, id: config.audioEndTokenID)
    }

    private static func tokenString(_ tokenizer: MossTTSTextTokenizing, id: Int) throws -> String {
        if let value = tokenizer.tokenString(for: id), !value.isEmpty {
            return value
        }
        let decoded = tokenizer.decode([id])
        guard !decoded.isEmpty else {
            throw AudioGenerationError.invalidInput("Tokenizer cannot resolve MOSS special token id \(id)")
        }
        return decoded
    }

    public func buildUserMessage(
        text: String? = nil,
        reference: [MLXArray?]? = nil,
        instruction: String? = nil,
        tokens: Int? = nil,
        quality: String? = nil,
        soundEvent: String? = nil,
        ambientSound: String? = nil,
        language: String? = nil,
        scene: String? = nil
    ) -> MossTTSConversationMessage {
        var audioCodes: [MLXArray] = []
        let referenceText: String
        if let reference {
            var parts: [String] = []
            for (index, item) in reference.enumerated() {
                if let item {
                    parts.append("[S\(index + 1)]:\n\(mossTTSAudioPlaceholder)")
                    audioCodes.append(item)
                } else {
                    parts.append("[S\(index + 1)]: None")
                }
            }
            referenceText = parts.joined(separator: "\n")
        } else {
            referenceText = "None"
        }

        var fields: [(name: String, value: String)] = [
            ("Reference(s)", referenceText),
            ("Instruction", instruction ?? "None"),
            ("Tokens", tokens.map(String.init) ?? "None"),
            ("Quality", quality ?? "None"),
            ("Sound Event", soundEvent ?? "None"),
            ("Ambient Sound", ambientSound ?? "None"),
            ("Language", language ?? "None"),
        ]
        if config.usesDialogueScenePrompt {
            fields.append(("Scene", scene ?? "None"))
        }
        fields.append(("Text", text ?? "None"))

        let content = """
        <user_inst>
        \(fields.map { "- \($0.name):\n\($0.value)" }.joined(separator: "\n"))
        </user_inst>
        """

        return MossTTSConversationMessage(role: "user", content: content, audioCodesList: audioCodes)
    }

    public func buildAssistantMessage(
        audioCodesList: [MLXArray],
        content: String = mossTTSAudioPlaceholder
    ) -> MossTTSConversationMessage {
        MossTTSConversationMessage(role: "assistant", content: content, audioCodesList: audioCodesList)
    }

    public static func applyChatTemplate(
        role: String,
        content: String,
        addGenerationPrompt: Bool
    ) -> String {
        var rendered = "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        if addGenerationPrompt {
            rendered += "<|im_start|>assistant\n"
        }
        return rendered
    }

    private func replaceAudioPlaceholders(
        content: String,
        lengths: [Int],
        genSlotToken: String,
        delaySlotToken: String,
        audioStartToken: String,
        audioEndToken: String
    ) throws -> String {
        guard config.nVQ >= 1 else {
            throw AudioGenerationError.invalidInput("n_vq must be >= 1, got \(config.nVQ)")
        }
        guard content.mossTTSCount(of: mossTTSAudioPlaceholder) == lengths.count else {
            throw AudioGenerationError.invalidInput("Audio placeholders do not match audio code lengths")
        }
        var output = content
        for length in lengths {
            guard length >= 0 else {
                throw AudioGenerationError.invalidInput("length must be >= 0, got \(length)")
            }
            let block: String
            if length == 0 {
                block = "\(audioStartToken)\(audioEndToken)"
            } else if !delaySlotToken.isEmpty {
                block = audioStartToken
                    + String(repeating: genSlotToken, count: length)
                    + String(repeating: delaySlotToken, count: config.nVQ - 1)
                    + audioEndToken
            } else {
                block = audioStartToken + String(repeating: genSlotToken, count: length) + audioEndToken
            }
            if let range = output.range(of: mossTTSAudioPlaceholder) {
                output.replaceSubrange(range, with: block)
            }
        }
        return output
    }

    private func getUnifiedCodes(
        role: String,
        content: String,
        audioCodesList: [MLXArray],
        truncation: Bool
    ) throws -> MLXArray {
        let audioGenSlotToken: String
        let audioDelaySlotToken: String
        let effectiveTruncation: Bool
        if role == "user" {
            audioGenSlotToken = audioUserSlotToken
            audioDelaySlotToken = audioUserSlotToken
            effectiveTruncation = false
        } else {
            audioGenSlotToken = audioAssistantGenSlotToken
            audioDelaySlotToken = audioAssistantDelaySlotToken
            effectiveTruncation = truncation
        }

        var normalizedCodes = try normalizeAudioCodesList(audioCodesList, nVQ: config.nVQ)
        var renderedContent = content
        if normalizedCodes.count > 1 && renderedContent.contains(mossTTSAudioPlaceholder) {
            let merged = mergeConsecutiveAudioPlaceholders(renderedContent, audioCodesList: normalizedCodes)
            renderedContent = merged.content
            normalizedCodes = merged.audioCodesList
        }
        renderedContent = try replaceAudioPlaceholders(
            content: renderedContent,
            lengths: normalizedCodes.map { $0.dim(0) },
            genSlotToken: audioGenSlotToken,
            delaySlotToken: useDelayPattern ? audioDelaySlotToken : "",
            audioStartToken: audioStartToken,
            audioEndToken: audioEndToken
        )

        let textCodes = tokenizer.encode(renderedContent)
        let audioStartIndices = textCodes.enumerated()
            .compactMap { $0.element == config.audioStartTokenID ? $0.offset : nil }
        let audioEndIndices = textCodes.enumerated()
            .compactMap { $0.element == config.audioEndTokenID ? $0.offset : nil }
        guard audioStartIndices.count == normalizedCodes.count,
              audioEndIndices.count == normalizedCodes.count
        else {
            throw AudioGenerationError.invalidInput("Audio placeholders do not match the provided audio codes list")
        }

        let nVQ = config.nVQ
        let delayAudioCodes: MLXArray
        if normalizedCodes.isEmpty {
            delayAudioCodes = MLX.full([textCodes.count, nVQ], values: Int32(config.audioPadCode), type: Int32.self)
        } else {
            var sections: [MLXArray] = []
            var prefixIndex = 0
            for index in normalizedCodes.indices {
                let audioStartIndex = audioStartIndices[index]
                let audioEndIndex = audioEndIndices[index]
                let codes = normalizedCodes[index]
                let effectiveCodes = useDelayPattern
                    ? try mossTTSApplyDelayPattern(codes, padCode: config.audioPadCode)
                    : codes.asType(.int32)
                let padRows = max(audioStartIndex - prefixIndex + 1, 0)
                sections.append(MLX.full([padRows, nVQ], values: Int32(config.audioPadCode), type: Int32.self))
                sections.append(effectiveCodes)
                prefixIndex = audioEndIndex
            }

            if effectiveTruncation && useDelayPattern && nVQ > 1, let last = sections.popLast() {
                let keep = max(last.dim(0) - (nVQ - 1), 0)
                sections.append(last[0..<keep, 0...])
            } else if !effectiveTruncation, let lastEnd = audioEndIndices.last {
                sections.append(
                    MLX.full(
                        [max(textCodes.count - lastEnd, 0), nVQ],
                        values: Int32(config.audioPadCode),
                        type: Int32.self
                    )
                )
            }
            delayAudioCodes = MLX.concatenated(sections, axis: 0)
        }

        let outputLength = min(textCodes.count, delayAudioCodes.dim(0))
        let text = MLXArray(textCodes.prefix(outputLength).map(Int32.init), [outputLength, 1]).asType(.int32)
        return MLX.concatenated([text, delayAudioCodes[0..<outputLength, 0...]], axis: 1)
    }

    private func normalizeAudioCodesList(_ audioCodesList: [MLXArray], nVQ: Int) throws -> [MLXArray] {
        try audioCodesList.map { codes in
            guard codes.ndim == 2 else {
                throw AudioGenerationError.invalidInput("Expected audio codes shape [frames, n_vq], got \(codes.shape)")
            }
            var normalized = codes
            if normalized.dim(1) < nVQ && normalized.dim(0) >= nVQ {
                normalized = normalized.transposed(1, 0)
            }
            guard normalized.dim(1) >= nVQ else {
                throw AudioGenerationError.invalidInput(
                    "audio_codes channels (\(normalized.dim(1))) < model n_vq (\(nVQ))"
                )
            }
            return normalized[0..., 0..<nVQ].asType(.int32)
        }
    }

    private func mergeConsecutiveAudioPlaceholders(
        _ content: String,
        audioCodesList: [MLXArray]
    ) -> (content: String, audioCodesList: [MLXArray]) {
        let placeholder = mossTTSAudioPlaceholder
        guard content.mossTTSCount(of: placeholder) == audioCodesList.count,
              audioCodesList.count > 1
        else {
            return (content, audioCodesList)
        }

        var resultContent = ""
        var resultCodes: [MLXArray] = []
        var remaining = content[...]
        var codeIndex = 0

        while let range = remaining.range(of: placeholder) {
            resultContent += String(remaining[..<range.lowerBound])
            var merged = audioCodesList[codeIndex]
            codeIndex += 1
            var suffix = remaining[range.upperBound...]

            while codeIndex < audioCodesList.count,
                  let nextRange = suffix.range(of: placeholder),
                  suffix[..<nextRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                merged = MLX.concatenated([merged, audioCodesList[codeIndex]], axis: 0)
                codeIndex += 1
                suffix = suffix[nextRange.upperBound...]
            }

            resultContent += placeholder
            resultCodes.append(merged)
            remaining = suffix
        }
        resultContent += String(remaining)
        return (resultContent, resultCodes)
    }

    public func callAsFunction(
        _ conversations: [[MossTTSConversationMessage]],
        mode: String = "generation",
        applyChatTemplate: Bool = true
    ) throws -> MossTTSProcessorBatch {
        guard mode == "generation" || mode == "continuation" else {
            throw AudioGenerationError.invalidInput("mode must be generation or continuation")
        }

        let truncation = mode == "continuation"
        var inputIDsList: [MLXArray] = []
        for conversation in conversations {
            guard !conversation.isEmpty else {
                throw AudioGenerationError.invalidInput("Conversation must not be empty")
            }
            if (mode == "generation") == (conversation.count % 2 == 0) {
                throw AudioGenerationError.invalidInput("Invalid conversation length for mode")
            }
            if (mode == "generation") != (conversation.last?.role == "user") {
                throw AudioGenerationError.invalidInput("Invalid final role for mode")
            }

            var unified: [MLXArray] = []
            for (messageIndex, message) in conversation.enumerated() {
                let addGenerationPrompt = mode == "generation" && messageIndex == conversation.count - 1
                let content = applyChatTemplate
                    ? Self.applyChatTemplate(
                        role: message.role,
                        content: message.content,
                        addGenerationPrompt: addGenerationPrompt
                    )
                    : message.content
                unified.append(
                    try getUnifiedCodes(
                        role: message.role,
                        content: content,
                        audioCodesList: message.audioCodesList,
                        truncation: truncation
                    )
                )
            }
            var inputIDs = MLX.concatenated(unified, axis: 0)
            if appendAudioStartForGeneration && mode == "generation" {
                var row = Array(repeating: Int32(config.audioPadCode), count: config.nVQ + 1)
                row[0] = Int32(config.audioStartTokenID)
                inputIDs = MLX.concatenated(
                    [inputIDs, MLXArray(row, [1, config.nVQ + 1]).asType(.int32)],
                    axis: 0
                )
            }
            inputIDsList.append(inputIDs)
        }
        return pad(inputIDsList)
    }

    private func pad(_ inputIDsList: [MLXArray]) -> MossTTSProcessorBatch {
        let maxLength = inputIDsList.map { $0.dim(0) }.max() ?? 0
        var padded: [MLXArray] = []
        var masks: [MLXArray] = []
        for var inputIDs in inputIDsList {
            let padLength = maxLength - inputIDs.dim(0)
            if padLength > 0 {
                var flat = Array(repeating: Int32(config.audioPadCode), count: padLength * (config.nVQ + 1))
                for row in 0 ..< padLength {
                    flat[row * (config.nVQ + 1)] = Int32(config.padTokenID)
                }
                inputIDs = MLX.concatenated(
                    [MLXArray(flat, [padLength, config.nVQ + 1]).asType(.int32), inputIDs],
                    axis: 0
                )
            }
            padded.append(inputIDs)
            masks.append(
                MLX.concatenated(
                    [
                        MLXArray.zeros([padLength], dtype: .bool),
                        MLXArray.ones([maxLength - padLength], dtype: .bool),
                    ],
                    axis: 0
                )
            )
        }
        return MossTTSProcessorBatch(
            inputIDs: MLX.stacked(padded, axis: 0).asType(.int32),
            attentionMask: MLX.stacked(masks, axis: 0).asType(.bool)
        )
    }
}

public final class MossTTSLocalProcessor: MossTTSDelayProcessor {
    public init(tokenizer: MossTTSTextTokenizing, config: MossTTSConfig) throws {
        try super.init(
            tokenizer: tokenizer,
            config: config,
            useDelayPattern: false,
            appendAudioStartForGeneration: true
        )
    }
}

private extension String {
    func mossTTSCount(of needle: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var searchRange = startIndex..<endIndex
        while let range = range(of: needle, range: searchRange) {
            count += 1
            searchRange = range.upperBound..<endIndex
        }
        return count
    }
}
