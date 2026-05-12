import Foundation
import MLX
import Tokenizers

enum MiMoPromptLanguage {
    case auto
    case chinese
    case english
}

struct MiMoSpecialTokens: Sendable {
    let sosp: Int32
    let eosp: Int32
    let empty: Int32
    let human: Int32?
    let speechLM: Int32?
    let sostm: Int32
    let eostm: Int32
    let eot: Int32
    let imStart: Int32
    let imEnd: Int32
}

struct MiMoInputSegment: Sendable {
    var text: String?
    var tokenIDs: [Int32]?
    var audioTokens: [Int32]?
    var addSospEosp = true

    init(text: String) {
        self.text = text
        self.tokenIDs = nil
        self.audioTokens = nil
    }

    init(tokenIDs: [Int32]) {
        self.text = nil
        self.tokenIDs = tokenIDs
        self.audioTokens = nil
    }

    init(audioTokens: [Int32], addSospEosp: Bool = true) {
        self.text = nil
        self.tokenIDs = nil
        self.audioTokens = audioTokens
        self.addSospEosp = addSospEosp
    }
}

enum MiMoPromptBuilder {
    static let fixedChineseTemplateTokenIDs: [Int32] = [44063, 111268, 43815, 105359, 12857, 87335, 68805]
    static let fixedChineseAssistantPrefixTokenIDs: [Int32] = [13708, 766, 1339, 522, 26865, 397, 27, 331, 7346, 29]
    static let fixedEnglishAssistantPrefixTokenIDs: [Int32] = [13708, 766, 1339, 522, 26865, 397, 27, 974, 975, 678, 29]

    static let chineseTemplates: [String] = [
        "请将这段语音转换为文字",
        "帮我识别这个音频文件中的内容",
        "把这段录音转成文本",
    ]

    static let englishTemplates: [String] = [
        "Please transcribe this audio file",
        "Convert this speech recording to text",
        "Transcribe the following voice message",
    ]

    static func resolveSpecialTokens(using tokenizer: any Tokenizer) throws -> MiMoSpecialTokens {
        try .init(
            sosp: singleTokenID("<|sosp|>", using: tokenizer),
            eosp: singleTokenID("<|eosp|>", using: tokenizer),
            empty: singleTokenID("<|empty|>", using: tokenizer),
            human: try? singleTokenID("<|Human|>", using: tokenizer),
            speechLM: try? singleTokenID("<|SpeechLM|>", using: tokenizer),
            sostm: singleTokenID("<|sostm|>", using: tokenizer),
            eostm: singleTokenID("<|eostm|>", using: tokenizer),
            eot: singleTokenID("<|eot|>", using: tokenizer),
            imStart: singleTokenID("<|im_start|>", using: tokenizer),
            imEnd: singleTokenID("<|im_end|>", using: tokenizer)
        )
    }

    static func buildASRPrompt(
        tokenizer: any Tokenizer,
        specialTokens: MiMoSpecialTokens,
        audioTokens: [Int32],
        groupSize: Int,
        audioChannels: Int,
        speechEmptyIDs: [Int32],
        language: MiMoPromptLanguage = .auto
    ) -> MLXArray {
        let templateSegment: MiMoInputSegment
        let assistantPrefixSegment: MiMoInputSegment
        switch language {
        case .chinese:
            templateSegment = .init(tokenIDs: fixedChineseTemplateTokenIDs)
            assistantPrefixSegment = .init(tokenIDs: fixedChineseAssistantPrefixTokenIDs)
        case .english:
            templateSegment = .init(text: englishTemplates[0])
            assistantPrefixSegment = .init(tokenIDs: fixedEnglishAssistantPrefixTokenIDs)
        case .auto:
            templateSegment = .init(tokenIDs: fixedChineseTemplateTokenIDs)
            assistantPrefixSegment = .init(
                tokenIDs: tokenizer.encode(
                    text: " thinking\n\n response\n<chinese>",
                    addSpecialTokens: false
                ).map(Int32.init)
            )
        }

        let segments: [MiMoInputSegment] = [
            .init(text: "<|im_start|>user\n"),
            .init(audioTokens: audioTokens),
            templateSegment,
            .init(text: "<|im_end|>\n"),
            .init(text: "<|im_start|>assistant\n"),
            assistantPrefixSegment,
        ]

        return buildInputIDs(
            segments: segments,
            tokenizer: tokenizer,
            specialTokens: specialTokens,
            groupSize: groupSize,
            audioChannels: audioChannels,
            speechEmptyIDs: speechEmptyIDs
        )
    }

    static func buildInputIDs(
        segments: [MiMoInputSegment],
        tokenizer: any Tokenizer,
        specialTokens: MiMoSpecialTokens,
        groupSize: Int,
        audioChannels: Int,
        speechEmptyIDs: [Int32]
    ) -> MLXArray {
        precondition(groupSize > 0)
        precondition(audioChannels > 0)
        precondition(speechEmptyIDs.count == audioChannels)

        var rows = Array(repeating: [Int32](), count: audioChannels + 1)

        for segment in segments {
            let chunk = segmentRows(
                for: segment,
                tokenizer: tokenizer,
                specialTokens: specialTokens,
                groupSize: groupSize,
                audioChannels: audioChannels,
                speechEmptyIDs: speechEmptyIDs
            )
            for index in chunk.indices {
                rows[index].append(contentsOf: chunk[index])
            }
        }

        let time = rows[0].count
        let flattened = rows.flatMap { $0 }
        return MLXArray(flattened).reshaped([audioChannels + 1, time])
    }

    static func flattenPrompt(_ inputIDs: MLXArray) -> MLXArray {
        precondition(inputIDs.ndim == 2)
        return inputIDs.transposed(1, 0).reshaped([1, -1])
    }

    private static func segmentRows(
        for segment: MiMoInputSegment,
        tokenizer: any Tokenizer,
        specialTokens: MiMoSpecialTokens,
        groupSize: Int,
        audioChannels: Int,
        speechEmptyIDs: [Int32]
    ) -> [[Int32]] {
        if let audioTokens = segment.audioTokens {
            return audioSegmentRows(
                audioTokens: audioTokens,
                specialTokens: specialTokens,
                groupSize: groupSize,
                audioChannels: audioChannels,
                speechEmptyIDs: speechEmptyIDs,
                addSospEosp: segment.addSospEosp
            )
        }

        let tokenizedText = segment.tokenIDs
            ?? tokenizer.encode(text: segment.text ?? "", addSpecialTokens: false).map(Int32.init)
        let textRow = insertBetween(tokenizedText, fillCount: groupSize - 1, fillValue: -100)

        var result = Array(repeating: [Int32](), count: audioChannels + 1)
        result[0] = textRow
        for channel in 0..<audioChannels {
            result[channel + 1] = Array(repeating: speechEmptyIDs[channel], count: textRow.count)
        }
        return result
    }

    private static func audioSegmentRows(
        audioTokens: [Int32],
        specialTokens: MiMoSpecialTokens,
        groupSize: Int,
        audioChannels: Int,
        speechEmptyIDs: [Int32],
        addSospEosp: Bool
    ) -> [[Int32]] {
        precondition(audioTokens.count.isMultiple(of: audioChannels), "Audio tokens must align with audio channels.")

        let sequenceLength = audioTokens.count / audioChannels
        precondition(sequenceLength.isMultiple(of: groupSize), "Audio token length must be divisible by group size.")

        let textLength = sequenceLength / groupSize
        var textTokens = Array(repeating: specialTokens.empty, count: textLength)
        if addSospEosp {
            textTokens.insert(specialTokens.sosp, at: 0)
            textTokens.append(specialTokens.eosp)
        }
        let textRow = insertBetween(textTokens, fillCount: groupSize - 1, fillValue: -100)

        var audioRows = Array(repeating: [Int32](), count: audioChannels)
        for timestep in 0..<sequenceLength {
            for channel in 0..<audioChannels {
                audioRows[channel].append(audioTokens[timestep * audioChannels + channel])
            }
        }

        if addSospEosp {
            for channel in 0..<audioChannels {
                let boundary = Array(repeating: speechEmptyIDs[channel], count: groupSize)
                audioRows[channel] = boundary + audioRows[channel] + boundary
            }
        }

        var result = Array(repeating: [Int32](), count: audioChannels + 1)
        result[0] = textRow
        for channel in 0..<audioChannels {
            result[channel + 1] = audioRows[channel]
        }
        return result
    }

    private static func singleTokenID(_ text: String, using tokenizer: any Tokenizer) throws -> Int32 {
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        guard ids.count == 1 else {
            throw NSError(
                domain: "MiMoPromptBuilder",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Expected a single token for \(text), got \(ids.count)."]
            )
        }
        return Int32(ids[0])
    }

    private static func insertBetween(_ tokens: [Int32], fillCount: Int, fillValue: Int32) -> [Int32] {
        guard fillCount > 0, !tokens.isEmpty else { return tokens }

        let expandedCount = tokens.count + tokens.count * fillCount
        var result = Array(repeating: fillValue, count: expandedCount)
        for (index, token) in tokens.enumerated() {
            result[index * (fillCount + 1)] = token
        }
        return result
    }
}
