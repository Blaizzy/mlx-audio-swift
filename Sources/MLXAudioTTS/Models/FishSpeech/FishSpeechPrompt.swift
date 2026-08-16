import Foundation
@preconcurrency import MLX

enum FishSpeechModality: String, Sendable {
    case text
    case voice
    case interleave
}

enum FishSpeechRole: String, Sendable {
    case system
    case user
    case assistant
}

struct FishSpeechTextPart: Sendable {
    let text: String
}

struct FishSpeechVQPart: Sendable {
    let codes: MLXArray

    init(_ codes: MLXArray) {
        self.codes = codes.asType(.int32)
    }
}

enum FishSpeechPart: Sendable {
    case text(FishSpeechTextPart)
    case vq(FishSpeechVQPart)
}

struct FishSpeechMessage: Sendable {
    let role: FishSpeechRole
    var parts: [FishSpeechPart] = []
    var addIMStart: Bool = true
    var addIMEnd: Bool = true
    var modality: FishSpeechModality? = nil
}

struct FishSpeechConversation: Sendable {
    var messages: [FishSpeechMessage] = []

    mutating func append(_ message: FishSpeechMessage) {
        messages.append(message)
    }

    func encodeForInference(
        tokenizer: some FishSpeechTokenizing,
        numCodebooks: Int
    ) -> MLXArray {
        typealias Segment = (tokens: [Int32], codes: [[Int32]]?)

        var segments: [Segment] = []
        segments.reserveCapacity(messages.count * 4)

        for message in messages {
            if message.addIMStart {
                let modalityToken = message.modality.flatMap { fishSpeechModalityTokens[$0] } ?? ""
                let text = "\(fishSpeechIMStartToken)\(message.role.rawValue)\n\(modalityToken)"
                segments.append((tokenizer.encode(text, addSpecialTokens: false).map(Int32.init), nil))
            }

            for part in message.parts {
                switch part {
                case .text(let textPart):
                    let tokenIDs = tokenizer.encode(textPart.text, addSpecialTokens: false).map(Int32.init)
                    segments.append((tokenIDs, nil))

                case .vq(let vqPart):
                    let rowCount = vqPart.codes.dim(0)
                    let time = vqPart.codes.dim(1)
                    var codes: [[Int32]] = []
                    codes.reserveCapacity(rowCount)
                    for row in 0 ..< rowCount {
                        codes.append(vqPart.codes[row, 0..<time].asArray(Int32.self))
                    }
                    let semanticTokens = codes.first?.map { $0 + Int32(tokenizer.semanticBeginID) } ?? []
                    segments.append((semanticTokens, codes))
                }
            }

            if message.addIMEnd {
                let tokenIDs = tokenizer.encode("\(fishSpeechIMEndToken)\n", addSpecialTokens: false).map(Int32.init)
                segments.append((tokenIDs, nil))
            }
        }

        guard !segments.isEmpty else {
            return MLXArray.zeros([numCodebooks + 1, 0], dtype: .int32)
        }

        let totalLength = segments.reduce(into: 0) { $0 += $1.tokens.count }
        var rows = Array(
            repeating: Array(repeating: Int32(0), count: totalLength),
            count: numCodebooks + 1
        )

        var cursor = 0
        for segment in segments {
            let end = cursor + segment.tokens.count
            rows[0].replaceSubrange(cursor..<end, with: segment.tokens)
            if let codes = segment.codes {
                let codebookCount = min(numCodebooks, codes.count)
                for row in 0 ..< codebookCount {
                    rows[row + 1].replaceSubrange(cursor..<end, with: codes[row])
                }
            }
            cursor = end
        }

        let flat = rows.flatMap { $0 }
        return MLXArray(flat).reshaped([numCodebooks + 1, totalLength])
    }
}

func fishSpeechSplitTextBySpeaker(_ text: String) -> [String] {
    let pattern = #"<\|speaker:\d+\|>"#
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }

    let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
    let matches = regex.matches(in: text, options: [], range: fullRange)
    guard !matches.isEmpty else { return [] }

    var turns: [String] = []
    turns.reserveCapacity(matches.count)

    for (index, match) in matches.enumerated() {
        let start = match.range.location
        let end = index + 1 < matches.count
            ? matches[index + 1].range.location
            : fullRange.length
        let range = NSRange(location: start, length: end - start)
        guard let stringRange = Range(range, in: text) else { continue }
        let turn = text[stringRange].trimmingCharacters(in: .whitespacesAndNewlines)
        if !turn.isEmpty {
            turns.append(turn)
        }
    }

    return turns
}

func fishSpeechSplitTextIntoBatches(_ text: String, maxBytes: Int) -> [String] {
    let limit = max(1, maxBytes)
    guard text.lengthOfBytes(using: .utf8) > limit else {
        return text.isEmpty ? [] : [text]
    }

    var batches: [String] = []
    var batchStart = text.startIndex
    var cursor = batchStart
    var batchBytes = 0
    var lastWhitespaceEnd: String.Index?

    while cursor < text.endIndex {
        let next = text.index(after: cursor)
        let character = text[cursor]
        let characterBytes = String(character).utf8.count
        if characterBytes > limit {
            if cursor > batchStart {
                batches.append(String(text[batchStart..<cursor]))
            }
            var fragment = ""
            var fragmentBytes = 0
            for scalar in String(character).unicodeScalars {
                let scalarText = String(scalar)
                let scalarBytes = scalarText.utf8.count
                if !fragment.isEmpty, fragmentBytes + scalarBytes > limit {
                    batches.append(fragment)
                    fragment = ""
                    fragmentBytes = 0
                }
                fragment.append(contentsOf: scalarText)
                fragmentBytes += scalarBytes
            }
            if !fragment.isEmpty {
                batches.append(fragment)
            }
            batchStart = next
            cursor = next
            batchBytes = 0
            lastWhitespaceEnd = nil
            continue
        }
        if batchBytes + characterBytes > limit, cursor > batchStart {
            let whitespaceSplit = lastWhitespaceEnd.flatMap { index -> String.Index? in
                String(text[batchStart..<index]).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? nil
                    : index
            }
            let split = whitespaceSplit ?? cursor
            batches.append(String(text[batchStart..<split]))
            batchStart = split
            cursor = split
            batchBytes = 0
            lastWhitespaceEnd = nil
            continue
        }

        batchBytes += characterBytes
        if character.isWhitespace {
            lastWhitespaceEnd = next
        }
        cursor = next
    }

    if batchStart < text.endIndex {
        batches.append(String(text[batchStart...]))
    }
    return batches
}

func fishSpeechSplitSpeakerTurn(_ turn: String, maxBytes: Int) -> [String] {
    guard turn.hasPrefix("<|speaker:"),
          let markerEnd = turn.range(of: "|>")?.upperBound
    else {
        return fishSpeechSplitTextIntoBatches(turn, maxBytes: maxBytes)
    }

    let marker = String(turn[..<markerEnd])
    let payload = String(turn[markerEnd...])
    let payloadLimit = max(1, maxBytes - marker.lengthOfBytes(using: .utf8))
    return fishSpeechSplitTextIntoBatches(payload, maxBytes: payloadLimit).map {
        marker + $0
    }
}

func fishSpeechGenerationBatches(_ text: String, maxBytes: Int) -> [String] {
    let turns = fishSpeechSplitTextBySpeaker(text)
    let batches = turns.isEmpty
        ? fishSpeechSplitTextIntoBatches(text, maxBytes: maxBytes)
        : fishSpeechGroupTurnsIntoBatches(
            turns.flatMap { fishSpeechSplitSpeakerTurn($0, maxBytes: maxBytes) },
            maxSpeakers: 5,
            maxBytes: maxBytes
        )
    return batches.filter {
        !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

func fishSpeechGroupTurnsIntoBatches(
    _ turns: [String],
    maxSpeakers: Int = 5,
    maxBytes: Int = 200
) -> [String] {
    guard !turns.isEmpty else { return [] }

    var batches: [String] = []
    var currentBatch: [String] = []
    var currentBytes = 0

    for turn in turns {
        let turnBytes = turn.lengthOfBytes(using: .utf8)
        let separatorBytes = currentBatch.isEmpty ? 0 : 1
        let exceedsSpeakers = currentBatch.count >= maxSpeakers
        let exceedsBytes = !currentBatch.isEmpty
            && (currentBytes + separatorBytes + turnBytes > maxBytes)

        if exceedsSpeakers || exceedsBytes {
            batches.append(currentBatch.joined(separator: "\n"))
            currentBatch = [turn]
            currentBytes = turnBytes
        } else {
            currentBatch.append(turn)
            currentBytes += separatorBytes + turnBytes
        }
    }

    if !currentBatch.isEmpty {
        batches.append(currentBatch.joined(separator: "\n"))
    }

    return batches
}
