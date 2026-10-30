import Foundation
import MLX
import MLXAudioCore
import Tokenizers

struct VibeVoiceTTSTurn: Equatable {
    let speaker: Int
    let text: String
}
struct VibeVoiceTTSPrompt {
    let tokens: [Int]
    let referenceRanges: [Range<Int>]
    static let system = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    static func parse(_ text: String) throws -> [VibeVoiceTTSTurn] {
        let lines = text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        guard !lines.isEmpty else { throw AudioGenerationError.invalidInput("Text is empty") }
        let regex = try NSRegularExpression(pattern: #"^Speaker\s+(\d+)\s*:\s*(.*)$"#, options: .caseInsensitive)
        var turns: [VibeVoiceTTSTurn] = []
        var current: Int?; var parts: [String] = []; var labeled = false
        func flush() {
            if let current, !parts.isEmpty { turns.append(.init(speaker: current, text: " " + parts.joined(separator: " "))) }
            parts = []
        }
        for line in lines {
            if let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
               let speakerRange = Range(match.range(at: 1), in: line), let textRange = Range(match.range(at: 2), in: line) {
                labeled = true; flush(); current = Int(line[speakerRange])!
                let value = line[textRange].trimmingCharacters(in: .whitespaces)
                parts = value.isEmpty ? [] : [value]
            } else { if labeled && current == nil { current = 0 }; parts.append(line) }
        }
        if !labeled { return [.init(speaker: 0, text: " " + parts.joined(separator: " "))] }
        flush()
        guard !turns.isEmpty else { throw AudioGenerationError.invalidInput("No valid speaker lines in script") }
        let speakers = Set(turns.map(\.speaker)).sorted()
        if speakers.first! > 0 {
            let map = Dictionary(uniqueKeysWithValues: speakers.enumerated().map { ($0.element, $0.offset) })
            turns = turns.map { .init(speaker: map[$0.speaker]!, text: $0.text) }
        }
        return turns
    }
    static func speakerOrder(_ turns: [VibeVoiceTTSTurn]) -> [Int] {
        var seen: Set<Int> = []; return turns.compactMap { seen.insert($0.speaker).inserted ? $0.speaker : nil }
    }
    init(turns: [VibeVoiceTTSTurn], referenceSteps: [Int]?, tokenizer: any Tokenizer) {
        var tokens = tokenizer.encode(text: Self.system, addSpecialTokens: true)
        var ranges: [Range<Int>] = []
        func append(_ text: String) { tokens += tokenizer.encode(text: text, addSpecialTokens: false) }
        if let referenceSteps {
            append(" Voice input:\n")
            for (speaker, steps) in zip(Self.speakerOrder(turns), referenceSteps) {
                append(" Speaker \(speaker):")
                tokens.append(VibeVoiceTTSModel.speechStartID)
                ranges.append(tokens.count..<(tokens.count + steps))
                tokens += Array(repeating: VibeVoiceTTSModel.speechDiffusionID, count: steps)
                tokens.append(VibeVoiceTTSModel.speechEndID); append("\n")
            }
        }
        append(" Text input:\n")
        for turn in turns { append(" Speaker \(turn.speaker):\(turn.text)\n") }
        append(" Speech output:\n"); tokens.append(VibeVoiceTTSModel.speechStartID)
        self.tokens = tokens; referenceRanges = ranges
    }
}
