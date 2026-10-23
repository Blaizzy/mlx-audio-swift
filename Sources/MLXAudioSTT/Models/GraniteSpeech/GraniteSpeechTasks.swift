import Foundation
import Tokenizers

/// Granite Speech Plus also supports speaker-attributed and word-timestamp transcripts.
public enum GraniteSpeechTask: String, Sendable {
    case transcription = "asr"
    case speakerAttribution = "saa"
    case timestamps

    var instruction: String {
        switch self {
        case .transcription:
            return "can you transcribe the speech into a written format?"
        case .speakerAttribution:
            return
                "Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
        case .timestamps:
            return
                "Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
        }
    }
}

extension GraniteSpeechModel {
    public var isPlus: Bool {
        config.modelType == "granite_speech_plus" || !config.encoderConfig.catHiddenLayers.isEmpty
    }

    static let plusSystemPrompt = """
        Knowledge Cutoff Date: April 2024.
        Today's Date: December 19, 2024.
        You are Granite, developed by IBM. You are a helpful AI assistant
        """

    func resolvePrompt(
        prompt: String?, language: String?, task: GraniteSpeechTask,
        sourceLanguage: String?, hotwords: [String], punctuate: Bool
    ) -> String {
        if task != .transcription {
            precondition(isPlus, "Speaker attribution and timestamps require Granite Speech Plus")
            precondition(
                prompt == nil && language == nil && sourceLanguage == nil && hotwords.isEmpty
                    && !punctuate,
                "Structured transcription uses the model's task prompt")
            return task.instruction
        }
        precondition(
            sourceLanguage == nil || language == nil,
            "Choose a source language or a translation language")
        let names = [
            "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
            "ja": "Japanese", "it": "Italian", "zh": "Chinese",
        ]
        let keywords = hotwords.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter {
            !$0.isEmpty
        }
        var instruction: String
        if let prompt {
            instruction = prompt
        } else if let language {
            let name = names[language.lowercased()] ?? language
            instruction =
                "Translate the speech to \(name)\(punctuate ? " with proper punctuation and capitalization" : "")."
        } else if punctuate {
            instruction = "transcribe the speech with proper punctuation and capitalization."
        } else if !keywords.isEmpty {
            instruction = "transcribe the speech to text."
        } else {
            instruction = task.instruction
        }
        if let sourceLanguage {
            let name = names[sourceLanguage.lowercased()] ?? sourceLanguage
            instruction =
                "The speech is in \(name). Transcribe it exactly; do not translate. \(instruction)"
        }
        if !keywords.isEmpty { instruction += " Keywords: " + keywords.joined(separator: ", ") }
        return instruction
    }

    func promptMessages(numAudioTokens: Int, instruction: String, systemPrompt: String?)
        -> [Tokenizers.Message]
    {
        let audio = String(repeating: "<|audio|>", count: numAudioTokens)
        let content =
            isPlus
            ? audio + " " + instruction.trimmingCharacters(in: .whitespaces) : audio + instruction
        var messages: [Tokenizers.Message] = []
        if let system = systemPrompt ?? (isPlus ? Self.plusSystemPrompt : nil), !system.isEmpty {
            messages.append(["role": "system", "content": system])
        }
        messages.append(["role": "user", "content": content])
        return messages
    }

    /// Returns structured fields only when every model-generated tag is well formed.
    /// The original tagged transcript is always retained in `STTOutput.text`.
    static func parseSegments(_ text: String, task: GraniteSpeechTask) -> [[String: Any]]? {
        guard task != .transcription else { return nil }
        let pattern = task == .speakerAttribution ? #"\[Speaker (\d+)\]:"# : #"\[T:(\d+)\]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return nil }
        func trimmed(_ range: NSRange) -> String {
            ns.substring(with: range).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if task == .speakerAttribution {
            guard trimmed(NSRange(location: 0, length: matches[0].range.location)).isEmpty else {
                return nil
            }
            var seen: Set<Int> = []
            var result: [[String: Any]] = []
            for (i, match) in matches.enumerated() {
                guard let speaker = Int(ns.substring(with: match.range(at: 1))) else { return nil }
                if !seen.contains(speaker) {
                    guard speaker == seen.count + 1 else { return nil }
                    seen.insert(speaker)
                }
                let start = NSMaxRange(match.range)
                let end = i + 1 < matches.count ? matches[i + 1].range.location : ns.length
                let body = trimmed(NSRange(location: start, length: end - start))
                guard !body.isEmpty else { return nil }
                result.append(["speaker_id": speaker, "text": body])
            }
            return result
        }
        var cursor = 0
        var endOfTag = 0
        var words: [[String: Any]] = []
        for match in matches {
            let word = trimmed(NSRange(location: endOfTag, length: match.range.location - endOfTag))
            guard word.split(whereSeparator: { $0.isWhitespace }).count == 1,
                let tag = Int(ns.substring(with: match.range(at: 1)))
            else { return nil }
            var end = tag
            if tag < 1000 {
                let (absolute, overflow) = ((cursor / 1000) * 1000).addingReportingOverflow(end)
                guard !overflow else { return nil }
                end = absolute
                if end < cursor {
                    let (wrapped, overflow) = end.addingReportingOverflow(1000)
                    guard !overflow else { return nil }
                    end = wrapped
                }
            }
            guard end >= cursor else { return nil }
            if word != "_" {
                words.append([
                    "word": word, "start": Double(cursor) / 100, "end": Double(end) / 100,
                ])
            }
            cursor = end
            endOfTag = NSMaxRange(match.range)
        }
        guard trimmed(NSRange(location: endOfTag, length: ns.length - endOfTag)).isEmpty else {
            return nil
        }
        guard let first = words.first, let last = words.last else { return [] }
        return [
            [
                "text": words.compactMap { $0["word"] as? String }.joined(separator: " "),
                "start": first["start"]!, "end": last["end"]!, "words": words,
            ]
        ]
    }
}
