//
//  SparkPrompt.swift
//  MLXAudio
//
//  Prompt construction and generated-token parsing for Spark-TTS, mirroring
//  spark.py (process_prompt / process_prompt_control) and utils/token_parser.py.
//

import Foundation

public enum SparkTaskToken {
    public static let tts = "<|task_tts|>"
    public static let controllableTTS = "<|task_controllable_tts|>"
}

public enum SparkLevel: String, CaseIterable, Sendable {
    case veryLow = "very_low"
    case low
    case moderate
    case high
    case veryHigh = "very_high"

    public var id: Int {
        switch self {
        case .veryLow: return 0
        case .low: return 1
        case .moderate: return 2
        case .high: return 3
        case .veryHigh: return 4
        }
    }
}

public enum SparkGender: String, Sendable {
    case female
    case male

    public var id: Int { self == .female ? 0 : 1 }
}

enum SparkPrompt {
    /// Voice-cloning prompt: reference global (+ optional semantic) tokens followed
    /// by the target text. `globalTokenIds`/`semanticTokenIds` come from BiCodec.
    static func clone(
        text: String,
        refText: String?,
        globalTokenIds: [Int],
        semanticTokenIds: [Int]?
    ) -> String {
        let globalTokens = globalTokenIds.map { "<|bicodec_global_\($0)|>" }.joined()

        var parts: [String] = [SparkTaskToken.tts, "<|start_content|>"]
        if let refText, let semanticTokenIds {
            let semanticTokens = semanticTokenIds.map { "<|bicodec_semantic_\($0)|>" }.joined()
            parts += [
                refText, text, "<|end_content|>",
                "<|start_global_token|>", globalTokens, "<|end_global_token|>",
                "<|start_semantic_token|>", semanticTokens,
            ]
        } else {
            parts += [
                text, "<|end_content|>",
                "<|start_global_token|>", globalTokens, "<|end_global_token|>",
            ]
        }
        return parts.joined()
    }

    /// Controllable-TTS prompt: gender/pitch/speed style labels, no reference audio.
    static func control(
        gender: SparkGender,
        pitch: SparkLevel,
        speed: SparkLevel,
        text: String
    ) -> String {
        let attribute = "<|gender_\(gender.id)|><|pitch_label_\(pitch.id)|><|speed_label_\(speed.id)|>"
        return [
            SparkTaskToken.controllableTTS,
            "<|start_content|>", text, "<|end_content|>",
            "<|start_style_label|>", attribute, "<|end_style_label|>",
        ].joined()
    }

    /// Extract the integer ids from `<|bicodec_<kind>_N|>` markers in decoded text.
    static func extractTokenIds(_ text: String, kind: String) -> [Int] {
        guard let re = try? NSRegularExpression(pattern: "bicodec_\(kind)_(\\d+)") else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return re.matches(in: text, range: range).compactMap { m in
            guard let r = Range(m.range(at: 1), in: text) else { return nil }
            return Int(text[r])
        }
    }
}
