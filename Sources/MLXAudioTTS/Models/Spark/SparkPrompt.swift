//
//  SparkPrompt.swift
//  MLXAudio
//
//  Prompt construction and generated-token parsing for Spark-TTS, mirroring
//  spark.py (process_prompt_control) and utils/token_parser.py.
//

import Foundation

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
    /// Controllable-TTS prompt: gender/pitch/speed style labels, no reference audio.
    static func control(
        gender: SparkGender,
        pitch: SparkLevel,
        speed: SparkLevel,
        text: String
    ) -> String {
        let attribute = "<|gender_\(gender.id)|><|pitch_label_\(pitch.id)|><|speed_label_\(speed.id)|>"
        return [
            "<|task_controllable_tts|>",
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
