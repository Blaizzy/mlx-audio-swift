import Foundation

public enum MingText {
    public static func semanticLength(_ text: String) -> Int {
        matches("[一-龥]|[a-zA-Z]+", text).count
    }
    private static func matches(_ pattern: String, _ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }.filter { !$0.isEmpty }
    }
    public static func addPeriod(_ text: String) -> String {
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = value.last else { return value }
        let period = matches("[一-鿿]", value).isEmpty ? "." : "。"
        if ",，、;；".contains(last) { value.removeLast(); return value + period }
        if last.isLetter || last.isNumber { value += period }
        return value
    }
    public static func fragments(_ text: String, maxLength: Int = 24) -> [String] {
        precondition(maxLength > 0)
        let text = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard semanticLength(text) > maxLength else { return [text] }
        func pack(_ pieces: [String]) -> [String] {
            var out: [String] = [], current = ""
            for piece in pieces {
                if !current.isEmpty && semanticLength(current + piece) > maxLength {
                    out.append(current); current = piece
                } else { current += piece }
            }
            if !current.isEmpty { out.append(current) }
            return out
        }
        var pieces: [String] = []
        for sentence in matches(".*?(?:[。！？]|[.!?](?=\\s)|$)\\s*", text) {
            if semanticLength(sentence) <= maxLength { pieces.append(sentence); continue }
            for clause in matches(".*?(?:[，；、]|[,;](?=\\s)|$)\\s*", sentence) {
                if semanticLength(clause) > maxLength {
                    pieces += pack(matches("\\[[^\\]]*\\]|[一-龥]|[^\\s一-龥]+|\\s+", clause))
                } else { pieces.append(clause) }
            }
        }
        return pack(pieces).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }
    public static func isVoice(_ voice: String) -> Bool {
        voice.range(of: "^(?:(?:US|UK)_[A-Z][a-z]+|[一-鿿][一-鿿_]*)$", options: .regularExpression) != nil
    }
    public static func isStyle(_ instruction: String?) -> Bool {
        guard let instruction else { return false }
        guard let data = instruction.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sequence = root["audio_sequence"] as? [[String: Any]], let item = sequence.first else { return true }
        let style = item["风格"] ?? item["style"]
        return style != nil && !(style is NSNull)
    }
    public static func instruction(dialect: String? = nil, style: String? = nil, speed: String? = nil,
                                   pitch: String? = nil, volume: String? = nil, emotion: String? = nil,
                                   voice: String? = nil, englishSchema: Bool? = nil) -> String {
        let english = englishSchema ?? (dialect != nil)
        let keys = english ? ["id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion", "ip"]
            : ["序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感", "IP"]
        func quoted(_ value: String?) -> String {
            guard let value else { return "null" }
            return String(data: try! JSONEncoder().encode(value), encoding: .utf8)!
        }
        let values = ["1", quoted("speaker_1"), quoted(dialect), quoted(style), quoted(speed), quoted(pitch), quoted(volume), quoted(emotion)]
        var fields = zip(keys.prefix(8), values).map { "\(quoted($0.0)): \($0.1)" }
        fields.append("\"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, \"ENV\": null, \"SNR\": null}")
        fields.append("\(quoted(keys[8])): \(quoted(voice))")
        return "{\"audio_sequence\": [{" + fields.joined(separator: ", ") + "}]}"
    }
    static func targetLanguage(_ language: String?) -> String? {
        let names = ["ja": "Japanese", "japanese": "Japanese", "ko": "Korean", "korean": "Korean",
            "de": "German", "german": "German", "fr": "French", "french": "French", "es": "Spanish", "spanish": "Spanish"]
        return names[language?.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) ?? ""]
    }
}
