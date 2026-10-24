import Foundation
import Tokenizers

struct MossMusicPrompt {
    let config: MossMusicConfig
    let tokenizer: Tokenizer
    static let audioSpan = try! NSRegularExpression(
        pattern: #"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"#)

    func audioTokens(count: Int, timeMarkers: Bool) -> [Int] {
        guard timeMarkers else { return Array(repeating: config.audioTokenId, count: count) }
        var ids: [Int] = []
        var consumed = 0
        let seconds = Int(Double(count) / 12.5)
        for second in stride(from: 2, through: seconds, by: 2) {
            let position = second / 2 * 25
            ids += Array(repeating: config.audioTokenId, count: position - consumed)
            consumed = position
            for digit in String(second) {
                ids.append(tokenizer.encode(text: String(digit), addSpecialTokens: false)[0])
            }
        }
        ids += Array(repeating: config.audioTokenId, count: count - consumed)
        return ids
    }

    func inputIds(
        audioTokens count: Int, prompt: String?, timeMarkers: Bool, enableThinking: Bool? = nil
    ) -> [Int] {
        var text = (prompt?.isEmpty == false ? prompt : nil) ?? config.defaultPrompt
        if Self.audioSpan.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) == nil
        {
            text =
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                + text + "<|im_end|>\n<|im_start|>assistant\n"
        }
        if enableThinking == false && text.hasSuffix("<|im_start|>assistant\n") {
            text += "<think>\n\n</think>\n\n"
        }
        let spans = Self.audioSpan.matches(in: text, range: NSRange(text.startIndex..., in: text))
        precondition(spans.count == 1, "MOSS-Music expects one audio span for the recording")
        let range = Range(spans[0].range, in: text)!
        var ids = tokenizer.encode(text: String(text[..<range.lowerBound]), addSpecialTokens: false)
        ids.append(config.audioStartId)
        ids += audioTokens(count: count, timeMarkers: timeMarkers)
        ids.append(config.audioEndId)
        ids += tokenizer.encode(text: String(text[range.upperBound...]), addSpecialTokens: false)
        return ids
    }
}

enum MossMusicText {
    static func response(_ text: String, stripThinking: Bool, final: Bool = true) -> String {
        guard stripThinking else { return text }
        var visible = text.replacingOccurrences(
            of: #"(?s)<think>.*?</think>"#, with: "", options: .regularExpression)
        if let open = visible.range(of: "<think>") { visible = String(visible[..<open.lowerBound]) }
        if !final, let tag = visible.lastIndex(of: "<") {
            let suffix = String(visible[tag...])
            if "<think>".hasPrefix(suffix) || "</think>".hasPrefix(suffix) {
                visible = String(visible[..<tag])
            }
        }
        return visible.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private struct Marker {
        let start: Double
        let end: Double?
        let range: NSRange
        let raw: String
    }
    private static let timeValue =
        #"(?:\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*s|\d+(?:[.,]\d+)?)"#
    private static let lineValue = #"(?:\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*s)"#
    private static let separator = #"(?:-|–|—|~|to|→)"#
    private static let bracketed = try! NSRegularExpression(
        pattern:
            #"[\[(<【]\s*("# + timeValue + #")\s*(?:"# + separator + #"\s*("# + timeValue
            + #"))?\s*[\])>】]"#, options: .caseInsensitive)
    private static let lined = try! NSRegularExpression(
        pattern:
            #"(?:^|\n)\s*("# + lineValue + #")\s*(?:"# + separator + #"\s*("# + lineValue
            + #"))?\s*[:：\-–—]\s*"#,
        options: [.caseInsensitive, .anchorsMatchLines])

    static func seconds(_ value: String) -> Double? {
        var text = value.lowercased().replacingOccurrences(of: ",", with: ".").trimmingCharacters(
            in: .whitespaces)
        if text.hasSuffix("s") {
            text.removeLast()
            text = text.trimmingCharacters(in: .whitespaces)
        }
        let parts = text.split(separator: ":").compactMap { Double($0) }
        guard !parts.isEmpty, parts.count <= 3, parts.count == text.split(separator: ":").count
        else { return nil }
        return parts.reduce(0) { $0 * 60 + $1 }
    }

    static func segments(_ text: String, duration: Double) -> [[String: Any]] {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        var markers: [Marker] = []
        for regex in [bracketed, lined] {
            for match in regex.matches(in: text, range: range) {
                guard let start = seconds(ns.substring(with: match.range(at: 1))) else { continue }
                let endRange = match.range(at: 2)
                let end =
                    endRange.location == NSNotFound ? nil : seconds(ns.substring(with: endRange))
                markers.append(
                    Marker(
                        start: start, end: end, range: match.range,
                        raw: ns.substring(with: match.range).trimmingCharacters(
                            in: .whitespacesAndNewlines)))
            }
        }
        markers.sort {
            $0.range.location == $1.range.location
                ? $0.range.length < $1.range.length : $0.range.location < $1.range.location
        }
        var unique: [Marker] = []
        var lastEnd = -1
        for marker in markers where marker.range.location >= lastEnd {
            unique.append(marker)
            lastEnd = NSMaxRange(marker.range)
        }
        var result: [[String: Any]] = []
        func rounded(_ value: Double) -> Double { (value * 1000).rounded() / 1000 }
        for (i, marker) in unique.enumerated() {
            let next = i + 1 < unique.count ? unique[i + 1] : nil
            let start = NSMaxRange(marker.range)
            let end = next?.range.location ?? ns.length
            let body = ns.substring(with: NSRange(location: start, length: end - start))
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(
                    of: #"^[\s:：,;|\-–—>]+|[\s|]+$"#, with: "", options: .regularExpression
                )
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if body.isEmpty && marker.end == nil { continue }
            result.append([
                "text": body, "start": rounded(marker.start),
                "end": rounded(max(marker.start, marker.end ?? next?.start ?? duration)),
                "kind": "timestamped_text", "marker": marker.raw,
            ])
        }
        if result.isEmpty {
            return [
                [
                    "text": text, "start": 0.0, "end": rounded(duration), "kind": "text",
                    "marker": NSNull(),
                ]
            ]
        }
        return result
    }
}
