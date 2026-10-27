import Foundation
import MLXAudioCore

/// The speech request framing shipped with Voxtral's Tekken v7 assets.
final class VoxtralTTSTextProcessor {
    let tokenizer: TekkenTokenizer
    let voiceLengths: [String: Int]
    let special: [String: Int]
    init(directory: URL) throws {
        let url = directory.appendingPathComponent("tekken.json")
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let entries = json?["special_tokens"] as? [[String: Any]] ?? []
        special = Dictionary(entries.compactMap { item in
            guard let name = item["token_str"] as? String, let id = item["rank"] as? Int else { return nil }
            return (name,id)
        }, uniquingKeysWith: { first, _ in first })
        voiceLengths = (json?["audio"] as? [String: Any])?["voice_num_audio_tokens"] as? [String: Int] ?? [:]
        tokenizer = try TekkenTokenizer(tekkenURL: url)
        guard ["<s>","[BEGIN_AUDIO]","[AUDIO]","[NEXT_AUDIO_TEXT]","[REPEAT_AUDIO_TEXT]"].allSatisfy({ special[$0] != nil }),
              !voiceLengths.isEmpty, voiceLengths.values.allSatisfy({ $0 > 0 }) else {
            throw AudioGenerationError.invalidInput("Voxtral tekken.json is missing speech metadata")
        }
    }
    func encode(_ text: String, voice: String, sanitize: Bool) throws -> [Int] {
        guard let count = voiceLengths[voice] else { throw AudioGenerationError.invalidInput("Unknown Voxtral voice: \(voice)") }
        let input = sanitize ? try VoxtralTTSText.sanitize(text) : text
        guard !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AudioGenerationError.invalidInput("Voxtral speech input is empty") }
        return [special["<s>"]!, special["[BEGIN_AUDIO]"]!] + Array(repeating: special["[AUDIO]"]!, count: count)
            + [special["[NEXT_AUDIO_TEXT]"]!] + tokenizer.encode(input)
            + [special["[REPEAT_AUDIO_TEXT]"]!, special["[BEGIN_AUDIO]"]!]
    }
}

// Text normalization follows mlx-audio's Voxtral demo frontend.
enum VoxtralTTSText {
    static func replace(_ text: String, _ pattern: String, _ replacement: String) -> String {
        text.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
    }
    static func transform(_ text: String, _ pattern: String, _ body: (NSTextCheckingResult, NSString) -> String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let source = text as NSString
        var result = text
        for match in regex.matches(in: text, range: NSRange(location: 0, length: source.length)).reversed() {
            guard let range = Range(match.range, in: result) else { continue }
            result.replaceSubrange(range, with: body(match,source))
        }
        return result
    }
    static let ones = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    static let teens = ["ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
    static let tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
    static func subThousand(_ n: Int, and: Bool = false) -> String {
        var parts: [String] = []
        if n >= 100 { parts.append(ones[n / 100] + " hundred"); if and && n % 100 > 0 { parts.append("and") } }
        let r = n % 100
        if r >= 20 { parts.append(tens[r / 10] + (r % 10 > 0 ? "-" + ones[r % 10] : "")) }
        else if r >= 10 { parts.append(teens[r - 10]) }
        else if r > 0 { parts.append(ones[r]) }
        return parts.isEmpty ? "zero" : parts.joined(separator: " ")
    }
    static func integer(_ value: Int64, and: Bool = false) -> String {
        if value == 0 { return "zero" }
        let scales = ["","thousand","million","billion","trillion","quadrillion"]
        var n = value, groups: [Int] = []
        while n > 0 { groups.append(Int(n % 1000)); n /= 1000 }
        return groups.indices.reversed().filter { groups[$0] > 0 }.map { i in
            subThousand(groups[i], and: and && i == 0 && groups.count > 1) + (scales[i].isEmpty ? "" : " " + scales[i])
        }.joined(separator: " ")
    }
    static func asciiDigits(_ text: String) -> String {
        text.map { character in
            if let value = character.wholeNumberValue, value < 10 { return String(value) }
            return String(character)
        }.joined()
    }
    static func number(_ source: String, and: Bool = false, years: Bool = true) -> String? {
        let clean = asciiDigits(source).replacingOccurrences(of: ",", with: "")
        if clean.hasPrefix("-") { return number(String(clean.dropFirst()), and: and, years: years).map { "negative " + $0 } }
        let parts = clean.split(separator: ".", omittingEmptySubsequences: false)
        guard let whole = Int64(parts[0]), whole >= 0, whole < 1_000_000_000_000_000_000 else { return nil }
        if parts.count == 2 {
            guard parts[1].allSatisfy({ $0.isASCII && $0.isNumber }) else { return nil }
            return integer(whole, and: and) + " point " + parts[1].map { ones[$0.wholeNumberValue!] }.joined(separator: "-")
        }
        if years && source.count == 4 && whole >= 1400 && whole < 2100 {
            if whole == 2000 { return "two thousand" }
            if whole <= 1999 { return integer(whole / 100) + (whole % 100 == 0 ? " hundred" : " " + subThousand(Int(whole % 100))) }
            if whole < 2010 { return "two thousand " + ones[Int(whole % 100)] }
            return "twenty " + subThousand(Int(whole % 100))
        }
        return integer(whole, and: and)
    }
    static let currencies = ["$": ["dollar","dollars","cent","cents"], "£": ["pound","pounds","penny","pence"],
                             "€": ["euro","euros","cent","cents"], "¥": ["yen","yen"],
                             "₹": ["rupee","rupees","paise","paise"], "¢": ["cent","cents"]]
    static func currency(_ source: String, short: Bool) -> String? {
        guard let first = source.first, let units = currencies[String(first)] else { return nil }
        let parts = asciiDigits(String(source.dropFirst())).replacingOccurrences(of: ",", with: "").split(separator: ".", omittingEmptySubsequences: false)
        guard let whole = Int64(parts[0]), whole < 1_000_000_000_000_000_000 else { return nil }
        let unit = units[whole == 1 ? 0 : 1]
        let fraction = parts.count > 1 ? String(parts[1]) : nil
        if let fraction, fraction.count > 2 || (units.count == 2 && fraction.contains(where: { $0 != "0" })) {
            return integer(whole) + " point " + fraction.map { ones[$0.wholeNumberValue!] }.joined(separator: "-") + " " + unit
        }
        let minor = fraction.flatMap { Int(String(($0 + "00").prefix(2))) } ?? 0
        let short = short && first == "$"
        if units.count == 2 || minor == 0 { return integer(whole, and: !short) + " " + unit }
        let minorWords = integer(Int64(minor)), minorUnit = units[minor == 1 ? 2 : 3]
        if whole == 0 { return minorWords + " " + minorUnit }
        if short { return integer(whole) + (minor <= 9 ? " oh " + ones[minor] : " " + minorWords) }
        return integer(whole, and: true) + " " + unit + " and " + minorWords + " " + minorUnit
    }
    static func numbers(_ text: String) -> String {
        let pattern = #"[$£€¥₹¢]\d[\d,]*(?:\.\d+)?|(?<!\w)-?\d[\d,]*(?:\.\d+)?\b"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        let currencyMatches = matches.filter { currencies[String(ns.substring(with: $0.range).prefix(1))] != nil }
        let repeatedDollars = currencyMatches.count > 1 && currencyMatches.allSatisfy { ns.substring(with: $0.range).hasPrefix("$") }
        let firstDollar = currencyMatches.first?.range.location
        return transform(text, pattern) { match, string in
            let token = string.substring(with: match.range)
            if currencies[String(token.prefix(1))] != nil { return currency(token, short: repeatedDollars && match.range.location != firstDollar) ?? token }
            let intPart = token.split(separator: ".")[0]
            if intPart.contains(","), let value = Int64(asciiDigits(String(intPart)).replacingOccurrences(of: ",", with: "").replacingOccurrences(of: "-", with: "")), value >= 1_000_000 {
                return number(token, and: true) ?? token
            }
            return token
        }
    }
    static func sanitize(_ source: String) throws -> String {
        var text = replace(source, #"```[\s\S]*?```|~~~[\s\S]*?~~~"#, " Code example omitted. ")
        text = htmlUnescape(text)
        text = replace(text, #"(?i)<br\s*/?>"#, " ")
        text = transform(text, #"!\[([^\]]*)\]\(([^)]*)\)"#) { m,s in
            let label = s.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            return label.isEmpty ? "image" : label
        }
        for (pattern,replacement) in [
            (#"\[([^\]]+)\]\(([^)]*)\)"#, "$1"), (#"<https?://[^>]+>"#, "link"), (#"https?://\S+"#, "link"),
            (#"`([^`]+)`"#, "$1"), (#"(?m)^\s{0,3}#{1,6}\s*"#, ""), (#"(?m)^\s*[-*+]\s+"#, ""),
            (#"(?m)^\s*\d{1,4}[.)]\s+"#, ""), (#"(?m)^\s{0,3}>\s?"#, ""), (#"</?[a-zA-Z][^>]*>"#, " "),
            ("[\u{200b}\u{200e}-\u{200f}\u{2028}-\u{2029}\u{2060}-\u{2064}\u{00ad}\u{180e}\u{feff}\u{fff9}-\u{fffb}]", ""),
            (#"(?:\r\n|\r|\n)+"#, " ")
        ] { text = replace(text, pattern, replacement) }
        text = numbers(text)
        text = transform(text, #"\s*\(([^()]+)\)\s*"#) { m,s in
            var inner = s.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if inner.range(of: #"^\s*[$€£¥₹¢]?\s*[+\-−–]?\d+(?:[.,]\d+)?\s*%?\s*$"#, options: .regularExpression) != nil { return s.substring(with: m.range) }
            for (pattern,replacement) in [(#"(?i)\bN\s*/\s*A\b"#, "not available"), (#"\s*(?:\+/-|±)\s*"#, " plus or minus "),
                (#"\s*>=\s*"#, " greater than or equal to "), (#"\s*<=\s*"#, " less than or equal to "),
                (#"\s*>\s*"#, " greater than "), (#"\s*<\s*"#, " less than "), (#"\s*~\s*"#, " about "),
                (#"#\s*(\d+)\b"#, "number $1"), (#"(?i)(\d+(?:\.\d+)?)\s*ms\b"#, "$1 milliseconds"),
                (#"(?i)(\d+(?:\.\d+)?)\s*h\b"#, "$1 hours"), (#"(\d+(?:\.\d+)?)\s*%"#, "$1 percent"),
                (#"\s*\+\s*"#, " plus "), (#"\s*=\s*"#, " equals "), (#"\s+"#, " ")]
            { inner = replace(inner, pattern, replacement) }
            return "—" + inner.trimmingCharacters(in: .whitespacesAndNewlines) + "—"
        }
        for (pattern,replacement) in [("[\u{2010}\u{2011}]", "-"), (#"\.{3,}"#, "..."), (#"([!?])\1+"#, "$1"), (#"[\-–−]{2,}"#, "—")] { text = replace(text, pattern, replacement) }
        text = transform(text, #"(?<=\s)[\-–−](?=\s)"#) { m,s in
            let tail = s.substring(from: NSMaxRange(m.range)).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let first = tail.first else { return s.substring(with: m.range) }
            return first.isNumber || "$€£¥₹¢".contains(first) ? s.substring(with: m.range) : "—"
        }
        text = replace(text, #"\s+"#, " ").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw AudioGenerationError.invalidInput("Voxtral speech input is empty after sanitization") }
        if let last = text.last, !".!?…؟。！？".contains(last) { text += "." }
        return text
    }
    static func htmlUnescape(_ text: String) -> String {
        transform(text, #"&(#(?:[xX][0-9a-fA-F]+|[0-9]+);?|[a-zA-Z][a-zA-Z0-9]{0,31};?)"#) { m,s in
            let token = s.substring(with: m.range(at: 1))
            if token.hasPrefix("#") {
                let raw = token.dropFirst().trimmingCharacters(in: CharacterSet(charactersIn: ";"))
                let hex = raw.hasPrefix("x") || raw.hasPrefix("X")
                let value = UInt32(hex ? String(raw.dropFirst()) : raw, radix: hex ? 16 : 10) ?? 0xfffd
                if value == 0 || value > 0x10ffff || (0xd800...0xdfff).contains(value) { return "�" }
                let windows: [UInt32: UInt32] = [0x80:0x20ac,0x82:0x201a,0x83:0x192,0x84:0x201e,0x85:0x2026,0x86:0x2020,0x87:0x2021,0x88:0x2c6,0x89:0x2030,0x8a:0x160,0x8b:0x2039,0x8c:0x152,0x8e:0x17d,0x91:0x2018,0x92:0x2019,0x93:0x201c,0x94:0x201d,0x95:0x2022,0x96:0x2013,0x97:0x2014,0x98:0x2dc,0x99:0x2122,0x9a:0x161,0x9b:0x203a,0x9c:0x153,0x9e:0x17e,0x9f:0x178,13:13]
                if let mapped = windows[value] { return String(UnicodeScalar(mapped)!) }
                if (1...8).contains(value) || value == 11 || (14...31).contains(value)
                    || (0x7f...0x9f).contains(value) && ![0x81,0x8d,0x8f,0x90,0x9d].contains(value)
                    || (0xfdd0...0xfdef).contains(value) || value & 0xffff == 0xfffe || value & 0xffff == 0xffff { return "" }
                return UnicodeScalar(value).map(String.init) ?? "�"
            }
            // Include legacy names without a semicolon, as required by HTML5.
            for end in stride(from: token.count, through: 1, by: -1) {
                let prefix = String(token.prefix(end))
                if let value = VoxtralHTML.entities[prefix] { return value + token.dropFirst(end) }
            }
            return s.substring(with: m.range)
        }
    }
}
