import Foundation
import MLXAudioCore

/// Written-to-spoken normalization for Maya-2's twelve supported languages.
public enum Maya2TextNormalizer {
    private struct Tables: Decodable, Sendable {
        let uri_acronyms: [String]
        let month_aliases: [String: Int]
        let id_words: [String]
        let label_words: [String]
        let languages: [String: LanguageData]
    }
    private struct LanguageData: Decodable, Sendable {
        let ones: [String], hundreds: [String], hundreds_comb: [String], year_prefix: [String]
        let scales: [[String]], months: [String]
        let letters: [String: String], symbols: [String: String], currency_words: [String: String]
        let month_aliases: [String: Int], scale_words: [String: String], singular: [String: String]
        let periods: [String: String], period_words: [String: String], ampm: [String: String]
        let time_marker_endings: [String: [String: String]]
        let dash: String, count_one: String, time_hour: String, time_hm: String
        let year_words: [String], time_markers: [String]
        let thousand_exceptions: [String: String]?
        let scale_one: [[String]]?
        let date_days: [String]?
    }
    private static let tables: Result<Tables, Error> = Result {
        guard let url = Bundle.module.url(forResource: "maya2_normalization_data", withExtension: "json") else {
            throw AudioGenerationError.modelNotInitialized("Maya-2 normalization tables are missing")
        }
        return try JSONDecoder().decode(Tables.self, from: Data(contentsOf: url))
    }
    private static let numberPattern = #"\d+(?:,\d+)*(?:\.\d+)?"#
    private static let symbols = ["₹": "rupee", "$": "dollar", "€": "euro", "£": "pound"]
    private static let currencies = ["rs": "rupee", "rs.": "rupee", "inr": "rupee", "rupees": "rupee", "rupee": "rupee", "usd": "dollar", "dollars": "dollar", "dollar": "dollar", "eur": "euro", "euro": "euro", "euros": "euro", "gbp": "pound", "pound": "pound", "pounds": "pound"]
    private enum FormError: Error { case invalid }

    private struct Match {
        let text: String
        let result: NSTextCheckingResult
        var value: String { (text as NSString).substring(with: result.range) }
        func group(_ index: Int) -> String? {
            guard index < result.numberOfRanges, result.range(at: index).location != NSNotFound else { return nil }
            return (text as NSString).substring(with: result.range(at: index))
        }
    }
    private static func matches(_ pattern: String, _ text: String) -> [Match] {
        guard let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { return [] }
        return re.matches(in: text, range: NSRange(text.startIndex..., in: text)).map { Match(text: text, result: $0) }
    }
    private static func substitute(_ pattern: String, _ text: String, _ transform: (Match) -> String) -> String {
        let source = text as NSString
        var output = "", offset = 0
        for m in matches(pattern, text) {
            output += source.substring(with: NSRange(location: offset, length: m.result.range.location - offset)) + transform(m)
            offset = NSMaxRange(m.result.range)
        }
        return output + source.substring(from: offset)
    }
    private static func alternatives<S: Sequence>(_ values: S) -> String where S.Element == String {
        let values = values.sorted { $0.utf16.count > $1.utf16.count }
        return values.isEmpty ? "(?!)" : values.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
    }
    private static func dropLastScalar(_ text: String) -> String { String(String.UnicodeScalarView(text.unicodeScalars.dropLast())) }
    private static func dropFirstScalar(_ text: String) -> String { String(String.UnicodeScalarView(text.unicodeScalars.dropFirst())) }
    private static func ungroup(_ value: String) throws -> String {
        let integer = value.trimmingCharacters(in: CharacterSet(charactersIn: "+-")).components(separatedBy: ".")[0]
        if integer.contains(","), matches(#"^(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})$"#, integer).isEmpty {
            throw FormError.invalid
        }
        return value.replacingOccurrences(of: ",", with: "")
    }

    private struct Language {
        let code: String
        let data: LanguageData
        func symbol(_ key: String) -> String { data.symbols[key] ?? key }
        func join(_ input: [String]) -> String {
            var parts = input.filter { !$0.isEmpty }
            if code == "ta", parts.count > 1 {
                for i in 0..<(parts.count - 1) {
                    if parts[i].hasSuffix("த்து") || parts[i].hasSuffix("ற்று") {
                        let first = parts[i + 1].unicodeScalars.first.map(String.init) ?? ""
                        if let suffix = ["க": "க்", "ச": "ச்", "த": "த்", "ப": "ப்"][first] { parts[i] += suffix }
                    }
                }
            }
            return parts.joined(separator: " ")
        }
        func cardinal(_ number: Int) throws -> String {
            guard number >= 0 else { throw FormError.invalid }
            if number < 100 { return data.ones[number] }
            if number < 1000 {
                let h = number / 100, rest = number % 100
                return rest > 0 ? join([data.hundreds_comb[h - 1], data.ones[rest]]) : data.hundreds[h - 1]
            }
            var parts: [String] = [], value = number
            for (index, scale) in [(2, 10_000_000), (1, 100_000), (0, 1000)] {
                let count = value / scale
                value %= scale
                if count == 0 { continue }
                let combining = value > 0 ? 1 : 0
                var word = data.scales[index][combining]
                if index == 0 && code == "ta" {
                    guard count < data.ones.count else { throw FormError.invalid }
                    if count != 1 { word = dropLastScalar(data.ones[count]) + "ா" + dropFirstScalar(word) }
                } else if index == 0 && code == "ml" {
                    if let exception = data.thousand_exceptions?[String(count)] { word = exception }
                    else {
                        guard count < data.ones.count else { throw FormError.invalid }
                        let base = data.ones[count]
                        word = base.unicodeScalars.last?.value == 0x0D4D
                            ? dropLastScalar(base) + (base.hasSuffix("ത്") ? "ിനായിരം" : "ായിരം")
                            : base + " " + data.scales[0][0]
                    }
                    if combining == 1 { word = dropLastScalar(word) + "ത്തി" }
                } else if code == "te" && count == 1, let one = data.scale_one { word = one[index][combining] }
                else if !(code == "kn" && index == 0 && count == 1 && parts.isEmpty) {
                    let multiplier = count == 1 && ["ta", "ml"].contains(code) ? data.count_one : try cardinal(count)
                    word = multiplier + " " + word
                }
                parts.append(word)
            }
            if value > 0 { parts.append(try cardinal(value)) }
            return join(parts)
        }
        func year(_ value: Int) throws -> String {
            if !data.year_prefix.isEmpty, (1100...1999).contains(value) {
                let h = value / 100, rest = value % 100
                if code == "te" && rest == 0 { return data.year_prefix[h - 11] + "ు" }
                return data.year_prefix[h - 11] + (rest > 0 ? " " + data.ones[rest] : "")
            }
            return try cardinal(value)
        }
        func digits(_ value: String, comma: Bool = true) -> String {
            value.compactMap { $0.wholeNumberValue }.map { data.ones[$0] }.joined(separator: comma ? ", " : " ")
        }
        func oblique(_ words: String) -> String { code == "te" && words.hasSuffix("లు") ? dropLastScalar(words) : words }
        func number(_ input: String) throws -> String {
            var value = try ungroup(input)
            let sign = value.hasPrefix("-") ? symbol("minus") : value.hasPrefix("+") ? symbol("plus") : ""
            value = value.trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
            let parts = value.components(separatedBy: ".")
            guard let integer = Int(parts[0]) else { throw FormError.invalid }
            var words = try cardinal(integer)
            if parts.count > 1 { words = oblique(words) + " " + symbol("point") + " " + digits(parts[1], comma: false) }
            return [sign, words].filter { !$0.isEmpty }.joined(separator: " ")
        }
        func amount(_ words: String, _ noun: String) -> String {
            words == data.ones[1] ? data.count_one + " " + (data.singular[noun] ?? noun) : oblique(words) + " " + noun
        }
        func money(_ input: String, currency: String, scale: String?) throws -> String {
            guard let key = symbols[currency] ?? data.currency_words[currency.lowercased()] ?? currencies[currency.lowercased()] else { throw FormError.invalid }
            var value = try ungroup(input)
            let sign = value.hasPrefix("-") ? symbol("minus") : value.hasPrefix("+") ? symbol("plus") : ""
            value = value.trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
            let parts = value.components(separatedBy: "."), noun = symbol(key)
            var words: String
            if parts.count > 1, parts[1].count == 2, scale == nil {
                guard let integer = Int(parts[0]), let fraction = Int(parts[1]) else { throw FormError.invalid }
                words = amount(try cardinal(integer), noun)
                if fraction > 0 { words += " " + amount(try cardinal(fraction), symbol(key == "rupee" ? "paisa" : "cent")) }
            } else if let scale {
                guard let scaleKey = data.scale_words[scale.lowercased()], let index = ["thousand": 0, "lakh": 1, "crore": 2][scaleKey] else { throw FormError.invalid }
                let word = code == "te" && value == "1" ? (data.scale_one?[index][0] ?? data.scales[index][0]) : data.scales[index][0]
                words = amount(amount(try number(value), word), noun)
            } else { words = amount(try number(value), noun) }
            return sign.isEmpty ? words : sign + " " + words
        }
        func date(day: Int, month: Int, year value: Int? = nil, monthFirst: Bool = false) throws -> String {
            let year = value ?? 2000
            guard (1...9999).contains(year), (1...12).contains(month), day >= 1 else { throw FormError.invalid }
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
            let days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            guard day <= days[month - 1] else { throw FormError.invalid }
            let name = data.months[month - 1]
            let words: String
            if code == "ta", let dateDays = data.date_days { words = name + " " + dateDays[day] + " தேதி" }
            else { words = monthFirst && code != "en" ? name + " " + (try cardinal(day)) : (try cardinal(day)) + " " + name }
            return words + (try value.map { " " + (try self.year($0)) } ?? "")
        }
        func time(hour: Int, minute: Int, ampm: String?, sourcePeriod: String?) throws -> String {
            guard hour <= 23, minute <= 59, ampm == nil || (1...12).contains(hour) else { throw FormError.invalid }
            var period: String?
            if let ampm, data.ampm.isEmpty, sourcePeriod == nil {
                let h24 = hour % 12 + (ampm == "pm" ? 12 : 0)
                period = (4..<12).contains(h24) ? "morning" : (12..<16).contains(h24) ? "afternoon" : (16..<20).contains(h24) ? "evening" : "night"
            }
            var words: String
            if code == "te", (1...12).contains(hour) {
                let h = hour == 1 ? "ఒక" : data.ones[hour]
                let noun = hour == 1 ? "గంట" : minute == 0 ? "గంటలు" : "గంటల"
                words = h + " " + noun + (minute == 1 ? " ఒక నిమిషం" : minute > 0 ? " " + data.ones[minute] + " నిమిషాలు" : "")
            } else if code == "en" {
                words = data.ones[hour] + (minute > 0 ? " " + data.ones[minute] : ampm != nil ? "" : " o'clock")
                if hour == 0 { words = "zero " + (minute > 0 ? data.ones[minute] : "hundred") }
            } else {
                words = (minute > 0 ? data.time_hm : data.time_hour)
                    .replacingOccurrences(of: "{h}", with: data.ones[hour]).replacingOccurrences(of: "{m}", with: data.ones[minute])
            }
            if let sourcePeriod { words = sourcePeriod + " " + words }
            else if let period { words = (data.periods[period] ?? period) + " " + words }
            if let ampm, let spoken = data.ampm[ampm] { words += " " + spoken }
            return words
        }
    }

    private static func uri(_ input: String, language: Language, tables: Tables) -> String {
        let text = input.trimmingCharacters(in: CharacterSet(charactersIn: ".,;!?"))
        let tail = String(input.suffix(input.utf16.count - text.utf16.count))
        let address = substitute(#"^https?://"#, text) { _ in "" }
        let symbols = ["@": "at", ".": "dot", "_": "underscore", "/": "slash", "+": "plus", "#": "hash"]
        var parts: [String] = []
        for token in matches(#"[^\W\d_]+|\d+|[^\w\s]|_"#, address).map(\.value) {
            if token.allSatisfy({ $0.wholeNumberValue != nil }) { parts.append(language.digits(token, comma: false)) }
            else if let key = symbols[token] { parts.append(language.symbol(key)) }
            else if token == "-" { parts.append(language.data.dash) }
            else if token.lowercased() == "www" || tables.uri_acronyms.contains(token.lowercased()) || token.count == 1 && token.unicodeScalars.allSatisfy({ $0.isASCII && CharacterSet.letters.contains($0) }) {
                parts.append(token.uppercased().map { language.data.letters[String($0)] ?? String($0) }.joined(separator: " "))
            } else { parts.append(token) }
        }
        return parts.joined(separator: " ") + tail
    }

    public static func normalize(_ input: String, language code: String = "en") throws -> String {
        let tables = try tables.get()
        guard let data = tables.languages[code] else { throw AudioGenerationError.invalidInput("Unsupported Maya-2 language: \(code)") }
        let language = Language(code: code, data: data)
        var months = tables.month_aliases.merging(data.month_aliases) { _, new in new }
        for (i, name) in data.months.enumerated() { months[name.lowercased()] = i + 1 }
        let currencyPattern = alternatives(Set(data.currency_words.keys).union(symbols.keys).union(["Rs", "Rs.", "INR"]))
        let scales = alternatives(data.scale_words.keys), monthNames = alternatives(months.keys)
        let markers = alternatives(data.time_markers), periods = alternatives(data.period_words.keys)
        let patterns: [(String, String)] = [
            ("tag", #"\[[^\[\]\n]*\]"#),
            ("uri", #"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}|(?:https?://|www\.)[^\s\[\]]+|(?<![\w.])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?:/[^\s\[\]]*)?"#),
            ("date", #"(?<![\w-])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})(?![\w/-])"#),
            ("namedDate", #"(?<!\w)(?:\d{1,2}[ -](?:\#(monthNames))(?!\w)(?:[ ,\-]+\d{4})?|(?:\#(monthNames))\s+\d{1,2}(?:,?\s+\d{4})?)(?!\w)"#),
            ("time", #"(?:(?<!\w)(?:\#(periods))\s+)?(?<![\w.])\d{1,2}(?::\d{2}|\.\d{2}(?=\s*[ap]\.?m\.?)|(?=\s*(?:[ap]\.?m\.?|\#(markers))(?!\w)))(?:\s*[ap]\.?m\.?)?(?:\s*(?:\#(markers))(?!\w))?(?:\s+(?:\#(periods))(?!\w))?(?!\d)"#),
            ("money", #"(?<!\w)[+-]?(?:\#(currencyPattern))\s*[+-]?\#(numberPattern)(?:\s*(?:\#(scales))(?!\w))?|(?<!\w)[+-]?\#(numberPattern)\s+(?:\#(currencyPattern))(?!\w)"#),
            ("percent", #"(?<!\w)-?\#(numberPattern)\s*%"#),
            ("version", #"(?<!\w)v\d+(?:\.\d+){2,}(?!\w)"#),
            ("identifier", #"(?<!\w)(?:\+\d{1,3}[ -]*)?(?:\d{4}(?:[ -]\d{4}){3}|\d{5}[ -]\d{5}|\d{3}[ -]\d{3}[ -]\d{4})(?!\d|[.,]\d)|(?<!\w)[A-Za-z]+[-\w]*\d[-\w]*(?!\w)|[#]\s*\d+"#),
            ("number", #"(?<![\w.])[+-]?\#(numberPattern)(?![\w.]\d)"#)
        ]
        let regex = try NSRegularExpression(pattern: patterns.map { "(?<\($0.0)>\($0.1))" }.joined(separator: "|"), options: .caseInsensitive)
        var text = substitute(#"\[[^\[\]\n]*\]|\d"#, input) { m in
            m.value.hasPrefix("[") ? m.value : m.value.first?.wholeNumberValue.map(String.init) ?? m.value
        }
        text = substitute(#"\[[^\[\]\n]*\]|([\w\u0900-\u0dff]+)\s*:\s*(?=\d)"#, text) { m in
            if let label = m.group(1), tables.id_words.contains(label.lowercased()) { return label + ", " }
            return m.value
        }
        let source = text as NSString
        var output = "", offset = 0
        for result in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let match = Match(text: text, result: result), token = match.value
            let kind = patterns.first { result.range(withName: $0.0).location != NSNotFound }!.0
            let replacement: String
            do { replacement = try replace(token: token, kind: kind, match: match, language: language, tables: tables, months: months) }
            catch { replacement = token }
            output += source.substring(with: NSRange(location: offset, length: result.range.location - offset)) + replacement
            offset = NSMaxRange(result.range)
        }
        output += source.substring(from: offset)
        return substitute(#"\[[^\[\]\n]*\]|\s+"#, output) { $0.value.hasPrefix("[") ? $0.value : " " }.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func replace(token: String, kind: String, match: Match, language: Language, tables: Tables, months: [String: Int]) throws -> String {
        switch kind {
        case "tag": return token
        case "uri": return uri(token, language: language, tables: tables)
        case "date":
            guard Set(matches(#"[/\.\-]"#, token).map(\.value)).count == 1 else { throw FormError.invalid }
            let parts = token.components(separatedBy: CharacterSet(charactersIn: "/.-"))
            guard parts.count == 3, let a = Int(parts[0]), let b = Int(parts[1]), let c = Int(parts[2]) else { throw FormError.invalid }
            var (year, month, day) = parts[0].count == 4 ? (a, b, c) : (c, b, a)
            if parts[0].count != 4 && parts[2].count == 2 { year += year <= 50 ? 2000 : 1900 }
            return try language.date(day: day, month: month, year: year)
        case "namedDate":
            let dayFirst = matches(#"^(\d{1,2})[ -](.+?)(?:[ ,\-]+(\d{4}))?$"#, token).first
            let m = dayFirst ?? matches(#"^(.+?)\s+(\d{1,2})(?:,?\s+(\d{4}))?$"#, token).first
            guard let m, let day = Int(m.group(dayFirst == nil ? 2 : 1) ?? ""),
                  let month = months[(m.group(dayFirst == nil ? 1 : 2) ?? "").lowercased()] else { throw FormError.invalid }
            return try language.date(day: day, month: month, year: m.group(3).flatMap(Int.init), monthFirst: dayFirst == nil)
        case "time":
            guard let h = matches(#"(\d{1,2})(?:[.:](\d{2}))?"#, token).first, let hour = Int(h.group(1) ?? "") else { throw FormError.invalid }
            let ampm = matches(#"([ap])\.?m\.?"#, token).first?.group(1).map { $0.lowercased() + "m" }
            let period = matches(#"(?<!\w)(\#(alternatives(language.data.period_words.keys)))(?!\w)"#, token).first?.value
            let words = try language.time(hour: hour, minute: h.group(2).flatMap(Int.init) ?? 0, ampm: ampm, sourcePeriod: period)
            for (marker, endings) in language.data.time_marker_endings where !matches(NSRegularExpression.escapedPattern(for: marker) + #"(?!\w)"#, token).isEmpty {
                if let space = words.lastIndex(of: " ") {
                    let last = String(words[words.index(after: space)...])
                    return String(words[...space]) + (endings[last] ?? last)
                }
            }
            return words
        case "money":
            guard let n = matches(numberPattern, token).first else { throw FormError.invalid }
            let ns = token as NSString
            let before = ns.substring(to: n.result.range.location).trimmingCharacters(in: .whitespaces)
            let after = ns.substring(from: NSMaxRange(n.result.range)).trimmingCharacters(in: .whitespaces)
            let currency = before.trimmingCharacters(in: CharacterSet(charactersIn: "+- "))
            let value = (before.contains("-") ? "-" : before.contains("+") ? "+" : "") + n.value
            if currency.isEmpty { return try language.number(value) + " " + after }
            return try language.money(value, currency: currency, scale: after.isEmpty ? nil : after)
        case "percent": return try language.number(String(token.dropLast()).trimmingCharacters(in: .whitespaces)) + " " + language.symbol("percent")
        case "version":
            let parts = try token.dropFirst().components(separatedBy: ".").map { value -> String in
                guard let n = Int(value) else { throw FormError.invalid }; return try language.cardinal(n)
            }
            return (language.data.letters["V"] ?? "V") + " " + parts.joined(separator: " " + language.symbol("point") + " ")
        case "identifier":
            var words: [String] = []
            for char in token {
                if let digit = char.wholeNumberValue { words.append(language.data.ones[digit]) }
                else if char.unicodeScalars.allSatisfy({ $0.isASCII && CharacterSet.letters.contains($0) }) {
                    words.append(language.data.letters[String(char).uppercased()] ?? String(char))
                }
            }
            let separator = token.contains("-") && token.unicodeScalars.contains(where: { CharacterSet.letters.contains($0) }) ? " " : ", "
            return (token.hasPrefix("+") ? language.symbol("plus") + " " : "") + words.joined(separator: separator)
        default:
            let digits = token.replacingOccurrences(of: ",", with: "").trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
            if !digits.contains(".") && !token.contains(",") {
                let ns = match.text as NSString
                let beforeStart = max(0, match.result.range.location - 70)
                let before = ns.substring(with: NSRange(location: beforeStart, length: match.result.range.location - beforeStart)).lowercased()
                let afterStart = NSMaxRange(match.result.range)
                let after = ns.substring(with: NSRange(location: afterStart, length: min(50, ns.length - afterStart))).lowercased()
                let beforeTail = before.components(separatedBy: CharacterSet(charactersIn: ".!?;।॥\n")).last ?? ""
                let afterHead = after.components(separatedBy: CharacterSet(charactersIn: ".!?;।॥\n")).first ?? ""
                let context = Set(matches(#"[\w\u0900-\u0dff]+"#, beforeTail).suffix(3).map(\.value) + matches(#"[\w\u0900-\u0dff]+"#, afterHead).prefix(4).map(\.value))
                let identifier = !context.isDisjoint(with: tables.id_words) && context.isDisjoint(with: tables.label_words + language.data.year_words)
                if digits.hasPrefix("0") && digits.count > 1 || digits.count >= 7 || identifier && digits.count >= 3 {
                    let sign = token.hasPrefix("-") ? language.symbol("minus") + " " : token.hasPrefix("+") ? language.symbol("plus") + " " : ""
                    return sign + language.digits(digits)
                }
            }
            return try language.number(token)
        }
    }
}
