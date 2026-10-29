import Foundation

enum Zonos2Text {
    static let ones: [Int: String] = [
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine",
    ]
    static let teens: [Int: String] = [
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    ]
    static let tens: [Int: String] = [
        2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty",
        9: "ninety",
    ]
    static let ordinals: [String: String] = [
        "zero": "zeroth", "one": "first", "two": "second", "three": "third", "five": "fifth",
        "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
    ]
    static let months: [String: String] = [
        "jan": "january", "january": "january", "feb": "february", "february": "february",
        "mar": "march", "march": "march", "apr": "april", "april": "april", "may": "may", "jun": "june",
        "june": "june", "jul": "july", "july": "july", "aug": "august", "august": "august",
        "sep": "september", "sept": "september", "september": "september", "oct": "october",
        "october": "october", "nov": "november", "november": "november", "dec": "december",
        "december": "december",
    ]
    static let units: [String: [String]] = [
        "kg": ["kilogram", "kilograms"], "g": ["gram", "grams"], "mg": ["milligram", "milligrams"],
        "km": ["kilometer", "kilometers"], "cm": ["centimeter", "centimeters"],
        "mm": ["millimeter", "millimeters"], "m": ["meter", "meters"], "mi": ["mile", "miles"],
        "mph": ["mile per hour", "miles per hour"], "lb": ["pound", "pounds"],
        "lbs": ["pound", "pounds"], "ft": ["foot", "feet"], "in": ["inch", "inches"],
        "l": ["liter", "liters"], "ml": ["milliliter", "milliliters"], "hz": ["hertz", "hertz"],
        "khz": ["kilohertz", "kilohertz"], "mhz": ["megahertz", "megahertz"],
        "ghz": ["gigahertz", "gigahertz"], "kbps": ["kilobit per second", "kilobits per second"],
        "mbps": ["megabit per second", "megabits per second"],
        "gbps": ["gigabit per second", "gigabits per second"],
        "°c": ["degree celsius", "degrees celsius"], "°f": ["degree fahrenheit", "degrees fahrenheit"],
    ]
    static let currencies: [String: [String]] = [
        "$": ["dollar", "dollars", "cent", "cents"], "€": ["euro", "euros", "cent", "cents"],
        "£": ["pound", "pounds", "penny", "pence"],
    ]
    static let quantities: [String: String] = [
        "k": "thousand", "m": "million", "b": "billion", "bn": "billion", "thousand": "thousand",
        "million": "million", "billion": "billion", "trillion": "trillion",
    ]
    static func integer(_ n: Int) -> String {
        if n == Int.min { return String(n) }
        if n < 0 { return "negative " + integer(-n) }
        if n < 10 { return ones[n]! }
        if n < 20 { return teens[n]! }
        if n < 100 { return tens[n / 10]! + (n % 10 == 0 ? "" : " " + ones[n % 10]!) }
        if n < 1000 {
            return ones[n / 100]! + " hundred" + (n % 100 == 0 ? "" : " " + integer(n % 100))
        }
        for (scale, word) in [
            (1_000_000_000_000, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"),
            (1000, "thousand"),
        ] where n >= scale {
            return integer(n / scale) + " " + word + (n % scale == 0 ? "" : " " + integer(n % scale))
        }
        return String(n)
    }
    static func digits(_ s: String) -> String {
        s.compactMap { $0.wholeNumberValue.flatMap { ones[$0] } }.joined(separator: " ")
    }
    static func clean(_ s: String) -> String { s.replacingOccurrences(of: ",", with: "") }
    static func number(_ raw: String) -> String {
        let value = clean(raw)
        let negative = clean(raw).hasPrefix("-")
        let text = negative ? String(value.dropFirst()) : value
        let prefix = negative ? "negative " : ""
        if let dot = text.firstIndex(of: ".") {
            guard let n = Int(text[..<dot]) else { return raw }
            return prefix + integer(n) + " point " + digits(String(text[text.index(after: dot)...]))
        }
        if text.count > 1, text.first == "0" { return prefix + digits(text) }
        guard let n = Int(text), n < Int.max else { return raw }
        return prefix + integer(n)
    }
    static func ordinal(_ n: Int) -> String {
        var words = integer(n).split(separator: " ").map(String.init)
        guard let last = words.last else { return "" }
        words[words.count - 1] =
            ordinals[last] ?? (last.hasSuffix("y") ? String(last.dropLast()) + "ieth" : last + "th")
        return words.joined(separator: " ")
    }
    static func year(_ n: Int) -> String {
        if (1000...1999).contains(n) {
            return integer(n / 100) + (n % 100 == 0 ? " hundred" : " " + integer(n % 100))
        }
        if (2000...2009).contains(n) {
            return "two thousand" + (n == 2000 ? "" : " " + integer(n - 2000))
        }
        if (2010...2099).contains(n) { return "twenty " + integer(n - 2000) }
        return integer(n)
    }
    static let monthNames = [
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december",
    ]
    static func date(month: Int, day: Int, year y: String?) -> String? {
        guard (1...12).contains(month), (1...31).contains(day) else { return nil }
        var result = monthNames[month - 1] + " " + ordinal(day)
        if let y, var n = Int(y) {
            if n < 100 { n += n < 50 ? 2000 : 1900 }
            result += " " + year(n)
        }
        return result
    }
    struct Match {
        let match: NSTextCheckingResult
        let text: NSString
        func get(_ name: String) -> String? {
            let range = match.range(withName: name)
            return range.location == NSNotFound ? nil : text.substring(with: range)
        }
        var original: String { text.substring(with: match.range) }
    }
    static func replace(
        _ text: String, pattern: String, insensitive: Bool = false, transform: (Match) -> String
    ) -> String {
        guard
            let regex = try? NSRegularExpression(
                pattern: pattern, options: insensitive ? [.caseInsensitive] : [])
        else { return text }
        let original = text as NSString
        var result = text
        for match in regex.matches(in: text, range: NSRange(location: 0, length: original.length))
            .reversed()
        {
            if let range = Range(match.range, in: result) {
                result.replaceSubrange(range, with: transform(Match(match: match, text: original)))
            }
        }
        return result
    }
    static func normalize(_ input: String, language: String) -> String {
        guard ["en", "en_us", "en_gb"].contains(language.lowercased()),
            !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return input }
        var text = input
        text = replace(
            text, pattern: #"\b(?<year>\d{4})-(?<month>\d{1,2})-(?<day>\d{1,2})\b"#, insensitive: false
        ) { m in
            date(month: Int(m.get("month")!)!, day: Int(m.get("day")!)!, year: m.get("year"))
                ?? m.original
        }
        text = replace(
            text,
            pattern:
                #"\b(?<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(?<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)?(?<year>\d{4})?\b"#,
            insensitive: true
        ) { m in
            {
                let name = months[m.get("month")!.lowercased()]!
                let day = Int(m.get("day")!)!
                guard (1...31).contains(day) else { return m.original }
                return name + " " + ordinal(day) + (m.get("year").map { " " + year(Int($0)!) } ?? "")
            }()
        }
        text = replace(
            text, pattern: #"\b(?<month>\d{1,2})/(?<day>\d{1,2})/(?<year>\d{2,4})\b"#, insensitive: false
        ) { m in
            date(month: Int(m.get("month")!)!, day: Int(m.get("day")!)!, year: m.get("year"))
                ?? m.original
        }
        text = replace(
            text,
            pattern:
                #"(?<!\w)(?<hour>[01]?\d|2[0-3]):(?<minute>[0-5]\d)(?::(?<second>[0-5]\d))?\s*(?<suffix>[aApP]\.?\s?[mM]\.?)?\b"#,
            insensitive: false
        ) { m in
            {
                let h = Int(m.get("hour")!)!
                let minute = Int(m.get("minute")!)!
                let suffix = m.get("suffix")
                var parts = [integer(suffix == nil ? h : (h % 12 == 0 ? 12 : h % 12))]
                if minute != 0 { parts.append((minute < 10 ? "oh " : "") + integer(minute)) }
                if let second = m.get("second"), let value = Int(second), value != 0 {
                    parts += ["and", integer(value), "seconds"]
                }
                if let suffix { parts += [suffix.lowercased().hasPrefix("a") ? "a" : "p", "m"] }
                return parts.joined(separator: " ")
            }()
        }
        text = replace(
            text, pattern: #"(?<!\w)(?:\+?1[-.\s]?)?(?:(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4})(?!\w)"#,
            insensitive: false
        ) { m in
            {
                var value = m.original.filter { $0.isNumber }
                if value.count == 11, value.first == "1" { value.removeFirst() }
                return digits(value)
            }()
        }
        text = replace(
            text,
            pattern:
                #"(?<!\w)(?<symbol>[$€£])\s*(?<amount>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*(?<quantity>k|K|m|M|b|B|bn|BN|thousand|million|billion|trillion))?"#,
            insensitive: false
        ) { m in
            {
                let amount = clean(m.get("amount")!)
                let names = currencies[m.get("symbol")!]!
                if let quantity = m.get("quantity") {
                    return number(amount) + " " + quantities[quantity.lowercased()]! + " "
                        + names[Double(amount) == 1 ? 0 : 1]
                }
                let components = amount.split(separator: ".", omittingEmptySubsequences: false)
                guard let major = Int(components[0]) else { return m.original }
                let minor = components.count > 1 ? Int((String(components[1]) + "00").prefix(2))! : 0
                var parts: [String] = []
                if major != 0 { parts.append(integer(major) + " " + names[major == 1 ? 0 : 1]) }
                if minor != 0 { parts.append(integer(minor) + " " + names[minor == 1 ? 2 : 3]) }
                return parts.isEmpty ? "zero " + names[1] : parts.joined(separator: " and ")
            }()
        }
        text = replace(
            text, pattern: #"(?<!\w)(?<number>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*%"#,
            insensitive: false
        ) { m in number(m.get("number")!) + " percent" }
        text = replace(
            text,
            pattern:
                #"(?<!\w)(?<number>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?<unit>kbps|mbps|gbps|mph|lbs|khz|mhz|ghz|kg|mg|km|cm|mm|mi|lb|ft|in|ml|hz|°c|°f|g|m|l)\b"#,
            insensitive: true
        ) { m in
            {
                let n = m.get("number")!
                let names = units[m.get("unit")!.lowercased()]!
                return number(n) + " " + names[Double(clean(n)) == 1 ? 0 : 1]
            }()
        }
        text = replace(
            text, pattern: #"(?<!\w)(?<numerator>\d+)/(?<denominator>\d+)(?!\w)"#, insensitive: false
        ) { m in
            {
                guard let n = Int(m.get("numerator")!), let d = Int(m.get("denominator")!) else {
                    return m.original
                }
                let name =
                    d == 2
                    ? (n == 1 ? "half" : "halves")
                    : d == 4 ? (n == 1 ? "quarter" : "quarters") : ordinal(d) + (n == 1 ? "" : "s")
                return integer(n) + " " + name
            }()
        }
        text = replace(
            text, pattern: #"(?<!\w)(?<number>-?\d{1,3}(?:,\d{3})+|-?\d+)(?:st|nd|rd|th)\b"#,
            insensitive: false
        ) { m in Int(clean(m.get("number")!)).map(ordinal) ?? m.original }
        text = replace(
            text, pattern: #"(?<![\w.])(?<number>-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+)(?![\w.])"#,
            insensitive: false
        ) { m in number(m.get("number")!) }
        text = replace(
            text, pattern: #"(?<![\w.])(?<number>-?(?:\d{1,3}(?:,\d{3})+|\d+))(?![\w.])"#,
            insensitive: false
        ) { m in number(m.get("number")!) }
        text = replace(text, pattern: #"\b[A-Z]{2,}\b"#, insensitive: false) { m in
            m.original.lowercased().map(String.init).joined(separator: " ")
        }
        text = text.replacingOccurrences(of: "&", with: " and ").replacingOccurrences(
            of: "@", with: " at ")
        text = replace(text, pattern: #"\s+(?<punct>[.!?,;:])"#) { $0.get("punct")! }
        return replace(text, pattern: #"\s+"#) { _ in " " }.trimmingCharacters(
            in: .whitespacesAndNewlines)
    }
}
