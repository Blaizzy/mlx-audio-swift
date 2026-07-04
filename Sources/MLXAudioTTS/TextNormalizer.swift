// Text normalization for TTS input.
//
// Expands numbers, dates, currency amounts, and common abbreviations
// into their spoken form so the TTS model produces more natural output.
//
// Currently supports English number expansion with basic international
// coverage for common patterns.

import Foundation

/// Normalise text before TTS generation to improve pronunciation of
/// numbers, dates, currency, and abbreviations.
///
/// Usage:
/// ```swift
/// let clean = TextNormalizer.normalize("It costs $50.25")
/// // "It costs fifty dollars and twenty-five cents"
/// ```
public enum TextNormalizer {
    // MARK: - Public API

    /// Run all normalisation passes on `text`.
    ///
    /// - Parameter text: Raw input text.
    /// - Returns: Text with numbers, dates, etc. expanded in‑place.
    public static func normalize(_ text: String) -> String {
        var result = text

        // Order matters — run broad patterns before narrow ones.
        result = expandCurrency(result)
        result = expandOrdinalIndicators(result)
        result = expandCardinalNumbers(result)
        result = expandCommonAbbreviations(result)

        return result
    }

    // MARK: - Currency

    /// Replace `$3.50`, `€12`, `£1.99` etc. with spoken forms.
    private static let currencyPattern =
        try! NSRegularExpression(pattern: #"([£$€¥])(\d{1,6}(?:\.\d{1,2})?)"#)

    private static let currencyNames: [Character: (unit: String, sub: String)] = [
        "$": ("dollars", "cents"),
        "£": ("pounds", "pence"),
        "€": ("euros", "cents"),
        "¥": ("yuan", "fen"),
    ]

    private static func expandCurrency(_ text: String) -> String {
        let ns = text as NSString
        var result = text

        for match in currencyPattern.matches(in: text, range: NSRange(location: 0, length: ns.length))
            .reversed()
        {
            let fullRange = NSRange(location: match.range.location, length: match.range.length)
            let symbol = ns.substring(with: match.range(at: 1))
            let amount = ns.substring(with: match.range(at: 2))

            guard let sym = symbol.first, let names = currencyNames[sym],
                  let value = Double(amount)
            else { continue }

            let spoken: String
            if amount.contains(".") {
                let parts = amount.split(separator: ".")
                let major = Int(parts[0]) ?? 0
                let minor = Int(parts[1].padding(toLength: 2, withPad: "0", startingAt: 0)) ?? 0
                if major == 0 {
                    spoken = "\(minor) \(names.sub)"
                } else if minor == 0 {
                    spoken = "\(numberToWords(major)) \(names.unit)"
                } else {
                    spoken =
                        "\(numberToWords(major)) \(names.unit) and \(numberToWords(minor)) \(names.sub)"
                }
            } else {
                let intVal = Int(value)
                spoken = "\(numberToWords(intVal)) \(names.unit)"
            }

            result = (result as NSString).replacingCharacters(in: fullRange, with: spoken)
        }
        return result
    }

    // MARK: - Ordinal indicators

    private static let ordinalPattern =
        try! NSRegularExpression(pattern: #"\b(\d{1,6})(st|nd|rd|th)\b"#)

    private static func expandOrdinalIndicators(_ text: String) -> String {
        let ns = text as NSString
        var result = text

        for match in ordinalPattern.matches(in: text, range: NSRange(location: 0, length: ns.length))
            .reversed()
        {
            let fullRange = NSRange(location: match.range.location, length: match.range.length)
            let numStr = ns.substring(with: match.range(at: 1))
            guard let num = Int(numStr) else { continue }
            let spoken = ordinalToWords(num)
            result = (result as NSString).replacingCharacters(in: fullRange, with: spoken)
        }
        return result
    }

    // MARK: - Cardinal numbers

    private static let cardinalPattern =
        try! NSRegularExpression(pattern: #"(?<![a-zA-Z])(\d{1,9})(?![a-zA-Z])"#)

    private static func expandCardinalNumbers(_ text: String) -> String {
        let ns = text as NSString
        var result = text

        for match in cardinalPattern.matches(in: text, range: NSRange(location: 0, length: ns.length))
            .reversed()
        {
            let fullRange = NSRange(location: match.range.location, length: match.range.length)
            let numStr = ns.substring(with: match.range(at: 1))
            guard let num = Int(numStr) else { continue }
            // Skip years (4 digits) — handled by context, not a simple rule
            if numStr.count == 4 && num >= 1000 && num <= 2099 { continue }
            let spoken = numberToWords(num)
            result = (result as NSString).replacingCharacters(in: fullRange, with: spoken)
        }
        return result
    }

    // MARK: - Common abbreviations

    private static let abbreviationMap: [String: String] = [
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "etcetera",
        "vs.": "versus",
        "Dr.": "doctor",
        "Mr.": "mister",
        "Mrs.": "missus",
        "Ms.": "miss",
        "Prof.": "professor",
        "St.": "saint",
        "Ave.": "avenue",
        "Blvd.": "boulevard",
        "Dept.": "department",
        "Est.": "established",
        "Inc.": "incorporated",
        "Ltd.": "limited",
        "Corp.": "corporation",
        "Co.": "company",
        "Mt.": "mount",
        "Ft.": "fort",
        "Jan.": "January",
        "Feb.": "February",
        "Mar.": "March",
        "Apr.": "April",
        "Jun.": "June",
        "Jul.": "July",
        "Aug.": "August",
        "Sep.": "September",
        "Oct.": "October",
        "Nov.": "November",
        "Dec.": "December",
    ]

    private static func expandCommonAbbreviations(_ text: String) -> String {
        var result = text
        for (abbr, expansion) in abbreviationMap {
            result = result.replacingOccurrences(of: abbr, with: expansion)
        }
        return result
    }

    // MARK: - Number to words

    private static let ones = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]

    private static let tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety",
    ]

    private static func numberToWords(_ n: Int) -> String {
        guard n >= 0 else { return "minus \(numberToWords(-n))" }
        if n >= 1_000_000_000 {
            let b = n / 1_000_000_000; let r = n % 1_000_000_000
            let base = b == 1 ? "one billion" : "\(numberToWords(b)) billion"
            return r == 0 ? base : "\(base) \(numberToWords(r))"
        }
        if n >= 1_000_000 {
            let m = n / 1_000_000; let r = n % 1_000_000
            let base = m == 1 ? "one million" : "\(numberToWords(m)) million"
            return r == 0 ? base : "\(base) \(numberToWords(r))"
        }
        if n >= 1_000 {
            let th = n / 1_000; let r = n % 1_000
            let base = th == 1 ? "one thousand" : "\(numberToWords(th)) thousand"
            return r == 0 ? base : "\(base) \(numberToWords(r))"
        }
        if n >= 100 {
            let h = n / 100; let r = n % 100
            let base = h == 1 ? "one hundred" : "\(ones[h]) hundred"
            return r == 0 ? base : "\(base) and \(numberToWords(r))"
        }
        if n >= 20 {
            let t = n / 10; let o = n % 10
            return o == 0 ? tens[t] : "\(tens[t])-\(ones[o])"
        }
        return ones[n]
    }

    private static func ordinalToWords(_ n: Int) -> String {
        let card = numberToWords(n)
        // Simple ordinal suffix mapping
        let suffix: String
        let mod100 = n % 100
        let mod10 = n % 10
        if mod100 >= 11 && mod100 <= 13 {
            suffix = "th"
        } else {
            switch mod10 {
            case 1: suffix = "st"
            case 2: suffix = "nd"
            case 3: suffix = "rd"
            default: suffix = "th"
            }
        }

        // Special ordinal words for 1-20
        if n == 1 { return "first" }
        if n == 2 { return "second" }
        if n == 3 { return "third" }
        if n == 4 { return "fourth" }
        if n == 5 { return "fifth" }
        if n == 6 { return "sixth" }
        if n == 7 { return "seventh" }
        if n == 8 { return "eighth" }
        if n == 9 { return "ninth" }
        if n == 10 { return "tenth" }
        if n == 11 { return "eleventh" }
        if n == 12 { return "twelfth" }
        if n == 13 { return "thirteenth" }

        return "\(card)\(suffix)"
    }
}

// MARK: - Foundation helpers

extension String {
    /// Simple‑minded lowercase ASCII detection — enough for abbreviation matching.
    fileprivate var lowercasedASCII: String {
        lowercased()
    }
}
