//
//  TextUtils.swift
//  MLXAudio
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation

// MARK: - Number to Words

private let ones = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]

private let tens = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]

private let ordinals: [Int: String] = [
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
    30: "thirtieth", 40: "fortieth", 50: "fiftieth", 60: "sixtieth",
    70: "seventieth", 80: "eightieth", 90: "ninetieth"
]

private func numToWords(_ n: Int) -> String {
    if n < 0 {
        return "minus " + numToWords(-n)
    }
    if n == 0 {
        return "zero"
    }
    if n < 20 {
        return ones[n]
    }
    if n < 100 {
        let tensWord = tens[n / 10]
        let onesWord = n % 10 == 0 ? "" : " " + ones[n % 10]
        return tensWord + onesWord
    }
    if n < 1000 {
        let hundredsWord = ones[n / 100] + " hundred"
        let remainder = n % 100 == 0 ? "" : " " + numToWords(n % 100)
        return hundredsWord + remainder
    }
    if n < 1_000_000 {
        let thousandsWord = numToWords(n / 1000) + " thousand"
        let remainder = n % 1000 == 0 ? "" : " " + numToWords(n % 1000)
        return thousandsWord + remainder
    }
    if n < 1_000_000_000 {
        let millionsWord = numToWords(n / 1_000_000) + " million"
        let remainder = n % 1_000_000 == 0 ? "" : " " + numToWords(n % 1_000_000)
        return millionsWord + remainder
    }
    let billionsWord = numToWords(n / 1_000_000_000) + " billion"
    let remainder = n % 1_000_000_000 == 0 ? "" : " " + numToWords(n % 1_000_000_000)
    return billionsWord + remainder
}

private func ordinalToWords(_ n: Int) -> String {
    if let ordinal = ordinals[n] {
        return ordinal
    }
    if n < 100 {
        let tensVal = n / 10
        let onesVal = n % 10
        if onesVal == 0 {
            return ordinals[n] ?? tens[tensVal] + "th"
        }
        return tens[tensVal] + " " + (ordinals[onesVal] ?? ones[onesVal] + "th")
    }
    if n % 100 != 0 {
        return numToWords(n - n % 100) + " " + ordinalToWords(n % 100)
    }
    let base = numToWords(n)
    if base.hasSuffix("y") {
        return String(base.dropLast()) + "ieth"
    }
    return base + "th"
}

// MARK: - Abbreviations

private let abbreviations: [(pattern: String, replacement: String)] = [
    ("\\bmrs\\.", "misuss"),
    ("\\bms\\.", "miss"),
    ("\\bmr\\.", "mister"),
    ("\\bdr\\.", "doctor"),
    ("\\bst\\.", "saint"),
    ("\\bco\\.", "company"),
    ("\\bjr\\.", "junior"),
    ("\\bmaj\\.", "major"),
    ("\\bgen\\.", "general"),
    ("\\bdrs\\.", "doctors"),
    ("\\brev\\.", "reverend"),
    ("\\blt\\.", "lieutenant"),
    ("\\bhon\\.", "honorable"),
    ("\\bsgt\\.", "sergeant"),
    ("\\bcapt\\.", "captain"),
    ("\\besq\\.", "esquire"),
    ("\\bltd\\.", "limited"),
    ("\\bcol\\.", "colonel"),
    ("\\bft\\.", "fort")
]

private let casedAbbreviations: [(pattern: String, replacement: String)] = [
    ("\\bTTS\\b", "text to speech"),
    ("\\bHz\\b", "hertz"),
    ("\\bkHz\\b", "kilohertz"),
    ("\\bKBs\\b", "kilobytes"),
    ("\\bKB\\b", "kilobyte"),
    ("\\bMBs\\b", "megabytes"),
    ("\\bMB\\b", "megabyte"),
    ("\\bGBs\\b", "gigabytes"),
    ("\\bGB\\b", "gigabyte"),
    ("\\bTBs\\b", "terabytes"),
    ("\\bTB\\b", "terabyte"),
    ("\\bAPIs\\b", "a p i's"),
    ("\\bAPI\\b", "a p i"),
    ("\\bCLIs\\b", "c l i's"),
    ("\\bCLI\\b", "c l i"),
    ("\\bCPUs\\b", "c p u's"),
    ("\\bCPU\\b", "c p u"),
    ("\\bGPUs\\b", "g p u's"),
    ("\\bGPU\\b", "g p u"),
    ("\\bAve\\b", "avenue"),
    ("\\betc\\b", "etcetera")
]

private func expandAbbreviations(_ text: String) -> String {
    var result = text

    // Case-insensitive abbreviations
    for (pattern, replacement) in abbreviations {
        if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: replacement)
        }
    }

    // Case-sensitive abbreviations
    for (pattern, replacement) in casedAbbreviations {
        if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: replacement)
        }
    }

    return result
}

// MARK: - Number Normalization

private func expandNumPrefix(_ match: String) -> String {
    // #1 -> "number 1"
    let digit = String(match.dropFirst())
    return "number \(digit)"
}

private func expandNumSuffix(_ match: String) -> String {
    // 1K -> "1 thousand"
    let digit = String(match.dropLast())
    let suffix = match.last!.uppercased()
    let suffixes = ["K": "thousand", "M": "million", "B": "billion", "T": "trillion"]
    return "\(digit) \(suffixes[suffix] ?? "")"
}

private func expandDollars(_ match: String) -> String {
    // Remove $ and commas
    let cleaned = match.replacingOccurrences(of: ",", with: "")
    let parts = cleaned.split(separator: ".")

    if parts.count > 2 {
        return match + " dollars"
    }

    let dollars = parts.first.flatMap { Int($0) } ?? 0
    let cents = parts.count > 1 ? (Int(parts[1]) ?? 0) : 0

    if dollars > 0 && cents > 0 {
        let dollarUnit = dollars == 1 ? "dollar" : "dollars"
        let centUnit = cents == 1 ? "cent" : "cents"
        return "\(numToWords(dollars)) \(dollarUnit), \(numToWords(cents)) \(centUnit)"
    }
    if dollars > 0 {
        let dollarUnit = dollars == 1 ? "dollar" : "dollars"
        return "\(numToWords(dollars)) \(dollarUnit)"
    }
    if cents > 0 {
        let centUnit = cents == 1 ? "cent" : "cents"
        return "\(numToWords(cents)) \(centUnit)"
    }
    return "zero dollars"
}

private func expandOrdinal(_ match: String) -> String {
    // Remove suffix (st, nd, rd, th)
    let numStr = match.replacingOccurrences(of: "st", with: "")
        .replacingOccurrences(of: "nd", with: "")
        .replacingOccurrences(of: "rd", with: "")
        .replacingOccurrences(of: "th", with: "")
    if let num = Int(numStr) {
        return ordinalToWords(num)
    }
    return match
}

private func expandNumber(_ match: String) -> String {
    guard let num = Int(match) else { return match }

    if num > 1000 && num < 3000 {
        if num == 2000 {
            return "two thousand"
        }
        if num > 2000 && num < 2010 {
            return "two thousand " + numToWords(num % 100)
        }
        if num % 100 == 0 {
            return numToWords(num / 100) + " hundred"
        }
        // Year-like pronunciation
        let first = num / 100
        let second = num % 100
        if second < 10 {
            return numToWords(first) + " oh " + numToWords(second)
        }
        return numToWords(first) + " " + numToWords(second)
    }
    return numToWords(num)
}

private func normalizeNumbers(_ text: String) -> String {
    var result = text

    // #digit pattern
    if let regex = try? NSRegularExpression(pattern: "#\\d", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        let matches = regex.matches(in: result, range: range).reversed()
        for match in matches {
            if let matchRange = Range(match.range, in: result) {
                let matchStr = String(result[matchRange])
                result.replaceSubrange(matchRange, with: expandNumPrefix(matchStr))
            }
        }
    }

    // digit + K/M/B/T pattern
    if let regex = try? NSRegularExpression(pattern: "\\b\\d+[KMBTkmbt]\\b", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        let matches = regex.matches(in: result, range: range).reversed()
        for match in matches {
            if let matchRange = Range(match.range, in: result) {
                let matchStr = String(result[matchRange])
                result.replaceSubrange(matchRange, with: expandNumSuffix(matchStr))
            }
        }
    }

    // Remove commas from numbers
    if let regex = try? NSRegularExpression(pattern: "(\\d[\\d,]+\\d)", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        let matches = regex.matches(in: result, range: range).reversed()
        for match in matches {
            if let matchRange = Range(match.range, in: result) {
                let matchStr = String(result[matchRange])
                result.replaceSubrange(matchRange, with: matchStr.replacingOccurrences(of: ",", with: ""))
            }
        }
    }

    // Dollar amounts
    if let regex = try? NSRegularExpression(pattern: "\\$([\\d.,]*\\d+)", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        let matches = regex.matches(in: result, range: range).reversed()
        for match in matches {
            if let matchRange = Range(match.range, in: result) {
                let matchStr = String(result[matchRange]).dropFirst() // Remove $
                result.replaceSubrange(matchRange, with: expandDollars(String(matchStr)))
            }
        }
    }

    // Ordinals (1st, 2nd, 3rd, etc.)
    if let regex = try? NSRegularExpression(pattern: "\\d+(st|nd|rd|th)", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        let matches = regex.matches(in: result, range: range).reversed()
        for match in matches {
            if let matchRange = Range(match.range, in: result) {
                let matchStr = String(result[matchRange])
                result.replaceSubrange(matchRange, with: expandOrdinal(matchStr))
            }
        }
    }

    // Regular numbers
    if let regex = try? NSRegularExpression(pattern: "\\d+", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        let matches = regex.matches(in: result, range: range).reversed()
        for match in matches {
            if let matchRange = Range(match.range, in: result) {
                let matchStr = String(result[matchRange])
                result.replaceSubrange(matchRange, with: expandNumber(matchStr))
            }
        }
    }

    return result
}

// MARK: - Special Characters

private let specialCharacters: [(pattern: String, replacement: String)] = [
    ("@", " at "),
    ("&", " and "),
    ("%", " percent "),
    (":", "."),
    (";", ","),
    ("\\+", " plus "),
    ("\\\\", " backslash "),
    ("~", " about "),
    ("<", " less than "),
    (">", " greater than "),
    ("=", " equals "),
    ("/", " slash "),
    ("_", " ")
]

private func expandSpecialCharacters(_ text: String) -> String {
    var result = text
    for (pattern, replacement) in specialCharacters {
        if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: replacement)
        }
    }
    return result
}

// MARK: - Text Cleaning

private func convertToASCII(_ text: String) -> String {
    // Normalize unicode and convert to ASCII
    let normalized = text.precomposedStringWithCanonicalMapping
    return normalized.unicodeScalars
        .filter { $0.isASCII }
        .map { String($0) }
        .joined()
}

private func removeUnknownCharacters(_ text: String) -> String {
    var result = text

    // Keep only allowed characters
    if let regex = try? NSRegularExpression(pattern: "[^A-Za-z !\\$%&'\\*\\+,\\-./0123456789<>\\?_]", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "")
    }

    // Remove additional special chars
    if let regex = try? NSRegularExpression(pattern: "[<>/_+]", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "")
    }

    return result
}

private func collapseWhitespace(_ text: String) -> String {
    var result = text

    // Collapse multiple spaces
    if let regex = try? NSRegularExpression(pattern: "\\s+", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: " ")
    }

    // Remove space before punctuation
    if let regex = try? NSRegularExpression(pattern: " ([.?!,])", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "$1")
    }

    return result.trimmingCharacters(in: .whitespaces)
}

private func dedupPunctuation(_ text: String) -> String {
    var result = text

    // Multiple dots -> ...
    if let regex = try? NSRegularExpression(pattern: "\\.{3,}", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "...")
    }

    // Multiple commas
    if let regex = try? NSRegularExpression(pattern: ",+", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: ",")
    }

    // Mixed punctuation with period
    if let regex = try? NSRegularExpression(pattern: "[.,]*\\.[.,]*", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: ".")
    }

    // Mixed punctuation with exclamation
    if let regex = try? NSRegularExpression(pattern: "[.,!]*![.,!]*", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "!")
    }

    // Mixed punctuation with question
    if let regex = try? NSRegularExpression(pattern: "[.,!?]*\\?[.,!?]*", options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "?")
    }

    return result
}

// MARK: - Public API

/// Clean and normalize text for Soprano TTS.
///
/// - Parameter text: Input text to clean
/// - Returns: Cleaned and normalized text
public func cleanTextForSoprano(_ text: String) -> String {
    var result = text
    result = convertToASCII(result)
    result = normalizeNumbers(result)
    result = expandAbbreviations(result)
    result = expandSpecialCharacters(result)
    result = result.lowercased()
    result = removeUnknownCharacters(result)
    result = collapseWhitespace(result)
    result = dedupPunctuation(result)
    return result
}
