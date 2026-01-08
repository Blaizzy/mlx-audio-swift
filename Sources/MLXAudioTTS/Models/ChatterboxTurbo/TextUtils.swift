//
//  TextUtils.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation

enum ChatterboxTurboTextUtils {
    static func puncNorm(_ text: String) -> String {
        guard !text.isEmpty else {
            return "You need to add some text for me to talk."
        }

        var normalized = text

        if let first = normalized.first, first.isLowercase {
            normalized = normalized.prefix(1).uppercased() + normalized.dropFirst()
        }

        normalized = normalized
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")

        let replacements: [(String, String)] = [
            ("\u{2026}", ", "),
            (":", ","),
            ("\u{2014}", "-"),
            ("\u{2013}", "-"),
            (" ,", ","),
            ("\u{201C}", "\""),
            ("\u{201D}", "\""),
            ("\u{2018}", "'"),
            ("\u{2019}", "'"),
        ]

        for (oldValue, newValue) in replacements {
            normalized = normalized.replacingOccurrences(of: oldValue, with: newValue)
        }

        normalized = normalized.trimmingCharacters(in: .whitespaces)

        let sentenceEnders: Set<Character> = [".", "!", "?", "-", ","]
        if let last = normalized.last, !sentenceEnders.contains(last) {
            normalized.append(".")
        }

        return normalized
    }
}
