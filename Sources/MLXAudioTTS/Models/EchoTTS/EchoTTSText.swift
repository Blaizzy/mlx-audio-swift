import Foundation
import MLX

func echoTtsNormalizeTextPrompt(_ text: String) -> String {
    var normalized = text
    normalized = normalized.replacingOccurrences(of: "…", with: "...")
    normalized = normalized.replacingOccurrences(of: "’", with: "'")
    normalized = normalized.replacingOccurrences(of: "“", with: "\"")
    normalized = normalized.replacingOccurrences(of: "”", with: "\"")
    normalized = normalized.replacingOccurrences(of: "\n", with: " ")
    normalized = normalized.replacingOccurrences(of: ":", with: ",")
    normalized = normalized.replacingOccurrences(of: ";", with: ",")
    normalized = normalized.replacingOccurrences(of: "—", with: ", ")

    if !normalized.hasPrefix("[")
        && !normalized.hasPrefix("(")
        && !normalized.contains("S1")
        && !normalized.contains("S2")
    {
        normalized = "[S1] " + normalized
    }

    return normalized
}

func echoTtsTokenizerEncode(
    _ text: String,
    appendBOS: Bool = true,
    normalize: Bool = true
) -> MLXArray {
    let normalized = normalize ? echoTtsNormalizeTextPrompt(text) : text
    var tokens = normalized.utf8.map(Int32.init)
    if appendBOS {
        tokens.insert(0, at: 0)
    }
    return MLXArray(tokens)
}

func echoTtsTextInputIDsAndMask(
    _ texts: [String],
    maxLength: Int?,
    normalize: Bool = true,
    padToMax: Bool = true
) -> (inputIDs: MLXArray, mask: MLXArray, normalizedTexts: [String]) {
    let normalizedTexts = texts.map { normalize ? echoTtsNormalizeTextPrompt($0) : $0 }
    let encoded = normalizedTexts.map {
        echoTtsTokenizerEncode($0, appendBOS: true, normalize: false).asArray(Int32.self)
    }

    let resolvedMaxLength = maxLength ?? encoded.map(\.count).max() ?? 0
    let finalLength: Int
    if padToMax {
        finalLength = resolvedMaxLength
    } else {
        finalLength = encoded.map { min($0.count, resolvedMaxLength) }.max() ?? 0
    }

    var tokenValues = Array(repeating: Int32(0), count: texts.count * finalLength)
    var maskValues = Array(repeating: false, count: texts.count * finalLength)

    for (row, sequence) in encoded.enumerated() {
        let count = min(sequence.count, finalLength)
        guard count > 0 else { continue }
        let base = row * finalLength
        tokenValues.replaceSubrange(base..<(base + count), with: sequence.prefix(count))
        for index in 0 ..< count {
            maskValues[base + index] = true
        }
    }

    let inputIDs = MLXArray(tokenValues).reshaped([texts.count, finalLength])
    let mask = MLXArray(maskValues).reshaped([texts.count, finalLength])
    return (inputIDs, mask, normalizedTexts)
}
