import Foundation
import MLX

// MARK: - Text Normalization

/// Normalize text for Echo TTS processing.
func echoNormalizeTextPrompt(_ text: String) -> String {
    var result = text

    // Replace unicode characters with ASCII equivalents
    let replacements: [(String, String)] = [
        ("\u{2018}", "'"), ("\u{2019}", "'"),   // Curly single quotes
        ("\u{201C}", "\""), ("\u{201D}", "\""), // Curly double quotes
        ("\u{2014}", "-"), ("\u{2013}", "-"),   // Em dash, en dash
        ("\u{2026}", "..."),                     // Ellipsis
        (":", ","), (";", ","),                  // Colons/semicolons to commas
    ]
    for (from, to) in replacements {
        result = result.replacingOccurrences(of: from, with: to)
    }

    // Replace newlines with spaces
    result = result.replacingOccurrences(of: "\n", with: " ")
    result = result.replacingOccurrences(of: "\r", with: " ")

    // Auto-prepend [S1] if text doesn't start with special markers
    let trimmed = result.trimmingCharacters(in: .whitespaces)
    if !trimmed.hasPrefix("[") && !trimmed.hasPrefix("(") &&
       !trimmed.contains("S1") && !trimmed.contains("S2") {
        result = "[S1] " + result
    }

    return result
}

// MARK: - UTF-8 Byte Tokenization

/// Encode text as UTF-8 bytes with BOS token.
func echoTokenizerEncode(_ text: String) -> MLXArray {
    let bytes = Array(text.utf8)
    var tokens = [Int32](repeating: 0, count: bytes.count + 1)
    tokens[0] = 0  // BOS token
    for (i, byte) in bytes.enumerated() {
        tokens[i + 1] = Int32(byte)
    }
    return MLXArray(tokens)
}

/// Batch encode texts with padding and attention masks.
/// Returns (tokenIds: [B, maxLen], mask: [B, maxLen]).
func echoGetTextInputIdsAndMask(
    _ texts: [String], maxLength: Int, normalize: Bool = true
) -> (MLXArray, MLXArray, [String]) {
    var normalizedTexts: [String] = []
    var encodedTexts: [MLXArray] = []

    for text in texts {
        let normalized = normalize ? echoNormalizeTextPrompt(text) : text
        normalizedTexts.append(normalized)
        let encoded = echoTokenizerEncode(normalized)
        // Truncate to maxLength
        let truncated = encoded.dim(0) > maxLength
            ? encoded[..<maxLength]
            : encoded
        encodedTexts.append(truncated)
    }

    // Find max length in batch
    let maxLen = min(encodedTexts.map { $0.dim(0) }.max() ?? 0, maxLength)

    // Pad and create masks
    var paddedTokens: [MLXArray] = []
    var masks: [MLXArray] = []

    for encoded in encodedTexts {
        let seqLen = encoded.dim(0)
        if seqLen < maxLen {
            let padding = MLXArray.zeros([maxLen - seqLen], type: Int32.self)
            paddedTokens.append(MLX.concatenated([encoded, padding]))
            let maskOnes = MLXArray.ones([seqLen], type: Int32.self)
            let maskZeros = MLXArray.zeros([maxLen - seqLen], type: Int32.self)
            masks.append(MLX.concatenated([maskOnes, maskZeros]))
        } else {
            paddedTokens.append(encoded)
            masks.append(MLXArray.ones([maxLen], type: Int32.self))
        }
    }

    let tokenIds = MLX.stacked(paddedTokens)  // [B, maxLen]
    let mask = MLX.stacked(masks)              // [B, maxLen]

    return (tokenIds, mask, normalizedTexts)
}
