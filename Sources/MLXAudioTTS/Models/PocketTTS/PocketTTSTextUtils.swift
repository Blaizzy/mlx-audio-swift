import Foundation

public enum PocketTTSTextUtils {
    public static func prepareTextPrompt(_ text: String) throws -> (String, Int) {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty {
            throw NSError(domain: "PocketTTSTextUtils", code: 1, userInfo: [NSLocalizedDescriptionKey: "Text prompt cannot be empty"])
        }
        t = t.replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "  ", with: " ")

        let wordCount = t.split(separator: " ").count
        let framesAfterEosGuess = wordCount <= 4 ? 3 : 1

        if let first = t.first, !first.isUppercase {
            t = first.uppercased() + t.dropFirst()
        }
        if let last = t.last, last.isLetter || last.isNumber {
            t += "."
        }
        if t.split(separator: " ").count < 5 {
            t = String(repeating: " ", count: 8) + t
        }
        return (t, framesAfterEosGuess)
    }

    public static func splitIntoBestSentences(
        _ tokenizer: SentencePieceTokenizer,
        _ textToGenerate: String
    ) throws -> [String] {
        let (prepared, _) = try prepareTextPrompt(textToGenerate)
        let cleaned = prepared.trimmingCharacters(in: .whitespaces)

        let tokens = tokenizer.encode(cleaned)
        let endTokens = tokenizer.encode(".!...?")
        guard !tokens.isEmpty else { return [cleaned] }

        var endOfSentenceIndices: [Int] = [0]
        var previousWasEnd = false
        for (idx, tok) in tokens.enumerated() {
            if endTokens.contains(tok) {
                previousWasEnd = true
            } else {
                if previousWasEnd {
                    endOfSentenceIndices.append(idx)
                }
                previousWasEnd = false
            }
        }
        endOfSentenceIndices.append(tokens.count)

        var tokenChunks: [(count: Int, text: String)] = []
        for i in 0..<(endOfSentenceIndices.count - 1) {
            let start = endOfSentenceIndices[i]
            let end = endOfSentenceIndices[i + 1]
            let slice = Array(tokens[start..<end])
            let text = tokenizer.decode(slice)
            tokenChunks.append((end - start, text))
        }

        let maxNbTokensInChunk = 50
        var chunks: [String] = []
        var current = ""
        var currentCount = 0
        for (count, text) in tokenChunks {
            if current.isEmpty {
                current = text
                currentCount = count
                continue
            }
            if currentCount + count > maxNbTokensInChunk {
                chunks.append(current.trimmingCharacters(in: .whitespaces))
                current = text
                currentCount = count
            } else {
                current += " " + text
                currentCount += count
            }
        }
        if !current.isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespaces))
        }
        return chunks
    }
}
