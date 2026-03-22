public struct TextTokenizer: Sendable {
    public init() {}

    public func tokenize(_ normalizedText: String) -> [G2PToken] {
        var tokens: [G2PToken] = []
        var index = normalizedText.startIndex

        while index < normalizedText.endIndex {
            let ch = normalizedText[index]

            if ch.isWhitespace {
                let start = index
                while index < normalizedText.endIndex, normalizedText[index].isWhitespace {
                    index = normalizedText.index(after: index)
                }
                tokens.append(G2PToken(
                    surface: String(normalizedText[start..<index]),
                    kind: .whitespace,
                    rangeInNormalizedText: start..<index
                ))
            } else if ch.isPunctuation || ch.isSymbol {
                let start = index
                index = normalizedText.index(after: index)
                tokens.append(G2PToken(
                    surface: String(normalizedText[start..<index]),
                    kind: .punctuation,
                    rangeInNormalizedText: start..<index
                ))
            } else {
                let start = index
                while index < normalizedText.endIndex,
                      !normalizedText[index].isWhitespace {
                    let c = normalizedText[index]
                    if (c == "'" || c == "\u{2019}" || c == "-"),
                       index > start
                    {
                        let next = normalizedText.index(after: index)
                        if next < normalizedText.endIndex,
                           !normalizedText[next].isWhitespace,
                           !normalizedText[next].isPunctuation || normalizedText[next] == "'" || normalizedText[next] == "-"
                        {
                            index = normalizedText.index(after: index)
                            continue
                        }
                    }
                    if c.isPunctuation || c.isSymbol { break }
                    index = normalizedText.index(after: index)
                }
                tokens.append(G2PToken(
                    surface: String(normalizedText[start..<index]),
                    kind: .word,
                    rangeInNormalizedText: start..<index
                ))
            }
        }
        return tokens
    }

    public static let englishDefault = TextTokenizer()
}
