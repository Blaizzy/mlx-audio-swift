import Foundation

public struct ParakeetAlignedToken: Sendable {
    public let id: Int
    public let text: String
    public var start: Double
    public var duration: Double

    public init(id: Int, text: String, start: Double, duration: Double) {
        self.id = id
        self.text = text
        self.start = start
        self.duration = duration
    }

    public var end: Double {
        start + duration
    }
}

public struct ParakeetAlignedSentence: Sendable {
    public let text: String
    public let tokens: [ParakeetAlignedToken]

    public init(text: String, tokens: [ParakeetAlignedToken]) {
        self.text = text
        self.tokens = tokens.sorted { $0.start < $1.start }
    }

    public var start: Double {
        tokens.first?.start ?? 0
    }

    public var end: Double {
        tokens.last?.end ?? 0
    }

    public var duration: Double {
        end - start
    }
}

public struct ParakeetAlignedResult: Sendable {
    public let text: String
    public let sentences: [ParakeetAlignedSentence]

    public init(text: String, sentences: [ParakeetAlignedSentence]) {
        self.text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sentences = sentences
    }

    public var segments: [[String: Any]] {
        sentences.map {
            [
                "text": $0.text,
                "start": $0.start,
                "end": $0.end,
            ]
        }
    }
}

public struct ParakeetStreamingResult: Sendable {
    public let text: String
    public let tokens: [Int]
    public let isFinal: Bool
    public let startTime: Double
    public let endTime: Double
    public let progress: Double
    public let audioPosition: Double
    public let audioDuration: Double
    public let language: String

    public init(
        text: String,
        tokens: [Int],
        isFinal: Bool,
        startTime: Double,
        endTime: Double,
        progress: Double = 0,
        audioPosition: Double = 0,
        audioDuration: Double = 0,
        language: String = "en"
    ) {
        self.text = text
        self.tokens = tokens
        self.isFinal = isFinal
        self.startTime = startTime
        self.endTime = endTime
        self.progress = progress
        self.audioPosition = audioPosition
        self.audioDuration = audioDuration
        self.language = language
    }
}

enum ParakeetAlignment {
    static func tokensToSentences(_ tokens: [ParakeetAlignedToken]) -> [ParakeetAlignedSentence] {
        var sentences: [ParakeetAlignedSentence] = []
        var current: [ParakeetAlignedToken] = []

        for (i, token) in tokens.enumerated() {
            current.append(token)
            if shouldCloseSentence(token: token, index: i, allTokens: tokens) {
                let text = current.map(\.text).joined()
                sentences.append(ParakeetAlignedSentence(text: text, tokens: current))
                current.removeAll(keepingCapacity: true)
            }
        }

        if !current.isEmpty {
            let text = current.map(\.text).joined()
            sentences.append(ParakeetAlignedSentence(text: text, tokens: current))
        }

        return sentences
    }

    static func sentencesToResult(_ sentences: [ParakeetAlignedSentence]) -> ParakeetAlignedResult {
        ParakeetAlignedResult(text: sentences.map(\.text).joined(), sentences: sentences)
    }

    static func mergeLongestContiguous(
        _ a: [ParakeetAlignedToken],
        _ b: [ParakeetAlignedToken],
        overlapDuration: Double
    ) throws -> [ParakeetAlignedToken] {
        if a.isEmpty { return b }
        if b.isEmpty { return a }

        let aEnd = a[a.count - 1].end
        let bStart = b[0].start
        if aEnd <= bStart { return a + b }

        let overlapA = a.filter { $0.end > bStart - overlapDuration }
        let overlapB = b.filter { $0.start < aEnd + overlapDuration }
        let enoughPairs = overlapA.count / 2

        if overlapA.count < 2 || overlapB.count < 2 {
            let cutoff = (aEnd + bStart) / 2
            return a.filter { $0.end <= cutoff } + b.filter { $0.start >= cutoff }
        }

        var best: [(Int, Int)] = []
        for i in overlapA.indices {
            for j in overlapB.indices {
                if !matches(overlapA[i], overlapB[j], overlapDuration: overlapDuration / 2) {
                    continue
                }
                var chain: [(Int, Int)] = []
                var k = i
                var l = j
                while k < overlapA.count && l < overlapB.count
                    && matches(overlapA[k], overlapB[l], overlapDuration: overlapDuration / 2) {
                    chain.append((k, l))
                    k += 1
                    l += 1
                }
                if chain.count > best.count {
                    best = chain
                }
            }
        }

        if best.count < enoughPairs {
            throw ParakeetAlignmentError.noStrongOverlap
        }

        let aStart = a.count - overlapA.count
        let indicesA = best.map { aStart + $0.0 }
        let indicesB = best.map { $0.1 }

        var merged: [ParakeetAlignedToken] = []
        merged.append(contentsOf: a[..<indicesA[0]])

        for idx in best.indices {
            let ia = indicesA[idx]
            let ib = indicesB[idx]
            merged.append(a[ia])

            if idx < best.count - 1 {
                let nextA = indicesA[idx + 1]
                let nextB = indicesB[idx + 1]
                let gapA = Array(a[(ia + 1)..<nextA])
                let gapB = Array(b[(ib + 1)..<nextB])
                merged.append(contentsOf: gapB.count > gapA.count ? gapB : gapA)
            }
        }

        merged.append(contentsOf: b[(indicesB[indicesB.count - 1] + 1)...])
        return merged
    }

    static func mergeLongestCommonSubsequence(
        _ a: [ParakeetAlignedToken],
        _ b: [ParakeetAlignedToken],
        overlapDuration: Double
    ) -> [ParakeetAlignedToken] {
        if a.isEmpty { return b }
        if b.isEmpty { return a }

        let aEnd = a[a.count - 1].end
        let bStart = b[0].start
        if aEnd <= bStart { return a + b }

        let overlapA = a.filter { $0.end > bStart - overlapDuration }
        let overlapB = b.filter { $0.start < aEnd + overlapDuration }

        if overlapA.count < 2 || overlapB.count < 2 {
            let cutoff = (aEnd + bStart) / 2
            return a.filter { $0.end <= cutoff } + b.filter { $0.start >= cutoff }
        }

        let rows = overlapA.count + 1
        let cols = overlapB.count + 1
        var dp = Array(repeating: Array(repeating: 0, count: cols), count: rows)

        for i in 1..<rows {
            for j in 1..<cols {
                if matches(overlapA[i - 1], overlapB[j - 1], overlapDuration: overlapDuration / 2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = overlapA.count
        var j = overlapB.count
        while i > 0 && j > 0 {
            if matches(overlapA[i - 1], overlapB[j - 1], overlapDuration: overlapDuration / 2) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }
        pairs.reverse()

        if pairs.isEmpty {
            let cutoff = (aEnd + bStart) / 2
            return a.filter { $0.end <= cutoff } + b.filter { $0.start >= cutoff }
        }

        let aStart = a.count - overlapA.count
        let indicesA = pairs.map { aStart + $0.0 }
        let indicesB = pairs.map { $0.1 }

        var merged: [ParakeetAlignedToken] = []
        merged.append(contentsOf: a[..<indicesA[0]])

        for idx in pairs.indices {
            let ia = indicesA[idx]
            let ib = indicesB[idx]
            merged.append(a[ia])

            if idx < pairs.count - 1 {
                let nextA = indicesA[idx + 1]
                let nextB = indicesB[idx + 1]
                let gapA = Array(a[(ia + 1)..<nextA])
                let gapB = Array(b[(ib + 1)..<nextB])
                merged.append(contentsOf: gapB.count > gapA.count ? gapB : gapA)
            }
        }

        merged.append(contentsOf: b[(indicesB[indicesB.count - 1] + 1)...])
        return merged
    }

    private static func shouldCloseSentence(
        token: ParakeetAlignedToken,
        index: Int,
        allTokens: [ParakeetAlignedToken]
    ) -> Bool {
        if token.text.contains("!") || token.text.contains("?")
            || token.text.contains("。") || token.text.contains("？") || token.text.contains("！") {
            return true
        }
        if token.text.contains(".") {
            if index == allTokens.count - 1 { return true }
            return allTokens[index + 1].text.contains(" ")
        }
        return false
    }

    private static func matches(
        _ lhs: ParakeetAlignedToken,
        _ rhs: ParakeetAlignedToken,
        overlapDuration: Double
    ) -> Bool {
        lhs.id == rhs.id && abs(lhs.start - rhs.start) < overlapDuration
    }
}

enum ParakeetAlignmentError: Error {
    case noStrongOverlap
}
