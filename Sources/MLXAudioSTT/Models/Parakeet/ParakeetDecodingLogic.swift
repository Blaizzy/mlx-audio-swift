import Foundation

struct ParakeetDecodingLogic {
    struct RNNTReductionResult: Sendable {
        let nextTime: Int
        let nextNewSymbols: Int
        let emittedToken: Bool
    }

    struct TDTReductionResult: Sendable {
        let nextTime: Int
        let nextNewSymbols: Int
        let jump: Int
        let emittedToken: Bool
    }

    struct CTCSpan: Sendable, Equatable {
        let token: Int
        let startFrame: Int
        let endFrame: Int
    }

    static func rnntStep(
        predictedToken: Int,
        blankToken: Int,
        time: Int,
        newSymbols: Int,
        maxSymbols: Int?
    ) -> RNNTReductionResult {
        if predictedToken == blankToken {
            return RNNTReductionResult(nextTime: time + 1, nextNewSymbols: 0, emittedToken: false)
        }

        let nextSymbols = newSymbols + 1
        if let maxSymbols, nextSymbols >= maxSymbols {
            return RNNTReductionResult(nextTime: time + 1, nextNewSymbols: 0, emittedToken: true)
        }
        return RNNTReductionResult(nextTime: time, nextNewSymbols: nextSymbols, emittedToken: true)
    }

    static func tdtStep(
        predictedToken: Int,
        blankToken: Int,
        decisionIndex: Int,
        durations: [Int],
        time: Int,
        newSymbols: Int,
        maxSymbols: Int?
    ) -> TDTReductionResult {
        let jump = durations.indices.contains(decisionIndex) ? durations[decisionIndex] : 1
        var nextTime = time + jump
        var nextNewSymbols = newSymbols + 1

        if jump != 0 {
            nextNewSymbols = 0
        } else if let maxSymbols, nextNewSymbols >= maxSymbols {
            nextTime += 1
            nextNewSymbols = 0
        }

        return TDTReductionResult(
            nextTime: nextTime,
            nextNewSymbols: nextNewSymbols,
            jump: jump,
            emittedToken: predictedToken != blankToken
        )
    }

    static func ctcSpans(bestTokens: [Int], blankToken: Int) -> [CTCSpan] {
        guard !bestTokens.isEmpty else { return [] }

        var spans: [CTCSpan] = []
        var previous = -1
        var currentStart: Int? = nil

        for (t, token) in bestTokens.enumerated() {
            if token == blankToken {
                if previous != -1, let startFrame = currentStart {
                    spans.append(CTCSpan(token: previous, startFrame: startFrame, endFrame: t))
                    previous = -1
                    currentStart = nil
                }
                continue
            }

            if token == previous {
                continue
            }

            if previous != -1, let startFrame = currentStart {
                spans.append(CTCSpan(token: previous, startFrame: startFrame, endFrame: t))
            }
            previous = token
            currentStart = t
        }

        if previous != -1, let startFrame = currentStart {
            var lastNonBlank = bestTokens.count - 1
            for t in stride(from: bestTokens.count - 1, through: startFrame, by: -1) {
                if bestTokens[t] != blankToken {
                    lastNonBlank = t
                    break
                }
            }
            spans.append(CTCSpan(token: previous, startFrame: startFrame, endFrame: lastNonBlank + 1))
        }

        return spans
    }
}
