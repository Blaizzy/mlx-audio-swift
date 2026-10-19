// Copyright © 2026 Apple Inc.

import Foundation

extension PrefillParameters {
    public func resolvedStepSize(
        defaultStepSize: Int = PrefillParameters.defaultStepSize, maximumStepSize: Int? = nil
    ) -> Int {
        max(1, min(stepSize ?? defaultStepSize, maximumStepSize ?? Int.max))
    }

    /// The per-forward chunk length for prefilling `count` positions, or `nil`
    /// when the prompt should go through in a single forward (``Chunking/unchunked``).
    ///
    /// `count` is the number of positions the caller intends to chunk — excluding
    /// any tail it reserves for the first sampled forward. Models pass their own
    /// `defaultStepSize` (used when ``stepSize`` is `nil`) and, if they have one,
    /// a `maximumStepSize` cap such as a sliding-window size.
    public func chunkLength(
        forChunking count: Int,
        defaultStepSize: Int = PrefillParameters.defaultStepSize,
        maximumStepSize: Int? = nil
    ) -> Int? {
        let step = resolvedStepSize(
            defaultStepSize: defaultStepSize, maximumStepSize: maximumStepSize)
        switch chunking {
        case .unchunked:
            return nil
        case .remainder:
            return step
        case .balanced:
            guard count > step else { return max(count, 1) }
            let chunkCount = (count + step - 1) / step
            return (count + chunkCount - 1) / chunkCount
        }
    }

    /// Drives a chunked prefill over `total` positions and returns the number
    /// of positions processed.
    ///
    /// Computes the chunk length once, then calls `body` with each chunk's
    /// range. The driver owns the loop's shared obligations: cooperative
    /// cancellation between chunks (a long prompt must be interruptible — see
    /// the note in `LLMModel.prepare`), an autorelease pool per chunk (long
    /// prompts run hundreds of forwards before any autorelease boundary), and
    /// the per-chunk ``progress`` report. `body` performs the model's forward
    /// and its own cache fencing (`asyncEval`/`eval`).
    ///
    /// `reserving` positions are left unprocessed for the caller's final
    /// forward — the one that produces logits (1 for every current caller).
    /// Under ``Chunking/unchunked`` no chunking happens and 0 is returned:
    /// the caller's final forward takes the whole prompt.
    public func forEachChunk(
        total: Int,
        reserving: Int = 1,
        defaultStepSize: Int = PrefillParameters.defaultStepSize,
        maximumStepSize: Int? = nil,
        _ body: (Range<Int>) throws -> Void
    ) throws -> Int {
        guard
            let chunkLength = chunkLength(
                forChunking: total - reserving,
                defaultStepSize: defaultStepSize, maximumStepSize: maximumStepSize)
        else {
            return 0
        }
        let slack = min(reserving, 1)
        var processed = 0
        while total - processed > reserving {
            try Task.checkCancellation()
            try autoreleasepool {
                let n = min(chunkLength, total - processed - slack)
                try body(processed ..< processed + n)
                processed += n
                progress?(processed, total)
            }
        }
        return processed
    }
}
