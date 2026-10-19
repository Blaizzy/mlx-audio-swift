// Copyright © 2026 Apple Inc.

import Foundation

/// Parameters controlling how a prompt is prefilled into the ``KVCache``.
///
/// Set via ``GenerateParameters/prefill`` and passed to
/// ``LanguageModel/prepare(_:cache:state:prefill:)``:
///
/// ```swift
/// var parameters = GenerateParameters()
/// parameters.prefill.stepSize = 1024
/// parameters.prefill.progress = { processed, total in ... }
/// ```
public struct PrefillParameters: Sendable {

    /// Ceiling on tokens evaluated per prefill forward. `nil` lets each model pick
    /// its own default (512 for the generic path).
    public var stepSize: Int?

    /// How the prompt is divided into forwards. See ``Chunking``.
    public var chunking: Chunking

    /// Called after each prefill chunk with `(processedPositions, totalPositions)`.
    ///
    /// Positions are the units the model prefills: prompt tokens for text
    /// models, merged embedding positions (after image expansion) for VLMs. A
    /// terminal `(total, total)` is delivered once the prompt is fully
    /// consumed, for every model — including those that prefill in a single
    /// forward. Because chunks are pipelined with `asyncEval`, intermediate
    /// calls report graph submission, slightly ahead of GPU completion.
    ///
    /// The closure is retained for as long as the containing parameters (e.g.
    /// by a stored `GenerateParameters`) — capture weakly anything long-lived.
    public var progress: (@Sendable (_ processed: Int, _ total: Int) -> Void)?

    /// Strategy dividing the prompt into prefill forwards.
    public enum Chunking: Sendable {
        /// The fewest equal chunks that respect the step-size ceiling, so no
        /// forward is disproportionately small at large KV length (the default).
        case balanced

        /// Fixed stride: chunks of exactly the step size. on the default `LLMModel` path the remainder rides along
        /// with the `AutoregressiveIterator`'s first forward; models that reserve one
        /// position for the logits keep that shape and stride the rest.
        case remainder

        /// The whole prompt in one forward — the reference computation that any
        /// chunking approximates. Attention scores are quadratic in the prompt
        /// length and the logits materialize at `[1, L, vocab]`, so this is for
        /// validation at short lengths, not production.
        case unchunked
    }

    public init(
        stepSize: Int? = nil,
        chunking: Chunking = .balanced,
        progress: (@Sendable (_ processed: Int, _ total: Int) -> Void)? = nil
    ) {
        self.stepSize = stepSize
        self.chunking = chunking
        self.progress = progress
    }

    /// The step size used when neither ``stepSize`` nor a model default applies.
    public static let defaultStepSize = 512

}
