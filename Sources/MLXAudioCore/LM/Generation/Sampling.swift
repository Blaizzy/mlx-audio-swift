// Copyright © 2024 Apple Inc.

import MLX
import MLXNN

public protocol LogitSampler {
    func sample(logits: MLXArray) -> MLXArray
}
/// Internal explicit-key capability; public samplers keep their existing interface.
protocol KeyedLogitSampler: LogitSampler {
    func sample(logits: MLXArray, key: MLXArray) -> MLXArray
}

extension ArgMaxSampler: KeyedLogitSampler {
    func sample(logits: MLXArray, key: MLXArray) -> MLXArray { sample(logits: logits) }
}

public protocol LogitProcessor {
    mutating func prompt(_ prompt: MLXArray)
    func process(logits: MLXArray) -> MLXArray
    mutating func didSample(token: MLXArray)
}

public struct ArgMaxSampler: LogitSampler {
    public init() {}

    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
/// to sample the logits.
///
/// Filters are applied in the same order as Python mlx-lm: top_p → min_p → top_k.
/// Each filter operates on the full vocabulary in original token order, masking
/// rejected tokens with `-inf`. This matches the composable filter chain in
/// `mlx_lm.sample_utils.make_sampler`.
public struct TopPSampler: KeyedLogitSampler {
    let temp: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    public init(
        temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0,
        seed: UInt64? = nil
    ) {
        self.temp = MLXArray(temperature)
        if topP > 0 && topP < 1 {
            self.topP = MLXArray(topP)
        } else {
            self.topP = nil
        }
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.negInf = MLXArray(-Float.infinity)
        // A seed makes sampling reproducible; nil keeps the prior
        // entropy-seeded behavior.
        self.randomState = seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        sample(logits: logits, key: randomState.next())
    }

    func sample(logits: MLXArray, key: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        var logprobs = logSoftmax(logits)
        // Preserve ordinary sampling's filter order and temperature placement.
        if let topP { logprobs = applyTopP(logprobs, topP: topP) }
        if let minP { logprobs = applyMinP(logprobs, minP: minP) }
        if let topK { logprobs = applyTopK(logprobs, topK: topK) }
        return categorical(logprobs * (1 / temp), key: key)
    }

    /// Keep tokens whose cumulative probability exceeds `1 - topP` (nucleus sampling).
    /// Matches `apply_top_p` from `mlx_lm/sample_utils.py`.
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Mask low-probability tail in sorted order, scatter back to original vocab order.
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP.
    /// Matches `apply_min_p` from `mlx_lm/sample_utils.py`.
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP)
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens.
    /// Mirrors `apply_top_k` from `mlx_lm/sample_utils.py`.
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        // O(V) partition on negated logprobs so top-k land at [0, topK).
        // Indices at [topK, V) are the tokens to mask out.
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// Sampler that uses `temperature` to sample the logits.
struct CategoricalSampler: KeyedLogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    init(temperature: Float, seed: UInt64? = nil) {
        self.temp = MLXArray(temperature)
        // A seed makes sampling reproducible; nil keeps the prior
        // entropy-seeded behavior.
        self.randomState = seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
    }

    func sample(logits: MLXArray) -> MLXArray {
        sample(logits: logits, key: randomState.next())
    }

    func sample(logits: MLXArray, key: MLXArray) -> MLXArray {
        categorical(logits * (1 / temp), key: key)
    }
}

/// GPU-resident ring buffer of recent token IDs.
///
/// Shared by penalty processors to avoid duplicating ring buffer logic.
/// Uses `MLX.where` mask operations for GPU-only updates (no CPU←GPU sync),
/// preserving `asyncEval()` pipelining in `TokenIterator`.
struct TokenRing {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    /// The valid portion of the ring (all of it once full), or `nil` if empty.
    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    /// Bulk-load from a prompt. Keeps the last `capacity` tokens.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let promptTokens = prompt.asType(.int32).flattened()
        let n = promptTokens.count
        if n <= capacity {
            if n < capacity {
                let padding = MLXArray.zeros([capacity - n], type: Int32.self)
                buffer = concatenated([promptTokens, padding])
            } else {
                buffer = promptTokens
            }
            count = n
            writeIndex = n % capacity
        } else {
            buffer = promptTokens[(-capacity)...]
            count = capacity
            writeIndex = 0
        }
    }

    /// Append a single token using GPU-only mask write (no CPU←GPU sync).
    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Processor that implements a `repetitionPenalty`.
struct RepetitionContext: LogitProcessor {
    private var ring: TokenRing
    let repetitionPenalty: Float

    init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        let broadcastIndices = indices[.newAxis, 0...]
        var selectedLogits = takeAlong(logits, broadcastIndices, axis: -1)

        selectedLogits = MLX.where(
            selectedLogits .< 0, selectedLogits * repetitionPenalty,
            selectedLogits / repetitionPenalty)

        return putAlong(logits, broadcastIndices, values: selectedLogits, axis: -1)
    }

    mutating func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive presence penalty to tokens in a recent context window.
///
/// The penalty is applied once per unique token via scatter-write (writing the
/// same value to the same index multiple times is idempotent).
struct PresencePenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let presencePenalty: Float

    init(presencePenalty: Float, presenceContextSize: Int) {
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        let broadcastIndices = indices[.newAxis, 0...]
        let selectedLogits = takeAlong(logits, broadcastIndices, axis: -1) - presencePenalty
        return putAlong(logits, broadcastIndices, values: selectedLogits, axis: -1)
    }

    mutating func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
///
/// Frequency counting is performed on GPU via `scatter_add` to build a histogram
/// of token occurrences, avoiding CPU←GPU synchronization.
struct FrequencyPenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let frequencyPenalty: Float

    init(frequencyPenalty: Float, frequencyContextSize: Int) {
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    mutating func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }

        let vocabSize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
            .at[validTokens.asType(.int32)].add(ones)

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }

    mutating func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that composes penalty processors in Python mlx-lm order.
struct PenaltyProcessor: LogitProcessor {
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?

    init(
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?
    ) {
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
    }

    mutating func prompt(_ prompt: MLXArray) {
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        return logits
    }

    mutating func didSample(token: MLXArray) {
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
    }
}

extension ModelSamplingOptions {
    func makeSampler() -> any LogitSampler { makeKeyedSampler() }

    func makeKeyedSampler() -> any KeyedLogitSampler {
        if temperature == 0 { return ArgMaxSampler() }
        if (topP > 0 && topP < 1) || topK > 0 || minP > 0 {
            return TopPSampler(
                temperature: temperature, topP: topP, topK: topK, minP: minP, seed: seed)
        }
        return CategoricalSampler(temperature: temperature, seed: seed)
    }
    func makeProcessor() -> PenaltyProcessor {
        PenaltyProcessor(
            repetitionContext: repetition.flatMap {
                $0.value != 0 && $0.contextSize > 0
                    ? RepetitionContext(
                        repetitionPenalty: $0.value, repetitionContextSize: $0.contextSize) : nil
            },
            presenceContext: presence.flatMap {
                $0.value != 0 && $0.contextSize > 0
                    ? PresencePenaltyContext(
                        presencePenalty: $0.value, presenceContextSize: $0.contextSize) : nil
            },
            frequencyContext: frequency.flatMap {
                $0.value != 0 && $0.contextSize > 0
                    ? FrequencyPenaltyContext(
                        frequencyPenalty: $0.value, frequencyContextSize: $0.contextSize) : nil
            })
    }
}
