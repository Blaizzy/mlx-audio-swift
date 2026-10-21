// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
///
/// See ``KVCache/ropeOffset``.
public enum RoPEOffset {
    case scalar(Int)
    case batch(MLXArray)
}

/// Implementation of KV cache functionality for MLX Swift
///
///
/// ## Quantized Cache Usage
///
/// **Standard caches:**
/// ```swift
/// let cache = KVCacheSimple()
/// let (keys, values) = cache.update(keys: keys, values: values)
/// let output = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, ...)
/// ```
///
/// **Quantized cache:**
/// ```swift
/// let quantizedCache = QuantizedKVCache(groupSize: 64, bits: 4)
/// let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
///
/// let output = quantizedScaledDotProductAttention(
///     queries: queries,
///     quantizedKeys: qKeys,
///     quantizedValues: qValues,
///     scale: scale,
///     mask: mask,
///     groupSize: quantizedCache.groupSize,
///     bits: quantizedCache.bits
/// )
/// ```
///
/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(configuration:)``
public protocol KVCache: Evaluatable {
    /// get the current offset
    var offset: Int { get }

    /// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
    var ropeOffset: RoPEOffset { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// Predict whether this cache can still be trimmed after appending `positions`.
    ///
    /// - Parameter positions: The nonnegative number of sequence positions that
    ///   would be appended.
    /// - Returns: `true` when a subsequent rewind would remain valid.
    func isTrimmable(after positions: Int) -> Bool

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Create an attention mask for this cache
    ///
    /// This method encapsulates cache-specific mask creation logic. Implementations should handle offset capping, window size logic,
    /// and optimization decisions (symbolic vs array masks).
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode

    /// Create an independent deep copy of this cache.
    func copy() -> any KVCache

    /// Prepare cache metadata for a batched sequence.
    func prepare(lengths: [Int]?)

    /// Prepare cache metadata for a batched sequence.
    func prepare(lengths: MLXArray?)

    /// Clear transient cache metadata after generation.
    func finalize()
}

extension KVCache {
    public var ropeOffset: RoPEOffset {
        .scalar(offset)
    }

    public func isTrimmable(after positions: Int) -> Bool {
        isTrimmable
    }

    public func prepare(lengths: [Int]?) {}

    public func prepare(lengths: MLXArray?) {}

    public func finalize() {}
}

/// Protocol for caches that support efficient quantized operations
///
/// **Usage Example:**
/// ```swift
/// // Efficient quantized path
/// if let quantizedCache = cache as? QuantizedKVCacheProtocol {
///     let (qKeys, qValues) = quantizedCache.updateQuantized(keys: k, values: v)
///     // Use native quantized operations
///     let scores = quantizedMM(queries, w: qKeys.0, scales: qKeys.1, biases: qKeys.2, ...)
/// } else {
///     // Regular path
///     let (k, v) = cache.update(keys: k, values: v)
///     let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, ...)
/// }
/// ```
public protocol QuantizedKVCacheProtocol: KVCache {
    /// The quantization group size used
    var groupSize: Int { get }

    /// The number of quantization bits used
    var bits: Int { get }

    /// Quantization mode
    var mode: QuantizationMode { get }

    /// Update cache and return quantized tuples for maximum efficiency
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )

    /// Get current quantized state without updating
    ///
    /// Useful for accessing cached data without adding new tokens.
    /// - Returns: Current quantized state, or nil if cache is empty
    func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))?
}

/// Protocol for caches that can update and compute attention from their own storage layout.
///
/// Used by compressed caches such as ``VarianceNormalizedKVCache`` that keep completed
/// tiles in a non-materialized representation and compute scores/values via cache-native
/// kernels (e.g. `quantizedMM` in a rotated domain).
public protocol KVCacheAttentionProtocol: KVCache {
    /// Update the cache with new K/V tensors and compute attention without first returning a
    /// fully materialized cache tensor pair.
    func updateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}
