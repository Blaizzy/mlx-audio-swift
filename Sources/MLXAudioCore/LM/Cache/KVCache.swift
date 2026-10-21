// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache, CacheCapabilityProviding {
    open var capabilities: CacheCapabilities { [] }
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    /// RoPE offset for this cache. `open` so subclasses can return a non-scalar
    /// offset (e.g. a batched cache's per-row `.batch(...)`).
    open var ropeOffset: RoPEOffset { .scalar(offset) }

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get { [""] }
        set {
            guard newValue.count == 1 && newValue[0].isEmpty else {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    open func isTrimmable(after positions: Int) -> Bool {
        isTrimmable
    }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    open func copy() -> any KVCache {
        fatalError("copy() must be implemented by subclass")
    }

    open func prepare(lengths: [Int]?) {}

    open func prepare(lengths: MLXArray?) {}

    open func finalize() {}

    /// Default implementation for caches without special mask requirements
    open func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // For single token, no mask needed
        if n == 1 {
            return .none
        }

        // For multi-token sequences
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }
}

public func createCausalMask(
    n: Int,
    offset: Int,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    if var lengths {
        lengths = lengths[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rinds .< lengths)
    }

    return mask
}

/// Create an attention mask matching mlx-lm's create_attention_mask helper.
///
/// This returns `.causal` when a symbolic mask is sufficient, avoiding
/// materializing a full mask array.
public func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }

    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }

    return .causal
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also `MultiHeadAttention.createAdditiveCausalMask(_:dtype:)` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

/// Create an attention mask with explicit window size parameter.
///
/// - Parameters:
///   - h: The input array (used to determine sequence length)
///   - cache: Optional single KV cache
///   - windowSize: Optional sliding window size (if provided, creates windowed attention)
///   - returnArray: Force return of array mask instead of symbolic "causal"
/// - Returns: Attention mask mode for scaled dot product attention
public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    // Delegate to cache's makeMask if available
    if let cache = cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    // Fallback for no cache
    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

public func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

/// Standard KV cache implementation based on Python's KVCache
/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
public class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    public override var capabilities: CacheCapabilities {
        [.attention, .snapshot, .fullRewind]
    }

    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    /// Order latent and indexer buffer updates that share a decoding dependency.
    func orderKeys(after other: KVCacheSimple) {
        guard let keys, let otherKeys = other.keys else { return }
        self.keys = depends(input: keys, dependencies: [otherKeys])
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(2)
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to a quantized cache for maximum efficiency.
    ///
    /// Use `updateQuantized()` and `quantizedScaledDotProductAttention()` for zero-overhead operation.
    ///
    /// - Throws: If neither the requested group size nor another supported group size can
    ///   represent both the key and value head dimensions.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) throws -> QuantizedKVCache {
        if let keys = self.keys, let values = self.values {
            // Quantize the current keys and values
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            guard
                let effectiveGroupSize = resolvedKVQuantizationGroupSize(
                    requested: groupSize,
                    keyHeadDim: currentKeys.dim(3),
                    valueHeadDim: currentValues.dim(3)
                )
            else {
                throw KVCacheError(
                    message:
                        "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(currentKeys.dim(3)). Value head dim: \(currentValues.dim(3))."
                )
            }
            let quantizedCache = QuantizedKVCache(groupSize: effectiveGroupSize, bits: bits)
            quantizedCache.offset = self.offset

            let quantizedKeys = quantized(
                currentKeys, groupSize: effectiveGroupSize, bits: bits)
            let quantizedValues = quantized(
                currentValues, groupSize: effectiveGroupSize, bits: bits)

            // Set the quantized state
            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }

            return quantizedCache
        }

        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset
        return quantizedCache
    }

    public override func copy() -> any KVCache {
        let new = KVCacheSimple()
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

/// Rotating KV cache for sliding window attention
public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    public override var capabilities: CacheCapabilities {
        [.attention, .snapshot, .prefixRewind]
    }

    internal enum CapacityOrigin: String {
        case modelNative
        case requested
    }

    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var maxCacheSize: Int
    private var step: Int
    private var idx: Int = 0

    /// In ring layout all rows are live, with `idx...` preceding `keep ..< idx`.
    /// At the end of the buffer the ring is already in temporal order. Otherwise,
    /// temporal layout holds only the first `idx` rows, even when `offset` is larger.
    /// `nil` defers legacy inference until arrays and metadata have both been restored.
    /// Every write or trim resolves it before changing either buffers or counters.
    private var wrappedFlag: Bool? = false

    private var wrapped: Bool {
        wrappedFlag ?? (idx < (keys?.dim(2) ?? 0) && offset > idx)
    }

    /// Model-native sliding-window caches deliberately keep their architectural
    /// window and do not participate in requested-capacity validation.
    internal var capacityOrigin = CapacityOrigin.modelNative

    internal var preservedPrefixTokens: Int { keep }

    public override var maxSize: Int? { maxCacheSize }

    /// Number of leading tokens that are never rotated out of the window.
    var keepCount: Int { keep }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused.
        // `idx` bounds the live rows: after a post-wrap trim the logical `offset`
        // exceeds the rows actually held, so the layout question is `wrapped`,
        // never an `idx`/`offset` comparison.
        if idx == array.dim(2) {
            return array
        } else if wrapped {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    /// The trailing `tail` cache entries in chronological order, without mutating the cache.
    ///
    /// `update(keys:values:)` is the only other way to read a rotating cache's contents, and it
    /// necessarily writes. This is the read-only counterpart: `keys`, `values`, `idx` and
    /// `offset` are all left exactly as they were, so a caller can present the ring's history
    /// alongside K/V it has not committed yet.
    ///
    /// The result is built by the same two steps the multi-token write path uses --
    /// ``temporalOrder(_:)`` to linearize, then the front-trim that preserves the pinned `keep`
    /// prefix -- so a view of length `n` holds exactly the entries a write that front-trimmed to
    /// `n` rows would have presented. When the ring is already chronological (`idx` at the end of
    /// the buffer, which is where every multi-token write leaves it) both steps degrade to
    /// slices and nothing is copied.
    ///
    /// The pinned `keep` prefix is a floor, not just a splice point: a `tail` below it still
    /// comes back, because those entries are not evictable and a view that dropped them would be
    /// a context this ring can never present. With `keep == 0` -- every sliding-window model --
    /// the floor is zero and the length is exactly `min(tail, count)`.
    ///
    /// - Parameter tail: Requested number of trailing entries. Clamped to what the cache holds;
    ///   a negative value is read as zero.
    /// - Returns: `(keys, values)` shaped `[B, kvHeads, n, headDim]` where
    ///   `n == max(min(tail, count), min(keep, count))`, or `nil` before the first write.
    internal func logicalView(tail: Int) -> (MLXArray, MLXArray)? {
        guard let keys = self.keys, let values = self.values else { return nil }

        let orderedKeys = temporalOrder(keys)
        let orderedValues = temporalOrder(values)

        let available = orderedKeys.dim(2)
        // Raising the bound to the pinned prefix is what keeps the front-trim's second slice in
        // range; stated here rather than left to slice clamping, since the length it produces is
        // the documented contract.
        let requested = Swift.min(Swift.max(tail, 0), available)
        let bound = Swift.max(requested, Swift.min(keep, available))
        let trimSize = available - bound
        guard trimSize > 0 else { return (orderedKeys, orderedValues) }

        // `keep == 0` is the sliding-window case (Gemma 3/3n/4, GPT-OSS, Exaone4): no pinned
        // prefix to splice around, so the trailing window is one slice per array.
        if keep == 0 {
            return (
                orderedKeys[.ellipsis, trimSize..., 0...],
                orderedValues[.ellipsis, trimSize..., 0...]
            )
        }
        return (
            trim(trimSize: trimSize, orderedKeys),
            trim(trimSize: trimSize, orderedValues)
        )
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            // Allow temporary cache growth during multi-token processing (e.g., prompt prefill).
            // The largest size is maxCacheSize + S - 1 to ensure
            // every token gets at least maxCacheSize context
            let trimSize = idx - maxCacheSize + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)
        wrappedFlag = false

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)

        // May not have hit the max size yet, so potentially keep growing the cache.
        // Fill is tracked by `idx`, not `offset`: after a post-wrap trim the logical
        // offset exceeds the rows actually held, and growth must resume from the rows.
        let filled = self.keys?.dim(2) ?? 0
        if self.keys == nil || (!wrapped && idx >= filled && filled < maxCacheSize) {
            let newSize = min(step, maxCacheSize - filled)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        // Rotate if we've hit the end
        if idx == maxCacheSize {
            idx = keep
            wrappedFlag = true
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // Return the appropriate cache slice: live rows are bounded by `idx` in
        // temporal layout, while a wrapped ring is fully live.
        if !wrapped, idx < self.keys!.dim(2) {
            return (
                self.keys![.ellipsis, ..<idx, 0...],
                self.values![.ellipsis, ..<idx, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        wrappedFlag = wrapped
        let result =
            if keys.dim(2) == 1 {
                updateInPlace(keys: keys, values: values)
            } else {
                updateConcat(keys: keys, values: values)
            }
        return result
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if !wrapped, idx < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<idx, 0...],
                    values[.ellipsis, ..<idx, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Note: RotatingKVCache doesn't set offset from keys like KVCache does
            // The offset is managed through meta_state
        }
    }

    public override var metaState: [String] {
        get {
            return [
                String(keep), String(maxCacheSize), String(step), String(offset), String(idx),
                capacityOrigin.rawValue, String(wrapped),
            ]
        }
        set {
            guard (5 ... 7).contains(newValue.count) else {
                fatalError("RotatingKVCache metaState must have 5 to 7 values")
            }
            guard let keepVal = Int(newValue[0]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            if newValue[1] == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(newValue[1]) else {
                fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
            if newValue.count >= 6 {
                guard let origin = CapacityOrigin(rawValue: newValue[5]) else {
                    fatalError("Invalid RotatingKVCache capacity origin '\(newValue[5])'")
                }
                self.capacityOrigin = origin
            } else {
                self.capacityOrigin = .modelNative
            }
            if newValue.count == 7 {
                guard let wrappedValue = Bool(newValue[6]) else {
                    fatalError("Invalid RotatingKVCache wrapped flag '\(newValue[6])'")
                }
                self.wrappedFlag = wrappedValue
            } else {
                // Either setter may run first. Defer inference until the restored
                // arrays are available, then freeze the layout before any mutation.
                self.wrappedFlag = nil
            }
        }
    }

    public override var isTrimmable: Bool {
        isTrimmable(after: 0)
    }

    public override func isTrimmable(after positions: Int) -> Bool {
        // This is the *exact rewind* predicate: past the window a trim is merely
        // consistent (see `trim`), because rows the rewound writes overwrote are
        // gone. Consumers that must undo writes exactly -- the staged-round and
        // prompt-cache-reuse machinery -- key off this and fall back to staging,
        // snapshots, or a rebuild once it turns false.
        offset + positions < maxCacheSize
    }

    /// Rewind the newest `n` positions.
    ///
    /// Before the ring wraps this is exact bookkeeping. After it wraps, the ring is
    /// linearized and the newest rows are cut: the cache stays consistent and the
    /// logical offset rewinds, but rows the rewound writes overwrote at the old edge
    /// of the window cannot come back, so the window is up to `n` rows short until
    /// it refills. Callers that need an exact rewind must gate on
    /// ``isTrimmable(after:)`` instead of calling this unconditionally.
    /// Once older rows have been evicted, trimming stops at the pinned `keep` prefix.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0, let keys, let values else { return 0 }
        wrappedFlag = wrapped
        let live = wrapped ? keys.dim(2) : idx
        // A gap between history and live rows means eviction has occurred. Preserve
        // the pinned prefix regardless of layout, including after repeated trims.
        // Without a gap, an exact rewind can still remove any of the original rows.
        let minimum = offset > live ? Swift.min(keep, live) : 0
        let trimmed = Swift.min(n, live - minimum)
        guard trimmed > 0 else { return 0 }
        let bound = live - trimmed

        if wrapped || keys.dim(2) > maxCacheSize {
            // Linearize a ring before cutting its newest rows. Also shrink oversized
            // prefill buffers: the next single-token write compacts those buffers to
            // maxCacheSize and must not treat a discarded suffix as live history.
            self.keys = temporalOrder(keys)[.ellipsis, ..<bound, 0...]
            self.values = temporalOrder(values)[.ellipsis, ..<bound, 0...]
        }
        idx = bound
        offset -= trimmed
        wrappedFlag = false
        return trimmed
    }

    /// Optimized mask creation for rotating cache with offset capping
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            // Multi-token case. The mask must span the rows the write will present:
            // in temporal layout that is the live rows (`idx`), which fall below the
            // logical offset after a post-wrap trim.
            let actualWindowSize = windowSize ?? maxCacheSize
            let liveRows = wrapped ? maxCacheSize : idx
            let cappedOffset = min(maxCacheSize - 1, liveRows)

            // Decide if we need an array mask
            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            // Single token case (n == 1)
            guard let windowSize = windowSize else {
                return .none
            }

            // May need a mask when window_size < max_size and cache has wrapped
            if offset >= windowSize, maxCacheSize > windowSize {
                var currentIdx = idx
                if currentIdx >= maxCacheSize {
                    currentIdx = 0
                }

                let maskSize = (!wrapped && idx < maxCacheSize) ? idx + 1 : maxCacheSize
                let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)

                // Roll the mask to account for rotation
                let rolledMask = roll(mask, shift: currentIdx + 1)

                return .array(rolledMask)
            }
            return .none
        }
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize.description), keep: \(keep), idx: \(idx), wrapped: \(wrapped)"
    }

    public override func copy() -> any KVCache {
        let new = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Restore a round's original ring, including rows overwritten by rejected tokens.
    internal func restoreDraftSnapshot(_ snapshot: RotatingKVCache) {
        keys = snapshot.keys.map { $0[.ellipsis] }
        values = snapshot.values.map { $0[.ellipsis] }
        offset = snapshot.offset
        idx = snapshot.idx
        wrappedFlag = snapshot.wrappedFlag
    }

    /// Convert to a quantized cache.
    ///
    /// Rotating-cache quantization needs a representation that preserves temporal ordering and
    /// rotation metadata. Until that representation exists, callers can recover by retaining the
    /// full-precision rotating cache.
    ///
    /// - Throws: Always, because rotating-cache quantization is not implemented.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) throws -> QuantizedKVCache {
        throw KVCacheError(
            message:
                "RotatingKVCache quantization is not implemented because its temporal ordering requires dedicated handling."
        )
    }
}

func resolvedKVQuantizationGroupSize(
    requested: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Int? {
    let requested = max(1, requested)
    let compatible = [32, 64, 128].filter {
        keyHeadDim.isMultiple(of: $0) && valueHeadDim.isMultiple(of: $0)
    }
    guard !compatible.isEmpty else { return nil }
    return compatible.min { lhs, rhs in
        let lhsDistance = abs(lhs - requested)
        let rhsDistance = abs(rhs - requested)
        if lhsDistance == rhsDistance {
            return lhs < rhs
        }
        return lhsDistance < rhsDistance
    }
}

/// Quantized KV cache for memory efficiency using MLX quantization
public class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    public override var capabilities: CacheCapabilities {
        [.attention, .snapshot, .fullRewind]
    }

    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    public private(set) var groupSize: Int
    public private(set) var bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
        self.mode = mode
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    /// Tree map equivalent for applying function to tuple elements
    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
        -> (T, T, T?)
    {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))

        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    /// Tree map for two tuples (like Python's tree_map over (keys, values))
    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
        _ tuple2: (MLXArray, MLXArray, MLXArray?)
    ) -> ((T, T, T?), (T, T, T?)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    /// Create initial quantized tuples (like Python's init_quant)
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        // Create temporary zero arrays and quantize them using native MLX Swift
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    /// Expand quantized tuple
    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray?
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    /// Update cache and return quantized tuples (Python's update_and_fetch)
    /// This is needed because `update` in Swift must return `(MLXArray, MLXArray)`
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset
        let effectiveGroupSize = resolvedKVQuantizationGroupSize(
            requested: groupSize,
            keyHeadDim: kHeadDim,
            valueHeadDim: vHeadDim
        )
        if let effectiveGroupSize,
            effectiveGroupSize != groupSize,
            self.keys == nil,
            self.values == nil,
            offset == 0
        {
            self.groupSize = effectiveGroupSize
        }
        guard effectiveGroupSize != nil else {
            fatalError(
                "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(kHeadDim). Value head dim: \(vHeadDim)."
            )
        }

        // Check if we need to expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                // Trim if needed
                if prev % step != 0 {
                    // Use tree_map equivalent to trim both keys and values
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                // Expand using tree_map equivalent (Python's tree_map(expand_quant, ...))
                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
            } else {
                // Initialize new quantized cache
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

        // Convert named tuples to positional tuples
        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        // Assign to storage
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        // Update each component of the quantized tuples
        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        if let qKeysBiases = qKeys.2 {
            currentKeys.2![.ellipsis, prev ..< offset, 0...] = qKeysBiases
        }

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        if let qValuesBiases = qValues.2 {
            currentValues.2![.ellipsis, prev ..< offset, 0...] = qValuesBiases
        }

        self.keys = currentKeys
        self.values = currentValues

        // Return quantized tuples
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with QuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    /// Array of keys and values -- this will have either 6 elements or 4 elements (if biases are nil).
    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                // Trim to current offset using tree_map
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                // Flatten tuples to array for serialization
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ].compactMap { $0 }
            } else {
                // Flatten tuples to array for serialization
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
            }
        }
        set {
            switch newValue.count {
            case 4:
                // nil biases case
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }
            guard
                let offset = Int(newValue[1]),
                let groupSize = Int(newValue[2]),
                let bits = Int(newValue[3])
            else {
                fatalError("Failed to convert QuantizedKVCache metaState values to integers")
            }

            self.offset = offset
            self.groupSize = groupSize
            self.bits = bits
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Convert to unquantized cache
    public func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            // Dequantize the current state using tree_map approach
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            // Set the unquantized state
            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

/// Chunked KV cache for processing large contexts in chunks
public class ChunkedKVCache: KVCacheSimple {
    public override var capabilities: CacheCapabilities { [.attention] }

    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    public func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            keys.dim(2) >= chunkSize
        else { return }

        startPosition += keys.dim(2) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)
        let end = offset - startPosition
        self.keys![.ellipsis, prev ..< end, 0...] = keys
        self.values![.ellipsis, prev ..< end, 0...] = values

        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = ChunkedKVCache(chunkSize: chunkSize)
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public override var metaState: [String] {
        get {
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            } else {
                self.chunkSize = Int(newValue[0])
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }
}

/// Base cache for array-based state storage
public class ArraysCache: BaseKVCache {
    public override var capabilities: CacheCapabilities {
        [.recurrent, .snapshot, .sequenceLengths]
    }

    fileprivate var cache: [MLXArray?]
    internal var leftPadding: MLXArray?
    internal var lengths: MLXArray?

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get {
            return cache.compactMap { $0 }
        }
        set {
            cache = newValue.map { $0 as MLXArray? }
        }
    }

    public override func copy() -> any KVCache {
        let new = ArraysCache(size: cache.count)
        copyContents(to: new)
        return new
    }

    internal func copyContents(to new: ArraysCache) {
        new.cache = cache.map { $0?[.ellipsis] }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        new.lengths = self.lengths
    }

    internal var batchSize: Int {
        cache.lazy.compactMap { $0?.dim(0) }.first ?? leftPadding?.size ?? lengths?.size ?? 1
    }

    /// In-place filter to keep just the given indices in the cache
    public func filter(batchIndices: MLXArray) {
        cache = cache.map { c in
            c?[batchIndices]
        }
        leftPadding = leftPadding?[batchIndices]
        lengths = lengths?[batchIndices]
    }

    /// In-place extend this cache with the other cache
    public func extend(other: ArraysCache) {
        let aBatch = batchSize
        let bBatch = other.batchSize

        func concatenate(_ a: MLXArray?, _ b: MLXArray?) -> MLXArray? {
            guard let example = a ?? b else {
                return nil
            }

            let suffixShape = Array(example.shape.dropFirst())
            let dtype = example.dtype
            let lhs = a ?? MLXArray.zeros([aBatch] + suffixShape, dtype: dtype)
            let rhs = b ?? MLXArray.zeros([bBatch] + suffixShape, dtype: dtype)
            return MLX.concatenated([lhs, rhs])
        }

        cache = zip(cache, other.cache).map { c, o in
            concatenate(c, o)
        }
        leftPadding = concatenate(leftPadding, other.leftPadding)
        lengths = concatenate(lengths, other.lengths)
    }

    public override func prepare(lengths: [Int]?) {
        self.lengths = lengths.map { MLXArray($0) }
    }

    public override func prepare(lengths: MLXArray?) {
        self.lengths = lengths
    }

    public override func finalize() {
        lengths = nil
        leftPadding = nil
    }

    public func advance(_ N: Int) {
        if let currentLengths = lengths {
            lengths = currentLengths - N
        }
        if let currentLeftPadding = leftPadding {
            leftPadding = currentLeftPadding - N
        }
    }

    public var currentLengths: MLXArray? {
        lengths
    }

    internal var leftPaddingValues: [Int]? {
        guard let leftPadding else { return nil }
        return leftPadding.asArray(Int.self)
    }

    internal var lengthsValues: [Int]? {
        guard let lengths else { return nil }
        return lengths.asArray(Int.self)
    }

    internal var presentSlotIndices: [Int] {
        cache.enumerated().compactMap { (i, v) in v != nil ? i : nil }
    }

    internal var slotCount: Int { cache.count }

    /// Create attention mask based on left padding or prepared sequence lengths
    public func makeMask(N: Int) -> MLXArray? {
        let positions = MLXArray(0 ..< N)
        if let leftPadding {
            return positions .>= leftPadding[0..., .newAxis]
        } else if let lengths {
            return positions .< lengths[0..., .newAxis]
        } else {
            return nil
        }
    }

    // MARK: - Serialization

    /// metaState format: [slotCount, presentSlots, leftPadding?, lengths?]
    /// Legacy format (BaseKVCache default): [""]
    public override var metaState: [String] {
        get {
            let leftPaddingState = Self.serializeMetadata(leftPadding)
            let lengthsState = Self.serializeMetadata(lengths)
            var result = [
                "\(cache.count)",
                presentSlotIndices.map(String.init).joined(separator: ","),
            ]
            if let leftPaddingState {
                result.append(leftPaddingState)
            } else if lengthsState != nil {
                result.append("")
            }
            if let lengthsState {
                result.append(lengthsState)
            }
            return result
        }
        set {
            assertionFailure(
                "ArraysCache.metaState should not be set directly. Use restoreFromMetaState() instead"
            )
        }
    }

    /// Restore from saved metaState + state arrays. Handles both new (slot-aware) and legacy formats.
    internal func restoreFromMetaState(state: [MLXArray], savedMetaState: [String]) {
        // Detect new format: first element parses as int (slotCount), second element is present slots
        if savedMetaState.count >= 2, let slotCount = Int(savedMetaState[0]) {
            let presentSlots =
                savedMetaState[1].isEmpty
                ? [] : savedMetaState[1].split(separator: ",").compactMap { Int($0) }

            self.cache = Array(repeating: nil, count: slotCount)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < state.count {
                self.cache[slotIdx] = state[arrayIdx]
            }
            self.leftPadding = Self.metadataArray(savedMetaState, at: 2)
            self.lengths = Self.metadataArray(savedMetaState, at: 3)
        } else {
            // Legacy: best-effort, state is compacted
            self.cache = state.map { $0 as MLXArray? }
        }
    }

    private static func serializeMetadata(_ array: MLXArray?) -> String? {
        array?.asArray(Int.self).map(String.init).joined(separator: ",")
    }

    private static func metadataArray(_ state: [String], at index: Int) -> MLXArray? {
        guard state.indices.contains(index), !state[index].isEmpty else { return nil }
        return MLXArray(state[index].split(separator: ",").compactMap { Int($0) })
    }
}

/// Simple cache for Mamba-style state space models
public class MambaCache: ArraysCache {
    private struct SpeculativeCheckpoint {
        var state: [MLXArray?]
        var offset: Int
        var leftPadding: MLXArray?
        var lengths: MLXArray?
    }

    private var speculativeCheckpoint: SpeculativeCheckpoint?

    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }

    /// History exists only for the active verification block. Prefill can request
    /// just its final prefix boundary instead of retaining every recurrent state.
    private struct SpeculativeHistory {
        let generation: Int
        let length: Int
        let positions: [Int]
        let initial: [MLXArray?]
        let offset: Int
        let leftPadding: MLXArray?
        let lengths: MLXArray?
        var convolution: (source: MLXArray, width: Int)?
        var recurrent: [Int: MLXArray] = [:]
    }
    private var speculativeHistory: SpeculativeHistory?
    private var historyGeneration = 0

    internal var speculativeStatePositions: [Int]? { speculativeHistory?.positions }

    internal func beginSpeculation(length: Int, prefixOnly: Bool = false) throws -> Int {
        guard length > 0, speculativeHistory == nil, !hasSpeculativeCheckpoint else {
            throw KVCacheError(
                message:
                    "invalid or overlapping recurrent cache transaction")
        }
        historyGeneration += 1
        speculativeHistory = SpeculativeHistory(
            generation: historyGeneration, length: length,
            positions: prefixOnly ? (length > 1 ? [length - 1] : []) : Array(1 ..< length),
            initial: cache, offset: offset, leftPadding: leftPadding, lengths: lengths)
        return historyGeneration
    }

    internal func recordSpeculativeWindow(_ source: MLXArray, width: Int) {
        guard let history = speculativeHistory else { return }
        precondition(source.dim(1) == history.length + width)
        speculativeHistory?.convolution = (source, width)
    }

    internal func recordSpeculativeRecurrent(_ states: MLXArray?, final: MLXArray) {
        guard let history = speculativeHistory else { return }
        precondition(history.positions.count == (states?.dim(1) ?? 0))
        for (index, position) in history.positions.enumerated() {
            speculativeHistory?.recurrent[position] = states![0..., index]
        }
        speculativeHistory?.recurrent[history.length] = final
    }

    internal func validateSpeculation(retaining count: Int, generation: Int) throws {
        guard let history = speculativeHistory, history.generation == generation,
            (0 ... history.length).contains(count),
            count == 0 || (history.convolution != nil && history.recurrent[count] != nil)
        else {
            throw KVCacheError(
                message:
                    "stale or incomplete recurrent cache transaction")
        }
    }

    internal func commitSpeculation(retaining count: Int, generation: Int) throws {
        try validateSpeculation(retaining: count, generation: generation)
        let history = speculativeHistory!
        if count == 0 {
            cache = history.initial
        } else {
            let window = history.convolution!
            cache[0] = contiguous(window.source[0..., count ..< (count + window.width), 0...])
            let selected = history.recurrent[count]!
            // A slice would pin the entire block's state history in every
            // saved session checkpoint. Gather into one standalone state.
            cache[1] =
                count == history.length
                ? selected
                : take(
                    selected, MLXArray((0 ..< selected.dim(0)).map(Int32.init)), axis: 0)
        }
        offset = history.offset
        leftPadding = history.leftPadding.map { $0 - count }
        lengths = history.lengths.map { $0 - count }
        speculativeHistory = nil
    }

    internal func abortSpeculation(generation: Int) {
        guard let history = speculativeHistory, history.generation == generation else { return }
        cache = history.initial
        offset = history.offset
        leftPadding = history.leftPadding
        lengths = history.lengths
        speculativeHistory = nil
    }

    /// Save the recurrent state at the last unconditionally committed token
    /// inside a speculative verification pass.
    internal func saveSpeculativeCheckpoint(
        convState: MLXArray,
        recurrentState: MLXArray,
        advancedBy tokenCount: Int
    ) {
        speculativeCheckpoint = SpeculativeCheckpoint(
            state: [convState, recurrentState],
            offset: offset,
            leftPadding: leftPadding.map { $0 - tokenCount },
            lengths: lengths.map { $0 - tokenCount })
    }

    internal var hasSpeculativeCheckpoint: Bool {
        speculativeCheckpoint != nil
    }

    @discardableResult
    internal func restoreSpeculativeCheckpoint() -> Bool {
        guard let checkpoint = speculativeCheckpoint else { return false }
        cache = checkpoint.state
        offset = checkpoint.offset
        leftPadding = checkpoint.leftPadding
        lengths = checkpoint.lengths
        speculativeCheckpoint = nil
        return true
    }

    internal func discardSpeculativeCheckpoint() {
        speculativeCheckpoint = nil
    }

    public override func copy() -> any KVCache {
        let new = MambaCache()
        copyContents(to: new)
        return new
    }
}

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    public override var capabilities: CacheCapabilities {
        let children = caches.map(cacheCapabilities)
        var result: CacheCapabilities = []
        for kind: CacheCapabilities in [.attention, .recurrent] {
            if children.contains(where: { $0.contains(kind) }) { result.insert(kind) }
        }
        for operation: CacheCapabilities in [.snapshot, .fullRewind, .sequenceLengths] {
            if children.allSatisfy({ $0.contains(operation) }) { result.insert(operation) }
        }
        if children.allSatisfy({ !$0.intersection([.fullRewind, .prefixRewind]).isEmpty }) {
            result.insert(.prefixRewind)
        }
        return result
    }

    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    /// Internal initializer for reconstruction from deserialized children.
    internal init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override func copy() -> any KVCache {
        let copiedCaches = caches.map { $0.copy() }
        let new = CacheList(caches: copiedCaches)
        return new
    }

    /// Recursively apply a transformation to every non-composite child cache.
    ///
    /// `CacheList` children are descended into; any other cache is passed to
    /// `transform` and replaced by the returned value. This is the primitive
    /// used by dynamic cache quantization and other cache-wide rewrites for
    /// models with hybrid attention/recurrent caches (e.g. Falcon-H1).
    public func mapChildren(_ transform: (KVCache) -> KVCache) {
        caches = caches.map { child in
            if let list = child as? CacheList {
                list.mapChildren(transform)
                return list
            }
            return transform(child)
        }
    }

    /// Rewrite every non-composite child while preserving its stable tree path.
    func rewriteLeaves(
        path: [Int],
        using transform: (KVCacheLeaf) -> KVCache
    ) {
        caches = caches.enumerated().map { index, child in
            let childPath = path + [index]
            if let list = child as? CacheList {
                list.rewriteLeaves(path: childPath, using: transform)
                return list
            }
            let leaf = KVCacheLeaf(path: childPath, cache: child)
            return transform(leaf)
        }
    }

    public override func prepare(lengths: [Int]?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func prepare(lengths: MLXArray?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func finalize() {
        caches.forEach { $0.finalize() }
    }

    public override var isTrimmable: Bool {
        isTrimmable(after: 0)
    }

    public override func isTrimmable(after positions: Int) -> Bool {
        caches.allSatisfy { $0.isTrimmable(after: positions) }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }

    /// Internal accessor for child caches (used by serialization and policy reporting).
    internal var children: [KVCache] { caches }

    // MARK: - Serialization

    /// metaState format: [childCount, (className, stateCount, metaStateCount, ...metaState)*]
    ///
    /// Like Python's CacheList.meta_state which returns [child_class_names, child_meta_states],
    /// but flattened for Swift's [String] format.
    public override var metaState: [String] {
        get {
            var result = ["\(caches.count)"]
            for cache in caches {
                let className = cacheClassName(cache)
                let meta = cache.metaState
                result.append(className)
                result.append("\(cache.state.count)")
                result.append("\(meta.count)")
                result.append(contentsOf: meta)
            }
            return result
        }
        set {
            assertionFailure(
                "CacheList serialized restoration is unsupported. Use copy() for in-memory snapshots."
            )
        }
    }

}

// MARK: - Error Types

struct KVCacheError: Error, LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

// MARK: - Utility Functions

/// Map a cache instance to its Python-compatible class name for serialization.
private func cacheClassName(_ cache: KVCache) -> String {
    switch cache {
    case is ChunkedKVCache: return "ChunkedKVCache"
    case is MambaCache: return "MambaCache"
    case is ArraysCache: return "ArraysCache"
    case is RotatingKVCache: return "RotatingKVCache"
    case is VarianceNormalizedKVCache: return "VarianceNormalizedKVCache"
    case is QuantizedKVCache: return "QuantizedKVCache"
    case is TurboQuantKVCache: return "TurboQuantKVCache"
    case is KVCacheSimple: return "KVCache"
    case is CacheList: return "CacheList"
    default: return "KVCache"
    }
}

/// Check if model's cache can be trimmed.
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    cache.dropFirst().forEach { $0.trim(numTokens) }
    return cache.first?.trim(numTokens) ?? 0
}

// MARK: - Type Aliases

// MARK: - Quantized Attention Operations

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine
) -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Apply mask
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, MLXArray.maskFill(for: scores.dtype))

    case .array(let maskArray):
        scores = applyMask(maskArray, to: scores)

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            scores = applyMask(maskArray, to: scores)
        }

    case .none:
        break
    }

    let attentionWeights = softmax(scores, axis: -1)

    // Compute output using quantized matmul
    var output = quantizedMM(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output

    // Apply a boolean/additive mask, broadcasting batched masks over the GQA
    // head-group axis: per-sequence masks are `[B, 1, L, S]`, but with
    // `nRepeats > 1` the scores are 5-D `[B, nKVHeads, nRepeats, L, S]`, so a
    // 4-D mask needs an extra axis to line up `B` with the batch dimension.
    func applyMask(_ maskArray: MLXArray, to scores: MLXArray) -> MLXArray {
        var maskArray = maskArray
        if nRepeats > 1 && maskArray.ndim == 4 {
            maskArray = expandedDimensions(maskArray, axis: -3)
        }
        if maskArray.dtype == .bool {
            return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
        } else {
            return scores + maskArray
        }
    }
}

// MARK: - Dynamic Cache Quantization

@discardableResult
func maybeAffineQuantizeKVCache(
    cache: inout [KVCache],
    bits: Int,
    groupSize: Int,
    compressionStart: Int
) -> Bool {
    var awaitsCompressionStart = false
    KVCacheTree.rewrite(&cache) { leaf in
        guard case .simple(let simple) = leaf.kind else { return leaf.cache }
        guard simple.offset > compressionStart else {
            awaitsCompressionStart = true
            return simple
        }

        guard let quantized = try? simple.toQuantized(groupSize: groupSize, bits: bits) else {
            return simple
        }
        return quantized
    }
    return !awaitsCompressionStart
}

@discardableResult
func maybeVarianceNormalizeKVCache(
    cache: inout [KVCache],
    keyBits: Int,
    valueBits: Int,
    tileSize: Int,
    sinkhornIterations: Int,
    compressionStart: Int
) -> Bool {
    var awaitsCompressionStart = false
    KVCacheTree.rewrite(&cache) { leaf in
        guard case .simple(let simple) = leaf.kind else { return leaf.cache }
        guard simple.offset > compressionStart else {
            awaitsCompressionStart = true
            return simple
        }

        let state = simple.innerState()
        if state.count >= 2 {
            guard
                supportsVarianceNormalizedKVCache(
                    keyHeadDim: state[0].dim(3),
                    valueHeadDim: state[1].dim(3),
                    tileSize: tileSize)
            else {
                return simple
            }
        }

        return simple.toVarianceNormalized(
            tileSize: tileSize,
            keyBits: keyBits,
            valueBits: valueBits,
            sinkhornIterations: sinkhornIterations)
    }
    return !awaitsCompressionStart
}

// MARK: - Attention Helpers

/// Apply a symbolic or array attention mask to score logits.
func applyAttentionMask(
    scores: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        return MLX.where(causalMask, scores, MLXArray.maskFill(for: scores.dtype))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
        } else {
            return scores + maskArray
        }

    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                return MLX.where(maskArray, scores, MLXArray.maskFill(for: scores.dtype))
            } else {
                return scores + maskArray
            }
        }
        return scores

    case .none:
        return scores
    }
}

func attentionScores(
    queries: MLXArray,
    keys: MLXArray,
    scale: Float
) -> MLXArray {
    let (batchSize, queryHeadCount, queryLength, headDim) = (
        queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
    )
    let kvHeadCount = keys.dim(1)
    let repeats = queryHeadCount / kvHeadCount
    let scaledQueries = queries * scale

    if repeats > 1 {
        let groupedQueries = scaledQueries.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, headDim,
        ])
        let groupedKeys = expandedDimensions(keys, axis: -3)
        return matmul(groupedQueries, groupedKeys.transposed(0, 1, 2, 4, 3))
            .reshaped(batchSize, queryHeadCount, queryLength, keys.dim(2))
    } else {
        return matmul(scaledQueries, keys.transposed(0, 1, 3, 2))
    }
}

func attentionValues(
    weights: MLXArray,
    values: MLXArray,
    queryHeadCount: Int
) -> MLXArray {
    let (batchSize, _, queryLength, keyLength) = (
        weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(3)
    )
    let kvHeadCount = values.dim(1)
    let repeats = queryHeadCount / kvHeadCount

    if repeats > 1 {
        let groupedWeights = weights.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, keyLength,
        ])
        let groupedValues = expandedDimensions(values, axis: -3)
        return matmul(groupedWeights, groupedValues)
            .reshaped(batchSize, queryHeadCount, queryLength, values.dim(3))
    } else {
        return matmul(weights, values)
    }
}
