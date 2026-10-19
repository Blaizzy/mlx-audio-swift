// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Layer kinds

/// How a single model layer retains state across generation steps.
///
/// This is the stable, descriptive counterpart to concrete ``KVCache`` instances.
/// Obtain it through ``KVCacheStatus`` rather than inspecting or
/// casting runtime cache objects.
public enum CacheLayerKind: Sendable, Hashable {
    /// Standard or implementation-defined attention KV.
    ///
    /// A `nil` `maxSize` is unbounded. A non-nil value describes a bounded
    /// custom cache that is not a ``RotatingKVCache``.
    case attention(maxSize: Int?)

    /// Sliding / rotating attention window (``RotatingKVCache``).
    ///
    /// Used both for architecture-imposed sliding-window attention and for
    /// caller-requested capacity limits on full-attention layers.
    case rotatingAttention(maxSize: Int, keep: Int)

    /// Fixed-size state-space / Mamba / GDN recurrent state (``MambaCache``).
    ///
    /// These caches are not token-windowed; a ``KVCacheConfiguration/capacity``
    /// policy does not apply to them.
    case stateSpace

    /// Composite per-layer cache (``CacheList``), e.g. Mamba + attention.
    case composite([CacheLayerKind])

    /// Whether this kind stores attention keys/values that can be size-limited.
    public var isAttention: Bool {
        switch self {
        case .attention, .rotatingAttention:
            return true
        case .composite(let children):
            return children.contains(where: \.isAttention)
        case .stateSpace:
            return false
        }
    }

    /// Whether this kind is a bounded attention window.
    public var isBoundedAttention: Bool {
        switch self {
        case .attention(let maxSize):
            return maxSize != nil
        case .rotatingAttention:
            return true
        case .composite(let children):
            return children.contains(where: \.isBoundedAttention)
        case .stateSpace:
            return false
        }
    }

    /// Effective attention token budget, if any.
    public var attentionMaxSize: Int? {
        switch self {
        case .attention(let maxSize):
            return maxSize
        case .rotatingAttention(let maxSize, _):
            return maxSize
        case .composite(let children):
            return children.compactMap(\.attentionMaxSize).max()
        case .stateSpace:
            return nil
        }
    }

    fileprivate var containsStateSpace: Bool {
        switch self {
        case .stateSpace:
            return true
        case .composite(let children):
            return children.contains(where: \.containsStateSpace)
        case .attention, .rotatingAttention:
            return false
        }
    }
}

// MARK: - Unified status

/// Authoritative planned or realized state of a model's KV / recurrent cache.
///
/// Construct this from locally owned caches and an optional owned configuration.
/// It describes topology, capacity, strategy application and model-wide progress.
public struct KVCacheStatus: Sendable, Hashable {
    public enum Phase: Sendable, Hashable {
        /// The cache layout has been constructed for inspection but has not run generation.
        case planned
        /// The status describes cache storage owned by an active session.
        case realized
    }

    /// Relationship between requested capacity and the attention layers to
    /// which that request applies.
    public enum CapacityDisposition: Sendable, Hashable {
        case notRequested
        case fullyApplied
        case partiallyApplied
        case unsupported
        case ignored
    }

    public let phase: Phase
    /// `nil` when no capacity or compression behavior was requested.
    public let requestedConfiguration: KVCacheConfiguration?
    /// Flattened leaf layers. ``KVCacheLayerStatus/path`` preserves composite topology.
    public let layers: [KVCacheLayerStatus]
    /// Authoritative model-wide position for realized session storage.
    public let processedTokenCount: Int?
    public let capacityDisposition: CapacityDisposition
    public let capacityAppliedLayerCount: Int
    public let capacityUnappliedLayerCount: Int

    /// `true` when the topology contains both attention and recurrent state.
    public var isHybrid: Bool {
        layers.contains { $0.kind.isAttention }
            && layers.contains { $0.kind.containsStateSpace }
    }

    /// Per-leaf attention token bounds (`nil` for unbounded or non-attention state).
    public var attentionMaxSizes: [Int?] {
        layers.map { $0.kind.attentionMaxSize }
    }

    public var requestedStrategy: KVCacheStrategyIdentifier? {
        requestedConfiguration?.strategy.identifier
    }

    public var compressedLayerCount: Int {
        layers.count {
            $0.state == .active && $0.resolvedStrategy != .fullPrecision
        }
    }

    public var pendingLayerCount: Int {
        layers.count { $0.state == .pending }
    }

    public var skippedLayerCount: Int {
        layers.count { $0.state == .skipped }
    }

    public init(
        cache: [KVCache],
        configuration: KVCacheConfiguration? = nil,
        phase: Phase = .realized,
        processedTokenCount: Int? = nil
    ) {
        let layers = kvCacheLayerStatuses(cache: cache, configuration: configuration)
        let capacity = Self.capacityOutcome(
            layers: layers,
            capacity: configuration?.capacity)

        self.phase = phase
        self.requestedConfiguration = configuration
        self.layers = layers
        self.processedTokenCount = processedTokenCount
        self.capacityDisposition = capacity.disposition
        self.capacityAppliedLayerCount = capacity.applied
        self.capacityUnappliedLayerCount = capacity.unapplied
    }

    private static func capacityOutcome(
        layers: [KVCacheLayerStatus],
        capacity: KVCacheConfiguration.Capacity?
    ) -> (disposition: CapacityDisposition, applied: Int, unapplied: Int) {
        guard let capacity else { return (.notRequested, 0, 0) }

        var applied = 0
        var unapplied = 0
        for layer in layers where layer.kind.isAttention {

            if layer.capacitySource == .modelDefined {
                continue
            }
            if case .rotatingAttention(let maxSize, let keep) = layer.kind,
                layer.capacitySource == .requested,
                maxSize == capacity.maxTokens,
                keep == capacity.preservedPrefixTokens
            {
                applied += 1
            } else {
                unapplied += 1
            }

        }

        if applied + unapplied == 0 { return (.unsupported, 0, 0) }
        if unapplied == 0 { return (.fullyApplied, applied, 0) }
        if applied == 0 { return (.ignored, 0, unapplied) }
        return (.partiallyApplied, applied, unapplied)
    }
}

// MARK: - Kind inference from concrete caches

extension CacheLayerKind {
    /// Best-effort classification of a concrete cache instance.
    init(cache: KVCache) {
        switch cache {
        case let materialized as MaterializedKVCache:
            self = Self(cache: materialized.storage)
        case let rotating as RotatingKVCache:
            self = .rotatingAttention(maxSize: rotating.maxSize ?? 0, keep: rotating.keepCount)
        case is MambaCache:
            self = .stateSpace
        case let list as CacheList:
            self = .composite(list.children.map { CacheLayerKind(cache: $0) })
        case let arrays as ArraysCache where !(arrays is MambaCache):
            // Generic slot cache used by some SSM variants.
            self = .stateSpace
        default:
            self = .attention(maxSize: cache.maxSize)
        }
    }
}

// MARK: - Construction helpers

/// Build a standard full-attention cache kind that honors the requested capacity.
///
/// - Throws: ``KVCacheConfigurationError`` when the request is invalid.
public func attentionCacheKind(configuration: KVCacheConfiguration?) throws -> CacheLayerKind {
    guard let capacity = configuration?.capacity else {
        return .attention(maxSize: nil)
    }
    return .rotatingAttention(
        maxSize: capacity.maxTokens,
        keep: capacity.preservedPrefixTokens)
}

/// Build a standard full-attention ``KVCache`` that honors the requested capacity.
///
/// - Throws: ``KVCacheConfigurationError`` when the request is invalid.
public func makeAttentionKVCache(configuration: KVCacheConfiguration?) throws -> KVCache {
    if let capacity = configuration?.capacity {
        return capacity.makeRotatingCache()
    }
    return KVCacheSimple()
}

/// Architecture-imposed sliding-window attention kind.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func slidingWindowCacheKind(
    configuration: KVCacheConfiguration?,
    window: Int
) throws -> CacheLayerKind {
    .rotatingAttention(
        maxSize: try validatedSlidingWindow(window),
        keep: 0)
}

/// Architecture-imposed sliding-window attention cache.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func makeSlidingWindowKVCache(configuration: KVCacheConfiguration?, window: Int) throws
    -> KVCache
{
    RotatingKVCache(maxSize: try validatedSlidingWindow(window))
}

/// Full-attention kind if the layer is not sliding-window; otherwise the
/// architecture sliding window.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func hybridAttentionCacheKind(
    configuration: KVCacheConfiguration?,
    slidingWindow: Int?,
    usesSlidingWindow: Bool
) throws -> CacheLayerKind {
    if usesSlidingWindow, let slidingWindow {
        return try slidingWindowCacheKind(configuration: configuration, window: slidingWindow)
    }
    return try attentionCacheKind(configuration: configuration)
}

/// Full-attention cache if the layer is not sliding-window; otherwise the
/// architecture sliding window.
///
/// - Throws: ``KVCacheConfigurationError`` when the request or window is invalid.
public func makeHybridAttentionKVCache(
    configuration: KVCacheConfiguration?,
    slidingWindow: Int?,
    usesSlidingWindow: Bool
) throws -> KVCache {
    if usesSlidingWindow, let slidingWindow {
        return try makeSlidingWindowKVCache(configuration: configuration, window: slidingWindow)
    }
    return try makeAttentionKVCache(configuration: configuration)
}

/// Validate the architecture-defined sliding window.
private func validatedSlidingWindow(_ window: Int) throws -> Int {
    guard window > 0 else {
        throw KVCacheConfigurationError.invalidSlidingWindow(window)
    }
    return window
}
