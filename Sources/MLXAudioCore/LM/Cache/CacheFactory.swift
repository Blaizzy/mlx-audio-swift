// Copyright © 2024 Apple Inc.

/// Default cache construction for locally owned attention models.
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try kvHeads.map { _ in try makeAttentionKVCache(configuration: configuration) }
    }
}

extension KVCacheConfiguration.Capacity {
    func makeRotatingCache() -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxTokens, keep: preservedPrefixTokens)
        cache.capacityOrigin = .requested
        return cache
    }
}

public func withPreparedCache<Result>(
    _ cache: [any KVCache],
    lengths: [Int]?,
    _ body: () throws -> Result
) rethrows -> Result {
    guard let lengths else {
        return try body()
    }
    for cache in cache {
        cache.prepare(lengths: lengths)
    }
    defer {
        for cache in cache {
            cache.finalize()
        }
    }
    return try body()
}
