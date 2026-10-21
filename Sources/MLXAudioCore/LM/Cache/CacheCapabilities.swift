/// Verified operations on a cache. Absence means unsupported, not best-effort.
/// Rewind capabilities still require checking the current offset/isTrimmable.
/// Persistence and batch row insertion/removal are deliberately not advertised.
public struct CacheCapabilities: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
    public static let attention = Self(rawValue: 1 << 0)
    public static let recurrent = Self(rawValue: 1 << 1)
    public static let snapshot = Self(rawValue: 1 << 2)
    public static let fullRewind = Self(rawValue: 1 << 3)
    public static let prefixRewind = Self(rawValue: 1 << 4)
    public static let sequenceLengths = Self(rawValue: 1 << 5)
}

public protocol CacheCapabilityProviding: KVCache {
    var capabilities: CacheCapabilities { get }
}

func cacheCapabilities(_ cache: KVCache) -> CacheCapabilities {
    if let cache = cache as? any CacheCapabilityProviding { return cache.capabilities }
    return []
}
