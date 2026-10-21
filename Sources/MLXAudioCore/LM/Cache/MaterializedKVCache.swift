import MLX

/// Keeps attention state materialized for models that reuse stored keys/values or
/// compute attention scores directly. Runtime compression must not replace this cache.
final class MaterializedKVCache: KVCache, CacheCapabilityProviding {
    private(set) var storage: KVCache

    init(_ storage: KVCache) { self.storage = storage }

    var offset: Int { storage.offset }
    var ropeOffset: RoPEOffset { storage.ropeOffset }
    var maxSize: Int? { storage.maxSize }
    var capabilities: CacheCapabilities { cacheCapabilities(storage) }
    var state: [MLXArray] {
        get { storage.state }
        set { storage.state = newValue }
    }
    var metaState: [String] {
        get { storage.metaState }
        set { storage.metaState = newValue }
    }
    var isTrimmable: Bool { storage.isTrimmable }
    func isTrimmable(after positions: Int) -> Bool { storage.isTrimmable(after: positions) }
    func innerState() -> [MLXArray] { storage.innerState() }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        storage.update(keys: keys, values: values)
    }
    func trim(_ n: Int) -> Int { storage.trim(n) }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        storage.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    func copy() -> any KVCache { MaterializedKVCache(storage.copy()) }
    func prepare(lengths: [Int]?) { storage.prepare(lengths: lengths) }
    func prepare(lengths: MLXArray?) { storage.prepare(lengths: lengths) }
    func finalize() { storage.finalize() }
}
