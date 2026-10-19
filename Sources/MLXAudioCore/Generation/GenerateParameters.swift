import MLX

/// Shared audio-generation settings backed by the locally owned sampling and cache implementations.
public struct GenerateParameters: Sendable {
    public var maxTokens: Int?
    public var maxKVSize: Int?
    public var kvCache: KVCacheConfiguration?
    public var kvBits: Int?
    public var kvGroupSize: Int
    public var quantizedKVStart: Int
    public var kvScheme: String?
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var seed: UInt64?
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    public var presencePenalty: Float?
    public var presenceContextSize: Int
    public var frequencyPenalty: Float?
    public var frequencyContextSize: Int
    public var prefill: PrefillParameters
    public var toolCallPolicy: ToolCallPolicy

    public var prefillStepSize: Int? {
        get { prefill.stepSize }
        set { prefill.stepSize = newValue }
    }

    public init(
        maxTokens: Int? = nil, maxKVSize: Int? = nil,
        kvCache: KVCacheConfiguration? = nil, kvBits: Int? = nil,
        kvGroupSize: Int = 64, quantizedKVStart: Int = 0, kvScheme: String? = nil,
        temperature: Float = 0.6, topP: Float = 1, topK: Int = 0, minP: Float = 0,
        repetitionPenalty: Float? = nil, repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil, presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil, frequencyContextSize: Int = 20,
        prefill: PrefillParameters = .init(), seed: UInt64? = nil,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvCache = kvCache
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.kvScheme = kvScheme
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefill = prefill
        self.seed = seed
        self.toolCallPolicy = toolCallPolicy
    }

    public init(
        maxTokens: Int? = nil, maxKVSize: Int? = nil, kvBits: Int? = nil,
        kvGroupSize: Int = 64, quantizedKVStart: Int = 0, kvScheme: String? = nil,
        temperature: Float = 0.6, topP: Float = 1, topK: Int = 0, minP: Float = 0,
        repetitionPenalty: Float? = nil, repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil, presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil, frequencyContextSize: Int = 20,
        prefillStepSize: Int?, seed: UInt64? = nil
    ) {
        self.init(
            maxTokens: maxTokens, maxKVSize: maxKVSize, kvBits: kvBits,
            kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart, kvScheme: kvScheme,
            temperature: temperature, topP: topP, topK: topK, minP: minP,
            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
            presencePenalty: presencePenalty, presenceContextSize: presenceContextSize,
            frequencyPenalty: frequencyPenalty, frequencyContextSize: frequencyContextSize,
            prefill: .init(stepSize: prefillStepSize), seed: seed)
    }

    private var sampling: ModelSamplingOptions {
        ModelSamplingOptions(
            temperature: temperature, topP: topP, topK: topK, minP: minP, seed: seed,
            repetition: repetitionPenalty.map { .init($0, contextSize: repetitionContextSize) },
            presence: presencePenalty.map { .init($0, contextSize: presenceContextSize) },
            frequency: frequencyPenalty.map { .init($0, contextSize: frequencyContextSize) })
    }

    public func sampler() -> any LogitSampler { sampling.makeSampler() }

    public func processor() -> (any LogitProcessor)? {
        let processor = sampling.makeProcessor()
        guard
            processor.repetitionContext != nil || processor.presenceContext != nil
                || processor.frequencyContext != nil
        else { return nil }
        return processor
    }

    package func resolvedCacheConfiguration() throws -> KVCacheConfiguration? {
        if let kvCache {
            guard maxKVSize == nil, kvBits == nil, kvScheme == nil,
                kvGroupSize == 64, quantizedKVStart == 0
            else {
                throw KVCacheConfigurationError.conflictingLegacyConfiguration
            }
            return kvCache
        }
        let capacity = try maxKVSize.map {
            try KVCacheConfiguration.Capacity(
                maxTokens: $0, preservedPrefixTokens: min(4, max(0, $0 - 1)))
        }
        let strategy: KVCacheConfiguration.Strategy
        if let kvScheme {
            strategy = try cacheStrategy(for: kvScheme, compressionStart: quantizedKVStart)
        } else if let kvBits {
            strategy = .affine(
                try .init(
                    bits: kvBits, groupSize: kvGroupSize, compressionStart: quantizedKVStart))
        } else {
            strategy = .fullPrecision
        }
        guard capacity != nil || strategy != .fullPrecision else { return nil }
        return KVCacheConfiguration(
            capacity: capacity, strategy: strategy, compatibility: .allowPartial)
    }
}

extension LanguageModel {
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let configuration = try parameters?.resolvedCacheConfiguration()
        let cache = try newCache(configuration: configuration)
        if let configuration {
            try validateKVCacheCompatibility(cache, configuration: configuration)
        }
        return cache
    }

    public func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        try cacheStatus(configuration: parameters?.resolvedCacheConfiguration())
    }
}

/// Convert eligible caches after their retained context crosses the compression threshold.
public func maybeQuantizeKVCache(
    cache: inout [KVCache], kvBits: Int?, kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0, kvScheme: String? = nil
) {
    let parameters = GenerateParameters(
        kvBits: kvBits, kvGroupSize: kvGroupSize,
        quantizedKVStart: quantizedKVStart, kvScheme: kvScheme)
    guard let configuration = try? parameters.resolvedCacheConfiguration() else { return }
    _ = try? applyKVCacheConfiguration(cache: &cache, configuration: configuration)
}

private func cacheStrategy(
    for scheme: String, compressionStart: Int
) throws -> KVCacheConfiguration.Strategy {
    switch scheme {
    case "affine4", "affine8":
        return .affine(
            try .init(
                bits: scheme == "affine4" ? 4 : 8, compressionStart: compressionStart))
    case "varn", "varn4v2", "varn4v2t128", "varn4v2t32", "varn4v2t64",
        "varn4", "varn4v4", "varn4v4t32", "varn2", "varn2v2":
        let tileSize = scheme.hasSuffix("t32") ? 32 : scheme.hasSuffix("t64") ? 64 : 128
        let symmetric = ["varn4", "varn4v4", "varn4v4t32"].contains(scheme)
        return .varianceNormalized(
            try .init(
                keyBits: scheme.hasPrefix("varn2") ? 2 : 4,
                valueBits: symmetric ? 4 : 2, tileSize: tileSize, compressionStart: compressionStart
            ))
    default:
        let precisions:
            [String: (
                TurboQuantKVCacheConfiguration.KeyPrecision,
                TurboQuantKVCacheConfiguration.ValuePrecision
            )] = [
                "turbo0v4": (.fp16, .fourBit), "turbo0v3": (.fp16, .threeBit),
                "turbo0v2": (.fp16, .twoBit), "turbo8v4": (.affineEightBit, .fourBit),
                "turbo8v3": (.affineEightBit, .threeBit), "turbo8v2": (.affineEightBit, .twoBit),
                "turbo4": (.fourBit, .fourBit), "turbo4v2": (.fourBit, .twoBit),
                "turbo3": (.threeBit, .threeBit), "turbo2": (.twoBit, .twoBit),
            ]
        guard let precision = precisions[scheme] else {
            throw KVCacheConfigurationError.unsupportedLegacyScheme(scheme)
        }
        return .turboQuant(
            try .init(
                keyPrecision: precision.0, valuePrecision: precision.1,
                compressionStart: compressionStart))
    }
}
