import Foundation
import MLX
import Testing

@testable import MLXAudioCore

@Suite("Local language-model support", .serialized)
struct LMCommonTests {
    @Test func samplingPreservesGreedyAndSeededBehavior() {
        let logits = MLXArray([Float(-4), -1, 3, 1]).reshaped([1, 4])
        #expect(
            GenerateParameters(temperature: 0).sampler().sample(logits: logits).item(Int.self) == 2)

        let parameters = GenerateParameters(
            temperature: 0.8, topP: 0.95, topK: 2, minP: 0.01, seed: 42)
        let first = parameters.sampler()
        let second = parameters.sampler()
        let firstTokens = (0 ..< 16).map { _ in first.sample(logits: logits).item(Int.self) }
        let secondTokens = (0 ..< 16).map { _ in second.sample(logits: logits).item(Int.self) }
        #expect(firstTokens == secondTokens)
        #expect(firstTokens.allSatisfy { $0 == 2 || $0 == 3 })
        #expect(GenerateParameters().processor() == nil)
    }

    @Test func penaltiesComposeAndAdvanceTheirContext() throws {
        let parameters = GenerateParameters(
            repetitionPenalty: 2, repetitionContextSize: 3,
            presencePenalty: 0.5, presenceContextSize: 3,
            frequencyPenalty: 0.25, frequencyContextSize: 3)
        var processor = try #require(parameters.processor())
        processor.prompt(MLXArray([Int32(1), 1, 2]))
        let logits = MLXArray([Float(-2), -4, 8, 1]).reshaped([1, 4])
        #expect(processor.process(logits: logits).asArray(Float.self) == [-2, -9, 3.25, 1])
        processor.didSample(token: MLXArray(Int32(2)))
        #expect(processor.process(logits: logits).asArray(Float.self) == [-2, -8.75, 3, 1])
    }

    @Test func configurationPreservesMixedQuantizationAndJSONValues() throws {
        let data = Data(
            """
            {
              "model_type": "qwen2", "text_config": {"eos_token_id": [2, 3]},
              "quantization": {
                "group_size": 64, "bits": 4,
                "model.layers.0": {"group_size": 32, "bits": 8},
                "model.norm": false
              }
            }
            """.utf8)
        let configuration = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let quantization = try #require(configuration.perLayerQuantization)
        #expect(quantization.quantization(layer: "model.layers.0")?.groupSize == 32)
        #expect(quantization.quantization(layer: "model.layers.0")?.bits == 8)
        #expect(quantization.quantization(layer: "model.layers.1")?.bits == 4)
        #expect(quantization.quantization(layer: "model.norm") == nil)
        #expect(configuration.effectiveEOSTokenIds == [2, 3])
        let roundTrip = try JSONDecoder().decode(
            BaseConfiguration.self, from: JSONEncoder().encode(configuration))
        #expect(roundTrip.perLayerQuantization?.quantization(layer: "model.layers.0")?.bits == 8)

        let values = try JSONDecoder().decode(
            [String: StringOrNumber].self,
            from: Data(#"{"type":"linear","factor":2.5,"sections":[1,2,3],"enabled":true}"#.utf8))
        #expect(values["factor"]?.asFloat() == 2.5)
        #expect(values["sections"]?.asInts() == [1, 2, 3])
        #expect(values["enabled"] == .bool(true))
    }

    @Test func dynamicQuantizationPreservesThresholdOffsetAndHeadDimensions() throws {
        var cache: [any KVCache] = [KVCacheSimple()]
        let keys = MLXArray.ones([1, 1, 2, 32])
        _ = cache[0].update(keys: keys, values: keys)
        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 2)
        #expect(cache[0] is KVCacheSimple)

        let next = MLXArray.ones([1, 1, 1, 32])
        _ = cache[0].update(keys: next, values: next)
        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 2)
        let quantized = try #require(cache[0] as? QuantizedKVCache)
        #expect(quantized.offset == 3)
        #expect(quantized.groupSize == 32)
        #expect(quantized.bits == 8)
        #expect(quantized.getQuantizedState() != nil)
    }

    @Test func quantizedAttentionSupportsBatchedGroupedQueriesAndBooleanMasks() {
        let queries = sin(MLXArray.arange(2 * 4 * 3 * 32).asType(.float32)).reshaped([2, 4, 3, 32])
        let keys = cos(MLXArray.arange(2 * 2 * 3 * 32).asType(.float32)).reshaped([2, 2, 3, 32])
        let values = sin(keys)
        let mask = broadcast(createCausalMask(n: 3, offset: 0), to: [2, 1, 3, 3])
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 0.125, mask: .array(mask))
        let cache = QuantizedKVCache(groupSize: 32, bits: 8)
        let actual = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: 0.125, mask: .array(mask))
        #expect(actual.shape == expected.shape)
        #expect(cache.offset == 3)
        #expect(abs(actual - expected).max().item(Float.self) < 0.03)
    }

    @Test func recurrentCacheCopiesPreserveIndependentState() throws {
        let cache = MambaCache()
        cache[0] = MLXArray.ones([1, 2, 4])
        let copy = try #require(cache.copy() as? MambaCache)
        cache[0] = MLXArray.zeros([1, 2, 4])
        #expect(copy[0]?.sum().item(Float.self) == 8)
        #expect(copy[1] == nil)
        #expect(cache[0]?.sum().item(Float.self) == 0)
    }

    @Test func cacheParametersPreserveLegacySchemesAndRejectConflictingSettings() throws {
        let affine = try #require(try GenerateParameters(kvBits: 4).resolvedCacheConfiguration())
        #expect(affine.strategy.identifier == .affine)
        #expect(affine.compatibility == .allowPartial)
        let turbo = try #require(
            try GenerateParameters(kvScheme: "turbo8v3").resolvedCacheConfiguration())
        #expect(turbo.strategy.identifier == .turboQuant)
        let variance = try #require(
            try GenerateParameters(kvScheme: "varn4v2t64").resolvedCacheConfiguration())
        #expect(variance.strategy.identifier == .varianceNormalized)
        let bounded = try #require(
            try GenerateParameters(maxKVSize: 3).resolvedCacheConfiguration())
        #expect(bounded.capacity?.maxTokens == 3)
        #expect(bounded.capacity?.preservedPrefixTokens == 2)
        #expect(throws: KVCacheConfigurationError.conflictingLegacyConfiguration) {
            try GenerateParameters(kvCache: .init(), kvBits: 4).resolvedCacheConfiguration()
        }
        #expect(throws: KVCacheConfigurationError.unsupportedLegacyScheme("unknown")) {
            try GenerateParameters(kvScheme: "unknown").resolvedCacheConfiguration()
        }
    }
}
