import Foundation
import MLX
import MLXAudioCore
import Testing

@testable import MLXAudioTTS

@Suite("Vendored LLM backbones", .serialized)
struct LLMBackboneTests {
    // Compare CPU/GPU matmuls to detect the reduced float32 precision used by
    // TF32 hardware, following mlx-swift-lm's continuation-test precision probe.
    private static let relativeMatmulError: Float = {
        let a = withRandomState(MLXRandom.RandomState(seed: 0)) { MLXRandom.normal([64, 512]) }
        let b = withRandomState(MLXRandom.RandomState(seed: 1)) { MLXRandom.normal([512, 64]) }
        let gpu = matmul(a, b, stream: .gpu)
        let cpu = matmul(a, b, stream: .cpu)
        return abs(gpu - cpu).max().item(Float.self) / abs(cpu).max().item(Float.self)
    }()

    private static var parityTolerance: Float {
        relativeMatmulError > 1e-5 ? 2e-2 : 1e-4
    }

    @Test func matmulPrecisionProbeIsUnambiguous() {
        #expect(Self.relativeMatmulError < 1e-5 || Self.relativeMatmulError > 1e-4)
    }

    private func makeQwen2(tied: Bool) throws -> Qwen2Model {
        let data = try JSONSerialization.data(withJSONObject: [
            "hidden_size": 8, "num_hidden_layers": 2, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001, "vocab_size": 16,
            "tie_word_embeddings": tied,
            "rope_scaling": ["type": "linear", "factor": 2.0],
        ])
        let config = try JSONDecoder().decode(Qwen2Configuration.self, from: data)
        return withRandomState(MLXRandom.RandomState(seed: 0)) { Qwen2Model(config) }
    }

    @Test(arguments: [false, true])
    func qwen2CachedDecodingMatchesFullForward(tied: Bool) throws {
        let model = try makeQwen2(tied: tied)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped([1, 5])
        let expected = model(tokens, cache: nil)
        let cache = try model.newCache(parameters: nil)
        let prefix = model(tokens[0..., ..<4], cache: cache)
        eval(prefix, cache)
        let actual = model(tokens[0..., 4...], cache: cache)

        #expect(actual.shape == [1, 1, 16])
        #expect(cache.allSatisfy { $0.offset == 5 })
        #expect(abs(actual - expected[0..., 4...]).max().item(Float.self) < Self.parityTolerance)
    }

    @Test(arguments: [false, true])
    func qwen2SanitizesTiedAndQuantizedOutputHeads(tied: Bool) throws {
        let model = try makeQwen2(tied: tied)
        let tensor = MLXArray.ones([1])
        let weights = [
            "model.embed_tokens.weight": tensor,
            "lm_head.weight": tensor,
            "lm_head.scales": tensor,
            "wrapper.lm_head.biases": tensor,
            "other_lm_head.weight": tensor,
            "model.layers.0.self_attn.rotary_emb.inv_freq": tensor,
        ]
        let sanitized = model.sanitize(weights: weights)
        let expected: Set<String> =
            tied
            ? ["model.embed_tokens.weight", "other_lm_head.weight"]
            : Set(weights.keys).subtracting(["model.layers.0.self_attn.rotary_emb.inv_freq"])
        #expect(Set(sanitized.keys) == expected)
    }

    @Test(arguments: [
        PrefillParameters.Chunking.balanced, .remainder, .unchunked,
    ])
    func qwen2PrefillPreservesTailAndMatchesFullForward(chunking: PrefillParameters.Chunking) throws
    {
        let model = try makeQwen2(tied: false)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5])
        let expected = model(tokens[.newAxis], cache: nil)
        let cache = try model.newCache(parameters: nil)
        let prepared = try model.prepare(
            LMInput(tokens: tokens), cache: cache, state: nil,
            prefill: PrefillParameters(stepSize: 2, chunking: chunking))
        guard case .tokens(let tail) = prepared else {
            Issue.record("Prefill must leave the remaining tokens for the caller")
            return
        }

        let remaining = tail.tokens.size
        #expect(remaining > 0)
        #expect(cache.allSatisfy { $0.offset == 5 - remaining })
        let actual = model(tail.tokens[.newAxis], cache: cache)
        #expect(cache.allSatisfy { $0.offset == 5 })
        #expect(
            abs(actual - expected[0..., (5 - remaining)...]).max().item(Float.self)
                < Self.parityTolerance)
    }

    @Test func marvisCachedDecodingMatchesFullForward() {
        let config = CSMLlamaConfiguration(
            hiddenSize: 8, hiddenLayers: 2, intermediateSize: 16, attentionHeads: 2,
            rmsNormEps: 1e-6, vocabularySize: 16, kvHeads: 1,
            maxPositionEmbeddings: 16)
        let model = withRandomState(MLXRandom.RandomState(seed: 0)) { CSMLlamaModel(config) }
        let inputs = MLXArray.arange(40).asType(.float32).reshaped([1, 5, 8]) / 40
        let expected = model(inputs, cache: nil)
        let cache = model.kvHeads.map { _ in KVCacheSimple() }
        let prefix = model(inputs[0..., ..<4], cache: cache)
        eval(prefix, cache)
        let actual = model(inputs[0..., 4...], cache: cache)

        #expect(cache.allSatisfy { $0.offset == 5 })
        #expect(abs(actual - expected[0..., 4...]).max().item(Float.self) < Self.parityTolerance)
    }
}
