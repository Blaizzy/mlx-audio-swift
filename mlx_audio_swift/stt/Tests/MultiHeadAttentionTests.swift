import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioSTT

/// Tests for WhisperMultiHeadAttention
/// Note: Tests requiring MLX array operations need a Metal-capable environment to run.
/// Run with `swift test --filter MultiHeadAttentionTests` on a Mac with Metal support.
struct MultiHeadAttentionTests {

    // MARK: - MLX Array Operation Tests
    // These tests require Metal support to run

    @Test func selfAttentionOutputShape() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 10, 512])  // [batch, seq, dim]

        let (output, _) = attn(x: x)
        #expect(output.shape == [1, 10, 512])
    }

    @Test func crossAttentionOutputShape() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 5, 512])  // Query: [batch, query_seq, dim]
        let xa = MLXArray.zeros([1, 10, 512])  // Key/Value: [batch, kv_seq, dim]

        let (output, _) = attn(x: x, xa: xa)
        #expect(output.shape == [1, 5, 512])
    }

    @Test func crossAttentionReturnsWeights() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 5, 512])
        let xa = MLXArray.zeros([1, 10, 512])

        let (_, weights) = attn(x: x, xa: xa)

        // Cross-attention should return weights for AlignAtt
        #expect(weights != nil)
        #expect(weights!.shape == [1, 8, 5, 10])  // [batch, heads, query_seq, kv_seq]
    }

    @Test func selfAttentionNoWeights() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 10, 512])

        let (_, weights) = attn(x: x)

        // Self-attention doesn't need to return weights
        #expect(weights == nil)
    }

    @Test func maskedAttentionWorks() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        let x = MLXArray.zeros([1, 10, 512])

        // Causal mask for autoregressive decoding
        let mask = WhisperMultiHeadAttention.causalMask(size: 10)

        let (output, _) = attn(x: x, mask: mask)
        #expect(output.shape == [1, 10, 512])
    }

    @Test func causalMaskShape() throws {
        let mask = WhisperMultiHeadAttention.causalMask(size: 5)
        #expect(mask.shape == [5, 5])
    }

    @Test func causalMaskIsUpperTriangularNegInf() throws {
        let mask = WhisperMultiHeadAttention.causalMask(size: 3)
        let values = mask.asArray(Float.self)

        // Diagonal and below should be 0
        #expect(values[0] == 0)  // [0,0]
        #expect(values[3] == 0)  // [1,0]
        #expect(values[4] == 0)  // [1,1]
        #expect(values[6] == 0)  // [2,0]
        #expect(values[7] == 0)  // [2,1]
        #expect(values[8] == 0)  // [2,2]

        // Above diagonal should be large negative
        #expect(values[1] < -1e8)  // [0,1]
        #expect(values[2] < -1e8)  // [0,2]
        #expect(values[5] < -1e8)  // [1,2]
    }

    @Test func kvCacheUpdateAppends() throws {
        let cache = KVCache()

        let keys1 = MLXArray.ones([1, 2, 512])
        let values1 = MLXArray.ones([1, 2, 512])

        let (k1, v1) = cache.update(keys: keys1, values: values1)
        #expect(k1.shape == [1, 2, 512])
        #expect(v1.shape == [1, 2, 512])

        let keys2 = MLXArray.ones([1, 1, 512])
        let values2 = MLXArray.ones([1, 1, 512])

        let (k2, v2) = cache.update(keys: keys2, values: values2)
        #expect(k2.shape == [1, 3, 512])
        #expect(v2.shape == [1, 3, 512])
    }

    @Test func kvCacheReset() throws {
        let cache = KVCache()

        let keys = MLXArray.ones([1, 2, 512])
        let values = MLXArray.ones([1, 2, 512])
        _ = cache.update(keys: keys, values: values)

        cache.reset()

        // After reset, new update should start fresh
        let (k, v) = cache.update(keys: keys, values: values)
        #expect(k.shape == [1, 2, 512])
        #expect(v.shape == [1, 2, 512])
    }

    @Test func headDimCalculation() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        // headDim should be 512 / 8 = 64
        // Verify by checking the weight dimensions of query projection
        #expect(attn.query.weight.shape == [512, 512])
    }

    @Test func keyProjectionHasNoBias() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        // Key projection should have no bias (matching OpenAI Whisper)
        #expect(attn.key.bias == nil)
    }

    @Test func otherProjectionsHaveBias() throws {
        let attn = WhisperMultiHeadAttention(nState: 512, nHead: 8)
        // Query, value, and out projections should have bias
        #expect(attn.query.bias != nil)
        #expect(attn.value.bias != nil)
        #expect(attn.out.bias != nil)
    }
}
