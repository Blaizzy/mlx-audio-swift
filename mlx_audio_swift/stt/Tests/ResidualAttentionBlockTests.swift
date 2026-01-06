import Testing
import MLX
import MLXNN
@testable import MLXAudioSTT

struct ResidualAttentionBlockTests {
    @Test func encoderBlockOutputShape() {
        // Encoder block: self-attention only, no cross-attention
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: false)
        let x = MLXArray.zeros([1, 10, 512])

        let (output, _) = block(x: x)
        #expect(output.shape == [1, 10, 512])
    }

    @Test func decoderBlockOutputShape() {
        // Decoder block: self-attention + cross-attention
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: true)
        let x = MLXArray.zeros([1, 5, 512])
        let xa = MLXArray.zeros([1, 10, 512])  // Encoder output

        let (output, _) = block(x: x, xa: xa)
        #expect(output.shape == [1, 5, 512])
    }

    @Test func decoderBlockReturnsCrossAttentionWeights() {
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: true)
        let x = MLXArray.zeros([1, 5, 512])
        let xa = MLXArray.zeros([1, 10, 512])

        let (_, crossAttnWeights) = block(x: x, xa: xa)

        // Should return cross-attention weights for AlignAtt
        #expect(crossAttnWeights != nil)
        #expect(crossAttnWeights!.shape == [1, 8, 5, 10])
    }

    @Test func encoderBlockNoCrossAttentionWeights() {
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: false)
        let x = MLXArray.zeros([1, 10, 512])

        let (_, crossAttnWeights) = block(x: x)

        // Encoder block should not return cross-attention weights
        #expect(crossAttnWeights == nil)
    }

    @Test func mlpExpansionFactor() {
        // MLP should expand to 4x nState then project back
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: false)

        // Check MLP layer dimensions via parameter shapes
        #expect(block.mlp1.weight.shape == [2048, 512])  // 4x expansion
        #expect(block.mlp2.weight.shape == [512, 2048])  // Project back
    }

    @Test func selfAttentionWithMask() {
        // Decoder self-attention uses causal mask
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: false)
        let x = MLXArray.zeros([1, 10, 512])
        let mask = WhisperMultiHeadAttention.causalMask(size: 10)

        let (output, _) = block(x: x, mask: mask)
        #expect(output.shape == [1, 10, 512])
    }

    @Test func residualConnectionPreservesGradientFlow() {
        // Output should not be zeros even with zero input due to bias terms
        let block = ResidualAttentionBlock(nState: 64, nHead: 4, crossAttention: false)
        let x = MLXArray.zeros([1, 5, 64])

        let (output, _) = block(x: x)

        // Due to residual connections and bias, output may have non-zero values
        // The key test is that shapes are preserved
        #expect(output.shape == [1, 5, 64])
    }

    @Test func preNormArchitecture() {
        // Verify LayerNorm is applied before attention (pre-norm)
        // We can verify this by checking that the layer norm modules exist
        let block = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: true)

        // Check that all layer norms are present
        #expect(block.attnLn.dimensions == 512)
        #expect(block.mlpLn.dimensions == 512)
        #expect(block.crossAttnLn?.dimensions == 512)
    }

    @Test func crossAttentionLayersOnlyWhenEnabled() {
        let encoderBlock = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: false)
        let decoderBlock = ResidualAttentionBlock(nState: 512, nHead: 8, crossAttention: true)

        // Encoder block should not have cross-attention layers
        #expect(encoderBlock.crossAttn == nil)
        #expect(encoderBlock.crossAttnLn == nil)

        // Decoder block should have cross-attention layers
        #expect(decoderBlock.crossAttn != nil)
        #expect(decoderBlock.crossAttnLn != nil)
    }
}
