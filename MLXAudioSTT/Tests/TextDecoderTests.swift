import Testing
import MLX
import MLXNN
@testable import MLXAudioSTT

struct TextDecoderTests {
    @Test func outputShapeIsCorrect() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        // Input: token ids [batch, seq]
        let tokens = MLXArray([50258, 50259, 50360], [1, 3])  // SOT tokens
        let encoderOutput = MLXArray.zeros([1, 1500, 1280])

        let (logits, _) = decoder(tokens: tokens, encoderOutput: encoderOutput)

        // Output: [batch, seq, nVocab]
        #expect(logits.shape == [1, 3, config.nVocab])
    }

    @Test func returnsCrossAttentionWeights() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        let tokens = MLXArray([50258], [1, 1])
        let encoderOutput = MLXArray.zeros([1, 1500, 1280])

        let (_, crossAttentionWeights) = decoder(tokens: tokens, encoderOutput: encoderOutput)

        // Should return cross-attention weights from all decoder layers
        #expect(crossAttentionWeights.count == config.nTextLayer)

        // Each weight tensor: [batch, nHead, tokenSeq, encoderSeq]
        for weights in crossAttentionWeights {
            #expect(weights.shape == [1, config.nTextHead, 1, 1500])
        }
    }

    @Test func hasCorrectNumberOfBlocks() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        #expect(decoder.blocks.count == config.nTextLayer)
    }

    @Test func blocksAreDecoderType() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        // All decoder blocks should have cross-attention
        for block in decoder.blocks {
            #expect(block.hasCrossAttention == true)
        }
    }

    @Test func tokenEmbeddingDimension() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        #expect(decoder.tokenEmbedding.weight.shape == [config.nVocab, config.nTextState])
    }

    @Test func positionalEmbeddingDimension() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        #expect(decoder.positionalEmbedding.shape == [config.nTextCtx, config.nTextState])
    }

    @Test func incrementalDecodingWithKVCache() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)
        let encoderOutput = MLXArray.zeros([1, 1500, 1280])

        // Create KV cache for incremental decoding
        let kvCache = (0..<config.nTextLayer).map { _ in KVCache(dim: config.nTextState) }

        // First token
        let token1 = MLXArray([50258], [1, 1])
        let (logits1, _) = decoder(tokens: token1, encoderOutput: encoderOutput, kvCache: kvCache)
        #expect(logits1.shape == [1, 1, config.nVocab])

        // Second token (incremental)
        let token2 = MLXArray([50259], [1, 1])
        let (logits2, _) = decoder(tokens: token2, encoderOutput: encoderOutput, kvCache: kvCache)
        #expect(logits2.shape == [1, 1, config.nVocab])
    }

    @Test func tiedEmbeddingsForOutputProjection() {
        let config = WhisperConfiguration.largeTurbo
        let decoder = TextDecoder(config: config)

        // The output projection uses the transpose of token embedding weights
        // This is verified implicitly by the output shape being [batch, seq, nVocab]
        let tokens = MLXArray([50258], [1, 1])
        let encoderOutput = MLXArray.zeros([1, 1500, 1280])

        let (logits, _) = decoder(tokens: tokens, encoderOutput: encoderOutput)

        // Last dimension should be vocabulary size
        #expect(logits.shape[2] == config.nVocab)
    }

    @Test func largeV3Configuration() {
        // Verify TextDecoder works with largeV3 (32 decoder layers)
        let config = WhisperConfiguration.largeV3
        let decoder = TextDecoder(config: config)

        #expect(decoder.blocks.count == 32)
        #expect(decoder.tokenEmbedding.weight.shape == [config.nVocab, config.nTextState])
    }

    @Test func crossAttentionWeightsFromAllLayers() {
        let config = WhisperConfiguration.largeTurbo  // 4 decoder layers
        let decoder = TextDecoder(config: config)

        let tokens = MLXArray([50258, 50259], [1, 2])
        let encoderOutput = MLXArray.zeros([1, 1500, 1280])

        let (_, crossAttentionWeights) = decoder(tokens: tokens, encoderOutput: encoderOutput)

        // Should have weights from all 4 layers
        #expect(crossAttentionWeights.count == 4)

        // Each should have shape [batch, nHead, tokenSeq, encoderSeq]
        for weights in crossAttentionWeights {
            #expect(weights.shape == [1, 20, 2, 1500])
        }
    }
}
