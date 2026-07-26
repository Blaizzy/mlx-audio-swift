//  Whisper uses decoder.embed_tokens as both the input embedding and output
//  projection. These tests pin the module-routed projection for plain and
//  quantized checkpoints so packed 4-bit weights are never multiplied as if
//  they were dense floating-point values.

import MLX
import MLXNN
import Testing

@testable import MLXAudioSTT

struct WhisperQuantizedTiedEmbeddingTests {
    private static func smallConfig() -> WhisperConfig {
        WhisperConfig(
            vocabSize: 96,
            numMelBins: 80,
            dModel: 64,
            encoderLayers: 1,
            encoderAttentionHeads: 2,
            encoderFfnDim: 128,
            maxSourcePositions: 32,
            decoderLayers: 1,
            decoderAttentionHeads: 2,
            decoderFfnDim: 128,
            maxTargetPositions: 32
        )
    }

    @Test func tiedProjectionMatchesDenseEmbeddingWeight() {
        let decoder = WhisperDecoder(config: Self.smallConfig())
        let weight = decoder.embedTokens.weight
        let hidden = MLXRandom.normal([weight.shape[1]]).asType(weight.dtype)
        let expected = MLX.matmul(hidden, weight.transposed(1, 0))

        let logits = decoder.projectToVocab(hidden)

        #expect(logits.shape == expected.shape)
        #expect(MLX.abs(logits - expected).max().item(Float.self) < 1e-5)
    }

    @Test func tiedProjectionDequantizesPackedEmbedding() {
        let decoder = WhisperDecoder(config: Self.smallConfig())
        quantize(model: decoder, groupSize: 32, bits: 4) { path, module in
            path == "embed_tokens" && module is Embedding
        }
        guard let quantized = decoder.embedTokens as? QuantizedEmbedding else {
            Issue.record("embed_tokens was not replaced by a QuantizedEmbedding")
            return
        }

        let dense = dequantized(
            quantized.weight,
            scales: quantized.scales,
            biases: quantized.biases,
            groupSize: quantized.groupSize,
            bits: quantized.bits,
            mode: quantized.mode
        )
        let hidden = MLXRandom.normal([dense.shape[1]]).asType(dense.dtype)
        let expected = MLX.matmul(hidden, dense.transposed(1, 0))

        let logits = decoder.projectToVocab(hidden)

        #expect(logits.shape == expected.shape)
        #expect(MLX.abs(logits - expected).max().item(Float.self) < 1e-2)
    }
}
