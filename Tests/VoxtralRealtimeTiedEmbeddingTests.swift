//  The Voxtral Realtime tied embedding doubles as the LM head. These tests pin the
//  decoder's tied-head routing: for a plain fp16/fp32 embedding the module-routed
//  embedToken/logits are value-identical to the raw-weight formulas they replaced,
//  and a quantized tied embedding (packed weight + scales/biases, as converted
//  checkpoints ship it) dequantizes through the module instead of exposing packed
//  bits to the decode path.
//
//  Run:
//    xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
//      -only-testing:'MLXAudioTests/VoxtralRealtimeTiedEmbeddingTests' \
//      CODE_SIGNING_ALLOWED=NO

import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioSTT

struct VoxtralRealtimeTiedEmbeddingTests {
    private static func smallConfig() -> VoxtralRealtimeDecoderConfig {
        VoxtralRealtimeDecoderConfig(
            dim: 64,
            nLayers: 1,
            nHeads: 2,
            nKvHeads: 1,
            headDim: 32,
            hiddenDim: 128,
            vocabSize: 96,
            normEps: 1e-5,
            ropeTheta: 1_000_000,
            slidingWindow: 8,
            tiedEmbeddings: true,
            adaRmsNormTCond: false,
            adaRmsNormTCondDim: 32
        )
    }

    @Test func moduleRoutedTiedHeadMatchesRawWeightFormulas() {
        let decoder = VoxtralRealtimeDecoder(Self.smallConfig())
        let w = decoder.tokEmbeddings.weight
        let h = MLXRandom.normal([w.shape[1]]).asType(w.dtype)

        // The raw-weight formulas the module-routed paths replaced.
        let expectedEmbed = w[7]
        let expectedLogits = MLX.matmul(h, w.transposed(1, 0))

        // The gather is the identical op — exact. The projection is the same
        // contraction lifted to rank 2, so allow kernel-order rounding only.
        #expect(
            MLX.abs(decoder.embedToken(tokenId: 7) - expectedEmbed)
                .max().item(Float.self) == 0
        )
        let logits = decoder.logits(h)
        #expect(logits.shape == expectedLogits.shape)
        #expect(
            MLX.abs(logits - expectedLogits).max().item(Float.self) < 1e-5
        )
    }

    @Test func quantizedTiedEmbeddingRoutesThroughDequantization() {
        let decoder = VoxtralRealtimeDecoder(Self.smallConfig())
        let dim = decoder.tokEmbeddings.weight.shape[1]

        quantize(model: decoder, groupSize: 32, bits: 4) { path, module in
            path == "tok_embeddings" && module is Embedding
        }
        guard let quantized = decoder.tokEmbeddings as? QuantizedEmbedding else {
            Issue.record("tok_embeddings was not replaced by a QuantizedEmbedding")
            return
        }

        let dequant = dequantized(
            quantized.weight,
            scales: quantized.scales,
            biases: quantized.biases,
            groupSize: quantized.groupSize,
            bits: quantized.bits,
            mode: quantized.mode
        )
        let h = MLXRandom.normal([dim]).asType(dequant.dtype)

        // embedToken must return the dequantized row, not packed bits.
        #expect(
            MLX.abs(decoder.embedToken(tokenId: 7) - dequant[7])
                .max().item(Float.self) == 0
        )

        // The tied projection must equal the contraction over the dequantized
        // weight, up to the quantized kernel's accumulation order.
        let expected = MLX.matmul(h, dequant.transposed(1, 0))
        let logits = decoder.logits(h)
        #expect(logits.shape == expected.shape)
        #expect(MLX.abs(logits - expected).max().item(Float.self) < 1e-2)
    }
}
