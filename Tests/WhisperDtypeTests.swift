import MLX
import MLXNN
import Testing

@testable import MLXAudioSTT

struct WhisperDtypeTests {
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

    @Test func encoderKeepsWeightDtypeWithFloat32Features() {
        let encoder = WhisperEncoder(config: Self.smallConfig())
        let fp16 = encoder.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) }
        encoder.update(parameters: ModuleParameters.unflattened(fp16))

        let features = MLXRandom.normal([1, 64, 80])
        #expect(features.dtype == .float32)
        #expect(encoder(features).dtype == .float16)
    }

    @Test func synthesizedEncoderPositionsUseWeightDtype() {
        let weights: [String: MLXArray] = [
            "encoder.conv2.weight": MLXArray.zeros([64, 3, 64], dtype: .float16),
            "encoder.blocks.0.mlp1.weight": MLXArray.zeros([128, 64], dtype: .float16),
        ]

        let sanitized = WhisperModel.sanitize(weights: weights, config: Self.smallConfig())

        #expect(sanitized["model.encoder.embed_positions.weight"]?.dtype == .float16)
    }
}
