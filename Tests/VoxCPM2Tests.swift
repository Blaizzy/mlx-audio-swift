import Foundation
import MLX
import Testing

@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite("VoxCPM2", .serialized)
struct VoxCPM2Tests {
    private func tinyVAE() -> VoxAudioVAE {
        VoxAudioVAE(VoxCPM2AudioVAEConfig(
            encoderDim: 4, encoderRates: [2], latentDim: 3,
            decoderDim: 8, decoderRates: [2], depthwise: true
        ))
    }

    @Test func rawVAEWeightsConvertConvolutionAndSnakeLayouts() throws {
        let vae = tinyVAE()
        let conv = MLXArray(0..<28).asType(.float32).reshaped(4, 1, 7)
        let transposedConv = MLXArray(0..<128).asType(.float32).reshaped(8, 4, 4)
        let alpha = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 4, 1)
        let weights = vae.sanitize(weights: [
            "encoder.block.0.weight": conv,
            "decoder.model.2.block.1.weight": transposedConv,
            "decoder.model.2.block.2.block.0.alpha": alpha,
        ])

        let convWeight = try #require(weights["encoder.conv_in.conv.weight"])
        let transposedWeight = try #require(weights["decoder.blocks.0.conv_t.conv.weight"])
        let snakeWeight = try #require(weights["decoder.blocks.0.res1.snake1.alpha"])
        #expect(convWeight.shape == [4, 7, 1])
        #expect(convWeight.asArray(Float.self) == conv.transposed(0, 2, 1).asArray(Float.self))
        #expect(transposedWeight.shape == [4, 4, 8])
        #expect(transposedWeight.asArray(Float.self) == transposedConv.transposed(1, 2, 0).asArray(Float.self))
        #expect(snakeWeight.shape == [1, 1, 4])
        #expect(snakeWeight.asArray(Float.self) == [1, 2, 3, 4])
        try vae.update(parameters: .unflattened(weights), verify: [.shapeMismatch, .noUnusedKeys])
    }

    @Test func rawWeightNormIsFusedBeforeLayoutConversion() throws {
        let vae = tinyVAE()
        let weights = vae.sanitize(weights: [
            "encoder.block.0.weight_g": MLXArray.ones([4, 1, 1]) * 2,
            "encoder.block.0.weight_v": MLXArray.ones([4, 1, 7]),
        ])
        let weight = try #require(weights["encoder.conv_in.conv.weight"])
        #expect(weight.shape == [4, 7, 1])
        let expected = Float(2) / sqrt(Float(7))
        #expect(weight.asArray(Float.self).allSatisfy { abs($0 - expected) < 1e-6 })
    }

    @Test func convertedPythonVAEWeightsPreserveParametersAndOutput() throws {
        let reference = tinyVAE()
        let model = tinyVAE()
        let parameters = reference.parameters().flattened()
        var pythonWeights: [String: MLXArray] = [:]
        for (key, value) in parameters {
            var pythonKey = key.replacingOccurrences(of: ".blocks.", with: ".blocks.layers.")
                .replacingOccurrences(of: "srCondLayers", with: "sr_cond_layers")
                .replacingOccurrences(of: "srBoundaries", with: "_sr_boundaries")
            for suffix in ["weight", "bias"] {
                pythonKey = pythonKey.replacingOccurrences(of: ".conv.\(suffix)", with: ".\(suffix)")
            }
            pythonWeights[pythonKey] = value
        }
        let sanitized = model.sanitize(weights: pythonWeights)
        #expect(Set(sanitized.keys) == Set(parameters.map(\.0)))
        for (key, value) in parameters {
            let loaded = try #require(sanitized[key])
            #expect(loaded.shape == value.shape)
            #expect(loaded.asArray(Float.self) == value.asArray(Float.self))
        }
        try model.update(parameters: .unflattened(sanitized), verify: .all)
        let audio = MLXArray(0..<32).asType(.float32).reshaped(1, 32, 1) / 32
        let expected = reference.decode(reference.encode(audio))
        let actual = model.decode(model.encode(audio))
        #expect(actual.shape == expected.shape)
        #expect(actual.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(MLX.max(MLX.abs(actual - expected)).item(Float.self) < 1e-6)
    }

    @Test func referenceAudioLoadingPreservesDurationAtModelRate() throws {
        let url = try #require(Bundle.module.url(
            forResource: "conversational_a", withExtension: "wav", subdirectory: "media"
        ))
        let (sourceRate, source) = try loadAudioArray(from: url)
        let (modelRate, resampled) = try loadAudioArray(from: url, sampleRate: 48_000)
        #expect(sourceRate == 24_000)
        #expect(modelRate == 48_000)
        #expect(resampled.size == source.size * 2)
        #expect(resampled.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
