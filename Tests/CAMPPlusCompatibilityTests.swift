import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXAudioCodecs

@Suite("CAMPPlus checkpoint layouts", .serialized)
struct CAMPPlusCompatibilityTests {
    @Test func legacyContextKeepsBatchNormAndPartialSegmentMean() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = CAMLayer(inChannels: 2, outChannels: 2, kernelSize: 1, padding: 0)
            var weights: [String: MLXArray] = [
                "linear_local.weight": MLXArray([Float(1), 0, 0, 1]).reshaped([2, 1, 2]),
                "linear1.weight": MLXArray([Float(0.5), 0.25]).reshaped([1, 1, 2]), "linear1.bias": MLXArray([Float(0)]),
                "linear2.weight": MLXArray([Float(0.25), 0.5]).reshaped([2, 1, 1]), "linear2.bias": MLXArray([Float(-0.125), 0.25]),
                "bn1.running_mean": MLXArray([Float(0.5)]), "bn1.running_var": MLXArray([Float(4)]),
                "bn1.weight": MLXArray([Float(2)]), "bn1.bias": MLXArray([Float(0.25)]),
                "bn2.running_mean": MLXArray([Float(0.5), -0.5]), "bn2.running_var": MLXArray([Float(1), 4]),
                "bn2.weight": MLXArray([Float(1), 2]), "bn2.bias": MLXArray([Float(0.125), -0.25])]
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all); model.train(false)
            let input = MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped([1, 2, 3])
            let hidden = (Float(4.5) - 0.5) / sqrt(Float(4.00001)) * 2 + 0.25
            let a = (hidden * 0.25 - 0.125 - 0.5) / sqrt(Float(1.00001)) + 0.125
            let b = (hidden * 0.5 + 0.25 + 0.5) / sqrt(Float(4.00001)) * 2 - 0.25
            let expected = input * sigmoid(MLXArray([a, b]).reshaped([1, 2, 1]))
            #expect(abs(model(input) - expected).max().item(Float.self) < 2e-6)
            let compatible = CAMLayer(inChannels: 2, outChannels: 2, kernelSize: 1, padding: 0, referenceCompatible: true)
            weights = weights.filter { !$0.key.hasPrefix("bn") }
            try compatible.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            let contextHidden: Float = 2.06 * 0.5 + 5.15 * 0.25
            let gates = sigmoid(MLXArray([contextHidden * 0.25 - 0.125, contextHidden * 0.5 + 0.25]).reshaped([1, 2, 1]))
            #expect(abs(compatible(input) - input * gates).max().item(Float.self) < 2e-6)
        }
    }
    @Test func publishedLayoutHasNoUntrainedAffineOrContextParameters() {
        let legacy = Set(CAMPPlus().parameters().flattened().map(\.0))
        let published = Set(CAMPPlus(referenceCompatible: true).parameters().flattened().map(\.0))
        #expect(legacy.contains("blocks.0.layers.0.cam_layer.bn1.weight"))
        #expect(legacy.contains("dense.nonlinear.0.weight"))
        #expect(legacy.contains("transits.0.linear.bias"))
        #expect(!published.contains { $0.contains(".cam_layer.bn") })
        #expect(!published.contains("dense.nonlinear.0.weight"))
        #expect(!published.contains("transits.0.linear.bias"))
        #expect(published.contains("dense.nonlinear.0.running_var"))
    }
}
