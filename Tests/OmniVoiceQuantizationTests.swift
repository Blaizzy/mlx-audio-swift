import Foundation
import MLX
import Testing
@testable import MLXAudioTTS

@Suite("OmniVoice row-wise int8 checkpoint packing", .serialized)
struct OmniVoiceQuantizationTests {
    @Test func packsSignedCodesAndAffineBiases() throws {
        let codes: [Int8] = [-127, -3, 0, 127, 1, -1, 42, -42]
        let scales = MLXArray([Float(0.01), Float(0.02)], [2, 1])
        let weights = [
            "layer.weight": MLXArray(codes, [2, 4]),
            "layer.weight.scales": scales
        ]

        let packed = try OmniVoiceModel.packRowwiseInt8ForMLX(weights: weights, groupSize: 4)

        #expect(packed["layer.weight"]?.shape == [2, 1])
        #expect(packed["layer.scales"]?.shape == [2, 1])
        #expect(packed["layer.biases"]?.shape == [2, 1])

        let packedCodes = try #require(packed["layer.weight"]).asArray(UInt32.self)
        let expectedFirst = UInt32(1) | (UInt32(125) << 8) | (UInt32(128) << 16) | (UInt32(255) << 24)
        let expectedSecond = UInt32(129) | (UInt32(127) << 8) | (UInt32(170) << 16) | (UInt32(86) << 24)
        #expect(packedCodes == [expectedFirst, expectedSecond])

        let packedScales = try #require(packed["layer.scales"]).asArray(Float.self)
        let packedBiases = try #require(packed["layer.biases"]).asArray(Float.self)
        #expect(packedScales == [0.01, 0.02])
        #expect(packedBiases[0].isApproximatelyEqual(to: -1.28, absoluteTolerance: 1e-5))
        #expect(packedBiases[1].isApproximatelyEqual(to: -2.56, absoluteTolerance: 1e-5))
    }
}

private extension Float {
    func isApproximatelyEqual(to other: Float, absoluteTolerance: Float) -> Bool {
        abs(self - other) <= absoluteTolerance
    }
}
