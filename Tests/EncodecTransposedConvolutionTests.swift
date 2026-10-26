import MLX
import MLXNN
import Testing
@testable import MLXAudioCodecs

struct EncodecTransposedConvolutionTests {
    @Test func nativeTransposeMatchesOverlapAddWithPaddingAndBias() throws {
        let (batch, length, inputs, outputs, kernel, stride, padding) = (2, 3, 2, 3, 4, 2, 1)
        let layer = EncodecBaseConvTranspose1d(inputChannels: inputs, outputChannels: outputs,
            kernelSize: kernel, stride: stride, padding: padding)
        let values: [Float] = (0..<(batch * length * inputs)).map { Float($0 - 4) / 8 }
        let weights: [Float] = (0..<(outputs * kernel * inputs)).map { Float($0 - 8) / 16 }
        let bias: [Float] = [0.125, -0.25, 0.375]
        try layer.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray(weights, [outputs, kernel, inputs]), "bias": MLXArray(bias)
        ]), verify: .all)
        let outputLength = (length - 1) * stride - 2 * padding + kernel
        var expected = [Float](repeating: 0, count: batch * outputLength * outputs)
        for b in 0..<batch {
            for t in 0..<outputLength {
                for o in 0..<outputs { expected[(b * outputLength + t) * outputs + o] = bias[o] }
            }
            for t in 0..<length {
                for k in 0..<kernel {
                    let out = t * stride + k - padding
                    guard (0..<outputLength).contains(out) else { continue }
                    for o in 0..<outputs {
                        for i in 0..<inputs {
                            expected[(b * outputLength + out) * outputs + o]
                                += values[(b * length + t) * inputs + i] * weights[(o * kernel + k) * inputs + i]
                        }
                    }
                }
            }
        }
        let result = layer(MLXArray(values, [batch, length, inputs]))
        #expect(result.shape == [batch, outputLength, outputs])
        #expect(abs(result - MLXArray(expected, result.shape)).max().item(Float.self) < 1e-5)
    }
}
