import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - AdainResBlock1d

public class AdainResBlock1d: Module {
    public let dimIn: Int
    public let upsampleType: String
    @ModuleInfo public var conv1: WeightNormedConv
    @ModuleInfo public var conv2: WeightNormedConv
    @ModuleInfo public var norm1: AdaIN1d
    @ModuleInfo public var norm2: AdaIN1d
    @ModuleInfo public var upsample: UpSample1d
    @ModuleInfo public var dropout: MLXNN.Dropout
    public var conv1x1: WeightNormedConv?
    public var pool: WeightNormedConv?

    public init(dimIn: Int, dimOut: Int, styleDim: Int = 64, upsample: Bool = false, dropoutP: Float = 0.0) {
        self.dimIn = dimIn
        self.upsampleType = upsample ? "upsample" : "none"
        _conv1 = ModuleInfo(wrappedValue: WeightNormedConv(inChannels: dimIn, outChannels: dimOut, kernelSize: 3, padding: 1))
        _conv2 = ModuleInfo(wrappedValue: WeightNormedConv(inChannels: dimOut, outChannels: dimOut, kernelSize: 3, padding: 1))
        _norm1 = ModuleInfo(wrappedValue: AdaIN1d(styleDim: styleDim, numFeatures: dimIn))
        _norm2 = ModuleInfo(wrappedValue: AdaIN1d(styleDim: styleDim, numFeatures: dimOut))
        _upsample = ModuleInfo(wrappedValue: UpSample1d(layerType: upsample ? "upsample" : "none"))
        _dropout = ModuleInfo(wrappedValue: MLXNN.Dropout(p: dropoutP))
        if dimIn != dimOut {
            conv1x1 = WeightNormedConv(inChannels: dimIn, outChannels: dimOut, kernelSize: 1, padding: 0, bias: false)
        }
        if upsample {
            pool = WeightNormedConv(inChannels: 1, outChannels: dimIn, kernelSize: 3, stride: 2, padding: 1, groups: dimIn)
        }
    }

    private func shortcut(_ x: MLXArray) -> MLXArray {
        var h = x.swappedAxes(2, 1)
        h = upsample(h)
        h = h.swappedAxes(2, 1)
        if let conv1x1 {
            h = conv1x1(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
        }
        return h
    }

    private func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var h = norm1(x, s)
        h = leakyRelu(h, negativeSlope: 0.2)

        if upsampleType != "none", let pool {
            h = pool(h.swappedAxes(2, 1), op: .convTranspose1d)
            h = MLX.padded(h, widths: [.init((0, 0)), .init((0, 1)), .init((0, 0))])
            h = h.swappedAxes(2, 1)
        }

        h = conv1(dropout(h.swappedAxes(2, 1)), op: .conv1d).swappedAxes(2, 1)
        h = norm2(h, s)
        h = leakyRelu(h, negativeSlope: 0.2)
        h = conv2(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
        return h
    }

    public func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let out = residual(x, s)
        return (out + shortcut(x)) / Float(2).squareRoot()
    }
}

// MARK: - AdaINResBlock1 (Snake activation, for Generator)

public class AdaINResBlock1: Module {
    public var convs1: [WeightNormedConv]
    public var convs2: [WeightNormedConv]
    public var adain1: [AdaIN1d]
    public var adain2: [AdaIN1d]
    public var alpha1_0: MLXArray
    public var alpha1_1: MLXArray
    public var alpha1_2: MLXArray
    public var alpha2_0: MLXArray
    public var alpha2_1: MLXArray
    public var alpha2_2: MLXArray

    public init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5], styleDim: Int = 64) {
        convs1 = dilation.map { d in
            WeightNormedConv(inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                             padding: (kernelSize * d - d) / 2, dilation: d)
        }
        convs2 = (0..<3).map { _ in
            WeightNormedConv(inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                             padding: (kernelSize - 1) / 2)
        }
        adain1 = (0..<3).map { _ in AdaIN1d(styleDim: styleDim, numFeatures: channels) }
        adain2 = (0..<3).map { _ in AdaIN1d(styleDim: styleDim, numFeatures: channels) }
        alpha1_0 = MLXArray.ones([1, channels, 1])
        alpha1_1 = MLXArray.ones([1, channels, 1])
        alpha1_2 = MLXArray.ones([1, channels, 1])
        alpha2_0 = MLXArray.ones([1, channels, 1])
        alpha2_1 = MLXArray.ones([1, channels, 1])
        alpha2_2 = MLXArray.ones([1, channels, 1])
    }

    public func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let alphas1 = [alpha1_0, alpha1_1, alpha1_2]
        let alphas2 = [alpha2_0, alpha2_1, alpha2_2]
        var h = x
        for idx in 0..<3 {
            let a1 = alphas1[idx]
            let a2 = alphas2[idx]
            var xt = adain1[idx](h, s)
            xt = xt + (1 / a1) * MLX.pow(MLX.sin(a1 * xt), 2)
            xt = convs1[idx](xt.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
            xt = adain2[idx](xt, s)
            xt = xt + (1 / a2) * MLX.pow(MLX.sin(a2 * xt), 2)
            xt = convs2[idx](xt.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
            h = xt + h
        }
        return h
    }
}
