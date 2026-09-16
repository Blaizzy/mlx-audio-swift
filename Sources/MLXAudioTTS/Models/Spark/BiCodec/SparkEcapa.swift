import Foundation
@preconcurrency import MLX
import MLXNN

private func swapCT(_ x: MLXArray) -> MLXArray { x.swappedAxes(1, 2) }

private final class SparkEcapaConvReluBn: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(inChannels: Int, outChannels: Int, kernel: Int, padding: Int, dilation: Int = 1) {
        self._conv = ModuleInfo(
            wrappedValue: Conv1d(
                inputChannels: inChannels, outputChannels: outChannels,
                kernelSize: kernel, stride: 1, padding: padding, dilation: dilation, bias: true),
            key: "conv")
        self._bn = ModuleInfo(wrappedValue: BatchNorm(featureCount: outChannels), key: "bn")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = swapCT(conv(swapCT(x)))
        h = MLXNN.relu(h)
        return swapCT(bn(swapCT(h)))
    }
}

private final class SparkEcapaRes2: Module {
    @ModuleInfo(key: "convs") var convs: [Conv1d]
    @ModuleInfo(key: "bns") var bns: [BatchNorm]
    let scale: Int
    let nums: Int

    init(channels: Int, kernel: Int, padding: Int, dilation: Int, scale: Int) {
        self.scale = scale
        self.nums = scale == 1 ? scale : scale - 1
        let width = channels / scale
        self._convs = ModuleInfo(
            wrappedValue: (0 ..< nums).map { _ in
                Conv1d(inputChannels: width, outputChannels: width, kernelSize: kernel,
                       stride: 1, padding: padding, dilation: dilation, bias: true)
            }, key: "convs")
        self._bns = ModuleInfo(wrappedValue: (0 ..< nums).map { _ in BatchNorm(featureCount: width) }, key: "bns")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let spx = MLX.split(x, parts: scale, axis: 1)
        var out: [MLXArray] = []
        var sp = spx[0]
        for i in 0 ..< nums {
            if i >= 1 { sp = sp + spx[i] }
            sp = convs[i](swapCT(sp))
            sp = swapCT(bns[i](MLXNN.relu(sp)))
            out.append(sp)
        }
        if scale != 1 { out.append(spx[nums]) }
        return MLX.concatenated(out, axis: 1)
    }
}

private final class SparkEcapaSEConnect: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(channels: Int, bottleneck: Int = 128) {
        self._linear1 = ModuleInfo(wrappedValue: Linear(channels, bottleneck), key: "linear1")
        self._linear2 = ModuleInfo(wrappedValue: Linear(bottleneck, channels), key: "linear2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = MLX.mean(x, axis: 2)
        out = MLXNN.relu(linear1(out))
        out = MLX.sigmoid(linear2(out))
        return x * out.expandedDimensions(axis: -1)
    }
}

private final class SparkEcapaSERes2Block: Module {
    @ModuleInfo(key: "se_res2block") fileprivate var seRes2Block: [Module]

    init(channels: Int, kernel: Int, padding: Int, dilation: Int, scale: Int) {
        self._seRes2Block = ModuleInfo(
            wrappedValue: [
                SparkEcapaConvReluBn(inChannels: channels, outChannels: channels, kernel: 1, padding: 0),
                SparkEcapaRes2(channels: channels, kernel: kernel, padding: padding, dilation: dilation, scale: scale),
                SparkEcapaConvReluBn(inChannels: channels, outChannels: channels, kernel: 1, padding: 0),
                SparkEcapaSEConnect(channels: channels),
            ], key: "se_res2block")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = (seRes2Block[0] as! SparkEcapaConvReluBn)(x)
        h = (seRes2Block[1] as! SparkEcapaRes2)(h)
        h = (seRes2Block[2] as! SparkEcapaConvReluBn)(h)
        h = (seRes2Block[3] as! SparkEcapaSEConnect)(h)
        return h + x
    }
}

/// ECAPA-TDNN speaker encoder (the latent trunk used by the BiCodec tokenizer):
/// layer1 + three SE-Res2 blocks concatenated and 1x1-convolved to the 1536-d
/// latent feature map. The pooling/embedding head is not needed for tokenization.
public final class SparkEcapaTDNN: Module {
    @ModuleInfo(key: "layer1") fileprivate var layer1: SparkEcapaConvReluBn
    @ModuleInfo(key: "layer2") fileprivate var layer2: SparkEcapaSERes2Block
    @ModuleInfo(key: "layer3") fileprivate var layer3: SparkEcapaSERes2Block
    @ModuleInfo(key: "layer4") fileprivate var layer4: SparkEcapaSERes2Block
    @ModuleInfo(key: "conv") fileprivate var conv: Conv1d

    public init(featDim: Int, channels: Int = 512, scale: Int = 8) {
        self._layer1 = ModuleInfo(
            wrappedValue: SparkEcapaConvReluBn(inChannels: featDim, outChannels: channels, kernel: 5, padding: 2),
            key: "layer1")
        self._layer2 = ModuleInfo(
            wrappedValue: SparkEcapaSERes2Block(channels: channels, kernel: 3, padding: 2, dilation: 2, scale: scale),
            key: "layer2")
        self._layer3 = ModuleInfo(
            wrappedValue: SparkEcapaSERes2Block(channels: channels, kernel: 3, padding: 3, dilation: 3, scale: scale),
            key: "layer3")
        self._layer4 = ModuleInfo(
            wrappedValue: SparkEcapaSERes2Block(channels: channels, kernel: 3, padding: 4, dilation: 4, scale: scale),
            key: "layer4")
        self._conv = ModuleInfo(
            wrappedValue: Conv1d(inputChannels: channels * 3, outputChannels: channels * 3, kernelSize: 1, bias: true),
            key: "conv")
    }

    /// Mel features [B, T, feat_dim] -> latent [B, channels*3, T].
    public func latent(_ mel: MLXArray) -> MLXArray {
        let x = mel.transposed(0, 2, 1)
        let out1 = layer1(x)
        let out2 = layer2(out1)
        let out3 = layer3(out2)
        let out4 = layer4(out3)
        let cat = MLX.concatenated([out2, out3, out4], axis: 1)
        let out = swapCT(conv(swapCT(cat)))
        return MLXNN.relu(out)
    }
}
