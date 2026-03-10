import MLX
import MLXNN

public class EcapaTdnnBackbone: Module {
    @ModuleInfo(key: "block0") var block0: TDNNBlock
    @ModuleInfo(key: "block1") var block1: SERes2NetBlock
    @ModuleInfo(key: "block2") var block2: SERes2NetBlock
    @ModuleInfo(key: "block3") var block3: SERes2NetBlock
    @ModuleInfo var mfa: TDNNBlock
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo(key: "asp_bn") var aspBn: BatchNorm
    @ModuleInfo var fc: MLXNN.Conv1d

    public init(config: EcapaTdnnConfig) {
        let channels = config.channels

        _block0.wrappedValue = TDNNBlock(
            inputChannels: config.inputSize,
            outputChannels: channels,
            kernelSize: config.kernelSizes[0]
        )
        _block1.wrappedValue = SERes2NetBlock(
            channels: channels,
            kernelSize: config.kernelSizes[1],
            dilation: config.dilations[1],
            res2netScale: config.res2netScale,
            seChannels: config.seChannels
        )
        _block2.wrappedValue = SERes2NetBlock(
            channels: channels,
            kernelSize: config.kernelSizes[2],
            dilation: config.dilations[2],
            res2netScale: config.res2netScale,
            seChannels: config.seChannels
        )
        _block3.wrappedValue = SERes2NetBlock(
            channels: channels,
            kernelSize: config.kernelSizes[3],
            dilation: config.dilations[3],
            res2netScale: config.res2netScale,
            seChannels: config.seChannels
        )
        _mfa.wrappedValue = TDNNBlock(
            inputChannels: channels * 3,
            outputChannels: channels * 3,
            kernelSize: config.kernelSizes[4]
        )
        _asp.wrappedValue = AttentiveStatisticsPooling(
            channels: channels * 3,
            attentionChannels: config.attentionChannels,
            globalContext: config.globalContext
        )
        _aspBn.wrappedValue = BatchNorm(featureCount: channels * 6)
        _fc.wrappedValue = MLXNN.Conv1d(
            inputChannels: channels * 6,
            outputChannels: config.embedDim,
            kernelSize: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = block0(x)
        var layers: [MLXArray] = []
        out = block1(out)
        layers.append(out)
        out = block2(out)
        layers.append(out)
        out = block3(out)
        layers.append(out)

        out = concatenated(layers, axis: -1)
        out = mfa(out)
        out = asp(out)
        out = aspBn(out)
        out = expandedDimensions(out, axis: 1)
        out = fc(out)

        if out.ndim == 3 {
            if out.dim(1) == 1 {
                return out.squeezed(axis: 1)
            }
            if out.dim(2) == 1 {
                return out.squeezed(axis: 2)
            }
        }

        return out
    }
}

class TDNNBlock: Module {
    @ModuleInfo var conv: MLXNN.Conv1d
    @ModuleInfo var norm: BatchNorm

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, dilation: Int = 1, groups: Int = 1) {
        let padding = (kernelSize - 1) * dilation / 2
        _conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: true
        )
        _norm.wrappedValue = BatchNorm(featureCount: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(relu(conv(x)))
    }
}

class Res2NetBlock: Module {
    let scale: Int
    @ModuleInfo var blocks: [TDNNBlock]

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1, scale: Int = 8) {
        self.scale = scale
        let hidden = channels / scale
        _blocks.wrappedValue = (0 ..< (scale - 1)).map { _ in
            TDNNBlock(
                inputChannels: hidden,
                outputChannels: hidden,
                kernelSize: kernelSize,
                dilation: dilation
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = split(x, parts: scale, axis: -1)
        var outputs = [chunks[0]]
        for index in 0 ..< blocks.count {
            let input = index > 0 ? chunks[index + 1] + outputs.last! : chunks[index + 1]
            outputs.append(blocks[index](input))
        }
        return concatenated(outputs, axis: -1)
    }
}

class SEBlock: Module {
    @ModuleInfo var conv1: MLXNN.Conv1d
    @ModuleInfo var conv2: MLXNN.Conv1d

    init(inputDim: Int, bottleneck: Int = 128) {
        _conv1.wrappedValue = MLXNN.Conv1d(inputChannels: inputDim, outputChannels: bottleneck, kernelSize: 1)
        _conv2.wrappedValue = MLXNN.Conv1d(inputChannels: bottleneck, outputChannels: inputDim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var squeeze = mean(x, axis: 1, keepDims: true)
        squeeze = relu(conv1(squeeze))
        squeeze = sigmoid(conv2(squeeze))
        return x * squeeze
    }
}

class SERes2NetBlock: Module {
    @ModuleInfo var tdnn1: TDNNBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: TDNNBlock
    @ModuleInfo(key: "se_block") var seBlock: SEBlock

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1, res2netScale: Int = 8, seChannels: Int = 128) {
        _tdnn1.wrappedValue = TDNNBlock(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _res2netBlock.wrappedValue = Res2NetBlock(
            channels: channels,
            kernelSize: kernelSize,
            dilation: dilation,
            scale: res2netScale
        )
        _tdnn2.wrappedValue = TDNNBlock(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _seBlock.wrappedValue = SEBlock(inputDim: channels, bottleneck: seChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = tdnn1(x)
        out = res2netBlock(out)
        out = tdnn2(out)
        out = seBlock(out)
        return out + residual
    }
}

class AttentiveStatisticsPooling: Module {
    let globalContext: Bool
    @ModuleInfo var tdnn: TDNNBlock
    @ModuleInfo var conv: MLXNN.Conv1d

    init(channels: Int, attentionChannels: Int = 128, globalContext: Bool = false) {
        self.globalContext = globalContext
        let tdnnInput = globalContext ? channels * 3 : channels
        _tdnn.wrappedValue = TDNNBlock(
            inputChannels: tdnnInput,
            outputChannels: attentionChannels,
            kernelSize: 1
        )
        _conv.wrappedValue = MLXNN.Conv1d(inputChannels: attentionChannels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attentionInput: MLXArray
        if globalContext {
            let meanTensor = mean(x, axis: 1, keepDims: true)
            let varianceTensor = variance(x, axis: 1, keepDims: true)
            let stdTensor = sqrt(varianceTensor + 1e-9)
            attentionInput = concatenated([
                x,
                broadcast(meanTensor, to: x.shape),
                broadcast(stdTensor, to: x.shape),
            ], axis: -1)
        } else {
            attentionInput = x
        }

        var attention = tdnn(attentionInput)
        attention = tanh(attention)
        attention = conv(attention)
        attention = softmax(attention, axis: 1)

        let weightedMean = sum(attention * x, axis: 1)
        let weightedVariance = sum(attention * (x * x), axis: 1) - weightedMean * weightedMean
        let weightedStd = sqrt(maximum(weightedVariance, 1e-9))
        return concatenated([weightedMean, weightedStd], axis: -1)
    }
}
