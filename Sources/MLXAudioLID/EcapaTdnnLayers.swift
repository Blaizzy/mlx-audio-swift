import Foundation
import MLX
import MLXNN

// MARK: - TDNN Block

class TDNNBlock: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo var norm: BatchNorm

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, dilation: Int = 1, groups: Int = 1) {
        let padding = (kernelSize - 1) * dilation / 2
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, padding: padding, dilation: dilation,
            groups: groups, bias: true
        )
        _norm.wrappedValue = BatchNorm(featureCount: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(relu(conv(x)))
    }
}

// MARK: - Res2Net Block

class Res2NetBlock: Module {
    let scale: Int
    @ModuleInfo var blocks: [TDNNBlock]

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1, scale: Int = 8) {
        self.scale = scale
        let hidden = channels / scale
        _blocks.wrappedValue = (0..<(scale - 1)).map { _ in
            TDNNBlock(inputChannels: hidden, outputChannels: hidden, kernelSize: kernelSize, dilation: dilation)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = split(x, parts: scale, axis: -1)
        var y = [chunks[0]]
        for i in 0..<blocks.count {
            let inp = i > 0 ? chunks[i + 1] + y.last! : chunks[i + 1]
            y.append(blocks[i](inp))
        }
        return concatenated(y, axis: -1)
    }
}

// MARK: - Squeeze-Excitation Block

class SEBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(inputDim: Int, bottleneck: Int = 128) {
        _conv1.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: bottleneck, kernelSize: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: bottleneck, outputChannels: inputDim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var s = mean(x, axis: 1, keepDims: true)
        s = relu(conv1(s))
        s = sigmoid(conv2(s))
        return x * s
    }
}

// MARK: - SE-Res2Net Block

class SERes2NetBlock: Module {
    @ModuleInfo var tdnn1: TDNNBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: TDNNBlock
    @ModuleInfo(key: "se_block") var seBlock: SEBlock

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1, res2netScale: Int = 8, seChannels: Int = 128) {
        _tdnn1.wrappedValue = TDNNBlock(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _res2netBlock.wrappedValue = Res2NetBlock(
            channels: channels, kernelSize: kernelSize, dilation: dilation, scale: res2netScale
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

// MARK: - Attentive Statistics Pooling

class AttentiveStatisticsPooling: Module {
    @ModuleInfo var tdnn: TDNNBlock
    @ModuleInfo var conv: Conv1d

    init(channels: Int, attentionChannels: Int = 128) {
        _tdnn.wrappedValue = TDNNBlock(inputChannels: channels * 3, outputChannels: attentionChannels, kernelSize: 1)
        _conv.wrappedValue = Conv1d(inputChannels: attentionChannels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let m = mean(x, axis: 1, keepDims: true)
        let v = variance(x, axis: 1, keepDims: true)
        let s = sqrt(v + 1e-9)
        let mExpanded = broadcast(m, to: x.shape)
        let sExpanded = broadcast(s, to: x.shape)

        let attnInput = concatenated([x, mExpanded, sExpanded], axis: -1)
        var attn = tdnn(attnInput)
        attn = tanh(attn)
        attn = conv(attn)
        attn = softmax(attn, axis: 1)

        let weightedMean = sum(attn * x, axis: 1)
        let weightedVar = sum(attn * (x * x), axis: 1) - weightedMean * weightedMean
        let weightedStd = sqrt(maximum(weightedVar, 1e-9))

        return concatenated([weightedMean, weightedStd], axis: -1)
    }
}

// MARK: - Heterogeneous Blocks Wrapper

class EcapaTdnnBlocks: Module {
    @ModuleInfo(key: "block0") var block0: TDNNBlock
    @ModuleInfo(key: "block1") var block1: SERes2NetBlock
    @ModuleInfo(key: "block2") var block2: SERes2NetBlock
    @ModuleInfo(key: "block3") var block3: SERes2NetBlock

    init(config: EcapaTdnnConfig) {
        _block0.wrappedValue = TDNNBlock(
            inputChannels: config.nMels, outputChannels: config.channels, kernelSize: 5
        )
        _block1.wrappedValue = SERes2NetBlock(
            channels: config.channels, kernelSize: 3, dilation: 2,
            res2netScale: config.res2netScale, seChannels: config.seChannels
        )
        _block2.wrappedValue = SERes2NetBlock(
            channels: config.channels, kernelSize: 3, dilation: 3,
            res2netScale: config.res2netScale, seChannels: config.seChannels
        )
        _block3.wrappedValue = SERes2NetBlock(
            channels: config.channels, kernelSize: 3, dilation: 4,
            res2netScale: config.res2netScale, seChannels: config.seChannels
        )
    }
}

// MARK: - Embedding Model

class EcapaTdnnEmbedding: Module {
    @ModuleInfo var blocks: EcapaTdnnBlocks
    @ModuleInfo var mfa: TDNNBlock
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo(key: "asp_bn") var aspBn: BatchNorm
    @ModuleInfo var fc: Conv1d

    init(config: EcapaTdnnConfig) {
        _blocks.wrappedValue = EcapaTdnnBlocks(config: config)
        _mfa.wrappedValue = TDNNBlock(
            inputChannels: config.channels * 3, outputChannels: config.channels * 3, kernelSize: 1
        )
        _asp.wrappedValue = AttentiveStatisticsPooling(
            channels: config.channels * 3, attentionChannels: config.attentionChannels
        )
        _aspBn.wrappedValue = BatchNorm(featureCount: config.channels * 6)
        _fc.wrappedValue = Conv1d(
            inputChannels: config.channels * 6, outputChannels: config.embeddingDim, kernelSize: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = blocks.block0(x)
        var xl = [MLXArray]()
        out = blocks.block1(out); xl.append(out)
        out = blocks.block2(out); xl.append(out)
        out = blocks.block3(out); xl.append(out)

        out = concatenated(xl, axis: -1)
        out = mfa(out)
        out = asp(out)
        out = aspBn(out)
        out = expandedDimensions(out, axis: 1)
        out = fc(out)
        return out
    }
}

// MARK: - Classifier Components

class DNNLinear: Module {
    @ModuleInfo var w: Linear

    init(inputDim: Int, outputDim: Int) {
        _w.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w(x) }
}

class DNNBlock: Module {
    @ModuleInfo var linear: DNNLinear
    @ModuleInfo var norm: BatchNorm

    init(inputDim: Int, outputDim: Int) {
        _linear.wrappedValue = DNNLinear(inputDim: inputDim, outputDim: outputDim)
        _norm.wrappedValue = BatchNorm(featureCount: outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(norm(linear(x)))
    }
}

class DNN: Module {
    @ModuleInfo(key: "block_0") var block0: DNNBlock

    init(inputDim: Int, outputDim: Int) {
        _block0.wrappedValue = DNNBlock(inputDim: inputDim, outputDim: outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block0(x) }
}

class ClassifierLinear: Module {
    @ModuleInfo var w: Linear

    init(inputDim: Int, outputDim: Int) {
        _w.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w(x) }
}

class EcapaClassifier: Module {
    @ModuleInfo var norm: BatchNorm
    @ModuleInfo var DNN: DNN
    @ModuleInfo var out: ClassifierLinear

    init(config: EcapaTdnnConfig) {
        _norm.wrappedValue = BatchNorm(featureCount: config.embeddingDim)
        _DNN.wrappedValue = .init(inputDim: config.embeddingDim, outputDim: config.classifierHiddenDim)
        _out.wrappedValue = ClassifierLinear(inputDim: config.classifierHiddenDim, outputDim: config.numClasses)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.squeezed(axis: 1)
        out = norm(out)
        out = DNN(out)
        out = self.out(out)
        return logSoftmax(out, axis: -1)
    }
}
