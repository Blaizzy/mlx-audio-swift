import Foundation
import MLX
import MLXNN

final class ParakeetFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dModel: Int, dFF: Int, useBias: Bool = true) {
        self._linear1.wrappedValue = Linear(dModel, dFF, bias: useBias)
        self._linear2.wrappedValue = Linear(dFF, dModel, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

final class ParakeetConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(args: ParakeetConformerConfig) {
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: args.dModel,
            outputChannels: args.dModel * 2,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: args.useBias
        )
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: args.dModel,
            outputChannels: args.dModel,
            kernelSize: args.convKernelSize,
            stride: 1,
            padding: (args.convKernelSize - 1) / 2,
            groups: args.dModel,
            bias: args.useBias
        )
        self._batchNorm.wrappedValue = BatchNorm(featureCount: args.dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: args.dModel,
            outputChannels: args.dModel,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: args.useBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pw = pointwiseConv1(x)
        let split = pw.split(parts: 2, axis: 2)
        let gated = split[0] * sigmoid(split[1])
        let dw = depthwiseConv(gated)
        return pointwiseConv2(silu(batchNorm(dw)))
    }
}

final class ParakeetDwStridingSubsampling: Module {
    let subsamplingFactor: Int
    let samplingNum: Int
    let stride: Int = 2
    let kernelSize: Int = 3
    let padding: Int = 1
    let convChannels: Int

    @ModuleInfo(key: "conv0") var conv0: Conv2d
    @ModuleInfo(key: "depthwise_layers") var depthwiseLayers: [Conv2d]
    @ModuleInfo(key: "pointwise_layers") var pointwiseLayers: [Conv2d]
    @ModuleInfo(key: "out") var out: Linear

    init(args: ParakeetConformerConfig) {
        self.subsamplingFactor = args.subsamplingFactor
        self.samplingNum = Int(log2(Double(args.subsamplingFactor)))
        self.convChannels = args.subsamplingConvChannels

        var finalFreqDim = args.featIn
        for _ in 0..<samplingNum {
            finalFreqDim = Int(floor(Double(finalFreqDim + 2 * padding - kernelSize) / Double(stride)) + 1.0)
            if finalFreqDim < 1 {
                finalFreqDim = 1
            }
        }

        self._conv0.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        var depthwise: [Conv2d] = []
        var pointwise: [Conv2d] = []
        if samplingNum > 1 {
            depthwise.reserveCapacity(samplingNum - 1)
            pointwise.reserveCapacity(samplingNum - 1)
            for _ in 0..<(samplingNum - 1) {
                depthwise.append(
                    Conv2d(
                        inputChannels: convChannels,
                        outputChannels: convChannels,
                        kernelSize: 3,
                        stride: 2,
                        padding: 1,
                        groups: convChannels
                    )
                )
                pointwise.append(
                    Conv2d(
                        inputChannels: convChannels,
                        outputChannels: convChannels,
                        kernelSize: 1,
                        stride: 1,
                        padding: 0,
                        groups: 1
                    )
                )
            }
        }
        self._depthwiseLayers.wrappedValue = depthwise
        self._pointwiseLayers.wrappedValue = pointwise
        self._out.wrappedValue = Linear(convChannels * finalFreqDim, args.dModel)
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var outLengths = lengths.asType(.float32)
        for _ in 0..<samplingNum {
            outLengths = floor((outLengths + Float(2 * padding - kernelSize)) / Float(stride)) + 1
        }
        let intLengths = outLengths.asType(.int32)

        var y = x.expandedDimensions(axis: 3)  // [B, T, F, 1]
        y = relu(conv0(y))
        if !depthwiseLayers.isEmpty {
            for i in depthwiseLayers.indices {
                y = relu(pointwiseLayers[i](depthwiseLayers[i](y)))
            }
        }

        let batch = y.shape[0]
        let time = y.shape[1]
        let freq = y.shape[2]
        let channels = y.shape[3]
        y = y.transposed(0, 1, 3, 2).reshaped([batch, time, channels * freq])
        y = out(y)
        return (y, intLengths)
    }
}

final class ParakeetConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ParakeetFeedForward

    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var relSelfAttn: ParakeetRelPositionMultiHeadAttention?
    let selfAttn: ParakeetMultiHeadAttention?

    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ParakeetConvolution

    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ParakeetFeedForward

    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    let usesRelPos: Bool

    init(args: ParakeetConformerConfig) {
        let ffHidden = args.dModel * args.ffExpansionFactor

        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._feedForward1.wrappedValue = ParakeetFeedForward(dModel: args.dModel, dFF: ffHidden, useBias: args.useBias)

        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: args.dModel)
        if args.selfAttentionModel == "rel_pos" {
            self._relSelfAttn.wrappedValue = ParakeetRelPositionMultiHeadAttention(
                nHead: args.nHeads,
                nFeat: args.dModel,
                bias: args.useBias
            )
            self.selfAttn = nil
            self.usesRelPos = true
        } else {
            self._relSelfAttn.wrappedValue = nil
            self.selfAttn = ParakeetMultiHeadAttention(
                nHead: args.nHeads,
                nFeat: args.dModel,
                bias: args.useBias
            )
            self.usesRelPos = false
        }

        self._normConv.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._conv.wrappedValue = ParakeetConvolution(args: args)

        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._feedForward2.wrappedValue = ParakeetFeedForward(dModel: args.dModel, dFF: ffHidden, useBias: args.useBias)
        self._normOut.wrappedValue = LayerNorm(dimensions: args.dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
        var y = x + 0.5 * feedForward1(normFeedForward1(x))

        let xNorm = normSelfAtt(y)
        if usesRelPos {
            guard let rel = relSelfAttn, let posEmb else {
                fatalError("Expected relative positional attention and posEmb")
            }
            y = y + rel(xNorm, xNorm, xNorm, posEmb: posEmb, mask: mask)
        } else {
            guard let mha = selfAttn else {
                fatalError("Expected multi-head attention module")
            }
            y = y + mha(xNorm, xNorm, xNorm, mask: mask)
        }

        y = y + conv(normConv(y))
        y = y + 0.5 * feedForward2(normFeedForward2(y))
        return normOut(y)
    }
}

final class ParakeetConformer: Module {
    let args: ParakeetConformerConfig
    let posEnc: ParakeetRelPositionalEncoding?

    @ModuleInfo(key: "pre_encode") var preEncodeDw: ParakeetDwStridingSubsampling?
    let preEncodeLinear: Linear?
    @ModuleInfo(key: "layers") var layers: [ParakeetConformerBlock]

    init(args: ParakeetConformerConfig) {
        self.args = args
        if args.selfAttentionModel == "rel_pos" {
            self.posEnc = ParakeetRelPositionalEncoding(
                dModel: args.dModel,
                maxLen: args.posEmbMaxLen,
                scaleInput: args.xscaling
            )
        } else {
            self.posEnc = nil
        }

        if args.subsamplingFactor > 1 && args.subsampling == "dw_striding" && !args.causalDownsampling {
            self._preEncodeDw.wrappedValue = ParakeetDwStridingSubsampling(args: args)
            self.preEncodeLinear = nil
        } else {
            self._preEncodeDw.wrappedValue = nil
            self.preEncodeLinear = Linear(args.featIn, args.dModel)
        }
        self._layers.wrappedValue = (0..<args.nLayers).map { _ in
            ParakeetConformerBlock(args: args)
        }
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let inLengths = lengths ?? MLXArray(Array(repeating: Int32(x.shape[1]), count: x.shape[0]))

        let encoded: (MLXArray, MLXArray)
        if let preEncodeDw {
            encoded = preEncodeDw(x, lengths: inLengths)
        } else if let preEncodeLinear {
            encoded = (preEncodeLinear(x), inLengths)
        } else {
            fatalError("ParakeetConformer pre_encode is not configured")
        }

        var h = encoded.0
        let outLengths = encoded.1
        var pos: MLXArray? = nil
        if let posEnc {
            let encoded = posEnc(h)
            h = encoded.0
            pos = encoded.1
        }

        for layer in layers {
            h = layer(h, posEmb: pos)
        }
        return (h, outLengths)
    }
}
