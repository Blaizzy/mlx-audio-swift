import Foundation
import MLX
import MLXNN

private let nemotronASRNegInf: Float = -1e9

enum NemotronASRAttentionMask {
    static func createChunkedLimitedMask(seqLen: Int, leftContext: Int, rightContext: Int) -> MLXArray {
        let chunkSize = max(rightContext + 1, 1)
        let leftChunks = leftContext >= 0 ? leftContext / chunkSize : 1_000_000

        let positions = MLX.arange(seqLen, dtype: .float32)
        let chunkIndex = floor(positions / Float(chunkSize)).asType(.int32)
        let queryChunks = chunkIndex.expandedDimensions(axis: 1)
        let keyChunks = chunkIndex.expandedDimensions(axis: 0)
        let diff = queryChunks - keyChunks
        let visible = logicalAnd(diff .>= MLXArray(Int32(0)), diff .<= MLXArray(Int32(leftChunks)))
        let mask = MLX.where(visible, MLXArray(Float(0)), MLXArray(nemotronASRNegInf)).asType(.float32)
        return mask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    }
}

final class NemotronASRFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dModel: Int, dFF: Int, useBias: Bool) {
        self._linear1.wrappedValue = Linear(dModel, dFF, bias: useBias)
        self._linear2.wrappedValue = Linear(dFF, dModel, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

final class NemotronASRConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    let padLeft: Int
    let padRight: Int

    init(args: NemotronASRConformerConfig) {
        let context = args.convContextSize
        switch context {
        case .causal:
            self.padLeft = args.convKernelSize - 1
            self.padRight = 0
        case .explicit(let left, let right):
            self.padLeft = left
            self.padRight = right
        }

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
            padding: 0,
            groups: args.dModel,
            bias: args.useBias
        )
        self._batchNorm.wrappedValue = LayerNorm(dimensions: args.dModel)
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
        var y = split[0] * sigmoid(split[1])
        y = MLX.padded(y, widths: [.init(0), .init((padLeft, padRight)), .init(0)])
        y = depthwiseConv(y)
        y = batchNorm(y)
        y = silu(y)
        return pointwiseConv2(y)
    }
}

final class NemotronASRCausalDwStridingSubsampling: Module {
    let subsamplingFactor: Int
    let samplingNum: Int
    let stride: Int = 2
    let kernelSize: Int = 3
    let padLeft: Int = 2
    let padRight: Int = 1
    let convChannels: Int

    @ModuleInfo(key: "conv0") var conv0: Conv2d
    @ModuleInfo(key: "depthwise_layers") var depthwiseLayers: [Conv2d]
    @ModuleInfo(key: "pointwise_layers") var pointwiseLayers: [Conv2d]
    @ModuleInfo(key: "out") var out: Linear

    init(args: NemotronASRConformerConfig) {
        self.subsamplingFactor = args.subsamplingFactor
        self.samplingNum = Int(log2(Double(args.subsamplingFactor)))
        self.convChannels = args.subsamplingConvChannels

        var finalFreqDim = args.featIn
        for _ in 0..<samplingNum {
            finalFreqDim = Int(floor(Double(finalFreqDim + padLeft + padRight - kernelSize) / Double(stride)) + 1)
            if finalFreqDim < 1 {
                finalFreqDim = 1
            }
        }

        self._conv0.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 0
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
                        padding: 0,
                        groups: convChannels
                    )
                )
                pointwise.append(
                    Conv2d(
                        inputChannels: convChannels,
                        outputChannels: convChannels,
                        kernelSize: 1,
                        stride: 1,
                        padding: 0
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
            outLengths = floor((outLengths + Float(padLeft + padRight - kernelSize)) / Float(stride)) + 1
        }

        var y = x.expandedDimensions(axis: 3)
        y = causalPad2D(y)
        y = relu(conv0(y))

        if !depthwiseLayers.isEmpty {
            for i in depthwiseLayers.indices {
                y = causalPad2D(y)
                y = pointwiseLayers[i](depthwiseLayers[i](y))
                y = relu(y)
            }
        }

        let batch = y.shape[0]
        let time = y.shape[1]
        let freq = y.shape[2]
        let channels = y.shape[3]
        y = y.transposed(0, 1, 3, 2).reshaped([batch, time, channels * freq])
        y = out(y)
        return (y, outLengths.asType(.int32))
    }

    private func causalPad2D(_ x: MLXArray) -> MLXArray {
        MLX.padded(
            x,
            widths: [.init(0), .init((padLeft, padRight)), .init((padLeft, padRight)), .init(0)]
        )
    }
}

final class NemotronASRConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: NemotronASRFeedForward

    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: NemoRelPositionMultiHeadAttention

    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: NemotronASRConvolution

    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: NemotronASRFeedForward

    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(args: NemotronASRConformerConfig) {
        let ffHidden = args.dModel * args.ffExpansionFactor

        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._feedForward1.wrappedValue = NemotronASRFeedForward(dModel: args.dModel, dFF: ffHidden, useBias: args.useBias)

        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._selfAttn.wrappedValue = NemoRelPositionMultiHeadAttention(
            nHead: args.nHeads,
            nFeat: args.dModel,
            bias: args.useBias
        )

        self._normConv.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._conv.wrappedValue = NemotronASRConvolution(args: args)

        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: args.dModel)
        self._feedForward2.wrappedValue = NemotronASRFeedForward(dModel: args.dModel, dFF: ffHidden, useBias: args.useBias)
        self._normOut.wrappedValue = LayerNorm(dimensions: args.dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var y = x + MLXArray(Float(0.5)).asType(x.dtype) * feedForward1(normFeedForward1(x))
        let yNorm = normSelfAtt(y)
        y = y + selfAttn(yNorm, yNorm, yNorm, posEmb: posEmb, mask: mask)
        y = y + conv(normConv(y))
        y = y + MLXArray(Float(0.5)).asType(y.dtype) * feedForward2(normFeedForward2(y))
        return normOut(y)
    }
}

final class NemotronASRConformer: Module {
    let args: NemotronASRConformerConfig

    @ModuleInfo(key: "pos_enc") var posEnc: NemoRelPositionalEncoding
    @ModuleInfo(key: "pre_encode") var preEncode: NemotronASRCausalDwStridingSubsampling
    @ModuleInfo(key: "layers") var layers: [NemotronASRConformerBlock]

    init(args: NemotronASRConformerConfig) {
        self.args = args
        self._posEnc.wrappedValue = NemoRelPositionalEncoding(
            dModel: args.dModel,
            maxLen: args.posEmbMaxLen,
            scaleInput: args.xscaling
        )
        self._preEncode.wrappedValue = NemotronASRCausalDwStridingSubsampling(args: args)
        self._layers.wrappedValue = (0..<args.nLayers).map { _ in
            NemotronASRConformerBlock(args: args)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        lengths: MLXArray? = nil,
        attContextSize: [Int]? = nil
    ) -> (MLXArray, MLXArray) {
        let inLengths = lengths ?? MLXArray(Array(repeating: Int32(x.shape[1]), count: x.shape[0])).asType(.int32)
        let encoded = preEncode(x, lengths: inLengths)
        var h = encoded.0
        let outLengths = encoded.1
        let positional = posEnc(h)
        h = positional.0
        let posEmb = positional.1

        let context = attContextSize ?? args.attContextSize.first ?? [56, 13]
        let leftContext = context.first ?? 56
        let rightContext = context.dropFirst().first ?? 13
        let mask: MLXArray?
        if args.attContextStyle == "chunked_limited" {
            mask = NemotronASRAttentionMask
                .createChunkedLimitedMask(seqLen: h.shape[1], leftContext: leftContext, rightContext: rightContext)
                .asType(h.dtype)
        } else {
            mask = nil
        }

        for layer in layers {
            h = layer(h, posEmb: posEmb, mask: mask)
        }

        return (h, outLengths)
    }
}
