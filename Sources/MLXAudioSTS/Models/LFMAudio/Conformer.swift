// LFM2.5-Audio: FastConformer Encoder
// Port of mlx_audio/sts/models/lfm_audio/conformer.py

import Foundation
import MLX
import MLXNN

// MARK: - Relative Positional Encoding

class RelativePositionalEncoding: Module {
    let dModel: Int
    let maxLen: Int
    let xscale: Float?
    let divTerm: MLXArray
    var pe: MLXArray?

    init(dModel: Int, maxLen: Int = 5000, xscale: Bool = true) {
        self.dModel = dModel
        self.maxLen = maxLen
        self.xscale = xscale ? sqrt(Float(dModel)) : nil

        // div_term: 10000^(-2i/d_model)
        self.divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
                * MLXArray(Float(-log(10000.0) / Float(dModel)))
        )
    }

    func extendPE(length: Int) {
        let neededSize = 2 * length - 1
        if let pe = pe, pe.shape[0] >= neededSize { return }

        // Positions from (length-1) to -(length-1)
        let positions = MLXArray(
            stride(from: Float(length - 1), through: Float(-(length - 1)), by: -1).map { $0 }
        ).expandedDimensions(axis: 1)

        var newPE = MLXArray.zeros([neededSize, dModel])
        let sinVals = MLX.sin(positions * divTerm)
        let cosVals = MLX.cos(positions * divTerm)

        // Fill even indices with sin, odd with cos
        newPE = newPE.at[0..., .stride(by: 2)].add(sinVals)
        newPE = newPE.at[0..., .stride(from: 1, by: 2)].add(cosVals)

        self.pe = newPE
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = x.shape[1]
        extendPE(length: seqLen)

        var result = x
        if let scale = xscale {
            result = result * MLXArray(scale)
        }

        let center = pe!.shape[0] / 2
        let start = center - seqLen + 1
        let end = center + seqLen
        let posEmb = pe![start..<end]

        return (result, posEmb)
    }
}

// MARK: - Conformer Feed Forward

class ConformerFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "dropout") var dropout: Dropout

    init(dModel: Int, dFF: Int, dropoutRate: Float = 0.1) {
        self._linear1.wrappedValue = Linear(dModel, dFF)
        self._linear2.wrappedValue = Linear(dFF, dModel)
        self._dropout.wrappedValue = Dropout(p: dropoutRate)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = silu(h)
        h = dropout(h)
        h = linear2(h)
        h = dropout(h)
        return h
    }
}

// MARK: - Conformer Convolution

class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Linear
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Linear
    @ModuleInfo(key: "dropout") var dropout: Dropout

    init(dModel: Int, kernelSize: Int = 31, normType: String = "batch_norm", dropoutRate: Float = 0.1) {
        self._pointwiseConv1.wrappedValue = Linear(dModel, 2 * dModel)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: kernelSize,
            padding: (kernelSize - 1) / 2,
            groups: dModel
        )
        self._norm.wrappedValue = BatchNorm(featureCount: dModel)
        self._pointwiseConv2.wrappedValue = Linear(dModel, dModel)
        self._dropout.wrappedValue = Dropout(p: dropoutRate)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pointwiseConv1(x)

        // GLU: split and apply sigmoid gate
        let parts = h.split(parts: 2, axis: -1)
        h = parts[0] * sigmoid(parts[1])

        // Depthwise conv (MLX Conv1d expects (B, L, C))
        h = depthwiseConv(h)
        h = norm(h)
        h = silu(h)
        h = pointwiseConv2(h)
        h = dropout(h)
        return h
    }
}

// MARK: - Relative Multi-Head Attention

class RelativeMultiHeadAttention: Module {
    let dModel: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "pos_proj") var posProj: Linear
    @ModuleInfo(key: "dropout") var dropout: Dropout

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(dModel: Int, numHeads: Int, dropoutRate: Float = 0.1) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.headDim = dModel / numHeads
        self.scale = 1.0 / sqrt(Float(self.headDim))

        self._qProj.wrappedValue = Linear(dModel, dModel)
        self._kProj.wrappedValue = Linear(dModel, dModel)
        self._vProj.wrappedValue = Linear(dModel, dModel)
        self._outProj.wrappedValue = Linear(dModel, dModel)
        self._posProj.wrappedValue = Linear(dModel, dModel, bias: false)
        self._dropout.wrappedValue = Dropout(p: dropoutRate)
        self._posBiasU.wrappedValue = MLXArray.zeros([numHeads, self.headDim])
        self._posBiasV.wrappedValue = MLXArray.zeros([numHeads, self.headDim])
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let (B, H, T, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Pad left: (B, H, T, 2T-1) -> (B, H, T, 2T)
        var shifted = MLX.padded(x, widths: [
            IntOrPair((0, 0)), IntOrPair((0, 0)),
            IntOrPair((0, 0)), IntOrPair((1, 0)),
        ])
        // Reshape: (B, H, T, 2T) -> (B, H, 2T, T)
        shifted = shifted.reshaped(B, H, posLen + 1, T)
        // Remove first row
        shifted = shifted[0..., 0..., 1..., 0...]
        // Reshape back
        shifted = shifted.reshaped(B, H, T, posLen)
        // Take first T columns
        return shifted[0..., 0..., 0..., ..<T]
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, T, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(B, T, numHeads, headDim)
        let k = kProj(x).reshaped(B, T, numHeads, headDim)
        let v = vProj(x).reshaped(B, T, numHeads, headDim)

        // Position projection
        let pInput = posEmb.ndim == 2 ? posEmb.expandedDimensions(axis: 0) : posEmb
        let p = posProj(pInput).reshaped(1, -1, numHeads, headDim)

        // Add positional biases and transpose to (B, H, T, D)
        let qWithBiasU = (q + posBiasU.expandedDimensions(axes: [0, 1])).transposed(0, 2, 1, 3)
        let qWithBiasV = (q + posBiasV.expandedDimensions(axes: [0, 1])).transposed(0, 2, 1, 3)

        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)
        let pT = p.transposed(0, 2, 1, 3)

        // Content-based scores: (B, H, T, T)
        let matrixAC = MLX.matmul(qWithBiasU, kT.transposed(0, 1, 3, 2))
        // Position-based scores: (B, H, T, 2T-1)
        var matrixBD = MLX.matmul(qWithBiasV, pT.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)

        var scores = (matrixAC + matrixBD) * MLXArray(scale)

        if let mask = mask {
            scores = scores + mask
        }

        var attn = softmax(scores, axis: -1)
        attn = dropout(attn)

        let out = MLX.matmul(attn, vT).transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return outProj(out)
    }
}

// MARK: - Conformer Layer

class ConformerLayer: Module {
    @ModuleInfo(key: "ff1_norm") var ff1Norm: LayerNorm
    @ModuleInfo(key: "ff1") var ff1: ConformerFeedForward
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm
    @ModuleInfo(key: "attn") var attn: RelativeMultiHeadAttention
    @ModuleInfo(key: "conv_norm") var convNorm: LayerNorm
    @ModuleInfo(key: "conv") var conv: ConformerConvolution
    @ModuleInfo(key: "ff2_norm") var ff2Norm: LayerNorm
    @ModuleInfo(key: "ff2") var ff2: ConformerFeedForward
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(
        dModel: Int, numHeads: Int, ffExpansionFactor: Int = 4,
        convKernelSize: Int = 31, convNormType: String = "batch_norm",
        dropout: Float = 0.1, dropoutAtt: Float = 0.1
    ) {
        let dFF = dModel * ffExpansionFactor
        self._ff1Norm.wrappedValue = LayerNorm(dimensions: dModel)
        self._ff1.wrappedValue = ConformerFeedForward(dModel: dModel, dFF: dFF, dropoutRate: dropout)
        self._attnNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._attn.wrappedValue = RelativeMultiHeadAttention(dModel: dModel, numHeads: numHeads, dropoutRate: dropoutAtt)
        self._convNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ConformerConvolution(dModel: dModel, kernelSize: convKernelSize, normType: convNormType, dropoutRate: dropout)
        self._ff2Norm.wrappedValue = LayerNorm(dimensions: dModel)
        self._ff2.wrappedValue = ConformerFeedForward(dModel: dModel, dFF: dFF, dropoutRate: dropout)
        self._finalNorm.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // FF1 (half residual)
        var h = x + 0.5 * ff1(ff1Norm(x))
        // Attention
        h = h + attn(attnNorm(h), posEmb: posEmb, mask: mask)
        // Conv
        h = h + conv(convNorm(h))
        // FF2 (half residual)
        h = h + 0.5 * ff2(ff2Norm(h))
        // Final norm
        return finalNorm(h)
    }
}

// MARK: - Conv Subsampling (2D Depthwise Separable)

class ConvSubsampling: Module {
    let subsamplingFactor: Int
    let convChannels: Int
    let inChannels: Int
    let conv: [Conv2d?]
    @ModuleInfo(key: "out") var out: Linear

    init(inChannels: Int, outChannels: Int, subsamplingFactor: Int = 8, convChannels: Int = 256) {
        self.subsamplingFactor = subsamplingFactor
        self.convChannels = convChannels
        self.inChannels = inChannels

        // 3 stages of stride-2 convs for 8x subsampling
        self.conv = [
            Conv2d(inputChannels: 1, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1),
            nil, // ReLU placeholder
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1, groups: convChannels),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 1, stride: 1, padding: 0),
            nil, // ReLU placeholder
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1, groups: convChannels),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 1, stride: 1, padding: 0),
        ]

        self._out.wrappedValue = Linear(convChannels * (inChannels / subsamplingFactor), outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, D) = (x.dim(0), x.dim(1), x.dim(2))

        // (B, T, D) -> (B, T, D, 1) for NHWC Conv2d
        var h = x.expandedDimensions(axis: 3)

        // Stage 1: standard conv + relu
        h = relu(conv[0]!(h))
        // Stage 2: depthwise + pointwise + relu
        h = conv[2]!(h)
        h = relu(conv[3]!(h))
        // Stage 3: depthwise + pointwise + relu
        h = conv[5]!(h)
        h = relu(conv[6]!(h))

        let (B2, TOut, DOut, C) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        // (B, T_out, D_out, C) -> (B, T_out, C, D_out) -> (B, T_out, C*D_out)
        h = h.transposed(0, 1, 3, 2).reshaped(B2, TOut, -1)
        return out(h)
    }
}

// MARK: - Conformer Encoder

public class ConformerEncoder: Module {
    let config: ConformerEncoderConfig

    @ModuleInfo(key: "pre_encode") var preEncode: ConvSubsampling
    let posEnc: RelativePositionalEncoding
    @ModuleInfo(key: "pre_dropout") var preDropout: Dropout
    let layers: [ConformerLayer]

    public init(_ config: ConformerEncoderConfig) {
        self.config = config

        self._preEncode.wrappedValue = ConvSubsampling(
            inChannels: config.featIn,
            outChannels: config.dModel,
            subsamplingFactor: config.subsamplingFactor,
            convChannels: config.subsamplingConvChannels
        )

        self.posEnc = RelativePositionalEncoding(
            dModel: config.dModel,
            maxLen: config.posEmbMaxLen,
            xscale: false
        )

        self._preDropout.wrappedValue = Dropout(p: config.dropoutPreEncoder)

        self.layers = (0..<config.nLayers).map { _ in
            ConformerLayer(
                dModel: config.dModel,
                numHeads: config.nHeads,
                ffExpansionFactor: config.ffExpansionFactor,
                convKernelSize: config.convKernelSize,
                convNormType: config.convNormType,
                dropout: config.dropout,
                dropoutAtt: config.dropoutAtt
            )
        }
    }

    public func callAsFunction(_ x: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        // Subsampling
        var h = preEncode(x)

        // Update lengths
        let newLengths: MLXArray
        if let lengths = lengths {
            newLengths = lengths / config.subsamplingFactor
        } else {
            newLengths = MLXArray([Int32(h.shape[1])] + Array(repeating: Int32(h.shape[1]), count: h.shape[0] - 1))
        }

        // Positional encoding
        let (scaledH, posEmb) = posEnc(h)
        h = scaledH

        // Pre-encoder dropout
        h = preDropout(h)

        // Attention mask from lengths
        var mask: MLXArray? = nil
        let maxLen = h.shape[1]
        let idx = MLXArray(0..<Int32(maxLen)).expandedDimensions(axis: 0)
        let paddingMask = idx .>= newLengths.expandedDimensions(axis: 1)
        mask = MLX.which(
            paddingMask.expandedDimensions(axes: [1, 2]),
            MLXArray(Float(-1e9)),
            MLXArray(Float(0))
        )

        // Apply conformer layers
        for layer in layers {
            h = layer(h, posEmb: posEmb, mask: mask)
        }

        return (h, newLengths)
    }
}

// MARK: - MLP Adapter

public class AdapterMLP: Module {
    let layers: [Module]

    public init(inChannels: Int, outChannels: Int, hiddenDims: [Int], useLayerNorm: Bool = true, dropout: Float = 0.0) {
        let channels = [inChannels] + hiddenDims + [outChannels]
        var layerList: [Module] = []

        if useLayerNorm {
            layerList.append(LayerNorm(dimensions: channels[0]))
        }

        for i in 0..<(channels.count - 1) {
            layerList.append(Linear(channels[i], channels[i + 1]))
            if i != channels.count - 2 {
                layerList.append(GELU())
            }
        }

        self.layers = layerList
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let linear = layer as? Linear {
                h = linear(h)
            } else if let norm = layer as? LayerNorm {
                h = norm(h)
            } else if layer is GELU {
                h = MLXNN.gelu(h)
            }
        }
        return h
    }
}
