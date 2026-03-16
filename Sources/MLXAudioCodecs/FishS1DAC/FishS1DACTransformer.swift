import Foundation
import MLX
import MLXFast
import MLXNN

public final class FishS1ConvNeXtBlock: Module, UnaryLayer {
    @ModuleInfo(key: "dwconv") public var dwconv: FishS1CausalConvNet
    @ModuleInfo(key: "norm") public var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") public var pwconv1: Linear
    @ModuleInfo(key: "act") public var act: GELU
    @ModuleInfo(key: "pwconv2") public var pwconv2: Linear

    public var gamma: MLXArray?

    public init(
        dim: Int,
        layerScaleInitValue: Float = 1e-6,
        mlpRatio: Float = 4.0,
        kernelSize: Int = 7,
        dilation: Int = 1
    ) {
        self._dwconv = ModuleInfo(wrappedValue: FishS1CausalConvNet(
            inChannels: dim,
            outChannels: dim,
            kernelSize: kernelSize,
            dilation: dilation,
            stride: 1,
            groups: dim
        ))
        self._norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim, eps: 1e-6))
        self._pwconv1 = ModuleInfo(wrappedValue: Linear(dim, Int(Float(dim) * mlpRatio)))
        self._act = ModuleInfo(wrappedValue: GELU())
        self._pwconv2 = ModuleInfo(wrappedValue: Linear(Int(Float(dim) * mlpRatio), dim))
        self.gamma = layerScaleInitValue > 0 ? MLXArray.ones([dim]) * layerScaleInitValue : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = x
        var y = dwconv(x).transposed(0, 2, 1)
        y = norm(y)
        y = pwconv1(y)
        y = act(y)
        y = pwconv2(y)
        if let gamma {
            y = gamma * y
        }
        y = y.transposed(0, 2, 1)
        return input + y
    }
}

func fishS1PrecomputeFreqsCis(
    seqLen: Int,
    headDim: Int,
    base: Double = 10_000,
    dtype: DType = .float32
) -> MLXArray {
    let freqs = 1.0 / MLX.pow(
        MLXArray(base),
        MLX.arange(0, headDim, step: 2, dtype: .float32) / MLXArray(Float(headDim))
    )
    let time = MLX.arange(seqLen, dtype: .float32)
    let angles = MLX.outer(time, freqs)
    return MLX.stacked([MLX.cos(angles), MLX.sin(angles)], axis: -1).asType(dtype)
}

func fishS1ApplyRotaryEmb(_ x: MLXArray, freqsCis: MLXArray) -> MLXArray {
    let reshaped = x.asType(.float32).reshaped(x.shape.dropLast() + [x.shape.last! / 2, 2])
    let freqs = freqsCis.reshaped([1, freqsCis.shape[0], 1, freqsCis.shape[1], 2])
    let real = reshaped[0..., 0..., 0..., 0..., 0]
    let imag = reshaped[0..., 0..., 0..., 0..., 1]
    let cosVal = freqs[0..., 0..., 0..., 0..., 0]
    let sinVal = freqs[0..., 0..., 0..., 0..., 1]
    let out = MLX.stacked([
        real * cosVal - imag * sinVal,
        imag * cosVal + real * sinVal
    ], axis: -1)
    return out.reshaped(x.shape).asType(x.dtype)
}

public final class FishS1TFRMSNorm: Module, UnaryLayer {
    public let eps: Float
    public var weight: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = x.asType(.float32)
        let normed = out * MLX.rsqrt(MLX.mean(out * out, axis: -1, keepDims: true) + eps)
        return normed.asType(x.dtype) * weight
    }
}

public final class FishS1LayerScale: Module, UnaryLayer {
    public var gamma: MLXArray

    public init(dim: Int, initValues: Float = 1e-2) {
        self.gamma = MLXArray.ones([dim]) * initValues
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

public final class FishS1Attention: Module {
    @ModuleInfo(key: "wqkv") public var wqkv: Linear
    @ModuleInfo(key: "wo") public var wo: Linear

    public let nHead: Int
    public let headDim: Int
    public let nLocalHeads: Int
    public let posEmbedType: String

    public init(config: FishS1DACModelArgs) {
        let totalHeadDim = (config.nHead + 2 * config.nLocalHeads) * config.headDim
        self._wqkv = ModuleInfo(wrappedValue: Linear(config.dim, totalHeadDim, bias: false))
        self._wo = ModuleInfo(wrappedValue: Linear(config.headDim * config.nHead, config.dim, bias: false))
        self.nHead = config.nHead
        self.headDim = config.headDim
        self.nLocalHeads = config.nLocalHeads
        self.posEmbedType = config.posEmbedType
    }

    public func callAsFunction(_ x: MLXArray, freqsCis: MLXArray?, mask: MLXArray?) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]
        let kvSize = nLocalHeads * headDim

        let qkv = wqkv(x)
        let qkvSplit = qkv.split(indices: [kvSize, 2 * kvSize], axis: -1)
        var q = qkvSplit[0]
        var k = qkvSplit[1]
        var v = qkvSplit[2]

        q = q.reshaped([batch, seqLen, nHead, headDim])
        k = k.reshaped([batch, seqLen, nLocalHeads, headDim])
        v = v.reshaped([batch, seqLen, nLocalHeads, headDim])

        if posEmbedType == "rope", let freqsCis {
            q = fishS1ApplyRotaryEmb(q, freqsCis: freqsCis)
            k = fishS1ApplyRotaryEmb(k, freqsCis: freqsCis)
        }

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        if nLocalHeads != nHead {
            let repeatFactor = nHead / nLocalHeads
            k = MLX.repeated(k, count: repeatFactor, axis: 1)
            v = MLX.repeated(v, count: repeatFactor, axis: 1)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / sqrt(Float(headDim)),
            mask: mask
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, seqLen, headDim * nHead])
        return wo(merged)
    }
}

public final class FishS1FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "w1") public var w1: Linear
    @ModuleInfo(key: "w3") public var w3: Linear
    @ModuleInfo(key: "w2") public var w2: Linear

    public init(config: FishS1DACModelArgs) {
        self._w1 = ModuleInfo(wrappedValue: Linear(config.dim, config.intermediateSize, bias: false))
        self._w3 = ModuleInfo(wrappedValue: Linear(config.dim, config.intermediateSize, bias: false))
        self._w2 = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.dim, bias: false))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

public final class FishS1TransformerBlock: Module {
    @ModuleInfo(key: "attention") public var attention: FishS1Attention
    @ModuleInfo(key: "feed_forward") public var feedForward: FishS1FeedForward
    @ModuleInfo(key: "ffn_norm") public var ffnNorm: FishS1TFRMSNorm
    @ModuleInfo(key: "attention_norm") public var attentionNorm: FishS1TFRMSNorm
    @ModuleInfo(key: "attention_layer_scale") public var attentionLayerScale: FishS1LayerScale
    @ModuleInfo(key: "ffn_layer_scale") public var ffnLayerScale: FishS1LayerScale

    public init(config: FishS1DACModelArgs) {
        self._attention = ModuleInfo(wrappedValue: FishS1Attention(config: config))
        self._feedForward = ModuleInfo(wrappedValue: FishS1FeedForward(config: config))
        self._ffnNorm = ModuleInfo(wrappedValue: FishS1TFRMSNorm(dim: config.dim, eps: config.normEps))
        self._attentionNorm = ModuleInfo(wrappedValue: FishS1TFRMSNorm(dim: config.dim, eps: config.normEps))
        self._attentionLayerScale = ModuleInfo(wrappedValue: FishS1LayerScale(dim: config.dim, initValues: 1e-2))
        self._ffnLayerScale = ModuleInfo(wrappedValue: FishS1LayerScale(dim: config.dim, initValues: 1e-2))
    }

    public func callAsFunction(_ x: MLXArray, freqsCis: MLXArray?, mask: MLXArray) -> MLXArray {
        let h = x + attentionLayerScale(attention(attentionNorm(x), freqsCis: freqsCis, mask: mask))
        return h + ffnLayerScale(feedForward(ffnNorm(h)))
    }
}

public class FishS1Transformer: Module, UnaryLayer {
    @ModuleInfo(key: "layers") public var layers: [FishS1TransformerBlock]
    @ModuleInfo(key: "norm") public var norm: FishS1TFRMSNorm

    public let config: FishS1DACModelArgs
    public let freqsCis: MLXArray?

    public init(config: FishS1DACModelArgs) {
        self.config = config
        self._layers = ModuleInfo(wrappedValue: (0..<config.nLayer).map { _ in
            FishS1TransformerBlock(config: config)
        })
        self._norm = ModuleInfo(wrappedValue: FishS1TFRMSNorm(dim: config.dim, eps: config.normEps))
        self.freqsCis = config.posEmbedType == "rope"
            ? fishS1PrecomputeFreqsCis(
                seqLen: config.blockSize,
                headDim: config.headDim,
                base: config.ropeBase
            )
            : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.shape[1]
        let localFreqs = freqsCis.map { $0[0..<seqLen, 0...] }
        let row = MLX.arange(seqLen).expandedDimensions(axis: 1)
        let col = MLX.arange(seqLen).expandedDimensions(axis: 0)
        var mask = (row .>= col)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        if mask.dtype == .bool {
            mask = MLX.where(mask, MLXArray(0.0, dtype: x.dtype), MLXArray(-1e9, dtype: x.dtype))
        }
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, freqsCis: localFreqs, mask: mask)
        }
        return norm(hidden)
    }
}

public final class FishS1WindowLimitedTransformer: FishS1Transformer {
    public let windowSize: Int?
    public let causal: Bool
    public let channelsFirst: Bool

    @ModuleInfo(key: "look_ahead_conv") public var lookAheadConv: Module
    @ModuleInfo(key: "input_proj") public var inputProj: Module
    @ModuleInfo(key: "output_proj") public var outputProj: Module

    public init(
        config: FishS1DACModelArgs,
        inputDim: Int = 512,
        windowSize: Int? = nil,
        causal: Bool = true
    ) {
        self.windowSize = windowSize
        self.causal = causal
        self.channelsFirst = config.channelsFirst
        self._lookAheadConv = ModuleInfo(wrappedValue: FishS1Identity())
        self._inputProj = ModuleInfo(wrappedValue: inputDim == config.dim ? FishS1Identity() : Linear(inputDim, config.dim))
        self._outputProj = ModuleInfo(wrappedValue: inputDim == config.dim ? FishS1Identity() : Linear(config.dim, inputDim))
        super.init(config: config)
    }

    func makeWindowLimitedMask(_ length: Int) -> MLXArray {
        let row = MLX.arange(length).expandedDimensions(axis: 1)
        let col = MLX.arange(length).expandedDimensions(axis: 0)
        let window = windowSize ?? length
        let validRange = MLX.maximum(row - window + 1, 0)
        return ((col .>= validRange) & (col .<= row))
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = channelsFirst ? x.transposed(0, 2, 1) : x
        hidden = fishS1CallUnary(inputProj, hidden)
        hidden = fishS1CallUnary(lookAheadConv, hidden)

        let seqLen = hidden.shape[1]
        let localFreqs = freqsCis.map { $0[0..<seqLen] }
        var mask = makeWindowLimitedMask(seqLen)
        if mask.dtype == .bool {
            mask = MLX.where(mask, MLXArray(0.0, dtype: hidden.dtype), MLXArray(-1e9, dtype: hidden.dtype))
        }

        for layer in layers {
            hidden = layer(hidden, freqsCis: localFreqs, mask: mask)
        }
        hidden = norm(hidden)
        hidden = fishS1CallUnary(outputProj, hidden)
        return channelsFirst ? hidden.transposed(0, 2, 1) : hidden
    }
}
