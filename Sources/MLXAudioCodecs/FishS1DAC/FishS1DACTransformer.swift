import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RoPE Helpers

func fishPrecomputeFreqsCis(blockSize: Int, headDim: Int, ropeBase: Float) -> (MLXArray, MLXArray) {
    let halfDim = headDim / 2
    var freqs = [Float](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        freqs[i] = 1.0 / pow(ropeBase, Float(2 * i) / Float(headDim))
    }
    let freqsArray = MLXArray(freqs)

    var positions = [Float](repeating: 0, count: blockSize)
    for i in 0..<blockSize {
        positions[i] = Float(i)
    }
    let posArray = MLXArray(positions)

    // [blockSize, halfDim]
    let angles = posArray.reshaped([blockSize, 1]) * freqsArray.reshaped([1, halfDim])
    let cosFreqs = MLX.cos(angles)
    let sinFreqs = MLX.sin(angles)
    return (cosFreqs, sinFreqs)
}

func fishApplyRotaryEmb(_ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray) -> MLXArray {
    // x: [B, nHeads, seqLen, headDim]
    let headDim = x.dim(-1)
    let halfDim = headDim / 2

    let x1 = x[0..., 0..., 0..., ..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]

    let rotated = MLX.concatenated([
        x1 * cosFreqs - x2 * sinFreqs,
        x1 * sinFreqs + x2 * cosFreqs
    ], axis: -1)
    return rotated
}

// MARK: - Attention

public class FishAttention: Module {
    let nHead: Int
    let nLocalHeads: Int
    let headDim: Int
    let dim: Int
    let totalHeadDim: Int

    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    public init(config: FishS1DACConfig) {
        self.nHead = config.nHead
        self.nLocalHeads = config.nLocalHeads
        self.headDim = config.headDim
        self.dim = config.dim
        self.totalHeadDim = (config.nHead + 2 * config.nLocalHeads) * config.headDim

        self._wqkv.wrappedValue = Linear(config.dim, totalHeadDim, bias: false)
        self._wo.wrappedValue = Linear(config.nHead * config.headDim, config.dim, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (batchSize, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Project QKV
        let qkv = wqkv(x)

        // Split into Q, K, V
        let qDim = nHead * headDim
        let kvDim = nLocalHeads * headDim

        let q = qkv[0..., 0..., ..<qDim]
            .reshaped([batchSize, seqLen, nHead, headDim])
            .transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., qDim..<(qDim + kvDim)]
            .reshaped([batchSize, seqLen, nLocalHeads, headDim])
            .transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., (qDim + kvDim)...]
            .reshaped([batchSize, seqLen, nLocalHeads, headDim])
            .transposed(0, 2, 1, 3)

        // Apply RoPE
        let qRoped = fishApplyRotaryEmb(q, cosFreqs: cosFreqs, sinFreqs: sinFreqs)
        let kRoped = fishApplyRotaryEmb(k, cosFreqs: cosFreqs, sinFreqs: sinFreqs)

        // Expand KV for grouped query attention
        var kExpanded = kRoped
        var vExpanded = v
        if nLocalHeads != nHead {
            let repeats = nHead / nLocalHeads
            kExpanded = MLX.repeated(kRoped, count: repeats, axis: 1)
            vExpanded = MLX.repeated(v, count: repeats, axis: 1)
        }

        // Scaled dot-product attention
        let scale = sqrt(Float(headDim))
        let output = MLXFast.scaledDotProductAttention(
            queries: qRoped, keys: kExpanded, values: vExpanded,
            scale: 1.0 / scale, mask: mask
        )

        // Reshape and project output
        let outputReshaped = output.transposed(0, 2, 1, 3)
            .reshaped([batchSize, seqLen, nHead * headDim])
        return wo(outputReshaped)
    }
}

// MARK: - Feed-Forward (SwiGLU)

public class FishFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(dim: Int, intermediateSize: Int) {
        self._w1.wrappedValue = Linear(dim, intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(intermediateSize, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, intermediateSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Transformer Block

public class FishTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: FishAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FishFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: FishTFRMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: FishTFRMSNorm
    @ModuleInfo(key: "attention_layer_scale") var attentionLayerScale: FishLayerScale
    @ModuleInfo(key: "ffn_layer_scale") var ffnLayerScale: FishLayerScale

    public init(config: FishS1DACConfig) {
        self._attention.wrappedValue = FishAttention(config: config)
        self._feedForward.wrappedValue = FishFeedForward(
            dim: config.dim, intermediateSize: config.intermediateSize
        )
        self._attentionNorm.wrappedValue = FishTFRMSNorm(dim: config.dim, eps: config.normEps)
        self._ffnNorm.wrappedValue = FishTFRMSNorm(dim: config.dim, eps: config.normEps)
        self._attentionLayerScale.wrappedValue = FishLayerScale(dim: config.dim)
        self._ffnLayerScale.wrappedValue = FishLayerScale(dim: config.dim)
    }

    public func callAsFunction(
        _ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var h = x + attentionLayerScale(
            attention(attentionNorm(x), cosFreqs: cosFreqs, sinFreqs: sinFreqs, mask: mask)
        )
        h = h + ffnLayerScale(feedForward(ffnNorm(h)))
        return h
    }
}

// MARK: - Transformer

public class FishTransformer: Module {
    let config: FishS1DACConfig

    @ModuleInfo(key: "layers") var layers: [FishTransformerBlock]
    @ModuleInfo(key: "norm") var norm: FishTFRMSNorm

    public init(config: FishS1DACConfig) {
        self.config = config
        self._layers.wrappedValue = (0..<config.nLayer).map { _ in
            FishTransformerBlock(config: config)
        }
        self._norm.wrappedValue = FishTFRMSNorm(dim: config.dim, eps: config.normEps)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let seqLen = x.dim(1)
        let (cosFreqs, sinFreqs) = fishPrecomputeFreqsCis(
            blockSize: seqLen, headDim: config.headDim, ropeBase: config.ropeBase
        )

        var h = x
        for layer in layers {
            h = layer(h, cosFreqs: cosFreqs, sinFreqs: sinFreqs, mask: mask)
        }
        return norm(h)
    }
}

// MARK: - Window-Limited Transformer

/// Transformer with window-limited causal attention and optional input/output projections.
/// Operates on NCL tensors when channelsFirst=true.
public class FishWindowLimitedTransformer: Module {
    let channelsFirst: Bool
    let windowSize: Int
    let dim: Int

    @ModuleInfo(key: "transformer") var transformer: FishTransformer
    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "output_proj") var outputProj: Linear?

    public init(
        config: FishS1DACConfig,
        inputDim: Int? = nil,
        outputDim: Int? = nil,
        channelsFirst: Bool = true,
        windowSize: Int = 128
    ) {
        self.channelsFirst = channelsFirst
        self.windowSize = windowSize
        self.dim = config.dim

        self._transformer.wrappedValue = FishTransformer(config: config)

        if let inputDim = inputDim, inputDim != config.dim {
            self._inputProj.wrappedValue = Linear(inputDim, config.dim, bias: false)
        } else {
            self._inputProj.wrappedValue = nil
        }

        if let outputDim = outputDim, outputDim != config.dim {
            self._outputProj.wrappedValue = Linear(config.dim, outputDim, bias: false)
        } else {
            self._outputProj.wrappedValue = nil
        }
    }

    func makeWindowLimitedMask(maxLength: Int) -> MLXArray {
        // Create causal mask with window limitation
        var maskData = [[Float]](repeating: [Float](repeating: -Float.infinity, count: maxLength), count: maxLength)
        for i in 0..<maxLength {
            let start = max(0, i - windowSize + 1)
            for j in start...i {
                maskData[i][j] = 0.0
            }
        }

        let flat = maskData.flatMap { $0 }
        return MLXArray(flat, [maxLength, maxLength])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x

        // NCL -> NLC if needed
        if channelsFirst {
            h = h.transposed(0, 2, 1)
        }

        if let proj = inputProj {
            h = proj(h)
        }

        let seqLen = h.dim(1)
        let mask = makeWindowLimitedMask(maxLength: seqLen)
        h = transformer(h, mask: mask)

        if let proj = outputProj {
            h = proj(h)
        }

        // NLC -> NCL if needed
        if channelsFirst {
            h = h.transposed(0, 2, 1)
        }

        return h
    }
}
