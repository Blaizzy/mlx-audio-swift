import Foundation
@preconcurrency import MLX
import MLXNN

private func tadaAlignerGELU(_ x: MLXArray) -> MLXArray {
    x * 0.5 * (1.0 + MLX.erf(x / MLXArray(Float(2.0).squareRoot())))
}

final class TADAW2VGroupNorm: Module {
    var weight = MLXArray.ones([512])
    var bias = MLXArray.zeros([512])
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let m = mean(x, axis: 1, keepDims: true)
        let v = mean(square(x - m), axis: 1, keepDims: true)
        return (x - m) * rsqrt(v + 1e-5) * weight + bias
    }
}
final class TADAW2VConvLayer: Module {
    let conv: Conv1d
    @ModuleInfo(key: "layer_norm") var norm: TADAW2VGroupNorm?
    init(inDim: Int, outDim: Int, kernel: Int, stride: Int) {
        conv = Conv1d(inputChannels: inDim, outputChannels: outDim, kernelSize: kernel, stride: stride, bias: false)
        _norm = ModuleInfo(wrappedValue: inDim == 1 ? TADAW2VGroupNorm() : nil, key: "layer_norm")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = conv(x)
        return tadaAlignerGELU(norm?(h) ?? h)
    }
}
final class TADAW2VPosConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    private let padRemove: Int

    init(hiddenSize: Int, kernel: Int, groups: Int) {
        self.padRemove = kernel % 2 == 0 ? 1 : 0
        self._conv = ModuleInfo(
            wrappedValue: Conv1d(inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: kernel, padding: kernel / 2, groups: groups),
            key: "conv")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        if padRemove > 0 { h = h[0..., 0 ..< (h.shape[1] - padRemove), 0...] }
        return tadaAlignerGELU(h)
    }
}

final class TADAW2VAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scaling: Float

    init(embedDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scaling = powf(Float(headDim), -0.5)
        self._qProj = ModuleInfo(wrappedValue: Linear(embedDim, embedDim), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(embedDim, embedDim), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(embedDim, embedDim), key: "v_proj")
        self._outProj = ModuleInfo(wrappedValue: Linear(embedDim, embedDim), key: "out_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0], t = x.shape[1]
        let q = (qProj(x) * scaling).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        let scores = MLX.matmul(q, k.transposed(0, 1, 3, 2))
        let weights = MLX.softmax(scores, axis: -1)
        let out = MLX.matmul(weights, v).transposed(0, 2, 1, 3).reshaped([b, t, numHeads * headDim])
        return outProj(out)
    }
}

final class TADAW2VFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._intermediateDense = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize), key: "intermediate_dense")
        self._outputDense = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize), key: "output_dense")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(tadaAlignerGELU(intermediateDense(x)))
    }
}

final class TADAW2VEncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: TADAW2VAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: TADAW2VFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int, eps: Float) {
        self._attention = ModuleInfo(wrappedValue: TADAW2VAttention(embedDim: hiddenSize, numHeads: numHeads), key: "attention")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps), key: "layer_norm")
        self._feedForward = ModuleInfo(wrappedValue: TADAW2VFeedForward(hiddenSize: hiddenSize, intermediateSize: intermediateSize), key: "feed_forward")
        self._finalLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps), key: "final_layer_norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = layerNorm(x + attention(x))
        return finalLayerNorm(h + feedForward(h))
    }
}

final class TADAW2VFeatureEncoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [TADAW2VConvLayer]

    override init() {
        let dims = [512, 512, 512, 512, 512, 512, 512]
        let strides = [5, 2, 2, 2, 2, 2, 2]
        let kernels = [10, 3, 3, 3, 3, 2, 2]
        var layers: [TADAW2VConvLayer] = []
        for i in 0 ..< dims.count {
            let inDim = i == 0 ? 1 : dims[i - 1]
            layers.append(TADAW2VConvLayer(inDim: inDim, outDim: dims[i], kernel: kernels[i], stride: strides[i]))
        }
        self._convLayers = ModuleInfo(wrappedValue: layers, key: "conv_layers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in convLayers { h = layer(h) }
        return h
    }
}

final class TADAW2VFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    override init() {
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: 512, eps: 1e-5), key: "layer_norm")
        self._projection = ModuleInfo(wrappedValue: Linear(512, 1024), key: "projection")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}

final class TADAW2VEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: TADAW2VPosConv
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [TADAW2VEncoderLayer]

    override init() {
        self._posConvEmbed = ModuleInfo(
            wrappedValue: TADAW2VPosConv(hiddenSize: 1024, kernel: 128, groups: 16), key: "pos_conv_embed")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: 1024, eps: 1e-5), key: "layer_norm")
        self._layers = ModuleInfo(
            wrappedValue: (0 ..< 24).map { _ in
                TADAW2VEncoderLayer(hiddenSize: 1024, numHeads: 16, intermediateSize: 4096, eps: 1e-5)
            }, key: "layers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + posConvEmbed(x)
        h = layerNorm(h)
        for layer in layers { h = layer(h) }
        return h
    }
}

final class TADAW2VNetwork: Module {
    @ModuleInfo(key: "feature_extractor") var extractor: TADAW2VFeatureEncoder
    @ModuleInfo(key: "feature_projection") var projection: TADAW2VFeatureProjection
    let encoder: TADAW2VEncoder
    @ModuleInfo(key: "lm_head") var head: Linear
    override init() {
        _extractor = ModuleInfo(wrappedValue: TADAW2VFeatureEncoder(), key: "feature_extractor")
        _projection = ModuleInfo(wrappedValue: TADAW2VFeatureProjection(), key: "feature_projection")
        encoder = TADAW2VEncoder()
        _head = ModuleInfo(wrappedValue: Linear(1024, 128256), key: "lm_head")
    }
    func hidden(_ audio: MLXArray) -> MLXArray {
        encoder(projection(extractor(audio.reshaped(1, -1, 1))))
    }
}

/// Native Wav2Vec2 CTC aligner bundled with the published TADA MLX packages.
final class TADAAligner: Module {
    let wav2vec2 = TADAW2VNetwork()
    func callAsFunction(_ audio: MLXArray) -> MLXArray { wav2vec2.head(wav2vec2.hidden(audio)) }
}
