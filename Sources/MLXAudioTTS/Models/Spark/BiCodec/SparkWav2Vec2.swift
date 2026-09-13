import Foundation
@preconcurrency import MLX
import MLXNN

private func geluExact(_ x: MLXArray) -> MLXArray {
    x * 0.5 * (1.0 + MLX.erf(x / MLXArray(Float(2.0).squareRoot())))
}

private final class SparkW2VConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(inDim: Int, outDim: Int, kernel: Int, stride: Int) {
        self._conv = ModuleInfo(
            wrappedValue: Conv1d(inputChannels: inDim, outputChannels: outDim, kernelSize: kernel, stride: stride, bias: true),
            key: "conv")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: outDim), key: "layer_norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x.swappedAxes(-2, -1))
        h = layerNorm(h).swappedAxes(-2, -1)
        return geluExact(h)
    }
}

private final class SparkW2VWNConv: Module {
    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray
    let padding: Int
    let groups: Int

    init(inChannels: Int, outChannels: Int, kernel: Int, padding: Int, groups: Int) {
        self.padding = padding
        self.groups = groups
        self.weight_g = MLXArray.zeros([1, kernel, 1])
        self.weight_v = MLXArray.zeros([outChannels, kernel, inChannels / groups])
        self.bias = MLXArray.zeros([outChannels])
    }

    private func normExceptDim1(_ x: MLXArray) -> MLXArray {
        MLX.sqrt(MLX.sum(x * x, axes: [0, 2], keepDims: true))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weight_g * weight_v / normExceptDim1(weight_v)
        return MLX.conv1d(x, weight, stride: 1, padding: padding, dilation: 1, groups: groups) + bias
    }
}

private final class SparkW2VPosConv: Module {
    @ModuleInfo(key: "conv") var conv: SparkW2VWNConv
    private let padRemove: Int

    init(hiddenSize: Int, kernel: Int, groups: Int) {
        self.padRemove = kernel % 2 == 0 ? 1 : 0
        self._conv = ModuleInfo(
            wrappedValue: SparkW2VWNConv(
                inChannels: hiddenSize, outChannels: hiddenSize, kernel: kernel,
                padding: kernel / 2, groups: groups),
            key: "conv")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        if padRemove > 0 { h = h[0..., 0 ..< (h.shape[1] - padRemove), 0...] }
        return geluExact(h)
    }
}

private final class SparkW2VAttention: Module {
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

private final class SparkW2VFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._intermediateDense = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize), key: "intermediate_dense")
        self._outputDense = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize), key: "output_dense")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(geluExact(intermediateDense(x)))
    }
}

private final class SparkW2VEncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: SparkW2VAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: SparkW2VFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int, eps: Float) {
        self._attention = ModuleInfo(wrappedValue: SparkW2VAttention(embedDim: hiddenSize, numHeads: numHeads), key: "attention")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps), key: "layer_norm")
        self._feedForward = ModuleInfo(wrappedValue: SparkW2VFeedForward(hiddenSize: hiddenSize, intermediateSize: intermediateSize), key: "feed_forward")
        self._finalLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps), key: "final_layer_norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(layerNorm(x))
        h = h + feedForward(finalLayerNorm(h))
        return h
    }
}

/// Wav2Vec2-large-xlsr-53 encoder returning the hidden-state mix (layers 11/14/16)
/// the Spark BiCodec tokenizer uses as feature input.
public final class SparkWav2Vec2: Module {
    @ModuleInfo(key: "feature_extractor") fileprivate var featureExtractor: SparkW2VFeatureEncoder
    @ModuleInfo(key: "feature_projection") fileprivate var featureProjection: SparkW2VFeatureProjection
    @ModuleInfo(key: "encoder") fileprivate var encoder: SparkW2VEncoder

    public override init() {
        self._featureExtractor = ModuleInfo(wrappedValue: SparkW2VFeatureEncoder(), key: "feature_extractor")
        self._featureProjection = ModuleInfo(wrappedValue: SparkW2VFeatureProjection(), key: "feature_projection")
        self._encoder = ModuleInfo(wrappedValue: SparkW2VEncoder(), key: "encoder")
    }

    /// Raw 16 kHz waveform [N] -> feature mix [1, T, 1024].
    public func features(_ wav: MLXArray) -> MLXArray {
        let mono = wav.ndim > 1 ? wav.reshaped([-1]) : wav
        let mean = mono.mean()
        let variance = MLX.mean(MLX.square(mono - mean))
        let normed = (mono - mean) / MLX.sqrt(variance + MLXArray(Float(1e-7)))
        let inputValues = normed.reshaped([1, 1, mono.shape[0]])

        var h = featureExtractor(inputValues).transposed(0, 2, 1)
        h = featureProjection(h)
        return encoder.hiddenStateMix(h, layers: [11, 14, 16])
    }

    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (rawKey, rawValue) in weights {
            var key = rawKey
            if key.hasPrefix("wav2vec2.") { key = String(key.dropFirst("wav2vec2.".count)) }
            if key.hasPrefix("lm_head.") || key.hasPrefix("quantizer.")
                || key.hasPrefix("project_") || key == "masked_spec_embed"
                || key.contains(".adapter_layer.") { continue }
            var v = rawValue
            if key.hasSuffix(".conv.weight") { v = v.swappedAxes(1, 2) }
            if key.hasSuffix(".parametrizations.weight.original0") {
                key = key.replacingOccurrences(of: ".parametrizations.weight.original0", with: ".weight_g")
                v = v.swappedAxes(1, 2)
            }
            if key.hasSuffix(".parametrizations.weight.original1") {
                key = key.replacingOccurrences(of: ".parametrizations.weight.original1", with: ".weight_v")
                v = v.swappedAxes(1, 2)
            }
            out[key] = v
        }
        return out
    }
}

private final class SparkW2VFeatureEncoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [SparkW2VConvLayer]

    override init() {
        let dims = [512, 512, 512, 512, 512, 512, 512]
        let strides = [5, 2, 2, 2, 2, 2, 2]
        let kernels = [10, 3, 3, 3, 3, 2, 2]
        var layers: [SparkW2VConvLayer] = []
        for i in 0 ..< dims.count {
            let inDim = i == 0 ? 1 : dims[i - 1]
            layers.append(SparkW2VConvLayer(inDim: inDim, outDim: dims[i], kernel: kernels[i], stride: strides[i]))
        }
        self._convLayers = ModuleInfo(wrappedValue: layers, key: "conv_layers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in convLayers { h = layer(h) }
        return h
    }
}

private final class SparkW2VFeatureProjection: Module {
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

private final class SparkW2VEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: SparkW2VPosConv
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [SparkW2VEncoderLayer]

    override init() {
        self._posConvEmbed = ModuleInfo(
            wrappedValue: SparkW2VPosConv(hiddenSize: 1024, kernel: 128, groups: 16), key: "pos_conv_embed")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: 1024, eps: 1e-5), key: "layer_norm")
        self._layers = ModuleInfo(
            wrappedValue: (0 ..< 24).map { _ in
                SparkW2VEncoderLayer(hiddenSize: 1024, numHeads: 16, intermediateSize: 4096, eps: 1e-5)
            }, key: "layers")
    }

    func hiddenStateMix(_ x: MLXArray, layers wanted: [Int]) -> MLXArray {
        var h = x + posConvEmbed(x)
        var all: [MLXArray] = []
        for layer in layers {
            all.append(h)
            h = layer(h)
        }
        all.append(layerNorm(h))
        var sum = all[wanted[0]]
        for idx in wanted.dropFirst() { sum = sum + all[idx] }
        return sum / MLXArray(Float(wanted.count))
    }
}
