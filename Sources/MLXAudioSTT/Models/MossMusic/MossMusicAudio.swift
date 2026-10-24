import Foundation
import MLX
import MLXFast
import MLXNN

final class MossMusicAudioAttention: Module {
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    init(_ c: MossMusicAudioConfig) {
        heads = c.numAttentionHeads
        headDim = c.hiddenSize / heads
        _qProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let q = (qProj(x) * pow(Float(headDim), -0.5)).reshaped(b, t, heads, headDim).transposed(
            0, 2, 1, 3)
        let k = kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let h = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1, mask: .array(mask))
        return outProj(h.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

final class MossMusicAudioLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MossMusicAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    init(_ c: MossMusicAudioConfig) {
        _attention.wrappedValue = MossMusicAudioAttention(c)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize)
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(attentionNorm(x), mask: mask)
        return h + fc2(gelu(fc1(finalNorm(h))))
    }
}

final class MossMusicEncoder: Module {
    let config: MossMusicAudioConfig
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var conv3: Conv2d
    @ModuleInfo(key: "stem_proj") var stemProjection: Linear
    @ModuleInfo var layers: [MossMusicAudioLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "out_proj") var outProjection: Linear?
    private let _positions: MLXArray
    init(_ c: MossMusicAudioConfig) {
        config = c
        _conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: c.downsampleHiddenSize, kernelSize: 3, stride: 2,
            padding: 1)
        _conv2.wrappedValue = Conv2d(
            inputChannels: c.downsampleHiddenSize, outputChannels: c.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1)
        _conv3.wrappedValue = Conv2d(
            inputChannels: c.downsampleHiddenSize, outputChannels: c.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1)
        _stemProjection.wrappedValue = Linear(
            c.downsampleHiddenSize * MossMusicAudioConfig.outputLength(c.numMelBins), c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in MossMusicAudioLayer(c) }
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        if c.outputDim != c.hiddenSize {
            _outProjection.wrappedValue = Linear(c.hiddenSize, c.outputDim, bias: false)
        }
        let inv = exp(
            -log(Float(10000)) / Float(c.hiddenSize / 2 - 1)
                * MLXArray(0 ..< (c.hiddenSize / 2)).asType(.float32))
        let times =
            MLXArray(0 ..< c.maxSourcePositions).asType(.float32).expandedDimensions(axis: -1) * inv
        _positions = concatenated([sin(times), cos(times)], axis: -1)
    }
    func callAsFunction(_ features: MLXArray, lengths: [Int]) -> (MLXArray, [MLXArray]) {
        var x = features.expandedDimensions(axis: -1)
        x = gelu(conv3(gelu(conv2(gelu(conv1(x))))))
        let b = x.dim(0)
        let f = x.dim(1)
        let t = x.dim(2)
        let channels = x.dim(3)
        x = stemProjection(x.transposed(0, 2, 3, 1).reshaped(b, t, channels * f))
        let downsampled = lengths.map(MossMusicAudioConfig.outputLength)
        let length = downsampled.max()!
        x = x[0..., 0 ..< length, 0...]
        x = x + _positions[0 ..< length].asType(x.dtype).expandedDimensions(axis: 0)
        let valid =
            MLXArray(0 ..< length).expandedDimensions(axis: 0)
            .< MLXArray(downsampled.map(Int32.init)).expandedDimensions(axis: -1)
        let mask = MLX.where(
            valid.expandedDimensions(axes: [1, 2]), MLXArray(Float(0)), MLXArray(Float(-1e9))
        ).asType(x.dtype)
        var captured: [Int: MLXArray] = [:]
        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask)
            if config.deepstackLayerIndices.contains(i) { captured[i] = x }
        }
        x = norm(x)
        var deep = config.deepstackLayerIndices.compactMap { captured[$0] }
        if let outProjection {
            x = outProjection(x)
            deep = deep.map { outProjection($0) }
        }
        return (x, deep)
    }
}

final class MossMusicMerger: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: MossMusicConfig) {
        _gate.wrappedValue = Linear(c.audioConfig.outputDim, c.adapterHiddenSize, bias: false)
        _up.wrappedValue = Linear(c.audioConfig.outputDim, c.adapterHiddenSize, bias: false)
        _down.wrappedValue = Linear(c.adapterHiddenSize, c.languageConfig.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}
