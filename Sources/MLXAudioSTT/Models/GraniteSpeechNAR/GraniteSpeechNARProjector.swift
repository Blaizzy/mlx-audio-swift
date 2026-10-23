import MLX
import MLXNN

class GraniteSpeechNARCrossAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var out: Linear
    let numHeads: Int
    let headDim: Int

    init(_ config: GraniteSpeechNARProjectorConfig) {
        let dim = config.hiddenSize
        numHeads = config.numHeads
        headDim = dim / config.numHeads
        _q.wrappedValue = Linear(dim, dim, bias: config.attentionBias)
        _k.wrappedValue = Linear(dim, dim, bias: config.attentionBias)
        _v.wrappedValue = Linear(dim, dim, bias: config.attentionBias)
        _out.wrappedValue = Linear(dim, dim, bias: config.attentionBias)
    }
    func callAsFunction(_ query: MLXArray, keyValue: MLXArray) -> MLXArray {
        let b = query.dim(0)
        let length = query.dim(1)
        func heads(_ x: MLXArray) -> MLXArray {
            x.reshaped(b, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        }
        let attention = MLXFast.scaledDotProductAttention(
            queries: heads(q(query)), keys: heads(k(keyValue)), values: heads(v(keyValue)),
            scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return out(attention.transposed(0, 2, 1, 3).reshaped(b, length, -1)).asType(query.dtype)
    }
}

class GraniteSpeechNARProjectorMLP: Module {
    let fc1: Linear
    let fc2: Linear
    init(_ config: GraniteSpeechNARProjectorConfig) {
        fc1 = Linear(config.hiddenSize, config.hiddenSize * config.mlpRatio, bias: config.mlpBias)
        fc2 = Linear(config.hiddenSize * config.mlpRatio, config.hiddenSize, bias: config.mlpBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(silu(fc1(x))).asType(x.dtype) }
}

class GraniteSpeechNARProjectorLayer: Module {
    @ModuleInfo(key: "attn_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "cross_attention") var crossAttention: GraniteSpeechNARCrossAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    let mlp: GraniteSpeechNARProjectorMLP
    init(_ config: GraniteSpeechNARProjectorConfig) {
        _attentionNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _crossAttention.wrappedValue = GraniteSpeechNARCrossAttention(config)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        mlp = GraniteSpeechNARProjectorMLP(config)
    }
    func callAsFunction(_ query: MLXArray, keyValue: MLXArray) -> MLXArray {
        let h = query + crossAttention(attentionNorm(query), keyValue: keyValue)
        return (h + mlp(mlpNorm(h))).asType(query.dtype)
    }
}

class GraniteSpeechNARQFormer: Module {
    let layers: [GraniteSpeechNARProjectorLayer]
    init(_ config: GraniteSpeechNARProjectorConfig) {
        layers = (0 ..< config.numLayers).map { _ in GraniteSpeechNARProjectorLayer(config) }
    }
    func callAsFunction(_ query: MLXArray, keyValue: MLXArray) -> MLXArray {
        layers.reduce(query) { $1($0, keyValue: keyValue) }
    }
}

class GraniteSpeechNARProjector: Module {
    let config: GraniteSpeechNARProjectorConfig
    @ModuleInfo(key: "layer_norms") var layerNorms: [LayerNorm]
    @ModuleInfo(key: "layer_projector") var layerProjector: Linear
    var query: MLXArray
    @ParameterInfo(key: "window_positions") var windowPositions: MLXArray
    let qformer: GraniteSpeechNARQFormer
    @ModuleInfo(key: "out_norm") var outNorm: LayerNorm
    @ModuleInfo(key: "out_linear") var outLinear: Linear

    init(_ config: GraniteSpeechNARProjectorConfig) {
        self.config = config
        _layerNorms.wrappedValue = (0 ..< config.numEncoderLayers).map { _ in
            LayerNorm(dimensions: config.encoderDim, eps: config.layerNormEps)
        }
        _layerProjector.wrappedValue = Linear(
            config.numEncoderLayers * config.encoderDim, config.hiddenSize, bias: config.mlpBias)
        query = .zeros([1, config.blockSize / config.downsampleRate, config.hiddenSize])
        _windowPositions.wrappedValue = .zeros([1, config.blockSize, config.hiddenSize])
        qformer = GraniteSpeechNARQFormer(config)
        _outNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _outLinear.wrappedValue = Linear(config.hiddenSize, config.llmDim)
    }
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let b = hiddenStates.dim(0)
        let length = hiddenStates.dim(1)
        let block = config.blockSize
        let states = split(hiddenStates, parts: config.numEncoderLayers, axis: -1)
        let normalized = zip(layerNorms, states).map { $0($1) }
        var h = gelu(layerProjector(concatenated(normalized, axis: -1)))
        let padding = (block - length % block) % block
        if padding > 0 {
            h = padded(h, widths: [.init((0, 0)), .init((0, padding)), .init((0, 0))])
        }
        let blocks = h.dim(1) / block
        let queries = block / config.downsampleRate
        h = h.reshaped(b * blocks, block, config.hiddenSize)
        let means = h.reshaped(b * blocks, queries, config.downsampleRate, config.hiddenSize).mean(
            axis: -2)
        let q = query.asType(h.dtype) + means
        let kv = h + windowPositions.asType(h.dtype)
        let output = outLinear(outNorm(qformer(q, keyValue: kv)))
        return output.reshaped(b, blocks * queries, config.llmDim).asType(hiddenStates.dtype)
    }
}
