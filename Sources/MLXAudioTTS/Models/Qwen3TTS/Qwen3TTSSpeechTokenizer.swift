import Foundation
@preconcurrency import MLX
import MLXAudioCodecs
@preconcurrency import MLXLMCommon
import MLXNN

// MARK: - Vector Quantization

final class VectorQuantization: Module {
    @ModuleInfo(key: "project_out") var projectOut: Linear?
    @ModuleInfo var codebook: EuclideanCodebook
    let codebookDim: Int

    init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let cbDim = codebookDim ?? dim
        self.codebookDim = cbDim
        if cbDim != dim {
            _projectOut.wrappedValue = Linear(cbDim, dim)
        } else {
            _projectOut.wrappedValue = nil
        }
        _codebook.wrappedValue = EuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = codebook.decode(codes) // [batch, time, codebook_dim]
        if let proj = projectOut {
            quantized = proj(quantized)
        }
        return quantized.transposed(0, 2, 1) // [batch, dim, time]
    }
}

final class ResidualVectorQuantization: Module {
    @ModuleInfo var layers: [VectorQuantization]

    init(numQuantizers: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        _layers.wrappedValue = (0 ..< numQuantizers).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [num_quantizers, batch, time]
        var quantized = MLXArray.zeros([codes.dim(1), layers[0].codebookDim, codes.dim(2)])
        for (idx, layer) in layers.enumerated() {
            quantized = quantized + layer.decode(codes[idx])
        }
        return quantized
    }
}

final class ResidualVectorQuantizer: Module {
    let dimension: Int
    @ModuleInfo(key: "input_proj") var inputProj: MLXNN.Conv1d?
    @ModuleInfo(key: "output_proj") var outputProj: MLXNN.Conv1d?
    @ModuleInfo var vq: ResidualVectorQuantization

    init(dimension: Int = 128, inputDimension: Int? = nil, outputDimension: Int? = nil,
         nQ: Int = 8, bins: Int = 1024, forceProjection: Bool = false) {
        let inDim = inputDimension ?? dimension
        let outDim = outputDimension ?? dimension
        self.dimension = dimension

        if inDim == dimension, !forceProjection {
            _inputProj.wrappedValue = nil
        } else {
            _inputProj.wrappedValue = MLXNN.Conv1d(inputChannels: inDim, outputChannels: dimension, kernelSize: 1, bias: false)
        }
        if outDim == dimension, !forceProjection {
            _outputProj.wrappedValue = nil
        } else {
            _outputProj.wrappedValue = MLXNN.Conv1d(inputChannels: dimension, outputChannels: outDim, kernelSize: 1, bias: false)
        }

        _vq.wrappedValue = ResidualVectorQuantization(numQuantizers: nQ, dim: dimension, codebookSize: bins)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, num_quantizers, time]
        let transposed = codes.transposed(1, 0, 2) // [num_quantizers, batch, time]
        var quantized = vq.decode(transposed) // [batch, dim, time]
        if let proj = outputProj {
            // Conv1d expects NLC: [batch, time, channels]
            quantized = proj(quantized.transposed(0, 2, 1)).transposed(0, 2, 1)
        }
        return quantized
    }
}

final class SplitResidualVectorQuantizer: Module {
    let nQSemantic: Int
    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    init(nQ: Int = 8, nQSemantic: Int = 1, dimension: Int = 128,
         inputDimension: Int? = nil, outputDimension: Int? = nil, bins: Int = 1024) {
        self.nQSemantic = nQSemantic
        _rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension, inputDimension: inputDimension, outputDimension: outputDimension,
            nQ: nQSemantic, bins: bins, forceProjection: true
        )
        _rvqRest.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension, inputDimension: inputDimension, outputDimension: outputDimension,
            nQ: nQ - nQSemantic, bins: bins, forceProjection: true
        )
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, num_quantizers, time]
        var quantized = rvqFirst.decode(codes[0..., ..<nQSemantic])
        if codes.dim(1) > nQSemantic {
            quantized = quantized + rvqRest.decode(codes[0..., nQSemantic...])
        }
        return quantized
    }
}

// MARK: - Causal Convolutions

/// Container for depthwise conv weights to match PyTorch key structure
final class DepthwiseConvWeight: Module {
    var weight: MLXArray
    var bias: MLXArray

    init(outChannels: Int, kernelSize: Int, inPerGroup: Int) {
        self.weight = MLXArray.zeros([outChannels, kernelSize, inPerGroup])
        self.bias = MLXArray.zeros([outChannels])
    }
}

final class CausalConv1d: Module {
    let groups: Int
    let inChannels: Int
    let outChannels: Int
    let stride: Int
    let kernelSizeVal: Int
    let effectiveKernelSize: Int
    let dilation: Int
    let paddingAmount: Int

    // Use either regular conv or depthwise weight
    @ModuleInfo var conv: Module

    init(inChannels: Int, outChannels: Int, kernelSize: Int,
         stride: Int = 1, dilation: Int = 1, groups: Int = 1) {
        self.groups = groups
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.stride = stride
        self.kernelSizeVal = kernelSize
        self.effectiveKernelSize = (kernelSize - 1) * dilation + 1
        self.dilation = dilation
        self.paddingAmount = effectiveKernelSize - stride

        if groups == 1 {
            _conv.wrappedValue = MLXNN.Conv1d(
                inputChannels: inChannels, outputChannels: outChannels,
                kernelSize: kernelSize, stride: stride, padding: 0, dilation: dilation
            )
        } else {
            let inPerGroup = inChannels / groups
            _conv.wrappedValue = DepthwiseConvWeight(outChannels: outChannels, kernelSize: kernelSize, inPerGroup: inPerGroup)
        }
    }

    private func getExtraPadding(_ length: Int) -> Int {
        let nFrames = Float(length - effectiveKernelSize + paddingAmount) / Float(stride) + 1
        let idealLength = (Int(ceil(nFrames)) - 1) * stride + (effectiveKernelSize - paddingAmount)
        return idealLength - length
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        let extra = getExtraPadding(x.dim(-1))
        var result = padded(x, widths: [.init(0), .init(0), .init((paddingAmount, extra))])

        if groups == 1 {
            // MLX Conv1d expects NLC
            result = result.transposed(0, 2, 1)
            result = (conv as! MLXNN.Conv1d)(result)
            return result.transposed(0, 2, 1)
        } else {
            // Depthwise convolution
            let dwConv = conv as! DepthwiseConvWeight
            let (_, channels, time) = (result.dim(0), result.dim(1), result.dim(2))
            let kSize = dwConv.weight.dim(1)
            let outputTime = time - kSize + 1

            let windows = stacked((0 ..< kSize).map { i in result[0..., 0..., i ..< (i + outputTime)] }, axis: -1)
            let w = dwConv.weight.squeezed(axis: -1) // [channels, kernel]
            let out = (windows * w.reshaped(1, channels, 1, kSize)).sum(axis: -1)
            return out + dwConv.bias.reshaped(1, channels, 1)
        }
    }
}

// MARK: - SnakeBeta activation

final class SnakeBeta: Module {
    var alpha: MLXArray
    var beta: MLXArray
    let eps: Float = 1e-9

    init(channels: Int) {
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let a = exp(alpha).reshaped(1, -1, 1)
        let b = exp(beta).reshaped(1, -1, 1)
        let sinVal = MLX.sin(x * a)
        return x + (1.0 / (b + eps)) * sinVal * sinVal
    }
}

// MARK: - ConvNeXt Block

final class ConvNeXtBlock: Module {
    @ModuleInfo var dwconv: CausalConv1d
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    var gamma: MLXArray

    init(dim: Int) {
        _dwconv.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, groups: dim)
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _pwconv1.wrappedValue = Linear(dim, 4 * dim)
        _pwconv2.wrappedValue = Linear(4 * dim, dim)
        self.gamma = MLXArray.ones([dim]) * 1e-6
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = dwconv(x)
        h = h.transposed(0, 2, 1) // [B, T, C]
        h = norm(h)
        h = gelu(pwconv1(h))
        h = gamma * pwconv2(h)
        h = h.transposed(0, 2, 1) // [B, C, T]
        return residual + h
    }
}

// MARK: - Decoder Transformer

final class DecoderRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let v = mean(xf * xf, axis: -1, keepDims: true)
        return (weight * (xf * rsqrt(v + eps))).asType(x.dtype)
    }
}

final class LayerScale: Module {
    var scale: MLXArray

    init(channels: Int, initialScale: Float = 0.01) {
        self.scale = MLXArray.ones([channels]) * initialScale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        scale * x
    }
}

final class DecoderRotaryEmbedding: Module {
    let _invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int = 8000, base: Float = 10000.0) {
        let arange = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
        let exponent = arange / Float(dim)
        self._invFreq = 1.0 / MLXArray(base).pow(exponent)
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let inv = _invFreq.reshaped(1, -1, 1).asType(.float32)
        let pos = positionIds[0..., .newAxis, 0...].asType(.float32)
        let freqs = matmul(inv, pos).transposed(0, 2, 1)
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(x.dtype), MLX.sin(emb).asType(x.dtype))
    }
}

func decoderRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

final class DecoderAttention: Module {
    let headDim: Int
    let numHeads: Int
    let numKvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self.headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, seqLen, numKvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, seqLen, numKvHeads, headDim).transposed(0, 2, 1, 3)

        let (cosVal, sinVal) = positionEmbeddings
        let cosE = expandedDimensions(cosVal, axis: 1)
        let sinE = expandedDimensions(sinVal, axis: 1)
        q = q * cosE + decoderRotateHalf(q) * sinE
        k = k * cosE + decoderRotateHalf(k) * sinE

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
    }
}

final class DecoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTokenizerDecoderConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class DecoderTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DecoderAttention
    @ModuleInfo var mlp: DecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: DecoderRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: DecoderRMSNorm
    @ModuleInfo(key: "self_attn_layer_scale") var selfAttnLayerScale: LayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpLayerScale: LayerScale

    init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        _selfAttn.wrappedValue = DecoderAttention(config: config, layerIdx: layerIdx)
        _mlp.wrappedValue = DecoderMLP(config: config)
        _inputLayernorm.wrappedValue = DecoderRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = DecoderRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttnLayerScale.wrappedValue = LayerScale(channels: config.hiddenSize, initialScale: config.layerScaleInitialScale)
        _mlpLayerScale.wrappedValue = LayerScale(channels: config.hiddenSize, initialScale: config.layerScaleInitialScale)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        var out = x + selfAttnLayerScale(selfAttn(inputLayernorm(x), positionEmbeddings: positionEmbeddings, mask: mask, cache: cache))
        out = out + mlpLayerScale(mlp(postAttentionLayernorm(out)))
        return out
    }
}

final class DecoderTransformer: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let layers: [DecoderTransformerLayer]
    @ModuleInfo var norm: DecoderRMSNorm
    let rotaryEmb: DecoderRotaryEmbedding
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self.layers = (0 ..< config.numHiddenLayers).map { DecoderTransformerLayer(config: config, layerIdx: $0) }
        _norm.wrappedValue = DecoderRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = DecoderRotaryEmbedding(dim: config.headDim, maxPositionEmbeddings: config.maxPositionEmbeddings, base: config.ropeTheta)
        _inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize)
        _outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim)
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (inputsEmbeds.dim(0), inputsEmbeds.dim(1), inputsEmbeds.dim(2))

        var x = inputProj(inputsEmbeds)

        let offset = cache?.first?.offset ?? 0
        let posIds = broadcast(
            MLXArray(Int32(offset) ..< Int32(offset + seqLen)).reshaped(1, seqLen),
            to: [batch, seqLen]
        )
        let posEmb = rotaryEmb(x, positionIds: posIds)

        var causalMask = mask
        if causalMask == nil, seqLen > 1 {
            causalMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(x.dtype)
        }

        for (i, layer) in layers.enumerated() {
            x = layer(x, positionEmbeddings: posEmb, mask: causalMask, cache: cache?[i])
        }
        return outputProj(norm(x))
    }

    func makeCache() -> [any KVCache] {
        layers.map { _ in KVCacheSimple() }
    }
}

// MARK: - Decoder Blocks

final class DecoderResidualUnit: Module {
    @ModuleInfo var act1: SnakeBeta
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var act2: SnakeBeta
    @ModuleInfo var conv2: CausalConv1d

    init(dim: Int, dilation: Int = 1) {
        _act1.wrappedValue = SnakeBeta(channels: dim)
        _conv1.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, dilation: dilation)
        _act2.wrappedValue = SnakeBeta(channels: dim)
        _conv2.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(act2(conv1(act1(x))))
    }
}

/// Upsample conv wrapper matching PyTorch key structure: block.1.conv.*
final class DecoderBlockUpsample: Module {
    @ModuleInfo var conv: ConvTransposed1d
    let trimRight: Int

    init(inDim: Int, outDim: Int, upsampleRate: Int) {
        let kernelSize = 2 * upsampleRate
        _conv.wrappedValue = ConvTransposed1d(inputChannels: inDim, outputChannels: outDim, kernelSize: kernelSize, stride: upsampleRate, padding: 0)
        self.trimRight = kernelSize - upsampleRate
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: NCL → NLC for ConvTransposed1d
        var h = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        if trimRight > 0 {
            h = h[0..., 0..., ..<(-trimRight)]
        }
        return h
    }
}

final class DecoderBlock: Module {
    @ModuleInfo var block: [Module]

    init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        let inDim = config.decoderDim / (1 << layerIdx)
        let outDim = config.decoderDim / (1 << (layerIdx + 1))
        let upsampleRate = config.upsampleRates[layerIdx]

        _block.wrappedValue = [
            SnakeBeta(channels: inDim),
            DecoderBlockUpsample(inDim: inDim, outDim: outDim, upsampleRate: upsampleRate),
            DecoderResidualUnit(dim: outDim, dilation: 1),
            DecoderResidualUnit(dim: outDim, dilation: 3),
            DecoderResidualUnit(dim: outDim, dilation: 9),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let snake = layer as? SnakeBeta { h = snake(h) }
            else if let upsample = layer as? DecoderBlockUpsample { h = upsample(h) }
            else if let resUnit = layer as? DecoderResidualUnit { h = resUnit(h) }
        }
        return h
    }
}

/// Initial conv: decoder.decoder.0.conv.*
final class DecoderInitialConv: Module {
    @ModuleInfo var conv: MLXNN.Conv1d
    let kernelSize: Int

    init(latentDim: Int, decoderDim: Int, kernelSize: Int = 7) {
        _conv.wrappedValue = MLXNN.Conv1d(inputChannels: latentDim, outputChannels: decoderDim, kernelSize: kernelSize, padding: 0)
        self.kernelSize = kernelSize
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: NCL, left-pad for causal
        let h = padded(x, widths: [.init(0), .init(0), .init((kernelSize - 1, 0))])
        return conv(h.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

/// Output snake: decoder.decoder.5.*
final class DecoderOutputSnake: Module {
    var alpha: MLXArray
    var beta: MLXArray
    let eps: Float = 1e-9

    init(channels: Int) {
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha).reshaped(1, -1, 1)
        let b = exp(beta).reshaped(1, -1, 1)
        let sinVal = MLX.sin(x * a)
        return x + (1.0 / (b + eps)) * sinVal * sinVal
    }
}

/// Output conv: decoder.decoder.6.conv.*
final class DecoderOutputConv: Module {
    @ModuleInfo var conv: MLXNN.Conv1d
    let kernelSize: Int

    init(channels: Int, kernelSize: Int = 7) {
        _conv.wrappedValue = MLXNN.Conv1d(inputChannels: channels, outputChannels: 1, kernelSize: kernelSize, padding: 0)
        self.kernelSize = kernelSize
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = padded(x, widths: [.init(0), .init(0), .init((kernelSize - 1, 0))])
        return conv(h.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

// MARK: - Causal Transpose Conv (for upsampling blocks)

final class CausalTransposeConv1d: Module {
    @ModuleInfo var conv: ConvTransposed1d
    let trimRight: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        _conv.wrappedValue = ConvTransposed1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize, stride: stride, padding: 0)
        self.trimRight = kernelSize - stride
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        if trimRight > 0 {
            h = h[0..., 0..., ..<(-trimRight)]
        }
        return h
    }
}

// MARK: - Upsample Layer (CausalTransposeConv + ConvNeXt)

final class UpsampleLayer: Module {
    @ModuleInfo var layers: [Module]

    init(latentDim: Int, factor: Int) {
        _layers.wrappedValue = [
            CausalTransposeConv1d(inChannels: latentDim, outChannels: latentDim,
                                  kernelSize: factor, stride: factor),
            ConvNeXtBlock(dim: latentDim),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let ct = layer as? CausalTransposeConv1d { h = ct(h) }
            else if let cn = layer as? ConvNeXtBlock { h = cn(h) }
        }
        return h
    }
}

// MARK: - Speech Tokenizer Encoder

final class Qwen3TTSSpeechTokenizerEncoder: Module {
    let config: Qwen3TTSTokenizerEncoderConfig
    let validNumQuantizers: Int
    @ModuleInfo var encoder: SeanetEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: MLXAudioCodecs.ProjectedTransformer
    @ModuleInfo var downsample: MLXAudioCodecs.ConvDownsample1d
    @ModuleInfo var quantizer: MLXAudioCodecs.SplitResidualVectorQuantizer

    let encoderCache: [any KVCache]

    init(config: Qwen3TTSTokenizerEncoderConfig, validNumQuantizers: Int) {
        self.config = config
        self.validNumQuantizers = validNumQuantizers

        let ratioProduct = config.upsamplingRatios.reduce(1, *)
        let encoderFrameRate = Double(config.samplingRate) / Double(ratioProduct)
        let downsampleStride = max(1, Int(encoderFrameRate / Double(config.frameRate)))

        let seanetCfg = SeanetConfig(
            dimension: config.hiddenSize,
            channels: config.audioChannels,
            causal: config.useCausalConv,
            nfilters: config.numFilters,
            nresidualLayers: config.numResidualLayers,
            ratios: config.upsamplingRatios,
            ksize: config.kernelSize,
            residualKsize: config.residualKernelSize,
            lastKsize: config.lastKernelSize,
            dilationBase: config.dilationGrowthRate,
            padMode: .constant,
            trueSkip: !config.useConvShortcut,
            compress: config.compress
        )
        _encoder.wrappedValue = SeanetEncoder(cfg: seanetCfg)

        let transformerCfg = TransformerConfig(
            dModel: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            causal: config.useCausalConv,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: config.layerScaleInitialScale,
            positionalEmbedding: "rope",
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: true,
            gating: false,
            norm: "layer_norm",
            context: config.slidingWindow,
            maxPeriod: Int(config.ropeTheta),
            maxSeqLen: config.maxPositionEmbeddings,
            kvRepeat: config.numAttentionHeads / config.numKeyValueHeads,
            dimFeedforward: config.intermediateSize,
            convLayout: true
        )
        let projectedTransformer = MLXAudioCodecs.ProjectedTransformer(
            cfg: transformerCfg,
            inputDim: config.hiddenSize,
            outputDims: [config.hiddenSize]
        )
        _encoderTransformer.wrappedValue = projectedTransformer

        _downsample.wrappedValue = MLXAudioCodecs.ConvDownsample1d(
            stride: downsampleStride,
            dim: config.hiddenSize,
            causal: config.useCausalConv
        )
        _quantizer.wrappedValue = MLXAudioCodecs.SplitResidualVectorQuantizer(
            dim: config.codebookDim,
            inputDim: config.hiddenSize,
            outputDim: config.hiddenSize,
            nq: config.numQuantizers,
            bins: config.codebookSize
        )
        self.encoderCache = projectedTransformer.makeCache()
    }

    func encode(_ audio: MLXArray) -> MLXArray {
        encoder.resetState()
        for cache in encoderCache { cache.trim(cache.offset) }

        var hidden = encoder(audio)
        hidden = encoderTransformer(hidden, cache: encoderCache)[0]
        hidden = downsample(hidden)

        let codes = quantizer.encode(hidden)
        let selected = min(validNumQuantizers, codes.dim(1))
        return codes[0..., 0 ..< selected, 0...]
    }
}

// MARK: - Full Speech Tokenizer Decoder

final class Qwen3TTSSpeechTokenizerDecoder: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let totalUpsample: Int

    @ModuleInfo(key: "pre_transformer") var preTransformer: DecoderTransformer
    @ModuleInfo var quantizer: SplitResidualVectorQuantizer
    @ModuleInfo(key: "pre_conv") var preConv: CausalConv1d
    @ModuleInfo var upsample: [UpsampleLayer]
    @ModuleInfo var decoder: [Module]

    init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self.totalUpsample = (config.upsampleRates + config.upsamplingRatios).reduce(1, *)

        _preTransformer.wrappedValue = DecoderTransformer(config: config)
        _quantizer.wrappedValue = SplitResidualVectorQuantizer(
            nQ: config.numQuantizers,
            nQSemantic: config.numSemanticQuantizers,
            dimension: config.codebookDim / 2,
            inputDimension: config.codebookDim,
            outputDimension: config.codebookDim,
            bins: config.codebookSize
        )
        _preConv.wrappedValue = CausalConv1d(inChannels: config.codebookDim, outChannels: config.latentDim, kernelSize: 3)
        _upsample.wrappedValue = config.upsamplingRatios.map { factor in
            UpsampleLayer(latentDim: config.latentDim, factor: factor)
        }

        let outputDim = config.decoderDim / (1 << config.upsampleRates.count)
        _decoder.wrappedValue = [
            DecoderInitialConv(latentDim: config.latentDim, decoderDim: config.decoderDim, kernelSize: 7),
        ] + (0 ..< config.upsampleRates.count).map { DecoderBlock(config: config, layerIdx: $0) as Module } + [
            DecoderOutputSnake(channels: outputDim),
            DecoderOutputConv(channels: outputDim, kernelSize: 7),
        ]
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, num_quantizers, time]
        var hidden = quantizer.decode(codes) // [batch, codebook_dim, time]
        hidden = preConv(hidden) // [batch, latent_dim, time]
        hidden = hidden.transposed(0, 2, 1) // [batch, time, latent_dim]
        hidden = preTransformer(hidden)
        hidden = hidden.transposed(0, 2, 1) // [batch, latent_dim, time]

        for layer in upsample {
            hidden = layer(hidden)
        }

        var wav = hidden
        for layer in decoder {
            if let initConv = layer as? DecoderInitialConv { wav = initConv(wav) }
            else if let block = layer as? DecoderBlock { wav = block(wav) }
            else if let snake = layer as? DecoderOutputSnake { wav = snake(wav) }
            else if let outConv = layer as? DecoderOutputConv { wav = outConv(wav) }
        }
        return clip(wav, min: -1, max: 1)
    }

    func chunkedDecode(_ codes: MLXArray, chunkSize: Int = 300, leftContextSize: Int = 25) -> MLXArray {
        var wavs = [MLXArray]()
        var startIndex = 0
        let totalTime = codes.dim(-1)

        while startIndex < totalTime {
            let endIndex = min(startIndex + chunkSize, totalTime)
            let contextSize = startIndex - leftContextSize > 0 ? leftContextSize : startIndex
            let chunk = codes[0..., 0..., (startIndex - contextSize) ..< endIndex]
            let wavChunk = callAsFunction(chunk)
            wavs.append(wavChunk[0..., 0..., (contextSize * totalUpsample)...])
            startIndex = endIndex
        }
        return concatenated(wavs, axis: -1)
    }
}

// MARK: - Speech Tokenizer (wrapper)

final class Qwen3TTSSpeechTokenizer: Module {
    let config: Qwen3TTSTokenizerConfig
    let decodeUpsampleRate: Int
    @ModuleInfo var decoder: Qwen3TTSSpeechTokenizerDecoder
    @ModuleInfo(key: "encoder_model") var encoderModel: Qwen3TTSSpeechTokenizerEncoder?

    init(config: Qwen3TTSTokenizerConfig) {
        self.config = config
        self.decodeUpsampleRate = config.decodeUpsampleRate
        let decoderConfig = config.decoderConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTokenizerDecoderConfig.self, from: json)
        }()
        _decoder.wrappedValue = Qwen3TTSSpeechTokenizerDecoder(config: decoderConfig)
        if let encoderConfig = config.encoderConfig {
            _encoderModel.wrappedValue = Qwen3TTSSpeechTokenizerEncoder(
                config: encoderConfig,
                validNumQuantizers: config.encoderValidNumQuantizers
            )
        }
    }

    var hasEncoder: Bool {
        encoderModel != nil
    }

    func encode(_ audio: MLXArray) -> MLXArray {
        guard let encoderModel else {
            fatalError("Encoder not available for this speech tokenizer")
        }
        return encoderModel.encode(audio)
    }

    func decode(_ audioCodes: MLXArray) -> (MLXArray, MLXArray) {
        // audioCodes: [batch, time, num_quantizers]
        let codes = audioCodes.transposed(0, 2, 1) // [batch, num_quantizers, time]
        let wav = decoder.chunkedDecode(codes).squeezed(axis: 1)

        // Calculate valid lengths
        let audioLengths = (audioCodes[0..., 0..., 0] .> 0).sum(axis: 1).asType(.int32) * Int32(decodeUpsampleRate)
        return (wav, audioLengths)
    }

    func streamingDecode(_ audioCodes: MLXArray, chunkTokens: Int = 100) -> [MLXArray] {
        let codes = audioCodes.transposed(0, 2, 1)
        let totalTokens = codes.dim(-1)
        let leftContextSize = 25
        var chunks = [MLXArray]()

        var startIndex = 0
        while startIndex < totalTokens {
            let endIndex = min(startIndex + chunkTokens, totalTokens)
            let contextSize = startIndex - leftContextSize > 0 ? leftContextSize : startIndex
            let chunk = codes[0..., 0..., (startIndex - contextSize) ..< endIndex]
            var wavChunk = decoder(chunk)
            wavChunk = wavChunk[0..., 0..., (contextSize * decoder.totalUpsample)...]
            wavChunk = wavChunk.squeezed(axis: 1)
            eval(wavChunk)
            chunks.append(wavChunk)
            Memory.clearCache()
            startIndex = endIndex
        }
        return chunks
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var codebookData = [String: [String: MLXArray]]()
        var encoderTransformerQKV = [Int: [String: MLXArray]]()
        var encoderCodebookData = [String: [String: MLXArray]]()

        let encoderLayerConvMap: [Int: String] = [
            0: "encoder_model.encoder.init_conv1d",
            3: "encoder_model.encoder.layers.0.downsample",
            6: "encoder_model.encoder.layers.1.downsample",
            9: "encoder_model.encoder.layers.2.downsample",
            12: "encoder_model.encoder.layers.3.downsample",
            14: "encoder_model.encoder.final_conv1d",
        ]
        let encoderResidualLayerMap: [Int: Int] = [1: 0, 4: 1, 7: 2, 10: 3]
        let encoderResidualBlockMap: [Int: Int] = [1: 0, 3: 1]

        func setEncoderQKV(layerIdx: Int, key: String, value: MLXArray) {
            var entry = encoderTransformerQKV[layerIdx] ?? [:]
            entry[key] = value
            encoderTransformerQKV[layerIdx] = entry
        }

        func stripKnownPrefixes(_ key: String) -> String {
            var normalized = key
            let prefixes = ["speech_tokenizer.", "encoder_model.", "decoder_model."]
            var didStrip = true
            while didStrip {
                didStrip = false
                for prefix in prefixes {
                    if normalized.hasPrefix(prefix) {
                        normalized = String(normalized.dropFirst(prefix.count))
                        didStrip = true
                        break
                    }
                }
            }
            return normalized
        }

        func splitBeforeLastCodebookPath(_ key: String) -> String {
            let marker = "._codebook."
            if let idx = key.range(of: marker, options: .backwards) {
                return String(key[..<idx.lowerBound])
            }
            return key
        }

        func splitBeforeCodebookPath(_ key: String) -> String {
            let marker = ".codebook."
            if let idx = key.range(of: marker, options: .backwards) {
                return String(key[..<idx.lowerBound])
            }
            return key
        }

        func mapEncoderQuantizerLayers(_ rest: String) -> String? {
            if rest.contains("codebook.") { return nil }
            if rest.hasPrefix("layers.") || rest.contains(".layers.") {
                if rest.hasPrefix("rvq_first.") {
                    let tail = String(rest.dropFirst("rvq_first.".count))
                    return "encoder_model.quantizer.rvq_first.vq.\(tail)"
                }
                if rest.hasPrefix("rvq_rest.") {
                    let tail = String(rest.dropFirst("rvq_rest.".count))
                    return "encoder_model.quantizer.rvq_rest.vq.\(tail)"
                }
                if rest.hasPrefix("semantic_residual_vector_quantizer.") {
                    let tail = String(rest.dropFirst("semantic_residual_vector_quantizer.".count))
                    return "encoder_model.quantizer.rvq_first.vq.\(tail)"
                }
                if rest.hasPrefix("acoustic_residual_vector_quantizer.") {
                    let tail = String(rest.dropFirst("acoustic_residual_vector_quantizer.".count))
                    return "encoder_model.quantizer.rvq_rest.vq.\(tail)"
                }
                if rest.hasPrefix("codebook.") {
                    return nil
                }
                if rest.hasPrefix("layers.") {
                    return "encoder_model.quantizer.rvq_rest.vq.\(rest)"
                }
            }
            return nil
        }

        func mapEncoderQuantizerPrefix(_ rest: String) -> String {
            if rest.hasPrefix("semantic_residual_vector_quantizer.") {
                return "encoder_model.quantizer.rvq_first"
            }
            if rest.hasPrefix("acoustic_residual_vector_quantizer.") {
                return "encoder_model.quantizer.rvq_rest"
            }
            if rest.hasPrefix("rvq_first.") {
                return "encoder_model.quantizer.rvq_first"
            }
            if rest.hasPrefix("rvq_rest.") {
                return "encoder_model.quantizer.rvq_rest"
            }
            return "encoder_model.quantizer.rvq_rest"
        }

        func encoderCodebookLayerIndex(from path: String) -> Int? {
            let parts = path.split(separator: ".")
            guard let layersIndex = parts.firstIndex(of: "layers"),
                  layersIndex + 1 < parts.count else { return nil }
            return Int(parts[layersIndex + 1])
        }

        func encoderCodebookPrefix(from basePath: String) -> String {
            if basePath.contains("rvq_first.") {
                return "encoder_model.quantizer.rvq_first"
            }
            if basePath.contains("rvq_rest.") {
                return "encoder_model.quantizer.rvq_rest"
            }
            if basePath.contains("semantic_residual_vector_quantizer") {
                return "encoder_model.quantizer.rvq_first"
            }
            return "encoder_model.quantizer.rvq_rest"
        }

        for (rawKey, var v) in weights {
            let k = stripKnownPrefixes(rawKey)
            if k.isEmpty || k == "encoder_model" || k == "decoder_model" || k == "speech_tokenizer" {
                continue
            }

            // Skip speaker encoder weights now handled separately
            if Qwen3TTSSpeakerEncoder.stripSpeakerEncoderPrefix(from: k) != nil {
                continue
            }

            // Collect decoder codebook cluster_usage and embedding_sum
            if k.contains("_codebook.cluster_usage") || k.contains("_codebook.embedding_sum") {
                let basePath = splitBeforeLastCodebookPath(k)
                if codebookData[basePath] == nil { codebookData[basePath] = [:] }
                if k.contains("cluster_usage") {
                    codebookData[basePath]!["cluster_usage"] = v
                } else {
                    codebookData[basePath]!["embedding_sum"] = v
                }
                continue
            }
            if k.contains("_codebook.initialized") || k.contains(".codebook.initialized") {
                continue
            }

            if k.hasPrefix("encoder.") {
                if k.hasPrefix("encoder.encoder.layers.") {
                    let parts = k.split(separator: ".")
                    guard parts.count >= 4, let layerIdx = Int(parts[3]) else { continue }
                    let n = layerIdx

                    if k.contains(".block.") {
                        guard let residualLayerIdx = encoderResidualLayerMap[n],
                              parts.count > 5,
                              let blockIdx = Int(parts[5]),
                              let convIdx = encoderResidualBlockMap[blockIdx] else {
                            continue
                        }
                        let basePath = "encoder_model.encoder.layers.\(residualLayerIdx).residuals.0.block.\(convIdx)"
                        let suffix = String(parts.dropFirst(6).joined(separator: "."))
                        let newKey = "\(basePath).conv.\(suffix)"
                        if suffix.hasSuffix("weight"), v.ndim == 3 {
                            v = v.transposed(0, 2, 1)
                        }
                        sanitized[newKey] = v
                    } else if let convMapPath = encoderLayerConvMap[n] {
                        let suffix = String(parts.dropFirst(4).joined(separator: "."))
                        let newKey = "\(convMapPath).conv.\(suffix)"
                        if suffix.hasSuffix("weight"), v.ndim == 3 {
                            v = v.transposed(0, 2, 1)
                        }
                        sanitized[newKey] = v
                    }
                    continue
                }

                if k.hasPrefix("encoder.encoder_transformer.layers.")
                    || k.hasPrefix("encoder.encoder_transformer.transformer.layers.") {
                    let parts = k.split(separator: ".")
                    let isNestedTransformer = parts.count >= 5 && parts[2] == "transformer" && parts[3] == "layers"
                    let layerIdxOffset = isNestedTransformer ? 4 : 3
                    guard parts.count > layerIdxOffset,
                          let layerIdx = Int(parts[layerIdxOffset]) else { continue }
                    let suffix = String(parts.dropFirst(layerIdxOffset + 1).joined(separator: "."))

                    if suffix.contains("self_attn.q_proj.weight") {
                        setEncoderQKV(layerIdx: layerIdx, key: "q", value: v)
                    } else if suffix.contains("self_attn.k_proj.weight") {
                        setEncoderQKV(layerIdx: layerIdx, key: "k", value: v)
                    } else if suffix.contains("self_attn.v_proj.weight") {
                        setEncoderQKV(layerIdx: layerIdx, key: "v", value: v)
                    } else if suffix.contains("self_attn.qkv.weight") && v.ndim == 2 {
                        let totalDim = v.dim(0)
                        let headDim = totalDim / 3
                        guard totalDim % 3 == 0, headDim > 0 else { continue }

                        setEncoderQKV(layerIdx: layerIdx, key: "q", value: v[0 ..< headDim])
                        setEncoderQKV(layerIdx: layerIdx, key: "k", value: v[headDim ..< headDim * 2])
                        setEncoderQKV(layerIdx: layerIdx, key: "v", value: v[headDim * 2 ..< totalDim])
                    } else if suffix.contains("self_attn.out_proj.weight")
                        || suffix.contains("self_attn.o_proj.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).self_attn.out_proj.weight"] = v
                    } else if suffix.contains("mlp.fc1.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).gating.linear1.weight"] = v
                    } else if suffix.contains("mlp.fc2.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).gating.linear2.weight"] = v
                    } else if suffix.contains("input_layernorm.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm1.weight"] = v
                    } else if suffix.contains("input_layernorm.bias") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm1.bias"] = v
                    } else if suffix.contains("post_attention_layernorm.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm2.weight"] = v
                    } else if suffix.contains("post_attention_layernorm.bias") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm2.bias"] = v
                    } else if suffix.contains("self_attn_layer_scale.scale") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).layer_scale_1.scale"] = v
                    } else if suffix.contains("mlp_layer_scale.scale") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).layer_scale_2.scale"] = v
                    }
                    continue
                }

                if k.hasPrefix("encoder.downsample.") {
                    let suffix = k.replacingOccurrences(of: "encoder.downsample.", with: "")
                    let newKey = "encoder_model.downsample.conv.conv.\(suffix)"
                    if suffix.hasSuffix("weight"), v.ndim == 3 {
                        v = v.transposed(0, 2, 1)
                    }
                    sanitized[newKey] = v
                    continue
                }

                if k.hasPrefix("encoder.quantizer.") {
                    let rest = k.replacingOccurrences(of: "encoder.quantizer.", with: "")

                    let isExplicitEmbedTensor =
                        rest.contains(".codebook.embed.weight")
                        || rest.hasSuffix(".codebook.embed")
                        || rest.hasSuffix("codebook.embed")
                    if isExplicitEmbedTensor {
                        continue
                    }

                    if rest.contains("codebook.cluster_usage") ||
                        rest.contains("codebook.embed_sum") ||
                        rest.contains("codebook.embedding_sum") {
                        let basePath = splitBeforeCodebookPath(rest)
                        if encoderCodebookData[basePath] == nil { encoderCodebookData[basePath] = [:] }
                        if rest.contains("cluster_usage") {
                            encoderCodebookData[basePath]!["cluster_usage"] = v
                        } else {
                            encoderCodebookData[basePath]!["embedding_sum"] = v
                        }
                        continue
                    }
                    if rest.contains("codebook.initialized") {
                        continue
                    }

                    if rest.contains("input_proj.weight") || rest.contains("output_proj.weight") {
                        let projType = rest.contains("input_proj") ? "input_proj" : "output_proj"
                        let basePath = mapEncoderQuantizerPrefix(rest)
                        let newKey = "\(basePath).\(projType).weight"
                        if rest.hasSuffix("weight"), v.ndim == 3 {
                            v = v.transposed(0, 2, 1)
                        }
                        sanitized[newKey] = v
                    }
                    if let mappedLayers = mapEncoderQuantizerLayers(rest) {
                        sanitized[mappedLayers] = v
                        continue
                    }
                    continue
                }

                continue
            }

            // Existing decoder weight handling
            if k.contains("_codebook.cluster_usage") || k.contains("_codebook.embedding_sum") {
                // handled above
                continue
            }

            // Transpose conv weights: PyTorch [out, in, kernel] → MLX format
            let isTransposeConv = (k.contains("upsample") && k.contains(".0.conv.weight"))
                || (k.contains("decoder.decoder") && k.contains("block.1.conv.weight"))

            if isTransposeConv, v.ndim == 3 {
                if !checkArrayShapeQwen3(v) {
                    v = v.transposed(1, 2, 0)
                }
            } else if k.contains("conv.weight"), v.ndim == 3 {
                if !checkArrayShapeQwen3(v) {
                    v = v.transposed(0, 2, 1)
                }
            } else if k.contains("_proj.weight"), v.ndim == 3 {
                if !checkArrayShapeQwen3(v) {
                    v = v.transposed(0, 2, 1)
                }
            }

            // Remap: upsample.X.Y.rest → upsample.X.layers.Y.rest
            // MLXNN unflattened() creates .array for numeric keys, but UpsampleLayer
            // exposes children via named "layers" property, so we insert "layers."
            var key = k
            if key.contains("upsample.") {
                key = key.replacingOccurrences(
                    of: #"upsample\.(\d+)\.(\d+)"#,
                    with: "upsample.$1.layers.$2",
                    options: .regularExpression
                )
            }
            sanitized[key] = v
        }

        for (layerIdx, qkv) in encoderTransformerQKV {
            guard let q = qkv["q"], let k = qkv["k"], let v = qkv["v"] else { continue }
            let inProj = concatenated([q, k, v], axis: 0)
            sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).self_attn.in_proj.weight"] = inProj
        }

        for (basePath, data) in encoderCodebookData {
            guard let clusterUsage = data["cluster_usage"],
                  let embeddingSum = data["embedding_sum"],
                  let layerIdx = encoderCodebookLayerIndex(from: basePath) else {
                continue
            }
            let groupPrefix = encoderCodebookPrefix(from: basePath)
            let prefix = "\(groupPrefix).vq.layers.\(layerIdx).codebook"
            sanitized["\(prefix).initialized"] = MLXArray.zeros([1], dtype: .float32)
            sanitized["\(prefix).cluster_usage"] = clusterUsage
            sanitized["\(prefix).embedding_sum"] = embeddingSum
        }

        // Keep decoder codebook statistics to derive embeddings via updateInPlace().
        for (basePath, data) in codebookData {
            guard let clusterUsage = data["cluster_usage"],
                  let embeddingSum = data["embedding_sum"] else { continue }
            sanitized["\(basePath).codebook.initialized"] = MLXArray.zeros([1], dtype: .float32)
            sanitized["\(basePath).codebook.cluster_usage"] = clusterUsage
            sanitized["\(basePath).codebook.embedding_sum"] = embeddingSum
        }

        return sanitized
    }
}

// MARK: - Conv weight shape heuristic

func checkArrayShapeQwen3(_ arr: MLXArray) -> Bool {
    guard arr.ndim == 3 else { return false }
    let (_, dim2, dim3) = (arr.dim(0), arr.dim(1), arr.dim(2))

    if dim2 == 1 {
        return dim3 > 64 // dim3 large → likely in_channels → MLX format
    } else if dim3 == 1 {
        return dim2 <= 64 // dim2 small → likely kernel → MLX format
    }
    return dim2 < dim3
}
