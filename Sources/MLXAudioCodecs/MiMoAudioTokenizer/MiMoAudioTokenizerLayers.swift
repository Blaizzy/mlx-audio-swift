import Foundation
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

private func mimoNorm(_ config: MiMoAudioTokenizerConfig, _ dimensions: Int) -> UnaryLayer {
    config.normalization == "RMSNorm"
        ? RMSNorm(dimensions: dimensions, eps: 1e-6)
        : LayerNorm(dimensions: dimensions, eps: 1e-5)
}

struct MiMoRotaryFrequencies { let value: MLXArray }

final class MiMoTokenizerAttention: Module, UnaryLayer {
    let heads: Int
    let headDim: Int
    let theta: Float
    let causal: Bool
    let window: [Int]
    var _inverseFrequencies: MiMoRotaryFrequencies?
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int, heads: Int, config: MiMoAudioTokenizerConfig, causal: Bool, window: [Int]) {
        self.heads = heads
        headDim = dim / heads
        theta = config.ropeTheta
        self.causal = causal
        self.window = window
        _qProj.wrappedValue = Linear(dim, dim)
        _kProj.wrappedValue = Linear(dim, dim, bias: false)
        _vProj.wrappedValue = Linear(dim, dim)
        _outProj.wrappedValue = Linear(dim, dim)
    }

    func rope(_ x: MLXArray) -> MLXArray {
        // The reference casts the frequency buffer to activation dtype before
        // forming its FP32 phases; ordinary RoPE would lose this BF16 rounding.
        let exponent = -MLXArray(stride(from: 0, to: headDim, by: 2).map(Float.init)) / Float(headDim)
        let frequencies = (_inverseFrequencies?.value ?? pow(MLXArray(theta), exponent)).asType(x.dtype).asType(.float32)
        let positions = MLXArray((0..<x.dim(2)).map(Float.init)).expandedDimensions(axis: -1)
        let angles = positions * frequencies
        let cosine = cos(angles).asType(x.dtype)
        let sine = sin(angles).asType(x.dtype)
        let halves = split(x, parts: 2, axis: -1)
        return concatenated([halves[0] * cosine - halves[1] * sine,
                             halves[1] * cosine + halves[0] * sine], axis: -1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), dim = x.dim(2)
        func project(_ layer: Linear) -> MLXArray {
            layer(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let q = rope(project(qProj)), k = rope(project(kProj)), v = project(vProj)
        var mask: MLXFast.ScaledDotProductAttentionMaskMode = causal ? .causal : .none
        if window[0] >= 0 || window[1] >= 0 {
            let positions = MLXArray(0..<length)
            let difference = positions.expandedDimensions(axis: 0) - positions.expandedDimensions(axis: 1)
            var valid = MLXArray.ones([length, length], dtype: .bool)
            if window[0] >= 0 { valid = valid .&& (difference .>= -window[0]) }
            if window[1] >= 0 { valid = valid .&& (difference .<= window[1]) }
            if causal { valid = valid .&& (difference .<= 0) }
            mask = .array(valid)
        }
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: 1 / sqrt(Float(headDim)), mask: mask)
        return outProj(y.transposed(0, 2, 1, 3).reshaped([batch, length, dim]))
    }
}

final class MiMoTokenizerTransformerLayer: Module, UnaryLayer {
    @ModuleInfo(key: "self_attn") var attention: MiMoTokenizerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: UnaryLayer
    @ModuleInfo(key: "final_layer_norm") var finalNorm: UnaryLayer
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(dim: Int, heads: Int, ffn: Int, config: MiMoAudioTokenizerConfig, causal: Bool, window: [Int]) {
        _attention.wrappedValue = MiMoTokenizerAttention(dim: dim, heads: heads, config: config,
                                                        causal: causal, window: window)
        _attentionNorm.wrappedValue = mimoNorm(config, dim)
        _finalNorm.wrappedValue = mimoNorm(config, dim)
        _fc1.wrappedValue = Linear(dim, ffn)
        _fc2.wrappedValue = Linear(ffn, dim)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = input + attention(attentionNorm(input))
        return x + fc2(gelu(fc1(finalNorm(x))))
    }
}

final class MiMoTokenizerRVQ: Module {
    @ModuleInfo var codebooks: [Embedding]

    init(dim: Int, sizes: [Int]) {
        _codebooks.wrappedValue = sizes.map { Embedding(embeddingCount: $0, dimensions: dim) }
    }

    func encode(_ features: MLXArray, count: Int) -> MLXArray {
        var residual = features.asType(.float32)
        var codes: [MLXArray] = []
        for codebook in codebooks.prefix(count) {
            let weight = codebook.weight.asType(.float32)
            let distance = sum(weight * weight, axis: -1) - 2 * matmul(residual, weight.T)
            let indices = argMin(distance, axis: -1)
            codes.append(indices)
            residual = residual - weight[indices]
        }
        return stacked(codes, axis: 0).asType(.int32)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var result = codebooks[0].weight.asType(.float32)[codes[0]]
        for i in 1..<codes.dim(0) { result = result + codebooks[i].weight.asType(.float32)[codes[i]] }
        return result
    }
}

final class MiMoTokenizerEncoder: Module {
    let config: MiMoAudioTokenizerConfig
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var layers: [MiMoTokenizerTransformerLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: UnaryLayer
    @ModuleInfo(key: "down_sample_layer") var downSample: MLXNN.Sequential?
    @ModuleInfo(key: "down_sample_norm") var downSampleNorm: UnaryLayer?
    @ModuleInfo var quantizer: MiMoTokenizerRVQ

    init(_ c: MiMoAudioTokenizerConfig) {
        config = c
        _conv1.wrappedValue = Conv1d(inputChannels: c.nMels, outputChannels: c.dModel,
                                    kernelSize: c.kernelSize, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: c.dModel, outputChannels: c.dModel,
                                    kernelSize: c.kernelSize, stride: c.strideSize, padding: 1)
        _layers.wrappedValue = (0..<c.encoderLayers).map { _ in
            MiMoTokenizerTransformerLayer(dim: c.dModel, heads: c.encoderAttentionHeads, ffn: c.encoderFFNDim,
                                           config: c, causal: c.encoderCausal, window: c.encoderAttentionWindow)
        }
        _layerNorm.wrappedValue = mimoNorm(c, c.dModel)
        if c.avgPooler != 1 {
            _downSample.wrappedValue = MLXNN.Sequential(layers: [
                Conv1d(inputChannels: c.dModel, outputChannels: c.dModel, kernelSize: c.avgPooler,
                       stride: c.avgPooler, bias: false), GELU()])
            _downSampleNorm.wrappedValue = mimoNorm(c, c.dModel)
        } else {
            _downSample.wrappedValue = nil
            _downSampleNorm.wrappedValue = nil
        }
        _quantizer.wrappedValue = MiMoTokenizerRVQ(dim: c.dModel, sizes: c.codebookSize)
    }

    func callAsFunction(_ mel: MLXArray, length: Int? = nil) -> MLXArray {
        let c = config
        let valid = ((length ?? mel.dim(1)) + 3 - c.kernelSize + 2 - c.kernelSize) / c.strideSize + 1
        var x = gelu(conv2(gelu(conv1(mel.asType(conv1.weight.dtype)))))
        let paddedLength = x.dim(1)
        x = x[0..., 0..<valid, 0...]
        var skip: MLXArray?
        for (i, layer) in layers.enumerated() {
            x = layer(x)
            if i + 1 == c.encoderSkipLayerID { skip = x }
        }
        if let skip { x = x + skip }
        x = layerNorm(x)
        if let downSample, let downSampleNorm {
            if valid < paddedLength {
                x = concatenated([x, broadcast(x[0..., (valid - 1)..<valid, 0...],
                                                to: [x.dim(0), paddedLength - valid, x.dim(2)])], axis: 1)
            }
            let extra = (c.avgPooler - paddedLength % c.avgPooler) % c.avgPooler
            if extra > 0 { x = padded(x, widths: [.init(0), .init((0, extra)), .init(0)]) }
            x = downSampleNorm(downSample(x))
            x = x[0..., 0..<((valid + c.avgPooler - 1) / c.avgPooler), 0...]
        }
        return x
    }
}

final class MiMoTokenizerCausalUpsample: Module, UnaryLayer {
    let trim: Int
    @ModuleInfo var conv: ConvTransposed1d
    @ModuleInfo var norm: GroupNorm

    init(input: Int, output: Int, kernel: Int, stride: Int) {
        trim = max(0, kernel - stride)
        _conv.wrappedValue = ConvTransposed1d(inputChannels: input, outputChannels: output,
                                             kernelSize: kernel, stride: stride)
        _norm.wrappedValue = GroupNorm(groupCount: 1, dimensions: output, eps: 1e-5,
                                       pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = norm(conv(x))
        return trim > 0 ? y[0..., 0..<(y.dim(1) - trim), 0...] : y
    }
}

final class MiMoTokenizerISTFT: Module, UnaryLayer {
    let nFFT: Int
    let hopLength: Int
    @ParameterInfo var window: MLXArray

    init(_ c: MiMoAudioTokenizerConfig) {
        nFFT = c.nFFT
        hopLength = c.hopLength
        _window.wrappedValue = mimoPeriodicHann(c.nFFT)
    }

    func callAsFunction(_ spectrum: MLXArray) -> MLXArray {
        mimoSameISTFT(spectrum, nFFT: nFFT, hopLength: hopLength, window: window.asType(.float32))
    }
}

final class MiMoTokenizerISTFTHead: Module, UnaryLayer {
    @ModuleInfo var out: Linear
    @ModuleInfo var istft: MiMoTokenizerISTFT

    init(_ c: MiMoAudioTokenizerConfig) {
        _out.wrappedValue = Linear(c.vocoderDim, c.nFFT + 2)
        _istft.wrappedValue = MiMoTokenizerISTFT(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(out(x), parts: 2, axis: -1)
        let magnitude = minimum(exp(parts[0]), MLXArray(Float(100))).asType(.float32)
        let phase = parts[1]
        let spectrum = magnitude * (cos(phase).asType(.float32)
            + MLXArray(real: Float(0), imaginary: Float(1)) * sin(phase).asType(.float32))
        return istft(spectrum).asType(x.dtype)
    }
}

final class MiMoTokenizerVocos: Module, UnaryLayer {
    @ModuleInfo var embeddings: Linear
    @ModuleInfo var layers: [MiMoTokenizerTransformerLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: UnaryLayer
    @ModuleInfo var head: MiMoTokenizerISTFTHead

    init(_ c: MiMoAudioTokenizerConfig) {
        _embeddings.wrappedValue = Linear(c.nMels, c.vocoderDim, bias: false)
        _layers.wrappedValue = (0..<c.vocoderNumLayers).map { _ in
            MiMoTokenizerTransformerLayer(dim: c.vocoderDim, heads: c.vocoderAttentionHeads,
                                           ffn: c.vocoderIntermediateDim, config: c,
                                           causal: false, window: c.vocoderAttentionWindow)
        }
        _layerNorm.wrappedValue = mimoNorm(c, c.vocoderDim)
        _head.wrappedValue = MiMoTokenizerISTFTHead(c)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = embeddings(input)
        for layer in layers { x = layer(x) }
        return head(layerNorm(x))
    }
}

final class MiMoTokenizerDecoder: Module, UnaryLayer {
    @ModuleInfo var dconv1: MiMoTokenizerCausalUpsample?
    @ModuleInfo var layers: [MiMoTokenizerTransformerLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: UnaryLayer
    @ModuleInfo var dconv2: MiMoTokenizerCausalUpsample
    @ModuleInfo var vocoder: MiMoTokenizerVocos

    init(_ c: MiMoAudioTokenizerConfig) {
        _dconv1.wrappedValue = c.avgPooler == 1 ? nil : MiMoTokenizerCausalUpsample(
            input: c.dModel, output: c.dModel, kernel: c.avgPooler, stride: c.avgPooler)
        _layers.wrappedValue = (0..<c.decoderLayers).map { _ in
            MiMoTokenizerTransformerLayer(dim: c.dModel, heads: c.decoderAttentionHeads, ffn: c.decoderFFNDim,
                                           config: c, causal: c.decoderCausal, window: c.decoderAttentionWindow)
        }
        _layerNorm.wrappedValue = mimoNorm(c, c.dModel)
        _dconv2.wrappedValue = MiMoTokenizerCausalUpsample(input: c.dModel, output: c.nMels,
                                                        kernel: c.decoderKernelSize, stride: c.decoderStrideSize)
        _vocoder.wrappedValue = MiMoTokenizerVocos(c)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let dtype = (layerNorm as? LayerNorm)?.weight?.dtype
            ?? (layerNorm as? RMSNorm)?.weight.dtype ?? input.dtype
        var x = input.asType(dtype)
        if let dconv1 { x = dconv1(x) }
        for layer in layers { x = layer(x) }
        return vocoder(dconv2(layerNorm(x)))
    }
}
