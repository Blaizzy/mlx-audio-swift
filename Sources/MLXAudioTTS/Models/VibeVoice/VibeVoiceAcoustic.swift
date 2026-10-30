// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXNN

/// One invocation owns one cache. All tensors use (batch, time, channels).
final class VibeVoiceConvolutionCache {
    var state: [ObjectIdentifier: MLXArray] = [:]
    func evaluate() { eval(Array(state.values)) }
}
final class VibeVoiceSynthesisConv: Module {
    let conv: Conv1d
    let padding: Int
    let stride: Int
    let inputChannels: Int
    let depthwise: Bool
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int = 1, depthwise: Bool = false, bias: Bool = true) {
        padding = kernel - stride
        self.stride = stride
        inputChannels = input
        self.depthwise = depthwise
        conv = Conv1d(inputChannels: input, outputChannels: output, kernelSize: kernel, stride: stride, groups: depthwise ? input : 1, bias: bias)
    }
    func callAsFunction(_ input: MLXArray, cache: VibeVoiceConvolutionCache? = nil) -> MLXArray {
        var x: MLXArray
        if let cache {
            precondition(input.dim(1) % stride == 0, "Streaming convolution chunks must align with stride")
            let left = cache.state[ObjectIdentifier(self)] ?? .zeros([input.dim(0), padding, inputChannels], dtype: input.dtype)
            x = concatenated([left, input], axis: 1)
            if padding > 0 { cache.state[ObjectIdentifier(self)] = x[0..., (x.dim(1) - padding)..., 0...] }
        } else {
            let extra = (stride - input.dim(1) % stride) % stride
            x = padded(input, widths: [.init((0, 0)), .init((padding, extra)), .init((0, 0))])
        }
        if !depthwise { return conv(x) }
        let kernel = conv.weight.dim(1)
        let count = (x.dim(1) - kernel) / stride + 1
        // Grouped convolution accumulates products in Float32, then casts before bias.
        // Preserve the original input tensors in the streaming cache.
        let outputDType = x.dtype == conv.weight.dtype ? x.dtype : .float32
        let full = x.asType(.float32), weight = conv.weight.asType(.float32)
        var y = full[0..., .stride(from: 0, to: count * stride, by: stride), 0...] * weight[0..., 0, 0]
        for k in 1..<kernel {
            y = y + full[0..., .stride(from: k, to: k + count * stride, by: stride), 0...] * weight[0..., k, 0]
        }
        y = y.asType(outputDType)
        if let bias = conv.bias { y = y + bias }
        return y
    }
}
final class VibeVoiceSynthesisTransposeConv: Module {
    let convtr: ConvTransposed1d
    let stride: Int
    let padding: Int
    init(_ input: Int, _ output: Int, ratio: Int, bias: Bool) {
        stride = ratio
        padding = ratio
        convtr = ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: ratio * 2, stride: ratio, bias: bias)
    }
    func callAsFunction(_ x: MLXArray, cache: VibeVoiceConvolutionCache? = nil) -> MLXArray {
        guard let cache else {
            let y = convtr(x)
            return y[0..., ..<(y.dim(1) - padding), 0...]
        }
        // Sum overlap before adding bias, so each emitted sample receives bias once.
        var y = convTransposed1d(x, convtr.weight, stride: stride)
        if let pending = cache.state[ObjectIdentifier(self)], pending.dim(1) > 0 {
            let overlap = min(pending.dim(1), y.dim(1))
            y = concatenated([y[0..., ..<overlap, 0...] + pending[0..., ..<overlap, 0...], y[0..., overlap..., 0...]], axis: 1)
        }
        let count = x.dim(1) * stride
        cache.state[ObjectIdentifier(self)] = y[0..., count..., 0...]
        let output = y[0..., ..<count, 0...]
        return convtr.bias.map { output + $0 } ?? output
    }
}
final class VibeVoiceSynthesisFFN: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    init(_ dim: Int, bias: Bool) {
        _linear1.wrappedValue = Linear(dim, dim * 4, bias: bias)
        _linear2.wrappedValue = Linear(dim * 4, dim, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}
final class VibeVoiceSynthesisBlock: Module {
    let norm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    let mixer: VibeVoiceSynthesisConv
    let ffn: VibeVoiceSynthesisFFN
    @ParameterInfo var gamma: MLXArray?
    @ParameterInfo(key: "ffn_gamma") var ffnGamma: MLXArray?
    init(_ dim: Int, _ c: VibeVoiceAcousticConfig) {
        norm = RMSNorm(dimensions: dim, eps: c.layernormEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: c.layernormEps)
        mixer = VibeVoiceSynthesisConv(dim, dim, kernel: 7, depthwise: true, bias: c.convBias)
        ffn = VibeVoiceSynthesisFFN(dim, bias: c.convBias)
        if c.layerScaleInitValue > 0 {
            _gamma.wrappedValue = .ones([dim]) * c.layerScaleInitValue
            _ffnGamma.wrappedValue = .ones([dim]) * c.layerScaleInitValue
        }
    }
    func callAsFunction(_ input: MLXArray, cache: VibeVoiceConvolutionCache? = nil) -> MLXArray {
        var h = mixer(norm(input), cache: cache)
        if let gamma { h = h * gamma }
        h = input + h
        var y = ffn(ffnNorm(h))
        if let ffnGamma { y = y * ffnGamma }
        return h + y
    }
}
final class VibeVoiceSynthesisStage: Module {
    let blocks: [VibeVoiceSynthesisBlock]
    init(_ depth: Int, dimension: Int, config: VibeVoiceAcousticConfig) {
        blocks = (0..<depth).map { _ in VibeVoiceSynthesisBlock(dimension, config) }
    }
}
final class VibeVoiceSynthesisEncoder: Module {
    @ModuleInfo(key: "downsample_layers") var downsampleLayers: [VibeVoiceSynthesisConv]
    let stages: [VibeVoiceSynthesisStage]
    let norm: RMSNorm?
    let head: VibeVoiceSynthesisConv
    init(_ c: VibeVoiceAcousticConfig) {
        var layers = [VibeVoiceSynthesisConv(c.channels, c.encoderNFilters, kernel: 7, bias: c.convBias)]
        let ratios = Array(c.encoderRatios.reversed())
        var dim = c.encoderNFilters
        var stages: [VibeVoiceSynthesisStage] = []
        for i in c.encoderDepths.indices {
            if i > 0 {
                layers.append(VibeVoiceSynthesisConv(dim, dim * 2, kernel: ratios[i - 1] * 2, stride: ratios[i - 1], bias: c.convBias))
                dim *= 2
            }
            stages.append(VibeVoiceSynthesisStage(c.encoderDepths[i], dimension: dim, config: c))
        }
        _downsampleLayers.wrappedValue = layers
        self.stages = stages
        norm = c.disableLastNorm ? nil : RMSNorm(dimensions: dim, eps: c.layernormEps)
        head = VibeVoiceSynthesisConv(dim, c.vaeDim, kernel: 7, bias: c.convBias)
    }
    func callAsFunction(_ audio: MLXArray, cache: VibeVoiceConvolutionCache? = nil) -> MLXArray {
        var x = audio.ndim == 2 ? audio.expandedDimensions(axis: -1) : audio
        for i in stages.indices {
            x = downsampleLayers[i](x, cache: cache)
            for block in stages[i].blocks { x = block(x, cache: cache) }
            eval(x)
        }
        return head(norm?(x) ?? x, cache: cache)
    }
}
final class VibeVoiceSynthesisUpsample: Module {
    let conv: VibeVoiceSynthesisConv?
    let convtr: VibeVoiceSynthesisTransposeConv?
    init(_ input: Int, _ output: Int, ratio: Int?, bias: Bool) {
        if let ratio {
            conv = nil
            convtr = VibeVoiceSynthesisTransposeConv(input, output, ratio: ratio, bias: bias)
        } else {
            conv = VibeVoiceSynthesisConv(input, output, kernel: 7, bias: bias)
            convtr = nil
        }
    }
    func callAsFunction(_ x: MLXArray, cache: VibeVoiceConvolutionCache?) -> MLXArray {
        if let conv { return conv(x, cache: cache) }
        return convtr!(x, cache: cache)
    }
}
final class VibeVoiceSynthesisDecoder: Module {
    @ModuleInfo(key: "upsample_layers") var upsampleLayers: [VibeVoiceSynthesisUpsample]
    let stages: [VibeVoiceSynthesisStage]
    let head: VibeVoiceSynthesisConv
    init(_ c: VibeVoiceAcousticConfig) {
        let ratios = c.decoderRatios ?? c.encoderRatios
        let depths = c.decoderDepths ?? Array(c.encoderDepths.reversed())
        let filters = c.decoderNFilters
        var dim = filters * (1 << (depths.count - 1))
        var layers = [VibeVoiceSynthesisUpsample(c.vaeDim, dim, ratio: nil, bias: c.convBias)]
        var stages: [VibeVoiceSynthesisStage] = []
        for i in depths.indices {
            stages.append(VibeVoiceSynthesisStage(depths[i], dimension: dim, config: c))
            if i < ratios.count {
                let next = i < ratios.count - 1 ? dim / 2 : filters
                layers.append(VibeVoiceSynthesisUpsample(dim, next, ratio: ratios[i], bias: c.convBias))
                dim = next
            }
        }
        _upsampleLayers.wrappedValue = layers
        self.stages = stages
        head = VibeVoiceSynthesisConv(filters, c.channels, kernel: 7, bias: c.convBias)
    }
    func callAsFunction(_ latents: MLXArray, cache: VibeVoiceConvolutionCache? = nil) -> MLXArray {
        var x = upsampleLayers[0](latents, cache: cache)
        for i in stages.indices {
            for block in stages[i].blocks { x = block(x, cache: cache) }
            if i + 1 < upsampleLayers.count { x = upsampleLayers[i + 1](x, cache: cache) }
            eval(x)
        }
        return head(x, cache: cache)
    }
}
final class VibeVoiceAcousticTokenizer: Module {
    let encoder: VibeVoiceSynthesisEncoder?
    let decoder: VibeVoiceSynthesisDecoder
    init(_ config: VibeVoiceAcousticConfig, includeEncoder: Bool = false) {
        encoder = includeEncoder ? VibeVoiceSynthesisEncoder(config) : nil
        decoder = VibeVoiceSynthesisDecoder(config)
    }
    /// Returns (batch, samples, channels). Latents are always (batch, frames, dimension).
    func decode(_ latents: MLXArray, cache: VibeVoiceConvolutionCache? = nil) -> MLXArray { decoder(latents, cache: cache) }
}
func canonicalVibeVoiceSynthesisPath(_ key: String) -> String {
    var key = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
    key = key.replacingOccurrences(of: #"\.(upsample_layers|downsample_layers)\.(\d+)\.0\."#, with: ".$1.$2.", options: .regularExpression)
    key = key.replacingOccurrences(of: #"\.downsample_layers\.(\d+)\.conv\.conv\."#, with: ".downsample_layers.$1.conv.", options: .regularExpression)
    key = key.replacingOccurrences(of: #"\.stages\.(\d+)\.(\d+)\."#, with: ".stages.$1.blocks.$2.", options: .regularExpression)
    key = key.replacingOccurrences(of: ".head.conv.conv.", with: ".head.conv.")
    key = key.replacingOccurrences(of: ".mixer.conv.conv.conv.", with: ".mixer.conv.")
    key = key.replacingOccurrences(of: #"\.t_embedder\.mlp\.(?:layers\.)?0(?=\.|$)"#, with: ".t_embedder.mlp.fc1", options: .regularExpression)
    key = key.replacingOccurrences(of: #"\.t_embedder\.mlp\.(?:layers\.)?2(?=\.|$)"#, with: ".t_embedder.mlp.fc2", options: .regularExpression)
    key = key.replacingOccurrences(of: #"\.adaLN_modulation\.(?:layers\.)?1(?=\.|$)"#, with: ".adaLN_modulation.linear", options: .regularExpression)
    return key
}
