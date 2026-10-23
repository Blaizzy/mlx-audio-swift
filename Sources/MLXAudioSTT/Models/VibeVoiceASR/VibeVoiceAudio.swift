// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import MLX
import MLXNN

final class VibeVoiceConv: Module {
    let conv: Conv1d
    let kernelSize: Int
    let stride: Int
    let depthwise: Bool
    init(
        _ input: Int, _ output: Int, kernel: Int, stride: Int = 1, depthwise: Bool = false,
        bias: Bool = true
    ) {
        self.kernelSize = kernel
        self.stride = stride
        self.depthwise = depthwise
        conv = Conv1d(
            inputChannels: input, outputChannels: output, kernelSize: kernel,
            stride: stride, padding: 0, groups: depthwise ? input : 1, bias: bias)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let extra = (stride - input.dim(1) % stride) % stride
        let x = MLX.padded(
            input, widths: [.init((0, 0)), .init((kernelSize - stride, extra)), .init((0, 0))])
        if !depthwise { return conv(x) }
        // Accumulating the short depthwise kernel avoids oversized grouped-convolution workspaces.
        let count = (x.dim(1) - kernelSize) / stride + 1
        var y =
            x[0..., .stride(from: 0, to: count * stride, by: stride), 0...]
            * conv.weight[0..., 0, 0]
        for k in 1 ..< kernelSize {
            y =
                y + x[0..., .stride(from: k, to: k + count * stride, by: stride), 0...]
                * conv.weight[0..., k, 0]
        }
        if let bias = conv.bias { y = y + bias }
        return y
    }
}

final class VibeVoiceMixer: Module {
    let conv: VibeVoiceConv
    init(_ dim: Int, _ c: VibeVoiceTokenizerConfig) {
        conv = VibeVoiceConv(dim, dim, kernel: 7, depthwise: true, bias: c.convBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class VibeVoiceFFN: Module {
    let linear1: Linear
    let linear2: Linear
    init(_ dim: Int, bias: Bool) {
        linear1 = Linear(dim, dim * 4, bias: bias)
        linear2 = Linear(dim * 4, dim, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let step = max(1, 8 * 1024 * 1024 / (x.dim(0) * linear1.weight.dim(0)))
        if x.dim(1) <= step { return linear2(gelu(linear1(x))) }
        var chunks: [MLXArray] = []
        for start in Swift.stride(from: 0, to: x.dim(1), by: step) {
            let y = linear2(gelu(linear1(x[0..., start ..< min(start + step, x.dim(1)), 0...])))
            eval(y)
            chunks.append(y)
        }
        return concatenated(chunks, axis: 1)
    }
}

final class VibeVoiceBlock: Module {
    let norm: Module
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Module
    let mixer: VibeVoiceMixer
    let ffn: VibeVoiceFFN
    @ParameterInfo var gamma: MLXArray?
    @ParameterInfo(key: "ffn_gamma") var ffnGamma: MLXArray?
    init(_ dim: Int, _ c: VibeVoiceTokenizerConfig) {
        norm =
            c.layernorm == "RMSNorm"
            ? RMSNorm(dimensions: dim, eps: c.layernormEps)
            : LayerNorm(dimensions: dim, eps: c.layernormEps)
        _ffnNorm.wrappedValue =
            c.layernorm == "RMSNorm"
            ? RMSNorm(dimensions: dim, eps: c.layernormEps)
            : LayerNorm(dimensions: dim, eps: c.layernormEps)
        mixer = VibeVoiceMixer(dim, c)
        ffn = VibeVoiceFFN(dim, bias: c.convBias)
        if c.layerScaleInitValue > 0 {
            _gamma.wrappedValue = .ones([dim]) * c.layerScaleInitValue
            _ffnGamma.wrappedValue = .ones([dim]) * c.layerScaleInitValue
        }
    }
    private func normalize(_ module: Module, _ x: MLXArray) -> MLXArray {
        if let norm = module as? RMSNorm { return norm(x) }
        return (module as! LayerNorm)(x)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var mixed = mixer(normalize(norm, input))
        if let gamma { mixed = mixed * gamma }
        let h = input + mixed
        var out = ffn(normalize(ffnNorm, h))
        if let ffnGamma { out = out * ffnGamma }
        return h + out
    }
}

final class VibeVoiceEncoder: Module {
    @ModuleInfo(key: "downsample_layers") var downsampleLayers: [VibeVoiceConv]
    let stages: [[VibeVoiceBlock]]
    let norm: RMSNorm?
    let head: VibeVoiceConv
    let hopLength: Int
    let receptiveField: Int
    init(_ c: VibeVoiceTokenizerConfig) {
        let ratios = Array(c.encoderRatios.reversed())
        var downs = [VibeVoiceConv(c.channels, c.encoderNFilters, kernel: 7, bias: c.convBias)]
        var stages: [[VibeVoiceBlock]] = []
        var dim = c.encoderNFilters
        var field = 7
        var hop = 1
        for stage in c.encoderDepths.indices {
            if stage > 0 {
                let ratio = ratios[stage - 1]
                downs.append(
                    VibeVoiceConv(dim, dim * 2, kernel: ratio * 2, stride: ratio, bias: c.convBias))
                field += (ratio * 2 - 1) * hop
                hop *= ratio
                dim *= 2
            }
            stages.append((0 ..< c.encoderDepths[stage]).map { _ in VibeVoiceBlock(dim, c) })
            field += c.encoderDepths[stage] * 6 * hop
        }
        _downsampleLayers.wrappedValue = downs
        self.stages = stages
        norm = c.disableLastNorm ? nil : RMSNorm(dimensions: dim, eps: c.layernormEps)
        head = VibeVoiceConv(dim, c.vaeDim, kernel: 7, bias: c.convBias)
        hopLength = hop
        receptiveField = field + 6 * hop
    }
    func forward(_ waveform: MLXArray) -> MLXArray {
        var x = waveform.expandedDimensions(axis: -1)
        for i in stages.indices {
            x = downsampleLayers[i](x)
            eval(x)
            for block in stages[i] {
                x = block(x)
                eval(x)
            }
        }
        if let norm { x = norm(x) }
        return head(x)
    }
    func callAsFunction(_ waveform: MLXArray, chunkSamples: Int = 240000) -> MLXArray {
        let step = max(1, chunkSamples / hopLength) * hopLength
        if waveform.dim(1) <= step { return forward(waveform) }
        // These finite causal convolutions need only their receptive field as left context.
        // Align every overlap to the total stride so chunking preserves the encoder frame grid.
        let context = ((receptiveField + hopLength - 1) / hopLength) * hopLength
        var parts: [MLXArray] = []
        for start in Swift.stride(from: 0, to: waveform.dim(1), by: step) {
            let left = max(0, start - context)
            let right = min(start + step, waveform.dim(1))
            let out = forward(waveform[0..., left ..< right])
            let part = out[0..., ((start - left) / hopLength)..., 0...]
            eval(part)
            parts.append(part)
        }
        return concatenated(parts, axis: 1)
    }
}

final class VibeVoiceTokenizerEncoder: Module {
    let encoder: VibeVoiceEncoder
    init(_ config: VibeVoiceTokenizerConfig) { encoder = VibeVoiceEncoder(config) }
    func callAsFunction(_ waveform: MLXArray) -> MLXArray { encoder(waveform) }
}
