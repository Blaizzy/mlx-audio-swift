import Foundation
import MLX
import MLXNN
import MLXRandom

final class LongCatSnake: Module, UnaryLayer {
    @ParameterInfo var alpha: MLXArray
    @ParameterInfo var beta: MLXArray
    init(_ channels: Int) {
        _alpha.wrappedValue = .zeros([channels])
        _beta.wrappedValue = .zeros([channels])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + sin(x * exp(alpha)).square() / (exp(beta) + 1e-9)
    }
}

final class LongCatActivation: Module, UnaryLayer {
    let useTanh: Bool
    init(tanh: Bool = false) { useTanh = tanh }
    func callAsFunction(_ x: MLXArray) -> MLXArray { useTanh ? MLX.tanh(x) : elu(x) }
}

private func longCatActivation(_ snake: Bool, _ channels: Int) -> any UnaryLayer {
    snake ? LongCatSnake(channels) : LongCatActivation()
}

final class LongCatConv: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray?
    let stride: Int, padding: Int, dilation: Int
    let transposed: Bool
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int = 1, padding: Int = 0,
         dilation: Int = 1, bias: Bool = true, transposed: Bool = false) {
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.transposed = transposed
        _weight.wrappedValue = .zeros([output, kernel, input])
        _bias.wrappedValue = bias ? .zeros([output]) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = transposed ? convTransposed1d(x, weight, stride: stride, padding: padding)
            : conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        return bias.map { h + $0 } ?? h
    }
}

func longCatPixelUnshuffle(_ x: MLXArray, factor: Int) -> MLXArray {
    let b = x.dim(0), w = x.dim(1), c = x.dim(2)
    return x.reshaped(b, w / factor, factor, c).transposed(0, 1, 3, 2).reshaped(b, w / factor, c * factor)
}

func longCatPixelShuffle(_ x: MLXArray, factor: Int) -> MLXArray {
    let b = x.dim(0), w = x.dim(1), c = x.dim(2) / factor
    return x.reshaped(b, w, c, factor).transposed(0, 1, 3, 2).reshaped(b, w * factor, c)
}

final class LongCatShortcut: Module, UnaryLayer {
    let input: Int, output: Int, factor: Int
    let upsample: Bool
    init(_ input: Int, _ output: Int, factor: Int, upsample: Bool) {
        self.input = input; self.output = output; self.factor = factor; self.upsample = upsample
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if upsample { return longCatPixelShuffle(repeated(x, count: output * factor / input, axis: 2), factor: factor) }
        let h = longCatPixelUnshuffle(x, factor: factor)
        return mean(h.reshaped(h.dim(0), h.dim(1), output, input * factor / output), axis: 3)
    }
}

final class LongCatResidualUnit: Module, UnaryLayer {
    @ModuleInfo var layers: [any UnaryLayer]
    init(_ channels: Int, dilation: Int, snake: Bool) {
        _layers.wrappedValue = [longCatActivation(snake, channels),
            LongCatConv(channels, channels, kernel: 7, padding: dilation * 3, dilation: dilation),
            longCatActivation(snake, channels), LongCatConv(channels, channels, kernel: 1)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + layers.reduce(x) { $1($0) } }
}

final class LongCatVAEBlock: Module, UnaryLayer {
    @ModuleInfo var layers: [any UnaryLayer]
    @ModuleInfo var res: LongCatShortcut?
    init(_ input: Int, _ output: Int, stride: Int, config: LongCatVAEConfiguration, decode: Bool) {
        let snake = config.useSnake, padding = (stride + 1) / 2
        if decode {
            _layers.wrappedValue = [longCatActivation(snake, input),
                LongCatConv(input, output, kernel: stride * 2, stride: stride, padding: padding, transposed: true),
                LongCatResidualUnit(output, dilation: 1, snake: snake),
                LongCatResidualUnit(output, dilation: 3, snake: snake),
                LongCatResidualUnit(output, dilation: 9, snake: snake)]
            _res.wrappedValue = config.upsampleShortcut == "duplicating" ? LongCatShortcut(input, output, factor: stride, upsample: true) : nil
        } else {
            _layers.wrappedValue = [LongCatResidualUnit(input, dilation: 1, snake: snake),
                LongCatResidualUnit(input, dilation: 3, snake: snake),
                LongCatResidualUnit(input, dilation: 9, snake: snake), longCatActivation(snake, input),
                LongCatConv(input, output, kernel: stride * 2, stride: stride, padding: padding)]
            _res.wrappedValue = config.downsampleShortcut == "averaging" ? LongCatShortcut(input, output, factor: stride, upsample: false) : nil
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = layers.reduce(x) { $1($0) }
        return res.map { h + $0(x) } ?? h
    }
}

final class LongCatVAEEncoder: Module, UnaryLayer {
    @ModuleInfo var layers: [any UnaryLayer]
    @ModuleInfo var shortcut: LongCatShortcut?
    init(_ c: LongCatVAEConfiguration) {
        let channels = ([1] + c.cMults).map { $0 * c.channels }
        var modules: [any UnaryLayer] = [LongCatConv(c.inChannels, channels[0], kernel: 7, padding: 3)]
        for i in c.strides.indices {
            modules.append(LongCatVAEBlock(channels[i], channels[i + 1], stride: c.strides[i], config: c, decode: false))
        }
        modules.append(LongCatConv(channels.last!, c.encoderLatentDim, kernel: 3, padding: 1))
        _layers.wrappedValue = modules
        _shortcut.wrappedValue = c.outShortcut == "averaging" ? LongCatShortcut(channels.last!, c.encoderLatentDim, factor: 1, upsample: false) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = layers.dropLast().reduce(x) { $1($0) }
        return layers.last!(h) + (shortcut?(h) ?? MLXArray(0))
    }
}

final class LongCatVAEDecoder: Module, UnaryLayer {
    @ModuleInfo var layers: [any UnaryLayer]
    @ModuleInfo var shortcut: LongCatShortcut?
    let finalTanh: Bool
    init(_ c: LongCatVAEConfiguration) {
        finalTanh = c.finalTanh
        let channels = ([1] + c.cMults).map { $0 * c.channels }
        var modules: [any UnaryLayer] = [LongCatConv(c.latentDim, channels.last!, kernel: 7, padding: 3)]
        for i in c.strides.indices.reversed() {
            modules.append(LongCatVAEBlock(channels[i + 1], channels[i], stride: c.strides[i], config: c, decode: true))
        }
        modules.append(longCatActivation(c.useSnake, channels[0]))
        modules.append(LongCatConv(channels[0], c.inChannels, kernel: 7, padding: 3, bias: false))
        _layers.wrappedValue = modules
        _shortcut.wrappedValue = c.inShortcut == "duplicating" ? LongCatShortcut(c.latentDim, channels.last!, factor: 1, upsample: true) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = layers[0](x)
        if let shortcut { h = h + shortcut(x) }
        for layer in layers.dropFirst() { h = layer(h) }
        return finalTanh ? tanh(h) : h
    }
}

final class LongCatVAE: Module {
    @ModuleInfo var encoder: LongCatVAEEncoder
    @ModuleInfo var decoder: LongCatVAEDecoder
    let config: LongCatVAEConfiguration
    init(_ config: LongCatVAEConfiguration) {
        self.config = config
        _encoder.wrappedValue = LongCatVAEEncoder(config)
        _decoder.wrappedValue = LongCatVAEDecoder(config)
    }
    func encode(_ audio: MLXArray, noise: MLXArray? = nil) -> MLXArray {
        let moments = split(encoder(audio), parts: 2, axis: -1)
        let noise = noise ?? MLXRandom.normal(moments[0].shape)
        return (moments[0] + noise * (softplus(moments[1]) + 1e-4)) / config.scale
    }
    func decode(_ latent: MLXArray) -> MLXArray { decoder(latent * config.scale) }
}
