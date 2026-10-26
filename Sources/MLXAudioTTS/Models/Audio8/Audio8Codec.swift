import Foundation
import MLX
import MLXNN

@inline(__always) private func audio8Unary(_ m: Module, _ x: MLXArray) -> MLXArray {
    (m as! UnaryLayer)(x)
}

final class Audio8CodecTransformer: Module, UnaryLayer {
    let layers: [Audio8Block]
    let norm: Audio8Norm
    let window: Int, headDim: Int
    init(layers: Int, dim: Int, heads: Int, kvHeads: Int? = nil, intermediate: Int, window: Int) {
        self.window = window; headDim = 64
        self.layers = (0..<layers).map { _ in Audio8Block(dim: dim, intermediate: intermediate,
            heads: heads, kvHeads: kvHeads ?? heads, headDim: 64, layerScale: true) }
        norm = Audio8Norm(dim, eps: 1e-5)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let t = input.shape[1]
        let q = MLX.arange(t).expandedDimensions(axis: 1)
        let k = MLX.arange(t).expandedDimensions(axis: 0)
        let mask = (k .<= q) .&& (k .>= maximum(q - Int32(window - 1), 0))
        let rope = Audio8RoPE(length: t, dim: headDim, base: 10000)
        var x = input
        for layer in layers { x = layer(x, rope: rope, mask: .array(mask.reshaped([1, 1, t, t]))) }
        return norm(x)
    }
}

final class Audio8CausalConv: Module, UnaryLayer {
    let conv: Conv1d
    let stride: Int, kernel: Int, left: Int
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int = 1, dilation: Int = 1, groups: Int = 1) {
        conv = Conv1d(inputChannels: input, outputChannels: output, kernelSize: kernel,
                      stride: stride, dilation: dilation, groups: groups)
        self.stride = stride; self.kernel = (kernel - 1) * dilation + 1
        left = self.kernel - stride
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let t = x.shape[1]
        let frames = Int(ceil(Double(t - kernel + left) / Double(stride) + 1))
        let right = (frames - 1) * stride + kernel - left - t
        return conv(padded(x, widths: [0, .init((left, right)), 0]))
    }
}

final class Audio8CausalTranspose: Module, UnaryLayer {
    let conv: ConvTransposed1d
    let crop: Int
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int) {
        conv = ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: kernel, stride: stride)
        crop = kernel - stride
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        return crop == 0 ? y : y[0..., 0..<(y.shape[1] - crop), 0...]
    }
}

final class Audio8Snake: Module, UnaryLayer {
    var alpha: MLXArray
    init(_ channels: Int) { alpha = .ones([1, 1, channels]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + square(sin(alpha * x)) / (alpha + 1e-9) }
}

final class Audio8Residual: Module, UnaryLayer {
    let block: [Module]
    init(_ dim: Int, dilation: Int) {
        block = [Audio8Snake(dim), Audio8CausalConv(dim, dim, kernel: 7, dilation: dilation),
                 Audio8Snake(dim), Audio8CausalConv(dim, dim, kernel: 1)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + block.reduce(x) { audio8Unary($1, $0) } }
}

final class Audio8CodecEncoderBlock: Module, UnaryLayer {
    let block: [Module]
    init(_ dim: Int, stride: Int, transformer: Int) {
        var b: [Module] = [Audio8Residual(dim / 2, dilation: 1), Audio8Residual(dim / 2, dilation: 3),
            Audio8Residual(dim / 2, dilation: 9), Audio8Snake(dim / 2),
            Audio8CausalConv(dim / 2, dim, kernel: 2 * stride, stride: stride)]
        if transformer > 0 { b.append(Audio8CodecTransformer(layers: transformer, dim: dim,
                    heads: dim / 64, intermediate: dim * 3, window: 512)) }
        block = b
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { audio8Unary($1, $0) } }
}

final class Audio8CodecEncoder: Module, UnaryLayer {
    let block: [Module]
    override init() {
        var b: [Module] = [Audio8CausalConv(1, 64, kernel: 7)]
        var dim = 64
        for (stride, count) in zip([2, 4, 8, 8], [0, 0, 0, 4]) {
            dim *= 2; b.append(Audio8CodecEncoderBlock(dim, stride: stride, transformer: count))
        }
        b += [Audio8Snake(dim), Audio8CausalConv(dim, 1024, kernel: 3)]
        block = b
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { audio8Unary($1, $0) } }
}

final class Audio8CodecDecoderBlock: Module, UnaryLayer {
    let block: [Module]
    init(_ input: Int, _ output: Int, stride: Int) {
        block = [Audio8Snake(input), Audio8CausalTranspose(input, output, kernel: 2 * stride, stride: stride),
            Audio8Residual(output, dilation: 1), Audio8Residual(output, dilation: 3), Audio8Residual(output, dilation: 9)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { audio8Unary($1, $0) } }
}

final class Audio8CodecDecoder: Module, UnaryLayer {
    let model: [Module]
    override init() {
        var b: [Module] = [Audio8CausalConv(1024, 1536, kernel: 7)]
        for (i, stride) in [8, 8, 4, 2].enumerated() {
            b.append(Audio8CodecDecoderBlock(1536 / (1 << i), 1536 / (1 << (i + 1)), stride: stride))
        }
        b += [Audio8Snake(96), Audio8CausalConv(96, 1, kernel: 7)]
        model = b
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { tanh(model.reduce(x) { audio8Unary($1, $0) }) }
}

final class Audio8VectorQuantizer: Module {
    @ModuleInfo(key: "in_proj") var inProj: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Conv1d
    let codebook: Embedding
    let size: Int
    init(size: Int) {
        self.size = size
        _inProj.wrappedValue = Conv1d(inputChannels: 1024, outputChannels: 8, kernelSize: 1)
        _outProj.wrappedValue = Conv1d(inputChannels: 8, outputChannels: 1024, kernelSize: 1)
        codebook = Embedding(embeddingCount: size, dimensions: 8)
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let p = inProj(x)
        let flat = p.reshaped([-1, 8])
        let f = flat / maximum(sqrt(sum(square(flat), axis: -1, keepDims: true)), 1e-12)
        let w = codebook.weight
        let c = w / maximum(sqrt(sum(square(w), axis: -1, keepDims: true)), 1e-12)
        let distance = sum(square(f), axis: -1, keepDims: true) - 2 * matmul(f, c.T)
            + sum(square(c), axis: -1, keepDims: true).T
        let ids = argMax(-distance, axis: -1).reshaped([x.shape[0], x.shape[1]])
        return (outProj(codebook(ids)), ids)
    }
    func decode(_ ids: MLXArray) -> MLXArray { outProj(codebook(clip(ids, min: 0, max: size - 1))) }
}

final class Audio8ResidualQuantizer: Module {
    let quantizers: [Audio8VectorQuantizer]
    init(count: Int, size: Int) { quantizers = (0..<count).map { _ in Audio8VectorQuantizer(size: size) } }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var residual = x, total = MLXArray.zeros(x.shape, dtype: x.dtype)
        var codes = [MLXArray]()
        for q in quantizers { let (y, c) = q(residual); total = total + y; residual = residual - y; codes.append(c) }
        return (total, stacked(codes, axis: 1))
    }
    func decode(_ ids: MLXArray) -> MLXArray {
        let values = quantizers.enumerated().map { $0.element.decode(ids[0..., $0.offset, 0...]) }
        return values.dropFirst().reduce(values[0], +)
    }
}

final class Audio8ConvNeXt: Module, UnaryLayer {
    let dwconv: Audio8CausalConv
    let norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    var gamma: MLXArray
    init(_ dim: Int) {
        dwconv = Audio8CausalConv(dim, dim, kernel: 7, groups: dim)
        norm = LayerNorm(dimensions: dim, eps: 1e-6)
        _pwconv1.wrappedValue = Linear(dim, dim * 4)
        _pwconv2.wrappedValue = Linear(dim * 4, dim)
        gamma = MLXArray.ones([dim]) * 1e-6
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + gamma * pwconv2(gelu(pwconv1(norm(dwconv(x)))))
    }
}

final class Audio8CodecResample: Module, UnaryLayer {
    let layers: [Module]
    init(up: Bool) {
        let conv: Module = up ? Audio8CausalTranspose(1024, 1024, kernel: 2, stride: 2)
            : Audio8CausalConv(1024, 1024, kernel: 2, stride: 2)
        layers = [conv, Audio8ConvNeXt(1024)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers.reduce(x) { audio8Unary($1, $0) } }
}

final class Audio8DownsampleQuantizer: Module {
    @ModuleInfo(key: "semantic_quantizer") var semantic: Audio8ResidualQuantizer
    let quantizer: Audio8ResidualQuantizer
    let downsample: [Audio8CodecResample], upsample: [Audio8CodecResample]
    @ModuleInfo(key: "pre_module") var pre: Audio8CodecTransformer
    @ModuleInfo(key: "post_module") var post: Audio8CodecTransformer
    init(_ c: Audio8Config) {
        _semantic.wrappedValue = Audio8ResidualQuantizer(count: 1, size: 4096)
        quantizer = Audio8ResidualQuantizer(count: 9, size: 1024)
        downsample = (0..<2).map { _ in Audio8CodecResample(up: false) }
        upsample = (0..<2).map { _ in Audio8CodecResample(up: true) }
        _pre.wrappedValue = Audio8CodecTransformer(layers: 8, dim: 1024, heads: 16, intermediate: 3072, window: 128)
        _post.wrappedValue = Audio8CodecTransformer(layers: c.codecPostNLayer, dim: 1024,
            heads: c.codecPostNHead, kvHeads: c.codecPostNLocalHeads, intermediate: c.codecPostIntermediateSize, window: 128)
    }
    func encode(_ x: MLXArray) -> MLXArray {
        let z = pre(downsample.reduce(x) { $1($0) })
        let (s, sc) = semantic(z)
        let (_, rc) = quantizer(z - s)
        return concatenated([sc, rc], axis: 1)
    }
    func decode(_ ids: MLXArray) -> MLXArray {
        let s = semantic.decode(ids[0..., 0..<1, 0...])
        let r = quantizer.decode(ids[0..., 1..., 0...])
        return upsample.reduce(post(s + r)) { $1($0) }
    }
}

/// The bundled Audio8 codec operates at 44.1 kHz with 2,048 samples per frame.
final class Audio8Codec: Module {
    let encoder: Audio8CodecEncoder
    let quantizer: Audio8DownsampleQuantizer
    let decoder: Audio8CodecDecoder
    init(_ config: Audio8Config) {
        encoder = Audio8CodecEncoder(); quantizer = Audio8DownsampleQuantizer(config); decoder = Audio8CodecDecoder()
    }
    func encode(_ audio: MLXArray) -> MLXArray {
        let waveform = audio.reshaped([1, audio.size, 1])
        let right = (2048 - audio.size % 2048) % 2048
        return quantizer.encode(encoder(padded(waveform, widths: [0, .init((0, right)), 0]))).asType(.int32)
    }
    func decode(_ codes: MLXArray) -> MLXArray {
        decoder(quantizer.decode(codes))[0, 0..., 0]
    }
}
