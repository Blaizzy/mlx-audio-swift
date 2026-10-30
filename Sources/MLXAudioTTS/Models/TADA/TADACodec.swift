import Foundation
import MLX
import MLXNN

func tadaGELU(_ x: MLXArray) -> MLXArray { x * 0.5 * (1 + erf(x / sqrt(Float(2)))) }

final class TADASnake: Module {
    var alpha: MLXArray
    init(_ dim: Int) { alpha = .ones([1, 1, dim]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + square(sin(alpha * x)) / alpha }
}
final class TADAResidual: Module {
    let snake1: TADASnake
    let snake2: TADASnake
    let conv1: Conv1d
    let conv2: Conv1d
    init(_ dim: Int, dilation: Int) {
        snake1 = TADASnake(dim); snake2 = TADASnake(dim)
        conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3 * dilation, dilation: dilation)
        conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + conv2(snake2(conv1(snake1(x)))) }
}
final class TADADecoderBlock: Module {
    let snake: TADASnake
    @ModuleInfo(key: "conv_transpose") var conv: ConvTransposed1d
    let res1: TADAResidual
    let res2: TADAResidual
    let res3: TADAResidual
    init(_ dim: Int, stride: Int) {
        snake = TADASnake(dim)
        let pad = (stride + 1) / 2
        _conv = ModuleInfo(wrappedValue: ConvTransposed1d(inputChannels: dim, outputChannels: dim / 2,
            kernelSize: 2 * stride, stride: stride, padding: pad, outputPadding: 0), key: "conv_transpose")
        res1 = TADAResidual(dim / 2, dilation: 1)
        res2 = TADAResidual(dim / 2, dilation: 3)
        res3 = TADAResidual(dim / 2, dilation: 9)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { res3(res2(res1(conv(snake(x))))) }
}
final class TADAWaveDecoder: Module {
    @ModuleInfo(key: "initial_conv") var initial: Conv1d
    let blocks: [TADADecoderBlock]
    @ModuleInfo(key: "final_snake") var snake: TADASnake
    @ModuleInfo(key: "final_conv") var final: Conv1d
    init(hidden: Int = 1024, dim: Int = 96, strides: [Int] = [4, 4, 5, 6]) {
        let channels = dim * (1 << strides.count)
        _initial = ModuleInfo(wrappedValue: Conv1d(inputChannels: hidden, outputChannels: channels, kernelSize: 7, padding: 3), key: "initial_conv")
        blocks = strides.enumerated().map { i, stride in TADADecoderBlock(channels / (1 << i), stride: stride) }
        _snake = ModuleInfo(wrappedValue: TADASnake(dim), key: "final_snake")
        _final = ModuleInfo(wrappedValue: Conv1d(inputChannels: dim, outputChannels: 1, kernelSize: 7, padding: 3), key: "final_conv")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = initial(x)
        for block in blocks { h = block(h) }
        return tanh(final(snake(h)))
    }
}
final class TADAEncoderBlock: Module {
    let res1: TADAResidual
    let res2: TADAResidual
    let res3: TADAResidual
    let snake: TADASnake
    let conv: Conv1d
    init(_ dim: Int, stride: Int) {
        res1 = TADAResidual(dim, dilation: 1); res2 = TADAResidual(dim, dilation: 3); res3 = TADAResidual(dim, dilation: 9)
        snake = TADASnake(dim)
        conv = Conv1d(inputChannels: dim, outputChannels: dim * 2, kernelSize: 2 * stride, stride: stride, padding: (stride + 1) / 2)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(snake(res3(res2(res1(x))))) }
}
final class TADAWaveEncoder: Module {
    @ModuleInfo(key: "initial_conv") var initial: Conv1d
    let blocks: [TADAEncoderBlock]
    @ModuleInfo(key: "final_snake") var snake: TADASnake
    @ModuleInfo(key: "final_conv") var final: Conv1d
    init(hidden: Int = 1024, dim: Int = 64, strides: [Int] = [6, 5, 4, 4]) {
        _initial = ModuleInfo(wrappedValue: Conv1d(inputChannels: 1, outputChannels: dim, kernelSize: 7, padding: 3), key: "initial_conv")
        blocks = strides.enumerated().map { i, stride in TADAEncoderBlock(dim * (1 << i), stride: stride) }
        let channels = dim * (1 << strides.count)
        _snake = ModuleInfo(wrappedValue: TADASnake(channels), key: "final_snake")
        _final = ModuleInfo(wrappedValue: Conv1d(inputChannels: channels, outputChannels: hidden, kernelSize: 3, padding: 1), key: "final_conv")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = initial(x)
        for block in blocks { h = block(h) }
        return final(snake(h))
    }
}

/// True entries block attention; a frame sees its own and previous text segment.
func tadaSegmentMask(_ boundaries: MLXArray) -> MLXArray {
    let ids = cumsum(boundaries, axis: 1) - boundaries
    let row = ids.expandedDimensions(axis: 2), column = ids.expandedDimensions(axis: 1)
    return .!(column .== row .|| column .== (row - 1))
}
func tadaEncoderSegmentMask(_ boundaries: MLXArray) -> MLXArray {
    let ids = cumsum(boundaries, axis: 1)
    let row = ids.expandedDimensions(axis: 2), column = ids.expandedDimensions(axis: 1)
    let markedRow = boundaries.asType(.bool).expandedDimensions(axis: 2)
    let markedColumn = boundaries.asType(.bool).expandedDimensions(axis: 1)
    let same = row .== column
    let sameValid = same .&& (.!markedColumn .|| (markedRow .&& same))
    let previousValid = (column .== (row - 1)) .&& .!markedColumn
    return .!(sameValid .|| (markedRow .&& previousValid))
}
final class TADALocalAttention: Module {
    let qkv: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    let heads: Int
    let headDim: Int
    init(_ dim: Int, heads: Int) {
        self.heads = heads; headDim = dim / heads
        qkv = Linear(dim, 3 * dim)
        _out = ModuleInfo(wrappedValue: Linear(dim, dim), key: "out_proj")
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim), key: "layer_norm")
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), dim = x.dim(2)
        let parts = split(qkv(x).reshaped(b, n, 3, heads, headDim).transposed(2, 0, 3, 1, 4), parts: 3, axis: 0)
        let q = MLXFast.RoPE(parts[0].squeezed(axis: 0), dimensions: headDim, traditional: true, base: 10000, scale: 1, offset: 0)
        let k = MLXFast.RoPE(parts[1].squeezed(axis: 0), dimensions: headDim, traditional: true, base: 10000, scale: 1, offset: 0)
        let v = parts[2].squeezed(axis: 0)
        let additive = MLX.where(mask.expandedDimensions(axis: 1), MLXArray(-Float.infinity), MLXArray(Float(0)))
        let h = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(headDim)), mask: additive)
        return norm(x + out(h.transposed(0, 2, 1, 3).reshaped(b, n, dim)))
    }
}
final class TADALocalLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: TADALocalAttention
    let linear1: Linear
    let linear2: Linear
    let norm: LayerNorm
    init(_ dim: Int, heads: Int, inner: Int) {
        _attention = ModuleInfo(wrappedValue: TADALocalAttention(dim, heads: heads), key: "self_attn")
        linear1 = Linear(dim, inner); linear2 = Linear(inner, dim); norm = LayerNorm(dimensions: dim)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = attention(x, mask: mask)
        return norm(h + linear2(tadaGELU(linear1(h))))
    }
}
final class TADALocalTransformer: Module {
    let layers: [TADALocalLayer]
    @ModuleInfo(key: "final_norm") var norm: LayerNorm
    init(dim: Int = 1024, layers: Int = 6, heads: Int = 8, inner: Int = 4096) {
        self.layers = (0 ..< layers).map { _ in TADALocalLayer(dim, heads: heads, inner: inner) }
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim), key: "final_norm")
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h, mask: mask) }
        return norm(h)
    }
}
final class TADACodecDecoder: Module {
    @ModuleInfo(key: "decoder_proj") var projection: Linear
    @ModuleInfo(key: "local_attention_decoder") var transformer: TADALocalTransformer
    @ModuleInfo(key: "wav_decoder") var waveform: TADAWaveDecoder
    init(hidden: Int = 1024, embed: Int = 512, dim: Int = 96, layers: Int = 6, heads: Int = 8, inner: Int = 4096, strides: [Int] = [4, 4, 5, 6]) {
        _projection = ModuleInfo(wrappedValue: Linear(embed, hidden), key: "decoder_proj")
        _transformer = ModuleInfo(wrappedValue: TADALocalTransformer(dim: hidden, layers: layers, heads: heads, inner: inner), key: "local_attention_decoder")
        _waveform = ModuleInfo(wrappedValue: TADAWaveDecoder(hidden: hidden, dim: dim, strides: strides), key: "wav_decoder")
    }
    func callAsFunction(_ features: MLXArray, masks: MLXArray) -> MLXArray {
        waveform(transformer(projection(features), mask: tadaSegmentMask(masks)))
    }
}
final class TADACodecEncoder: Module {
    @ModuleInfo(key: "wav_encoder") var waveform: TADAWaveEncoder
    @ModuleInfo(key: "local_attention_encoder") var transformer: TADALocalTransformer
    @ModuleInfo(key: "hidden_linear") var projection: Linear
    @ModuleInfo(key: "pos_emb") var position: Embedding
    let padding: Int
    init(hidden: Int = 1024, embed: Int = 512, dim: Int = 64, layers: Int = 6, heads: Int = 8, inner: Int = 4096, strides: [Int] = [6, 5, 4, 4]) {
        padding = 2 * strides.reduce(1, *)
        _waveform = ModuleInfo(wrappedValue: TADAWaveEncoder(hidden: hidden, dim: dim, strides: strides), key: "wav_encoder")
        _transformer = ModuleInfo(wrappedValue: TADALocalTransformer(dim: hidden, layers: layers, heads: heads, inner: inner), key: "local_attention_encoder")
        _projection = ModuleInfo(wrappedValue: Linear(hidden, embed), key: "hidden_linear")
        _position = ModuleInfo(wrappedValue: Embedding(embeddingCount: 2, dimensions: hidden), key: "pos_emb")
    }
    func features(_ audio: MLXArray, masks: MLXArray) -> MLXArray {
        let input = padded(audio.reshaped(1, -1, 1), widths: [.init(0), .init((0, padding)), .init(0)])
        let h = waveform(input)
        let n = h.dim(1)
        let mask = masks.dim(1) < n ? padded(masks, widths: [.init(0), .init((0, n - masks.dim(1)))]) : masks[0..., 0 ..< n]
        return projection(transformer(h + position(mask), mask: tadaEncoderSegmentMask(mask)))
    }
}
