import Foundation
import MLX
import MLXNN

/// Weight-normalized causal convolution; normalized checkpoint kernels use [out,in,kernel].
final class VoxtralTTSConv: Module {
    @ParameterInfo var weight: MLXArray
    let stride: Int
    let padding: String
    init(output: Int, input: Int, kernel: Int, stride: Int = 1, padding: String = "replicate") {
        self.stride = stride; self.padding = padding
        _weight.wrappedValue = .zeros([output,input,kernel])
    }
    func pad(_ x: MLXArray, left: Int) -> MLXArray {
        guard left > 0 else { return x }
        if padding == "replicate" { return concatenated([repeated(x[0...,0..<1,0...], count: left, axis: 1), x], axis: 1) }
        var h = x
        let extra = max(0, left - x.dim(1) + 1)
        if extra > 0 { h = padded(h, widths: [.init(0), .init((0,extra)), .init(0)]) }
        let indices = MLXArray(Array((1...left).reversed()))
        let reflected = take(h, indices, axis: 1)
        let result = concatenated([reflected,h], axis: 1)
        return extra > 0 ? result[0...,0..<(result.dim(1) - extra),0...] : result
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if stride > 1 {
            return convTransposed1d(x, weight.transposed(1,2,0), stride: stride)[0...,0..<(x.dim(1) * stride),0...]
        }
        return conv1d(pad(x, left: weight.dim(2) - 1), weight.transposed(0,2,1))
    }
}

final class VoxtralTTSDecoderTransformer: Module {
    let layers: [VoxtralTTSBlock]
    init(_ c: VoxtralTTSConfiguration, count: Int) {
        layers = (0..<count).map { _ in VoxtralTTSBlock(dim: c.tokenizerDim, hidden: c.tokenizerHiddenDim,
            heads: c.tokenizerHeads, kvHeads: c.tokenizerKVHeads, headDim: c.tokenizerHeadDim,
            eps: c.tokenizerNormEps, qkEps: c.qkNorm ? c.qkNormEps : nil,
            layerScale: c.layerScale ? c.layerScaleInit : nil) }
    }
    func callAsFunction(_ input: MLXArray, slopes: MLXArray, window: Int) -> MLXArray {
        var x = input
        for layer in layers { x = layer(x, slopes: slopes, window: window) }
        return x
    }
}

final class VoxtralTTSDecoderStage: Module {
    let conv: VoxtralTTSConv
    let transformer: VoxtralTTSDecoderTransformer
    init(_ c: VoxtralTTSConfiguration, index: Int) {
        conv = VoxtralTTSConv(output: c.tokenizerDim, input: index == 0 ? c.semanticDim + c.acousticCodebooks : c.tokenizerDim,
                              kernel: c.decoderKernels[index], stride: c.decoderStrides[index])
        transformer = VoxtralTTSDecoderTransformer(c, count: c.decoderLengths[index])
    }
}

final class VoxtralTTSAudioDecoder: Module {
    @ParameterInfo(key: "cluster_usage") var clusterUsage: MLXArray
    @ParameterInfo(key: "embedding_sum") var embeddingSum: MLXArray
    let stages: [VoxtralTTSDecoderStage]
    @ModuleInfo(key: "output_proj") var outputProjection: VoxtralTTSConv
    private let configuration: VoxtralTTSConfiguration
    private let _slopes: MLXArray
    init(_ c: VoxtralTTSConfiguration) {
        configuration = c
        _clusterUsage.wrappedValue = .ones([c.semanticSize])
        _embeddingSum.wrappedValue = .zeros([c.semanticSize,c.semanticDim])
        stages = c.decoderLengths.indices.map { VoxtralTTSDecoderStage(c, index: $0) }
        _outputProjection.wrappedValue = VoxtralTTSConv(output: c.patchSize, input: c.tokenizerDim,
                                                       kernel: c.patchKernel, padding: "reflect")
        _slopes = Self.alibiSlopes(c.tokenizerHeads)
    }
    static func alibiSlopes(_ heads: Int) -> MLXArray {
        func powerOfTwo(_ n: Int) -> [Float] {
            let start = pow(Float(2), -pow(Float(2), -(log2(Float(n)) - 3)))
            return (0..<n).map { start * pow(start, Float($0)) }
        }
        let closest = 1 << Int(floor(log2(Double(heads))))
        var values = powerOfTwo(closest)
        if closest != heads { values += stride(from: 0, to: 2 * closest, by: 2).prefix(heads - closest).map { powerOfTwo(2 * closest)[$0] } }
        return MLXArray(values)
    }
    func codebook(_ codes: MLXArray) -> MLXArray {
        let semantic = codes[0...,0...,0] - 2
        let acoustic = codes[0...,0...,1...] - 2
        let vectors = embeddingSum.asType(.float32) / maximum(clusterUsage.asType(.float32)[0...,.newAxis], 1e-8)
        return concatenated([vectors[semantic], 2 * acoustic.asType(.float32) / Float(configuration.acousticSize - 1) - 1], axis: -1)
    }
    func decode(_ codes: MLXArray) -> MLXArray {
        var x = codebook(codes)
        for (index, stage) in stages.enumerated() {
            x = stage.conv(x)
            let window = index < 3 ? 2 << index : 16
            x = stage.transformer(x, slopes: _slopes, window: window)
        }
        return outputProjection(x).reshaped(codes.dim(0),-1)
    }
}
