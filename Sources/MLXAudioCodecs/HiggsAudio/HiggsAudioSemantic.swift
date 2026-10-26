import Foundation
import MLX
import MLXFast
import MLXNN

public struct HiggsAudioSemanticConfig: Codable, Sendable {
    public var hiddenSize = 768
    public var numHiddenLayers = 12
    public var numAttentionHeads = 12
    public var intermediateSize = 3072
    public var convDim = [512, 512, 512, 512, 512, 512, 512]
    public var convKernel = [10, 3, 3, 3, 3, 2, 2]
    public var convStride = [5, 2, 2, 2, 2, 2, 2]
    public var kernelSize = 3
    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", intermediateSize = "intermediate_size"
        case convDim = "conv_dim", convKernel = "conv_kernel", convStride = "conv_stride"
        case kernelSize = "kernel_size"
    }
    public init(from decoder: Swift.Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? numAttentionHeads
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        convDim = try c.decodeIfPresent([Int].self, forKey: .convDim) ?? convDim
        convKernel = try c.decodeIfPresent([Int].self, forKey: .convKernel) ?? convKernel
        convStride = try c.decodeIfPresent([Int].self, forKey: .convStride) ?? convStride
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? kernelSize
    }
}




final class HiggsAudioSemanticConv1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray  // PyTorch [out, in/groups, K]
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let strideVal: Int
    let paddingVal: Int
    let dilationVal: Int
    let groupsVal: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.strideVal = stride
        self.paddingVal = padding
        self.dilationVal = dilation
        self.groupsVal = groups

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outChannels, inChannels / groups, kernelSize]
        )
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weight.transposed(0, 2, 1).asType(.float32)  // [out, K, in/groups]
        var h = MLX.conv1d(
            x.asType(.float32), w,
            stride: strideVal, padding: paddingVal, dilation: dilationVal, groups: groupsVal
        )
        if let b = bias {
            h = h + b.asType(.float32)
        }
        return h.asType(x.dtype)
    }
}


final class HiggsAudioHubertConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: HiggsAudioSemanticConv1d
    @ModuleInfo(key: "layer_norm") var groupNorm: GroupNorm?

    init(inDim: Int, outDim: Int, kernel: Int, stride: Int, bias: Bool, useGroupNorm: Bool) {
        self._conv.wrappedValue = HiggsAudioSemanticConv1d(
            inChannels: inDim, outChannels: outDim, kernelSize: kernel,
            stride: stride, bias: bias
        )
        self._groupNorm.wrappedValue =
            useGroupNorm
            ? GroupNorm(groupCount: outDim, dimensions: outDim, affine: true, pytorchCompatible: true)
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        if let groupNorm {
            h = groupNorm(h)
        }
        return gelu(h)
    }
}

final class HiggsAudioHubertFeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [HiggsAudioHubertConvLayer]

    init(config: HiggsAudioSemanticConfig) {
        var layers: [HiggsAudioHubertConvLayer] = []
        for i in 0..<config.convDim.count {
            layers.append(
                HiggsAudioHubertConvLayer(
                    inDim: i == 0 ? 1 : config.convDim[i - 1],
                    outDim: config.convDim[i],
                    kernel: config.convKernel[i],
                    stride: config.convStride[i],
                    bias: false,
                    useGroupNorm: i == 0  // feat_extract_norm == "group"
                ))
        }
        self._convLayers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.expandedDimensions(axis: -1)  // [B, T, 1]
        for layer in convLayers {
            h = layer(h)
        }
        return h
    }
}

final class HiggsAudioHubertFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    init(config: HiggsAudioSemanticConfig) {
        let convOut = config.convDim.last ?? 512
        self._layerNorm.wrappedValue = LayerNorm(dimensions: convOut, eps: 1e-5)
        self._projection.wrappedValue = Linear(convOut, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}


final class HiggsAudioWeightNormConv1d: Module {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    let kernelSize: Int
    let paddingVal: Int
    let groupsVal: Int

    init(dim: Int, kernelSize: Int, padding: Int, groups: Int) {
        self.kernelSize = kernelSize
        self.paddingVal = padding
        self.groupsVal = groups
        self._weightG.wrappedValue = MLXArray.ones([1, 1, kernelSize])
        self._weightV.wrappedValue = MLXRandom.uniform(
            low: -0.01, high: 0.01, [dim, dim / groups, kernelSize]
        )
        self._bias.wrappedValue = MLXArray.zeros([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let v = weightV.transposed(0, 2, 1).asType(.float32)
        let g = weightG.transposed(0, 2, 1).asType(.float32)
        let norm = MLX.sqrt(MLX.sum(v * v, axes: [0, 2], keepDims: true))
        let w = g * v / norm

        var h = MLX.conv1d(
            x.asType(.float32), w,
            stride: 1, padding: paddingVal, groups: groupsVal
        )
        h = h + bias.asType(.float32)
        return h.asType(x.dtype)
    }
}

final class HiggsAudioHubertPositionalConvEmbedding: Module {
    @ModuleInfo(key: "conv") var conv: HiggsAudioWeightNormConv1d

    init(config: HiggsAudioSemanticConfig) {
        self._conv.wrappedValue = HiggsAudioWeightNormConv1d(
            dim: config.hiddenSize, kernelSize: 128, padding: 64, groups: 16
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        h = h[0..., 0..<(h.shape[1] - 1), 0...]
        return gelu(h)
    }
}

final class HiggsAudioHubertAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scaling: Float

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scaling = pow(Float(headDim), -0.5)
        self._qProj.wrappedValue = Linear(dim, dim)
        self._kProj.wrappedValue = Linear(dim, dim)
        self._vProj.wrappedValue = Linear(dim, dim)
        self._outProj.wrappedValue = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, D) = (x.shape[0], x.shape[1], x.shape[2])
        func shaped(_ t: MLXArray) -> MLXArray {
            t.reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        }
        let q = shaped(qProj(x) * scaling)
        let k = shaped(kProj(x))
        let v = shaped(vProj(x))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: .none
        )
        return outProj(attn.transposed(0, 2, 1, 3).reshaped([B, T, D]))
    }
}

final class HiggsAudioHubertFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(dim: Int, intermediate: Int) {
        self._intermediateDense.wrappedValue = Linear(dim, intermediate)
        self._outputDense.wrappedValue = Linear(intermediate, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }
}

final class HiggsAudioHubertEncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: HiggsAudioHubertAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: HiggsAudioHubertFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: HiggsAudioSemanticConfig) {
        self._attention.wrappedValue = HiggsAudioHubertAttention(
            dim: config.hiddenSize, numHeads: config.numAttentionHeads
        )
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._feedForward.wrappedValue = HiggsAudioHubertFeedForward(
            dim: config.hiddenSize, intermediate: config.intermediateSize
        )
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = layerNorm(x + attention(x))
        h = finalLayerNorm(h + feedForward(h))
        return h
    }
}

final class HiggsAudioHubertEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: HiggsAudioHubertPositionalConvEmbedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [HiggsAudioHubertEncoderLayer]

    init(config: HiggsAudioSemanticConfig) {
        self._posConvEmbed.wrappedValue = HiggsAudioHubertPositionalConvEmbedding(config: config)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            HiggsAudioHubertEncoderLayer(config: config)
        }
    }

    func hiddenStates(_ x: MLXArray) -> [MLXArray] {
        var h = layerNorm(x + posConvEmbed(x))
        var all: [MLXArray] = [h]
        for layer in layers {
            h = layer(h)
            all.append(h)
        }
        return all
    }
}

public final class HiggsAudioHubertModel: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: HiggsAudioHubertFeatureExtractor
    @ModuleInfo(key: "feature_projection") var featureProjection: HiggsAudioHubertFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: HiggsAudioHubertEncoder

    public init(config: HiggsAudioSemanticConfig) {
        self._featureExtractor.wrappedValue = HiggsAudioHubertFeatureExtractor(config: config)
        self._featureProjection.wrappedValue = HiggsAudioHubertFeatureProjection(config: config)
        self._encoder.wrappedValue = HiggsAudioHubertEncoder(config: config)
    }

    public func meanHiddenStates(_ x: MLXArray) -> MLXArray {
        let features = featureExtractor(x)
        let projected = featureProjection(features)
        let all = encoder.hiddenStates(projected)
        return MLX.mean(MLX.stacked(all, axis: 0), axis: 0)
    }
}


final class HiggsAudioSemanticResidualUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: HiggsAudioSemanticConv1d
    @ModuleInfo(key: "conv2") var conv2: HiggsAudioSemanticConv1d

    init(dim: Int, dilation: Int = 1, kernelSize: Int = 3) {
        let pad = (kernelSize - 1) * dilation / 2
        self._conv1.wrappedValue = HiggsAudioSemanticConv1d(
            inChannels: dim, outChannels: dim, kernelSize: kernelSize,
            padding: pad, dilation: dilation, bias: false
        )
        self._conv2.wrappedValue = HiggsAudioSemanticConv1d(
            inChannels: dim, outChannels: dim, kernelSize: 1, bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = elu(x)
        y = conv1(y)
        y = elu(y)
        y = conv2(y)
        return x + y
    }
}

final class HiggsAudioSemanticConvBlock: Module {
    @ModuleInfo(key: "res_units") var resUnits: [HiggsAudioSemanticResidualUnit]
    @ModuleInfo(key: "conv") var conv: HiggsAudioSemanticConv1d

    init(dim: Int, stride: Int, dilation: Int, kernelSize: Int, unitKernelSize: Int) {
        self._resUnits.wrappedValue = [
            HiggsAudioSemanticResidualUnit(dim: dim, dilation: dilation, kernelSize: unitKernelSize),
            HiggsAudioSemanticResidualUnit(dim: dim, dilation: dilation, kernelSize: unitKernelSize),
        ]
        self._conv.wrappedValue = HiggsAudioSemanticConv1d(
            inChannels: dim, outChannels: dim, kernelSize: kernelSize,
            stride: stride, padding: (kernelSize - 1) / 2, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for unit in resUnits {
            h = unit(h)
        }
        return conv(h)
    }
}

public final class HiggsAudioSemanticEncoder: Module {
    @ModuleInfo(key: "conv") var conv: HiggsAudioSemanticConv1d
    @ModuleInfo(key: "conv_blocks") var convBlocks: [HiggsAudioSemanticConvBlock]

    public init(config: HiggsAudioSemanticConfig) {
        let dim = config.hiddenSize
        let kernel = config.kernelSize
        self._conv.wrappedValue = HiggsAudioSemanticConv1d(
            inChannels: dim, outChannels: dim, kernelSize: kernel,
            padding: (kernel - 1) / 2, bias: false
        )
        self._convBlocks.wrappedValue = (0..<2).map { _ in
            HiggsAudioSemanticConvBlock(
                dim: dim, stride: 1, dilation: 1,
                kernelSize: kernel, unitKernelSize: config.kernelSize
            )
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        for block in convBlocks {
            h = block(h)
        }
        return h
    }
}


public func higgsAudioSincResample(
    _ waveform: [Float],
    from origFreq: Int,
    to newFreq: Int,
    lowpassFilterWidth: Int = 6,
    rolloff: Double = 0.99
) -> [Float] {
    if origFreq == newFreq { return waveform }

    func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
    let g = gcd(origFreq, newFreq)
    let origR = origFreq / g
    let newR = newFreq / g

    let baseFreq = Double(min(origR, newR)) * rolloff
    let width = Int(ceil(Double(lowpassFilterWidth * origR) / baseFreq))

    let kTaps = 2 * width + origR
    var kernel = [[Float]](repeating: [Float](repeating: 0, count: kTaps), count: newR)
    for phase in 0..<newR {
        for k in 0..<kTaps {
            let idx = Double(-width + k) / Double(origR)
            var t = (-Double(phase) / Double(newR) + idx) * baseFreq
            t = min(max(t, -Double(lowpassFilterWidth)), Double(lowpassFilterWidth))
            let window = pow(cos(t * Double.pi / Double(lowpassFilterWidth) / 2), 2)
            let tPi = t * Double.pi
            let sinc = tPi == 0 ? 1.0 : sin(tPi) / tPi
            kernel[phase][k] = Float(sinc * window * (baseFreq / Double(origR)))
        }
    }

    let length = waveform.count
    var padded = [Float](repeating: 0, count: width + length + width + origR)
    padded.replaceSubrange(width..<(width + length), with: waveform)

    let outLen = Int(ceil(Double(length * newR) / Double(origR)))
    var result = [Float](repeating: 0, count: outLen)
    for phase in 0..<newR {
        let taps = kernel[phase]
        var pos = phase
        var start = 0
        while pos < outLen {
            var acc: Float = 0
            for k in 0..<kTaps {
                acc += padded[start + k] * taps[k]
            }
            result[pos] = acc
            pos += newR
            start += origR
        }
    }
    return result
}
