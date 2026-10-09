import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - HiggsAudioV2 semantic encode path (HuBERT + SemanticEncoder)
//
// Port of mlx_audio (Python):
//   - mlx_audio/stt/models/wav2vec/wav2vec.py  (Wav2Vec2Model, model_type=hubert)
//   - mlx_audio/codec/models/higgs_audio/semantic.py  (SemanticEncoder)
//   - mlx_audio/codec/models/higgs_audio/higgs_audio.py  (_sinc_resample, encode fusion)
//
// All modules run channels-last (NLC) and store conv weights in the
// checkpoint's PyTorch layout [out, in/groups, K], transposing at call time
// (same convention as OmniVoiceConv1d, minus the NCL round-trips).

// MARK: - Conv primitive (PyTorch weight layout, NLC data)

final class OmniVoiceSemanticConv1d: Module {
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

    /// x: [B, T, C_in] -> [B, T', C_out]
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

// MARK: - HuBERT feature extractor

/// One conv layer of the HuBERT feature extractor. Layer 0 carries a
/// per-channel GroupNorm under the (misleading) checkpoint key `layer_norm`.
final class OmniVoiceHubertConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: OmniVoiceSemanticConv1d
    @ModuleInfo(key: "layer_norm") var groupNorm: GroupNorm?

    init(inDim: Int, outDim: Int, kernel: Int, stride: Int, bias: Bool, useGroupNorm: Bool) {
        self._conv.wrappedValue = OmniVoiceSemanticConv1d(
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

final class OmniVoiceHubertFeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [OmniVoiceHubertConvLayer]

    init(config: OmniVoiceAudioTokenizerConfig) {
        var layers: [OmniVoiceHubertConvLayer] = []
        for i in 0..<config.convDim.count {
            layers.append(
                OmniVoiceHubertConvLayer(
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

    /// x: [B, T] waveform -> [B, T', conv_dim[-1]]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.expandedDimensions(axis: -1)  // [B, T, 1]
        for layer in convLayers {
            h = layer(h)
        }
        return h
    }
}

final class OmniVoiceHubertFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    init(config: OmniVoiceAudioTokenizerConfig) {
        let convOut = config.convDim.last ?? 512
        self._layerNorm.wrappedValue = LayerNorm(dimensions: convOut, eps: 1e-5)
        self._projection.wrappedValue = Linear(convOut, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}

// MARK: - HuBERT encoder

/// Weight-normed grouped conv (PyTorch weight_norm(dim=2) layout):
///   weight_g: [1, 1, K], weight_v: [out, in/groups, K]
final class OmniVoiceWeightNormConv1d: Module {
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

    /// x: [B, T, C] -> [B, T', C]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX layout [out, K, in/groups]; weight-norm over all axes except K.
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

/// Weight-normed grouped positional conv embedding (kernel 128, groups 16),
/// followed by even-kernel same-pad trim and GELU.
final class OmniVoiceHubertPositionalConvEmbedding: Module {
    @ModuleInfo(key: "conv") var conv: OmniVoiceWeightNormConv1d

    init(config: OmniVoiceAudioTokenizerConfig) {
        // num_conv_pos_embeddings = 128, num_conv_pos_embedding_groups = 16
        self._conv.wrappedValue = OmniVoiceWeightNormConv1d(
            dim: config.hiddenSize, kernelSize: 128, padding: 64, groups: 16
        )
    }

    /// x: [B, T, D] -> [B, T, D]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        // SamePadLayer: even kernel emits one extra frame; drop the last.
        h = h[0..., 0..<(h.shape[1] - 1), 0...]
        return gelu(h)
    }
}

final class OmniVoiceHubertAttention: Module {
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

final class OmniVoiceHubertFeedForward: Module {
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

/// Post-norm transformer layer (do_stable_layer_norm = false).
final class OmniVoiceHubertEncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: OmniVoiceHubertAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: OmniVoiceHubertFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: OmniVoiceAudioTokenizerConfig) {
        self._attention.wrappedValue = OmniVoiceHubertAttention(
            dim: config.hiddenSize, numHeads: config.numAttentionHeads
        )
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._feedForward.wrappedValue = OmniVoiceHubertFeedForward(
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

final class OmniVoiceHubertEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: OmniVoiceHubertPositionalConvEmbedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [OmniVoiceHubertEncoderLayer]

    init(config: OmniVoiceAudioTokenizerConfig) {
        self._posConvEmbed.wrappedValue = OmniVoiceHubertPositionalConvEmbedding(config: config)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            OmniVoiceHubertEncoderLayer(config: config)
        }
    }

    /// Returns all hidden states: the post-layernorm input plus each layer
    /// output (num_hidden_layers + 1 entries).
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

/// HuBERT model matching the `semantic_model.*` checkpoint keys.
final class OmniVoiceHubertModel: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: OmniVoiceHubertFeatureExtractor
    @ModuleInfo(key: "feature_projection") var featureProjection: OmniVoiceHubertFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: OmniVoiceHubertEncoder

    init(config: OmniVoiceAudioTokenizerConfig) {
        self._featureExtractor.wrappedValue = OmniVoiceHubertFeatureExtractor(config: config)
        self._featureProjection.wrappedValue = OmniVoiceHubertFeatureProjection(config: config)
        self._encoder.wrappedValue = OmniVoiceHubertEncoder(config: config)
    }

    /// x: [B, T] 16 kHz waveform -> mean over all hidden states, [B, T', hidden]
    /// (HiggsAudioV2 averages the full hidden-state stack, not just the last.)
    func meanHiddenStates(_ x: MLXArray) -> MLXArray {
        let features = featureExtractor(x)
        let projected = featureProjection(features)
        let all = encoder.hiddenStates(projected)
        return MLX.mean(MLX.stacked(all, axis: 0), axis: 0)
    }
}

// MARK: - SemanticEncoder (post-HuBERT CNN)

final class OmniVoiceSemanticResidualUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceSemanticConv1d
    @ModuleInfo(key: "conv2") var conv2: OmniVoiceSemanticConv1d

    init(dim: Int, dilation: Int = 1, kernelSize: Int = 3) {
        let pad = (kernelSize - 1) * dilation / 2
        self._conv1.wrappedValue = OmniVoiceSemanticConv1d(
            inChannels: dim, outChannels: dim, kernelSize: kernelSize,
            padding: pad, dilation: dilation, bias: false
        )
        self._conv2.wrappedValue = OmniVoiceSemanticConv1d(
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

final class OmniVoiceSemanticConvBlock: Module {
    @ModuleInfo(key: "res_units") var resUnits: [OmniVoiceSemanticResidualUnit]
    @ModuleInfo(key: "conv") var conv: OmniVoiceSemanticConv1d

    init(dim: Int, stride: Int, dilation: Int, kernelSize: Int, unitKernelSize: Int) {
        self._resUnits.wrappedValue = [
            OmniVoiceSemanticResidualUnit(dim: dim, dilation: dilation, kernelSize: unitKernelSize),
            OmniVoiceSemanticResidualUnit(dim: dim, dilation: dilation, kernelSize: unitKernelSize),
        ]
        self._conv.wrappedValue = OmniVoiceSemanticConv1d(
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

/// SemanticEncoder matching the `encoder_semantic.*` checkpoint keys.
/// Input/output: [B, T, hidden] (strides are [1, 1] — no downsampling).
final class OmniVoiceSemanticEncoder: Module {
    @ModuleInfo(key: "conv") var conv: OmniVoiceSemanticConv1d
    @ModuleInfo(key: "conv_blocks") var convBlocks: [OmniVoiceSemanticConvBlock]

    init(config: OmniVoiceAudioTokenizerConfig) {
        let dim = config.hiddenSize
        let kernel = config.kernelSize
        self._conv.wrappedValue = OmniVoiceSemanticConv1d(
            inChannels: dim, outChannels: dim, kernelSize: kernel,
            padding: (kernel - 1) / 2, bias: false
        )
        // strides/dilations/channel_ratios are [1, 1] for this checkpoint.
        self._convBlocks.wrappedValue = (0..<2).map { _ in
            OmniVoiceSemanticConvBlock(
                dim: dim, stride: 1, dilation: 1,
                kernelSize: kernel, unitKernelSize: config.kernelSize
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        for block in convBlocks {
            h = block(h)
        }
        return h
    }
}

// MARK: - Sinc resampling (torchaudio sinc_interp_hann parity)

/// Resample using Hann-windowed sinc interpolation. Numerical port of
/// `_sinc_resample` in mlx_audio higgs_audio (which itself matches
/// torchaudio.functional.resample). Required for parity with the Python
/// semantic encode path — AVAudioConverter resampling does NOT match.
func omniVoiceSincResample(
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

    // kernel[phase][k], k in 0..<(2*width + origR)
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
