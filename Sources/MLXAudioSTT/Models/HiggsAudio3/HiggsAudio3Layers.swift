import Foundation
import MLX
import MLXFast
import MLXNN
import MLXAudioCore

final class HiggsAudio3Attention: Module {
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    init(_ config: HiggsAudio3EncoderConfig) {
        heads = config.encoderAttentionHeads
        headDim = config.dModel / heads
        _q.wrappedValue = Linear(config.dModel, config.dModel)
        _k.wrappedValue = Linear(config.dModel, config.dModel, bias: false)
        _v.wrappedValue = Linear(config.dModel, config.dModel)
        _out.wrappedValue = Linear(config.dModel, config.dModel)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), d = x.dim(2)
        let queries = (q(x) * pow(Float(headDim), -0.5)).reshaped(b,t,heads,headDim).transposed(0,2,1,3)
        let keys = k(x).reshaped(b,t,heads,headDim).transposed(0,2,1,3)
        let values = v(x).reshaped(b,t,heads,headDim).transposed(0,2,1,3)
        return out(MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values,
            scale: 1, mask: .none).transposed(0,2,1,3).reshaped(b,t,d))
    }
}

final class HiggsAudio3EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: HiggsAudio3Attention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    init(_ c: HiggsAudio3EncoderConfig) {
        _attention.wrappedValue = HiggsAudio3Attention(c)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: c.dModel)
        _fc1.wrappedValue = Linear(c.dModel, c.encoderFfnDim)
        _fc2.wrappedValue = Linear(c.encoderFfnDim, c.dModel)
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.dModel)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(attentionNorm(x))
        return h + fc2(gelu(fc1(finalNorm(h))))
    }
}

public final class HiggsAudio3Encoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positions: Embedding
    @ModuleInfo(key: "layers") var layers: [HiggsAudio3EncoderLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    public init(_ c: HiggsAudio3EncoderConfig) {
        _conv1.wrappedValue = Conv1d(inputChannels: c.numMelBins, outputChannels: c.dModel, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: c.dModel, outputChannels: c.dModel, kernelSize: 3, stride: 2, padding: 1)
        _positions.wrappedValue = Embedding(embeddingCount: c.maxSourcePositions, dimensions: c.dModel)
        _layers.wrappedValue = (0..<c.encoderLayers).map { _ in HiggsAudio3EncoderLayer(c) }
        _norm.wrappedValue = LayerNorm(dimensions: c.dModel)
    }
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = gelu(conv1(features.transposed(0,2,1)))
        x = gelu(conv2(x))
        x = x + positions.weight[0..<x.dim(1)].expandedDimensions(axis: 0)
        for layer in layers { x = layer(x) }
        let b = x.dim(0), pairs = x.dim(1) / 2, d = x.dim(2)
        return norm(x[0..., 0..<(pairs * 2), 0...].reshaped(b,pairs,2,d).mean(axis: 2))
    }
}

public final class HiggsAudio3Projector: Module {
    @ModuleInfo(key: "temporal") var temporal: Conv1d?
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    public init(_ c: HiggsAudio3Config) {
        let d = c.audioEncoderConfig.dModel
        _temporal.wrappedValue = c.projectorTemporalDownsample > 1
            ? Conv1d(inputChannels: d, outputChannels: d, kernelSize: 3, stride: 2, padding: 1, groups: d) : nil
        _linear1.wrappedValue = Linear(d, 2048)
        _linear2.wrappedValue = Linear(2048, c.textConfig.hiddenSize)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(relu(linear1(temporal?(x) ?? x)))
    }
}

/// The Higgs frontend uses a symmetric Hann window and Slaney mel scale.
/// It deliberately drops the final centered STFT frame, matching Python.
public enum HiggsAudio3Frontend {
    public static func features(_ waveform: MLXArray, numMelBins: Int = 128, sampleRate: Int = 16000) -> MLXArray {
        var audio = waveform.reshaped([-1]).asType(.float32)
        // Keep reflection and the encoder's pair pooling defined for tiny clips.
        if audio.size < 640 { audio = padded(audio, widths: [.init((0, 640 - audio.size))]) }
        let spectrum = stft(audio: audio, window: hanningWindow(size: 400), nFft: 400, hopLength: 160)
        let power = abs(spectrum[0..<(spectrum.dim(0) - 1)]).square()
        let mel = matmul(power, melFilters(sampleRate: sampleRate, nFft: 400, nMels: numMelBins,
            norm: "slaney", melScale: .slaney))
        var logMel = log10(maximum(mel, MLXArray(Float(1e-10))))
        logMel = maximum(logMel, logMel.max() - 8)
        return ((logMel + 4) / 4).transposed().expandedDimensions(axis: 0)
    }
}
