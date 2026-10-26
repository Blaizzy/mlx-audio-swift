import Foundation
import MLX
import MLXFast
import MLXNN
import MLXAudioCore

private let higgsV2SwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = MLX.compile { (gate: MLXArray, up: MLXArray) in
    silu(gate) * up
}

final class HiggsV2RoPE: Module {
    let dims: Int
    let traditional: Bool

    private let _freqs: MLXArray

    init(
        dims: Int,
        traditional: Bool = false,
        base: Float = 500000.0,
        scaleFactor: Float = 32.0,
        lowFreqFactor: Float = 1.0,
        highFreqFactor: Float = 4.0,
        oldContextLen: Float = 8192.0
    ) {
        precondition(dims % 2 == 0, "RoPE dims must be even")
        self.dims = dims
        self.traditional = traditional

        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        let exponents = indices / MLXArray(Float(dims))
        var freqs = MLX.pow(MLXArray(base), exponents)  // base^(i/d), NOT negative!

        let wavelens = MLXArray(2.0 * Float.pi) * freqs

        let lowFreqWavelen = oldContextLen / lowFreqFactor   // e.g., 8192 / 1 = 8192
        let highFreqWavelen = oldContextLen / highFreqFactor // e.g., 8192 / 4 = 2048

        freqs = MLX.where(wavelens .> MLXArray(lowFreqWavelen), freqs * scaleFactor, freqs)

        let isMediumFreq = logicalAnd(wavelens .> MLXArray(highFreqWavelen), wavelens .< MLXArray(lowFreqWavelen))

        let smoothFactors = (MLXArray(oldContextLen) / wavelens - MLXArray(lowFreqFactor))
            / MLXArray(highFreqFactor - lowFreqFactor)

        let denominator = (MLXArray(1.0) - smoothFactors) / MLXArray(scaleFactor) + smoothFactors
        let smoothFreqs = freqs / denominator

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, freqs)

        super.init()
    }

    convenience init(dims: Int, config: LlamaTTSConfiguration) {
        let base = config.ropeTheta
        let rs = config.ropeScaling

        func num(_ k: String, _ d: Float) -> Float {
            guard let v = rs?[k] else { return d }
            switch v {
            case .float(let x): return Float(x)
            case .int(let x): return Float(x)
            case .string(let s): return Float(s) ?? d
            default:
                assertionFailure("unexpected value for \(k): \(v)")
                return d
            }
        }

        self.init(
            dims: dims,
            traditional: config.ropeTraditional,
            base: base,
            scaleFactor: num("factor", rs == nil ? 1.0 : 32.0),
            lowFreqFactor: num("low_freq_factor", 1.0),
            highFreqFactor: num("high_freq_factor", 4.0),
            oldContextLen: num("original_max_position_embeddings", 8192.0)
        )
    }

    func callAsFunction(_ x: MLXArray, offset: Int? = nil) -> MLXArray {
        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,  // Use custom freqs instead
            scale: 1.0,
            offset: offset ?? 0,
            freqs: _freqs
        )
    }
}


final class HiggsV2Attention: Module {
    let args: LlamaTTSConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: HiggsV2RoPE

    init(_ args: LlamaTTSConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = HiggsV2RoPE(dims: headDim, config: args)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(output)
    }
}


final class HiggsV2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: LlamaTTSConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(higgsV2SwiGLU(gate(x), up(x)))
    }
}


final class HiggsV2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: HiggsV2Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "audio_input_layernorm") var audioInputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "audio_post_attention_layernorm") var audioPostNorm: RMSNorm
    @ModuleInfo var mlp: HiggsV2MLP
    @ModuleInfo(key: "audio_mlp") var audioMLP: HiggsV2MLP

    init(_ config: LlamaTTSConfiguration) {
        _attention.wrappedValue = HiggsV2Attention(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _audioInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _audioPostNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = HiggsV2MLP(config)
        _audioMLP.wrappedValue = HiggsV2MLP(config)
    }

    func callAsFunction(_ x: MLXArray, audioMask: MLXArray,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
                        audioOnly: Bool = false) -> MLXArray {
        if audioOnly {
            let h = x + attention(audioInputNorm(x), mask: mask, cache: cache)
            return h + audioMLP(audioPostNorm(h))
        }
        let route = audioMask.expandedDimensions(axis: -1)
        let h = x + attention(MLX.where(route, audioInputNorm(x), inputNorm(x)), mask: mask, cache: cache)
        return h + MLX.where(route, audioMLP(audioPostNorm(h)), mlp(postNorm(h)))
    }
}

final class HiggsV2DecoderProjector: Module {
    @ModuleInfo(key: "text_lm_head") var textHead: Linear
    @ModuleInfo(key: "audio_lm_head") var audioHead: Linear
    init(_ config: HiggsV2Configuration) {
        _textHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabularySize, bias: false)
        _audioHead.wrappedValue = Linear(config.textConfig.hiddenSize,
            config.audioNumCodebooks * (config.audioCodebookSize + 2), bias: false)
    }
}
