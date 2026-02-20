import Foundation
import MLX
import MLXNN

struct VoxtralRealtimeEncoderKVCache {
    var keys: MLXArray   // [kv_len, n_heads * head_dim]
    var values: MLXArray // [kv_len, n_heads * head_dim]
    var positionOffset: Int
}

func voxtralComputeRopeFrequencies(
    positions: MLXArray,
    headDim: Int,
    theta: Float
) -> (cos: MLXArray, sin: MLXArray) {
    let idx = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
    let invFreq = MLX.exp((-log(theta)) * (idx / Float(headDim)))
    let angles = positions.asType(.float32).expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
    return (MLX.cos(angles), MLX.sin(angles))
}

func voxtralApplyInterleavedRoPE(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    nHeads: Int,
    headDim: Int
) -> MLXArray {
    let seqLen = x.shape[0]
    let halfDim = headDim / 2

    let reshaped = x.reshaped(seqLen, nHeads, halfDim, 2)
    let x1 = reshaped[0..., 0..., 0..., 0]
    let x2 = reshaped[0..., 0..., 0..., 1]

    let cosE = cos.expandedDimensions(axis: 1)
    let sinE = sin.expandedDimensions(axis: 1)

    let o1 = x1 * cosE - x2 * sinE
    let o2 = x2 * cosE + x1 * sinE

    let out = MLX.concatenated(
        [o1.expandedDimensions(axis: -1), o2.expandedDimensions(axis: -1)],
        axis: -1
    )
    return out.reshaped(seqLen, nHeads * headDim)
}

final class VoxtralRealtimeCausalConv1d: Module {
    let kernelSize: Int
    let stride: Int
    let padding: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = kernelSize - stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        if padding > 0 {
            out = MLX.padded(
                out,
                widths: [
                    IntOrPair(0),
                    IntOrPair((padding, 0)),
                    IntOrPair(0),
                ]
            )
        }
        return conv(out)
    }
}

final class VoxtralRealtimeEncoderAttention: Module {
    let nHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: VoxtralRealtimeEncoderConfig) {
        nHeads = config.nHeads
        headDim = config.headDim
        slidingWindow = config.slidingWindow
        ropeTheta = config.ropeTheta
        scale = pow(Float(config.headDim), -0.5)

        let attnDim = config.nHeads * config.headDim
        self._wq.wrappedValue = Linear(config.dim, attnDim, bias: true)
        self._wk.wrappedValue = Linear(config.dim, attnDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, attnDim, bias: true)
        self._wo.wrappedValue = Linear(attnDim, config.dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        cache: VoxtralRealtimeEncoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeEncoderKVCache) {
        let seqLen = x.shape[0]

        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        let (cos, sin) = voxtralComputeRopeFrequencies(
            positions: positions,
            headDim: headDim,
            theta: ropeTheta
        )

        q = voxtralApplyInterleavedRoPE(q, cos: cos, sin: sin, nHeads: nHeads, headDim: headDim)
        k = voxtralApplyInterleavedRoPE(k, cos: cos, sin: sin, nHeads: nHeads, headDim: headDim)

        var positionOffset = cache?.positionOffset ?? 0
        if let cache {
            k = MLX.concatenated([cache.keys, k], axis: 0)
            v = MLX.concatenated([cache.values, v], axis: 0)
        }

        var kvLen = k.shape[0]
        if kvLen > slidingWindow {
            let trim = kvLen - slidingWindow
            k = k[trim...]
            v = v[trim...]
            kvLen = slidingWindow
            positionOffset += trim
        }

        let newCache = VoxtralRealtimeEncoderKVCache(
            keys: k,
            values: v,
            positionOffset: positionOffset
        )

        let q4 = q.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let k4 = k.reshaped(1, kvLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let v4 = v.reshaped(1, kvLen, nHeads, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen == 1 {
            maskMode = .none
        } else if cache == nil && seqLen <= slidingWindow {
            maskMode = .causal
        } else {
            let qPos = positions.expandedDimensions(axis: 1)
            let kPos = MLXArray(positionOffset..<(positionOffset + kvLen)).asType(.int32).expandedDimensions(axis: 0)
            let causal = kPos .<= qPos
            let window = kPos .>= (qPos - MLXArray(Int32(slidingWindow - 1)))
            let allowed = logicalAnd(causal, window)
            let mask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9))
            maskMode = .array(mask)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q4,
            keys: k4,
            values: v4,
            scale: scale,
            mask: maskMode
        )

        let out = attn.transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)
        return (wo(out), newCache)
    }
}

final class VoxtralRealtimeEncoderLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralRealtimeEncoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    @ModuleInfo(key: "feed_forward_w1") var feedForwardW1: Linear
    @ModuleInfo(key: "feed_forward_w3") var feedForwardW3: Linear
    @ModuleInfo(key: "feed_forward_w2") var feedForwardW2: Linear

    init(_ config: VoxtralRealtimeEncoderConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = VoxtralRealtimeEncoderAttention(config)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        self._feedForwardW1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        cache: VoxtralRealtimeEncoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeEncoderKVCache) {
        var out = x

        var h = attentionNorm(out)
        let attnOut = attention(h, positions: positions, cache: cache)
        h = attnOut.0
        out = out + h

        h = ffnNorm(out)
        let gate = silu(feedForwardW1(h))
        let up = feedForwardW3(h)
        out = out + feedForwardW2(gate * up)

        return (out, attnOut.1)
    }
}

final class VoxtralRealtimeAudioEncoder: Module {
    let config: VoxtralRealtimeEncoderConfig
    let decoderDim: Int

    @ModuleInfo(key: "conv_layers_0_conv") var convLayers0Conv: VoxtralRealtimeCausalConv1d
    @ModuleInfo(key: "conv_layers_1_conv") var convLayers1Conv: VoxtralRealtimeCausalConv1d

    @ModuleInfo(key: "transformer_layers") var transformerLayers: [VoxtralRealtimeEncoderLayer]
    @ModuleInfo(key: "transformer_norm") var transformerNorm: RMSNorm

    @ModuleInfo(key: "audio_language_projection_0") var audioLanguageProjection0: Linear
    @ModuleInfo(key: "audio_language_projection_2") var audioLanguageProjection2: Linear

    init(_ config: VoxtralRealtimeEncoderConfig, decoderDim: Int = 3072) {
        self.config = config
        self.decoderDim = decoderDim

        self._convLayers0Conv.wrappedValue = VoxtralRealtimeCausalConv1d(
            inChannels: 128,
            outChannels: config.dim,
            kernelSize: 3,
            stride: 1
        )
        self._convLayers1Conv.wrappedValue = VoxtralRealtimeCausalConv1d(
            inChannels: config.dim,
            outChannels: config.dim,
            kernelSize: 3,
            stride: 2
        )

        self._transformerLayers.wrappedValue = (0..<config.nLayers).map { _ in
            VoxtralRealtimeEncoderLayer(config)
        }
        self._transformerNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        let adapterInputDim = config.dim * config.downsampleFactor
        self._audioLanguageProjection0.wrappedValue = Linear(adapterInputDim, decoderDim, bias: false)
        self._audioLanguageProjection2.wrappedValue = Linear(decoderDim, decoderDim, bias: false)
    }

    func convStem(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed(1, 0).expandedDimensions(axis: 0)
        x = gelu(convLayers0Conv(x))
        x = gelu(convLayers1Conv(x))
        x = x.squeezed(axis: 0)

        let trunc = x.shape[0] % config.downsampleFactor
        if trunc > 0 {
            x = x[trunc...]
        }

        return x
    }

    func encodeFull(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.shape[0]
        let positions = MLXArray(0..<seqLen).asType(.int32)

        var x = convOut
        for layer in transformerLayers {
            x = layer(x, positions: positions, cache: nil).0
        }

        x = transformerNorm(x)
        return downsampleAndProject(x)
    }

    func encodeChunked(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.shape[0]
        let sw = config.slidingWindow

        if seqLen <= sw {
            return encodeFull(convOut)
        }

        var caches: [VoxtralRealtimeEncoderKVCache?] = Array(repeating: nil, count: transformerLayers.count)
        var outputs: [MLXArray] = []

        var chunkStart = 0
        while chunkStart < seqLen {
            let chunkEnd = min(chunkStart + sw, seqLen)
            var x = convOut[chunkStart..<chunkEnd, 0...]
            let positions = MLXArray(chunkStart..<chunkEnd).asType(.int32)

            for i in transformerLayers.indices {
                let next = transformerLayers[i](x, positions: positions, cache: caches[i])
                x = next.0
                caches[i] = next.1
            }

            outputs.append(transformerNorm(x))
            chunkStart = chunkEnd
        }

        let encoded = outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 0)
        return downsampleAndProject(encoded)
    }

    func downsampleAndProject(_ encoded: MLXArray) -> MLXArray {
        let seqLen = encoded.shape[0]
        let ds = config.downsampleFactor
        let dsLen = seqLen / ds

        if dsLen == 0 {
            return encoded[0..<0, 0...]
        }

        var x = encoded[0..<(dsLen * ds), 0...].reshaped(dsLen, config.dim * ds)
        x = gelu(audioLanguageProjection0(x))
        return audioLanguageProjection2(x)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let convOut = convStem(mel)
        if convOut.shape[0] <= config.slidingWindow {
            return encodeFull(convOut)
        }
        return encodeChunked(convOut)
    }
}
