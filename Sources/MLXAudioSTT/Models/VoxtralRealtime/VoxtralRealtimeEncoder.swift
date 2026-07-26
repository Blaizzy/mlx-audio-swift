import Foundation
import MLX
import MLXNN

struct VoxtralRealtimeEncoderKVCache {
    var keys: MLXArray   // [kv_len, n_heads * head_dim]
    var values: MLXArray // [kv_len, n_heads * head_dim]
    var positionOffset: Int
}

/// Carried causal-conv state for the incremental conv stem — see `convStemStep`.
/// `nil` means "not started": the first step seeds each carry with the causal
/// zero left-pad the offline `VoxtralRealtimeCausalConv1d` would apply.
struct VoxtralRealtimeConvStemState {
    var conv1Carry: MLXArray? // [1, 2, nMels] — last two cast mel input frames
    var conv2Carry: MLXArray? // [1, 1..2, dim] — conv1 output suffix from the next stride-2 window
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

    // cos/sin are float32 for precision; cast down so rotating fp16 q/k stays fp16.
    let cosE = cos.expandedDimensions(axis: 1).asType(x.dtype)
    let sinE = sin.expandedDimensions(axis: 1).asType(x.dtype)

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

/// Attention inputs identical for every transformer layer of one encoder forward
/// pass: the interleaved-RoPE cos/sin tables for `positions` and the SDPA mask.
/// Built once per pass instead of per layer — same operations, so layer outputs
/// are bit-identical; only the `nLayers` redundant kernel launches go away (which
/// dominate streaming encoder steps, where each pass covers a few new frames).
struct VoxtralRealtimeEncoderAttentionInputs {
    let ropeCos: MLXArray
    let ropeSin: MLXArray
    let mask: MLXFast.ScaledDotProductAttentionMaskMode

    /// Build the shared inputs for one forward pass of `seqLen` frames at
    /// `positions`, extending `caches`. One mask can serve every layer because the
    /// encoder always advances its layer caches in lockstep (created, extended,
    /// trimmed, and reset together — asserted here). `dtype` is the q/k dtype the
    /// mask must match; the uniform-precision projections preserve the input dtype.
    static func build(
        positions: MLXArray,
        seqLen: Int,
        caches: [VoxtralRealtimeEncoderKVCache?],
        slidingWindow: Int,
        headDim: Int,
        ropeTheta: Float,
        dtype: DType
    ) -> VoxtralRealtimeEncoderAttentionInputs {
        let (cos, sin) = voxtralComputeRopeFrequencies(
            positions: positions,
            headDim: headDim,
            theta: ropeTheta
        )

        // Collapses "no caches" and "[nil, ...]" into one cache-less case. Nil-ness
        // must be uniform across layers: nil and present-but-empty caches select
        // different mask branches below.
        let cache = caches.first ?? nil
        let cachedLen = cache?.keys.shape[0] ?? 0
        let cachedOffset = cache?.positionOffset ?? 0
        precondition(
            caches.allSatisfy {
                ($0 == nil) == (cache == nil)
                    && ($0?.keys.shape[0] ?? 0) == cachedLen
                    && ($0?.positionOffset ?? 0) == cachedOffset
            },
            "encoder layer caches must advance in lockstep to share one attention mask"
        )

        // Mirror the concat + sliding-window trim the attention applies to its
        // cache, so the mask covers the exact key positions each layer will use.
        var positionOffset = cachedOffset
        var kvLen = cachedLen + seqLen
        if kvLen > slidingWindow {
            positionOffset += kvLen - slidingWindow
            kvLen = slidingWindow
        }

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
            // Match the activation dtype: a float32 mask over fp16 q/k aborts SDPA.
            let mask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9)).asType(dtype)
            maskMode = .array(mask)
        }

        return VoxtralRealtimeEncoderAttentionInputs(
            ropeCos: cos,
            ropeSin: sin,
            mask: maskMode
        )
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
        inputs: VoxtralRealtimeEncoderAttentionInputs,
        cache: VoxtralRealtimeEncoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeEncoderKVCache) {
        let seqLen = x.shape[0]

        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        q = voxtralApplyInterleavedRoPE(
            q, cos: inputs.ropeCos, sin: inputs.ropeSin, nHeads: nHeads, headDim: headDim)
        k = voxtralApplyInterleavedRoPE(
            k, cos: inputs.ropeCos, sin: inputs.ropeSin, nHeads: nHeads, headDim: headDim)

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

        let attn = MLXFast.scaledDotProductAttention(
            queries: q4,
            keys: k4,
            values: v4,
            scale: scale,
            mask: inputs.mask
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
        inputs: VoxtralRealtimeEncoderAttentionInputs,
        cache: VoxtralRealtimeEncoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeEncoderKVCache) {
        var out = x

        var h = attentionNorm(out)
        let attnOut = attention(h, inputs: inputs, cache: cache)
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
        // mel is float32; cast to the conv weight dtype so encoder activations stay fp16.
        var x = mel.asType(convLayers0Conv.conv.weight.dtype).transposed(1, 0).expandedDimensions(axis: 0)
        x = gelu(convLayers0Conv(x))
        x = gelu(convLayers1Conv(x))
        x = x.squeezed(axis: 0)

        let trunc = x.shape[0] % config.downsampleFactor
        if trunc > 0 {
            x = x[trunc...]
        }

        return x
    }

    /// Shared per-pass attention inputs (RoPE tables + mask) for `seqLen` frames
    /// at `positions`, extending `caches` — see
    /// `VoxtralRealtimeEncoderAttentionInputs`.
    private func attentionInputs(
        positions: MLXArray,
        seqLen: Int,
        caches: [VoxtralRealtimeEncoderKVCache?],
        dtype: DType
    ) -> VoxtralRealtimeEncoderAttentionInputs {
        VoxtralRealtimeEncoderAttentionInputs.build(
            positions: positions,
            seqLen: seqLen,
            caches: caches,
            slidingWindow: config.slidingWindow,
            headDim: config.headDim,
            ropeTheta: config.ropeTheta,
            dtype: dtype
        )
    }

    /// Incremental counterpart of `convStem`: consume new mel columns
    /// (`[nMels, nNew]`) and return every conv-stem row (`[nNewRows, dim]`) they
    /// complete, matching the rows `convStem` produces at the same absolute indices.
    /// Both convs are causal (zero left-pad), so a row is exact once its input
    /// window is exact: conv1 (k3 s1) carries its last 2 cast input frames; conv2
    /// (k3 s2) carries the suffix from the start of the next stride-2 window
    /// (1 frame after an even total input count, 2 after odd), keeping the
    /// downsample phase aligned for arbitrary chunk sizes. `convStem`'s leading
    /// `% downsampleFactor` truncation is a no-op for the session's whole-token
    /// streams and is not replicated.
    func convStemStep(_ mel: MLXArray, state: inout VoxtralRealtimeConvStemState) -> MLXArray {
        let dtype = convLayers0Conv.conv.weight.dtype
        guard mel.shape[1] > 0 else { return MLXArray.zeros([0, config.dim], type: Float.self).asType(dtype) }

        // Same cast + layout as `convStem`: [1, nNew, nMels].
        let x = mel.asType(dtype).transposed(1, 0).expandedDimensions(axis: 0)

        let pad1 = convLayers0Conv.padding
        let carry1 = state.conv1Carry
            ?? MLXArray.zeros([1, pad1, x.shape[2]], type: Float.self).asType(dtype)
        let in1 = MLX.concatenated([carry1, x], axis: 1)
        state.conv1Carry = in1[0..., (in1.shape[1] - pad1)..., 0...]
        let h = gelu(convLayers0Conv.conv(in1)) // stride 1 ⇒ one row per new mel column

        let pad2 = convLayers1Conv.padding
        let carry2 = state.conv2Carry
            ?? MLXArray.zeros([1, pad2, h.shape[2]], type: Float.self).asType(dtype)
        let in2 = MLX.concatenated([carry2, h], axis: 1)
        let kernel = convLayers1Conv.kernelSize
        let stride = convLayers1Conv.stride
        let newRows = in2.shape[1] >= kernel ? (in2.shape[1] - kernel) / stride + 1 : 0
        state.conv2Carry = in2[0..., (newRows * stride)..., 0...]
        guard newRows > 0 else { return MLXArray.zeros([0, config.dim], type: Float.self).asType(dtype) }

        // The valid (unpadded) conv over `in2` yields exactly `newRows` rows; the
        // 1–2 trailing frames it cannot window are what `conv2Carry` re-presents
        // to the next call.
        return gelu(convLayers1Conv.conv(in2)).squeezed(axis: 0)
    }

    func encodeFull(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.shape[0]
        let positions = MLXArray(0..<seqLen).asType(.int32)
        let inputs = attentionInputs(
            positions: positions, seqLen: seqLen, caches: [], dtype: convOut.dtype)

        var x = convOut
        for layer in transformerLayers {
            x = layer(x, inputs: inputs, cache: nil).0
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
            // Per-chunk, not per-pass: the caches' offset/history advance between
            // chunks, so these inputs cannot be hoisted out of this loop.
            let inputs = attentionInputs(
                positions: positions, seqLen: chunkEnd - chunkStart, caches: caches,
                dtype: x.dtype)

            for i in transformerLayers.indices {
                let next = transformerLayers[i](x, inputs: inputs, cache: caches[i])
                x = next.0
                caches[i] = next.1
            }

            outputs.append(transformerNorm(x))
            chunkStart = chunkEnd
        }

        let encoded = outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 0)
        return downsampleAndProject(encoded)
    }

    /// Feed a block of new conv-stem frames at absolute positions `[startPos, startPos+n)`
    /// through the transformer with persistent per-layer KV-caches, returning the
    /// transformer-normed frames (pre-downsample). While the total fed length stays
    /// `<= slidingWindow` the caches never trim, so the result is bit-identical to
    /// `encodeFull` over the same prefix — see `VoxtralRealtimeStreamSession`.
    func encodeIncremental(
        _ convBlock: MLXArray,
        startPos: Int,
        caches: inout [VoxtralRealtimeEncoderKVCache?]
    ) -> MLXArray {
        var x = convBlock
        let positions = MLXArray(startPos..<(startPos + convBlock.shape[0])).asType(.int32)
        let inputs = attentionInputs(
            positions: positions, seqLen: convBlock.shape[0], caches: caches, dtype: x.dtype)
        for i in transformerLayers.indices {
            let next = transformerLayers[i](x, inputs: inputs, cache: caches[i])
            x = next.0
            caches[i] = next.1
        }
        return transformerNorm(x)
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
