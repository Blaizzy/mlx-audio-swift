import Foundation
import MLX
import MLXNN

// Cache-aware streaming for Nemotron 3.5 ASR (mirrors the model's native mode).
// Each conformer layer keeps an attention cache (last `leftCache` attention-input
// frames) and a conv cache (last `convKernel-1` GLU-output frames); subsampling is
// incremental with a 16-frame mel cache. Output is frame-identical to the offline
// (chunked_limited) encoder at the native chunk size (rightContext + 1), so the
// streamed transcript equals `decode(...)`.

private let nemoPreEncodeMelCache = 16  // >= causal receptive field of 8x dw-striding

extension NemotronASRModel {
    private func nemoStreamBlock(
        _ block: NemotronASRConformerBlock,
        _ x: MLXArray,
        attnCache: MLXArray?,
        convCache: MLXArray?,
        leftCache: Int,
        convLeft: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        var residual = x + MLXArray(Float(0.5)).asType(x.dtype) * block.feedForward1(block.normFeedForward1(x))

        // cache-aware self-attention (Q = chunk, K/V = [cache ++ chunk])
        let xn = block.normSelfAtt(residual)
        let cacheLen = attnCache?.shape[1] ?? 0
        let kv = attnCache == nil ? xn : MLX.concatenated([attnCache!, xn], axis: 1)
        let posEmb = encoder.posEnc(xn, offset: cacheLen).1
        residual = residual + block.selfAttn(xn, kv, kv, posEmb: posEmb, mask: nil)
        let kvLen = kv.shape[1]
        let attnNext = kv[0..., max(0, kvLen - leftCache)..<kvLen, 0...]

        // cache-aware causal conv (prepend conv cache instead of zero-padding)
        let xc = block.normConv(residual)
        let pw = block.conv.pointwiseConv1(xc)
        let sp = pw.split(parts: 2, axis: 2)
        let g = sp[0] * sigmoid(sp[1])  // (1, c, d)
        let cc = convCache ?? MLXArray.zeros([g.shape[0], convLeft, g.shape[2]], dtype: g.dtype)
        let din = MLX.concatenated([cc, g], axis: 1)
        let dw = block.conv.depthwiseConv(din)
        let dinLen = din.shape[1]
        let convNext = din[0..., max(0, dinLen - convLeft)..<dinLen, 0...]
        var y = block.conv.batchNorm(dw)
        y = silu(y)
        residual = residual + block.conv.pointwiseConv2(y)

        residual = residual + MLXArray(Float(0.5)).asType(residual.dtype)
            * block.feedForward2(block.normFeedForward2(residual))
        return (block.normOut(residual), attnNext, convNext)
    }

    /// Run encoder + prompt fusion in cache-aware chunks, invoking `onChunk` with
    /// each chunk's post-prompt encoder frames (1, c, d). Frame-identical to offline.
    func cacheAwareStreamEncode(
        _ mel: MLXArray,
        language: String?,
        chunkFrames: Int? = nil,
        onChunk: (MLXArray) -> Void
    ) {
        var features = mel
        if features.ndim == 2 { features = features.expandedDimensions(axis: 0) }
        features = features.asType(computeDType)

        let sf = encoderConfig.subsamplingFactor
        let right = defaultAttContextSize.count > 1 ? defaultAttContextSize[1] : 13
        let cf = chunkFrames ?? max(1, right + 1)
        let chunkMel = cf * sf
        let leftCache = defaultAttContextSize.first ?? 56
        let convLeft = encoderConfig.convKernelSize - 1
        let n = encoder.layers.count
        let total = features.shape[1]

        var attnCache = [MLXArray?](repeating: nil, count: n)
        var convCache = [MLXArray?](repeating: nil, count: n)
        var melCache: MLXArray?
        var emitted = 0
        var consumed = 0

        while consumed < total {
            let end = min(consumed + chunkMel, total)
            let m = features[0..., consumed..<end, 0...]
            let cacheLen = melCache?.shape[1] ?? 0
            let win = melCache == nil ? m : MLX.concatenated([melCache!, m], axis: 1)
            let winLen = win.shape[1]
            let lengths = MLXArray([Int32(winLen)]).asType(.int32)
            let sub = encoder.preEncode(win, lengths: lengths).0  // (1, k, d)

            let isFinal = end >= total
            let base = (consumed - cacheLen) / sf
            let lo = emitted - base
            let hi = isFinal ? sub.shape[1] : (end / sf - base)
            consumed = end
            melCache = win[0..., max(0, winLen - nemoPreEncodeMelCache)..<winLen, 0...]

            if hi <= lo {
                emitted = base + max(lo, hi)
                continue
            }
            emitted = base + hi
            var h = sub[0..., lo..<hi, 0...]
            for li in encoder.layers.indices {
                let r = nemoStreamBlock(
                    encoder.layers[li], h,
                    attnCache: attnCache[li], convCache: convCache[li],
                    leftCache: leftCache, convLeft: convLeft
                )
                h = r.0
                attnCache[li] = r.1
                convCache[li] = r.2
            }
            onChunk(applyPrompt(h, language: language))
        }
    }
}
