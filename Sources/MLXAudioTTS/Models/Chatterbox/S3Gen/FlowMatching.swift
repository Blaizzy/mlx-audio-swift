// Ported from Python mlx-audio chatterbox_turbo s3gen/decoder.py + flow_matching.py + flow.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Sinusoidal Position Embedding (free function, matching Python)

/// Sinusoidal position embeddings for timestep encoding.
/// Python: `sinusoidal_pos_emb(timesteps, dim, scale=1000)`
private func sinusoidalPosEmb(_ timesteps: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
    var t = timesteps
    if t.ndim < 1 { t = t.expandedDimensions(axis: 0) }
    let halfDim = dim / 2
    let emb = exp(
        MLXArray(0 ..< halfDim).asType(.float32)
            * MLXArray(Float(-log(10000.0) / Float(halfDim - 1)))
    )
    let out = MLXArray(scale) * t.expandedDimensions(axis: 1) * emb.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.sin(out), MLX.cos(out)], axis: -1)
}

// MARK: - TimestepEmbedding (2-layer MLP with SiLU)

/// MLP for timestep embedding.
/// Python: `TimestepEmbedding(in_channels, time_embed_dim)` with SiLU activation.
/// Weight keys: `linear_1.{weight,bias}`, `linear_2.{weight,bias}`
class S3GenTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
    }

    func callAsFunction(_ sample: MLXArray) -> MLXArray {
        var out = linear1(sample)
        out = silu(out)
        return linear2(out)
    }
}

// MARK: - Conv1d Wrappers (matching Python Conv1dPT / ConvTranspose1dPT nesting)

/// Wrapper matching Python's `Conv1dPT` which has a `conv` child attribute.
/// This creates the extra `.conv` in weight keys (e.g., `conv.conv.weight`).
/// Handles B,C,T → B,T,C transpose for MLX Conv1d.
class S3GenConv1dPT: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int,
         stride: Int = 1, padding: Int = 0, dilation: Int = 1) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, stride: stride, padding: padding, dilation: dilation)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T) → transpose to (B, T, C) for MLX Conv1d → back to (B, C, T)
        var out = x.transposed(0, 2, 1)
        out = conv(out)
        return out.transposed(0, 2, 1)
    }
}

/// Wrapper matching Python's `ConvTranspose1dPT`.
class S3GenConvTranspose1dPT: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int,
         stride: Int = 1, padding: Int = 0) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, stride: stride, padding: padding)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        out = conv(out)
        return out.transposed(0, 2, 1)
    }
}

// MARK: - Causal Conv1d

/// Causal 1D convolution (left-padding only).
/// Python: `CausalConv1d` with `conv` child (Conv1dPT).
/// Weight keys: `conv.conv.{weight,bias}`
class S3GenCausalConv1d: Module {
    @ModuleInfo(key: "conv") var conv: S3GenConv1dPT
    let causalPadding: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1) {
        self.causalPadding = kernelSize - 1
        self._conv.wrappedValue = S3GenConv1dPT(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: 1, padding: 0, dilation: dilation)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T) — apply left padding in time dimension
        // Conv1dPT handles the B,C,T internally, but we need to pad before calling it
        // Pad on the time axis (axis=2 in B,C,T)
        let padded = MLX.padded(x, widths: [.init(0), .init(0), .init((causalPadding, 0))])
        return conv(padded)
    }
}

// MARK: - Block1D / CausalBlock1D

/// Causal 1D block using CausalConv1d + LayerNorm + Mish.
/// Python: `CausalBlock1D` with `block` list child: `block.0` = CausalConv1d, `block.1` = LayerNorm.
/// Weight keys: `block.0.conv.conv.{weight,bias}`, `block.1.{weight,bias}`
class S3GenCausalBlock1D: Module {
    /// Python stores `[CausalConv1d, LayerNorm]` as a ModuleList named `block`.
    /// Weight keys: `block.0.conv.conv.{weight,bias}`, `block.1.{weight,bias}`
    @ModuleInfo(key: "block") var block: [Module]

    let dimOut: Int

    init(dim: Int, dimOut: Int) {
        self.dimOut = dimOut
        self._block.wrappedValue = [
            S3GenCausalConv1d(inChannels: dim, outChannels: dimOut, kernelSize: 3),
            LayerNorm(dimensions: dimOut),
        ]
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let causalConv = block[0] as! S3GenCausalConv1d
        let norm = block[1] as! LayerNorm
        var out = causalConv(x * mask)
        // LayerNorm needs (B, T, C)
        out = out.transposed(0, 2, 1)
        out = norm(out)
        out = out.transposed(0, 2, 1)
        out = MLXNN.mish(out)
        return out * mask
    }
}

// MARK: - ResnetBlock1D

/// ResNet block with time embedding injection.
/// Python: `ResnetBlock1D` with `block1`, `block2` (CausalBlock1D), `mlp` list, `res_conv` (Conv1dPT).
/// Weight keys: `block1.block.{0,1}...`, `block2.block.{0,1}...`, `mlp.0.{weight,bias}`, `res_conv.conv.{weight,bias}`
class S3GenResnetBlock1D: Module {
    @ModuleInfo(key: "block1") var block1: S3GenCausalBlock1D
    @ModuleInfo(key: "block2") var block2: S3GenCausalBlock1D
    /// Python: `self.mlp = [nn.Linear(...)]` — a list with one element.
    /// Weight keys: `mlp.0.{weight,bias}`
    @ModuleInfo(key: "mlp") var mlp: [Linear]
    @ModuleInfo(key: "res_conv") var resConv: S3GenConv1dPT

    init(dim: Int, dimOut: Int, timeEmbDim: Int) {
        self._block1.wrappedValue = S3GenCausalBlock1D(dim: dim, dimOut: dimOut)
        self._block2.wrappedValue = S3GenCausalBlock1D(dim: dimOut, dimOut: dimOut)
        self._mlp.wrappedValue = [Linear(timeEmbDim, dimOut)]
        self._resConv.wrappedValue = S3GenConv1dPT(
            inputChannels: dim, outputChannels: dimOut, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = block1(x, mask: mask)
        // Apply Mish then linear to time embedding, broadcast across time
        h = h + mlp[0](MLXNN.mish(timeEmb)).expandedDimensions(axis: -1)
        h = block2(h, mask: mask)
        // Residual connection through 1x1 conv
        let xRes = resConv(x * mask)
        return h + xRes
    }
}

// MARK: - Downsample1D / Upsample1D

/// Downsample 1D with stride-2 convolution.
/// Python: `Downsample1D` with `conv` (Conv1dPT).
/// Weight keys: `conv.conv.{weight,bias}`
class S3GenDownsample1D: Module {
    @ModuleInfo(key: "conv") var conv: S3GenConv1dPT

    init(dim: Int) {
        self._conv.wrappedValue = S3GenConv1dPT(
            inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

/// Upsample 1D with transposed convolution.
/// Python: `Upsample1D` with `conv` (ConvTranspose1dPT).
/// Weight keys: `conv.conv.{weight,bias}`
class S3GenDecoderUpsample1D: Module {
    @ModuleInfo(key: "conv") var conv: S3GenConvTranspose1dPT

    init(dim: Int) {
        self._conv.wrappedValue = S3GenConvTranspose1dPT(
            inputChannels: dim, outputChannels: dim, kernelSize: 4, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

// MARK: - Self-Attention (Bidirectional)

/// Self-attention for transformer blocks. Bidirectional (no causal masking).
/// Python: `SelfAttention1D` with `to_q`, `to_k`, `to_v` (no bias), `to_out` list with `to_out.0` (Linear with bias).
/// Weight keys: `to_q.weight`, `to_k.weight`, `to_v.weight`, `to_out.0.{weight,bias}`
class S3GenSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    /// Python: `self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim)])`
    /// Weight keys: `to_out.0.{weight,bias}`
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    init(dim: Int, numHeads: Int, headDim: Int) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        let innerDim = numHeads * headDim

        self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toOut.wrappedValue = [Linear(innerDim, dim)]
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)

        let q = toQ(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = toK(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)

        // Apply padding mask if provided: (B, T) → (B, 1, 1, T) for broadcasting
        var attnMask: MLXArray? = nil
        if let mask = mask {
            // The mask is (B, T) with 1s for valid positions and 0s for padding
            // Convert to attention mask: 0 → -inf (large negative), 1 → 0
            let expanded = mask.expandedDimensions(axes: [1, 2]) // (B, 1, 1, T)
            attnMask = MLX.where(expanded .> 0, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: attnMask)

        let combined = out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim)
        return toOut[0](combined)
    }
}

// MARK: - GELU Activation Module

/// GELU activation with linear projection.
/// Python: `GELU` class with `proj` (Linear) → gelu activation.
/// Weight keys: `proj.{weight,bias}`
class S3GenGELU: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(dimIn: Int, dimOut: Int) {
        self._proj.wrappedValue = Linear(dimIn, dimOut)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return gelu(proj(x))
    }
}

// MARK: - FeedForward

/// Feed-forward network with GELU activation.
///
/// Python: `FeedForward` with `net = nn.ModuleList([GEGLU(...), nn.Linear(...)])`.
/// Original weight keys: `net.0.proj.{weight,bias}`, `net.1.{weight,bias}`
///
/// We avoid using `[Module]` arrays because MLX Swift's quantization update can't handle
/// heterogeneous arrays where some elements need recursive updates (S3GenGELU containing
/// a Linear→QuantizedLinear) and others are direct module replacements (Linear→QuantizedLinear).
/// Instead, we use non-numeric keys (`gelu_gate` and `out_proj`) and remap the weight keys
/// from `net.0.*` → `gelu_gate.*` and `net.1.*` → `out_proj.*` during sanitization.
class S3GenFeedForward: Module {
    @ModuleInfo(key: "gelu_gate") var geluGate: S3GenGELU
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int, mult: Int = 4) {
        let innerDim = dim * mult
        self._geluGate.wrappedValue = S3GenGELU(dimIn: dim, dimOut: innerDim)
        self._outProj.wrappedValue = Linear(innerDim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return outProj(geluGate(x))
    }
}

// MARK: - Transformer Block

/// Transformer block with self-attention and feed-forward.
/// Python: `TransformerBlock` with `norm1`, `attn1`, `norm3`, `ff` (pre-norm residual).
/// Note: Uses `norm3` (not `norm2`) to match PyTorch naming convention.
/// Weight keys: `norm1.{weight,bias}`, `attn1.{to_q,to_k,to_v,to_out}...`, `norm3.{weight,bias}`, `ff.net...`
class S3GenTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: S3GenSelfAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: S3GenFeedForward

    init(dim: Int, numHeads: Int, headDim: Int) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._attn1.wrappedValue = S3GenSelfAttention(dim: dim, numHeads: numHeads, headDim: headDim)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim)
        self._ff.wrappedValue = S3GenFeedForward(dim: dim, mult: 4)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Pre-norm self-attention with residual
        var out = x + attn1(norm1(x), mask: mask)
        // Pre-norm feed-forward with residual
        out = out + ff(norm3(out))
        return out
    }
}

// MARK: - U-Net Blocks (Down, Mid, Up)

/// Down block: ResNet + transformer blocks + downsample.
/// Weight keys: `resnet.{...}`, `transformer_blocks.{0-3}.{...}`, `downsample.{...}`
class S3GenDownBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: S3GenResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [S3GenTransformerBlock]
    /// When `isLast`: CausalConv1d (no downsample). Otherwise: Downsample1D (stride-2).
    @ModuleInfo(key: "downsample") var downsample: Module
    let isLast: Bool

    init(inputChannel: Int, outputChannel: Int, timeEmbDim: Int,
         nBlocks: Int, numHeads: Int, headDim: Int, isLast: Bool) {
        self.isLast = isLast
        self._resnet.wrappedValue = S3GenResnetBlock1D(
            dim: inputChannel, dimOut: outputChannel, timeEmbDim: timeEmbDim)
        var blocks: [S3GenTransformerBlock] = []
        for _ in 0 ..< nBlocks {
            blocks.append(S3GenTransformerBlock(
                dim: outputChannel, numHeads: numHeads, headDim: headDim))
        }
        self._transformerBlocks.wrappedValue = blocks
        if isLast {
            self._downsample.wrappedValue = S3GenCausalConv1d(
                inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
        } else {
            self._downsample.wrappedValue = S3GenDownsample1D(dim: outputChannel)
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> (MLXArray, MLXArray) {
        var h = resnet(x, mask: mask, timeEmb: timeEmb)
        // Transpose for transformer: (B, C, T) → (B, T, C)
        h = h.transposed(0, 2, 1)
        let maskT = mask[0..., 0, 0...]  // (B, 1, T) → (B, T)
        for block in transformerBlocks {
            h = block(h, mask: maskT)
        }
        h = h.transposed(0, 2, 1)  // Back to (B, C, T)
        let skipConnection = h
        if isLast {
            h = (downsample as! S3GenCausalConv1d)(h * mask)
        } else {
            h = (downsample as! S3GenDownsample1D)(h * mask)
        }
        return (h, skipConnection)
    }
}

/// Mid block: ResNet + transformer blocks (no down/upsample).
/// Weight keys: `resnet.{...}`, `transformer_blocks.{0-3}.{...}`
class S3GenMidBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: S3GenResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [S3GenTransformerBlock]

    init(channels: Int, timeEmbDim: Int, nBlocks: Int, numHeads: Int, headDim: Int) {
        self._resnet.wrappedValue = S3GenResnetBlock1D(
            dim: channels, dimOut: channels, timeEmbDim: timeEmbDim)
        var blocks: [S3GenTransformerBlock] = []
        for _ in 0 ..< nBlocks {
            blocks.append(S3GenTransformerBlock(
                dim: channels, numHeads: numHeads, headDim: headDim))
        }
        self._transformerBlocks.wrappedValue = blocks
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = resnet(x, mask: mask, timeEmb: timeEmb)
        h = h.transposed(0, 2, 1)
        let maskT = mask[0..., 0, 0...]
        for block in transformerBlocks {
            h = block(h, mask: maskT)
        }
        return h.transposed(0, 2, 1)
    }
}

/// Up block: ResNet (with skip connection concat) + transformer blocks + upsample.
/// Weight keys: `resnet.{...}`, `transformer_blocks.{0-3}.{...}`, `upsample.{...}`
class S3GenUpBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: S3GenResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [S3GenTransformerBlock]
    /// When `isLast`: CausalConv1d (no upsample). Otherwise: Upsample1D (transposed conv).
    @ModuleInfo(key: "upsample") var upsample: Module
    let isLast: Bool

    init(inputChannel: Int, outputChannel: Int, timeEmbDim: Int,
         nBlocks: Int, numHeads: Int, headDim: Int, isLast: Bool) {
        self.isLast = isLast
        // Input channel is doubled because of skip connection concatenation
        self._resnet.wrappedValue = S3GenResnetBlock1D(
            dim: inputChannel, dimOut: outputChannel, timeEmbDim: timeEmbDim)
        var blocks: [S3GenTransformerBlock] = []
        for _ in 0 ..< nBlocks {
            blocks.append(S3GenTransformerBlock(
                dim: outputChannel, numHeads: numHeads, headDim: headDim))
        }
        self._transformerBlocks.wrappedValue = blocks
        if isLast {
            self._upsample.wrappedValue = S3GenCausalConv1d(
                inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
        } else {
            self._upsample.wrappedValue = S3GenDecoderUpsample1D(dim: outputChannel)
        }
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        // Trim x to match skip connection length, then concatenate
        let trimmed = x[0..., 0..., ..<skip.dim(2)]
        var h = MLX.concatenated([trimmed, skip], axis: 1)
        h = resnet(h, mask: mask, timeEmb: timeEmb)
        h = h.transposed(0, 2, 1)
        let maskT = mask[0..., 0, 0...]
        for block in transformerBlocks {
            h = block(h, mask: maskT)
        }
        h = h.transposed(0, 2, 1)
        if isLast {
            h = (upsample as! S3GenCausalConv1d)(h * mask)
        } else {
            h = (upsample as! S3GenDecoderUpsample1D)(h * mask)
        }
        return h
    }
}

// MARK: - Conditional Decoder (U-Net)

/// Conditional decoder with U-Net architecture for flow matching.
/// Full U-Net: time embedding → concat inputs → down → mid → up → final.
///
/// Python: `ConditionalDecoder` with `time_mlp`, `time_embed_mixer` (meanflow),
/// `down_blocks`, `mid_blocks`, `up_blocks`, `final_block`, `final_proj`.
///
/// Weight keys match Python naming exactly.
class S3GenConditionalDecoder: Module {
    let inChannels: Int
    let outChannels: Int
    let meanflow: Bool

    @ModuleInfo(key: "time_mlp") var timeMLP: S3GenTimestepEmbedding
    @ModuleInfo(key: "time_embed_mixer") var timeEmbedMixer: Linear
    @ModuleInfo(key: "down_blocks") var downBlocks: [S3GenDownBlock]
    @ModuleInfo(key: "mid_blocks") var midBlocks: [S3GenMidBlock]
    @ModuleInfo(key: "up_blocks") var upBlocks: [S3GenUpBlock]
    @ModuleInfo(key: "final_block") var finalBlock: S3GenCausalBlock1D
    @ModuleInfo(key: "final_proj") var finalProj: S3GenConv1dPT

    init(
        inChannels: Int = 320, outChannels: Int = 80,
        channels: [Int] = [256],
        nBlocks: Int = 4, numMidBlocks: Int = 12,
        numHeads: Int = 8, attentionHeadDim: Int = 64,
        meanflow: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.meanflow = meanflow

        let timeEmbedDim = channels[0] * 4  // 256 * 4 = 1024

        // Time embedding MLP: sinusoidal → 2-layer MLP
        self._timeMLP.wrappedValue = S3GenTimestepEmbedding(
            inChannels: inChannels, timeEmbedDim: timeEmbedDim)

        // Meanflow: concatenate t_emb + r_emb → mix down to timeEmbedDim
        self._timeEmbedMixer.wrappedValue = Linear(timeEmbedDim * 2, timeEmbedDim, bias: false)

        // Down blocks
        var downBlockList: [S3GenDownBlock] = []
        var outputChannel = inChannels
        for (i, ch) in channels.enumerated() {
            let inputChannel = outputChannel
            outputChannel = ch
            let isLast = i == channels.count - 1
            downBlockList.append(S3GenDownBlock(
                inputChannel: inputChannel, outputChannel: outputChannel,
                timeEmbDim: timeEmbedDim, nBlocks: nBlocks,
                numHeads: numHeads, headDim: attentionHeadDim, isLast: isLast))
        }
        self._downBlocks.wrappedValue = downBlockList

        // Mid blocks
        let midCh = channels.last ?? 256
        var midBlockList: [S3GenMidBlock] = []
        for _ in 0 ..< numMidBlocks {
            midBlockList.append(S3GenMidBlock(
                channels: midCh, timeEmbDim: timeEmbedDim,
                nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim))
        }
        self._midBlocks.wrappedValue = midBlockList

        // Up blocks
        let channelsReversed = Array(channels.reversed()) + [channels[0]]
        let channelsUpCount = channelsReversed.count - 1
        var upBlockList: [S3GenUpBlock] = []
        for i in 0 ..< channelsUpCount {
            let skipCh = channelsReversed[i]  // From skip connection
            let outCh = channelsReversed[i + 1]
            let isLast = i == channelsUpCount - 1
            upBlockList.append(S3GenUpBlock(
                inputChannel: skipCh * 2, outputChannel: outCh,
                timeEmbDim: timeEmbedDim, nBlocks: nBlocks,
                numHeads: numHeads, headDim: attentionHeadDim, isLast: isLast))
        }
        self._upBlocks.wrappedValue = upBlockList

        // Final layers
        let finalCh = channels[0]
        self._finalBlock.wrappedValue = S3GenCausalBlock1D(dim: finalCh, dimOut: finalCh)
        self._finalProj.wrappedValue = S3GenConv1dPT(
            inputChannels: finalCh, outputChannels: outChannels, kernelSize: 1)
    }

    func callAsFunction(
        x: MLXArray, mask: MLXArray, mu: MLXArray, t: MLXArray,
        spks: MLXArray? = nil, cond: MLXArray? = nil,
        r: MLXArray? = nil
    ) -> MLXArray {
        // 1. Time embedding
        var tEmb = timeMLP(sinusoidalPosEmb(t, dim: inChannels))

        // Meanflow: mix t and r embeddings
        if meanflow, let r = r {
            let rEmb = timeMLP(sinusoidalPosEmb(r, dim: inChannels))
            let concatEmb = MLX.concatenated([tEmb, rEmb], axis: -1)
            tEmb = timeEmbedMixer(concatEmb)
        }

        // 2. Concatenate inputs: [x, mu, spks_expanded, cond] along channel axis
        var inputs: [MLXArray] = [x, mu]
        if let spks = spks {
            let spksExpanded = MLX.broadcast(
                spks.expandedDimensions(axis: -1),
                to: [spks.dim(0), spks.dim(1), x.dim(2)])
            inputs.append(spksExpanded)
        }
        if let cond = cond {
            inputs.append(cond)
        }
        var h = MLX.concatenated(inputs, axis: 1)

        // 3. Down path
        var hiddens: [MLXArray] = []
        var masks: [MLXArray] = [mask]
        for downBlock in downBlocks {
            let maskDown = masks.last!
            let (downOut, skip) = downBlock(h, mask: maskDown, timeEmb: tEmb)
            hiddens.append(skip)
            h = downOut
            // Halve the mask for next level (stride-2 selection along time axis)
            let strideIndices = MLXArray(Array(Swift.stride(from: Int32(0), to: Int32(maskDown.dim(2)), by: 2)))
            masks.append(maskDown.take(strideIndices, axis: 2))
        }
        masks.removeLast()  // Remove the last (smallest) mask

        // 4. Mid path
        let maskMid = masks.last!
        for midBlock in midBlocks {
            h = midBlock(h, mask: maskMid, timeEmb: tEmb)
        }

        // 5. Up path
        for upBlock in upBlocks {
            let maskUp = masks.removeLast()
            let skip = hiddens.removeLast()
            h = upBlock(h, skip: skip, mask: maskUp, timeEmb: tEmb)
        }

        // 6. Final block + projection
        let finalMask = mask
        h = finalBlock(h, mask: finalMask)

        let output = finalProj(h * finalMask)
        return output * finalMask
    }
}

// MARK: - Causal Conditional CFM (Flow Matching)

/// Causal Conditional Flow Matching with Euler ODE solver.
///
/// Supports both Regular and Turbo (meanflow) models with different solvers:
/// - **Turbo (meanflow=true)**: `basicEuler` — no CFG, passes `r` to estimator, linear time schedule
/// - **Regular (meanflow=false)**: `solveEulerCFG` — classifier-free guidance, cosine time schedule
///
/// Matches Python's `CausalConditionalCFM(ConditionalCFM)` exactly.
class CausalConditionalCFM: Module {
    static let melChannels = 80

    let sigmaMin: Float
    let cfgRate: Float
    let meanflow: Bool
    let tScheduler: String
    let nFeats: Int

    /// Pre-generated deterministic noise for Regular (non-meanflow) models.
    /// Python: `mx.random.seed(0); self.rand_noise = mx.random.normal((1, 80, 50*300))`
    /// Using fixed seed ensures reproducible inference. Turbo uses fresh random noise instead.
    let randNoise: MLXArray?

    @ModuleInfo(key: "estimator") var estimator: S3GenConditionalDecoder

    init(
        inChannels: Int = 320, outChannels: Int = 80,
        channels: [Int] = [256],
        nBlocks: Int = 4, numMidBlocks: Int = 12,
        numHeads: Int = 8, attentionHeadDim: Int = 64,
        sigmaMin: Float = 1e-6, cfgRate: Float = 0.7,
        tScheduler: String = "cosine",
        meanflow: Bool = true
    ) {
        self.sigmaMin = sigmaMin
        self.cfgRate = cfgRate
        self.tScheduler = tScheduler
        self.meanflow = meanflow
        self.nFeats = outChannels

        // Regular (non-meanflow) model uses deterministic noise from a fixed seed.
        // This matches Python's CausalConditionalCFM.__init__ which does:
        //   mx.random.seed(0)
        //   self.rand_noise = mx.random.normal((1, MEL_CHANNELS, 50 * 300))
        //
        // IMPORTANT: We must use seed-based (global state) generation, NOT key-based.
        // mx.random.seed(0) followed by mx.random.normal() (no explicit key) internally
        // splits the global key before generating, producing different values than
        // mx.random.normal(key=mx.random.key(0)). Using the wrong method gives
        // completely different starting noise for the ODE solver.
        if !meanflow {
            MLXRandom.seed(0)
            self.randNoise = MLXRandom.normal([1, Self.melChannels, 50 * 300])
            eval(self.randNoise!)
        } else {
            self.randNoise = nil
        }

        self._estimator.wrappedValue = S3GenConditionalDecoder(
            inChannels: inChannels, outChannels: outChannels,
            channels: channels, nBlocks: nBlocks, numMidBlocks: numMidBlocks,
            numHeads: numHeads, attentionHeadDim: attentionHeadDim,
            meanflow: meanflow)
    }

    /// Basic Euler solver WITHOUT classifier-free guidance.
    ///
    /// Used for meanflow/Turbo distilled models. Passes `r` (next timestep) to the
    /// estimator so the time embedding mixer can combine t and r embeddings.
    /// Python: `ConditionalCFM._basic_euler`
    private func basicEuler(
        z: MLXArray, tSpan: MLXArray, mu: MLXArray, mask: MLXArray,
        spks: MLXArray?, cond: MLXArray?
    ) -> MLXArray {
        var x = z
        let nSteps = tSpan.dim(0) - 1

        for i in 0 ..< nSteps {
            let t = tSpan[i ..< (i + 1)]
            let r = tSpan[(i + 1) ..< (i + 2)]

            // Predict velocity — passes r for meanflow time embedding mixing
            let dxdt = estimator(
                x: x, mask: mask, mu: mu, t: t,
                spks: spks, cond: cond, r: r)

            // Euler step
            let dt = r - t
            x = x + dt * dxdt
        }
        return x
    }

    /// Euler solver WITH classifier-free guidance.
    ///
    /// Used for Regular (non-meanflow) models. Duplicates the batch for conditional
    /// and unconditional predictions, then combines with CFG formula.
    ///
    /// Matches Python's `ConditionalCFM.solve_euler` exactly: initializes default
    /// zero-filled spks/cond tensors, uses variable dt between steps, and applies
    /// CFG formula `(1 + cfg_rate) * cond - cfg_rate * uncond`.
    private func solveEulerCFG(
        z: MLXArray, tSpan: MLXArray, mu: MLXArray, mask: MLXArray,
        spks: MLXArray?, cond: MLXArray?
    ) -> MLXArray {
        var x = z
        let batchSize = x.dim(0)
        let T_len = x.dim(2)

        // Python: initializes default zero-filled tensors even when spks/cond are None.
        // `spks_in = mx.zeros((2, self.spk_emb_dim))`
        // `cond_in = mx.zeros((2, self.n_feats, T_len))`
        var spksIn = MLXArray.zeros([2, nFeats])
        var condIn = MLXArray.zeros([2, nFeats, T_len])

        // Python: `t = mx.expand_dims(t_span[0], 0)` → shape (1,)
        // t_span[0] is scalar, expand_dims(0) gives 1D array of shape (1,)
        var t = tSpan[0].expandedDimensions(axis: 0) // (1,)
        var dt = tSpan[1] - tSpan[0]

        let nSteps = tSpan.dim(0) - 1
        for step in 1 ... nSteps {
            // Duplicate for classifier-free guidance: [conditional, unconditional]
            let xIn = MLX.concatenated([x, x], axis: 0)
            let maskIn = MLX.concatenated([mask, mask], axis: 0)
            let muIn = MLX.concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
            let tIn = MLX.concatenated([t, t], axis: 0)

            if let spks = spks {
                spksIn = MLX.concatenated([spks, MLXArray.zeros(like: spks)], axis: 0)
            }
            if let cond = cond {
                condIn = MLX.concatenated([cond, MLXArray.zeros(like: cond)], axis: 0)
            }

            // Predict velocity for both conditional and unconditional
            // Regular model: no `r` parameter (r is nil)
            let dxdt = estimator(
                x: xIn, mask: maskIn, mu: muIn, t: tIn,
                spks: spksIn, cond: condIn, r: nil)

            // Split conditional and unconditional predictions
            let dxdtCond = dxdt[0 ..< batchSize]
            let dxdtUncond = dxdt[batchSize...]

            // CFG formula: (1 + cfg_rate) * cond - cfg_rate * uncond
            let pred = (1 + cfgRate) * dxdtCond - cfgRate * dxdtUncond

            // Euler step
            x = x + dt * pred
            t = t + dt

            // Update dt for next step (Python: `dt = t_span[step+1] - t`)
            if step < nSteps {
                dt = tSpan[step + 1] - t
            }
        }
        return x
    }

    func callAsFunction(
        mu: MLXArray, mask: MLXArray, nTimesteps: Int,
        spks: MLXArray? = nil, cond: MLXArray? = nil,
        noisedMels: MLXArray? = nil
    ) -> MLXArray {
        let z: MLXArray

        if meanflow {
            // Turbo: fresh random noise + optional noised_mels splice
            var noise = MLXRandom.normal(mu.shape)
            if let noisedMels = noisedMels {
                let promptMelLen = mu.dim(2) - noisedMels.dim(2)
                if promptMelLen > 0 {
                    let promptPart = noise[0..., 0..., ..<promptMelLen]
                    noise = MLX.concatenated([promptPart, noisedMels], axis: 2)
                }
            }
            z = noise
        } else {
            // Regular: deterministic noise from pre-generated buffer (seed=0).
            // Python: `z = self.rand_noise[:, :, :T] * temperature`
            // Temperature is always 1.0 for CausalConditionalCFM.
            let T = mu.dim(2)
            z = randNoise![0..., 0..., ..<T]
        }

        // Time schedule: cosine for Regular, linear for Turbo (meanflow)
        // Python: `if (not meanflow) and (self.t_scheduler == "cosine"): t_span = 1 - cos(...)`
        let tSpan: MLXArray
        let linear = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
        if !meanflow && tScheduler == "cosine" {
            tSpan = 1.0 - MLX.cos(linear * Float32(Float.pi / 2))
        } else {
            tSpan = linear
        }

        // Meanflow (Turbo): basic Euler without CFG, passes r to estimator
        // Regular: CFG Euler solver with cosine schedule
        if meanflow {
            return basicEuler(z: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
        }

        return solveEulerCFG(z: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    }
}

// MARK: - CausalMaskedDiffWithXvec (S3Gen Flow Container)

/// Flow matching wrapper that combines Conformer encoder with flow matching decoder.
///
/// Pipeline: tokens → inputEmbedding → Conformer encoder (2x upsample) → encoderProj
/// → flow matching decoder (Euler ODE) → mel spectrogram → HiFi-GAN vocoder → waveform.
class CausalMaskedDiffWithXvec: Module {
    let outputSize: Int
    let vocabSize: Int
    let tokenMelRatio: Int
    let preLookaheadLen: Int
    let meanflow: Bool

    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "decoder") var decoder: CausalConditionalCFM
    @ModuleInfo(key: "mel2wav") var vocoderModule: HiFTGenerator
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: CAMPPlus

    init(
        inputSize: Int = 512, outputSize: Int = 80,
        spkEmbedDim: Int = 192, vocabSize: Int = 6561,
        decoderInChannels: Int = 320,
        encoderConfig: [String: Any]? = nil,
        decoderConfig: [String: Any]? = nil,
        meanflow: Bool = true
    ) {
        self.outputSize = outputSize
        self.vocabSize = vocabSize
        self.tokenMelRatio = 2
        self.preLookaheadLen = 3
        self.meanflow = meanflow

        self._inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
        self._spkEmbedAffineLayer.wrappedValue = Linear(spkEmbedDim, outputSize)

        // Conformer encoder
        self._encoder.wrappedValue = UpsampleConformerEncoder(
            inputSize: inputSize, outputSize: inputSize)

        // Encoder projection (encoder output dim → mel dim)
        self._encoderProj.wrappedValue = Linear(inputSize, outputSize)

        // Flow matching decoder — inChannels = 320 = 80*4 (x + mu + spks + cond)
        self._decoder.wrappedValue = CausalConditionalCFM(
            inChannels: decoderInChannels, outChannels: outputSize,
            meanflow: meanflow)

        // HiFi-GAN vocoder (weight key: mel2wav)
        self._vocoderModule.wrappedValue = HiFTGenerator()

        // CAMPPlus speaker encoder (x-vector extraction)
        self._speakerEncoder.wrappedValue = CAMPPlus()
    }

    /// Run vocoder (HiFi-GAN) on mel spectrogram to produce waveform.
    func vocoder(_ mel: MLXArray) -> (MLXArray, MLXArray) {
        return vocoderModule(mel)
    }

    /// Flow matching inference: speech tokens → mel spectrogram.
    ///
    /// - Parameters:
    ///   - token: Generated speech token IDs, shape `(B, T_gen)`, int32
    ///   - tokenLen: Length of generated tokens, shape `(B,)`
    ///   - promptToken: Prompt/reference speech token IDs, shape `(B, T_prompt)`, int32
    ///   - promptTokenLen: Length of prompt tokens, shape `(B,)`
    ///   - promptFeat: Prompt mel spectrogram features, shape `(B, T_mel, 80)`
    ///   - embedding: Speaker x-vector, shape `(B, 192)`
    ///   - nTimesteps: Number of Euler ODE steps for flow matching
    ///   - streaming: Whether to use streaming/causal mode
    /// - Returns: Generated mel spectrogram, shape `(B, 80, T_gen_mel)`
    func inference(
        token: MLXArray, tokenLen: MLXArray,
        promptToken: MLXArray, promptTokenLen: MLXArray,
        promptFeat: MLXArray,
        embedding: MLXArray,
        nTimesteps: Int = 10,
        streaming: Bool = false
    ) -> MLXArray {
        // 1. L2-normalize speaker embedding before projection
        let norm = MLX.sqrt((embedding * embedding).sum(axis: 1, keepDims: true))
        let normalizedEmb = embedding / (norm + 1e-8)
        let spkEmb = spkEmbedAffineLayer(normalizedEmb) // (B, 80)

        // 2. Concatenate prompt + generated tokens
        let combinedToken = MLX.concatenated([promptToken, token], axis: 1) // (B, T_prompt + T_gen)
        let combinedLen = promptTokenLen + tokenLen

        // Create embedding mask: (B, T, 1) — masks padding positions before embedding.
        // Python: `mask = (seq_range < seq_length).unsqueeze(-1).astype(dtype)`
        let maxLen = combinedToken.dim(1)
        let seqRange = MLXArray(0 ..< Int32(maxLen)).expandedDimensions(axis: 0)  // (1, T)
        let seqLenExpanded = combinedLen.expandedDimensions(axis: -1)  // (B, 1)
        let embMask = (seqRange .< seqLenExpanded).asType(.float32).expandedDimensions(axis: -1) // (B, T, 1)

        // Clip token IDs to valid range, embed, and apply mask
        let clipped = MLX.clip(combinedToken, min: 0, max: vocabSize - 1)
        let embedded = inputEmbedding(clipped) * embMask // (B, T_combined, 512) — masked

        // 3. Conformer encoder (includes 2x upsample internally)
        let (encoderOut, _) = encoder(xs: embedded, xsLens: combinedLen, streaming: streaming)
        // encoderOut: (B, T_up, 512) where T_up ≈ T_combined * 2

        // 4. Compute mel lengths from prompt feat
        let promptMelLen = promptFeat.dim(1)

        // 5. Project encoder output to mel dimension
        let h = encoderProj(encoderOut) // (B, T_up, 80)
        let totalMelLen = h.dim(1)

        // 6. Build conditioning signal from prompt mel features
        // Python: `conds = mx.zeros([1, mel_len1 + mel_len2, D]); conds[:, :mel_len1] = prompt_feat`
        var condsSlices: [MLXArray] = []
        if promptMelLen > 0 {
            let copyLen = min(promptMelLen, totalMelLen)
            condsSlices.append(promptFeat[0..., ..<copyLen, 0...])
            if copyLen < totalMelLen {
                condsSlices.append(MLXArray.zeros([1, totalMelLen - copyLen, outputSize]))
            }
        } else {
            condsSlices.append(MLXArray.zeros([1, totalMelLen, outputSize]))
        }
        let conds = MLX.concatenated(condsSlices, axis: 1).transposed(0, 2, 1) // (B, 80, T_up)

        // 7. Create decoder mask — Python uses all-ones: `mask = mx.ones([1, 1, total_len])`
        // The flow matching decoder doesn't need padding masks since batch=1 and
        // all positions are valid.
        let decoderMask = MLXArray.ones([1, 1, totalMelLen]).asType(.float32)

        // 8. Flow matching decode
        let mu = h.transposed(0, 2, 1) // (B, 80, T_up)

        // For meanflow, generate noised mels for the generated portion only
        let noisedMels: MLXArray?
        if meanflow {
            let genMelLen = token.dim(1) * tokenMelRatio
            noisedMels = MLXRandom.normal([1, outputSize, genMelLen])
        } else {
            noisedMels = nil
        }

        let mel = decoder(
            mu: mu, mask: decoderMask, nTimesteps: nTimesteps,
            spks: spkEmb, cond: conds, noisedMels: noisedMels)

        // 8. Extract only the generated portion (skip prompt mel region)
        if promptMelLen > 0 && promptMelLen < totalMelLen {
            return mel[0..., 0..., promptMelLen...]
        }
        return mel
    }
}
