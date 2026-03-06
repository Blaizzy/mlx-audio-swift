import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - RoPE Helpers

func echoPrecomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000.0) -> (MLXArray, MLXArray) {
    let halfDim = dim / 2
    var freqs = [Float](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        freqs[i] = 1.0 / pow(theta, Float(2 * i) / Float(dim))
    }
    let freqsArray = MLXArray(freqs)

    var positions = [Float](repeating: 0, count: end)
    for i in 0..<end {
        positions[i] = Float(i)
    }
    let posArray = MLXArray(positions)

    // [end, halfDim]
    let angles = posArray.reshaped([end, 1]) * freqsArray.reshaped([1, halfDim])
    return (MLX.cos(angles), MLX.sin(angles))
}

func echoApplyRotaryEmb(_ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray) -> MLXArray {
    // x: [B, nHeads, seqLen, headDim]
    let headDim = x.dim(-1)
    let halfDim = headDim / 2

    let x1 = x[0..., 0..., 0..., ..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]

    return MLX.concatenated([
        x1 * cosFreqs - x2 * sinFreqs,
        x1 * sinFreqs + x2 * cosFreqs
    ], axis: -1)
}

// MARK: - Timestep Embedding

func echoGetTimestepEmbedding(_ timestep: MLXArray, embedSize: Int) -> MLXArray {
    let halfDim = embedSize / 2
    var freqs = [Float](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        freqs[i] = exp(-log(Float(10000.0)) * Float(i) / Float(halfDim))
    }
    let freqsArray = MLXArray(freqs)

    // timestep: [B] or [B, 1]
    let t = timestep.reshaped([-1, 1])  // [B, 1]
    let args = t * freqsArray.reshaped([1, halfDim])  // [B, halfDim]

    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)  // [B, embedSize]
}

// MARK: - Mask Helpers

func echoBoolToAdditiveMask(_ mask: MLXArray) -> MLXArray {
    // Convert boolean mask (true=attend) to additive mask (0=attend, -1e9=ignore)
    let floatMask = mask.asType(.float32)
    return (1.0 - floatMask) * (-1e9)
}

func echoMakeCausalMask(seqLen: Int) -> MLXArray {
    var data = [Float](repeating: -Float.infinity, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0...i {
            data[i * seqLen + j] = 0.0
        }
    }
    return MLXArray(data, [seqLen, seqLen])
}

// MARK: - RMS Normalization

/// RMS normalization supporting both scalar and per-head dimensions.
class EchoRMSNorm: Module {
    let eps: Float
    let perHead: Bool

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.perHead = false
        self._weight.wrappedValue = MLXArray.ones([dim])
    }

    init(numHeads: Int, headDim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.perHead = true
        self._weight.wrappedValue = MLXArray.ones([numHeads, headDim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = MLX.sqrt(MLX.mean(x * x, axis: -1, keepDims: true) + eps)
        return (x / rms) * weight
    }
}

// MARK: - Low-Rank Adaptive Layer Normalization

/// Adaptive layer normalization with low-rank shift/scale/gate decomposition.
class EchoLowRankAdaLN: Module {
    let modelSize: Int
    let rank: Int

    @ModuleInfo(key: "shift_down") var shiftDown: Linear
    @ModuleInfo(key: "shift_up") var shiftUp: Linear
    @ModuleInfo(key: "shift_residual") var shiftResidual: Linear
    @ModuleInfo(key: "scale_down") var scaleDown: Linear
    @ModuleInfo(key: "scale_up") var scaleUp: Linear
    @ModuleInfo(key: "scale_residual") var scaleResidual: Linear
    @ModuleInfo(key: "gate_down") var gateDown: Linear
    @ModuleInfo(key: "gate_up") var gateUp: Linear
    @ModuleInfo(key: "gate_residual") var gateResidual: Linear
    @ModuleInfo(key: "norm") var norm: EchoRMSNorm

    init(modelSize: Int, condSize: Int, rank: Int) {
        self.modelSize = modelSize
        self.rank = rank

        // Shift path
        self._shiftDown.wrappedValue = Linear(condSize, rank, bias: false)
        self._shiftUp.wrappedValue = Linear(rank, modelSize, bias: false)
        self._shiftResidual.wrappedValue = Linear(condSize, modelSize, bias: false)

        // Scale path
        self._scaleDown.wrappedValue = Linear(condSize, rank, bias: false)
        self._scaleUp.wrappedValue = Linear(rank, modelSize, bias: false)
        self._scaleResidual.wrappedValue = Linear(condSize, modelSize, bias: false)

        // Gate path
        self._gateDown.wrappedValue = Linear(condSize, rank, bias: false)
        self._gateUp.wrappedValue = Linear(rank, modelSize, bias: false)
        self._gateResidual.wrappedValue = Linear(condSize, modelSize, bias: false)

        self._norm.wrappedValue = EchoRMSNorm(dim: modelSize)
    }

    /// Returns (normalized_x, gate).
    func callAsFunction(_ x: MLXArray, condEmbed: MLXArray) -> (MLXArray, MLXArray) {
        // condEmbed: [B, 1, condSize] split into 3 parts
        let condSize = condEmbed.dim(-1) / 3
        let shiftCond = condEmbed[0..., 0..., ..<condSize]
        let scaleCond = condEmbed[0..., 0..., condSize..<(2 * condSize)]
        let gateCond = condEmbed[0..., 0..., (2 * condSize)...]

        let shift = shiftUp(shiftDown(shiftCond)) + shiftResidual(shiftCond)
        let scale = scaleUp(scaleDown(scaleCond)) + scaleResidual(scaleCond)
        let gate = MLX.tanh(gateUp(gateDown(gateCond)) + gateResidual(gateCond))

        let normalized = norm(x) * (scale + 1.0) + shift
        return (normalized, gate)
    }
}

// MARK: - Self Attention (for Text/Speaker Encoders)

class EchoSelfAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm

    init(modelSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = modelSize / numHeads

        self._wq.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._wk.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._wv.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._wo.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._gate.wrappedValue = Linear(modelSize, modelSize, bias: false)
        self._qNorm.wrappedValue = EchoRMSNorm(numHeads: numHeads, headDim: headDim)
        self._kNorm.wrappedValue = EchoRMSNorm(numHeads: numHeads, headDim: headDim)
    }

    func callAsFunction(
        _ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = wq(x).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        // QK normalization
        q = qNorm(q)
        k = kNorm(k)

        // RoPE
        q = echoApplyRotaryEmb(q, cosFreqs: cosFreqs, sinFreqs: sinFreqs)
        k = echoApplyRotaryEmb(k, cosFreqs: cosFreqs, sinFreqs: sinFreqs)

        let scale = sqrt(Float(headDim))
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0 / scale, mask: mask
        )

        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([bsz, seqLen, numHeads * headDim])

        // Gated output
        return outputReshaped * sigmoid(gate(x))
    }
}

// MARK: - Joint Attention (for Main DiT Blocks)

/// Typealias for KV cache: list of (key, value) pairs per layer.
public typealias EchoKVCache = [(MLXArray, MLXArray)]

class EchoJointAttention: Module {
    let numHeads: Int
    let headDim: Int
    let modelSize: Int

    // Self-attention projections
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "gate") var gate: Linear

    // Cross-attention to text
    @ModuleInfo(key: "wk_text") var wkText: Linear
    @ModuleInfo(key: "wv_text") var wvText: Linear

    // Cross-attention to speaker
    @ModuleInfo(key: "wk_speaker") var wkSpeaker: Linear
    @ModuleInfo(key: "wv_speaker") var wvSpeaker: Linear

    // Optional cross-attention to latent prefix (blockwise)
    @ModuleInfo(key: "wk_latent") var wkLatent: Linear?
    @ModuleInfo(key: "wv_latent") var wvLatent: Linear?

    // QK normalization
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm

    init(config: EchoDiTConfig, hasLatentAttention: Bool = false) {
        self.numHeads = config.numHeads
        self.headDim = config.modelSize / config.numHeads
        self.modelSize = config.modelSize

        self._wq.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._wk.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._wv.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._wo.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)
        self._gate.wrappedValue = Linear(config.modelSize, config.modelSize, bias: false)

        self._wkText.wrappedValue = Linear(config.textModelSize, config.modelSize, bias: false)
        self._wvText.wrappedValue = Linear(config.textModelSize, config.modelSize, bias: false)

        self._wkSpeaker.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)
        self._wvSpeaker.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)

        if hasLatentAttention {
            self._wkLatent.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)
            self._wvLatent.wrappedValue = Linear(config.speakerModelSize, config.modelSize, bias: false)
        } else {
            self._wkLatent.wrappedValue = nil
            self._wvLatent.wrappedValue = nil
        }

        self._qNorm.wrappedValue = EchoRMSNorm(numHeads: config.numHeads, headDim: headDim)
        self._kNorm.wrappedValue = EchoRMSNorm(numHeads: config.numHeads, headDim: headDim)
    }

    /// Build text KV cache for a layer.
    func getKVCacheText(_ textState: MLXArray) -> (MLXArray, MLXArray) {
        let bsz = textState.dim(0)
        let seqLen = textState.dim(1)
        let k = kNorm(wkText(textState).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3))
        let v = wvText(textState).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        return (k, v)
    }

    /// Build speaker KV cache for a layer.
    func getKVCacheSpeaker(_ speakerState: MLXArray) -> (MLXArray, MLXArray) {
        let bsz = speakerState.dim(0)
        let seqLen = speakerState.dim(1)
        let k = kNorm(wkSpeaker(speakerState).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3))
        let v = wvSpeaker(speakerState).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        return (k, v)
    }

    func callAsFunction(
        _ x: MLXArray,
        cosFreqs: MLXArray, sinFreqs: MLXArray,
        textKV: (MLXArray, MLXArray),
        speakerKV: (MLXArray, MLXArray),
        latentKV: (MLXArray, MLXArray)? = nil,
        textMask: MLXArray? = nil,
        speakerMask: MLXArray? = nil
    ) -> MLXArray {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Self Q, K, V
        var q = wq(x).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        var selfK = wk(x).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        let selfV = wv(x).reshaped([bsz, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        // QK norm
        q = qNorm(q)
        selfK = kNorm(selfK)

        // Apply RoPE to only first half of Q and self-K positions
        let halfHead = headDim / 2
        let qFirst = q[0..., 0..., 0..., ..<halfHead]
        let qSecond = q[0..., 0..., 0..., halfHead...]
        let qRotated = echoApplyRotaryEmb(qFirst, cosFreqs: cosFreqs, sinFreqs: sinFreqs)
        q = MLX.concatenated([qRotated, qSecond], axis: -1)

        let kFirst = selfK[0..., 0..., 0..., ..<halfHead]
        let kSecond = selfK[0..., 0..., 0..., halfHead...]
        let kRotated = echoApplyRotaryEmb(kFirst, cosFreqs: cosFreqs, sinFreqs: sinFreqs)
        selfK = MLX.concatenated([kRotated, kSecond], axis: -1)

        // Concatenate all KV sources: [self, latent_prefix?, text, speaker]
        var allK = [selfK]
        var allV = [selfV]
        var totalKVLen = seqLen

        if let (lk, lv) = latentKV {
            allK.append(lk)
            allV.append(lv)
            totalKVLen += lk.dim(2)
        }

        let (textK, textV) = textKV
        allK.append(textK)
        allV.append(textV)
        totalKVLen += textK.dim(2)

        let (spkK, spkV) = speakerKV
        allK.append(spkK)
        allV.append(spkV)
        totalKVLen += spkK.dim(2)

        let kCat = MLX.concatenated(allK, axis: 2)
        let vCat = MLX.concatenated(allV, axis: 2)

        // Build combined attention mask
        // Self-attention part: no mask needed (all attend)
        // For text and speaker: use their respective masks
        var maskParts: [MLXArray] = []
        // Self: [seqLen] all zeros (attend)
        maskParts.append(MLXArray.zeros([bsz, 1, 1, seqLen]))

        if let (lk, _) = latentKV {
            maskParts.append(MLXArray.zeros([bsz, 1, 1, lk.dim(2)]))
        }

        if let tm = textMask {
            // tm: [B, textLen] -> [B, 1, 1, textLen] additive
            maskParts.append(echoBoolToAdditiveMask(tm).reshaped([bsz, 1, 1, -1]))
        } else {
            maskParts.append(MLXArray.zeros([bsz, 1, 1, textK.dim(2)]))
        }

        if let sm = speakerMask {
            maskParts.append(echoBoolToAdditiveMask(sm).reshaped([bsz, 1, 1, -1]))
        } else {
            maskParts.append(MLXArray.zeros([bsz, 1, 1, spkK.dim(2)]))
        }

        let combinedMask = MLX.concatenated(maskParts, axis: -1)

        let scale = sqrt(Float(headDim))
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: kCat, values: vCat,
            scale: 1.0 / scale, mask: combinedMask
        )

        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([bsz, seqLen, numHeads * headDim])
        return wo(outputReshaped) * sigmoid(gate(x))
    }
}

// MARK: - MLP (SwiGLU)

class EchoMLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(modelSize: Int, intermediateSize: Int) {
        self._w1.wrappedValue = Linear(modelSize, intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(intermediateSize, modelSize, bias: false)
        self._w3.wrappedValue = Linear(modelSize, intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Encoder Transformer Block (for Text/Speaker Encoders)

class EchoEncoderTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: EchoSelfAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_norm") var attentionNorm: EchoRMSNorm
    @ModuleInfo(key: "mlp_norm") var mlpNorm: EchoRMSNorm

    init(modelSize: Int, numHeads: Int, intermediateSize: Int) {
        self._attention.wrappedValue = EchoSelfAttention(modelSize: modelSize, numHeads: numHeads)
        self._mlp.wrappedValue = EchoMLP(modelSize: modelSize, intermediateSize: intermediateSize)
        self._attentionNorm.wrappedValue = EchoRMSNorm(dim: modelSize)
        self._mlpNorm.wrappedValue = EchoRMSNorm(dim: modelSize)
    }

    func callAsFunction(
        _ x: MLXArray, cosFreqs: MLXArray, sinFreqs: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var h = x + attention(attentionNorm(x), cosFreqs: cosFreqs, sinFreqs: sinFreqs, mask: mask)
        h = h + mlp(mlpNorm(h))
        return h
    }
}

// MARK: - Main DiT Transformer Block

class EchoTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: EchoJointAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_adaln") var attentionAdaLN: EchoLowRankAdaLN
    @ModuleInfo(key: "mlp_adaln") var mlpAdaLN: EchoLowRankAdaLN

    init(config: EchoDiTConfig, hasLatentAttention: Bool = false) {
        self._attention.wrappedValue = EchoJointAttention(config: config, hasLatentAttention: hasLatentAttention)
        self._mlp.wrappedValue = EchoMLP(modelSize: config.modelSize, intermediateSize: config.intermediateSize)
        // condSize is modelSize * 3 (for shift, scale, gate)
        self._attentionAdaLN.wrappedValue = EchoLowRankAdaLN(
            modelSize: config.modelSize, condSize: config.modelSize * 3, rank: config.adalnRank
        )
        self._mlpAdaLN.wrappedValue = EchoLowRankAdaLN(
            modelSize: config.modelSize, condSize: config.modelSize * 3, rank: config.adalnRank
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        condEmbed: MLXArray,
        cosFreqs: MLXArray, sinFreqs: MLXArray,
        textKV: (MLXArray, MLXArray),
        speakerKV: (MLXArray, MLXArray),
        latentKV: (MLXArray, MLXArray)? = nil,
        textMask: MLXArray? = nil,
        speakerMask: MLXArray? = nil
    ) -> MLXArray {
        // Attention with AdaLN
        let (xNormAttn, attnGate) = attentionAdaLN(x, condEmbed: condEmbed)
        let attnOutput = attention(
            xNormAttn,
            cosFreqs: cosFreqs, sinFreqs: sinFreqs,
            textKV: textKV, speakerKV: speakerKV,
            latentKV: latentKV,
            textMask: textMask, speakerMask: speakerMask
        )
        var h = x + attnGate * attnOutput

        // MLP with AdaLN
        let (hNormMLP, mlpGate) = mlpAdaLN(h, condEmbed: condEmbed)
        h = h + mlpGate * mlp(hNormMLP)

        return h
    }
}

// MARK: - Text Encoder

class EchoTextEncoder: Module {
    let modelSize: Int
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "layers") var layers: [EchoEncoderTransformerBlock]

    init(config: EchoDiTConfig) {
        self.modelSize = config.textModelSize
        self.numHeads = config.textNumHeads
        self.headDim = config.textModelSize / config.textNumHeads

        self._embedding.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize, dimensions: config.textModelSize
        )
        self._layers.wrappedValue = (0..<config.textNumLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: config.textModelSize,
                numHeads: config.textNumHeads,
                intermediateSize: config.textIntermediateSize
            )
        }
    }

    func callAsFunction(_ tokens: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let seqLen = tokens.dim(1)
        var h = embedding(tokens)

        let (cos, sin) = echoPrecomputeFreqsCis(dim: headDim, end: seqLen)

        // Non-causal: no causal mask, just key mask from padding
        let attnMask: MLXArray?
        if let m = mask {
            // [B, seqLen] -> [B, 1, 1, seqLen]
            attnMask = echoBoolToAdditiveMask(m).reshaped([m.dim(0), 1, 1, -1])
        } else {
            attnMask = nil
        }

        for layer in layers {
            h = layer(h, cosFreqs: cos, sinFreqs: sin, mask: attnMask)
        }
        return h
    }
}

// MARK: - Speaker Encoder

class EchoSpeakerEncoder: Module {
    let modelSize: Int
    let numHeads: Int
    let headDim: Int
    let patchSize: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "layers") var layers: [EchoEncoderTransformerBlock]

    init(config: EchoDiTConfig) {
        self.modelSize = config.speakerModelSize
        self.numHeads = config.speakerNumHeads
        self.headDim = config.speakerModelSize / config.speakerNumHeads
        self.patchSize = config.speakerPatchSize

        // Input projection from patched latent
        self._inProj.wrappedValue = Linear(
            config.latentSize * config.speakerPatchSize,
            config.speakerModelSize,
            bias: false
        )
        self._layers.wrappedValue = (0..<config.speakerNumLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: config.speakerModelSize,
                numHeads: config.speakerNumHeads,
                intermediateSize: config.speakerIntermediateSize
            )
        }
    }

    func callAsFunction(_ latent: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let bsz = latent.dim(0)
        let seqLen = latent.dim(1)
        let latentSize = latent.dim(2)

        // Patch: [B, T, latentSize] -> [B, T/patchSize, latentSize * patchSize]
        let patchedLen = seqLen / patchSize
        let patched = latent.reshaped([bsz, patchedLen, latentSize * patchSize])

        // Project and scale
        var h = inProj(patched) / 6.0

        let (cos, sin) = echoPrecomputeFreqsCis(dim: headDim, end: patchedLen)

        // Causal attention
        let causalMask = echoMakeCausalMask(seqLen: patchedLen)

        // Combine with key mask if provided
        var attnMask = causalMask
        if let m = mask {
            // Downsample mask by patchSize
            // Take every patchSize-th element
            let dsMask = m[0..., stride(from: 0, to: m.dim(1), by: patchSize)]
            let keyMask = echoBoolToAdditiveMask(dsMask).reshaped([bsz, 1, 1, -1])
            attnMask = causalMask + keyMask
        }

        for layer in layers {
            h = layer(h, cosFreqs: cos, sinFreqs: sin, mask: attnMask)
        }
        return h
    }
}

// MARK: - EchoDiT (Main Model)

public class EchoDiT: Module {
    let config: EchoDiTConfig

    @ModuleInfo(key: "text_encoder") var textEncoder: EchoTextEncoder
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: EchoSpeakerEncoder
    @ModuleInfo(key: "text_norm") var textNorm: EchoRMSNorm
    @ModuleInfo(key: "speaker_norm") var speakerNorm: EchoRMSNorm

    // Optional latent encoder for blockwise generation
    @ModuleInfo(key: "latent_encoder") var latentEncoder: EchoSpeakerEncoder?
    @ModuleInfo(key: "latent_norm") var latentNorm: EchoRMSNorm?

    // Condition module: timestep -> 3 * modelSize (for AdaLN shift/scale/gate)
    @ModuleInfo(key: "cond_module") var condModule: EchoSequential

    // Input/output projections
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_norm") var outNorm: EchoRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    // Main transformer blocks
    @ModuleInfo(key: "layers") var layers: [EchoTransformerBlock]

    public init(_ config: EchoDiTConfig, hasLatentAttention: Bool = false) {
        self.config = config

        self._textEncoder.wrappedValue = EchoTextEncoder(config: config)
        self._speakerEncoder.wrappedValue = EchoSpeakerEncoder(config: config)
        self._textNorm.wrappedValue = EchoRMSNorm(dim: config.textModelSize)
        self._speakerNorm.wrappedValue = EchoRMSNorm(dim: config.speakerModelSize)

        if hasLatentAttention {
            self._latentEncoder.wrappedValue = EchoSpeakerEncoder(config: config)
            self._latentNorm.wrappedValue = EchoRMSNorm(dim: config.speakerModelSize)
        } else {
            self._latentEncoder.wrappedValue = nil
            self._latentNorm.wrappedValue = nil
        }

        // Condition module: Sequential(Linear, SiLU, Linear, SiLU, Linear)
        // timestepEmbedSize -> modelSize -> modelSize -> modelSize * 3
        self._condModule.wrappedValue = EchoSequential(layers: [
            Linear(config.timestepEmbedSize, config.modelSize),
            SiLUModule(),
            Linear(config.modelSize, config.modelSize),
            SiLUModule(),
            Linear(config.modelSize, config.modelSize * 3),
        ])

        self._inProj.wrappedValue = Linear(config.latentSize, config.modelSize, bias: false)
        self._outNorm.wrappedValue = EchoRMSNorm(dim: config.modelSize)
        self._outProj.wrappedValue = Linear(config.modelSize, config.latentSize, bias: false)

        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            EchoTransformerBlock(config: config, hasLatentAttention: hasLatentAttention)
        }
    }

    /// Encode text and build per-layer KV caches.
    public func getKVCacheText(_ tokens: MLXArray, mask: MLXArray? = nil) -> EchoKVCache {
        var textState = textEncoder(tokens, mask: mask)
        textState = textNorm(textState)
        return layers.map { layer in
            layer.attention.getKVCacheText(textState)
        }
    }

    /// Encode speaker latent and build per-layer KV caches.
    public func getKVCacheSpeaker(_ latent: MLXArray, mask: MLXArray? = nil) -> EchoKVCache {
        var speakerState = speakerEncoder(latent, mask: mask)
        speakerState = speakerNorm(speakerState)
        return layers.map { layer in
            layer.attention.getKVCacheSpeaker(speakerState)
        }
    }

    /// Forward pass through the DiT.
    public func callAsFunction(
        _ x: MLXArray,
        timestep: MLXArray,
        textKVCache: EchoKVCache,
        speakerKVCache: EchoKVCache,
        latentKVCache: EchoKVCache? = nil,
        textMask: MLXArray? = nil,
        speakerMask: MLXArray? = nil
    ) -> MLXArray {
        let seqLen = x.dim(1)

        // Timestep conditioning
        let tEmbed = echoGetTimestepEmbedding(timestep, embedSize: config.timestepEmbedSize)
        let condEmbed = condModule(tEmbed).expandedDimensions(axis: 1)  // [B, 1, modelSize*3]

        // Project input
        var h = inProj(x)

        // Precompute RoPE
        let halfHead = (config.modelSize / config.numHeads) / 2
        let (cos, sin) = echoPrecomputeFreqsCis(dim: halfHead * 2, end: seqLen)

        // Downsample speaker mask by patchSize for attention
        var dsSpeakerMask = speakerMask
        if let sm = speakerMask {
            let patchSize = config.speakerPatchSize
            let patchedLen = sm.dim(1) / patchSize
            if patchedLen > 0 {
                dsSpeakerMask = sm[0..., stride(from: 0, to: sm.dim(1), by: patchSize)]
            }
        }

        // Run through transformer blocks
        for (i, layer) in layers.enumerated() {
            let textKV = textKVCache[i]
            let speakerKV = speakerKVCache[i]
            let latentKV = latentKVCache?[i]

            h = layer(
                h,
                condEmbed: condEmbed,
                cosFreqs: cos, sinFreqs: sin,
                textKV: textKV, speakerKV: speakerKV,
                latentKV: latentKV,
                textMask: textMask, speakerMask: dsSpeakerMask
            )
        }

        // Output projection
        h = outNorm(h)
        h = outProj(h).asType(.float32)
        return h
    }
}

// MARK: - Sequential Wrapper

/// Simple sequential module for the condition module.
class EchoSequential: Module, UnaryLayer {
    @ModuleInfo var layers: [Module]

    init(layers: [Module]) {
        self._layers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let unary = layer as? UnaryLayer {
                h = unary(h)
            }
        }
        return h
    }
}

/// SiLU activation as a module.
class SiLUModule: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(x)
    }
}
