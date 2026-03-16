import Foundation
import MLX
import MLXAudioCore
import MLXNN

typealias EchoTTSRotaryCache = (cos: MLXArray, sin: MLXArray)
typealias EchoTTSKVCache = (keys: MLXArray, values: MLXArray)

private func echoTtsCallUnary(_ layer: Module, _ x: MLXArray) -> MLXArray {
    (layer as! UnaryLayer).callAsFunction(x)
}

func echoTtsPrecomputeFreqsCis(dim: Int, end: Int, theta: Float = 10_000) -> EchoTTSRotaryCache {
    let indices = MLX.arange(0, dim, step: 2, dtype: .float32) / Float(dim)
    let freqs = 1 / MLX.pow(MLXArray(theta), indices)
    let positions = MLX.arange(end, dtype: .float32)
    let angles = MLX.outer(positions, freqs)
    return (MLX.cos(angles), MLX.sin(angles))
}

func echoTtsApplyRotaryEmb(_ x: MLXArray, freqsCis: EchoTTSRotaryCache) -> MLXArray {
    let xEven = x[0..., 0..., 0..., .stride(by: 2)]
    let xOdd = x[0..., 0..., 0..., .stride(from: 1, by: 2)]
    let cos = freqsCis.cos.expandedDimensions(axes: [0, 2])
    let sin = freqsCis.sin.expandedDimensions(axes: [0, 2])

    let rotatedEven = xEven * cos - xOdd * sin
    let rotatedOdd = xOdd * cos + xEven * sin
    return MLX.stacked([rotatedEven, rotatedOdd], axis: -1).reshaped(x.shape)
}

func echoTtsTimestepEmbedding(_ timestep: MLXArray, embedSize: Int) -> MLXArray {
    precondition(embedSize % 2 == 0, "embedSize must be even")
    let half = embedSize / 2
    let base = log(MLXArray(10_000, dtype: .float32))
    let exponent = MLX.arange(half, dtype: .float32) / Float(half)
    let freqs = 1_000 * MLX.exp(-base * exponent)
    let args = timestep.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(timestep.dtype)
}

private func echoTtsBoolToAdditiveMask(_ mask: MLXArray) -> MLXArray {
    let zero = MLXArray.zeros(mask.shape, dtype: .float32)
    let negInf = MLXArray.full(mask.shape, values: MLXArray(-1e9), dtype: .float32)
    return MLX.where(mask, zero, negInf).expandedDimensions(axis: 1)
}

private func echoTtsMakeCausalMask(_ length: Int) -> MLXArray {
    let row = MLX.arange(length).expandedDimensions(axis: 1)
    let col = MLX.arange(length).expandedDimensions(axis: 0)
    return row .>= col
}

final class EchoSequential: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ layers: [Module]) {
        self._layers = ModuleInfo(wrappedValue: layers)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = echoTtsCallUnary(layer, hidden)
        }
        return hidden
    }
}

final class EchoLowRankAdaLN: Module {
    let eps: Float

    @ModuleInfo(key: "shift_down") var shiftDown: Linear
    @ModuleInfo(key: "scale_down") var scaleDown: Linear
    @ModuleInfo(key: "gate_down") var gateDown: Linear
    @ModuleInfo(key: "shift_up") var shiftUp: Linear
    @ModuleInfo(key: "scale_up") var scaleUp: Linear
    @ModuleInfo(key: "gate_up") var gateUp: Linear

    init(modelSize: Int, rank: Int, eps: Float) {
        self.eps = eps
        self._shiftDown = ModuleInfo(wrappedValue: Linear(modelSize, rank, bias: false))
        self._scaleDown = ModuleInfo(wrappedValue: Linear(modelSize, rank, bias: false))
        self._gateDown = ModuleInfo(wrappedValue: Linear(modelSize, rank, bias: false))
        self._shiftUp = ModuleInfo(wrappedValue: Linear(rank, modelSize, bias: true))
        self._scaleUp = ModuleInfo(wrappedValue: Linear(rank, modelSize, bias: true))
        self._gateUp = ModuleInfo(wrappedValue: Linear(rank, modelSize, bias: true))
    }

    func callAsFunction(_ x: MLXArray, condEmbed: MLXArray) -> (MLXArray, MLXArray) {
        let parts = condEmbed.split(parts: 3, axis: -1)
        let shift = shiftUp(shiftDown(silu(parts[0]))) + parts[0]
        let scale = scaleUp(scaleDown(silu(parts[1]))) + parts[1]
        let gate = tanh(gateUp(gateDown(silu(parts[2]))) + parts[2])

        let xFloat = x.asType(.float32)
        let normalized = xFloat * rsqrt(mean(xFloat * xFloat, axis: -1, keepDims: true) + MLXArray(eps))
        let output = normalized * (scale + 1) + shift
        return (output.asType(x.dtype), gate)
    }
}

final class EchoRMSNorm: Module {
    let eps: Float
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(shape: [Int], eps: Float) {
        self.eps = eps
        self._weight = ModuleInfo(wrappedValue: MLXArray.ones(shape))
    }

    convenience init(dimensions: Int, eps: Float) {
        self.init(shape: [dimensions], eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let normalized = xFloat * rsqrt(mean(xFloat * xFloat, axis: -1, keepDims: true) + MLXArray(eps))
        return (normalized * weight).asType(x.dtype)
    }
}

final class EchoSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let isCausal: Bool
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm

    init(modelSize: Int, numHeads: Int, isCausal: Bool, normEps: Float) {
        precondition(modelSize % numHeads == 0, "modelSize must be divisible by numHeads")

        self.numHeads = numHeads
        self.headDim = modelSize / numHeads
        self.isCausal = isCausal
        self.scale = 1 / sqrt(Float(self.headDim))

        self._wq = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wk = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wv = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wo = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._gate = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._qNorm = ModuleInfo(wrappedValue: EchoRMSNorm(shape: [numHeads, self.headDim], eps: normEps))
        self._kNorm = ModuleInfo(wrappedValue: EchoRMSNorm(shape: [numHeads, self.headDim], eps: normEps))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, freqsCis: EchoTTSRotaryCache) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]

        var q = wq(x).reshaped([batch, seqLen, numHeads, headDim])
        var k = wk(x).reshaped([batch, seqLen, numHeads, headDim])
        let v = wv(x).reshaped([batch, seqLen, numHeads, headDim])
        let gateValues = gate(x)

        q = qNorm(q)
        k = kNorm(k)

        let slicedFreqs = (cos: freqsCis.cos[0..<seqLen], sin: freqsCis.sin[0..<seqLen])
        q = echoTtsApplyRotaryEmb(q, freqsCis: slicedFreqs)
        k = echoTtsApplyRotaryEmb(k, freqsCis: slicedFreqs)

        var maskBool: MLXArray?
        if let mask {
            maskBool = MLX.broadcast(mask.expandedDimensions(axis: 1), to: [batch, seqLen, seqLen])
        }
        if isCausal {
            let causal = MLX.broadcast(
                echoTtsMakeCausalMask(seqLen).expandedDimensions(axis: 0),
                to: [batch, seqLen, seqLen]
            )
            maskBool = maskBool == nil ? causal : (maskBool! .&& causal)
        }

        let attentionMask = maskBool.map(echoTtsBoolToAdditiveMask)
        let output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            scale: scale,
            mask: attentionMask
        )
        let merged = output.transposed(0, 2, 1, 3).reshaped([batch, seqLen, numHeads * headDim])
        return wo(merged * sigmoid(gateValues))
    }
}

final class EchoJointAttention: Module {
    let speakerPatchSize: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let useLatentKV: Bool

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wk_text") var wkText: Linear
    @ModuleInfo(key: "wv_text") var wvText: Linear
    @ModuleInfo(key: "wk_speaker") var wkSpeaker: Linear
    @ModuleInfo(key: "wv_speaker") var wvSpeaker: Linear
    @ModuleInfo(key: "wk_latent") var wkLatent: Linear?
    @ModuleInfo(key: "wv_latent") var wvLatent: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(
        modelSize: Int,
        numHeads: Int,
        textModelSize: Int,
        speakerModelSize: Int,
        speakerPatchSize: Int,
        normEps: Float,
        useLatentKV: Bool = true
    ) {
        precondition(modelSize % numHeads == 0, "modelSize must be divisible by numHeads")

        self.speakerPatchSize = speakerPatchSize
        self.numHeads = numHeads
        self.headDim = modelSize / numHeads
        self.scale = 1 / sqrt(Float(self.headDim))
        self.useLatentKV = useLatentKV

        self._wq = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wk = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wv = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wkText = ModuleInfo(wrappedValue: Linear(textModelSize, modelSize, bias: false))
        self._wvText = ModuleInfo(wrappedValue: Linear(textModelSize, modelSize, bias: false))
        self._wkSpeaker = ModuleInfo(wrappedValue: Linear(speakerModelSize, modelSize, bias: false))
        self._wvSpeaker = ModuleInfo(wrappedValue: Linear(speakerModelSize, modelSize, bias: false))
        self._wkLatent = ModuleInfo(wrappedValue: useLatentKV ? Linear(speakerModelSize, modelSize, bias: false) : nil)
        self._wvLatent = ModuleInfo(wrappedValue: useLatentKV ? Linear(speakerModelSize, modelSize, bias: false) : nil)
        self._qNorm = ModuleInfo(wrappedValue: EchoRMSNorm(shape: [numHeads, self.headDim], eps: normEps))
        self._kNorm = ModuleInfo(wrappedValue: EchoRMSNorm(shape: [numHeads, self.headDim], eps: normEps))
        self._gate = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
        self._wo = ModuleInfo(wrappedValue: Linear(modelSize, modelSize, bias: false))
    }

    private func applyRotaryToHalfHeads(_ x: MLXArray, freqsCis: EchoTTSRotaryCache) -> MLXArray {
        let halfHeads = x.shape[2] / 2
        let rotated = echoTtsApplyRotaryEmb(x[0..., 0..., 0..<halfHeads, 0...], freqsCis: freqsCis)
        let untouched = x[0..., 0..., halfHeads..., 0...]
        return MLX.concatenated([rotated, untouched], axis: 2)
    }

    func callAsFunction(
        _ x: MLXArray,
        textMask: MLXArray,
        speakerMask: MLXArray,
        freqsCis: EchoTTSRotaryCache,
        kvCacheText: EchoTTSKVCache,
        kvCacheSpeaker: EchoTTSKVCache,
        startPos: Int?,
        kvCacheLatent: EchoTTSKVCache?
    ) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]
        let resolvedStartPos = startPos ?? 0

        var q = wq(x).reshaped([batch, seqLen, numHeads, headDim])
        var kSelf = wk(x).reshaped([batch, seqLen, numHeads, headDim])
        let vSelf = wv(x).reshaped([batch, seqLen, numHeads, headDim])
        let gateValues = gate(x)

        q = qNorm(q)
        kSelf = kNorm(kSelf)

        let qFreqs = (
            cos: freqsCis.cos[resolvedStartPos..<(resolvedStartPos + seqLen)],
            sin: freqsCis.sin[resolvedStartPos..<(resolvedStartPos + seqLen)]
        )
        q = applyRotaryToHalfHeads(q, freqsCis: qFreqs)
        kSelf = applyRotaryToHalfHeads(kSelf, freqsCis: qFreqs)

        let latentKeys: MLXArray
        let latentValues: MLXArray
        let latentMask: MLXArray
        if let kvCacheLatent, kvCacheLatent.keys.shape[1] > 0 {
            latentKeys = kvCacheLatent.keys
            latentValues = kvCacheLatent.values
            let positions = MLX.arange(kvCacheLatent.keys.shape[1], dtype: .int32) * speakerPatchSize
            latentMask = MLX.broadcast(
                (positions.expandedDimensions(axis: 0) .< resolvedStartPos),
                to: [batch, kvCacheLatent.keys.shape[1]]
            )
        } else {
            latentKeys = MLXArray.zeros([batch, 0, numHeads, headDim], dtype: x.dtype)
            latentValues = MLXArray.zeros([batch, 0, numHeads, headDim], dtype: x.dtype)
            latentMask = MLXArray.zeros([batch, 0], dtype: .bool)
        }

        let keys = MLX.concatenated([kSelf, latentKeys, kvCacheText.keys, kvCacheSpeaker.keys], axis: 1)
        let values = MLX.concatenated([vSelf, latentValues, kvCacheText.values, kvCacheSpeaker.values], axis: 1)
        let selfMask = MLXArray.ones([batch, seqLen], dtype: .bool)
        let fullMask = MLX.concatenated([selfMask, latentMask, textMask, speakerMask], axis: 1)
        let attentionMask = echoTtsBoolToAdditiveMask(
            MLX.broadcast(fullMask.expandedDimensions(axis: 1), to: [batch, seqLen, fullMask.shape[1]])
        )

        let output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: keys.transposed(0, 2, 1, 3),
            values: values.transposed(0, 2, 1, 3),
            scale: scale,
            mask: attentionMask
        )
        let merged = output.transposed(0, 2, 1, 3).reshaped([batch, seqLen, numHeads * headDim])
        return wo(merged * sigmoid(gateValues))
    }

    func getKVCacheText(_ textState: MLXArray) -> EchoTTSKVCache {
        let batch = textState.shape[0]
        let length = textState.shape[1]
        let keys = kNorm(wkText(textState).reshaped([batch, length, numHeads, headDim]))
        let values = wvText(textState).reshaped([batch, length, numHeads, headDim])
        return (keys, values)
    }

    func getKVCacheSpeaker(_ speakerState: MLXArray) -> EchoTTSKVCache {
        let batch = speakerState.shape[0]
        let length = speakerState.shape[1]
        let keys = kNorm(wkSpeaker(speakerState).reshaped([batch, length, numHeads, headDim]))
        let values = wvSpeaker(speakerState).reshaped([batch, length, numHeads, headDim])
        return (keys, values)
    }

    func getKVCacheLatent(_ latentState: MLXArray, freqsCis: EchoTTSRotaryCache) throws -> EchoTTSKVCache {
        guard useLatentKV, let wkLatent, let wvLatent else {
            throw AudioGenerationError.invalidInput(
                "Latent prefix modules are disabled. Use deleteBlockwiseModules=false to enable blockwise generation."
            )
        }
        let batch = latentState.shape[0]
        let length = latentState.shape[1]
        let keys = kNorm(wkLatent(latentState).reshaped([batch, length, numHeads, headDim]))
        let values = wvLatent(latentState).reshaped([batch, length, numHeads, headDim])
        return (applyRotaryToHalfHeads(keys, freqsCis: freqsCis), values)
    }
}

final class EchoMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w3") var w3: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(modelSize: Int, intermediateSize: Int) {
        self._w1 = ModuleInfo(wrappedValue: Linear(modelSize, intermediateSize, bias: false))
        self._w3 = ModuleInfo(wrappedValue: Linear(modelSize, intermediateSize, bias: false))
        self._w2 = ModuleInfo(wrappedValue: Linear(intermediateSize, modelSize, bias: false))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class EchoEncoderTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: EchoSelfAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_norm") var attentionNorm: EchoRMSNorm
    @ModuleInfo(key: "mlp_norm") var mlpNorm: EchoRMSNorm

    init(
        modelSize: Int,
        numHeads: Int,
        intermediateSize: Int,
        isCausal: Bool,
        normEps: Float
    ) {
        self._attention = ModuleInfo(wrappedValue: EchoSelfAttention(
            modelSize: modelSize,
            numHeads: numHeads,
            isCausal: isCausal,
            normEps: normEps
        ))
        self._mlp = ModuleInfo(wrappedValue: EchoMLP(modelSize: modelSize, intermediateSize: intermediateSize))
        self._attentionNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: modelSize, eps: normEps))
        self._mlpNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: modelSize, eps: normEps))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, freqsCis: EchoTTSRotaryCache) -> MLXArray {
        let attended = attention(attentionNorm(x), mask: mask, freqsCis: freqsCis)
        let withAttention = x + attended
        return withAttention + mlp(mlpNorm(withAttention))
    }
}

final class EchoTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: EchoJointAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_adaln") var attentionAdaLN: EchoLowRankAdaLN
    @ModuleInfo(key: "mlp_adaln") var mlpAdaLN: EchoLowRankAdaLN

    init(
        modelSize: Int,
        numHeads: Int,
        intermediateSize: Int,
        normEps: Float,
        textModelSize: Int,
        speakerModelSize: Int,
        speakerPatchSize: Int,
        adalnRank: Int,
        useLatentKV: Bool = true
    ) {
        self._attention = ModuleInfo(wrappedValue: EchoJointAttention(
            modelSize: modelSize,
            numHeads: numHeads,
            textModelSize: textModelSize,
            speakerModelSize: speakerModelSize,
            speakerPatchSize: speakerPatchSize,
            normEps: normEps,
            useLatentKV: useLatentKV
        ))
        self._mlp = ModuleInfo(wrappedValue: EchoMLP(modelSize: modelSize, intermediateSize: intermediateSize))
        self._attentionAdaLN = ModuleInfo(wrappedValue: EchoLowRankAdaLN(modelSize: modelSize, rank: adalnRank, eps: normEps))
        self._mlpAdaLN = ModuleInfo(wrappedValue: EchoLowRankAdaLN(modelSize: modelSize, rank: adalnRank, eps: normEps))
    }

    func callAsFunction(
        _ x: MLXArray,
        condEmbed: MLXArray,
        textMask: MLXArray,
        speakerMask: MLXArray,
        freqsCis: EchoTTSRotaryCache,
        kvCacheText: EchoTTSKVCache,
        kvCacheSpeaker: EchoTTSKVCache,
        startPos: Int?,
        kvCacheLatent: EchoTTSKVCache?
    ) -> MLXArray {
        let (attentionInput, attentionGate) = attentionAdaLN(x, condEmbed: condEmbed)
        let attended = attention(
            attentionInput,
            textMask: textMask,
            speakerMask: speakerMask,
            freqsCis: freqsCis,
            kvCacheText: kvCacheText,
            kvCacheSpeaker: kvCacheSpeaker,
            startPos: startPos,
            kvCacheLatent: kvCacheLatent
        )
        let withAttention = x + attentionGate * attended

        let (mlpInput, mlpGate) = mlpAdaLN(withAttention, condEmbed: condEmbed)
        return withAttention + mlpGate * mlp(mlpInput)
    }
}

final class EchoTextEncoder: Module {
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [EchoEncoderTransformerBlock]

    let headDim: Int

    init(
        vocabSize: Int,
        modelSize: Int,
        numLayers: Int,
        numHeads: Int,
        intermediateSize: Int,
        normEps: Float
    ) {
        self._textEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: vocabSize, dimensions: modelSize))
        self._blocks = ModuleInfo(wrappedValue: (0 ..< numLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: modelSize,
                numHeads: numHeads,
                intermediateSize: intermediateSize,
                isCausal: false,
                normEps: normEps
            )
        })
        self.headDim = modelSize / numHeads
    }

    func callAsFunction(_ inputIDs: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = textEmbedding(inputIDs)
        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: inputIDs.shape[1])
        for block in blocks {
            hidden = block(hidden, mask: mask, freqsCis: freqs)
        }
        return hidden
    }
}

final class EchoSpeakerEncoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [EchoEncoderTransformerBlock]

    let patchSize: Int
    let headDim: Int

    init(
        latentSize: Int,
        patchSize: Int,
        modelSize: Int,
        numLayers: Int,
        numHeads: Int,
        intermediateSize: Int,
        normEps: Float
    ) {
        self.patchSize = patchSize
        self.headDim = modelSize / numHeads
        self._inProj = ModuleInfo(wrappedValue: Linear(latentSize * patchSize, modelSize, bias: true))
        self._blocks = ModuleInfo(wrappedValue: (0 ..< numLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: modelSize,
                numHeads: numHeads,
                intermediateSize: intermediateSize,
                isCausal: true,
                normEps: normEps
            )
        })
    }

    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let seqLenPatched = (latent.shape[1] / patchSize) * patchSize
        let trimmed = latent[0..., 0..<seqLenPatched, 0...]
        var hidden = trimmed.reshaped([
            trimmed.shape[0],
            seqLenPatched / patchSize,
            trimmed.shape[2] * patchSize,
        ])
        hidden = inProj(hidden) / 6

        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: hidden.shape[1])
        for block in blocks {
            hidden = block(hidden, mask: nil, freqsCis: freqs)
        }
        return hidden
    }
}

public final class EchoDiT: Module {
    public let speakerPatchSize: Int
    public let timestepEmbedSize: Int
    public let enableBlockwiseModules: Bool
    let headDim: Int

    @ModuleInfo(key: "text_encoder") var textEncoder: EchoTextEncoder
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: EchoSpeakerEncoder
    @ModuleInfo(key: "latent_encoder") var latentEncoder: EchoSpeakerEncoder?
    @ModuleInfo(key: "latent_norm") var latentNorm: EchoRMSNorm?
    @ModuleInfo(key: "text_norm") var textNorm: EchoRMSNorm
    @ModuleInfo(key: "speaker_norm") var speakerNorm: EchoRMSNorm
    @ModuleInfo(key: "cond_module") var condModule: EchoSequential
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [EchoTransformerBlock]
    @ModuleInfo(key: "out_norm") var outNorm: EchoRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(
        latentSize: Int,
        modelSize: Int,
        numLayers: Int,
        numHeads: Int,
        intermediateSize: Int,
        normEps: Float,
        textVocabSize: Int,
        textModelSize: Int,
        textNumLayers: Int,
        textNumHeads: Int,
        textIntermediateSize: Int,
        speakerPatchSize: Int,
        speakerModelSize: Int,
        speakerNumLayers: Int,
        speakerNumHeads: Int,
        speakerIntermediateSize: Int,
        timestepEmbedSize: Int,
        adalnRank: Int,
        enableBlockwiseModules: Bool = true
    ) {
        self.speakerPatchSize = speakerPatchSize
        self.timestepEmbedSize = timestepEmbedSize
        self.enableBlockwiseModules = enableBlockwiseModules
        self.headDim = modelSize / numHeads

        self._textEncoder = ModuleInfo(wrappedValue: EchoTextEncoder(
            vocabSize: textVocabSize,
            modelSize: textModelSize,
            numLayers: textNumLayers,
            numHeads: textNumHeads,
            intermediateSize: textIntermediateSize,
            normEps: normEps
        ))
        self._speakerEncoder = ModuleInfo(wrappedValue: EchoSpeakerEncoder(
            latentSize: latentSize,
            patchSize: speakerPatchSize,
            modelSize: speakerModelSize,
            numLayers: speakerNumLayers,
            numHeads: speakerNumHeads,
            intermediateSize: speakerIntermediateSize,
            normEps: normEps
        ))
        self._latentEncoder = ModuleInfo(wrappedValue: enableBlockwiseModules ? EchoSpeakerEncoder(
            latentSize: latentSize,
            patchSize: speakerPatchSize,
            modelSize: speakerModelSize,
            numLayers: speakerNumLayers,
            numHeads: speakerNumHeads,
            intermediateSize: speakerIntermediateSize,
            normEps: normEps
        ) : nil)
        self._latentNorm = ModuleInfo(wrappedValue: enableBlockwiseModules ? EchoRMSNorm(dimensions: speakerModelSize, eps: normEps) : nil)
        self._textNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: textModelSize, eps: normEps))
        self._speakerNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: speakerModelSize, eps: normEps))
        self._condModule = ModuleInfo(wrappedValue: EchoSequential([
            Linear(timestepEmbedSize, modelSize, bias: false),
            SiLU(),
            Linear(modelSize, modelSize, bias: false),
            SiLU(),
            Linear(modelSize, modelSize * 3, bias: false),
        ]))
        self._inProj = ModuleInfo(wrappedValue: Linear(latentSize, modelSize, bias: true))
        self._blocks = ModuleInfo(wrappedValue: (0 ..< numLayers).map { _ in
            EchoTransformerBlock(
                modelSize: modelSize,
                numHeads: numHeads,
                intermediateSize: intermediateSize,
                normEps: normEps,
                textModelSize: textModelSize,
                speakerModelSize: speakerModelSize,
                speakerPatchSize: speakerPatchSize,
                adalnRank: adalnRank,
                useLatentKV: enableBlockwiseModules
            )
        })
        self._outNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: modelSize, eps: normEps))
        self._outProj = ModuleInfo(wrappedValue: Linear(modelSize, latentSize, bias: true))
    }

    func callAsFunction(
        x: MLXArray,
        t: MLXArray,
        textMask: MLXArray,
        speakerMask: MLXArray,
        kvCacheText: [EchoTTSKVCache],
        kvCacheSpeaker: [EchoTTSKVCache],
        startPos: Int? = nil,
        kvCacheLatent: [EchoTTSKVCache]? = nil
    ) -> MLXArray {
        let resolvedStartPos = startPos ?? 0
        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: resolvedStartPos + x.shape[1])
        let speakerPatchMask = speakerMask[0..., .stride(from: 0, by: speakerPatchSize)]

        let condEmbed = condModule(echoTtsTimestepEmbedding(t, embedSize: timestepEmbedSize)).expandedDimensions(axis: 1)

        var hidden = inProj(x)
        for (index, block) in blocks.enumerated() {
            hidden = block(
                hidden,
                condEmbed: condEmbed,
                textMask: textMask,
                speakerMask: speakerPatchMask,
                freqsCis: freqs,
                kvCacheText: kvCacheText[index],
                kvCacheSpeaker: kvCacheSpeaker[index],
                startPos: startPos,
                kvCacheLatent: kvCacheLatent?[index]
            )
        }

        return outProj(outNorm(hidden)).asType(.float32)
    }

    func getKVCacheText(_ textInputIDs: MLXArray, textMask: MLXArray?) -> [EchoTTSKVCache] {
        let textState = textNorm(textEncoder(textInputIDs, mask: textMask))
        return blocks.map { $0.attention.getKVCacheText(textState) }
    }

    func getKVCacheSpeaker(_ speakerLatent: MLXArray) -> [EchoTTSKVCache] {
        let speakerState = speakerNorm(speakerEncoder(speakerLatent))
        return blocks.map { $0.attention.getKVCacheSpeaker(speakerState) }
    }

    func getKVCacheLatent(_ prefixLatent: MLXArray) throws -> [EchoTTSKVCache] {
        guard enableBlockwiseModules, let latentEncoder, let latentNorm else {
            throw AudioGenerationError.invalidInput(
                "Latent prefix modules are disabled. Use deleteBlockwiseModules=false to enable blockwise generation."
            )
        }

        if prefixLatent.shape[1] == 0 {
            return blocks.map { block in
                let shape = [prefixLatent.shape[0], 0, block.attention.numHeads, block.attention.headDim]
                return (
                    MLXArray.zeros(shape, dtype: prefixLatent.dtype),
                    MLXArray.zeros(shape, dtype: prefixLatent.dtype)
                )
            }
        }

        let latentState = latentNorm(latentEncoder(prefixLatent))
        let positions = MLX.arange(latentState.shape[1], dtype: .int32) * speakerPatchSize
        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: latentState.shape[1] * speakerPatchSize)
        let latentFreqs = (
            cos: freqs.cos[positions],
            sin: freqs.sin[positions]
        )
        return try blocks.map { try $0.attention.getKVCacheLatent(latentState, freqsCis: latentFreqs) }
    }
}
