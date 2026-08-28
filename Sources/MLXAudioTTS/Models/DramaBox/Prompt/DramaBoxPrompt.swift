import Foundation
@preconcurrency import MLX
import MLXNN

func dramaBoxConvertToAdditiveMask(_ attentionMask: MLXArray, dtype: DType) -> MLXArray {
    let B = attentionMask.dim(0)
    let T = attentionMask.dim(1)
    let maskMinusOne = (attentionMask.asType(.int32) - 1).asType(dtype).reshaped(B, 1, 1, T)
    return maskMinusOne * MLXArray(dramaBoxFinfoMax(dtype), dtype: dtype)
}

final class DramaBoxFeatureExtractor: Module {
    let embeddingDim: Int
    let outFeatures: Int
    let numLayers: Int
    let rescale: Float
    @ModuleInfo(key: "audio_aggregate_embed") var audioAggregateEmbed: Linear

    init(embeddingDim: Int = 3840, outFeatures: Int = 2048, numLayers: Int = 49) {
        self.embeddingDim = embeddingDim
        self.outFeatures = outFeatures
        self.numLayers = numLayers
        self.rescale = Float(sqrt(Double(outFeatures) / Double(embeddingDim)))
        self._audioAggregateEmbed.wrappedValue = Linear(
            embeddingDim * numLayers, outFeatures, bias: true
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: [MLXArray], attentionMask: MLXArray) -> MLXArray {
        let encoded = stacked(hiddenStates, axis: -1)
        let origDtype = encoded.dtype
        let x32 = encoded.asType(.float32)
        let variance = MLX.mean(x32 * x32, axis: 2, keepDims: true)
        var normed = x32 * MLX.rsqrt(variance + 1e-6)
        let B = encoded.dim(0)
        let T = encoded.dim(1)
        let D = encoded.dim(2)
        let L = encoded.dim(3)
        normed = normed.reshaped(B, T, D * L)
        let mask3d = attentionMask.asType(.bool).reshaped(B, T, 1)
        normed = which(mask3d, normed, MLXArray.zeros(like: normed))
        normed = normed.asType(origDtype)
        let rescaled = normed * MLXArray(rescale, dtype: origDtype)
        return audioAggregateEmbed(rescaled)
    }
}

final class DramaBoxBasicTransformerBlock1D: Module {
    @ModuleInfo var attn1: DramaBoxLTXAttention
    @ModuleInfo var ff: DramaBoxLTXFeedForward

    init(dim: Int, heads: Int, dimHead: Int) {
        self._attn1.wrappedValue = DramaBoxLTXAttention(
            queryDim: dim,
            heads: heads,
            dimHead: dimHead,
            applyGatedAttention: true,
            ropeType: "split"
        )
        self._ff.wrappedValue = DramaBoxLTXFeedForward(dim, dimOut: dim)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        ropeCosSin: (MLXArray, MLXArray)?
    ) -> MLXArray {
        var h = hiddenStates
        var norm = dramaBoxFunctionalRMSNorm(h)
        h = attn1(norm, mask: attentionMask, ropeCosSin: ropeCosSin) + h
        norm = dramaBoxFunctionalRMSNorm(h)
        return ff(norm) + h
    }
}

final class DramaBoxEmbeddings1DConnector: Module {
    let numAttentionHeads: Int
    let attentionHeadDim: Int
    let innerDim: Int
    let numLayers: Int
    let numLearnableRegisters: Int
    let positionalEmbeddingTheta: Float
    let positionalEmbeddingMaxPos: Int
    let seqLen: Int

    @ModuleInfo(key: "learnable_registers") var learnableRegisters: MLXArray
    @ModuleInfo(key: "transformer_1d_blocks") var transformer1dBlocks: [DramaBoxBasicTransformerBlock1D]

    init(
        numAttentionHeads: Int = 32,
        attentionHeadDim: Int = 64,
        numLayers: Int = 8,
        numLearnableRegisters: Int = 128,
        positionalEmbeddingTheta: Float = 10_000,
        positionalEmbeddingMaxPos: Int = 4096,
        seqLen: Int = 1024
    ) {
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        let dim = numAttentionHeads * attentionHeadDim
        self.innerDim = dim
        self.numLayers = numLayers
        self.numLearnableRegisters = numLearnableRegisters
        self.positionalEmbeddingTheta = positionalEmbeddingTheta
        self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos
        self.seqLen = seqLen
        self._learnableRegisters.wrappedValue = MLXArray.zeros([numLearnableRegisters, dim], dtype: .bfloat16)
        self._transformer1dBlocks.wrappedValue = (0..<numLayers).map { _ in
            DramaBoxBasicTransformerBlock1D(dim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim)
        }
        super.init()
    }

    func packValidToFront(_ hiddenStates: MLXArray, binaryMask: MLXArray) -> MLXArray {
        let T = binaryMask.dim(1)
        let position = MLXArray(Array(0..<T).map { Int32($0) }).reshaped([1, T])
        let key = binaryMask * Int32(T + 1) - position
        let order = argSort(-key, axis: 1)
        var packed = takeAlong(hiddenStates, order.asType(.int32).expandedDimensions(axis: 2), axis: 1)
        let numValid = MLX.sum(binaryMask, axis: 1, keepDims: true)
        let validMask = (MLXArray(Array(0..<T).map { Int32($0) }).reshaped([1, T]) .< numValid)
            .asType(packed.dtype)
        packed = packed * validMask.expandedDimensions(axis: 2)
        return packed
    }

    func replacePaddedWithRegisters(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(1)
        let D = hiddenStates.dim(2)
        let numDup = T / numLearnableRegisters
        var tiledRegisters = tiled(learnableRegisters, repetitions: [numDup, 1])
        tiledRegisters = MLX.broadcast(tiledRegisters, to: [B, T, D])
        let maskBT = (attentionMask.reshaped(B, T) .>= MLXArray(Float(-9000))).asType(.int32)
        let adjusted = packValidToFront(hiddenStates, binaryMask: maskBT)
        let flipped = maskBT[0..., .stride(by: -1)]
        let flipped3d = flipped.reshaped(B, T, 1).asType(hiddenStates.dtype)
        let out = flipped3d * adjusted + (1 - flipped3d) * tiledRegisters.asType(hiddenStates.dtype)
        return (out, MLXArray.zeros(like: attentionMask))
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray) -> (MLXArray, MLXArray) {
        var h = hiddenStates
        var mask = attentionMask
        if numLearnableRegisters > 0 {
            (h, mask) = replacePaddedWithRegisters(h, attentionMask: mask)
        }
        let T = h.dim(1)
        let rope = dramaBoxPrecomputeSplitFreqs1D(
            seqLen: T,
            innerDim: innerDim,
            numHeads: numAttentionHeads,
            theta: positionalEmbeddingTheta,
            maxPos: positionalEmbeddingMaxPos,
            outDtype: h.dtype
        )
        for block in transformer1dBlocks {
            h = block(h, attentionMask: mask, ropeCosSin: rope)
        }
        h = dramaBoxFunctionalRMSNorm(h)
        return (h, mask)
    }
}

struct DramaBoxEmbeddingsProcessorOutput {
    var audioEncoding: MLXArray
    var attentionMask: MLXArray
}

final class DramaBoxEmbeddingsProcessor: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: DramaBoxFeatureExtractor
    @ModuleInfo(key: "audio_connector") var audioConnector: DramaBoxEmbeddings1DConnector

    init(
        featureExtractor: DramaBoxFeatureExtractor,
        audioConnector: DramaBoxEmbeddings1DConnector
    ) {
        self._featureExtractor.wrappedValue = featureExtractor
        self._audioConnector.wrappedValue = audioConnector
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: [MLXArray],
        attentionMask: MLXArray
    ) -> DramaBoxEmbeddingsProcessorOutput {
        let audioFeats = featureExtractor(hiddenStates, attentionMask: attentionMask)
        let additive = dramaBoxConvertToAdditiveMask(attentionMask, dtype: audioFeats.dtype)
        let (audioEncoded, postMask) = audioConnector(audioFeats, attentionMask: additive)
        let B = postMask.dim(0)
        let T = postMask.dim(3)
        return DramaBoxEmbeddingsProcessorOutput(
            audioEncoding: audioEncoded,
            attentionMask: MLXArray.ones([B, T], type: Int32.self)
        )
    }
}

func loadDramaBoxFeatureExtractorWeights(
    _ model: DramaBoxFeatureExtractor,
    state: [String: MLXArray]
) throws {
    try dramaBoxLoadModuleWeights(model, state: state, prefix: "text_embedding_projection.")
}

func loadDramaBoxConnectorWeights(
    _ model: DramaBoxEmbeddings1DConnector,
    state: [String: MLXArray]
) throws {
    try dramaBoxLoadModuleWeights(
        model,
        state: state,
        prefix: "model.diffusion_model.audio_embeddings_connector."
    )
}
