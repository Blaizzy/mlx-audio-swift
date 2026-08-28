import Foundation
@preconcurrency import MLX
import MLXNN

final class DramaBoxLTXBlock: Module {
    let dim: Int
    let normEps: Float

    @ModuleInfo(key: "audio_attn1") var audioAttn1: DramaBoxLTXAttention
    @ModuleInfo(key: "audio_attn2") var audioAttn2: DramaBoxLTXAttention
    @ModuleInfo(key: "audio_ff") var audioFF: DramaBoxLTXFeedForward
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_prompt_scale_shift_table") var audioPromptScaleShiftTable: MLXArray

    init(
        dim: Int,
        heads: Int,
        dimHead: Int,
        contextDim: Int,
        applyGatedAttention: Bool = true,
        normEps: Float = 1e-6,
        ropeType: String = "split"
    ) {
        self.dim = dim
        self.normEps = normEps
        self._audioAttn1.wrappedValue = DramaBoxLTXAttention(
            queryDim: dim,
            heads: heads,
            dimHead: dimHead,
            applyGatedAttention: applyGatedAttention,
            ropeType: ropeType
        )
        self._audioAttn2.wrappedValue = DramaBoxLTXAttention(
            queryDim: dim,
            heads: heads,
            dimHead: dimHead,
            contextDim: contextDim,
            applyGatedAttention: applyGatedAttention,
            ropeType: ropeType
        )
        self._audioFF.wrappedValue = DramaBoxLTXFeedForward(dim, dimOut: dim, mult: 4)
        self._audioScaleShiftTable.wrappedValue = MLXArray.zeros([9, dim])
        self._audioPromptScaleShiftTable.wrappedValue = MLXArray.zeros([2, dim])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        adaEmb: MLXArray,
        promptAdaEmb: MLXArray,
        context: MLXArray,
        ropeCosSin: (MLXArray, MLXArray)?,
        selfAttentionMask: MLXArray? = nil,
        skipSelfAttn: Bool = false
    ) -> MLXArray {
        let B = x.dim(0)
        var hidden = x
        let ada: MLXArray
        if adaEmb.ndim == 2 {
            ada = adaEmb.reshaped(B, 1, 9, dim)
                + audioScaleShiftTable.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        } else {
            ada = adaEmb.reshaped(B, adaEmb.dim(1), 9, dim)
                + audioScaleShiftTable.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        }

        let shiftMSA = ada[0..., 0..., 0, 0...]
        let scaleMSA = ada[0..., 0..., 1, 0...]
        let gateMSA = ada[0..., 0..., 2, 0...]
        var h = dramaBoxFunctionalRMSNorm(hidden, eps: normEps) * (1 + scaleMSA) + shiftMSA
        hidden = hidden + audioAttn1(
            h, mask: selfAttentionMask, ropeCosSin: ropeCosSin, skipSelfAttn: skipSelfAttn
        ) * gateMSA

        let shiftQ = ada[0..., 0..., 6, 0...]
        let scaleQ = ada[0..., 0..., 7, 0...]
        let gate = ada[0..., 0..., 8, 0...]
        let ctxPA = audioPromptScaleShiftTable.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            + promptAdaEmb.reshaped(B, 1, 2, dim)
        let shiftKV = ctxPA[0..., 0..., 0, 0...]
        let scaleKV = ctxPA[0..., 0..., 1, 0...]
        let attnInput = dramaBoxFunctionalRMSNorm(hidden, eps: normEps) * (1 + scaleQ) + shiftQ
        let encoderHS = context * (1 + scaleKV) + shiftKV
        hidden = hidden + audioAttn2(attnInput, context: encoderHS) * gate

        let shiftMLP = ada[0..., 0..., 3, 0...]
        let scaleMLP = ada[0..., 0..., 4, 0...]
        let gateMLP = ada[0..., 0..., 5, 0...]
        h = dramaBoxFunctionalRMSNorm(hidden, eps: normEps) * (1 + scaleMLP) + shiftMLP
        hidden = hidden + audioFF(h) * gateMLP
        return hidden
    }
}
