import Foundation
@preconcurrency import MLX
import MLXNN

final class DramaBoxLTXModel: Module {
    let config: DramaBoxDiTConfig

    @ModuleInfo(key: "audio_patchify_proj") var audioPatchifyProj: Linear
    @ModuleInfo(key: "audio_adaln_single") var audioAdalnSingle: DramaBoxAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") var audioPromptAdalnSingle: DramaBoxAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [DramaBoxLTXBlock]
    @ModuleInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear

    init(_ config: DramaBoxDiTConfig = DramaBoxDiTConfig()) {
        self.config = config
        let hidden = config.audioInnerDim
        self._audioPatchifyProj.wrappedValue = Linear(config.audioInChannels, hidden, bias: true)
        self._audioAdalnSingle.wrappedValue = DramaBoxAdaLayerNormSingle(hidden: hidden, coeff: 9)
        self._audioPromptAdalnSingle.wrappedValue = DramaBoxAdaLayerNormSingle(hidden: hidden, coeff: 2)
        self._transformerBlocks.wrappedValue = (0..<config.numLayers).map { _ in
            DramaBoxLTXBlock(
                dim: hidden,
                heads: config.audioNumAttentionHeads,
                dimHead: config.audioAttentionHeadDim,
                contextDim: config.audioCrossAttentionDim,
                applyGatedAttention: config.applyGatedAttention,
                normEps: config.normEps,
                ropeType: config.ropeType
            )
        }
        self._audioScaleShiftTable.wrappedValue = MLXArray.zeros([2, hidden])
        self._audioProjOut.wrappedValue = Linear(hidden, config.audioOutChannels, bias: true)
        super.init()
    }

    func normOut(_ x: MLXArray) -> MLXArray {
        let orig = x.dtype
        let x32 = x.asType(.float32)
        let mean = MLX.mean(x32, axis: -1, keepDims: true)
        let variance = MLX.mean((x32 - mean) * (x32 - mean), axis: -1, keepDims: true)
        return ((x32 - mean) * MLX.rsqrt(variance + config.normEps)).asType(orig)
    }

    func callAsFunction(
        _ x: MLXArray,
        aCtx: MLXArray,
        sigma: MLXArray,
        positions: MLXArray? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        attentionMask: MLXArray? = nil,
        denoiseMask: MLXArray? = nil,
        stgBlocks: Set<Int> = []
    ) -> MLXArray {
        let B = x.dim(0)
        var hidden = audioPatchifyProj(x)
        let rope: (MLXArray, MLXArray)
        if let ropeCosSin {
            rope = ropeCosSin
        } else if let positions {
            rope = dramaBoxPrecomputeSplitFreqsFromPositions(
                positions: positions,
                innerDim: config.audioInnerDim,
                numHeads: config.audioNumAttentionHeads,
                theta: config.positionalEmbeddingTheta,
                maxPos: Float(config.audioPositionalEmbeddingMaxPos),
                outDtype: hidden.dtype
            )
        } else {
            fatalError("DramaBox DiT requires positions or ropeCosSin")
        }

        let scalarScaled = sigma.asType(.float32) * config.timestepScaleMultiplier
        let mainScaled: MLXArray
        if let denoiseMask {
            var dm = denoiseMask
            if dm.ndim == 3 {
                dm = dm[.ellipsis, 0]
            }
            mainScaled = scalarScaled.expandedDimensions(axis: 1) * dm.asType(.float32)
        } else {
            mainScaled = scalarScaled
        }

        let (adaEmb, embeddedT) = audioAdalnSingle(mainScaled, dtype: hidden.dtype)
        let (promptAda, _) = audioPromptAdalnSingle(scalarScaled, dtype: hidden.dtype)

        for (idx, block) in transformerBlocks.enumerated() {
            hidden = block(
                hidden,
                adaEmb: adaEmb,
                promptAdaEmb: promptAda,
                context: aCtx,
                ropeCosSin: rope,
                selfAttentionMask: attentionMask,
                skipSelfAttn: stgBlocks.contains(idx)
            )
        }

        let bias = audioScaleShiftTable.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let embedded: MLXArray
        if embeddedT.ndim == 2 {
            embedded = embeddedT.reshaped(B, 1, 1, -1)
        } else {
            embedded = embeddedT.reshaped(B, embeddedT.dim(1), 1, -1)
        }
        let scaleShift = bias + embedded
        let shiftFinal = scaleShift[0..., 0..., 0, 0...]
        let scaleFinal = scaleShift[0..., 0..., 1, 0...]
        hidden = normOut(hidden)
        hidden = hidden * (1 + scaleFinal) + shiftFinal
        return audioProjOut(hidden)
    }
}

func loadDramaBoxDiTWeights(_ model: DramaBoxLTXModel, state: [String: MLXArray]) throws {
    let prefix = "model.diffusion_model."
    var sub: [String: MLXArray] = [:]
    for (key, value) in state where key.hasPrefix(prefix) {
        let tail = String(key.dropFirst(prefix.count))
        if tail.hasPrefix("audio_embeddings_connector") { continue }
        sub[tail] = value
    }
    guard !sub.isEmpty else {
        throw DramaBoxError.generationFailed("No DiT keys under \(prefix)")
    }
    try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
}
