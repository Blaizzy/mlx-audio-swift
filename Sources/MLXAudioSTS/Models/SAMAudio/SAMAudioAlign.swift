import Foundation
import MLX
import MLXNN

/// Aligns projected features from another modality to an anchor sequence.
public final class AlignModalities: Module {
    let outChannels: Int
    let normalize: Bool

    @ModuleInfo(key: "conv_weight") var convWeight: MLXArray
    @ModuleInfo(key: "conv_bias") var convBias: MLXArray
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "gate") var gate: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        normalize: Bool = true,
        withGate: Bool = true
    ) {
        self.outChannels = outChannels
        self.normalize = normalize
        self._convWeight.wrappedValue = MLXArray.zeros([outChannels, 1, inChannels])
        self._convBias.wrappedValue = MLXArray.zeros([outChannels])
        self._layerNorm.wrappedValue = normalize ? LayerNorm(dimensions: outChannels) : nil
        self._gate.wrappedValue = withGate ? MLXArray([Float(0)]) : nil
    }

    public func callAsFunction(_ anchor: MLXArray, tgt: MLXArray? = nil) -> MLXArray {
        guard let tgt else {
            return anchor
        }

        let tgtT = tgt.transposed(0, 2, 1)  // (B, T, C_in)
        var postConv = MLX.conv1d(tgtT, convWeight)
        postConv = postConv + convBias

        if normalize, let layerNorm {
            postConv = layerNorm(postConv)
        }

        guard let gate else {
            return postConv
        }

        return anchor + MLX.tanh(gate) * postConv
    }
}

/// Embeds temporal anchors and injects them into time-aligned features.
public final class EmbedAnchors: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "gate") var gate: MLXArray
    @ModuleInfo(key: "proj") var proj: Linear

    public init(
        numEmbeddings: Int,
        embeddingDim: Int,
        outDim: Int
    ) {
        self._embed.wrappedValue = Embedding(embeddingCount: numEmbeddings + 1, dimensions: embeddingDim)
        self._gate.wrappedValue = MLXArray([Float(0)])
        self._proj.wrappedValue = Linear(embeddingDim, outDim, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil
    ) -> MLXArray {
        guard let anchorIDs, let anchorAlignment else {
            return x
        }

        let gatheredIDs = MLX.takeAlong(anchorIDs, anchorAlignment, axis: 1)
        let embs = embed(gatheredIDs)
        let projected = proj(embs)
        return x + MLX.tanh(gate) * projected
    }
}
