import Foundation
@preconcurrency import MLX
@preconcurrency import MLXFast
import MLXNN

final class DramaBoxLTXInnerRMSNorm: Module {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(_ dim: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        dramaBoxFunctionalRMSNorm(x, eps: eps, weight: weight)
    }
}

final class DramaBoxLTXAttention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let scale: Float
    let ropeType: String

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "q_norm") var qNorm: DramaBoxLTXInnerRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: DramaBoxLTXInnerRMSNorm
    @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear?
    /// Python `to_out = [Linear, Identity]` → keys `to_out.0.{weight,bias}`.
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    init(
        queryDim: Int,
        heads: Int,
        dimHead: Int,
        contextDim: Int? = nil,
        normEps: Float = 1e-6,
        applyGatedAttention: Bool = false,
        ropeType: String = "split"
    ) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.scale = pow(Float(dimHead), -0.5)
        self.ropeType = ropeType
        let ctx = contextDim ?? queryDim
        self._toQ.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(ctx, innerDim, bias: true)
        self._toV.wrappedValue = Linear(ctx, innerDim, bias: true)
        self._qNorm.wrappedValue = DramaBoxLTXInnerRMSNorm(innerDim, eps: normEps)
        self._kNorm.wrappedValue = DramaBoxLTXInnerRMSNorm(innerDim, eps: normEps)
        if applyGatedAttention {
            self._toGateLogits.wrappedValue = Linear(queryDim, heads, bias: true)
        } else {
            self._toGateLogits.wrappedValue = nil
        }
        self._toOut.wrappedValue = [Linear(innerDim, queryDim, bias: true)]
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray? = nil,
        mask: MLXArray? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        skipSelfAttn: Bool = false
    ) -> MLXArray {
        let ctx = context ?? x
        let B: Int
        let Tq: Int
        var out: MLXArray

        if skipSelfAttn {
            out = toV(ctx)
            B = out.dim(0)
            Tq = out.dim(1)
        } else {
            var q = toQ(x)
            var k = toK(ctx)
            let v = toV(ctx)
            q = qNorm(q)
            k = kNorm(k)
            if let (cos, sin) = ropeCosSin, ropeType == "split" {
                q = dramaBoxApplySplitRope(q, cos: cos, sin: sin)
                k = dramaBoxApplySplitRope(k, cos: cos, sin: sin)
            }
            B = q.dim(0)
            Tq = q.dim(1)
            let Tk = k.dim(1)
            q = q.reshaped(B, Tq, heads, dimHead).transposed(0, 2, 1, 3)
            k = k.reshaped(B, Tk, heads, dimHead).transposed(0, 2, 1, 3)
            let vHeads = v.reshaped(B, Tk, heads, dimHead).transposed(0, 2, 1, 3)
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: vHeads, scale: scale, mask: mask
            )
            out = out.transposed(0, 2, 1, 3).reshaped(B, Tq, heads * dimHead)
        }

        if let toGateLogits {
            let gates = 2.0 * MLX.sigmoid(toGateLogits(x))
            out = out.reshaped(B, Tq, heads, dimHead) * gates.expandedDimensions(axis: -1)
            out = out.reshaped(B, Tq, heads * dimHead)
        }
        return toOut[0](out)
    }
}
