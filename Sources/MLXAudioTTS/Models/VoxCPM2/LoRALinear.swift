// LoRA‑wrapped Linear layer for VoxCPM2 fine‑tuning.
//
// Wraps a frozen `Linear` base layer and adds trainable low‑rank matrices
// A (in_dim × r) and B (out_dim × r). The forward pass computes:
//
//     output = base(x) + (x · Aᵀ · Bᵀ) × (α / r)
//
// The LoRA branch can be toggled on/off at runtime via `enabled`.

import Foundation
import MLX
import MLXNN

/// A Linear layer augmented with a LoRA low‑rank adapter.
///
/// The base weights are frozen — only the LoRA matrices `loraA` and `loraB`
/// are intended for training.  The branch is controlled by `enabled` so the
/// same model can switch between fine‑tuned and base behaviour without
/// reloading weights.
public final class LoRALinear: Module, UnaryLayer {
    /// The original frozen linear layer.
    @ModuleInfo public var base: Linear

    /// LoRA down‑projection: `inDim → r`
    public var loraA: MLXArray

    /// LoRA up‑projection: `r → outDim`
    public var loraB: MLXArray

    /// Rank of the low‑rank matrices.
    public let r: Int

    /// Scaling factor.  Effective scale = `alpha / Float(r)`.
    public let alpha: Float

    /// Whether the LoRA branch is active.
    public var enabled: Bool = true

    /// Wrap an existing `Linear` layer with LoRA matrices.
    ///
    /// - Parameters:
    ///   - base: The linear layer to wrap (its weights are frozen).
    ///   - r: LoRA rank.
    ///   - alpha: Scaling factor (effective scale = alpha / r).
    public init(base: Linear, r: Int = 8, alpha: Int = 16) {
        self.r = r
        self.alpha = Float(alpha)

        let wShape = base.weight.shape
        let inDim = wShape[1]
        let outDim = wShape[0]

        self._base = ModuleInfo(wrappedValue: base, key: "base")

        // Initialise A with Kaiming‑uniform and B with zeros (the standard
        // LoRA init so that the branch starts at zero).
        let scale = sqrt(1.0 / Float(inDim))
        let aData = (MLXRandom.uniform(low: 0.0, high: 1.0, [inDim, r]) * 2.0 - 1.0) * scale
        self.loraA = aData
        self.loraB = MLXArray.zeros([outDim, r])

        super.init()
    }

    /// Convenience: wrap a Linear with an explicit LoRAConfig.
    public convenience init(base: Linear, config: LoRAConfig) {
        self.init(base: base, r: config.r, alpha: config.alpha)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let baseOut = base(x)
        guard enabled else { return baseOut }

        // LoRA path: x · A · Bᵀ  (not Aᵀ! MLX stores weight in (out, in) layout,
        // matching matmul convention where x is (..., in))
        let loraOut = x.matmul(loraA).matmul(loraB.T)
        let scale = alpha / Float(r)
        return baseOut + loraOut * scale
    }
}
