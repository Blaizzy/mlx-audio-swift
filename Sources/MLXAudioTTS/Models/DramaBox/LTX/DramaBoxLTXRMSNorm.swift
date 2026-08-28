import Foundation
@preconcurrency import MLX

/// Functional RMSNorm used by connector blocks and DiT AdaLN pre-norms.
/// No learnable weight unless supplied. Compute in fp32.
func dramaBoxFunctionalRMSNorm(_ x: MLXArray, eps: Float = 1e-6, weight: MLXArray? = nil) -> MLXArray {
    let origDtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    var out = x32 * MLX.rsqrt(variance + eps)
    if let weight {
        out = out * weight.asType(.float32)
    }
    return out.asType(origDtype)
}
