#if canImport(Accelerate)
import Accelerate
#endif

/// Computes `out[0..<n] = vec[0..<p] · mat[p×n]` (row-major), i.e. a 1×P by P×N matrix
/// product. This is the recurrent hidden-state projection in DeepFilterNet's GRU loop.
///
/// On Apple platforms this dispatches to Accelerate's `vDSP_mmul` (with M=1). On other
/// platforms (Linux) it falls back to a straightforward pure-Swift accumulation, which is
/// adequate because the projection runs on small hidden dimensions.
@inline(__always)
func dfnHiddenMatVec(
    _ vec: UnsafePointer<Float>,
    _ mat: UnsafePointer<Float>,
    _ out: UnsafeMutablePointer<Float>,
    n: Int,
    p: Int
) {
    #if canImport(Accelerate)
    vDSP_mmul(vec, 1, mat, 1, out, 1, vDSP_Length(1), vDSP_Length(n), vDSP_Length(p))
    #else
    for j in 0 ..< n { out[j] = 0 }
    for k in 0 ..< p {
        let a = vec[k]
        if a == 0 { continue }
        let row = mat + k * n
        for j in 0 ..< n {
            out[j] += a * row[j]
        }
    }
    #endif
}
