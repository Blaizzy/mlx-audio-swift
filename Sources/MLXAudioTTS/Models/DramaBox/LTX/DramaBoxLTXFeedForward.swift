import Foundation
@preconcurrency import MLX
import MLXNN

final class DramaBoxGELUApprox: Module {
    @ModuleInfo var proj: Linear

    init(dimIn: Int, dimOut: Int) {
        self._proj.wrappedValue = Linear(dimIn, dimOut, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXNN.geluApproximate(proj(x))
    }
}

final class DramaBoxLTXFeedForward: Module {
    /// Python `net = [GELUApprox, Identity, Linear]` → `net.0.proj.*` and `net.2.*`.
    @ModuleInfo var net: [Module]

    init(_ dim: Int, dimOut: Int, mult: Int = 4) {
        let inner = dim * mult
        self._net.wrappedValue = [
            DramaBoxGELUApprox(dimIn: dim, dimOut: inner),
            Identity(),
            Linear(inner, dimOut, bias: true),
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gelu = net[0] as! DramaBoxGELUApprox
        let projOut = net[2] as! Linear
        return projOut(gelu(x))
    }
}
