import Foundation
@preconcurrency import MLX

struct DramaBoxLTX2Scheduler {
    func execute(
        steps: Int,
        tokens: Int,
        maxShift: Float = 2.05,
        baseShift: Float = 0.95,
        stretch: Bool = true,
        terminal: Float = 0.1
    ) -> MLXArray {
        var sigmas = MLX.linspace(Float32(1), Float32(0), count: steps + 1).asType(.float32)
        let mm = (maxShift - baseShift) / (4096.0 - 1024.0)
        let b = baseShift - mm * 1024.0
        let sigmaShift = Float(tokens) * mm + b
        let expShift = exp(sigmaShift)
        let nonzero = sigmas .!= MLXArray(0 as Float)
        let safe = which(nonzero, sigmas, MLXArray(Float(1e-12)))
        let ratio = 1.0 / safe - 1.0
        let shifted = MLXArray(expShift) / (MLXArray(expShift) + ratio)
        sigmas = which(nonzero, shifted, MLXArray(0 as Float))
        if stretch {
            let oneMinus = 1.0 - sigmas
            let anchor = oneMinus[steps - 1]
            let scaleFactor = anchor / (1.0 - terminal)
            let stretched = 1.0 - oneMinus / scaleFactor
            sigmas = which(nonzero, stretched, MLXArray(0 as Float))
        }
        return sigmas.asType(.float32)
    }
}
