import Foundation
import MLX
import MLXNN

/// Tiny keeps the Euler state and all random draws in FP32, including BF16 checkpoints.
final class MingTinyFlow: Module {
    let cfm: MingCFM
    let latentDim: Int
    init(latentDim: Int, conditionDim: Int, config: MingTransformerConfiguration) {
        self.latentDim = latentDim
        cfm = MingCFM(MingDiT(latentDim: latentDim, conditionDim: conditionDim, config: config))
    }
    func sample(_ hidden: MLXArray, history: MLXArray, patchSize: Int, steps: Int,
                cfg: Float, sigma: Float, temperature: Float,
                noise: MLXArray? = nil, sdeNoise: MLXArray? = nil) -> MLXArray {
        var state = (noise ?? MLXRandom.normal([hidden.dim(0), latentDim, patchSize], dtype: .float32)
            .transposed(0, 2, 1)).asType(.float32)
        var times = mingFlowTimesteps(steps)
        times = times - (cos(Float.pi / 2 * times) - 1 + times)
        let dtype = cfm.model.dtype
        for index in 0..<steps {
            let current = state.asType(dtype), condition = hidden.asType(dtype), past = history.asType(dtype)
            let time = times[index].asType(dtype)
            let prediction: MLXArray
            if cfg == 1 || cfg < 1e-5 {
                let t = MLXArray.ones([current.dim(0)], dtype: dtype) * time
                let result = cfm.model(current, time: t,
                    condition: cfg == 1 ? condition : MLXArray.zeros(like: condition), history: past)
                prediction = result[0..., (result.dim(1) - patchSize)..., 0...]
            } else {
                let (conditional, unconditional) = cfm.model.guided(current, time: time,
                    condition: condition, history: past, patchSize: patchSize)
                prediction = conditional + (conditional - unconditional) * cfg
            }
            let dt = times[index + 1] - times[index]
            state = state + dt * prediction.asType(.float32)
            // The reference draws noise even on the last step, but returns the final Euler value.
            let perturbation = sdeNoise?[index] ?? MLXRandom.normal(state.shape, dtype: .float32)
            if index < steps - 1 {
                state = state + sigma * sqrt(temperature) * sqrt(abs(dt)) * perturbation
            }
            eval(state, perturbation)
        }
        return state
    }
}
