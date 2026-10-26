// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXRandom

/// Stochastic second-order midpoint SDE-DPM-Solver++, with a zero final sigma.
final class KugelAudioSDESolver {
    private let schedule: VibeVoiceDPMSolver
    private var index = 0
    private var previous: MLXArray?
    var timesteps: [Int] { schedule.timesteps }
    init(_ config: VibeVoiceDiffusionConfig, steps: Int) {
        schedule = VibeVoiceDPMSolver(trainingSteps: config.ddpmNumSteps,
            betaSchedule: config.ddpmBetaSchedule, predictionType: config.predictionType)
        schedule.setTimesteps(steps)
    }
    func reset() { index = 0; previous = nil }
    func step(_ output: MLXArray, sample: MLXArray, noise: MLXArray? = nil) -> MLXArray {
        let i = index
        precondition(i < timesteps.count)
        let x = sample.asType(.float32), output = output.asType(.float32)
        let clean: MLXArray
        switch schedule.predictionType {
        case "epsilon": clean = (x - schedule.sigma[i] * output) / schedule.alpha[i]
        case "sample": clean = output
        default: clean = schedule.alpha[i] * x - schedule.sigma[i] * output
        }
        let h = schedule.lambda[i + 1] - schedule.lambda[i]
        let decay = Foundation.exp(-h)
        let coefficient = schedule.alpha[i + 1] * (1 - decay * decay)
        var result = (schedule.sigma[i + 1] / schedule.sigma[i] * decay) * x + coefficient * clean
        if i > 0, i < timesteps.count - 1, let previous {
            let ratio = (schedule.lambda[i] - schedule.lambda[i - 1]) / h
            if ratio != 0 { result = result + (0.5 * coefficient / ratio) * (clean - previous) }
        }
        result = result + (schedule.sigma[i + 1] * sqrt(1 - decay * decay))
            * (noise ?? MLXRandom.normal(sample.shape)).asType(.float32)
        previous = clean
        index += 1
        return result
    }
}
