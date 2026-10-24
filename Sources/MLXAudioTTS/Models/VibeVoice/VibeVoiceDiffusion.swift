// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXNN

final class VibeVoiceSpeechConnector: Module {
    let fc1: Linear
    let norm: RMSNorm
    let fc2: Linear
    init(input: Int, output: Int, eps: Float = 1e-6) {
        fc1 = Linear(input, output)
        norm = RMSNorm(dimensions: output, eps: eps)
        fc2 = Linear(output, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(norm(fc1(x))) }
}
final class VibeVoiceBinaryClassifier: Module {
    let fc1: Linear
    let fc2: Linear
    init(_ dim: Int) { fc1 = Linear(dim, dim); fc2 = Linear(dim, 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(relu(fc1(x))) }
}
final class VibeVoiceTimestepMLP: Module {
    @ModuleInfo(key: "0") var fc1: Linear
    @ModuleInfo(key: "2") var fc2: Linear
    init(_ dim: Int) {
        _fc1.wrappedValue = Linear(256, dim, bias: false)
        _fc2.wrappedValue = Linear(dim, dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(silu(fc1(x))) }
}
final class VibeVoiceTimestepEmbedder: Module {
    let mlp: VibeVoiceTimestepMLP
    init(_ dim: Int) { mlp = VibeVoiceTimestepMLP(dim) }
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let frequencies = exp(MLXArray(0..<128).asType(.float32) * (-log(Float(10000)) / 128))
        let angles = t.asType(.float32).expandedDimensions(axis: -1) * frequencies
        return mlp(concatenated([cos(angles), sin(angles)], axis: -1).asType(t.dtype))
    }
}
final class VibeVoiceDiffusionFFN: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ dim: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(dim, hidden, bias: false)
        _up.wrappedValue = Linear(dim, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}
final class VibeVoiceAdaLN: Module {
    @ModuleInfo(key: "1") var linear: Linear
    init(_ dim: Int, multiplier: Int) { _linear.wrappedValue = Linear(dim, dim * multiplier, bias: false) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(silu(x)) }
}
final class VibeVoiceDiffusionLayer: Module {
    let norm: RMSNorm
    let ffn: VibeVoiceDiffusionFFN
    @ModuleInfo(key: "adaLN_modulation") var modulation: VibeVoiceAdaLN
    init(_ c: VibeVoiceDiffusionConfig) {
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        ffn = VibeVoiceDiffusionFFN(c.hiddenSize, Int(Float(c.hiddenSize) * c.headFFNRatio))
        _modulation.wrappedValue = VibeVoiceAdaLN(c.hiddenSize, multiplier: 3)
    }
    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        let parts = split(modulation(condition), parts: 3, axis: -1)
        return x + parts[2] * ffn(norm(x) * (1 + parts[1]) + parts[0])
    }
}
final class VibeVoiceDiffusionFinal: Module {
    let linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var modulation: VibeVoiceAdaLN
    let epsilon: Float
    init(_ c: VibeVoiceDiffusionConfig) {
        linear = Linear(c.hiddenSize, c.latentSize, bias: false)
        epsilon = c.rmsNormEps
        _modulation.wrappedValue = VibeVoiceAdaLN(c.hiddenSize, multiplier: 2)
    }
    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        let parts = split(modulation(condition), parts: 2, axis: -1)
        let full = x.asType(.float32)
        let norm = (full * rsqrt(mean(full * full, axis: -1, keepDims: true) + epsilon)).asType(x.dtype)
        return linear(norm * (1 + parts[1]) + parts[0])
    }
}
final class VibeVoiceDiffusionHead: Module {
    @ModuleInfo(key: "noisy_images_proj") var noisyProjection: Linear
    @ModuleInfo(key: "cond_proj") var conditionProjection: Linear
    @ModuleInfo(key: "t_embedder") var timestepEmbedder: VibeVoiceTimestepEmbedder
    let layers: [VibeVoiceDiffusionLayer]
    @ModuleInfo(key: "final_layer") var finalLayer: VibeVoiceDiffusionFinal
    init(_ c: VibeVoiceDiffusionConfig) {
        _noisyProjection.wrappedValue = Linear(c.latentSize, c.hiddenSize, bias: false)
        _conditionProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _timestepEmbedder.wrappedValue = VibeVoiceTimestepEmbedder(c.hiddenSize)
        layers = (0..<c.headLayers).map { _ in VibeVoiceDiffusionLayer(c) }
        _finalLayer.wrappedValue = VibeVoiceDiffusionFinal(c)
    }
    func projectCondition(_ x: MLXArray) -> MLXArray { conditionProjection(x) }
    func embedTimesteps(_ t: MLXArray) -> MLXArray { timestepEmbedder(t) }
    func forward(_ noisy: MLXArray, conditionWithTimestep: MLXArray) -> MLXArray {
        var x = noisyProjection(noisy)
        for layer in layers { x = layer(x, condition: conditionWithTimestep) }
        return finalLayer(x, condition: conditionWithTimestep)
    }
    func callAsFunction(_ noisy: MLXArray, timesteps: MLXArray, condition: MLXArray) -> MLXArray {
        forward(noisy, conditionWithTimestep: projectCondition(condition) + embedTimesteps(timesteps))
    }
}

/// Deterministic DPM-Solver++ second-order midpoint updates, with a zero final sigma.
final class VibeVoiceDPMSolver {
    let trainingAlpha: [Float]
    let predictionType: String
    private(set) var timesteps: [Int] = []
    private(set) var alpha: [Float] = []
    private(set) var sigma: [Float] = []
    private(set) var lambda: [Float] = []
    private var index = 0
    private var previous: MLXArray?
    init(trainingSteps: Int = 1000, betaSchedule: String = "cosine", predictionType: String = "v_prediction") {
        self.predictionType = predictionType
        var betas: [Float] = []
        func alphaBar(_ t: Double) -> Double { pow(cos((t + 0.008) / 1.008 * .pi / 2), 2) }
        for i in 0..<trainingSteps {
            if betaSchedule == "linear" { betas.append(0.0001 + Float(i) / Float(trainingSteps - 1) * 0.0199) }
            else { betas.append(Float(min(1 - alphaBar(Double(i + 1) / Double(trainingSteps)) / alphaBar(Double(i) / Double(trainingSteps)), 0.999))) }
        }
        // Match the reference's float32 cumulative product before scalar schedule construction.
        trainingAlpha = sqrt(cumprod(1 - MLXArray(betas))).asArray(Float.self)
    }
    func setTimesteps(_ steps: Int) {
        precondition(steps > 0 && steps <= trainingAlpha.count)
        timesteps = (0..<steps).map { Int((Double(trainingAlpha.count - 1) * (1 - Double($0) / Double(steps))).rounded(.toNearestOrEven)) }
        alpha = []; sigma = []; lambda = []
        for t in timesteps {
            let a = Double(trainingAlpha[t])
            let s = sqrt((1 - a * a) / (a * a))
            let currentAlpha = 1 / sqrt(s * s + 1)
            let currentSigma = s * currentAlpha
            alpha.append(Float(currentAlpha)); sigma.append(Float(currentSigma))
            lambda.append(Float(log(currentAlpha) - log(currentSigma)))
        }
        alpha.append(1); sigma.append(0); lambda.append(.infinity)
        reset()
    }
    func reset() { index = 0; previous = nil }
    func step(_ output: MLXArray, sample: MLXArray) -> MLXArray {
        let i = index
        precondition(i < timesteps.count)
        let x = sample.asType(.float32)
        let model = output.asType(.float32)
        let clean: MLXArray
        switch predictionType {
        case "epsilon": clean = (x - sigma[i] * model) / alpha[i]
        case "sample": clean = model
        default: clean = alpha[i] * x - sigma[i] * model
        }
        let h = lambda[i + 1] - lambda[i]
        let coefficient = -alpha[i + 1] * (Foundation.exp(-h) - 1)
        var result = (sigma[i + 1] / sigma[i]) * x + coefficient * clean
        if i > 0, i < timesteps.count - 1, let previous {
            let ratio = (lambda[i] - lambda[i - 1]) / h
            if ratio != 0 { result = result + (0.5 * coefficient / ratio) * (clean - previous) }
        }
        previous = clean
        index += 1
        return result.asType(output.dtype)
    }
}
