import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - SineGenerator

public class SineGenerator {
    public let sineAmp: Float
    public let noiseStd: Float
    public let harmonicNum: Int
    public let samplingRate: Int
    public let voicedThreshold: Float
    public let upsampleScale: Int

    public init(sampRate: Int, upsampleScale: Int, harmonicNum: Int = 0,
                sineAmp: Float = 0.1, noiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.upsampleScale = upsampleScale
    }

    public func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let harmonics = MLXArray(Array((1...Int32(harmonicNum + 1)).map { Float($0) }))
            .reshaped([1, 1, harmonicNum + 1])
        let fn = f0 * harmonics

        let radValues = (fn / Float(samplingRate)) % MLXArray(Float(1))

        let randIni = MLXRandom.normal([f0.shape[0], harmonicNum + 1])
        randIni[0..., 0] = MLXArray(Float(0))
        radValues[0..., 0, 0...] = radValues[0..., 0, 0...] + randIni

        let downscale = 1.0 / Float(upsampleScale)
        let downSize = max(1, Int(ceil(Float(radValues.shape[1]) * downscale)))
        let radDown = interpolate1d(radValues.transposed(0, 2, 1), size: downSize).transposed(0, 2, 1)
        let phaseDown = MLX.cumsum(radDown, axis: 1) * (2 * Float.pi)
        let phaseScaled = phaseDown.transposed(0, 2, 1) * Float(upsampleScale)
        let phase = interpolate1d(phaseScaled, size: radValues.shape[1]).transposed(0, 2, 1)

        let sineWaves = MLX.sin(phase) * sineAmp
        let uv = (f0 .> Float(voicedThreshold)).asType(.float32)
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)
        let result = sineWaves * uv + noise
        return (result, uv, noise)
    }
}

// MARK: - SourceModule

public class SourceModule: Module {
    public let sineGen: SineGenerator
    @ModuleInfo(key: "l_linear") public var lLinear: Linear

    public init(samplingRate: Int, upsampleScale: Int, harmonicNum: Int = 0,
                sineAmp: Float = 0.1, addNoiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        sineGen = SineGenerator(sampRate: samplingRate, upsampleScale: upsampleScale,
                                harmonicNum: harmonicNum, sineAmp: sineAmp,
                                noiseStd: addNoiseStd, voicedThreshold: voicedThreshold)
        _lLinear = ModuleInfo(wrappedValue: Linear(harmonicNum + 1, 1), key: "l_linear")
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (sineWavs, uv, _) = sineGen(x)
        let sineMerge = tanh(lLinear(sineWavs))
        let noise = MLXRandom.normal(uv.shape) * sineGen.sineAmp / 3
        return (sineMerge, noise, uv)
    }
}
