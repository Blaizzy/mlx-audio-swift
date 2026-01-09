//
//  S3GenHiFiGAN.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom

private func s3GetPadding(kernelSize: Int, dilation: Int = 1) -> Int {
    Int((kernelSize * dilation - dilation) / 2)
}

private func s3Elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    MLX.where(x .> MLXArray(Float(0)), x, MLXArray(alpha) * (MLX.exp(x) - MLXArray(Float(1.0))))
}

private func s3HannWindowPeriodic(size: Int) -> [Float] {
    guard size > 0 else { return [] }
    var window = [Float](repeating: 0, count: size)
    let denom = Float(size)
    for n in 0..<size {
        window[n] = 0.5 - 0.5 * cosf(2.0 * Float.pi * Float(n) / denom)
    }
    return window
}

final class Snake: Module {
    let inFeatures: Int
    let alphaLogscale: Bool
    @ModuleInfo(key: "alpha") private var alpha: MLXArray

    init(inFeatures: Int, alpha: Float = 1.0, alphaTrainable: Bool = true, alphaLogscale: Bool = false) {
        self.inFeatures = inFeatures
        self.alphaLogscale = alphaLogscale
        if alphaLogscale {
            self._alpha.wrappedValue = MLXArray.zeros([inFeatures], type: Float.self) * MLXArray(alpha)
        } else {
            self._alpha.wrappedValue = MLXArray.ones([inFeatures]) * MLXArray(alpha)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var alpha = self.alpha.reshaped(1, -1, 1)
        if alphaLogscale {
            alpha = MLX.exp(alpha)
        }

        let noDiv: Float = 1e-9
        let minAlpha: Float = 1e-4

        let sign = MLX.sign(alpha)
        let absVal = MLX.abs(alpha)
        var clamped = sign * MLX.maximum(absVal, MLXArray(minAlpha))
        clamped = MLX.where(absVal .< MLXArray(noDiv), MLXArray(minAlpha), clamped)

        return x + (MLXArray(Float(1.0)) / clamped) * MLX.pow(MLX.sin(x * alpha), 2.0)
    }
}

final class S3ResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1dPT]
    @ModuleInfo(key: "convs2") var convs2: [Conv1dPT]
    @ModuleInfo(key: "activations1") var activations1: [Snake]
    @ModuleInfo(key: "activations2") var activations2: [Snake]

    init(channels: Int = 512, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
        self._convs1.wrappedValue = dilations.map { dilation in
            Conv1dPT(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                padding: s3GetPadding(kernelSize: kernelSize, dilation: dilation),
                dilation: dilation
            )
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1dPT(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                padding: s3GetPadding(kernelSize: kernelSize, dilation: 1)
            )
        }
        self._activations1.wrappedValue = dilations.map { _ in Snake(inFeatures: channels) }
        self._activations2.wrappedValue = dilations.map { _ in Snake(inFeatures: channels) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for idx in 0..<convs1.count {
            var xt = activations1[idx](h)
            xt = convs1[idx](xt)
            xt = activations2[idx](xt)
            xt = convs2[idx](xt)
            h = xt + h
        }
        return h
    }
}

final class SineGen: Module {
    let sineAmp: Float
    let noiseStd: Float
    let harmonicNum: Int
    let samplingRate: Int
    let voicedThreshold: Float

    init(sampRate: Int, harmonicNum: Int = 0, sineAmp: Float = 0.1, noiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let batch = f0.shape[0]
        let harmonics = MLXArray.arange(1.0, Double(harmonicNum + 2), step: 1.0, dtype: .float32)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 2)
        let fMat = f0 * harmonics / MLXArray(Float(samplingRate))
        var theta = MLXArray(Float(2.0 * Float.pi)) * MLX.cumsum(fMat, axis: -1)
        let twoPi = MLXArray(Float(2.0 * Float.pi))
        theta = theta - MLX.floor(theta / twoPi) * twoPi

        let phaseVec: MLXArray
        if harmonicNum > 0 {
            let randomPhases = MLXRandom.uniform(low: -Float.pi, high: Float.pi, [batch, harmonicNum, 1])
            let zeroPhase = MLXArray.zeros([batch, 1, 1], type: Float.self)
            phaseVec = MLX.concatenated([zeroPhase, randomPhases], axis: 1)
        } else {
            phaseVec = MLXArray.zeros([batch, 1, 1], type: Float.self)
        }

        var sineWaves = MLXArray(Float(sineAmp)) * MLX.sin(theta + phaseVec)
        let uv = (f0 .> MLXArray(voicedThreshold)).asType(.float32)
        let noiseAmp = uv * MLXArray(noiseStd) + (MLXArray(Float(1.0)) - uv) * MLXArray(Float(sineAmp / 3.0))
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)
        sineWaves = sineWaves * uv + noise
        return (sineWaves, uv, noise)
    }
}

final class SourceModule: Module {
    private let sineAmp: Float
    private let noiseStd: Float
    private let lSinGen: SineGen
    @ModuleInfo(key: "l_linear") private var lLinear: Linear

    init(
        samplingRate: Int,
        upsampleScale: Int,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        addNoiseStd: Float = 0.003,
        voicedThreshold: Float = 10
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = addNoiseStd
        self.lSinGen = SineGen(
            sampRate: samplingRate,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: addNoiseStd,
            voicedThreshold: voicedThreshold
        )
        self._lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (sineWavs, uv, _) = lSinGen(f0.transposed(0, 2, 1))
        let sineMerged = MLX.tanh(lLinear(sineWavs.transposed(0, 2, 1)))
        let noise = MLXRandom.normal(uv.shape) * MLXArray(Float(sineAmp / 3.0))
        return (sineMerged, noise, uv.transposed(0, 2, 1))
    }
}

final class F0Predictor: Module {
    @ModuleInfo(key: "condnet") var condnet: [Conv1dPT]
    @ModuleInfo(key: "classifier") private var classifier: Linear

    init(inChannels: Int = 80, hiddenChannels: Int = 512, numLayers: Int = 5) {
        var layers: [Conv1dPT] = []
        for idx in 0..<numLayers {
            let inCh = idx == 0 ? inChannels : hiddenChannels
            layers.append(Conv1dPT(inChannels: inCh, outChannels: hiddenChannels, kernelSize: 3, padding: 1))
        }
        self._condnet.wrappedValue = layers
        self._classifier.wrappedValue = Linear(hiddenChannels, 1)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var h = mel
        for conv in condnet {
            h = conv(h)
            h = s3Elu(h)
        }
        h = h.transposed(0, 2, 1)
        var f0 = classifier(h)
        f0 = f0[0..., 0..., 0]
        return MLX.abs(f0)
    }
}

final class HiFTGenerator: Module {
    let samplingRate: Int
    let istftParams: [String: Int]
    let audioLimit: Float
    let numUpsamples: Int
    let numKernels: Int
    let f0UpsampleScale: Int

    @ModuleInfo(key: "f0_predictor") var f0Predictor: F0Predictor
    @ModuleInfo(key: "m_source") private var source: SourceModule
    @ModuleInfo(key: "conv_pre") private var convPre: Conv1dPT
    @ModuleInfo(key: "ups") private var ups: [ConvTranspose1dPT]
    @ModuleInfo(key: "source_downs") private var sourceDowns: [Conv1dPT]
    @ModuleInfo(key: "source_resblocks") private var sourceResblocks: [S3ResBlock]
    @ModuleInfo(key: "resblocks") private var resblocks: [S3ResBlock]
    @ModuleInfo(key: "conv_post") private var convPost: Conv1dPT
    private let stftWindow: MLXArray

    init(
        inChannels: Int = 80,
        baseChannels: Int = 512,
        nbHarmonics: Int = 8,
        samplingRate: Int = 24_000,
        nsfAlpha: Float = 0.1,
        nsfSigma: Float = 0.003,
        nsfVoicedThreshold: Float = 10,
        upsampleRates: [Int] = [8, 5, 3],
        upsampleKernelSizes: [Int] = [16, 11, 7],
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        sourceResblockKernelSizes: [Int] = [7, 7, 11],
        sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        istftParams: [String: Int] = ["n_fft": 16, "hop_len": 4],
        f0Predictor: F0Predictor? = nil
    ) {
        self.samplingRate = samplingRate
        self.istftParams = istftParams
        self.audioLimit = 0.99
        self.numUpsamples = upsampleRates.count
        self.numKernels = resblockKernelSizes.count

        let f0Predictor = f0Predictor ?? F0Predictor()
        self._f0Predictor.wrappedValue = f0Predictor

        let hopLen = istftParams["hop_len"] ?? 4
        let upsampleScale = upsampleRates.reduce(1, *) * hopLen
        self.f0UpsampleScale = upsampleScale
        self._source.wrappedValue = SourceModule(
            samplingRate: samplingRate,
            upsampleScale: upsampleScale,
            harmonicNum: nbHarmonics,
            sineAmp: nsfAlpha,
            addNoiseStd: nsfSigma,
            voicedThreshold: nsfVoicedThreshold
        )

        self._convPre.wrappedValue = Conv1dPT(inChannels: inChannels, outChannels: baseChannels, kernelSize: 7, padding: 3)

        var upLayers: [ConvTranspose1dPT] = []
        var resBlocks: [S3ResBlock] = []
        let base = baseChannels

        for (idx, pair) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            let u = pair.0
            let k = pair.1
            let inCh = base / (1 << idx)
            let outCh = base / (1 << (idx + 1))
            upLayers.append(ConvTranspose1dPT(inChannels: inCh, outChannels: outCh, kernelSize: k, stride: u, padding: (k - u) / 2))
        }

        for i in 0..<upLayers.count {
            let resCh = base / (1 << (i + 1))
            for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
                resBlocks.append(S3ResBlock(channels: resCh, kernelSize: k, dilations: d))
            }
        }

        var sourceDowns: [Conv1dPT] = []
        var sourceResBlocks: [S3ResBlock] = []

        let downsampleRates = [1] + upsampleRates.reversed().dropLast()
        var downsampleCum: [Int] = []
        var running = 1
        for rate in downsampleRates {
            running *= rate
            downsampleCum.append(running)
        }

        let nFft = istftParams["n_fft"] ?? 16
        for (idx, triple) in zip(downsampleCum.reversed(), zip(sourceResblockKernelSizes, sourceResblockDilationSizes)).enumerated() {
            let u = triple.0
            let kernel = triple.1.0
            let dilation = triple.1.1
            let outCh = base / (1 << (idx + 1))
            if u == 1 {
                sourceDowns.append(Conv1dPT(inChannels: nFft + 2, outChannels: outCh, kernelSize: 1))
            } else {
                sourceDowns.append(Conv1dPT(
                    inChannels: nFft + 2,
                    outChannels: outCh,
                    kernelSize: u * 2,
                    stride: u,
                    padding: u / 2
                ))
            }
            sourceResBlocks.append(S3ResBlock(channels: outCh, kernelSize: kernel, dilations: dilation))
        }

        self._ups.wrappedValue = upLayers
        self._resblocks.wrappedValue = resBlocks
        self._sourceDowns.wrappedValue = sourceDowns
        self._sourceResblocks.wrappedValue = sourceResBlocks

        let finalCh = base / (1 << upLayers.count)
        self._convPost.wrappedValue = Conv1dPT(inChannels: finalCh, outChannels: nFft + 2, kernelSize: 7, padding: 3)
        self.stftWindow = MLXArray(s3HannWindowPeriodic(size: nFft))
    }

    private func upsampleF0(_ f0: MLXArray) -> MLXArray {
        let expanded = f0.expandedDimensions(axis: 2)
        let broadcasted = expanded.broadcasted(to: [f0.shape[0], f0.shape[1], f0UpsampleScale])
        return broadcasted.reshaped([f0.shape[0], f0.shape[1] * f0UpsampleScale, 1])
    }

    private func stft(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let nFft = istftParams["n_fft"] ?? 16
        let hopLen = istftParams["hop_len"] ?? 4
        let batch = x.shape[0]
        let length = x.shape[1]
        let numFrames = length < nFft ? 1 : 1 + (length - nFft) / hopLen
        let freqBins = nFft / 2 + 1

        let window = stftWindow.asArray(Float.self)
        let twoPi = 2.0 * Double.pi

        var real = [Float](repeating: 0, count: batch * freqBins * numFrames)
        var imag = [Float](repeating: 0, count: batch * freqBins * numFrames)

        for b in 0..<batch {
            let wav = x[b].asArray(Float.self)
            for frameIndex in 0..<numFrames {
                let start = frameIndex * hopLen
                var frame = [Float](repeating: 0, count: nFft)
                for n in 0..<nFft {
                    let idx = start + n
                    let sample = idx < wav.count ? wav[idx] : 0
                    frame[n] = sample * window[n]
                }
                for k in 0..<freqBins {
                    var sumReal: Double = 0
                    var sumImag: Double = 0
                    for n in 0..<nFft {
                        let angle = twoPi * Double(k * n) / Double(nFft)
                        let value = Double(frame[n])
                        sumReal += value * cos(angle)
                        sumImag -= value * sin(angle)
                    }
                    let base = (b * freqBins + k) * numFrames + frameIndex
                    real[base] = Float(sumReal)
                    imag[base] = Float(sumImag)
                }
            }
        }

        let realArray = MLXArray(real).reshaped([batch, freqBins, numFrames])
        let imagArray = MLXArray(imag).reshaped([batch, freqBins, numFrames])
        return (realArray, imagArray)
    }

    private func istft(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        let nFft = istftParams["n_fft"] ?? 16
        let hopLen = istftParams["hop_len"] ?? 4
        let batch = magnitude.shape[0]
        let numFrames = magnitude.shape[2]

        let maxMag = MLXArray(Float(1e2))
        let magnitudeClamped = MLX.where(magnitude .< maxMag, magnitude, maxMag)
        let real = magnitudeClamped * MLX.cos(phase)
        let imag = magnitudeClamped * MLX.sin(phase)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(batch)

        let window = stftWindow.asArray(Float.self)
        let outputLength = (numFrames - 1) * hopLen + nFft

        for b in 0..<batch {
            let realB = real[b]
            let imagB = imag[b]
            let complexSpec = realB + MLXArray(real: Float(0), imaginary: Float(1)) * imagB
            let framesFreq = MLXFFT.irfft(complexSpec, axis: 0)
            let framesTime = framesFreq.transposed(1, 0)

            var audioSamples = [Float](repeating: 0, count: outputLength)
            var windowSum = [Float](repeating: 0, count: outputLength)

            for i in 0..<numFrames {
                let start = i * hopLen
                let frameData = framesTime[i].asArray(Float.self)
                for j in 0..<min(nFft, frameData.count) {
                    let idx = start + j
                    if idx < outputLength {
                        let win = window[j]
                        audioSamples[idx] += frameData[j] * win
                        windowSum[idx] += win * win
                    }
                }
            }

            for i in 0..<outputLength {
                let denom = max(windowSum[i], 1e-8)
                audioSamples[i] /= denom
            }

            let pad = nFft / 2
            let expectedLen = (numFrames - 1) * hopLen
            let end = min(outputLength, pad + expectedLen)
            let trimmed = pad < end ? Array(audioSamples[pad..<end]) : audioSamples
            outputs.append(MLXArray(trimmed))
        }

        return MLX.stacked(outputs, axis: 0)
    }

    func decode(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let sInput = s[0..., 0, 0...]
        let (sReal, sImag) = stft(sInput)
        let sStft = MLX.concatenated([sReal, sImag], axis: 1)

        var h = convPre(x)

        for i in 0..<numUpsamples {
            h = leakyRelu(h, negativeSlope: 0.1)
            h = ups[i](h)

            if i == numUpsamples - 1 {
                h = MLX.padded(h, widths: [.init(0), .init(0), .init((1, 0))])
            }

            var si = sourceDowns[i](sStft)
            si = sourceResblocks[i](si)
            let minLen = min(h.shape[2], si.shape[2])
            h = h[0..., 0..., 0..<minLen] + si[0..., 0..., 0..<minLen]

            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let idx = i * numKernels + j
                let resOut = resblocks[idx](h)
                xs = xs == nil ? resOut : xs! + resOut
            }
            if let xs {
                h = xs / MLXArray(Float(numKernels))
            }
        }

        h = leakyRelu(h, negativeSlope: 0.01)
        h = convPost(h)

        let nFft = istftParams["n_fft"] ?? 16
        let mag = MLX.exp(h[0..., 0..<(nFft / 2 + 1), 0...])
        let phase = MLX.sin(h[0..., (nFft / 2 + 1)..., 0...])
        var audio = istft(magnitude: mag, phase: phase)
        audio = clip(audio, min: MLXArray(-audioLimit), max: MLXArray(audioLimit))
        return audio
    }

    func callAsFunction(_ mel: MLXArray) -> (MLXArray, MLXArray) {
        let f0 = f0Predictor(mel)
        let f0Up = upsampleF0(f0)
        let (s, _, _) = source(f0Up)
        let sT = s.transposed(0, 2, 1)
        let audio = decode(mel, sT)
        return (audio, f0)
    }

    func inference(_ speechFeat: MLXArray, _ cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let mel = speechFeat.transposed(0, 2, 1)
        let f0 = f0Predictor(mel)
        let f0Up = upsampleF0(f0)
        let (s, _, _) = source(f0Up)
        var sT = s.transposed(0, 2, 1)

        if let cacheSource, cacheSource.shape[2] > 0 {
            let cacheLen = cacheSource.shape[2]
            let tail = sT[0..., 0..., cacheLen...]
            sT = MLX.concatenated([cacheSource, tail], axis: 2)
        }

        let audio = decode(mel, sT)
        return (audio, sT)
    }
}
