import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

// MARK: - Instance Normalization

class KittenInstanceNorm1d: Module {
    let numFeatures: Int
    let eps: Float

    init(numFeatures: Int, eps: Float = 1e-5) {
        self.numFeatures = numFeatures
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        return (x - mean) / MLX.sqrt(variance + eps)
    }
}

// MARK: - AdaIN1d

class KittenAdaIN1d: Module {
    @ModuleInfo var norm: KittenInstanceNorm1d
    @ModuleInfo var fc: Linear

    init(styleDim: Int, numFeatures: Int) {
        _norm = ModuleInfo(wrappedValue: KittenInstanceNorm1d(numFeatures: numFeatures))
        _fc = ModuleInfo(wrappedValue: Linear(styleDim, numFeatures * 2))
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var h = fc(s)
        h = h.expandedDimensions(axis: 2)
        let parts = MLX.split(h, parts: 2, axis: 1)
        let gamma = parts[0]
        let beta = parts[1]
        return (1 + gamma) * norm(x) + beta
    }
}

// MARK: - UpSample1d

class KittenUpSample1d: Module {
    let layerType: String

    init(layerType: String = "none") {
        self.layerType = layerType
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if layerType == "none" { return x }
        return Upsample(scaleFactor: .float(2), mode: .nearest)(x)
    }
}

// MARK: - AdainResBlk1d

class KittenAdainResBlk1d: Module {
    let dimIn: Int
    let upsampleType: String
    @ModuleInfo var conv1: KittenConvWeighted
    @ModuleInfo var conv2: KittenConvWeighted
    @ModuleInfo var norm1: KittenAdaIN1d
    @ModuleInfo var norm2: KittenAdaIN1d
    @ModuleInfo var upsample: KittenUpSample1d
    @ModuleInfo var dropout: MLXNN.Dropout
    var conv1x1: KittenConvWeighted?
    var pool: KittenConvWeighted?

    init(dimIn: Int, dimOut: Int, styleDim: Int = 64, upsample: Bool = false, dropoutP: Float = 0.0) {
        self.dimIn = dimIn
        self.upsampleType = upsample ? "upsample" : "none"
        _conv1 = ModuleInfo(wrappedValue: KittenConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 3, padding: 1))
        _conv2 = ModuleInfo(wrappedValue: KittenConvWeighted(inChannels: dimOut, outChannels: dimOut, kernelSize: 3, padding: 1))
        _norm1 = ModuleInfo(wrappedValue: KittenAdaIN1d(styleDim: styleDim, numFeatures: dimIn))
        _norm2 = ModuleInfo(wrappedValue: KittenAdaIN1d(styleDim: styleDim, numFeatures: dimOut))
        _upsample = ModuleInfo(wrappedValue: KittenUpSample1d(layerType: upsample ? "upsample" : "none"))
        _dropout = ModuleInfo(wrappedValue: MLXNN.Dropout(p: dropoutP))
        if dimIn != dimOut {
            conv1x1 = KittenConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 1, padding: 0, bias: false)
        }
        if upsample {
            pool = KittenConvWeighted(inChannels: 1, outChannels: dimIn, kernelSize: 3, stride: 2, padding: 1, groups: dimIn)
        }
    }

    private func shortcut(_ x: MLXArray) -> MLXArray {
        var h = x.swappedAxes(2, 1)
        h = upsample(h)
        h = h.swappedAxes(2, 1)
        if let conv1x1 {
            h = conv1x1(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
        }
        return h
    }

    private func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var h = norm1(x, s)
        h = kittenLeakyRelu(h, negativeSlope: 0.2)

        if upsampleType != "none", let pool {
            h = pool(h.swappedAxes(2, 1), op: .convTranspose1d)
            h = MLX.padded(h, widths: [.init((0, 0)), .init((0, 1)), .init((0, 0))])
            h = h.swappedAxes(2, 1)
        }

        h = conv1(dropout(h.swappedAxes(2, 1)), op: .conv1d).swappedAxes(2, 1)
        h = norm2(h, s)
        h = kittenLeakyRelu(h, negativeSlope: 0.2)
        h = conv2(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
        return h
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let out = residual(x, s)
        return (out + shortcut(x)) / Float(2).squareRoot()
    }
}

// MARK: - AdaINResBlock1 (Snake activation, for Generator)

class KittenAdaINResBlock1: Module {
    var convs1: [KittenConvWeighted]
    var convs2: [KittenConvWeighted]
    var adain1: [KittenAdaIN1d]
    var adain2: [KittenAdaIN1d]
    var alpha1_0: MLXArray
    var alpha1_1: MLXArray
    var alpha1_2: MLXArray
    var alpha2_0: MLXArray
    var alpha2_1: MLXArray
    var alpha2_2: MLXArray

    init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5], styleDim: Int = 64) {
        convs1 = dilation.map { d in
            KittenConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                               padding: (kernelSize * d - d) / 2, dilation: d)
        }
        convs2 = (0..<3).map { _ in
            KittenConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize,
                               padding: (kernelSize - 1) / 2)
        }
        adain1 = (0..<3).map { _ in KittenAdaIN1d(styleDim: styleDim, numFeatures: channels) }
        adain2 = (0..<3).map { _ in KittenAdaIN1d(styleDim: styleDim, numFeatures: channels) }
        alpha1_0 = MLXArray.ones([1, channels, 1])
        alpha1_1 = MLXArray.ones([1, channels, 1])
        alpha1_2 = MLXArray.ones([1, channels, 1])
        alpha2_0 = MLXArray.ones([1, channels, 1])
        alpha2_1 = MLXArray.ones([1, channels, 1])
        alpha2_2 = MLXArray.ones([1, channels, 1])
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let alphas1 = [alpha1_0, alpha1_1, alpha1_2]
        let alphas2 = [alpha2_0, alpha2_1, alpha2_2]
        var h = x
        for idx in 0..<3 {
            let a1 = alphas1[idx]
            let a2 = alphas2[idx]
            var xt = adain1[idx](h, s)
            xt = xt + (1 / a1) * MLX.pow(MLX.sin(a1 * xt), 2)
            xt = convs1[idx](xt.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
            xt = adain2[idx](xt, s)
            xt = xt + (1 / a2) * MLX.pow(MLX.sin(a2 * xt), 2)
            xt = convs2[idx](xt.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
            h = xt + h
        }
        return h
    }
}

// MARK: - 1D Linear Interpolation

private func interpolate1d(_ input: MLXArray, size: Int) -> MLXArray {
    let inWidth = input.shape[2]
    if inWidth == size { return input }
    if inWidth < 1 || size < 1 { return input }

    let scale = Float(inWidth) / Float(size)
    var xCoords = MLXArray(Array((0..<size).map { Float($0) * scale + 0.5 * scale - 0.5 }))
    xCoords = MLX.clip(xCoords, min: 0, max: Float(inWidth - 1))

    let xLow = MLX.floor(xCoords).asType(.int32)
    let xHigh = MLX.minimum(xLow + 1, MLXArray(Int32(inWidth - 1)))
    let xFrac = xCoords - xLow.asType(.float32)

    let yLow = input[0..., 0..., xLow]
    let yHigh = input[0..., 0..., xHigh]
    let fracExpanded = xFrac.reshaped([1, 1, size])
    return yLow * (1 - fracExpanded) + yHigh * fracExpanded
}

// MARK: - SineGen & SourceModule

class KittenSineGen {
    let sineAmp: Float
    let noiseStd: Float
    let harmonicNum: Int
    let samplingRate: Int
    let voicedThreshold: Float
    let upsampleScale: Int

    init(sampRate: Int, upsampleScale: Int, harmonicNum: Int = 0,
         sineAmp: Float = 0.1, noiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.upsampleScale = upsampleScale
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let harmonics = MLXArray(Array((1...Int32(harmonicNum + 1)).map { Float($0) }))
            .reshaped([1, 1, harmonicNum + 1])
        let fn = f0 * harmonics

        var radValues = (fn / Float(samplingRate)) % MLXArray(Float(1))

        // Initial phase noise (none for fundamental)
        var randIni = MLXRandom.normal([f0.shape[0], harmonicNum + 1])
        randIni[0..., 0] = MLXArray(Float(0))
        radValues[0..., 0, 0...] = radValues[0..., 0, 0...] + randIni

        // Downsample → cumsum → upsample (smooth phase interpolation)
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

class KittenSourceModule: Module {
    let sineGen: KittenSineGen
    @ModuleInfo(key: "l_linear") var lLinear: Linear

    init(samplingRate: Int, upsampleScale: Int, harmonicNum: Int = 0,
         sineAmp: Float = 0.1, addNoiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        sineGen = KittenSineGen(sampRate: samplingRate, upsampleScale: upsampleScale,
                                harmonicNum: harmonicNum, sineAmp: sineAmp,
                                noiseStd: addNoiseStd, voicedThreshold: voicedThreshold)
        _lLinear = ModuleInfo(wrappedValue: Linear(harmonicNum + 1, 1), key: "l_linear")
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (sineWavs, uv, _) = sineGen(x)
        let sineMerge = tanh(lLinear(sineWavs))
        let noise = MLXRandom.normal(uv.shape) * sineGen.sineAmp / 3
        return (sineMerge, noise, uv)
    }
}

// MARK: - Generator

class KittenGenerator: Module {
    let numKernels: Int
    let numUpsamples: Int
    let postNFft: Int

    @ModuleInfo(key: "m_source") var mSource: KittenSourceModule
    @ModuleInfo(key: "f0_upsamp") var f0Upsamp: Upsample
    @ModuleInfo(key: "noise_convs") var noiseConvs: [Conv1d]
    @ModuleInfo(key: "noise_res") var noiseRes: [KittenAdaINResBlock1]
    @ModuleInfo var ups: [KittenConvWeighted]
    @ModuleInfo var resblocks: [KittenAdaINResBlock1]
    @ModuleInfo(key: "conv_post") var convPost: KittenConvWeighted

    init(styleDim: Int, config: KittenTTSISTFTNetConfig) {
        numKernels = config.resblockKernelSizes.count
        numUpsamples = config.upsampleRates.count
        postNFft = config.genIstftNFft

        let upsampleProduct = config.upsampleRates.reduce(1, *)
        let totalUpsample = upsampleProduct * config.genIstftHopSize

        _mSource = ModuleInfo(wrappedValue: KittenSourceModule(
            samplingRate: 24000, upsampleScale: totalUpsample, harmonicNum: 8, voicedThreshold: 10), key: "m_source")
        _f0Upsamp = ModuleInfo(wrappedValue: Upsample(scaleFactor: .float(Float(totalUpsample))), key: "f0_upsamp")

        var upsArr = [KittenConvWeighted]()
        var noiseConvsArr = [Conv1d]()
        var noiseResArr = [KittenAdaINResBlock1]()
        var resArr = [KittenAdaINResBlock1]()

        let ch0 = config.upsampleInitialChannel
        for i in 0..<config.upsampleRates.count {
            let u = config.upsampleRates[i]
            let k = config.upsampleKernelSizes[i]
            let chOut = ch0 / (1 << (i + 1))
            let chIn = ch0 / (1 << i)
            upsArr.append(KittenConvWeighted(inChannels: chOut, outChannels: chIn, kernelSize: k, stride: u, padding: (k - u) / 2, encode: true))

            let cCur = ch0 / (1 << (i + 1))
            if i + 1 < config.upsampleRates.count {
                let strideF0 = config.upsampleRates[(i+1)...].reduce(1, *)
                noiseConvsArr.append(Conv1d(inputChannels: config.genIstftNFft + 2, outputChannels: cCur, kernelSize: strideF0 * 2, stride: strideF0, padding: (strideF0 + 1) / 2))
                noiseResArr.append(KittenAdaINResBlock1(channels: cCur, kernelSize: 7, dilation: [1, 3, 5], styleDim: styleDim))
            } else {
                noiseConvsArr.append(Conv1d(inputChannels: config.genIstftNFft + 2, outputChannels: cCur, kernelSize: 1))
                noiseResArr.append(KittenAdaINResBlock1(channels: cCur, kernelSize: 11, dilation: [1, 3, 5], styleDim: styleDim))
            }

            for j in 0..<config.resblockKernelSizes.count {
                let rk = config.resblockKernelSizes[j]
                let rd = config.resblockDilationSizes[j]
                resArr.append(KittenAdaINResBlock1(channels: cCur, kernelSize: rk, dilation: rd, styleDim: styleDim))
            }
        }

        _ups = ModuleInfo(wrappedValue: upsArr)
        _noiseConvs = ModuleInfo(wrappedValue: noiseConvsArr, key: "noise_convs")
        _noiseRes = ModuleInfo(wrappedValue: noiseResArr, key: "noise_res")
        _resblocks = ModuleInfo(wrappedValue: resArr)

        let lastCh = ch0 / (1 << config.upsampleRates.count)
        _convPost = ModuleInfo(wrappedValue: KittenConvWeighted(inChannels: lastCh, outChannels: config.genIstftNFft + 2, kernelSize: 7, padding: 3), key: "conv_post")
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ f0: MLXArray) -> MLXArray {
        let f0Up = f0Upsamp(f0[.newAxis].transposed(0, 2, 1))
        let (harSource, _, _) = mSource(f0Up)
        let harFlat = harSource.transposed(0, 2, 1).squeezed(axis: 1)

        let (harSpec, harPhase) = stftForward(harFlat)
        var har = MLX.concatenated([harSpec, harPhase], axis: 1)
        har = har.swappedAxes(2, 1)

        var h = x
        for i in 0..<numUpsamples {
            h = kittenLeakyRelu(h, negativeSlope: 0.1)
            let xSource = noiseRes[i](noiseConvs[i](har).swappedAxes(2, 1), s)
            h = ups[i](h.swappedAxes(2, 1), op: .convTranspose1d).swappedAxes(2, 1)
            if i == numUpsamples - 1 {
                h = MLX.padded(h, widths: [.init((0, 0)), .init((0, 0)), .init((1, 0))])
            }
            h = h + xSource

            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let blockOut = resblocks[i * numKernels + j](h, s)
                xs = xs.map { $0 + blockOut } ?? blockOut
            }
            h = xs! / Float(numKernels)
        }

        h = kittenLeakyRelu(h, negativeSlope: 0.01)
        h = convPost(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)

        let spec = MLX.exp(h[0..., ..<(postNFft / 2 + 1), 0...])
        let phase = MLX.sin(h[0..., (postNFft / 2 + 1)..., 0...])
        let result = istftInverse(spec, phase)
        return result
    }

    private func stftForward(_ audio: MLXArray) -> (magnitude: MLXArray, phase: MLXArray) {
        var input = audio
        if input.ndim == 1 {
            input = input.expandedDimensions(axis: 0)
        }
        var mags = [MLXArray]()
        var phases = [MLXArray]()
        for b in 0..<input.shape[0] {
            let result = stft(audio: input[b], window: hanningWindow(size: postNFft),
                              nFft: postNFft, hopLength: 5, padMode: .reflect)
            // stft returns [T, nFft/2+1], transpose to [nFft/2+1, T] to match Python
            let transposed = result.transposed(1, 0)
            mags.append(MLX.abs(transposed))
            let r = transposed.realPart()
            let im = transposed.imaginaryPart()
            phases.append(MLX.atan2(im, r))
        }
        return (MLX.stacked(mags, axis: 0), MLX.stacked(phases, axis: 0))
    }

    private func istftInverse(_ magnitude: MLXArray, _ phase: MLXArray) -> MLXArray {
        let hopSize = 5
        let winLength = postNFft
        let batchSize = magnitude.shape[0]
        var outputs = [MLXArray]()

        // Use hanning window matching Python (periodic, size N, drop last)
        let w = hanningWindow(size: winLength + 1)
        let window = w[0..<winLength]
        let windowArray = window.asArray(Float.self)
        // COLA normalization: window squared (matches Python normalized=True)
        let windowSqArray = windowArray.map { $0 * $0 }

        for b in 0..<batchSize {
            let realB = magnitude[b] * MLX.cos(phase[b])
            let imagB = magnitude[b] * MLX.sin(phase[b])
            let complexSpec = realB + MLXArray(real: Float(0), imaginary: Float(1)) * imagB
            // complexSpec shape: [nFft/2+1, numFrames]
            let framesFreq = MLXFFT.irfft(complexSpec, axis: 0)
            // framesFreq: [nFft, numFrames], transpose → [numFrames, nFft]
            let framesTime = framesFreq.transposed(1, 0)
            let windowedFrames = framesTime * window

            let numFrames = windowedFrames.shape[0]
            let outputLength = (numFrames - 1) * hopSize + winLength
            var audioSamples = [Float](repeating: 0, count: outputLength)
            var windowSum = [Float](repeating: 0, count: outputLength)

            for i in 0..<numFrames {
                let start = i * hopSize
                let frameData = windowedFrames[i].asArray(Float.self)
                for j in 0..<min(winLength, frameData.count) {
                    if start + j < outputLength {
                        audioSamples[start + j] += frameData[j]
                        windowSum[start + j] += windowSqArray[j]
                    }
                }
            }
            for i in 0..<outputLength {
                if windowSum[i] > 1e-10 {
                    audioSamples[i] /= windowSum[i]
                }
            }
            // Center trim: remove win_length//2 from both ends
            let start = winLength / 2
            let end = outputLength - winLength / 2
            if end > start {
                outputs.append(MLXArray(Array(audioSamples[start..<end])))
            } else {
                outputs.append(MLXArray(audioSamples))
            }
        }
        return MLX.stacked(outputs, axis: 0).expandedDimensions(axis: 1)
    }
}

// MARK: - KittenDecoder

class KittenDecoder: Module {
    @ModuleInfo var encode: KittenAdainResBlk1d
    @ModuleInfo var decode: [KittenAdainResBlk1d]
    @ModuleInfo(key: "F0_conv") var f0Conv: KittenConvWeighted
    @ModuleInfo(key: "N_conv") var nConv: KittenConvWeighted
    @ModuleInfo(key: "asr_res") var asrRes: [KittenConvWeighted]
    @ModuleInfo var generator: KittenGenerator

    init(config: KittenTTSConfig) {
        let dimIn = config.hiddenDim
        let styleDim = config.styleDim
        let maxConvDim = config.maxConvDim
        let decoderOutDim = config.decoderOutDim ?? config.maxConvDim
        let asrResDim = config.asrResDim

        _encode = ModuleInfo(wrappedValue: KittenAdainResBlk1d(dimIn: dimIn + 2, dimOut: maxConvDim, styleDim: styleDim))
        var decodeArr = [KittenAdainResBlk1d]()
        for _ in 0..<3 {
            decodeArr.append(KittenAdainResBlk1d(dimIn: maxConvDim + 2 + asrResDim, dimOut: maxConvDim, styleDim: styleDim))
        }
        decodeArr.append(KittenAdainResBlk1d(dimIn: maxConvDim + 2 + asrResDim, dimOut: decoderOutDim, styleDim: styleDim, upsample: true))
        _decode = ModuleInfo(wrappedValue: decodeArr)

        _f0Conv = ModuleInfo(wrappedValue: KittenConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1), key: "F0_conv")
        _nConv = ModuleInfo(wrappedValue: KittenConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1), key: "N_conv")
        _asrRes = ModuleInfo(wrappedValue: [KittenConvWeighted(inChannels: dimIn, outChannels: asrResDim, kernelSize: 1, padding: 0)], key: "asr_res")
        _generator = ModuleInfo(wrappedValue: KittenGenerator(styleDim: styleDim, config: config.istftnet))
    }

    func callAsFunction(_ asr: MLXArray, _ f0: MLXArray, _ n: MLXArray, _ s: MLXArray) -> MLXArray {
        let f0Curve = f0
        let f0Exp = f0.expandedDimensions(axis: 1).swappedAxes(2, 1)
        let f0Down = f0Conv(f0Exp, op: .conv1d).swappedAxes(2, 1)
        let nExp = n.expandedDimensions(axis: 1).swappedAxes(2, 1)
        let nDown = nConv(nExp, op: .conv1d).swappedAxes(2, 1)
        var x = MLX.concatenated([asr, f0Down, nDown], axis: 1)
        x = encode(x, s)
        let asrResOut = asrRes[0](asr.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
        var res = true
        for block in decode {
            if res {
                x = MLX.concatenated([x, asrResOut, f0Down, nDown], axis: 1)
            }
            x = block(x, s)
            if block.upsampleType != "none" {
                res = false
            }
        }
        return generator(x, s, f0Curve)
    }
}

func kittenLeakyRelu(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
    MLX.where(x .> 0, x, x * negativeSlope)
}
