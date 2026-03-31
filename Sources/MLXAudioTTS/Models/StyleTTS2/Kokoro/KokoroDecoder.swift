import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - STFT / iSTFT Utilities

private func kokoroHanning(length: Int) -> MLXArray {
    if length == 1 { return MLXArray(1.0) }
    let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
    return 0.5 + 0.5 * cos(n * (.pi / Float(length - 1)))
}

private func kokoroUnwrap(_ p: MLXArray) -> MLXArray {
    let period: Float = 2.0 * .pi
    let discont: Float = period / 2.0
    let pDiff = p[0..., 1..<p.shape[1]] - p[0..., 0..<(p.shape[1] - 1)]
    let intervalLow: Float = -period / 2.0
    var pDiffMod = (((pDiff - intervalLow) % period) + period) % period + intervalLow
    pDiffMod = MLX.where(
        pDiffMod .== intervalLow,
        MLX.where(pDiff .> 0, period / 2.0, pDiffMod),
        pDiffMod
    )
    var phCorrect = pDiffMod - pDiff
    phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)
    return MLX.concatenated([p[0..., 0..<1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

private func kokoroStft(x: MLXArray, nFft: Int, hopLength: Int, winLength: Int) -> MLXArray {
    var w = kokoroHanning(length: winLength + 1)[0..<winLength]
    if w.shape[0] < nFft {
        w = MLX.concatenated([w, MLXArray.zeros([nFft - w.shape[0]])])
    }
    let padding = nFft / 2
    let prefix = x[1..<(padding + 1)][.stride(by: -1)]
    let suffix = x[-(padding + 1)..<(-1)][.stride(by: -1)]
    let padded = MLX.concatenated([prefix, x, suffix])

    let numFrames = 1 + (padded.shape[0] - nFft) / hopLength
    let frames = MLX.asStrided(padded, [numFrames, nFft], strides: [hopLength, 1])
    return MLXFFT.rfft(frames * w).transposed(1, 0)
}

private func kokoroIstft(x: MLXArray, hopLength: Int, winLength: Int) -> MLXArray {
    var w = kokoroHanning(length: winLength + 1)[0..<winLength]
    if w.shape[0] < winLength {
        w = MLX.concatenated([w, MLXArray.zeros([winLength - w.shape[0]])])
    }
    let xT = x.transposed(1, 0)
    let t = (xT.shape[0] - 1) * hopLength + winLength
    let windowModLen = winLength / hopLength
    let wSquared = w * w
    let totalWsquared = MLX.concatenated(Array(repeating: wSquared, count: t / winLength + 1))
    let output = MLXFFT.irfft(xT, axis: 1) * w

    var outputs = [MLXArray]()
    var windowSums = [MLXArray]()
    for i in 0..<windowModLen {
        let outputStride = output[.stride(from: i, by: windowModLen), .ellipsis].reshaped([-1])
        let windowSumArray = totalWsquared[0..<outputStride.shape[0]]
        outputs.append(MLX.concatenated([
            MLXArray.zeros([i * hopLength]), outputStride,
            MLXArray.zeros([max(0, t - i * hopLength - outputStride.shape[0])]),
        ]))
        windowSums.append(MLX.concatenated([
            MLXArray.zeros([i * hopLength]), windowSumArray,
            MLXArray.zeros([max(0, t - i * hopLength - windowSumArray.shape[0])]),
        ]))
    }

    var reconstructed = outputs[0]
    var windowSum = windowSums[0]
    for i in 1..<windowModLen {
        reconstructed = reconstructed + outputs[i]
        windowSum = windowSum + windowSums[i]
    }
    let start = winLength / 2
    let end = reconstructed.shape[0] - winLength / 2
    return reconstructed[start..<end] / windowSum[start..<end]
}

// MARK: - KokoroSTFT

class KokoroSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int

    init(filterLength: Int, hopLength: Int, winLength: Int) {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
    }

    func transform(inputData: MLXArray) -> (magnitude: MLXArray, phase: MLXArray) {
        var audio = inputData
        if audio.ndim == 1 { audio = audio.expandedDimensions(axis: 0) }

        var magnitudes = [MLXArray]()
        var phases = [MLXArray]()
        for b in 0..<audio.shape[0] {
            let result = kokoroStft(x: audio[b], nFft: filterLength, hopLength: hopLength, winLength: winLength)
            magnitudes.append(MLX.abs(result))
            phases.append(MLX.atan2(result.imaginaryPart(), result.realPart()))
        }
        return (MLX.stacked(magnitudes, axis: 0), MLX.stacked(phases, axis: 0))
    }

    func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var reconstructed = [MLXArray]()
        for b in 0..<magnitude.shape[0] {
            let phaseCont = kokoroUnwrap(phase[b])
            let complex = magnitude[b] * MLX.exp(MLXArray(real: Float(0), imaginary: Float(1)) * phaseCont)
            reconstructed.append(kokoroIstft(x: complex, hopLength: hopLength, winLength: winLength))
        }
        return MLX.stacked(reconstructed, axis: 0).expandedDimensions(axis: 1)
    }
}

// MARK: - Generator

class KokoroGenerator: Module {
    let numKernels: Int
    let numUpsamples: Int
    let postNFft: Int
    let stftModule: KokoroSTFT

    @ModuleInfo(key: "m_source") var mSource: SourceModule
    @ModuleInfo(key: "f0_upsamp") var f0Upsamp: Upsample
    @ModuleInfo(key: "noise_convs") var noiseConvs: [Conv1d]
    @ModuleInfo(key: "noise_res") var noiseRes: [AdaINResBlock1]
    @ModuleInfo var ups: [WeightNormedConv]
    @ModuleInfo var resblocks: [AdaINResBlock1]
    @ModuleInfo(key: "conv_post") var convPost: WeightNormedConv

    init(styleDim: Int, config: ISTFTNetConfig) {
        numKernels = config.resblockKernelSizes.count
        numUpsamples = config.upsampleRates.count
        postNFft = config.genIstftNFft

        let upsampleProduct = config.upsampleRates.reduce(1, *)
        let totalUpsample = upsampleProduct * config.genIstftHopSize

        stftModule = KokoroSTFT(
            filterLength: config.genIstftNFft, hopLength: config.genIstftHopSize,
            winLength: config.genIstftNFft
        )

        _mSource = ModuleInfo(
            wrappedValue: SourceModule(
                samplingRate: 24000, upsampleScale: totalUpsample,
                harmonicNum: 8, voicedThreshold: 10
            ), key: "m_source"
        )
        _f0Upsamp = ModuleInfo(
            wrappedValue: Upsample(scaleFactor: .float(Float(totalUpsample))),
            key: "f0_upsamp"
        )

        var upsArr = [WeightNormedConv]()
        var noiseConvsArr = [Conv1d]()
        var noiseResArr = [AdaINResBlock1]()
        var resArr = [AdaINResBlock1]()

        let ch0 = config.upsampleInitialChannel
        for i in 0..<config.upsampleRates.count {
            let u = config.upsampleRates[i]
            let k = config.upsampleKernelSizes[i]
            let chOut = ch0 / (1 << (i + 1))
            let chIn = ch0 / (1 << i)
            upsArr.append(WeightNormedConv(
                inChannels: chOut, outChannels: chIn, kernelSize: k,
                stride: u, padding: (k - u) / 2, encode: true
            ))

            let cCur = ch0 / (1 << (i + 1))
            if i + 1 < config.upsampleRates.count {
                let strideF0 = config.upsampleRates[(i + 1)...].reduce(1, *)
                noiseConvsArr.append(Conv1d(
                    inputChannels: config.genIstftNFft + 2, outputChannels: cCur,
                    kernelSize: strideF0 * 2, stride: strideF0, padding: (strideF0 + 1) / 2
                ))
                noiseResArr.append(AdaINResBlock1(channels: cCur, kernelSize: 7, dilation: [1, 3, 5], styleDim: styleDim))
            } else {
                noiseConvsArr.append(Conv1d(
                    inputChannels: config.genIstftNFft + 2, outputChannels: cCur, kernelSize: 1
                ))
                noiseResArr.append(AdaINResBlock1(channels: cCur, kernelSize: 11, dilation: [1, 3, 5], styleDim: styleDim))
            }

            for j in 0..<config.resblockKernelSizes.count {
                let rk = config.resblockKernelSizes[j]
                let rd = config.resblockDilationSizes[j]
                resArr.append(AdaINResBlock1(channels: cCur, kernelSize: rk, dilation: rd, styleDim: styleDim))
            }
        }

        _ups = ModuleInfo(wrappedValue: upsArr)
        _noiseConvs = ModuleInfo(wrappedValue: noiseConvsArr, key: "noise_convs")
        _noiseRes = ModuleInfo(wrappedValue: noiseResArr, key: "noise_res")
        _resblocks = ModuleInfo(wrappedValue: resArr)

        let lastCh = ch0 / (1 << config.upsampleRates.count)
        _convPost = ModuleInfo(
            wrappedValue: WeightNormedConv(
                inChannels: lastCh, outChannels: config.genIstftNFft + 2,
                kernelSize: 7, padding: 3
            ), key: "conv_post"
        )
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ f0: MLXArray) -> MLXArray {
        let f0Up = f0Upsamp(f0[.newAxis].transposed(0, 2, 1))
        let (harSource, _, _) = mSource(f0Up)
        let harFlat = harSource.transposed(0, 2, 1).squeezed(axis: 1)

        let (harSpec, harPhase) = stftModule.transform(inputData: harFlat)
        var har = MLX.concatenated([harSpec, harPhase], axis: 1)
        har = har.swappedAxes(2, 1)

        var h = x
        for i in 0..<numUpsamples {
            h = leakyRelu(h, negativeSlope: 0.1)
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

        h = leakyRelu(h, negativeSlope: 0.01)
        h = convPost(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)

        let spec = MLX.exp(h[0..., ..<(postNFft / 2 + 1), 0...])
        let phase = MLX.sin(h[0..., (postNFft / 2 + 1)..., 0...])
        return stftModule.inverse(magnitude: spec, phase: phase)
    }
}

// MARK: - Decoder

class KokoroDecoder: Module {
    @ModuleInfo var encode: AdainResBlock1d
    @ModuleInfo var decode: [AdainResBlock1d]
    @ModuleInfo(key: "F0_conv") var f0Conv: WeightNormedConv
    @ModuleInfo(key: "N_conv") var nConv: WeightNormedConv
    @ModuleInfo(key: "asr_res") var asrRes: [WeightNormedConv]
    @ModuleInfo var generator: KokoroGenerator

    init(config: KokoroConfig) {
        let dimIn = config.hiddenDim
        let styleDim = config.styleDim
        let asrResDim = config.asrResDim
        let decoderDim = config.istftnet.upsampleInitialChannel * 2
        let outDim = config.istftnet.upsampleInitialChannel

        _encode = ModuleInfo(wrappedValue: AdainResBlock1d(
            dimIn: dimIn + 2, dimOut: decoderDim, styleDim: styleDim
        ))
        var decodeArr = [AdainResBlock1d]()
        for _ in 0..<3 {
            decodeArr.append(AdainResBlock1d(
                dimIn: decoderDim + 2 + asrResDim, dimOut: decoderDim, styleDim: styleDim
            ))
        }
        decodeArr.append(AdainResBlock1d(
            dimIn: decoderDim + 2 + asrResDim, dimOut: outDim,
            styleDim: styleDim, upsample: true
        ))
        _decode = ModuleInfo(wrappedValue: decodeArr)

        _f0Conv = ModuleInfo(
            wrappedValue: WeightNormedConv(
                inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1
            ), key: "F0_conv"
        )
        _nConv = ModuleInfo(
            wrappedValue: WeightNormedConv(
                inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1
            ), key: "N_conv"
        )
        _asrRes = ModuleInfo(
            wrappedValue: [WeightNormedConv(inChannels: dimIn, outChannels: asrResDim, kernelSize: 1, padding: 0)],
            key: "asr_res"
        )
        _generator = ModuleInfo(wrappedValue: KokoroGenerator(styleDim: styleDim, config: config.istftnet))
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
