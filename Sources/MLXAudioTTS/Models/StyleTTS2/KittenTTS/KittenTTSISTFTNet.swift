import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

// MARK: - Generator

class KittenGenerator: Module {
    let numKernels: Int
    let numUpsamples: Int
    let postNFft: Int

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

        _mSource = ModuleInfo(wrappedValue: SourceModule(
            samplingRate: 24000, upsampleScale: totalUpsample, harmonicNum: 8, voicedThreshold: 10), key: "m_source")
        _f0Upsamp = ModuleInfo(wrappedValue: Upsample(scaleFactor: .float(Float(totalUpsample))), key: "f0_upsamp")

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
            upsArr.append(WeightNormedConv(inChannels: chOut, outChannels: chIn, kernelSize: k, stride: u, padding: (k - u) / 2, encode: true))

            let cCur = ch0 / (1 << (i + 1))
            if i + 1 < config.upsampleRates.count {
                let strideF0 = config.upsampleRates[(i+1)...].reduce(1, *)
                noiseConvsArr.append(Conv1d(inputChannels: config.genIstftNFft + 2, outputChannels: cCur, kernelSize: strideF0 * 2, stride: strideF0, padding: (strideF0 + 1) / 2))
                noiseResArr.append(AdaINResBlock1(channels: cCur, kernelSize: 7, dilation: [1, 3, 5], styleDim: styleDim))
            } else {
                noiseConvsArr.append(Conv1d(inputChannels: config.genIstftNFft + 2, outputChannels: cCur, kernelSize: 1))
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
        _convPost = ModuleInfo(wrappedValue: WeightNormedConv(inChannels: lastCh, outChannels: config.genIstftNFft + 2, kernelSize: 7, padding: 3), key: "conv_post")
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

        let w = hanningWindow(size: winLength + 1)
        let window = w[0..<winLength]
        let windowArray = window.asArray(Float.self)
        let windowSqArray = windowArray.map { $0 * $0 }

        for b in 0..<batchSize {
            let realB = magnitude[b] * MLX.cos(phase[b])
            let imagB = magnitude[b] * MLX.sin(phase[b])
            let complexSpec = realB + MLXArray(real: Float(0), imaginary: Float(1)) * imagB
            let framesFreq = MLXFFT.irfft(complexSpec, axis: 0)
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
    @ModuleInfo var encode: AdainResBlock1d
    @ModuleInfo var decode: [AdainResBlock1d]
    @ModuleInfo(key: "F0_conv") var f0Conv: WeightNormedConv
    @ModuleInfo(key: "N_conv") var nConv: WeightNormedConv
    @ModuleInfo(key: "asr_res") var asrRes: [WeightNormedConv]
    @ModuleInfo var generator: KittenGenerator

    init(config: KittenTTSConfig) {
        let dimIn = config.hiddenDim
        let styleDim = config.styleDim
        let maxConvDim = config.maxConvDim
        let decoderOutDim = config.decoderOutDim ?? config.maxConvDim
        let asrResDim = config.asrResDim

        _encode = ModuleInfo(wrappedValue: AdainResBlock1d(dimIn: dimIn + 2, dimOut: maxConvDim, styleDim: styleDim))
        var decodeArr = [AdainResBlock1d]()
        for _ in 0..<3 {
            decodeArr.append(AdainResBlock1d(dimIn: maxConvDim + 2 + asrResDim, dimOut: maxConvDim, styleDim: styleDim))
        }
        decodeArr.append(AdainResBlock1d(dimIn: maxConvDim + 2 + asrResDim, dimOut: decoderOutDim, styleDim: styleDim, upsample: true))
        _decode = ModuleInfo(wrappedValue: decodeArr)

        _f0Conv = ModuleInfo(wrappedValue: WeightNormedConv(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1), key: "F0_conv")
        _nConv = ModuleInfo(wrappedValue: WeightNormedConv(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1), key: "N_conv")
        _asrRes = ModuleInfo(wrappedValue: [WeightNormedConv(inChannels: dimIn, outChannels: asrResDim, kernelSize: 1, padding: 0)], key: "asr_res")
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
