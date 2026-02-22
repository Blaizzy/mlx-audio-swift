import Foundation
import MLX
import MLXFFT
import MLXNN
import MLXRandom

// MARK: - STFT / iSTFT

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
    pDiffMod = MLX.where(pDiffMod .== intervalLow, MLX.where(pDiff .> 0, period / 2.0, pDiffMod), pDiffMod)
    var phCorrect = pDiffMod - pDiff
    phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)
    return MLX.concatenated([p[0..., 0..<1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

private func kokoroStft(x: MLXArray, nFft: Int, hopLength: Int, winLength: Int) -> MLXArray {
    var w = kokoroHanning(length: winLength + 1)[0..<winLength]
    if w.shape[0] < nFft {
        w = MLX.concatenated([w, MLXArray.zeros([nFft - w.shape[0]])])
    }
    // Reflect pad
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
    let windowModLen = 20 / 5
    let wSquared = w * w
    let totalWsquared = MLX.concatenated(Array(repeating: wSquared, count: t / winLength))
    let output = MLXFFT.irfft(xT, axis: 1) * w

    var outputs: [MLXArray] = []
    var windowSums: [MLXArray] = []
    for i in 0..<windowModLen {
        let outputStride = output[.stride(from: i, by: windowModLen), .ellipsis].reshaped([-1])
        let windowSumArray = totalWsquared[0..<outputStride.shape[0]]
        outputs.append(MLX.concatenated([
            MLXArray.zeros([i * hopLength]), outputStride,
            MLXArray.zeros([max(0, t - i * hopLength - outputStride.shape[0])])
        ]))
        windowSums.append(MLX.concatenated([
            MLXArray.zeros([i * hopLength]), windowSumArray,
            MLXArray.zeros([max(0, t - i * hopLength - windowSumArray.shape[0])])
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

class KokoroSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int

    init(filterLength: Int, hopLength: Int, winLength: Int) {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
    }

    func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
        var audio = inputData
        if audio.ndim == 1 { audio = audio.expandedDimensions(axis: 0) }

        var magnitudes: [MLXArray] = []
        var phases: [MLXArray] = []
        for b in 0..<audio.shape[0] {
            let stft = kokoroStft(x: audio[b], nFft: filterLength, hopLength: hopLength, winLength: winLength)
            magnitudes.append(MLX.abs(stft))
            phases.append(MLX.atan2(stft.imaginaryPart(), stft.realPart()))
        }
        return (MLX.stacked(magnitudes, axis: 0), MLX.stacked(phases, axis: 0))
    }

    func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var reconstructed: [MLXArray] = []
        for b in 0..<magnitude.shape[0] {
            let phaseCont = kokoroUnwrap(phase[b])
            let stft = magnitude[b] * MLX.exp(MLXArray(real: 0, imaginary: 1) * phaseCont)
            reconstructed.append(kokoroIstft(x: stft, hopLength: hopLength, winLength: winLength))
        }
        return MLX.stacked(reconstructed, axis: 0).expandedDimensions(axis: 1)
    }
}

// MARK: - Sine Generator

private class KokoroSineGen {
    let sineAmp: Float
    let noiseStd: Float
    let harmonicNum: Int
    let samplingRate: Int
    let voicedThreshold: Float
    let upsampleScale: Float

    init(sampRate: Int, upsampleScale: Float, harmonicNum: Int = 0,
         sineAmp: Float = 0.1, noiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.upsampleScale = upsampleScale
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let range = MLXArray(1...(harmonicNum + 1)).asType(.float32)
        let fn = f0 * range.reshaped([1, 1, range.shape[0]])

        // Generate sine from F0
        var radValues = (fn / Float(samplingRate)) % 1
        let randIni = MLXRandom.normal([fn.shape[0], fn.shape[2]])
        randIni[0..., 0] = MLXArray(0.0)
        radValues[0..<radValues.shape[0], 0, 0..<radValues.shape[2]] =
            radValues[0..<radValues.shape[0], 0, 0..<radValues.shape[2]] + randIni

        radValues = kokoroInterpolate(
            input: radValues.transposed(0, 2, 1),
            scaleFactor: [1 / upsampleScale], mode: "linear"
        ).transposed(0, 2, 1)

        var phase = MLX.cumsum(radValues, axis: 1) * 2 * Float.pi
        phase = kokoroInterpolate(
            input: phase.transposed(0, 2, 1) * upsampleScale,
            scaleFactor: [upsampleScale], mode: "linear"
        ).transposed(0, 2, 1)

        let sineWaves = MLX.sin(phase) * sineAmp
        let uv = (f0 .> voicedThreshold).asType(.float32)
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)
        return (sineWaves * uv + noise, uv, noise)
    }
}

// MARK: - Source Module (Harmonic + Noise Source Filter)

private class KokoroSourceModule: Module {
    let sineAmp: Float
    let sinGen: KokoroSineGen
    let lLinear: Linear

    init(weights: [String: MLXArray], samplingRate: Int, upsampleScale: Float,
         harmonicNum: Int = 0, sineAmp: Float = 0.1, voicedThreshold: Float = 0) {
        self.sineAmp = sineAmp
        sinGen = KokoroSineGen(
            sampRate: samplingRate, upsampleScale: upsampleScale,
            harmonicNum: harmonicNum, sineAmp: sineAmp, voicedThreshold: voicedThreshold
        )
        lLinear = Linear(
            weight: weights["decoder.generator.m_source.l_linear.weight"]!,
            bias: weights["decoder.generator.m_source.l_linear.bias"]!
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (sineWavs, uv, _) = sinGen(x)
        let sineMerge = tanh(lLinear(sineWavs))
        let noise = MLXRandom.normal(uv.shape) * (sineAmp / 3)
        return (sineMerge, noise, uv)
    }
}

// MARK: - Generator (iSTFT-based HiFi-GAN)

private class KokoroGenerator {
    let numKernels: Int
    let numUpsamples: Int
    let mSource: KokoroSourceModule
    let f0Upsample: Upsample
    let postNFft: Int
    var noiseConvs: [KokoroConv1d]
    var noiseRes: [KokoroAdaINResBlock1]
    var ups: [KokoroConvWeighted]
    var resBlocks: [KokoroAdaINResBlock1]
    let convPost: KokoroConvWeighted
    let reflectionPad: KokoroReflectionPad1d
    let stft: KokoroSTFT

    init(
        weights: [String: MLXArray], styleDim: Int,
        resblockKernelSizes: [Int], upsampleRates: [Int],
        upsampleInitialChannel: Int, resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int], genIstftNFft: Int, genIstftHopSize: Int
    ) {
        numKernels = resblockKernelSizes.count
        numUpsamples = upsampleRates.count

        let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
        let upsampleScaleNumVal: Int = upsampleScaleNum.item()

        mSource = KokoroSourceModule(
            weights: weights, samplingRate: 24000,
            upsampleScale: upsampleScaleNum.item(),
            harmonicNum: 8, voicedThreshold: 10
        )
        f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

        noiseConvs = []
        noiseRes = []
        ups = []

        for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            ups.append(KokoroConvWeighted(
                weightG: weights["decoder.generator.ups.\(i).weight_g"]!,
                weightV: weights["decoder.generator.ups.\(i).weight_v"]!,
                bias: weights["decoder.generator.ups.\(i).bias"]!,
                stride: u, padding: (k - u) / 2
            ))
        }

        resBlocks = []
        for i in 0..<ups.count {
            let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
            for (j, (k, d)) in zip(resblockKernelSizes, resblockDilationSizes).enumerated() {
                resBlocks.append(KokoroAdaINResBlock1(
                    weights: weights,
                    keyPrefix: "decoder.generator.resblocks.\((i * resblockKernelSizes.count) + j)",
                    channels: ch, kernelSize: k, dilation: d, styleDim: styleDim
                ))
            }

            let cCur = ch
            if i + 1 < upsampleRates.count {
                let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
                noiseConvs.append(KokoroConv1d(
                    weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
                    bias: weights["decoder.generator.noise_convs.\(i).bias"]!,
                    stride: strideF0, padding: (strideF0 + 1) / 2
                ))
                noiseRes.append(KokoroAdaINResBlock1(
                    weights: weights,
                    keyPrefix: "decoder.generator.noise_res.\(i)",
                    channels: cCur, kernelSize: 7, dilation: [1, 3, 5], styleDim: styleDim
                ))
            } else {
                noiseConvs.append(KokoroConv1d(
                    weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
                    bias: weights["decoder.generator.noise_convs.\(i).bias"]!
                ))
                noiseRes.append(KokoroAdaINResBlock1(
                    weights: weights,
                    keyPrefix: "decoder.generator.noise_res.\(i)",
                    channels: cCur, kernelSize: 11, dilation: [1, 3, 5], styleDim: styleDim
                ))
            }
        }

        postNFft = genIstftNFft
        convPost = KokoroConvWeighted(
            weightG: weights["decoder.generator.conv_post.weight_g"]!,
            weightV: weights["decoder.generator.conv_post.weight_v"]!,
            bias: weights["decoder.generator.conv_post.bias"]!,
            padding: 3
        )
        reflectionPad = KokoroReflectionPad1d(padding: (1, 0))
        stft = KokoroSTFT(filterLength: genIstftNFft, hopLength: genIstftHopSize, winLength: genIstftNFft)
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ f0Curve: MLXArray) -> MLXArray {
        var f0New = f0Curve[.newAxis, 0..., 0...].transposed(0, 2, 1)
        f0New = f0Upsample(f0New)

        var (harSource, _, _) = mSource(f0New)
        harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
        let (harSpec, harPhase) = stft.transform(inputData: harSource)
        var har = MLX.concatenated([harSpec, harPhase], axis: 1)
        har = MLX.swappedAxes(har, 2, 1)

        var out = x
        for i in 0..<numUpsamples {
            out = LeakyReLU(negativeSlope: 0.1)(out)
            var xSource = noiseConvs[i](har)
            xSource = MLX.swappedAxes(xSource, 2, 1)
            xSource = noiseRes[i](xSource, s)

            out = MLX.swappedAxes(out, 2, 1)
            out = ups[i](out, conv: MLX.convTransposed1d)
            out = MLX.swappedAxes(out, 2, 1)

            if i == numUpsamples - 1 { out = reflectionPad(out) }
            out = out + xSource

            var xs: MLXArray?
            for j in 0..<numKernels {
                let blk = resBlocks[i * numKernels + j](out, s)
                xs = xs.map { $0 + blk } ?? blk
            }
            out = xs! / numKernels
        }

        out = LeakyReLU(negativeSlope: 0.01)(out)
        out = MLX.swappedAxes(out, 2, 1)
        out = convPost(out, conv: MLX.conv1d)
        out = MLX.swappedAxes(out, 2, 1)

        let spec = MLX.exp(out[0..., 0..<(postNFft / 2 + 1), 0...])
        let phase = MLX.sin(out[0..., (postNFft / 2 + 1)..., 0...])
        return stft.inverse(magnitude: spec, phase: phase)
    }
}

// MARK: - Kokoro Decoder (Top-level)

class KokoroAudioDecoder {
    private let encode: KokoroAdainResBlk1d
    private var decode: [KokoroAdainResBlk1d]
    private let f0Conv: KokoroConvWeighted
    private let nConv: KokoroConvWeighted
    private let asrRes: [KokoroConvWeighted]
    private let generator: KokoroGenerator

    init(weights: [String: MLXArray], config: KokoroConfig) {
        let dimIn = config.hiddenDim
        let styleDim = config.styleDim
        let istft = config.istftNet

        encode = KokoroAdainResBlk1d(
            weights: weights, keyPrefix: "decoder.encode",
            dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim
        )
        decode = [
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "decoder.decode.0", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "decoder.decode.1", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "decoder.decode.2", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "decoder.decode.3", dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: "true"),
        ]
        f0Conv = KokoroConvWeighted(
            weightG: weights["decoder.F0_conv.weight_g"]!, weightV: weights["decoder.F0_conv.weight_v"]!,
            bias: weights["decoder.F0_conv.bias"]!, stride: 2, padding: 1, groups: 1
        )
        nConv = KokoroConvWeighted(
            weightG: weights["decoder.N_conv.weight_g"]!, weightV: weights["decoder.N_conv.weight_v"]!,
            bias: weights["decoder.N_conv.bias"]!, stride: 2, padding: 1, groups: 1
        )
        asrRes = [KokoroConvWeighted(
            weightG: weights["decoder.asr_res.0.weight_g"]!, weightV: weights["decoder.asr_res.0.weight_v"]!,
            bias: weights["decoder.asr_res.0.bias"]!, padding: 0
        )]
        generator = KokoroGenerator(
            weights: weights, styleDim: styleDim,
            resblockKernelSizes: istft.resblockKernelSizes,
            upsampleRates: istft.upsampleRates,
            upsampleInitialChannel: istft.upsampleInitialChannel,
            resblockDilationSizes: istft.resblockDilationSizes,
            upsampleKernelSizes: istft.upsampleKernelSizes,
            genIstftNFft: istft.genIstftNFft,
            genIstftHopSize: istft.genIstftHopSize
        )
    }

    func callAsFunction(asr: MLXArray, f0Curve: MLXArray, n: MLXArray, s: MLXArray) -> MLXArray {
        let f0Swapped = MLX.swappedAxes(f0Curve.reshaped([f0Curve.shape[0], 1, f0Curve.shape[1]]), 2, 1)
        let f0 = MLX.swappedAxes(f0Conv(f0Swapped, conv: MLX.conv1d), 2, 1)

        let nSwapped = MLX.swappedAxes(n.reshaped([n.shape[0], 1, n.shape[1]]), 2, 1)
        let nProcessed = MLX.swappedAxes(nConv(nSwapped, conv: MLX.conv1d), 2, 1)

        var x = MLX.concatenated([asr, f0, nProcessed], axis: 1)
        x = encode(x: x, s: s)

        let asrResidual = MLX.swappedAxes(asrRes[0](MLX.swappedAxes(asr, 2, 1), conv: MLX.conv1d), 2, 1)
        var res = true

        for block in decode {
            if res { x = MLX.concatenated([x, asrResidual, f0, nProcessed], axis: 1) }
            x = block(x: x, s: s)
            if block.upsampleType != "none" { res = false }
        }

        return generator(x, s, f0Curve)
    }
}
