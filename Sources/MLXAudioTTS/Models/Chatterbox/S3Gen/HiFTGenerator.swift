// Ported from Python mlx-audio chatterbox s3gen/hifigan.py + f0_predictor.py

import Foundation
import MLX
import MLXNN

// MARK: - Conv1d Wrapper (matching Python Conv1dPT)

/// Wrapper matching Python's Conv1dPT which stores `self.conv = nn.Conv1d(...)`.
/// This adds the `.conv` in weight keys so `condnet.0.conv.weight` maps correctly.
/// Input/output in MLX (B,T,C) format — caller handles any B,C,T transpose.
class HiFiConv1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int,
         stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation, groups: groups)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

// MARK: - ConvTranspose1d Wrapper (matching Python ConvTranspose1dPT)

/// Wrapper matching Python's ConvTranspose1dPT which stores `self.conv = nn.ConvTranspose1d(...)`.
/// Input/output in MLX (B,T,C) format — caller handles any B,C,T transpose.
class HiFiConvTranspose1d: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int,
         stride: Int = 1, padding: Int = 0) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, stride: stride, padding: padding)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

// MARK: - F0 Predictor

/// Convolutional F0 predictor for pitch estimation.
class ConvRNNF0Predictor: Module {
    let numClass: Int
    let numCondNets: Int

    @ModuleInfo(key: "condnet") var condnets: [HiFiConv1d]
    @ModuleInfo(key: "classifier") var classifier: Linear

    init(numClass: Int = 1, inChannels: Int = 80, condChannels: Int = 512) {
        self.numClass = numClass
        self.numCondNets = 5

        // 5 Conv1d layers (wrapped as HiFiConv1d to match Python Conv1dPT nesting)
        var convLayers: [HiFiConv1d] = []
        for i in 0 ..< 5 {
            let inC = i == 0 ? inChannels : condChannels
            let conv = HiFiConv1d(
                inputChannels: inC, outputChannels: condChannels,
                kernelSize: 3, padding: 1)
            convLayers.append(conv)
        }

        self._condnets.wrappedValue = convLayers
        self._classifier.wrappedValue = Linear(condChannels, numClass)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T) -> need (B, T, C) for Conv1d
        var out = x.transposed(0, 2, 1) // (B, T, C)

        // Apply conv layers with ELU activation
        for i in 0 ..< numCondNets {
            out = elu(condnets[i](out))
        }

        // Classify
        out = classifier(out) // (B, T, 1)
        out = out.squeezed(axis: -1) // (B, T)

        return abs(out)
    }
}

// MARK: - Snake Activation

/// Snake activation: x + (1/alpha) * sin^2(alpha * x).
class Snake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    let alphaLogscale: Bool

    init(inFeatures: Int, alpha: Float = 1.0, alphaLogscale: Bool = false) {
        self.alphaLogscale = alphaLogscale
        if alphaLogscale {
            self._alpha.wrappedValue = MLXArray.zeros([inFeatures]) * alpha
        } else {
            self._alpha.wrappedValue = MLXArray.ones([inFeatures]) * alpha
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)
        var a = alpha.reshaped(1, -1, 1)

        if alphaLogscale {
            a = exp(a)
        }

        let minAlpha = Float(1e-4)
        let alphaSign = MLX.sign(a)
        let alphaAbs = abs(a)
        var alphaClamped = alphaSign * MLX.maximum(alphaAbs, MLXArray(minAlpha))
        alphaClamped = MLX.where(alphaAbs .< 1e-9, MLXArray(minAlpha), alphaClamped)

        return x + (1.0 / alphaClamped) * MLX.pow(MLX.sin(x * alphaClamped), MLXArray(2))
    }
}

// MARK: - ResBlock (HiFi-GAN)

/// Residual block for HiFi-GAN with multiple dilations.
class HiFiResBlock: Module {
    let channels: Int
    let numDilations: Int

    @ModuleInfo(key: "activations1") var activations1: [Snake]
    @ModuleInfo(key: "convs1") var convs1: [HiFiConv1d]
    @ModuleInfo(key: "activations2") var activations2: [Snake]
    @ModuleInfo(key: "convs2") var convs2: [HiFiConv1d]

    init(channels: Int = 512, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
        self.channels = channels
        self.numDilations = dilations.count

        var acts1: [Snake] = []
        var cs1: [HiFiConv1d] = []
        var acts2: [Snake] = []
        var cs2: [HiFiConv1d] = []

        for (_, dilation) in dilations.enumerated() {
            let padding1 = (kernelSize * dilation - dilation) / 2
            let padding2 = (kernelSize - 1) / 2

            // First conv with dilation (wrapped to match Python Conv1dPT key nesting)
            let conv1 = HiFiConv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: padding1, dilation: dilation)

            // Second conv (dilation=1)
            let conv2 = HiFiConv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: padding2)

            // Snake activations
            let act1 = Snake(inFeatures: channels)
            let act2 = Snake(inFeatures: channels)

            acts1.append(act1)
            cs1.append(conv1)
            acts2.append(act2)
            cs2.append(conv2)
        }

        self._activations1.wrappedValue = acts1
        self._convs1.wrappedValue = cs1
        self._activations2.wrappedValue = acts2
        self._convs2.wrappedValue = cs2
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x // (B, C, T)

        for i in 0 ..< numDilations {
            // activation1 -> conv1 -> activation2 -> conv2 -> residual
            var xt = activations1[i](out) // Snake in (B,C,T)
            xt = xt.transposed(0, 2, 1) // -> (B,T,C)
            xt = convs1[i](xt) // Conv1d in (B,T,C)
            xt = xt.transposed(0, 2, 1) // -> (B,C,T)
            xt = activations2[i](xt) // Snake in (B,C,T)
            xt = xt.transposed(0, 2, 1) // -> (B,T,C)
            xt = convs2[i](xt) // Conv1d in (B,T,C)
            xt = xt.transposed(0, 2, 1) // -> (B,C,T)
            out = xt + out
        }

        return out
    }
}

// MARK: - Source Module (Neural Source Filter)

/// Sine wave generator for neural source filter.
class SineGen: Module {
    let sineAmp: Float
    let noiseStd: Float
    let harmonicNum: Int
    let samplingRate: Int
    let voicedThreshold: Float
    let useInterpolation: Bool
    let upsampleScale: Int

    init(
        sampRate: Int, harmonicNum: Int = 0,
        sineAmp: Float = 0.1, noiseStd: Float = 0.003,
        voicedThreshold: Float = 0, useInterpolation: Bool = false,
        upsampleScale: Int = 1
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.useInterpolation = useInterpolation
        self.upsampleScale = upsampleScale
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // f0: (B, 1, T)
        let B = f0.dim(0), T = f0.dim(2)

        // Create harmonics multiplier: [1, 2, ..., harmonicNum+1]
        let harmonicMult = (MLXArray(1 ... (harmonicNum + 1)).asType(.float32))
            .reshaped(1, -1, 1)
        let fMat = f0 * harmonicMult / Float(samplingRate)

        // Phase computation
        let thetaMat = 2.0 * Float.pi * (MLX.cumsum(fMat, axis: -1) % 1.0)

        // Random initial phase (zero for fundamental)
        var phaseVec = MLXRandom.uniform(
            low: -Float.pi, high: Float.pi,
            [B, harmonicNum + 1, 1])
        let phaseMask = (MLXArray(0 ... harmonicNum).reshaped(1, -1, 1) .> 0).asType(.float32)
        phaseVec = phaseVec * phaseMask

        var sineWaves = sineAmp * MLX.sin(thetaMat + phaseVec)

        // Voiced/unvoiced mask
        let uv = (f0 .> voicedThreshold).asType(.float32)

        // Noise
        let noiseAmp = uv * noiseStd + (1.0 - uv) * sineAmp / 3.0
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

        sineWaves = sineWaves * uv + noise

        return (sineWaves, uv, noise)
    }
}

/// Source module combining sine generator with linear merge.
class SourceModuleHnNSF: Module {
    let sineAmp: Float
    let noiseStd: Float

    @ModuleInfo(key: "l_sin_gen") var lSinGen: SineGen
    @ModuleInfo(key: "l_linear") var lLinear: Linear

    init(
        samplingRate: Int, upsampleScale: Int,
        harmonicNum: Int = 0, sineAmp: Float = 0.1,
        addNoiseStd: Float = 0.003, voicedThreshold: Float = 0,
        useInterpolation: Bool = false
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = addNoiseStd

        self._lSinGen.wrappedValue = SineGen(
            sampRate: samplingRate, harmonicNum: harmonicNum,
            sineAmp: sineAmp, noiseStd: addNoiseStd,
            voicedThreshold: voicedThreshold,
            useInterpolation: useInterpolation,
            upsampleScale: upsampleScale)

        self._lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // x: (B, 1, T) F0 values
        // Generate sine harmonics — SineGen expects (B, 1, T) directly
        var (sineWavs, uv, _) = lSinGen(x)
        // sineWavs: (B, harmonics+1, T), uv: (B, 1, T)

        // Merge harmonics with linear layer
        // sineWavs: (B, H+1, T) -> need (B, T, H+1) for Linear
        let sineMerge = tanh(lLinear(sineWavs.transposed(0, 2, 1))) // (B, T, 1)

        let noise = MLXRandom.normal(uv.shape) * sineAmp / 3.0

        return (sineMerge.transposed(0, 2, 1), noise, uv) // (B, 1, T)
    }
}

// MARK: - STFT/ISTFT

/// Reverse an MLXArray along a given axis by gathering with reversed indices.
private func reverseAlongAxis(_ x: MLXArray, axis: Int) -> MLXArray {
    let n = x.dim(axis)
    let indices = MLXArray(Array((0 ..< n).reversed()).map { Int32($0) })
    return x.take(indices, axis: axis)
}

/// Short-Time Fourier Transform for HiFi-GAN.
func hifigan_stft(x: MLXArray, nFft: Int, hopLength: Int, window: MLXArray) -> (MLXArray, MLXArray)
{
    let B = x.dim(0), T = x.dim(1)

    // Reflect padding
    let padLength = nFft / 2
    let leftSlice = x[0..., 1 ..< (padLength + 1)]
    let leftPad = reverseAlongAxis(leftSlice, axis: 1)
    let rightSlice = x[0..., (-(padLength + 1)) ..< (-1)]
    let rightPad = reverseAlongAxis(rightSlice, axis: 1)
    let xPadded = MLX.concatenated([leftPad, x, rightPad], axis: 1)

    // Frame the signal
    let numFrames = (xPadded.dim(1) - nFft) / hopLength + 1
    let frameStarts = MLXArray(0 ..< numFrames) * hopLength
    let sampleOffsets = MLXArray(0 ..< nFft)
    let allIndices =
        frameStarts.expandedDimensions(axis: 1) + sampleOffsets.expandedDimensions(axis: 0)

    var frames = MLX.take(xPadded, allIndices.flattened(), axis: 1)
        .reshaped(B, numFrames, nFft)
    frames = frames.transposed(0, 2, 1)  // (B, nFft, numFrames)

    // Apply window
    let windowExp = window.reshaped(1, -1, 1)
    frames = frames * windowExp

    // FFT
    let fftResult = MLXFFT.fft(frames, axis: 1)
    let halfSpec = fftResult[0..., ..<(nFft / 2 + 1), 0...]

    // Extract real and imaginary parts
    let realPart = halfSpec.asType(.float32)  // Taking real part via float cast
    let negImagUnit = MLXArray(real: Float(0), imaginary: Float(-1))
    let imagPart = (halfSpec * negImagUnit).asType(.float32)

    return (realPart, imagPart)
}

/// Inverse STFT for HiFi-GAN waveform reconstruction.
func hifigan_istft(
    magnitude: MLXArray, phase: MLXArray,
    nFft: Int, hopLength: Int, window: MLXArray
) -> MLXArray {
    let clippedMag = MLX.minimum(magnitude, MLXArray(Float(1e2)))

    // Convert to complex
    let real = clippedMag * MLX.cos(phase)
    let imag = clippedMag * MLX.sin(phase)

    let B = real.dim(0), F = real.dim(1), numFrames = real.dim(2)

    // Construct complex spectrum
    let imagUnit = MLXArray(real: Float(0), imaginary: Float(1))
    let complexSpec = real + imagUnit * imag  // (B, F, numFrames) complex

    // irfft along frequency axis (axis=1) to get time-domain frames
    var frames = MLXFFT.irfft(complexSpec, axis: 1)  // (B, nFft, numFrames)

    // Apply window
    let windowExp = window.reshaped(1, -1, 1)
    frames = frames * windowExp

    // Overlap-add
    let outputLength = (numFrames - 1) * hopLength + nFft
    let frameOffsets = MLXArray(0 ..< numFrames) * hopLength
    let sampleIndices = MLXArray(0 ..< nFft)
    let indices =
        (frameOffsets.expandedDimensions(axis: 1) + sampleIndices.expandedDimensions(axis: 0))
        .flattened()

    // Window normalization
    let windowSq = window * window
    var windowSum = MLXArray.zeros([outputLength])
    let windowUpdates = tiled(windowSq, repetitions: [numFrames])
    windowSum = windowSum.at[indices].add(windowUpdates)
    windowSum = MLX.maximum(windowSum, MLXArray(Float(1e-8)))

    // Overlap add per batch
    let frameData = frames.transposed(0, 2, 1)  // (B, numFrames, nFft)
    let updates = frameData.reshaped(B, -1)  // (B, numFrames * nFft)

    let batchIndices = MLX.repeated(MLXArray(0 ..< B), count: numFrames * nFft)
    let flatIndices = tiled(indices, repetitions: [B])
    let linearIndices = batchIndices * outputLength + flatIndices

    var output = MLXArray.zeros([B * outputLength])
    output = output.at[linearIndices].add(updates.flattened())
    output = output.reshaped(B, outputLength)

    output = output / windowSum

    // Remove padding
    let padLength = nFft / 2
    output = output[0..., padLength ..< (outputLength - padLength)]

    return output
}

// MARK: - Hann Window

/// Create periodic Hann window matching scipy fftbins=True.
func hannWindowPeriodic(size: Int) -> MLXArray {
    let values = (0 ..< size).map { n in
        Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double(n) / Double(size))))
    }
    return MLXArray(values)
}

// MARK: - HiFTGenerator

/// HiFi-GAN with Neural Source Filter (HiFT-Net) generator.
class HiFTGenerator: Module {
    let outChannels: Int = 1
    let nbHarmonics: Int
    let samplingRate: Int
    let istftParams: [String: Int]
    let lreluSlope: Float
    let audioLimit: Float
    let numKernels: Int
    let numUpsamples: Int
    let f0UpsampleScale: Int

    @ModuleInfo(key: "m_source") var mSource: SourceModuleHnNSF
    @ModuleInfo(key: "conv_pre") var convPre: HiFiConv1d
    @ModuleInfo(key: "conv_post") var convPost: HiFiConv1d
    @ModuleInfo(key: "f0_predictor") var f0Predictor: ConvRNNF0Predictor
    @ModuleInfo(key: "ups") var ups: [HiFiConvTranspose1d]
    @ModuleInfo(key: "source_downs") var sourceDowners: [HiFiConv1d]
    @ModuleInfo(key: "source_resblocks") var sourceResblocks: [HiFiResBlock]
    @ModuleInfo(key: "resblocks") var resblocks: [HiFiResBlock]

    var stftWindow: MLXArray

    init(
        inChannels: Int = 80, baseChannels: Int = 512,
        nbHarmonics: Int = 8, samplingRate: Int = 24000,
        nsfAlpha: Float = 0.1, nsfSigma: Float = 0.003,
        nsfVoicedThreshold: Float = 10,
        upsampleRates: [Int] = [8, 5, 3],
        upsampleKernelSizes: [Int] = [16, 11, 7],
        istftParams: [String: Int] = ["n_fft": 16, "hop_len": 4],
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        sourceResblockKernelSizes: [Int] = [7, 7, 11],
        sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lreluSlope: Float = 0.1, audioLimit: Float = 0.99,
        useInterpolation: Bool = false
    ) {
        self.nbHarmonics = nbHarmonics
        self.samplingRate = samplingRate
        self.istftParams = istftParams
        self.lreluSlope = lreluSlope
        self.audioLimit = audioLimit
        self.numKernels = resblockKernelSizes.count
        self.numUpsamples = upsampleRates.count

        let upsampleScale = upsampleRates.reduce(1, *) * istftParams["hop_len"]!
        self.f0UpsampleScale = upsampleScale

        // Neural Source Filter
        self._mSource.wrappedValue = SourceModuleHnNSF(
            samplingRate: samplingRate, upsampleScale: upsampleScale,
            harmonicNum: nbHarmonics, sineAmp: nsfAlpha,
            addNoiseStd: nsfSigma, voicedThreshold: nsfVoicedThreshold,
            useInterpolation: useInterpolation)

        // Pre-convolution
        self._convPre.wrappedValue = HiFiConv1d(
            inputChannels: inChannels, outputChannels: baseChannels,
            kernelSize: 7, stride: 1, padding: 3)

        // Upsampling layers (ConvTranspose1d)
        let ch = baseChannels
        var upsArr: [HiFiConvTranspose1d] = []
        for i in 0 ..< upsampleRates.count {
            let u = upsampleRates[i]
            let k = upsampleKernelSizes[i]
            let inCh = ch / (1 << i)
            let outCh = ch / (1 << (i + 1))
            upsArr.append(
                HiFiConvTranspose1d(
                    inputChannels: inCh, outputChannels: outCh,
                    kernelSize: k, stride: u, padding: (k - u) / 2))
        }
        self._ups.wrappedValue = upsArr

        // Source downsampling and source resblocks
        // Python: downsample_rates = [1] + upsample_rates[::-1][:-1]
        // For [8, 5, 3]: reversed = [3, 5, 8], [:-1] = [3, 5], so [1, 3, 5]
        // downsample_cum = cumprod([1, 3, 5]) = [1, 3, 15]
        // reversed = [15, 3, 1]
        var downsampleRates = [1] + Array(upsampleRates.reversed().dropLast())
        var downsampleCum: [Int] = []
        var cumProd = 1
        for r in downsampleRates {
            cumProd *= r
            downsampleCum.append(cumProd)
        }
        let downsampleCumReversed = Array(downsampleCum.reversed())

        let nFftPlus2 = istftParams["n_fft"]! + 2
        var srcDowns: [HiFiConv1d] = []
        var srcResblks: [HiFiResBlock] = []

        for i in 0 ..< downsampleCumReversed.count {
            let u = downsampleCumReversed[i]
            let k = sourceResblockKernelSizes[i]
            let d = sourceResblockDilationSizes[i]
            let outCh = ch / (1 << (i + 1))

            if u == 1 {
                srcDowns.append(
                    HiFiConv1d(
                        inputChannels: nFftPlus2, outputChannels: outCh,
                        kernelSize: 1))
            } else {
                srcDowns.append(
                    HiFiConv1d(
                        inputChannels: nFftPlus2, outputChannels: outCh,
                        kernelSize: u * 2, stride: u, padding: u / 2))
            }
            srcResblks.append(HiFiResBlock(channels: outCh, kernelSize: k, dilations: d))
        }
        self._sourceDowners.wrappedValue = srcDowns
        self._sourceResblocks.wrappedValue = srcResblks

        // Main resblocks: for each upsample level, numKernels resblocks
        var mainResblks: [HiFiResBlock] = []
        for i in 0 ..< upsampleRates.count {
            let resCh = ch / (1 << (i + 1))
            for j in 0 ..< resblockKernelSizes.count {
                mainResblks.append(
                    HiFiResBlock(
                        channels: resCh, kernelSize: resblockKernelSizes[j],
                        dilations: resblockDilationSizes[j]))
            }
        }
        self._resblocks.wrappedValue = mainResblks

        // Post-convolution
        let finalChannels = ch / (1 << upsampleRates.count)
        self._convPost.wrappedValue = HiFiConv1d(
            inputChannels: finalChannels, outputChannels: nFftPlus2,
            kernelSize: 7, stride: 1, padding: 3)

        self.stftWindow = hannWindowPeriodic(size: istftParams["n_fft"]!)

        self._f0Predictor.wrappedValue = ConvRNNF0Predictor(
            numClass: 1, inChannels: inChannels, condChannels: 512)
    }

    /// Upsample F0 using nearest-neighbor interpolation.
    func f0Upsample(_ f0: MLXArray) -> MLXArray {
        return MLX.repeated(f0, count: f0UpsampleScale, axis: 2)
    }

    /// Decode mel-spectrogram + source to waveform.
    /// x: (B, C, T) mel spectrogram in channel-first format
    /// s: (B, 1, T_audio) source signal
    func decode(x: MLXArray, s: MLXArray) -> MLXArray {
        // STFT of source signal
        let squeezedS = s.squeezed(axis: 1)  // (B, T_audio)
        let nFft = istftParams["n_fft"]!
        let hopLen = istftParams["hop_len"]!

        let (sReal, sImag) = hifigan_stft(
            x: squeezedS, nFft: nFft, hopLength: hopLen, window: stftWindow)
        let sStft = MLX.concatenated([sReal, sImag], axis: 1)  // (B, nFft+2, frames)

        // Pre-convolution: x is (B, C, T) -> transpose to (B, T, C) for conv
        var h = x.transposed(0, 2, 1)  // (B, T, C)
        h = convPre(h)  // Conv1d in (B, T, C)
        h = h.transposed(0, 2, 1)  // (B, C, T)

        // Upsampling with source fusion and resblocks
        for i in 0 ..< numUpsamples {
            h = leakyRelu(h, negativeSlope: lreluSlope)
            // ConvTranspose: (B, C, T) -> (B, T, C) -> conv -> (B, T', C') -> (B, C', T')
            h = h.transposed(0, 2, 1)
            h = ups[i](h)
            h = h.transposed(0, 2, 1)

            // Reflection padding at last upsample step
            if i == numUpsamples - 1 {
                // Pad time dimension: pad 1 on the left
                // h is (B, C, T), need to pad axis 2
                h = MLX.padded(h, widths: [.init(0), .init(0), .init((1, 0))])
            }

            // Source fusion: downsample source STFT and add to h
            // sStft is (B, nFft+2, frames) in channel-first format
            // source_downs expects channel-first, but HiFiConv1d wraps Conv1d which expects (B, T, C)
            var si = sStft.transposed(0, 2, 1)  // (B, frames, nFft+2)
            si = sourceDowners[i](si)  // Conv1d in (B, T, C) -> (B, T', outCh)
            si = si.transposed(0, 2, 1)  // (B, outCh, T')
            si = sourceResblocks[i](si)  // ResBlock in (B, C, T)

            // Match time dimensions (take minimum to handle off-by-one)
            let minLen = min(h.dim(2), si.dim(2))
            h = h[0..., 0..., ..<minLen] + si[0..., 0..., ..<minLen]

            // Apply resblocks and average
            var xs: MLXArray? = nil
            for j in 0 ..< numKernels {
                let idx = i * numKernels + j
                if xs == nil {
                    xs = resblocks[idx](h)
                } else {
                    xs = xs! + resblocks[idx](h)
                }
            }
            h = xs! / Float(numKernels)
        }

        // Final processing
        h = leakyRelu(h, negativeSlope: lreluSlope)
        h = h.transposed(0, 2, 1)  // (B, T, C)
        h = convPost(h)  // Conv1d in (B, T, C)
        h = h.transposed(0, 2, 1)  // (B, nFft+2, T)

        // Split into magnitude and phase
        let nFftHalf = nFft / 2 + 1
        let mag = exp(h[0..., ..<nFftHalf, 0...])
        let phase = MLX.sin(h[0..., nFftHalf..., 0...])

        // Inverse STFT
        var output = hifigan_istft(
            magnitude: mag, phase: phase,
            nFft: nFft, hopLength: hopLen, window: stftWindow)

        output = MLX.clip(output, min: -audioLimit, max: audioLimit)
        return output
    }

    func callAsFunction(_ speechFeat: MLXArray, cacheSource: MLXArray? = nil)
        -> (MLXArray, MLXArray)
    {
        var cache = cacheSource ?? MLXArray.zeros([1, 1, 0])

        // Predict F0
        let f0 = f0Predictor(speechFeat)

        // Upsample F0: (B, T) -> (B, 1, T_up) via repeat along axis 2
        var s = f0Upsample(f0.expandedDimensions(axis: 1))  // (B, 1, T_up)

        // Generate source from F0 — mSource expects (B, 1, T)
        let (sourceSignal, _, _) = mSource(s)
        // sourceSignal: (B, 1, T_up)
        s = sourceSignal

        // Apply cache for streaming continuity
        if cache.dim(2) != 0 {
            let cacheLen = cache.dim(2)
            s = MLX.concatenated([cache, s[0..., 0..., cacheLen...]], axis: 2)
        }

        // Decode
        let generatedSpeech = decode(x: speechFeat, s: s)

        return (generatedSpeech, s)
    }
}
