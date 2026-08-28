import Foundation
@preconcurrency import MLX
import MLXNN

func dramaBoxConv1dBCT(_ conv: Conv1d, _ x: MLXArray) -> MLXArray {
    conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
}

func dramaBoxConvTransposed1dBCT(_ conv: ConvTransposed1d, _ x: MLXArray) -> MLXArray {
    conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
}

func dramaBoxReplicatePad1d(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    if left == 0 && right == 0 { return x }
    var parts: [MLXArray] = []
    if left > 0 {
        parts.append(MLX.broadcast(x[.ellipsis, 0..<1], to: [x.dim(0), x.dim(1), left]))
    }
    parts.append(x)
    if right > 0 {
        parts.append(MLX.broadcast(x[.ellipsis, (x.dim(2) - 1)...], to: [x.dim(0), x.dim(1), right]))
    }
    return concatenated(parts, axis: -1)
}

func dramaBoxDepthwiseConv1d(_ x: MLXArray, filter: MLXArray, stride: Int) -> MLXArray {
    let C = x.dim(1)
    let K = filter.dim(-1)
    let xCL = x.transposed(0, 2, 1)
    let w = MLX.broadcast(filter.reshaped([1, K, 1]), to: [C, K, 1]).asType(x.dtype)
    return MLX.conv1d(xCL, w, stride: stride, padding: 0, groups: C).transposed(0, 2, 1)
}

func dramaBoxDepthwiseConvTranspose1d(
    _ x: MLXArray,
    filter: MLXArray,
    stride: Int,
    scale: Float
) -> MLXArray {
    let C = x.dim(1)
    let K = filter.dim(-1)
    let xCL = x.transposed(0, 2, 1)
    let w = MLX.broadcast(filter.reshaped([1, K, 1]), to: [C, K, 1]).asType(x.dtype) * scale
    return MLX.convTransposed1d(xCL, w, stride: stride, padding: 0, groups: C).transposed(0, 2, 1)
}

final class DramaBoxSnakeBeta: Module {
    @ModuleInfo var alpha: MLXArray
    @ModuleInfo var beta: MLXArray
    let alphaLogscale: Bool
    let eps: Float

    init(channels: Int, alphaLogscale: Bool = true) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
        self.alphaLogscale = alphaLogscale
        self.eps = 1e-9
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var a = alpha.asType(x.dtype)
        var b = beta.asType(x.dtype)
        if alphaLogscale {
            a = MLX.exp(a)
            b = MLX.exp(b)
        }
        a = a.reshaped([1, a.dim(0), 1])
        b = b.reshaped([1, b.dim(0), 1])
        return x + (1.0 / (b + eps)) * MLX.sin(x * a) * MLX.sin(x * a)
    }
}

final class DramaBoxLowPassFilter1d: Module {
    @ModuleInfo var filter: MLXArray
    let kernelSize: Int
    let padLeft: Int
    let padRight: Int
    let stride: Int

    init(kernelSize: Int = 12, stride: Int = 1) {
        self.kernelSize = kernelSize
        let even = kernelSize.isMultiple(of: 2)
        self.padLeft = kernelSize / 2 - (even ? 1 : 0)
        self.padRight = kernelSize / 2
        self.stride = stride
        self._filter.wrappedValue = MLXArray.zeros([1, 1, kernelSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = dramaBoxReplicatePad1d(x, left: padLeft, right: padRight)
        return dramaBoxDepthwiseConv1d(padded, filter: filter, stride: stride)
    }
}

final class DramaBoxUpSample1d: Module {
    @ModuleInfo var filter: MLXArray
    let ratio: Int
    let kernelSize: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int

    init(ratio: Int = 2, kernelSize: Int = 12) {
        self.ratio = ratio
        self.kernelSize = kernelSize
        self.pad = kernelSize / ratio - 1
        self.padLeft = pad * ratio + (kernelSize - ratio) / 2
        self.padRight = pad * ratio + (kernelSize - ratio + 1) / 2
        self._filter.wrappedValue = MLXArray.zeros([1, 1, kernelSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = dramaBoxReplicatePad1d(x, left: pad, right: pad)
        y = dramaBoxDepthwiseConvTranspose1d(y, filter: filter, stride: ratio, scale: Float(ratio))
        let end = y.dim(-1) - padRight
        return y[.ellipsis, padLeft..<end]
    }
}

final class DramaBoxDownSample1d: Module {
    @ModuleInfo var lowpass: DramaBoxLowPassFilter1d

    init(ratio: Int = 2, kernelSize: Int = 12) {
        self._lowpass.wrappedValue = DramaBoxLowPassFilter1d(kernelSize: kernelSize, stride: ratio)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lowpass(x)
    }
}

final class DramaBoxActivation1d: Module {
    @ModuleInfo var act: DramaBoxSnakeBeta
    @ModuleInfo var upsample: DramaBoxUpSample1d
    @ModuleInfo var downsample: DramaBoxDownSample1d

    init(channels: Int, upRatio: Int = 2, downRatio: Int = 2, kernelSize: Int = 12) {
        self._act.wrappedValue = DramaBoxSnakeBeta(channels: channels)
        self._upsample.wrappedValue = DramaBoxUpSample1d(ratio: upRatio, kernelSize: kernelSize)
        self._downsample.wrappedValue = DramaBoxDownSample1d(ratio: downRatio, kernelSize: kernelSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(act(upsample(x)))
    }
}

struct DramaBoxHannSincUpsampler {
    let ratio: Int
    let kernelSize: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int
    let filter: MLXArray

    init(ratio: Int) {
        self.ratio = ratio
        let rolloff = 0.99
        let lowpassFilterWidth = 6.0
        let width = Int(ceil(lowpassFilterWidth / rolloff))
        self.kernelSize = 2 * width * ratio + 1
        self.pad = width
        self.padLeft = 2 * width * ratio
        self.padRight = kernelSize - ratio
        var filt = [Float](repeating: 0, count: kernelSize)
        for i in 0..<kernelSize {
            let t = (Double(i) / Double(ratio) - Double(width)) * rolloff
            let tClamped = min(max(t, -lowpassFilterWidth), lowpassFilterWidth)
            let window = pow(cos(tClamped * Double.pi / lowpassFilterWidth / 2.0), 2)
            let sinc = t == 0 ? 1.0 : sin(Double.pi * t) / (Double.pi * t)
            filt[i] = Float(sinc * window * rolloff / Double(ratio))
        }
        self.filter = MLXArray(filt).reshaped([1, 1, kernelSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = dramaBoxReplicatePad1d(x, left: pad, right: pad)
        y = dramaBoxDepthwiseConvTranspose1d(y, filter: filter, stride: ratio, scale: Float(ratio))
        let end = y.dim(-1) - padRight
        return y[.ellipsis, padLeft..<end]
    }
}

struct DramaBoxVocoderArgs: Sendable {
    var upsampleInitialChannel: Int
    var upsampleRates: [Int]
    var upsampleKernelSizes: [Int]
    var resblockKernelSizes: [Int]
    var resblockDilationSizes: [[Int]]
    var inChannels: Int
    var outChannels: Int
    var useTanhAtFinal: Bool
    var applyFinalActivation: Bool
    var useBiasAtFinal: Bool

    var numUpsamples: Int { upsampleRates.count }
    var numKernels: Int { resblockKernelSizes.count }
    var finalChannels: Int {
        var ch = upsampleInitialChannel
        for _ in upsampleRates { ch /= 2 }
        return ch
    }

    static let main = DramaBoxVocoderArgs(
        upsampleInitialChannel: 1536,
        upsampleRates: [5, 2, 2, 2, 2, 2],
        upsampleKernelSizes: [11, 4, 4, 4, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        inChannels: 128,
        outChannels: 2,
        useTanhAtFinal: false,
        applyFinalActivation: true,
        useBiasAtFinal: false
    )

    static let bwe = DramaBoxVocoderArgs(
        upsampleInitialChannel: 512,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [12, 11, 4, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        inChannels: 128,
        outChannels: 2,
        useTanhAtFinal: false,
        applyFinalActivation: false,
        useBiasAtFinal: false
    )
}

final class DramaBoxAMPBlock1: Module {
    @ModuleInfo var convs1: [Conv1d]
    @ModuleInfo var convs2: [Conv1d]
    @ModuleInfo var acts1: [DramaBoxActivation1d]
    @ModuleInfo var acts2: [DramaBoxActivation1d]

    init(channels: Int, kernelSize: Int, dilation: [Int] = [1, 3, 5]) {
        func padding(_ k: Int, _ d: Int) -> Int { (k * d - d) / 2 }
        self._convs1.wrappedValue = dilation.map { d in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: padding(kernelSize, d),
                dilation: d,
                bias: true
            )
        }
        self._convs2.wrappedValue = dilation.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: padding(kernelSize, 1),
                dilation: 1,
                bias: true
            )
        }
        self._acts1.wrappedValue = dilation.map { _ in DramaBoxActivation1d(channels: channels) }
        self._acts2.wrappedValue = dilation.map { _ in DramaBoxActivation1d(channels: channels) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for i in 0..<convs1.count {
            var xt = acts1[i](y)
            xt = dramaBoxConv1dBCT(convs1[i], xt)
            xt = acts2[i](xt)
            xt = dramaBoxConv1dBCT(convs2[i], xt)
            y = y + xt
        }
        return y
    }
}

final class DramaBoxVocoder: Module {
    let args: DramaBoxVocoderArgs
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo var ups: [ConvTransposed1d]
    @ModuleInfo var resblocks: [DramaBoxAMPBlock1]
    @ModuleInfo(key: "act_post") var actPost: DramaBoxActivation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    init(_ args: DramaBoxVocoderArgs) {
        self.args = args
        var ch = args.upsampleInitialChannel
        self._convPre.wrappedValue = Conv1d(
            inputChannels: args.inChannels, outputChannels: ch, kernelSize: 7, stride: 1, padding: 3, bias: true
        )
        var ups: [ConvTransposed1d] = []
        for (stride, ks) in zip(args.upsampleRates, args.upsampleKernelSizes) {
            ups.append(
                ConvTransposed1d(
                    inputChannels: ch,
                    outputChannels: ch / 2,
                    kernelSize: ks,
                    stride: stride,
                    padding: (ks - stride) / 2
                )
            )
            ch /= 2
        }
        self._ups.wrappedValue = ups
        var resblocks: [DramaBoxAMPBlock1] = []
        ch = args.upsampleInitialChannel
        for _ in 0..<args.numUpsamples {
            ch /= 2
            for (k, d) in zip(args.resblockKernelSizes, args.resblockDilationSizes) {
                resblocks.append(DramaBoxAMPBlock1(channels: ch, kernelSize: k, dilation: d))
            }
        }
        self._resblocks.wrappedValue = resblocks
        self._actPost.wrappedValue = DramaBoxActivation1d(channels: args.finalChannels)
        self._convPost.wrappedValue = Conv1d(
            inputChannels: args.finalChannels,
            outputChannels: args.outChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            bias: args.useBiasAtFinal
        )
        super.init()
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x: MLXArray
        if mel.ndim == 4 {
            x = mel.transposed(0, 1, 3, 2)
            let B = x.dim(0)
            let S = x.dim(1)
            let F = x.dim(2)
            let T = x.dim(3)
            x = x.reshaped(B, S * F, T)
        } else {
            x = mel.transposed(0, 2, 1)
        }
        x = dramaBoxConv1dBCT(convPre, x)
        for i in 0..<args.numUpsamples {
            x = dramaBoxConvTransposed1dBCT(ups[i], x)
            let start = i * args.numKernels
            let end = start + args.numKernels
            var outs: [MLXArray] = []
            for idx in start..<end {
                outs.append(resblocks[idx](x))
            }
            x = MLX.mean(stacked(outs, axis: 0), axis: 0)
        }
        x = actPost(x)
        x = dramaBoxConv1dBCT(convPost, x)
        if args.applyFinalActivation {
            if args.useTanhAtFinal {
                x = MLX.tanh(x)
            } else {
                x = MLX.clip(x, min: -1, max: 1)
            }
        }
        return x
    }
}

final class DramaBoxSTFTFn: Module {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int
    @ModuleInfo(key: "forward_basis") var forwardBasis: MLXArray
    @ModuleInfo(key: "inverse_basis") var inverseBasis: MLXArray

    init(filterLength: Int, hopLength: Int, winLength: Int) {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
        let nFreqs = filterLength / 2 + 1
        self._forwardBasis.wrappedValue = MLXArray.zeros([2 * nFreqs, 1, filterLength])
        self._inverseBasis.wrappedValue = MLXArray.zeros([2 * nFreqs, 1, filterLength])
        super.init()
    }

    func callAsFunction(_ y: MLXArray) -> (MLXArray, MLXArray) {
        var work = y.ndim == 2 ? y.expandedDimensions(axis: 1) : y
        let leftPad = max(0, winLength - hopLength)
        if leftPad > 0 {
            let pad = MLXArray.zeros([work.dim(0), work.dim(1), leftPad], dtype: work.dtype)
            work = concatenated([pad, work], axis: -1)
        }
        let yCL = work.transposed(0, 2, 1)
        let w = forwardBasis.transposed(0, 2, 1)
        var spec = MLX.conv1d(yCL, w.asType(yCL.dtype), stride: hopLength, padding: 0)
        spec = spec.transposed(0, 2, 1)
        let nFreqs = spec.dim(1) / 2
        let real = spec[0..., 0..<nFreqs]
        let imag = spec[0..., nFreqs...]
        let magnitude = MLX.sqrt(real * real + imag * imag)
        let phase = MLX.atan2(imag.asType(.float32), real.asType(.float32)).asType(real.dtype)
        return (magnitude, phase)
    }
}

final class DramaBoxMelSTFT: Module {
    @ModuleInfo(key: "stft_fn") var stftFn: DramaBoxSTFTFn
    @ModuleInfo(key: "mel_basis") var melBasis: MLXArray

    init(filterLength: Int = 512, hopLength: Int = 80, winLength: Int = 512, nMelChannels: Int = 64) {
        self._stftFn.wrappedValue = DramaBoxSTFTFn(
            filterLength: filterLength, hopLength: hopLength, winLength: winLength
        )
        let nFreqs = filterLength / 2 + 1
        self._melBasis.wrappedValue = MLXArray.zeros([nMelChannels, nFreqs])
        super.init()
    }

    func melSpectrogram(_ y: MLXArray) -> MLXArray {
        let (magnitude, _) = stftFn(y)
        let mel = MLX.matmul(melBasis.asType(magnitude.dtype), magnitude)
        return MLX.log(MLX.maximum(mel, MLXArray(Float(1e-5))))
    }
}

final class DramaBoxVocoderWithBWE: Module {
    let inputSamplingRate: Int
    let outputSamplingRate: Int
    let hopLength: Int
    let ratio: Int
    let skipResampler: DramaBoxHannSincUpsampler

    @ModuleInfo var vocoder: DramaBoxVocoder
    @ModuleInfo(key: "bwe_generator") var bweGenerator: DramaBoxVocoder
    @ModuleInfo(key: "mel_stft") var melSTFT: DramaBoxMelSTFT

    init(
        mainArgs: DramaBoxVocoderArgs = .main,
        bweArgs: DramaBoxVocoderArgs = .bwe,
        inputSamplingRate: Int = 16_000,
        outputSamplingRate: Int = 48_000,
        hopLength: Int = 80,
        nFft: Int = 512,
        winLength: Int = 512,
        nMelChannels: Int = 64
    ) {
        self.inputSamplingRate = inputSamplingRate
        self.outputSamplingRate = outputSamplingRate
        self.hopLength = hopLength
        self.ratio = outputSamplingRate / inputSamplingRate
        self.skipResampler = DramaBoxHannSincUpsampler(ratio: ratio)
        self._vocoder.wrappedValue = DramaBoxVocoder(mainArgs)
        self._bweGenerator.wrappedValue = DramaBoxVocoder(bweArgs)
        self._melSTFT.wrappedValue = DramaBoxMelSTFT(
            filterLength: nFft, hopLength: hopLength, winLength: winLength, nMelChannels: nMelChannels
        )
        super.init()
    }

    func callAsFunction(_ melSpec: MLXArray) -> MLXArray {
        let inputDtype = melSpec.dtype
        let mel32 = melSpec.asType(.float32)
        var x = vocoder(mel32)
        let tLow = x.dim(-1)
        let outputLength = tLow * outputSamplingRate / inputSamplingRate
        let remainder = tLow % hopLength
        if remainder != 0 {
            let pad = MLXArray.zeros([x.dim(0), x.dim(1), hopLength - remainder], dtype: x.dtype)
            x = concatenated([x, pad], axis: -1)
        }
        let B = x.dim(0)
        let C = x.dim(1)
        let T = x.dim(2)
        let flat = x.reshaped(B * C, T)
        let mel = melSTFT.melSpectrogram(flat)
        let stereoMel = mel.reshaped(B, C, mel.dim(1), mel.dim(2))
        let melForBWE = stereoMel.transposed(0, 1, 3, 2)
        var residual = bweGenerator(melForBWE)
        var skip = skipResampler(x)
        let targetLen = min(skip.dim(-1), residual.dim(-1))
        skip = skip[.ellipsis, 0..<targetLen]
        residual = residual[.ellipsis, 0..<targetLen]
        let out = MLX.clip(residual + skip, min: -1, max: 1)[.ellipsis, 0..<outputLength]
        return out.asType(inputDtype)
    }
}

func loadDramaBoxVocoderWeights(_ model: DramaBoxVocoderWithBWE, state: [String: MLXArray]) throws {
    let prefix = "vocoder."
    var sub: [String: MLXArray] = [:]
    for (key, value) in state where key.hasPrefix(prefix) {
        let tail = String(key.dropFirst(prefix.count))
        var tensor = value
        if tensor.ndim == 3 && tail.hasSuffix(".weight") {
            if tail.contains(".ups.") {
                tensor = tensor.transposed(1, 2, 0)
            } else {
                tensor = tensor.transposed(0, 2, 1)
            }
        }
        sub[tail] = tensor
    }
    guard !sub.isEmpty else {
        throw DramaBoxError.generationFailed("No vocoder weights")
    }
    try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
}
