import Foundation
@preconcurrency import MLX
import MLXNN

struct DramaBoxAudioVAEConfig: Sendable {
    var inChannels: Int = 2
    var outCh: Int = 2
    var zChannels: Int = 8
    var ch: Int = 128
    var chMult: [Int] = [1, 2, 4]
    var numResBlocks: Int = 2
    var doubleZ: Bool = true
    var samplingRate: Int = 16_000
    var melBins: Int = 64
    var melHopLength: Int = 160
    var nFft: Int = 1024
    var audioLatentDownsampleFactor: Int = 4

    var numResolutions: Int { chMult.count }
}

func dramaBoxPixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let orig = x.dtype
    let x32 = x.asType(.float32)
    let meanSq = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    return (x32 / MLX.sqrt(meanSq + eps)).asType(orig)
}

final class DramaBoxCausalConv2d: Module {
    let padTop: Int
    let padBottom: Int
    let padLeft: Int
    let padRight: Int
    @ModuleInfo var conv: Conv2d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        let padH = (kernelSize - 1) * dilation
        let padW = (kernelSize - 1) * dilation
        self.padTop = padH
        self.padBottom = 0
        self.padLeft = padW / 2
        self.padRight = padW - padW / 2
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(0),
            dilation: IntOrPair(dilation),
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(
            x,
            widths: [.init(0), .init((padTop, padBottom)), .init((padLeft, padRight)), .init(0)]
        )
        return conv(padded)
    }
}

final class DramaBoxVAEResnetBlock: Module {
    @ModuleInfo var conv1: DramaBoxCausalConv2d
    @ModuleInfo var conv2: DramaBoxCausalConv2d
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: DramaBoxCausalConv2d?

    init(inChannels: Int, outChannels: Int) {
        self._conv1.wrappedValue = DramaBoxCausalConv2d(
            inChannels: inChannels, outChannels: outChannels, kernelSize: 3
        )
        self._conv2.wrappedValue = DramaBoxCausalConv2d(
            inChannels: outChannels, outChannels: outChannels, kernelSize: 3
        )
        if inChannels != outChannels {
            self._ninShortcut.wrappedValue = DramaBoxCausalConv2d(
                inChannels: inChannels, outChannels: outChannels, kernelSize: 1
            )
        } else {
            self._ninShortcut.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = dramaBoxPixelNorm(x)
        h = MLXNN.silu(h)
        h = conv1(h)
        h = dramaBoxPixelNorm(h)
        h = MLXNN.silu(h)
        h = conv2(h)
        let skip = ninShortcut?(x) ?? x
        return skip + h
    }
}

final class DramaBoxVAEDownsample: Module {
    @ModuleInfo var conv: Conv2d

    init(_ channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [.init(0), .init((2, 0)), .init((0, 1)), .init(0)])
        return conv(padded)
    }
}

final class DramaBoxVAEUpsample: Module {
    @ModuleInfo var conv: DramaBoxCausalConv2d

    init(_ channels: Int) {
        self._conv.wrappedValue = DramaBoxCausalConv2d(
            inChannels: channels, outChannels: channels, kernelSize: 3
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.repeated(x, count: 2, axis: 1)
        y = MLX.repeated(y, count: 2, axis: 2)
        y = conv(y)
        return y[0..., 1..., 0..., 0...]
    }
}

final class DramaBoxVAEMidBlock: Module {
    @ModuleInfo(key: "block_1") var block1: DramaBoxVAEResnetBlock
    @ModuleInfo(key: "block_2") var block2: DramaBoxVAEResnetBlock

    init(channels: Int) {
        self._block1.wrappedValue = DramaBoxVAEResnetBlock(inChannels: channels, outChannels: channels)
        self._block2.wrappedValue = DramaBoxVAEResnetBlock(inChannels: channels, outChannels: channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        block2(block1(x))
    }
}

final class DramaBoxVAEDownStage: Module {
    @ModuleInfo var block: [DramaBoxVAEResnetBlock]
    @ModuleInfo var downsample: DramaBoxVAEDownsample?

    init(inChannels: Int, outChannels: Int, numBlocks: Int, withDownsample: Bool) {
        var blocks: [DramaBoxVAEResnetBlock] = []
        var chIn = inChannels
        for _ in 0..<numBlocks {
            blocks.append(DramaBoxVAEResnetBlock(inChannels: chIn, outChannels: outChannels))
            chIn = outChannels
        }
        self._block.wrappedValue = blocks
        self._downsample.wrappedValue = withDownsample ? DramaBoxVAEDownsample(outChannels) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for b in block { h = b(h) }
        if let downsample { h = downsample(h) }
        return h
    }
}

final class DramaBoxVAEUpStage: Module {
    @ModuleInfo var block: [DramaBoxVAEResnetBlock]
    @ModuleInfo var upsample: DramaBoxVAEUpsample?

    init(inChannels: Int, outChannels: Int, numBlocks: Int, withUpsample: Bool) {
        var blocks: [DramaBoxVAEResnetBlock] = []
        var chIn = inChannels
        for _ in 0..<numBlocks {
            blocks.append(DramaBoxVAEResnetBlock(inChannels: chIn, outChannels: outChannels))
            chIn = outChannels
        }
        self._block.wrappedValue = blocks
        self._upsample.wrappedValue = withUpsample ? DramaBoxVAEUpsample(outChannels) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for b in block { h = b(h) }
        if let upsample { h = upsample(h) }
        return h
    }
}

final class DramaBoxAudioEncoder: Module {
    let config: DramaBoxAudioVAEConfig
    @ModuleInfo(key: "conv_in") var convIn: DramaBoxCausalConv2d
    @ModuleInfo var down: [DramaBoxVAEDownStage]
    @ModuleInfo var mid: DramaBoxVAEMidBlock
    @ModuleInfo(key: "conv_out") var convOut: DramaBoxCausalConv2d

    init(_ config: DramaBoxAudioVAEConfig) {
        self.config = config
        let ch = config.ch
        self._convIn.wrappedValue = DramaBoxCausalConv2d(
            inChannels: config.inChannels, outChannels: ch, kernelSize: 3
        )
        var stages: [DramaBoxVAEDownStage] = []
        var blockIn = ch
        for level in 0..<config.numResolutions {
            let blockOut = ch * config.chMult[level]
            stages.append(
                DramaBoxVAEDownStage(
                    inChannels: blockIn,
                    outChannels: blockOut,
                    numBlocks: config.numResBlocks,
                    withDownsample: level != config.numResolutions - 1
                )
            )
            blockIn = blockOut
        }
        self._down.wrappedValue = stages
        self._mid.wrappedValue = DramaBoxVAEMidBlock(channels: blockIn)
        let outChannels = config.doubleZ ? 2 * config.zChannels : config.zChannels
        self._convOut.wrappedValue = DramaBoxCausalConv2d(
            inChannels: blockIn, outChannels: outChannels, kernelSize: 3
        )
        super.init()
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var h = convIn(mel)
        for stage in down { h = stage(h) }
        h = mid(h)
        h = dramaBoxPixelNorm(h)
        h = MLXNN.silu(h)
        return convOut(h)
    }
}

final class DramaBoxAudioDecoder: Module {
    let config: DramaBoxAudioVAEConfig
    @ModuleInfo(key: "conv_in") var convIn: DramaBoxCausalConv2d
    @ModuleInfo var mid: DramaBoxVAEMidBlock
    @ModuleInfo var up: [DramaBoxVAEUpStage]
    @ModuleInfo(key: "conv_out") var convOut: DramaBoxCausalConv2d

    init(_ config: DramaBoxAudioVAEConfig) {
        self.config = config
        let ch = config.ch
        var blockIn = ch * config.chMult[config.numResolutions - 1]
        self._convIn.wrappedValue = DramaBoxCausalConv2d(
            inChannels: config.zChannels, outChannels: blockIn, kernelSize: 3
        )
        self._mid.wrappedValue = DramaBoxVAEMidBlock(channels: blockIn)
        var stages = [DramaBoxVAEUpStage?](repeating: nil, count: config.numResolutions)
        for level in (0..<config.numResolutions).reversed() {
            let blockOut = ch * config.chMult[level]
            stages[level] = DramaBoxVAEUpStage(
                inChannels: blockIn,
                outChannels: blockOut,
                numBlocks: config.numResBlocks + 1,
                withUpsample: level != 0
            )
            blockIn = blockOut
        }
        self._up.wrappedValue = stages.map { $0! }
        self._convOut.wrappedValue = DramaBoxCausalConv2d(
            inChannels: blockIn, outChannels: config.outCh, kernelSize: 3
        )
        super.init()
    }

    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var h = convIn(latent)
        h = mid(h)
        for level in (0..<config.numResolutions).reversed() {
            h = up[level](h)
        }
        h = dramaBoxPixelNorm(h)
        h = MLXNN.silu(h)
        return convOut(h)
    }
}

final class DramaBoxPerChannelStatistics: Module {
    @ModuleInfo(key: "mean_of_means") var meanOfMeans: MLXArray
    @ModuleInfo(key: "std_of_means") var stdOfMeans: MLXArray

    init(dim: Int = 128) {
        self._meanOfMeans.wrappedValue = MLXArray.zeros([dim])
        self._stdOfMeans.wrappedValue = MLXArray.ones([dim])
        super.init()
    }

    func normalize(_ x: MLXArray) -> MLXArray {
        (x - meanOfMeans.asType(x.dtype)) / stdOfMeans.asType(x.dtype)
    }

    func unNormalize(_ x: MLXArray) -> MLXArray {
        x * stdOfMeans.asType(x.dtype) + meanOfMeans.asType(x.dtype)
    }
}

final class DramaBoxAudioVAE: Module {
    let config: DramaBoxAudioVAEConfig
    @ModuleInfo var encoder: DramaBoxAudioEncoder
    @ModuleInfo var decoder: DramaBoxAudioDecoder
    @ModuleInfo(key: "per_channel_statistics") var perChannelStatistics: DramaBoxPerChannelStatistics

    init(_ config: DramaBoxAudioVAEConfig = DramaBoxAudioVAEConfig()) {
        self.config = config
        self._encoder.wrappedValue = DramaBoxAudioEncoder(config)
        self._decoder.wrappedValue = DramaBoxAudioDecoder(config)
        let latentMelBins = config.melBins / (1 << (config.numResolutions - 1))
        self._perChannelStatistics.wrappedValue = DramaBoxPerChannelStatistics(
            dim: config.zChannels * latentMelBins
        )
        super.init()
    }

    func encode(_ mel: MLXArray) -> MLXArray {
        let melCL = mel.transposed(0, 2, 3, 1)
        let raw = encoder(melCL)
        let meansCL = raw[.ellipsis, 0..<config.zChannels]
        let means = meansCL.transposed(0, 3, 1, 2)
        let patched = DramaBoxAudioPatchifier.patchify(means)
        let normed = perChannelStatistics.normalize(patched)
        return DramaBoxAudioPatchifier.unpatchify(
            normed, channels: config.zChannels, melBins: means.dim(3)
        )
    }

    func decode(_ latent: MLXArray) -> MLXArray {
        let patched = DramaBoxAudioPatchifier.patchify(latent)
        let denormed = perChannelStatistics.unNormalize(patched)
        let unp = DramaBoxAudioPatchifier.unpatchify(
            denormed, channels: latent.dim(1), melBins: latent.dim(3)
        )
        let outCL = decoder(unp.transposed(0, 2, 3, 1))
        return outCL.transposed(0, 3, 1, 2)
    }
}

func loadDramaBoxAudioVAEWeights(_ model: DramaBoxAudioVAE, state: [String: MLXArray]) throws {
    let prefix = "audio_vae."
    var sub: [String: MLXArray] = [:]
    for (key, value) in state where key.hasPrefix(prefix) {
        var tail = String(key.dropFirst(prefix.count))
        if tail.hasPrefix("per_channel_statistics.") {
            tail = tail
                .replacingOccurrences(of: "mean-of-means", with: "mean_of_means")
                .replacingOccurrences(of: "std-of-means", with: "std_of_means")
        }
        var tensor = value
        if tensor.ndim == 4 && tail.hasSuffix(".weight") {
            tensor = tensor.transposed(0, 2, 3, 1)
        }
        sub[tail] = tensor
    }
    guard !sub.isEmpty else {
        throw DramaBoxError.generationFailed("No audio_vae weights")
    }
    try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
}
