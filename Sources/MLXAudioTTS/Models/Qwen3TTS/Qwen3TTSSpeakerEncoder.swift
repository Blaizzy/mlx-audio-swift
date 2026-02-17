@preconcurrency import MLX
import MLXNN

// MARK: - Speaker Encoder Utilities

private func reflectPad1D(_ x: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return x }
    let timeLength = x.dim(1)
    guard timeLength > 0 else { return x }
    let clampedPad = Swift.min(pad, Swift.max(timeLength - 1, 0))
    guard clampedPad > 0 else { return x }

    var leftPieces: [MLXArray] = []
    leftPieces.reserveCapacity(clampedPad)
    if clampedPad > 0 {
        for i in stride(from: clampedPad, through: 1, by: -1) {
            leftPieces.append(x[0..., i ..< (i + 1), 0...])
        }
    }

    var rightPieces: [MLXArray] = []
    rightPieces.reserveCapacity(clampedPad)
    if clampedPad > 0, timeLength > 1 {
        let rightStart = timeLength - clampedPad - 1
        let rightFrom = timeLength - 2
        let rightTo = Swift.max(rightStart, 0)
        for i in stride(from: rightFrom, through: rightTo, by: -1) {
            rightPieces.append(x[0..., i ..< (i + 1), 0...])
        }
    }

    let left = leftPieces.isEmpty ? MLXArray([]) : concatenated(leftPieces, axis: 1)
    let right = rightPieces.isEmpty ? MLXArray([]) : concatenated(rightPieces, axis: 1)
    return concatenated([left, x, right], axis: 1)
}

// MARK: - Time Delay Net Block

final class TimeDelayNetBlock: Module {
    let pad: Int
    @ModuleInfo var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
        self.pad = (kernelSize - 1) * dilation / 2
        _conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        out = reflectPad1D(out, pad: pad)
        out = conv(out)
        return relu(out.transposed(0, 2, 1))
    }
}

// MARK: - Res2Net Block

final class Res2NetBlock: Module {
    let scale: Int
    let inChannel: Int
    let hiddenChannel: Int
    @ModuleInfo var blocks: [TimeDelayNetBlock]

    init(inChannels: Int, outChannels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        self.scale = scale
        self.inChannel = inChannels / scale
        self.hiddenChannel = outChannels / scale

        let blockCount = max(0, scale - 1)
        var builtBlocks: [TimeDelayNetBlock] = []
        builtBlocks.reserveCapacity(blockCount)
        for _ in 0 ..< blockCount {
            builtBlocks.append(
                TimeDelayNetBlock(
                    inChannels: inChannel,
                    outChannels: hiddenChannel,
                    kernelSize: kernelSize,
                    dilation: dilation
                )
            )
        }
        _blocks.wrappedValue = builtBlocks
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = split(x, parts: scale, axis: 1)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(scale)

        var outputPart: MLXArray? = nil
        for i in 0 ..< scale {
            if i == 0 {
                outputPart = chunks[i]
            } else if i == 1 {
                outputPart = blocks[i - 1](chunks[i])
            } else {
                if let previousPart = outputPart {
                    outputPart = blocks[i - 1](chunks[i] + previousPart)
                }
            }
            if let part = outputPart {
                outputs.append(part)
            }
        }

        return concatenated(outputs, axis: 1)
    }
}

// MARK: - Squeeze-and-Excitation

final class SqueezeExcitationBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(inChannels: Int, seChannels: Int, outChannels: Int) {
        _conv1.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: seChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        _conv2.wrappedValue = Conv1d(
            inputChannels: seChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var se = mean(x, axis: 2, keepDims: true)
        se = se.transposed(0, 2, 1)
        se = relu(conv1(se))
        se = sigmoid(conv2(se))
        se = se.transposed(0, 2, 1)
        return x * se
    }
}

// MARK: - SE + Res2Net + TDNN block

final class SqueezeExcitationRes2NetBlock: Module {
    let outChannels: Int
    @ModuleInfo var tdnn1: TimeDelayNetBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: TimeDelayNetBlock
    @ModuleInfo(key: "se_block") var seBlock: SqueezeExcitationBlock

    init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        self.outChannels = outChannels

        _tdnn1.wrappedValue = TimeDelayNetBlock(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        _res2netBlock.wrappedValue = Res2NetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            scale: res2netScale,
            kernelSize: kernelSize,
            dilation: dilation
        )
        _tdnn2.wrappedValue = TimeDelayNetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        _seBlock.wrappedValue = SqueezeExcitationBlock(
            inChannels: outChannels,
            seChannels: seChannels,
            outChannels: outChannels
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = tdnn1(x)
        out = res2netBlock(out)
        out = tdnn2(out)
        out = seBlock(out)
        return out + residual
    }
}

// MARK: - Attentive Statistics Pooling

final class AttentiveStatisticsPooling: Module {
    let eps: Float = 1e-12
    @ModuleInfo var tdnn: TimeDelayNetBlock
    @ModuleInfo var conv: Conv1d

    init(channels: Int, attentionChannels: Int = 128) {
        _tdnn.wrappedValue = TimeDelayNetBlock(
            inChannels: channels * 3,
            outChannels: attentionChannels,
            kernelSize: 1,
            dilation: 1
        )
        _conv.wrappedValue = Conv1d(
            inputChannels: attentionChannels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let channels = x.dim(1)
        let sequenceLength = x.dim(2)

        let meanTensor = mean(x, axis: 2, keepDims: true)
        let centered = x - meanTensor
        let stdTensor = sqrt(mean(centered * centered, axis: 2, keepDims: true) + eps)

        let meanExpanded = broadcast(meanTensor, to: [batch, channels, sequenceLength])
        let stdExpanded = broadcast(stdTensor, to: [batch, channels, sequenceLength])

        var attention = concatenated([x, meanExpanded, stdExpanded], axis: 1)
        attention = tdnn(attention)
        attention = tanh(attention)
        attention = conv(attention.transposed(0, 2, 1)).transposed(0, 2, 1)
        attention = softmax(attention, axis: 2)

        let meanOut = sum(attention * x, axis: 2, keepDims: true)
        let varOut = sum(attention * (x - meanOut) * (x - meanOut), axis: 2, keepDims: true)
        let stdOut = sqrt(clip(varOut, min: eps))

        return concatenated([meanOut, stdOut], axis: 1)
    }
}

// MARK: - Speaker Encoder

final class Qwen3TTSSpeakerEncoder: Module {
    let config: Qwen3TTSSpeakerEncoderConfig
    let channels: [Int]

    @ModuleInfo var blocks: [Module]
    @ModuleInfo var mfa: TimeDelayNetBlock
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo var fc: Conv1d

    init(config: Qwen3TTSSpeakerEncoderConfig) {
        self.config = config
        self.channels = config.encChannels

        var builtBlocks: [Module] = []

        builtBlocks.append(
            TimeDelayNetBlock(
                inChannels: config.melDim,
                outChannels: config.encChannels[0],
                kernelSize: config.encKernelSizes[0],
                dilation: config.encDilations[0]
            )
        )

        if config.encChannels.count > 1 {
            for i in 1 ..< config.encChannels.count - 1 {
                builtBlocks.append(
                    SqueezeExcitationRes2NetBlock(
                        inChannels: config.encChannels[i - 1],
                        outChannels: config.encChannels[i],
                        res2netScale: config.encRes2netScale,
                        seChannels: config.encSeChannels,
                        kernelSize: config.encKernelSizes[i],
                        dilation: config.encDilations[i]
                    )
                )
            }
        }

        _blocks.wrappedValue = builtBlocks

        _mfa.wrappedValue = TimeDelayNetBlock(
            inChannels: config.encChannels.last ?? 1,
            outChannels: config.encChannels.last ?? 1,
            kernelSize: config.encKernelSizes.last ?? 1,
            dilation: config.encDilations.last ?? 1
        )

        _asp.wrappedValue = AttentiveStatisticsPooling(
            channels: config.encChannels.last ?? 1,
            attentionChannels: config.encAttentionChannels
        )

        _fc.wrappedValue = Conv1d(
            inputChannels: (config.encChannels.last ?? 1) * 2,
            outputChannels: config.encDim,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var states = x.transposed(0, 2, 1)

        var hiddenStates: [MLXArray] = []
        for block in blocks {
            if let layer = block as? TimeDelayNetBlock {
                states = layer(states)
            } else if let layer = block as? SqueezeExcitationRes2NetBlock {
                states = layer(states)
            } else {
                fatalError("Unsupported speaker encoder block type: \(type(of: block))")
            }
            hiddenStates.append(states)
        }

        if hiddenStates.count >= 2 {
            states = concatenated(Array(hiddenStates[1...]), axis: 1)
        }

        states = mfa(states)
        states = asp(states)
        states = fc(states.transposed(0, 2, 1)).transposed(0, 2, 1)
        return states.squeezed(axis: -1)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (k, vOrig) in weights {
            guard let key = stripSpeakerEncoderPrefix(from: k) else { continue }
            if key.isEmpty {
                continue
            }

            var value = vOrig

            if key.hasSuffix(".weight"), value.ndim == 3, !checkArrayShapeQwen3(value) {
                value = value.transposed(0, 2, 1)
            }

            sanitized[key] = value
        }

        return sanitized
    }

    static func stripSpeakerEncoderPrefix(from key: String) -> String? {
        let parts = key.split(separator: ".")
        guard let markerIndex = parts.firstIndex(of: "speaker_encoder") else {
            return nil
        }

        let suffixParts = parts[(markerIndex + 1)...]
        guard !suffixParts.isEmpty else { return nil }
        return suffixParts.joined(separator: ".")
    }
}
