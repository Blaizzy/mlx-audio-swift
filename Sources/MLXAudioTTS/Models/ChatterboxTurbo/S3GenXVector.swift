//
//  S3GenXVector.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN

private func relu(_ x: MLXArray) -> MLXArray {
    MLX.maximum(x, MLXArray(Float(0)))
}

private func mish(_ x: MLXArray) -> MLXArray {
    x * MLX.tanh(MLX.log(MLX.exp(x) + MLXArray(Float(1.0))))
}

private func s3xvectorPadList(_ xs: [MLXArray], padValue: Float = 0) -> MLXArray {
    let batch = xs.count
    let maxLen = xs.map { $0.shape[0] }.max() ?? 0
    let featureShape = xs.first?.shape.dropFirst() ?? []
    let featureSize = featureShape.reduce(1, *)

    var padded = [Float](repeating: padValue, count: batch * maxLen * featureSize)

    for (i, x) in xs.enumerated() {
        let data = x.asArray(Float.self)
        let len = x.shape[0]
        for t in 0..<len {
            let srcStart = t * featureSize
            let dstStart = (i * maxLen + t) * featureSize
            padded[dstStart..<(dstStart + featureSize)] = data[srcStart..<(srcStart + featureSize)]
        }
    }

    let fullShape = [batch, maxLen] + featureShape
    return MLXArray(padded).reshaped(fullShape)
}

private func s3xvectorRepeat(_ x: MLXArray, count: Int, axis: Int) -> MLXArray {
    let expanded = x.expandedDimensions(axis: axis)
    var targetShape = expanded.shape
    targetShape[axis] = count
    let prefix = Array(x.shape[0..<axis])
    let suffix = Array(x.shape[axis...])
    return expanded.broadcasted(to: targetShape).reshaped(prefix + [count] + suffix)
}

private func s3xvectorStd(_ x: MLXArray, axis: Int) -> MLXArray {
    let mean = MLX.mean(x, axis: axis, keepDims: false)
    let meanExpanded = mean.expandedDimensions(axis: axis)
    let variance = MLX.mean((x - meanExpanded).square(), axis: axis, keepDims: false)
    return MLX.sqrt(variance + 1e-8)
}

private func s3xvectorMelScale(_ freq: Float) -> Float {
    1127.0 * logf(1.0 + freq / 700.0)
}

private func s3xvectorInverseMelScale(_ mel: Float) -> Float {
    700.0 * (expf(mel / 1127.0) - 1.0)
}

private func s3xvectorGetMelBanksKaldi(
    numBins: Int,
    paddedWindowSize: Int,
    sampleFreq: Float,
    lowFreq: Float,
    highFreq: Float
) -> [[Float]] {
    let numFftBins = paddedWindowSize / 2
    let nyquist = 0.5 * sampleFreq
    let highFreqVal: Float = highFreq <= 0.0 ? highFreq + nyquist : highFreq

    let fftBinWidth = sampleFreq / Float(paddedWindowSize)
    let melLow = s3xvectorMelScale(lowFreq)
    let melHigh = s3xvectorMelScale(highFreqVal)
    let melDelta = (melHigh - melLow) / Float(numBins + 1)

    var bins = Array(repeating: Array(repeating: Float(0), count: numFftBins), count: numBins)

    for i in 0..<numBins {
        let leftMel = melLow + Float(i) * melDelta
        let centerMel = melLow + Float(i + 1) * melDelta
        let rightMel = melLow + Float(i + 2) * melDelta

        for j in 0..<numFftBins {
            let freq = fftBinWidth * Float(j)
            let mel = s3xvectorMelScale(freq)
            if mel > leftMel && mel <= centerMel {
                bins[i][j] = (mel - leftMel) / (centerMel - leftMel)
            } else if mel > centerMel && mel < rightMel {
                bins[i][j] = (rightMel - mel) / (rightMel - centerMel)
            }
        }
    }

    return bins
}

private func s3xvectorPoveyWindow(length: Int) -> [Float] {
    guard length > 1 else { return [Float](repeating: 1, count: length) }
    var window = [Float](repeating: 0, count: length)
    let denom = Float(length - 1)
    for n in 0..<length {
        let val = 0.5 - 0.5 * cosf(2.0 * Float.pi * Float(n) / denom)
        window[n] = powf(val, 0.85)
    }
    return window
}

private func s3xvectorExtractFbankFeatures(
    _ audio: MLXArray,
    numMelBins: Int = 80,
    sampleRate: Int = 16_000,
    frameLengthMs: Float = 25.0,
    frameShiftMs: Float = 10.0,
    lowFreq: Float = 20.0,
    highFreq: Float = 0.0,
    preemphasisCoeff: Float = 0.97,
    removeDcOffset: Bool = true,
    usePower: Bool = true,
    snipEdges: Bool = true
) -> MLXArray {
    var audio = audio
    if audio.ndim == 1 {
        audio = audio.expandedDimensions(axis: 0)
    }

    let batch = audio.shape[0]
    let frameLength = Int(Float(sampleRate) * frameLengthMs * 0.001)
    let frameShift = Int(Float(sampleRate) * frameShiftMs * 0.001)

    var paddedLength = 1
    while paddedLength < frameLength {
        paddedLength *= 2
    }

    var melBanks = s3xvectorGetMelBanksKaldi(
        numBins: numMelBins,
        paddedWindowSize: paddedLength,
        sampleFreq: Float(sampleRate),
        lowFreq: lowFreq,
        highFreq: highFreq
    )
    for i in 0..<melBanks.count {
        melBanks[i].append(0)
    }

    let melBanksFlat = melBanks.flatMap { $0 }
    let melBanksArray = MLXArray(melBanksFlat)
        .reshaped([numMelBins, paddedLength / 2 + 1])
    let melBanksT = melBanksArray.transposed(1, 0)

    let window = s3xvectorPoveyWindow(length: frameLength)
    let windowPadded = window + [Float](repeating: 0, count: paddedLength - frameLength)
    let windowArray = MLXArray(windowPadded)

    let epsilon = Float.ulpOfOne

    var features: [MLXArray] = []
    features.reserveCapacity(batch)

    for b in 0..<batch {
        var wav = audio[b].asArray(Float.self)

        let numFrames: Int
        if snipEdges {
            if wav.count < frameLength {
                numFrames = 0
            } else {
                numFrames = 1 + (wav.count - frameLength) / frameShift
            }
        } else {
            numFrames = (wav.count + frameShift / 2) / frameShift
        }

        if numFrames == 0 {
            let zeros = MLXArray.zeros([1, numMelBins], type: Float.self)
            features.append(zeros)
            continue
        }

        var frames = [Float](repeating: 0, count: numFrames * frameLength)
        for i in 0..<numFrames {
            let start = i * frameShift
            let end = start + frameLength
            for j in 0..<frameLength {
                let idx = start + j
                let value = idx < wav.count ? wav[idx] : 0
                frames[i * frameLength + j] = value
            }
        }

        if removeDcOffset {
            for i in 0..<numFrames {
                let start = i * frameLength
                let end = start + frameLength
                let slice = frames[start..<end]
                let mean = slice.reduce(0, +) / Float(frameLength)
                for j in 0..<frameLength {
                    frames[start + j] -= mean
                }
            }
        }

        if preemphasisCoeff != 0 {
            var preemph = frames
            for i in 0..<numFrames {
                let base = i * frameLength
                preemph[base] = frames[base]
                if frameLength > 1 {
                    for j in 1..<frameLength {
                        preemph[base + j] = frames[base + j] - preemphasisCoeff * frames[base + j - 1]
                    }
                }
            }
            frames = preemph
        }

        var paddedFrames = [Float](repeating: 0, count: numFrames * paddedLength)
        for i in 0..<numFrames {
            let srcStart = i * frameLength
            let dstStart = i * paddedLength
            paddedFrames[dstStart..<(dstStart + frameLength)] = frames[srcStart..<(srcStart + frameLength)]
        }

        let paddedArray = MLXArray(paddedFrames).reshaped([numFrames, paddedLength])
        let windowExpanded = windowArray.expandedDimensions(axis: 0).broadcasted(to: [numFrames, paddedLength])
        let windowed = paddedArray * windowExpanded
        let fftOut = MLXFFT.rfft(windowed, axis: 1)
        var spectrum = MLX.abs(fftOut)
        if usePower {
            spectrum = spectrum.square()
        }
        let melEnergies = MLX.matmul(spectrum, melBanksT)
        let logMel = MLX.log(MLX.maximum(melEnergies, MLXArray(epsilon)))
        features.append(logMel)
    }

    let maxLen = features.map { $0.shape[0] }.max() ?? 1
    var paddedFeatures: [MLXArray] = []
    paddedFeatures.reserveCapacity(features.count)

    for feat in features {
        if feat.shape[0] < maxLen {
            let padLen = maxLen - feat.shape[0]
            let pad = MLXArray.zeros([padLen, numMelBins], type: Float.self)
            paddedFeatures.append(MLX.concatenated([feat, pad], axis: 0))
        } else {
            paddedFeatures.append(feat)
        }
    }

    return MLX.stacked(paddedFeatures, axis: 0)
}

final class S3BasicResBlock: Module {
    static let expansion = 1

    @ModuleInfo(key: "conv1") private var conv1: Conv2d
    @ModuleInfo(key: "bn1") private var bn1: BatchNorm
    @ModuleInfo(key: "conv2") private var conv2: Conv2d
    @ModuleInfo(key: "bn2") private var bn2: BatchNorm

    private let useShortcut: Bool
    @ModuleInfo(key: "shortcut_conv") private var shortcutConv: Conv2d?
    @ModuleInfo(key: "shortcut_bn") private var shortcutBn: BatchNorm?

    init(inPlanes: Int, planes: Int, stride: Int = 1) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inPlanes,
            outputChannels: planes,
            kernelSize: 3,
            stride: [stride, 1],
            padding: 1,
            bias: false
        )
        self._bn1.wrappedValue = BatchNorm(featureCount: planes)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: planes,
            outputChannels: planes,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: false
        )
        self._bn2.wrappedValue = BatchNorm(featureCount: planes)

        self.useShortcut = stride != 1 || inPlanes != planes * Self.expansion
        if useShortcut {
            let conv = Conv2d(
                inputChannels: inPlanes,
                outputChannels: planes * Self.expansion,
                kernelSize: 1,
                stride: [stride, 1],
                bias: false
            )
            let bn = BatchNorm(featureCount: planes * Self.expansion)
            self._shortcutConv.wrappedValue = conv
            self._shortcutBn.wrappedValue = bn
        } else {
            self._shortcutConv.wrappedValue = nil
            self._shortcutBn.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1(x)
        out = relu(bn1(out))
        out = bn2(conv2(out))

        let shortcut: MLXArray
        if useShortcut, let shortcutConv, let shortcutBn {
            shortcut = shortcutBn(shortcutConv(x))
        } else {
            shortcut = x
        }

        out = relu(out + shortcut)
        return out
    }
}

final class S3FCM: Module {
    private let featDim: Int
    private let outChannels: Int

    @ModuleInfo(key: "conv1") private var conv1: Conv2d
    @ModuleInfo(key: "bn1") private var bn1: BatchNorm
    @ModuleInfo(key: "conv2") private var conv2: Conv2d
    @ModuleInfo(key: "bn2") private var bn2: BatchNorm

    private let layer1: [S3BasicResBlock]
    private let layer2: [S3BasicResBlock]

    init(mChannels: Int = 32, featDim: Int = 80) {
        self.featDim = featDim
        self.outChannels = mChannels * (featDim / 8)

        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: mChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: false
        )
        self._bn1.wrappedValue = BatchNorm(featureCount: mChannels)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: mChannels,
            outputChannels: mChannels,
            kernelSize: 3,
            stride: [2, 1],
            padding: 1,
            bias: false
        )
        self._bn2.wrappedValue = BatchNorm(featureCount: mChannels)

        self.layer1 = [
            S3BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
            S3BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 1)
        ]
        self.layer2 = [
            S3BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
            S3BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 1)
        ]
    }

    var outputChannels: Int { outChannels }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = h.expandedDimensions(axis: -1)

        h = relu(bn1(conv1(h)))
        for layer in layer1 { h = layer(h) }
        for layer in layer2 { h = layer(h) }
        h = relu(bn2(conv2(h)))

        let batch = h.dim(0)
        let fReduced = h.dim(1)
        let time = h.dim(2)
        let channels = h.dim(3)

        let reshaped = h.transposed(0, 3, 1, 2).reshaped(batch, channels * fReduced, time)
        return reshaped
    }
}

final class S3TDNNLayer: Module {
    @ModuleInfo(key: "linear") private var linear: Conv1d
    @ModuleInfo(key: "bn") private var bn: BatchNorm

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = false
    ) {
        var padding = padding
        if padding < 0 {
            precondition(kernelSize % 2 == 1)
            padding = (kernelSize - 1) / 2 * dilation
        }
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        self._bn.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = linear(h)
        h = bn(h)
        h = relu(h)
        return h.transposed(0, 2, 1)
    }
}

final class S3CAMLayer: Module {
    @ModuleInfo(key: "linear_local") private var linearLocal: Conv1d
    @ModuleInfo(key: "linear1") private var linear1: Conv1d
    @ModuleInfo(key: "linear2") private var linear2: Conv1d

    init(
        bnChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int,
        dilation: Int,
        bias: Bool,
        reduction: Int = 2
    ) {
        self._linearLocal.wrappedValue = Conv1d(
            inputChannels: bnChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        self._linear1.wrappedValue = Conv1d(
            inputChannels: bnChannels,
            outputChannels: bnChannels / reduction,
            kernelSize: 1
        )
        self._linear2.wrappedValue = Conv1d(
            inputChannels: bnChannels / reduction,
            outputChannels: outChannels,
            kernelSize: 1
        )
    }

    private func segPooling(_ x: MLXArray, segLen: Int = 100) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let channels = x.dim(2)

        let nSegs = (time + segLen - 1) / segLen
        let padLen = nSegs * segLen - time
        var padded = x
        if padLen > 0 {
            let pad = MLXArray.zeros([batch, padLen, channels], type: Float.self)
            padded = MLX.concatenated([padded, pad], axis: 1)
        }

        let reshaped = padded.reshaped(batch, nSegs, segLen, channels)
        let seg = MLX.mean(reshaped, axis: 2)
        let expanded = seg.expandedDimensions(axis: 2).broadcasted(to: [batch, nSegs, segLen, channels])
            .reshaped(batch, nSegs * segLen, channels)

        return padLen > 0 ? expanded[0..<batch, 0..<time, 0..<channels] : expanded
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        let y = linearLocal(h)

        let context = MLX.mean(h, axis: 1, keepDims: true) + segPooling(h)
        var m = relu(linear1(context))
        m = MLX.sigmoid(linear2(m))

        let result = y * m
        return result.transposed(0, 2, 1)
    }
}

final class S3CAMDenseTDNNLayer: Module {
    @ModuleInfo(key: "bn1") private var bn1: BatchNorm
    @ModuleInfo(key: "linear1") private var linear1: Conv1d
    @ModuleInfo(key: "bn2") private var bn2: BatchNorm
    @ModuleInfo(key: "cam_layer") private var camLayer: S3CAMLayer

    init(
        inChannels: Int,
        outChannels: Int,
        bnChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = false
    ) {
        let padding = (kernelSize - 1) / 2 * dilation
        self._bn1.wrappedValue = BatchNorm(featureCount: inChannels)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: bnChannels,
            kernelSize: 1,
            bias: false
        )
        self._bn2.wrappedValue = BatchNorm(featureCount: bnChannels)
        self._camLayer.wrappedValue = S3CAMLayer(
            bnChannels: bnChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = bn1(h)
        h = relu(h)
        h = linear1(h)
        h = bn2(h)
        h = relu(h)
        h = h.transposed(0, 2, 1)
        return camLayer(h)
    }
}

final class S3CAMDenseTDNNBlock: Module {
    private let layers: [S3CAMDenseTDNNLayer]

    init(
        numLayers: Int,
        inChannels: Int,
        outChannels: Int,
        bnChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = false
    ) {
        self.layers = (0..<numLayers).map { index in
            S3CAMDenseTDNNLayer(
                inChannels: inChannels + index * outChannels,
                outChannels: outChannels,
                bnChannels: bnChannels,
                kernelSize: kernelSize,
                stride: stride,
                dilation: dilation,
                bias: bias
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = MLX.concatenated([h, layer(h)], axis: 1)
        }
        return h
    }
}

final class S3TransitLayer: Module {
    @ModuleInfo(key: "bn") private var bn: BatchNorm
    @ModuleInfo(key: "linear") private var linear: Conv1d

    init(inChannels: Int, outChannels: Int, bias: Bool = true) {
        self._bn.wrappedValue = BatchNorm(featureCount: inChannels)
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = bn(h)
        h = relu(h)
        h = linear(h)
        return h.transposed(0, 2, 1)
    }
}

final class S3StatsPool: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: 2, keepDims: false)
        let std = s3xvectorStd(x, axis: 2)
        return MLX.concatenated([mean, std], axis: 1)
    }
}

final class S3DenseLayer: Module {
    @ModuleInfo(key: "linear") private var linear: Conv1d
    @ModuleInfo(key: "bn") private var bn: BatchNorm

    init(inChannels: Int, outChannels: Int, bias: Bool = false) {
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: bias
        )
        self._bn.wrappedValue = BatchNorm(featureCount: outChannels, affine: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.ndim == 2 {
            var h = x.expandedDimensions(axis: 1)
            h = linear(h)
            h = h.squeezed(axis: 1)
            return bn(h)
        }

        var h = x.transposed(0, 2, 1)
        h = linear(h)
        h = h.transposed(0, 2, 1)
        let bnInput = h.transposed(0, 2, 1)
        let bnOut = bn(bnInput)
        return bnOut.transposed(0, 2, 1)
    }
}

final class CAMPPlus: Module {
    @ModuleInfo(key: "head") var head: S3FCM
    @ModuleInfo(key: "tdnn") var tdnn: S3TDNNLayer
    @ModuleInfo(key: "blocks") var blocks: [S3CAMDenseTDNNBlock]
    @ModuleInfo(key: "transits") var transits: [S3TransitLayer]
    @ModuleInfo(key: "out_bn") var outBn: BatchNorm
    @ModuleInfo(key: "stats") var stats: S3StatsPool
    @ModuleInfo(key: "dense") var dense: S3DenseLayer

    init(
        featDim: Int = 80,
        embeddingSize: Int = 192,
        growthRate: Int = 32,
        bnSize: Int = 4,
        initChannels: Int = 128
    ) {
        let head = S3FCM(mChannels: 32, featDim: featDim)
        self._head.wrappedValue = head
        var channels = head.outputChannels

        self._tdnn.wrappedValue = S3TDNNLayer(
            inChannels: channels,
            outChannels: initChannels,
            kernelSize: 5,
            stride: 2,
            padding: -1,
            dilation: 1
        )
        channels = initChannels

        let blockConfigs = [
            (12, 3, 1),
            (24, 3, 2),
            (16, 3, 2)
        ]

        var blocks: [S3CAMDenseTDNNBlock] = []
        var transits: [S3TransitLayer] = []

        for (index, config) in blockConfigs.enumerated() {
            let numLayers = config.0
            let kernelSize = config.1
            let dilation = config.2

            let block = S3CAMDenseTDNNBlock(
                numLayers: numLayers,
                inChannels: channels,
                outChannels: growthRate,
                bnChannels: growthRate * bnSize,
                kernelSize: kernelSize,
                dilation: dilation
            )
            blocks.append(block)
            channels += numLayers * growthRate

            let transitOut = channels / 2
            let transit = S3TransitLayer(inChannels: channels, outChannels: transitOut, bias: false)
            transits.append(transit)
            channels = transitOut
        }

        self._blocks.wrappedValue = blocks
        self._transits.wrappedValue = transits
        self._outBn.wrappedValue = BatchNorm(featureCount: channels)
        self._stats.wrappedValue = S3StatsPool()
        self._dense.wrappedValue = S3DenseLayer(inChannels: channels * 2, outChannels: embeddingSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = head(x)
        h = tdnn(h)

        for (block, transit) in zip(blocks, transits) {
            h = block(h)
            h = transit(h)
        }

        var hT = h.transposed(0, 2, 1)
        hT = outBn(hT)
        hT = relu(hT)
        h = hT.transposed(0, 2, 1)

        h = stats(h)
        h = dense(h)
        return h
    }

    func inference(_ wavs: [MLXArray]) -> MLXArray {
        var padded: [MLXArray] = []
        padded.reserveCapacity(wavs.count)
        let maxLen = wavs.map { $0.shape[0] }.max() ?? 1

        for wav in wavs {
            if wav.shape[0] < maxLen {
                let padLen = maxLen - wav.shape[0]
                let pad = MLXArray.zeros([padLen], type: Float.self)
                padded.append(MLX.concatenated([wav, pad], axis: 0))
            } else {
                padded.append(wav)
            }
        }

        let audio = MLX.stacked(padded, axis: 0)
        var features = s3xvectorExtractFbankFeatures(audio)
        let mean = MLX.mean(features, axis: 1, keepDims: true)
        features = features - mean
        return callAsFunction(features)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        let blockRegex = try? NSRegularExpression(pattern: "block(\\d+)\\.tdnnd(\\d+)")
        let transitRegex = try? NSRegularExpression(pattern: "transit(\\d+)")

        for (key, value) in weights {
            if key.contains("num_batches_tracked") {
                continue
            }

            var newKey = key

            if newKey.hasPrefix("xvector.") {
                newKey = String(newKey.dropFirst("xvector.".count))
                newKey = newKey.replacingOccurrences(of: "tdnn.nonlinear.batchnorm", with: "tdnn.bn")

                if let blockRegex {
                    let range = NSRange(newKey.startIndex..<newKey.endIndex, in: newKey)
                    if let match = blockRegex.firstMatch(in: newKey, range: range) {
                        let blockRange = match.range(at: 1)
                        let layerRange = match.range(at: 2)
                        if let blockRange = Range(blockRange, in: newKey),
                           let layerRange = Range(layerRange, in: newKey),
                           let blockIdx = Int(newKey[blockRange]),
                           let layerIdx = Int(newKey[layerRange])
                        {
                            let old = "block\(blockIdx).tdnnd\(layerIdx)"
                            let replacement = "blocks.\(blockIdx - 1).layers.\(layerIdx - 1)"
                            newKey = newKey.replacingOccurrences(of: old, with: replacement)
                        }
                    }
                }

                if let transitRegex {
                    let range = NSRange(newKey.startIndex..<newKey.endIndex, in: newKey)
                    if let match = transitRegex.firstMatch(in: newKey, range: range) {
                        let transitRange = match.range(at: 1)
                        if let transitRange = Range(transitRange, in: newKey),
                           let transitIdx = Int(newKey[transitRange])
                        {
                            let old = "transit\(transitIdx)"
                            let replacement = "transits.\(transitIdx - 1)"
                            newKey = newKey.replacingOccurrences(of: old, with: replacement)
                        }
                    }
                }

                newKey = newKey.replacingOccurrences(of: "nonlinear.batchnorm", with: "bn")
                newKey = newKey.replacingOccurrences(of: "nonlinear1.batchnorm", with: "bn1")
                newKey = newKey.replacingOccurrences(of: "nonlinear2.batchnorm", with: "bn2")
                newKey = newKey.replacingOccurrences(of: "out_nonlinear.batchnorm", with: "out_bn")
                newKey = newKey.replacingOccurrences(of: "dense.nonlinear.batchnorm", with: "dense.bn")
            } else if newKey.hasPrefix("head.") {
                newKey = newKey.replacingOccurrences(of: "shortcut.0", with: "shortcut_conv")
                newKey = newKey.replacingOccurrences(of: "shortcut.1", with: "shortcut_bn")
            }

            var newValue = value
            if newKey.contains("weight"), value.ndim >= 3 {
                if value.ndim == 4 {
                    if value.shape[2] == value.shape[3] {
                        newValue = value.transposed(0, 2, 3, 1)
                    }
                } else if value.ndim == 3 {
                    if value.shape[1] > value.shape[2] {
                        newValue = value.transposed(0, 2, 1)
                    }
                }
            }

            sanitized[newKey] = newValue
        }

        return sanitized
    }
}
