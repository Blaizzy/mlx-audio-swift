// Ported from Python mlx-audio chatterbox s3gen/xvector.py (CAM++ speaker encoder)
// Produces 192-dimensional x-vector speaker embeddings from 16kHz audio.

import Foundation
import MLX
import MLXAudioCore
import MLXNN

// MARK: - Kaldi-style Feature Extraction

/// Extract 80-bin Kaldi-compatible log filterbank features from 16kHz audio.
/// Uses Povey window, pre-emphasis 0.97, HTK mel scale.
func kaldiFbank(
    audio: MLXArray,
    sampleRate: Int = 16000,
    numMels: Int = 80,
    frameLength: Float = 0.025,
    frameShift: Float = 0.010,
    preemphasis: Float = 0.97,
    energyFloor: Float = 1.1920929e-07
) -> MLXArray {
    let frameLenSamples = Int(frameLength * Float(sampleRate))  // 400
    let frameShiftSamples = Int(frameShift * Float(sampleRate)) // 160
    let nFft = nextPowerOf2(frameLenSamples)                    // 512

    let audioLen = audio.dim(0)
    let numFrames = max(1, 1 + (audioLen - frameLenSamples) / frameShiftSamples)

    // Frame extraction using strided view
    let frames = asStrided(
        audio, [numFrames, frameLenSamples],
        strides: [frameShiftSamples, 1], offset: 0
    ).asType(.float32)

    // DC removal per frame
    let frameMean = frames.mean(axis: 1, keepDims: true)
    var processed = frames - frameMean

    // Pre-emphasis: y[0] = x[0] (unchanged), y[n] = x[n] - 0.97 * x[n-1] for n >= 1
    let preemphasized = MLX.concatenated(
        [processed[0..., ..<1],
         processed[0..., 1...] - preemphasis * processed[0..., ..<(frameLenSamples - 1)]],
        axis: 1
    )
    processed = preemphasized

    // Povey window: hann^0.85
    let poveyWindow = makePoveyWindow(size: frameLenSamples)
    processed = processed * poveyWindow

    // Zero-pad to nFft if needed
    if frameLenSamples < nFft {
        let padSize = nFft - frameLenSamples
        processed = MLX.padded(processed, widths: [.init(0), .init((0, padSize))])
    }

    // FFT and power spectrum
    let fft = MLXFFT.rfft(processed, axis: 1) // (numFrames, nFft/2+1)
    let powerSpec = abs(fft).square()

    // HTK mel filterbank (Kaldi default: fMin=20 Hz)
    let filters = melFilters(
        sampleRate: sampleRate, nFft: nFft, nMels: numMels,
        fMin: 20.0, fMax: Float(sampleRate) / 2.0,
        norm: nil, melScale: .htk
    ) // (nFreqs, nMels)

    // Apply: (numFrames, nFreqs) @ (nFreqs, nMels) -> (numFrames, nMels)
    var melEnergy = matmul(powerSpec, filters)
    melEnergy = MLX.maximum(melEnergy, MLXArray(energyFloor))
    let logMel = MLX.log(melEnergy)

    return logMel // (numFrames, numMels)
}

/// Hann window raised to power 0.85 (Povey window).
private func makePoveyWindow(size: Int) -> MLXArray {
    let values = (0 ..< size).map { n -> Float in
        let hann = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(size - 1)))
        return pow(hann, 0.85)
    }
    return MLXArray(values)
}

/// Next power of 2 >= n.
private func nextPowerOf2(_ n: Int) -> Int {
    var p = 1
    while p < n { p <<= 1 }
    return p
}

// MARK: - BatchNorm Helpers

/// Apply BatchNorm in PyTorch (B,C,T) format by transposing to (B,T,C) and back.
private func batchNormBCT(_ bn: BatchNorm, _ x: MLXArray) -> MLXArray {
    let transposed = x.transposed(0, 2, 1) // (B, T, C)
    let normed = bn(transposed)
    return normed.transposed(0, 2, 1) // (B, C, T)
}

/// Apply Conv1d in PyTorch (B,C,T) format by transposing to (B,T,C) and back.
private func conv1dBCT(_ conv: Conv1d, _ x: MLXArray) -> MLXArray {
    let transposed = x.transposed(0, 2, 1) // (B, T, C)
    let result = conv(transposed)
    return result.transposed(0, 2, 1) // (B, C', T)
}

// MARK: - Nonlinear Factory

/// Create nonlinear layers from config string like "batchnorm-relu".
private func makeNonlinear(configStr: String, channels: Int) -> [Module] {
    var layers: [Module] = []
    for part in configStr.split(separator: "-") {
        switch String(part).lowercased() {
        case "batchnorm":
            layers.append(BatchNorm(featureCount: channels))
        case "relu":
            layers.append(ReLUModule())
        default:
            break
        }
    }
    return layers
}

/// Wrapper to make ReLU a Module for storage in arrays.
private class ReLUModule: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(x) }
}

// MARK: - Statistics Pooling

/// Compute mean and standard deviation pooling over time dimension.
private func statisticsPooling(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let mean = x.mean(axis: axis)
    let variance = x.variance(axis: axis)
    let std = MLX.sqrt(variance + 1e-5)
    return MLX.concatenated([mean, std], axis: -1)
}

// MARK: - Segment Pooling (for CAMLayer)

/// Segment-level average pooling: divides time axis into segments,
/// pools each segment, expands back to original length.
private func segPooling(_ x: MLXArray, segLen: Int = 100) -> MLXArray {
    // x: (B, C, T)
    let B = x.dim(0), C = x.dim(1), T = x.dim(2)

    if T <= segLen {
        // Single segment — just mean, expand to match T
        let mean = x.mean(axis: 2, keepDims: true) // (B, C, 1)
        let ones = MLXArray.ones([1, 1, T])
        return mean * ones
    }

    let numFullSegs = T / segLen
    let remainder = T % segLen

    var segments: [MLXArray] = []
    for i in 0 ..< numFullSegs {
        let start = i * segLen
        let end = start + segLen
        let segMean = x[0..., 0..., start ..< end].mean(axis: 2, keepDims: true) // (B, C, 1)
        // Broadcast to segment length by multiplying with ones
        let ones = MLXArray.ones([1, 1, segLen])
        segments.append(segMean * ones)
    }
    if remainder > 0 {
        let start = numFullSegs * segLen
        let segMean = x[0..., 0..., start...].mean(axis: 2, keepDims: true)
        let ones = MLXArray.ones([1, 1, remainder])
        segments.append(segMean * ones)
    }

    return MLX.concatenated(segments, axis: 2) // (B, C, T)
}

// MARK: - BasicResBlock (for FCM)

/// Basic residual block for 2D convolution (used in FCM ResNet head).
/// Stride is applied only in the H (frequency) dimension via stride=(s, 1).
class BasicResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    // Optional shortcut when stride != 1 or channels change
    @ModuleInfo(key: "shortcut") var shortcut: [Module]

    let hasShortcut: Bool

    init(inPlanes: Int, planes: Int, stride: Int = 1) {
        self.hasShortcut = (stride != 1) || (inPlanes != planes)

        // Both convs use kernel 3x3, padding 1
        // Stride applied only in H dimension: stride=(stride, 1)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inPlanes, outputChannels: planes,
            kernelSize: .init((3, 3)), stride: .init((stride, 1)), padding: .init((1, 1)),
            bias: false
        )
        self._bn1.wrappedValue = BatchNorm(featureCount: planes)

        self._conv2.wrappedValue = Conv2d(
            inputChannels: planes, outputChannels: planes,
            kernelSize: .init((3, 3)), stride: .init((1, 1)), padding: .init((1, 1)),
            bias: false
        )
        self._bn2.wrappedValue = BatchNorm(featureCount: planes)

        if hasShortcut {
            self._shortcut.wrappedValue = [
                Conv2d(inputChannels: inPlanes, outputChannels: planes,
                       kernelSize: .init((1, 1)), stride: .init((stride, 1)), bias: false),
                BatchNorm(featureCount: planes),
            ]
        } else {
            self._shortcut.wrappedValue = []
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C) — MLX NHWC format
        var identity = x

        var out = conv1(x)
        out = batchNorm2dNHWC(bn1, out)
        out = relu(out)

        out = conv2(out)
        out = batchNorm2dNHWC(bn2, out)

        if hasShortcut, shortcut.count >= 2 {
            if let shortConv = shortcut[0] as? Conv2d,
               let shortBn = shortcut[1] as? BatchNorm
            {
                identity = shortConv(identity)
                identity = batchNorm2dNHWC(shortBn, identity)
            }
        }

        return relu(out + identity)
    }
}

/// Apply BatchNorm on 4D NHWC data: (B, H, W, C) → reshape to (B*H*W, C) → BN → reshape back.
private func batchNorm2dNHWC(_ bn: BatchNorm, _ x: MLXArray) -> MLXArray {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let flat = x.reshaped([B * H * W, C])
    let normed = bn(flat)
    return normed.reshaped([B, H, W, C])
}

// MARK: - FCM (Frequency Context Mask) — Full ResNet2D Head

/// Frequency Context Mask: 2D ResNet that processes mel spectrogram spatially.
/// Input: (B, F=80, T) → output: (B, mChannels * F/8, T) = (B, 320, T).
class FCM: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [BasicResBlock]
    @ModuleInfo(key: "layer2") var layer2: [BasicResBlock]
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    let mChannels: Int
    let featDim: Int

    init(featDim: Int = 80, mChannels: Int = 32) {
        self.mChannels = mChannels
        self.featDim = featDim

        // Initial conv: 1 -> mChannels
        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: mChannels,
            kernelSize: .init((3, 3)), stride: .init((1, 1)), padding: .init((1, 1)),
            bias: false
        )
        self._bn1.wrappedValue = BatchNorm(featureCount: mChannels)

        // layer1: 2 blocks, first with stride=2 in freq dim
        self._layer1.wrappedValue = [
            BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
            BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 1),
        ]

        // layer2: 2 blocks, first with stride=2 in freq dim
        self._layer2.wrappedValue = [
            BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
            BasicResBlock(inPlanes: mChannels, planes: mChannels, stride: 1),
        ]

        // Final conv with stride=2 in freq dim
        self._conv2.wrappedValue = Conv2d(
            inputChannels: mChannels, outputChannels: mChannels,
            kernelSize: .init((3, 3)), stride: .init((2, 1)), padding: .init((1, 1)),
            bias: false
        )
        self._bn2.wrappedValue = BatchNorm(featureCount: mChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, F, T) in PyTorch NCHW convention
        // MLX Conv2d expects NHWC: (B, H, W, C)
        // Treat F as H, T as W, C=1
        let B = x.dim(0), F = x.dim(1), T = x.dim(2)

        // (B, F, T) -> (B, F, T, 1) — NHWC with C=1
        var out = x.expandedDimensions(axis: 3)

        // Initial conv + BN + ReLU
        out = conv1(out)                     // (B, F, T, mChannels)
        out = batchNorm2dNHWC(bn1, out)
        out = relu(out)

        // Layer 1: stride=2 in H -> F/2
        for block in layer1 { out = block(out) }
        // Layer 2: stride=2 in H -> F/4
        for block in layer2 { out = block(out) }

        // Final conv: stride=2 in H -> F/8
        out = conv2(out)
        out = batchNorm2dNHWC(bn2, out)
        out = relu(out)

        // out shape: (B, F/8, T, mChannels) — reshape to (B, mChannels * F/8, T)
        let newH = out.dim(1)
        let newW = out.dim(2)
        let C = out.dim(3)
        // Rearrange: (B, H, W, C) -> (B, C*H, W) = (B, mChannels * F/8, T)
        out = out.transposed(0, 3, 1, 2) // (B, C, H, W)
        out = out.reshaped([B, C * newH, newW])

        return out // (B, 320, T)
    }
}

// MARK: - TDNNLayer

/// Time-delay neural network layer with 1D convolution + nonlinear.
/// Operates in PyTorch (B,C,T) format.
class TDNNLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: [Module]

    init(
        inChannels: Int, outChannels: Int,
        kernelSize: Int, stride: Int = 1, dilation: Int = 1,
        configStr: String = "batchnorm-relu", bias: Bool = false
    ) {
        let padding = (kernelSize - 1) / 2 * dilation
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation, bias: bias
        )
        self._nonlinear.wrappedValue = makeNonlinear(configStr: configStr, channels: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)
        var out = conv1dBCT(linear, x) // (B, C', T)
        for layer in nonlinear {
            if let bn = layer as? BatchNorm {
                out = batchNormBCT(bn, out)
            } else if let reluLayer = layer as? ReLUModule {
                out = reluLayer(out)
            }
        }
        return out
    }
}

// MARK: - CAMLayer (Context-Aware Masking)

/// Context-aware masking layer with local conv, global mean, and segment pooling.
class CAMLayer: Module {
    @ModuleInfo(key: "linear_local") var linearLocal: Conv1d
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "linear2") var linear2: Conv1d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    let segLen: Int
    let outChannels: Int

    init(
        inChannels: Int, outChannels: Int,
        kernelSize: Int = 3, stride: Int = 1, padding: Int = 1,
        dilation: Int = 1, segLen: Int = 100
    ) {
        self.segLen = segLen
        self.outChannels = outChannels

        // Reduction factor for bottleneck: inChannels / 2
        // Python: linear1 = Conv1d(bn_channels, bn_channels // reduction, 1) where reduction=2
        let innerChannels = inChannels / 2

        self._linearLocal.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation, bias: false
        )
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: innerChannels,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )
        self._linear2.wrappedValue = Conv1d(
            inputChannels: innerChannels, outputChannels: outChannels,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )
        self._bn1.wrappedValue = BatchNorm(featureCount: innerChannels)
        self._bn2.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)

        // Local branch: Conv1d
        let y = conv1dBCT(linearLocal, x) // (B, outChannels, T)

        // Global context: mean + segment pooling
        let globalMean = x.mean(axis: 2, keepDims: true) // (B, C, 1)
        let segPool = segPooling(x, segLen: segLen)        // (B, C, T)
        let context = globalMean + segPool                 // (B, C, T)

        // Bottleneck: reduce -> relu -> expand -> sigmoid
        var m = conv1dBCT(linear1, context)
        m = batchNormBCT(bn1, m)
        m = relu(m)
        m = conv1dBCT(linear2, m)
        m = batchNormBCT(bn2, m)
        m = sigmoid(m)

        return y * m // (B, outChannels, T)
    }
}

// MARK: - CAMDenseTDNNLayer (Single Layer in Dense Block)

/// One layer within a dense block: nonlinear1 -> linear1 (bottleneck) -> nonlinear2 -> CAMLayer.
class CAMDenseTDNNLayer: Module {
    @ModuleInfo(key: "nonlinear1") var nonlinear1: [Module]
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "nonlinear2") var nonlinear2: [Module]
    @ModuleInfo(key: "cam_layer") var camLayer: CAMLayer

    init(
        inChannels: Int, outChannels: Int, bnChannels: Int,
        kernelSize: Int, stride: Int = 1, dilation: Int = 1,
        configStr: String = "batchnorm-relu"
    ) {
        let padding = (kernelSize - 1) / 2 * dilation

        self._nonlinear1.wrappedValue = makeNonlinear(configStr: configStr, channels: inChannels)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: bnChannels,
            kernelSize: 1, stride: 1, padding: 0, bias: false
        )
        self._nonlinear2.wrappedValue = makeNonlinear(configStr: configStr, channels: bnChannels)
        self._camLayer.wrappedValue = CAMLayer(
            inChannels: bnChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)
        var out = x
        for layer in nonlinear1 {
            if let bn = layer as? BatchNorm { out = batchNormBCT(bn, out) }
            else if let r = layer as? ReLUModule { out = r(out) }
        }
        out = conv1dBCT(linear1, out)
        for layer in nonlinear2 {
            if let bn = layer as? BatchNorm { out = batchNormBCT(bn, out) }
            else if let r = layer as? ReLUModule { out = r(out) }
        }
        out = camLayer(out)
        return out // (B, outChannels, T)
    }
}

// MARK: - CAMDenseTDNNBlock (Dense Block with Skip Connections)

/// Dense block: each layer receives concatenation of all previous outputs.
/// Channel count grows by outChannels per layer.
class CAMDenseTDNNBlock: Module {
    @ModuleInfo(key: "layers") var layers: [CAMDenseTDNNLayer]

    init(
        numLayers: Int, inChannels: Int, outChannels: Int, bnChannels: Int,
        kernelSize: Int, dilation: Int = 1,
        configStr: String = "batchnorm-relu"
    ) {
        var layerList: [CAMDenseTDNNLayer] = []
        for i in 0 ..< numLayers {
            let layerInChannels = inChannels + i * outChannels
            layerList.append(CAMDenseTDNNLayer(
                inChannels: layerInChannels, outChannels: outChannels,
                bnChannels: bnChannels, kernelSize: kernelSize,
                dilation: dilation, configStr: configStr
            ))
        }
        self._layers.wrappedValue = layerList
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, inChannels, T)
        var features = [x]
        for layer in layers {
            let input = MLX.concatenated(features, axis: 1) // Dense concat
            let out = layer(input)
            features.append(out)
        }
        return MLX.concatenated(features, axis: 1) // (B, inChannels + numLayers*outChannels, T)
    }
}

// MARK: - TransitLayer

/// Transition layer: nonlinear + 1x1 Conv1d to reduce channels.
class TransitLayer: Module {
    @ModuleInfo(key: "nonlinear") var nonlinear: [Module]
    @ModuleInfo(key: "linear") var linear: Conv1d

    init(inChannels: Int, outChannels: Int, configStr: String = "batchnorm-relu") {
        self._nonlinear.wrappedValue = makeNonlinear(configStr: configStr, channels: inChannels)
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 1, stride: 1, padding: 0, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in nonlinear {
            if let bn = layer as? BatchNorm { out = batchNormBCT(bn, out) }
            else if let r = layer as? ReLUModule { out = r(out) }
        }
        return conv1dBCT(linear, out)
    }
}

// MARK: - DenseLayer (Output)

/// Dense output layer: 1x1 Conv1d + optional nonlinear.
class DenseLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: [Module]

    init(inChannels: Int, outChannels: Int, configStr: String = "batchnorm", bias: Bool = false) {
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 1, stride: 1, padding: 0, bias: bias
        )
        self._nonlinear.wrappedValue = makeNonlinear(configStr: configStr, channels: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C) or (B, C, T)
        let was2D = x.ndim == 2
        var out = was2D ? x.expandedDimensions(axis: 2) : x

        out = conv1dBCT(linear, out)
        for layer in nonlinear {
            if let bn = layer as? BatchNorm { out = batchNormBCT(bn, out) }
            else if let r = layer as? ReLUModule { out = r(out) }
        }

        return was2D ? out.squeezed(axis: 2) : out
    }
}

// MARK: - StatsPool

/// Statistics pooling: mean + standard deviation over time.
class StatsPool: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T)
        return statisticsPooling(x, axis: 2) // (B, 2*C)
    }
}

// MARK: - CAMPPlus (Full X-Vector Speaker Encoder)

/// CAM++ speaker encoder for S3Gen conditioning.
/// Architecture: FCM → TDNN → 3 Dense Blocks with Transit Layers → StatsPool → Dense → 192-dim.
///
/// Default config: feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4, init_channels=128
/// Block config: (12, 24, 16) layers with kernel_sizes=(3, 3, 3) and dilations=(1, 2, 2).
class CAMPPlus: Module {
    @ModuleInfo(key: "head") var head: FCM
    @ModuleInfo(key: "tdnn") var tdnn: TDNNLayer
    @ModuleInfo(key: "blocks") var blocks: [CAMDenseTDNNBlock]
    @ModuleInfo(key: "transits") var transits: [TransitLayer]
    @ModuleInfo(key: "out_nonlinear") var outNonlinear: [Module]
    @ModuleInfo(key: "pool") var pool: StatsPool
    @ModuleInfo(key: "dense") var dense: DenseLayer

    let embeddingSize: Int

    init(
        featDim: Int = 80, embeddingSize: Int = 192,
        growthRate: Int = 32, bnSize: Int = 4,
        initChannels: Int = 128, mChannels: Int = 32,
        configStr: String = "batchnorm-relu"
    ) {
        self.embeddingSize = embeddingSize

        let bnChannels = bnSize * growthRate // 128
        let fcmOutChannels = mChannels * (featDim / 8) // 32 * 10 = 320

        // FCM head
        self._head.wrappedValue = FCM(featDim: featDim, mChannels: mChannels)

        // TDNN: 320 -> 128, kernel=5, stride=2
        self._tdnn.wrappedValue = TDNNLayer(
            inChannels: fcmOutChannels, outChannels: initChannels,
            kernelSize: 5, stride: 2, configStr: configStr
        )

        // Dense blocks and transit layers
        // Block configs: (numLayers, kernelSize, dilation)
        let blockConfigs: [(Int, Int, Int)] = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]

        var blockList: [CAMDenseTDNNBlock] = []
        var transitList: [TransitLayer] = []
        var channels = initChannels

        for (numLayers, kernelSize, dilation) in blockConfigs {
            blockList.append(CAMDenseTDNNBlock(
                numLayers: numLayers, inChannels: channels,
                outChannels: growthRate, bnChannels: bnChannels,
                kernelSize: kernelSize, dilation: dilation,
                configStr: configStr
            ))

            let blockOutChannels = channels + numLayers * growthRate
            let transitOutChannels = blockOutChannels / 2
            transitList.append(TransitLayer(
                inChannels: blockOutChannels, outChannels: transitOutChannels,
                configStr: configStr
            ))
            channels = transitOutChannels
        }

        self._blocks.wrappedValue = blockList
        self._transits.wrappedValue = transitList

        // Output: BN+ReLU → StatsPool → Dense
        self._outNonlinear.wrappedValue = makeNonlinear(configStr: configStr, channels: channels)
        self._pool.wrappedValue = StatsPool()
        self._dense.wrappedValue = DenseLayer(
            inChannels: channels * 2, outChannels: embeddingSize,
            configStr: "batchnorm", bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, T, F) fbank features
        var out = x.transposed(0, 2, 1) // (B, F, T)

        // FCM head
        out = head(out) // (B, 320, T)

        // TDNN
        out = tdnn(out) // (B, 128, T/2)

        // Dense blocks + transits
        for (block, transit) in zip(blocks, transits) {
            out = block(out)
            out = transit(out)
        }

        // Output nonlinear
        for layer in outNonlinear {
            if let bn = layer as? BatchNorm { out = batchNormBCT(bn, out) }
            else if let r = layer as? ReLUModule { out = r(out) }
        }

        // Stats pooling: (B, C, T) -> (B, 2*C)
        out = pool(out)

        // Dense: (B, 2*C) -> (B, embedDim)
        out = dense(out)

        return out // (B, 192)
    }

    /// Run inference on raw 16kHz audio waveform(s).
    /// Extracts Kaldi fbank features, pads to uniform length, runs forward pass.
    func inference(_ wavs: [MLXArray], sampleRate: Int = 16000) -> MLXArray {
        // Extract fbank features for each wav
        var feats: [MLXArray] = []
        for wav in wavs {
            let fbank = kaldiFbank(audio: wav, sampleRate: sampleRate)
            feats.append(fbank) // (T, 80)
        }

        // Mean normalization per utterance
        feats = feats.map { f in
            let mean = f.mean(axis: 0, keepDims: true)
            return f - mean
        }

        // Pad to max length
        let maxLen = feats.map { $0.dim(0) }.max() ?? 0
        var padded: [MLXArray] = []
        for f in feats {
            if f.dim(0) < maxLen {
                let pad = MLXArray.zeros([maxLen - f.dim(0), f.dim(1)])
                padded.append(MLX.concatenated([f, pad], axis: 0))
            } else {
                padded.append(f)
            }
        }

        let batch = MLX.stacked(padded, axis: 0) // (B, T, 80)
        return callAsFunction(batch) // (B, 192)
    }

    /// Sanitize weight keys from Python checkpoint format to Swift module hierarchy.
    ///
    /// Handles both raw PyTorch and pre-converted MLX-community weight formats.
    /// Conv weight transpositions are only applied when the weight shape doesn't
    /// already match the model's expected parameter shape.
    static func sanitize(weights: [String: MLXArray], model: CAMPPlus) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        // Flatten model parameters to a dict of "a.b.c.weight" → MLXArray for shape comparison
        let flatParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        for (key, var value) in weights {
            var newKey = key

            // Drop PyTorch-only BatchNorm tracking state
            if newKey.hasSuffix(".num_batches_tracked") { continue }

            // --- Turbo model key format: xvector.blockN.tdnndM.* → blocks.(N-1).layers.(M-1).* ---

            // xvector.blockN.tdnndM.suffix → blocks.(N-1).layers.(M-1).suffix
            // Use named capture groups for reliable number extraction
            if let regex = try? NSRegularExpression(pattern: #"^xvector\.block(\d+)\.tdnnd(\d+)\."#),
               let match = regex.firstMatch(in: newKey, range: NSRange(newKey.startIndex..., in: newKey)),
               let blockRange = Range(match.range(at: 1), in: newKey),
               let layerRange = Range(match.range(at: 2), in: newKey),
               let blockNum = Int(newKey[blockRange]),
               let layerNum = Int(newKey[layerRange])
            {
                let matchEnd = newKey.index(newKey.startIndex, offsetBy: match.range.length)
                let suffix = String(newKey[matchEnd...])
                newKey = "blocks.\(blockNum - 1).layers.\(layerNum - 1).\(suffix)"
            }

            // xvector.transitN.* → transits.(N-1).*
            if let regex = try? NSRegularExpression(pattern: #"^xvector\.transit(\d+)\."#),
               let match = regex.firstMatch(in: newKey, range: NSRange(newKey.startIndex..., in: newKey)),
               let numRange = Range(match.range(at: 1), in: newKey),
               let num = Int(newKey[numRange])
            {
                let matchEnd = newKey.index(newKey.startIndex, offsetBy: match.range.length)
                let suffix = String(newKey[matchEnd...])
                newKey = "transits.\(num - 1).\(suffix)"
            }

            // xvector.tdnn.* → tdnn.*
            if newKey.hasPrefix("xvector.tdnn.") {
                newKey = String(newKey.dropFirst("xvector.".count))
            }

            // xvector.out_nonlinear.* → out_nonlinear.*
            if newKey.hasPrefix("xvector.out_nonlinear.") {
                newKey = String(newKey.dropFirst("xvector.".count))
            }

            // xvector.dense.* → dense.*
            if newKey.hasPrefix("xvector.dense.") {
                newKey = String(newKey.dropFirst("xvector.".count))
            }

            // --- Regular model keys: blocks.N.layers.M.* — already correct, no transform needed ---

            // head.* stays as head.*

            // Remap PyTorch batchnorm keys to array index 0.
            // In Python, nonlinear = nn.Sequential(BatchNorm1d(...), ReLU()),
            // so batchnorm is at index 0. This handles all patterns:
            //   nonlinear.batchnorm.*  → nonlinear.0.*
            //   nonlinear1.batchnorm.* → nonlinear1.0.*
            //   nonlinear2.batchnorm.* → nonlinear2.0.*
            //   out_nonlinear.batchnorm.* → out_nonlinear.0.*
            newKey = newKey.replacingOccurrences(of: ".batchnorm.", with: ".0.")

            // Conv weight transposition — ONLY when model reference shape exists and differs.
            // MLX-community models ship with weights already in MLX format.
            // Raw PyTorch models need transposition (shapes will differ from model expectation).
            // If no model reference is found, assume MLX format (do NOT transpose).
            if newKey.hasSuffix(".weight") && (value.ndim == 4 || value.ndim == 3) {
                if let expectedWeight = flatParams[newKey] {
                    if value.shape != expectedWeight.shape {
                        if value.ndim == 4 {
                            // Conv2d: PyTorch (O,I,H,W) → MLX (O,H,W,I)
                            value = value.transposed(0, 2, 3, 1)
                        } else if value.ndim == 3 {
                            // Conv1d: PyTorch (O,I,K) → MLX (O,K,I)
                            value = value.swappedAxes(1, 2)
                        }
                    }
                }
                // If no model reference found, leave weight as-is (assume MLX format)
            }

            sanitized[newKey] = value
        }

        return sanitized
    }
}
