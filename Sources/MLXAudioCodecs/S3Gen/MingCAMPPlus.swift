import Foundation
import MLX
import MLXNN
import MLXAudioCore

private final class MingCAMPResidual: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    let shortcut: [Conv2d]
    init(stride: Int) {
        conv1 = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3,
                       stride: .init((stride, 1)), padding: 1)
        conv2 = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3, padding: 1)
        shortcut = stride == 1 ? [] : [Conv2d(inputChannels: 32, outputChannels: 32,
            kernelSize: 1, stride: .init((stride, 1)))]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(conv2(relu(conv1(x))) + (shortcut.first.map { $0(x) } ?? x))
    }
}
private final class MingCAMPFront: Module {
    let conv1 = Conv2d(inputChannels: 1, outputChannels: 32, kernelSize: 3, padding: 1)
    let conv2 = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3,
                       stride: .init((2, 1)), padding: 1)
    let layer1 = [MingCAMPResidual(stride: 2), MingCAMPResidual(stride: 1)]
    let layer2 = [MingCAMPResidual(stride: 2), MingCAMPResidual(stride: 1)]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(conv1(x.expandedDimensions(axis: -1)))
        for layer in layer1 + layer2 { h = layer(h) }
        h = relu(conv2(h))
        return h.transposed(0, 3, 1, 2).reshaped(h.dim(0), -1, h.dim(2)).transposed(0, 2, 1)
    }
}
private final class MingCAMPTimeDelay: Module {
    let linear = Conv1d(inputChannels: 320, outputChannels: 128, kernelSize: 5, stride: 2, padding: 2)
    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(linear(x)) }
}
private final class MingCAMPContext: Module {
    @ModuleInfo(key: "linear_local") var local: Conv1d
    let linear1 = Conv1d(inputChannels: 128, outputChannels: 64, kernelSize: 1)
    let linear2 = Conv1d(inputChannels: 64, outputChannels: 32, kernelSize: 1)
    init(dilation: Int) {
        _local.wrappedValue = Conv1d(inputChannels: 128, outputChannels: 32,
            kernelSize: 3, padding: dilation, dilation: dilation, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(1), segments = (length + 99) / 100
        let padded = MLX.padded(x, widths: [.init(0), .init((0, segments * 100 - length)), .init(0)])
        let sums = padded.reshaped(x.dim(0), segments, 100, x.dim(2)).sum(axis: 2)
        let counts = minimum(100, length - MLXArray(0..<segments) * 100).asType(x.dtype)
        let segmentMean = sums / counts[.newAxis, 0..., .newAxis]
        let context = x.mean(axis: 1, keepDims: true)
            + repeated(segmentMean, count: 100, axis: 1)[0..., ..<length, 0...]
        return local(x) * sigmoid(linear2(relu(linear1(context))))
    }
}
private final class MingCAMPLayer: Module {
    let nonlinear1: [BatchNorm]
    let linear1: Conv1d
    @ModuleInfo(key: "cam_layer") var context: MingCAMPContext
    init(input: Int, dilation: Int) {
        nonlinear1 = [BatchNorm(featureCount: input)]
        linear1 = Conv1d(inputChannels: input, outputChannels: 128, kernelSize: 1)
        _context.wrappedValue = MingCAMPContext(dilation: dilation)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { context(relu(linear1(relu(nonlinear1[0](x))))) }
}
private final class MingCAMPBlock: Module {
    let layers: [MingCAMPLayer]
    init(input: Int, count: Int, dilation: Int) {
        layers = (0..<count).map { MingCAMPLayer(input: input + $0 * 32, dilation: dilation) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = concatenated([h, layer(h)], axis: -1) }
        return h
    }
}
private final class MingCAMPTransit: Module {
    let nonlinear: [BatchNorm]
    let linear: Conv1d
    init(input: Int, bias: Bool) {
        nonlinear = [BatchNorm(featureCount: input)]
        linear = Conv1d(inputChannels: input, outputChannels: input / 2, kernelSize: 1, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(relu(nonlinear[0](x))) }
}
private final class MingCAMPDense: Module {
    let linear = Conv1d(inputChannels: 1024, outputChannels: 192, kernelSize: 1, bias: false)
    let nonlinear = [BatchNorm(featureCount: 192, affine: false)]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        nonlinear[0](linear(x.expandedDimensions(axis: 1))).squeezed(axis: 1)
    }
}

/// Native folded CAMPPlus speaker encoder used by Ming dense and Tiny.
/// Accepts [batch, frames, 80] mean-normalized Kaldi features; returns [batch, 192].
public final class MingCAMPPlus: Module {
    private let head = MingCAMPFront()
    private let tdnn = MingCAMPTimeDelay()
    private let blocks = [MingCAMPBlock(input: 128, count: 12, dilation: 1),
                          MingCAMPBlock(input: 256, count: 24, dilation: 2),
                          MingCAMPBlock(input: 512, count: 16, dilation: 2)]
    private let transits = [MingCAMPTransit(input: 512, bias: false),
                            MingCAMPTransit(input: 1024, bias: false),
                            MingCAMPTransit(input: 1024, bias: true)]
    private let dense = MingCAMPDense()
    public override init() { super.init() }
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var h = tdnn(head(features.transposed(0, 2, 1)))
        for (block, transit) in zip(blocks, transits) { h = transit(block(h)) }
        h = relu(h)
        let count = h.dim(1)
        precondition(count > 1, "CAMPPlus requires at least two feature frames")
        let variance = h.variance(axis: 1) * Float(count) / Float(count - 1)
        return dense(concatenated([h.mean(axis: 1), sqrt(maximum(variance, 0))], axis: -1))
    }
    /// Ming's no-dither Povey filterbank frontend, with no waveform amplitude scaling.
    public static func features(_ audio16k: MLXArray) -> MLXArray {
        precondition(audio16k.ndim == 1 && audio16k.size >= 720)
        let count = 1 + (audio16k.size - 400) / 160
        let frames = asStrided(audio16k.asType(.float32), [count, 400], strides: [160, 1])
        let centered = frames - frames.mean(axis: 1, keepDims: true)
        let emphasized = concatenated([centered[0..., ..<1],
            centered[0..., 1...] - 0.97 * centered[0..., ..<399]], axis: 1)
        let window = pow(0.5 - 0.5 * cos(2 * Float.pi * MLXArray(0..<400).asType(.float32) / 399), 0.85)
        let padded = MLX.padded(emphasized * window, widths: [.init(0), .init((0, 112))])
        let spectrum = abs(MLXFFT.rfft(padded, axis: -1)).square()
        let melLow = Float(1127 * log(1 + Float(20) / 700))
        let melHigh = Float(1127 * log(1 + Float(8000) / 700))
        let delta = (melHigh - melLow) / 81
        let bin = MLXArray(0..<80).asType(.float32)[0..., .newAxis]
        let mel = (1127 * log(1 + MLXArray(0..<256).asType(.float32) * (Float(16000) / 512 / 700)))[.newAxis]
        let left = melLow + bin * delta
        let center = melLow + (bin + 1) * delta
        let right = melLow + (bin + 2) * delta
        var filters = maximum(0, minimum((mel - left) / (center - left), (right - mel) / (right - center)))
        filters = MLX.padded(filters, widths: [.init(0), .init((0, 1))])
        let features = log(maximum(matmul(spectrum, filters.T), 1e-8))
        return features - features.mean(axis: 0, keepDims: true)
    }
    public static func fromWeights(_ weights: [String: MLXArray]) throws -> MingCAMPPlus {
        let model = MingCAMPPlus()
        try model.update(parameters: ModuleParameters.unflattened(sanitize(weights)), verify: .all)
        model.train(false); eval(model)
        return model
    }
    /// Lossless remap for the published Ming initializer safetensors and the pinned
    /// StepAudio2 native safetensors. No ONNX graph or conversion runtime is needed.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let raw = weights["onnx::Conv_4423"] != nil
        guard raw else { return weights }
        var numbered = ["head.conv1", "head.layer1.0.conv1", "head.layer1.0.conv2", "head.layer1.0.shortcut.0",
            "head.layer1.1.conv1", "head.layer1.1.conv2", "head.layer2.0.conv1", "head.layer2.0.conv2",
            "head.layer2.0.shortcut.0", "head.layer2.1.conv1", "head.layer2.1.conv2", "head.conv2", "tdnn.linear"]
        for (block, count) in [12, 24, 16].enumerated() {
            for layer in 0..<count { numbered.append("blocks.\(block).layers.\(layer).linear1") }
        }
        numbered.append("transits.2.linear")
        var map: [String: String] = [:]
        for (index, name) in numbered.enumerated() {
            map["onnx::Conv_\(4423 + index * 3)"] = name + ".weight"
            map["onnx::Conv_\(4424 + index * 3)"] = name + ".bias"
        }
        var result: [String: MLXArray] = [:]
        for (key, tensor) in weights {
            var name = map[key] ?? key
            for block in 1...3 {
                for layer in 1...[12, 24, 16][block - 1] {
                    let prefix = "xvector.block\(block).tdnnd\(layer)."
                    if name.hasPrefix(prefix) { name = "blocks.\(block - 1).layers.\(layer - 1)." + name.dropFirst(prefix.count) }
                }
                name = name.replacingOccurrences(of: "xvector.transit\(block).", with: "transits.\(block - 1).")
            }
            name = name.replacingOccurrences(of: "xvector.dense.", with: "dense.")
                .replacingOccurrences(of: ".batchnorm.", with: ".0.")
            let value = tensor.ndim == 4 ? tensor.transposed(0, 2, 3, 1)
                : tensor.ndim == 3 ? tensor.transposed(0, 2, 1) : tensor
            result[name] = value
        }
        return result
    }
}
