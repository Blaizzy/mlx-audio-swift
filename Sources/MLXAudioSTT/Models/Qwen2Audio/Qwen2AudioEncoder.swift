import Foundation
import MLX
import MLXAudioCore
import MLXNN

final class Qwen2AudioEncoder: Module {
    let config: Qwen2AudioEncoderConfig
    let conv1: Conv1d
    let conv2: Conv1d
    let layers: [WhisperEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    private var _positions: MLXArray

    init(_ config: Qwen2AudioEncoderConfig) {
        self.config = config
        conv1 = Conv1d(
            inputChannels: config.numMelBins, outputChannels: config.dModel, kernelSize: 3,
            padding: 1)
        conv2 = Conv1d(
            inputChannels: config.dModel, outputChannels: config.dModel, kernelSize: 3, stride: 2,
            padding: 1)
        var whisper = WhisperConfig()
        whisper.dModel = config.dModel
        whisper.encoderAttentionHeads = config.encoderAttentionHeads
        whisper.encoderFfnDim = config.encoderFFNDim
        layers = (0 ..< config.encoderLayers).map { _ in WhisperEncoderLayer(config: whisper) }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        let inverse = exp(
            -Float(log(10000.0) / Double(config.dModel / 2 - 1))
                * arange(config.dModel / 2).asType(.float32))
        let phases =
            arange(config.maxSourcePositions + 1).asType(.float32).expandedDimensions(axis: 1)
            * inverse.expandedDimensions(axis: 0)
        _positions = concatenated([sin(phases), cos(phases)], axis: -1)
    }

    func preparePositions() {
        _positions = _positions.asType(conv1.weight.dtype)
        eval(_positions)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var h = gelu(conv2(gelu(conv1(features.transposed(0, 2, 1)))))
        h = h + _positions[..<h.dim(1)]
        for layer in layers { h = layer(h) }
        let b = h.dim(0)
        let pooledLength = h.dim(1) / 2
        h = h[0..., ..<(pooledLength * 2), 0...].reshaped(b, pooledLength, 2, config.dModel).mean(
            axis: 2)
        return layerNorm(h)
    }
}

final class Qwen2AudioProjector: Module {
    let linear: Linear
    init(_ config: Qwen2AudioConfig) {
        linear = Linear(config.audioConfig.dModel, config.textConfig.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(x) }
}
