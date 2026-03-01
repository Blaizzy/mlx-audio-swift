import Foundation
import MLX
import MLXNN

// MARK: - Feature Extractor

class Wav2Vec2FeatureExtractorLayer: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int) {
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, stride: stride, bias: true
        )
        _layerNorm.wrappedValue = LayerNorm(dimensions: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(layerNorm(conv(x)))
    }
}

class Wav2Vec2FeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [Wav2Vec2FeatureExtractorLayer]

    init(config: Wav2Vec2LIDConfig) {
        let inChannels = [1] + Array(config.convDim.dropLast())
        _convLayers.wrappedValue = zip(
            zip(inChannels, config.convDim),
            zip(config.convKernel, config.convStride)
        ).map {
            Wav2Vec2FeatureExtractorLayer(
                inputChannels: $0.0, outputChannels: $0.1,
                kernelSize: $1.0, stride: $1.1
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convLayers.reduce(x) { $1($0) }
    }
}

// MARK: - Feature Projection

class Wav2Vec2FeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var projection: Linear

    init(inputDim: Int, outputDim: Int) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: inputDim)
        _projection.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}

// MARK: - Positional Convolutional Embedding

class Wav2Vec2PositionalConvEmbedding: Module {
    @ModuleInfo var conv: Conv1d

    init(hiddenSize: Int, kernelSize: Int, groups: Int) {
        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize,
            kernelSize: kernelSize, padding: kernelSize / 2,
            groups: groups, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(conv(x))
    }
}

// MARK: - Attention

class Wav2Vec2Attention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        let q = qProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)

        let scale = Float(headDim).squareRoot()
        var attn = matmul(q, k.transposed(0, 1, 3, 2)) / scale
        attn = softmax(attn, axis: -1)

        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return outProj(out)
    }
}

// MARK: - Feed Forward

class Wav2Vec2FeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _intermediateDense.wrappedValue = Linear(hiddenSize, intermediateSize)
        _outputDense.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }
}

// MARK: - Encoder Layer

class Wav2Vec2EncoderLayer: Module {
    @ModuleInfo var attention: Wav2Vec2Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Wav2Vec2FeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int) {
        _attention.wrappedValue = Wav2Vec2Attention(hiddenSize: hiddenSize, numHeads: numHeads)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _feedForward.wrappedValue = Wav2Vec2FeedForward(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize
        )
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + attention(layerNorm(x))
        out = out + feedForward(finalLayerNorm(out))
        return out
    }
}

// MARK: - Encoder

class Wav2Vec2Encoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: Wav2Vec2PositionalConvEmbedding
    @ModuleInfo var layers: [Wav2Vec2EncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(config: Wav2Vec2LIDConfig) {
        _posConvEmbed.wrappedValue = Wav2Vec2PositionalConvEmbedding(
            hiddenSize: config.hiddenSize,
            kernelSize: config.numConvPosEmbeddings,
            groups: config.numConvPosEmbeddingGroups
        )
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            Wav2Vec2EncoderLayer(
                hiddenSize: config.hiddenSize,
                numHeads: config.numAttentionHeads,
                intermediateSize: config.intermediateSize
            )
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        let pos = posConvEmbed(out)
        out = out + pos[0..., ..<out.dim(1), 0...]

        for layer in layers {
            out = layer(out)
        }
        return layerNorm(out)
    }
}
