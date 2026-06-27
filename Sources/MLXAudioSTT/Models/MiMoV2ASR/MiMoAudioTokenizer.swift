import Foundation
import MLX
import MLXFast
import MLXNN
import MLXAudioCodecs

final class MiMoAudioTokenizerAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: RoPE

    init(embedDim: Int, numHeads: Int, ropeTheta: Int) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _kProj.wrappedValue = Linear(embedDim, embedDim, bias: false)
        _vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: Float(ropeTheta))
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let batch = hiddenStates.shape[0]
        let time = hiddenStates.shape[1]

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped([batch, time, numHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([batch, time, numHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([batch, time, numHeads, headDim]).transposed(0, 2, 1, 3)

        queries = rope(queries)
        keys = rope(keys)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: nil
        )

        return outProj(output.transposed(0, 2, 1, 3).reshaped([batch, time, embedDim]))
    }
}

final class MiMoAudioTokenizerEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MiMoAudioTokenizerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: MiMoAudioTokenizerConfig) {
        _selfAttention.wrappedValue = MiMoAudioTokenizerAttention(
            embedDim: config.dModel,
            numHeads: config.encoderAttentionHeads,
            ropeTheta: config.ropeTheta
        )
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _fc1.wrappedValue = Linear(config.dModel, config.encoderFFNDim, bias: true)
        _fc2.wrappedValue = Linear(config.encoderFFNDim, config.dModel, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var residual = hiddenStates
        var output = selfAttentionLayerNorm(hiddenStates)
        output = selfAttention(output)
        output = residual + output

        residual = output
        output = finalLayerNorm(output)
        output = gelu(fc1(output))
        output = fc2(output)
        return residual + output
    }
}

final class MiMoAudioTokenizerCodebook: Module {
    var embed: MLXArray

    init(codebookSize: Int, dim: Int) {
        self.embed = MLXArray.zeros([codebookSize, dim], type: Float.self)
    }
}

final class MiMoAudioTokenizerVectorQuantization: Module {
    @ModuleInfo(key: "_codebook") var codebook: MiMoAudioTokenizerCodebook

    init(codebookSize: Int, dim: Int) {
        _codebook.wrappedValue = MiMoAudioTokenizerCodebook(codebookSize: codebookSize, dim: dim)
    }

    func encode(_ hiddenStates: MLXArray) -> MLXArray {
        let x = hiddenStates.asType(.float32)
        let embedding = codebook.embed.asType(.float32)
        let distances =
            -(MLX.sum(x * x, axis: -1, keepDims: true)
              - 2 * MLX.matmul(x, embedding.transposed())
              + MLX.sum(embedding * embedding, axis: -1).expandedDimensions(axis: 0))
        return distances.argMax(axis: -1).asType(.int32)
    }

    func decode(_ indices: MLXArray) -> MLXArray {
        take(codebook.embed, indices, axis: 0)
    }
}

final class MiMoAudioTokenizerResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [MiMoAudioTokenizerVectorQuantization]

    init(config: MiMoAudioTokenizerConfig, activeQuantizers: Int) {
        let codebookSizes = Array(config.codebookSize.prefix(max(1, activeQuantizers)))
        _layers.wrappedValue = codebookSizes.map {
            MiMoAudioTokenizerVectorQuantization(codebookSize: $0, dim: config.dModel)
        }
    }

    func encode(_ hiddenStates: MLXArray, nQuantizers: Int? = nil) -> MLXArray {
        let activeQuantizers = min(nQuantizers ?? layers.count, layers.count)
        var residual = hiddenStates.asType(.float32)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(activeQuantizers)

        for layerIndex in 0..<activeQuantizers {
            let indices = layers[layerIndex].encode(residual)
            let quantized = layers[layerIndex].decode(indices)
            residual = residual - quantized
            outputs.append(indices.expandedDimensions(axis: 1))
        }

        return MLX.concatenated(outputs, axis: 1)
    }
}

final class MiMoAudioTokenizerQuantizer: Module {
    @ModuleInfo(key: "vq") var vq: MiMoAudioTokenizerResidualVectorQuantization

    init(config: MiMoAudioTokenizerConfig, activeQuantizers: Int) {
        _vq.wrappedValue = MiMoAudioTokenizerResidualVectorQuantization(
            config: config,
            activeQuantizers: activeQuantizers
        )
    }

    func encode(_ hiddenStates: MLXArray, nQuantizers: Int? = nil) -> MLXArray {
        vq.encode(hiddenStates, nQuantizers: nQuantizers)
    }
}

final class MiMoAudioTokenizerEncoder: Module {
    let config: MiMoAudioTokenizerConfig
    let activeQuantizers: Int

    @ModuleInfo(key: "conv1") var conv1: MLXAudioCodecs.Conv1d
    @ModuleInfo(key: "conv2") var conv2: MLXAudioCodecs.Conv1d
    @ModuleInfo(key: "layers") var layers: [MiMoAudioTokenizerEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "down_sample_layer") var downSampleLayer: [MLXAudioCodecs.Conv1d]
    @ModuleInfo(key: "down_sample_norm") var downSampleNorm: LayerNorm
    @ModuleInfo(key: "quantizer") var quantizer: MiMoAudioTokenizerQuantizer

    init(config: MiMoAudioTokenizerConfig, activeQuantizers: Int) {
        self.config = config
        self.activeQuantizers = activeQuantizers

        _conv1.wrappedValue = MLXAudioCodecs.Conv1d(
            inChannels: config.nMels,
            outChannels: config.dModel,
            ksize: config.kernelSize,
            stride: 1,
            padding: 1,
            bias: true
        )
        _conv2.wrappedValue = MLXAudioCodecs.Conv1d(
            inChannels: config.dModel,
            outChannels: config.dModel,
            ksize: config.kernelSize,
            stride: config.strideSize,
            padding: 1,
            bias: true
        )
        _layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            MiMoAudioTokenizerEncoderLayer(config: config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _downSampleLayer.wrappedValue = [
            MLXAudioCodecs.Conv1d(
                inChannels: config.dModel,
                outChannels: config.dModel,
                ksize: config.avgPooler,
                stride: config.avgPooler,
                padding: 0,
                bias: false
            )
        ]
        _downSampleNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _quantizer.wrappedValue = MiMoAudioTokenizerQuantizer(
            config: config,
            activeQuantizers: activeQuantizers
        )
    }

    func encodeMel(_ mel: MLXArray, nQuantizers: Int? = nil) -> MLXArray {
        var hiddenStates = mel
        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates.expandedDimensions(axis: 0)
        }
        if hiddenStates.shape[1] != config.nMels {
            hiddenStates = hiddenStates.transposed(0, 2, 1)
        }

        hiddenStates = gelu(conv1(hiddenStates))
        hiddenStates = gelu(conv2(hiddenStates))
        hiddenStates = hiddenStates.transposed(0, 2, 1)

        var skipConnection = MLXArray.zeros(hiddenStates.shape, dtype: hiddenStates.dtype)
        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates)
            if let skipLayerID = config.encoderSkipLayerID, index == skipLayerID - 1 {
                skipConnection = hiddenStates
            }
        }

        hiddenStates = layerNorm(hiddenStates + skipConnection)

        let remainder = hiddenStates.shape[1] % config.avgPooler
        if remainder != 0 {
            let padLength = config.avgPooler - remainder
            hiddenStates = MLX.padded(
                hiddenStates,
                widths: [.init((0, 0)), .init((0, padLength)), .init((0, 0))]
            )
        }

        hiddenStates = downSampleLayer[0](hiddenStates.transposed(0, 2, 1))
        hiddenStates = gelu(hiddenStates)
        hiddenStates = hiddenStates.transposed(0, 2, 1)
        hiddenStates = downSampleNorm(hiddenStates)

        let packed = hiddenStates.reshaped([-1, config.dModel])
        return quantizer.encode(packed, nQuantizers: nQuantizers)
    }
}

final class MiMoAudioTokenizerModel: Module {
    let config: MiMoAudioTokenizerConfig
    let activeQuantizers: Int

    @ModuleInfo(key: "encoder") var encoder: MiMoAudioTokenizerEncoder

    init(config: MiMoAudioTokenizerConfig, activeQuantizers: Int) {
        self.config = config
        self.activeQuantizers = activeQuantizers
        _encoder.wrappedValue = MiMoAudioTokenizerEncoder(
            config: config,
            activeQuantizers: activeQuantizers
        )
    }

    func encodeMel(_ mel: MLXArray, nQuantizers: Int? = nil) -> MLXArray {
        encoder.encodeMel(mel, nQuantizers: nQuantizers ?? activeQuantizers)
    }

    static func load(
        modelDirectory: URL,
        config: MiMoAudioTokenizerConfig,
        activeQuantizers: Int
    ) throws -> MiMoAudioTokenizerModel {
        let model = MiMoAudioTokenizerModel(config: config, activeQuantizers: activeQuantizers)
        let weights = try loadWeights(from: modelDirectory)
        let sanitized = sanitize(weights: weights, activeQuantizers: activeQuantizers)
        if let quantization = config.quantization {
            quantize(model: model) { path, _ in
                guard sanitized["\(path).scales"] != nil else {
                    return nil
                }
                return quantization.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [])
        eval(model)
        return model
    }

    private static func loadWeights(from modelDirectory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        let safetensors = files.filter { $0.pathExtension == "safetensors" }
        guard !safetensors.isEmpty else {
            throw NSError(
                domain: "MiMoAudioTokenizerModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No safetensors files found in \(modelDirectory.path)"]
            )
        }

        var merged: [String: MLXArray] = [:]
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            merged.merge(shard) { _, new in new }
        }
        return merged
    }

    private static func sanitize(
        weights: [String: MLXArray],
        activeQuantizers: Int
    ) -> [String: MLXArray] {
        let quantizerPrefix = "encoder.quantizer.vq.layers."
        let hasHFStyleKeys = weights.keys.contains { $0.hasPrefix("encoder.") || $0.hasPrefix("decoder.") }

        return weights.filter { key, _ in
            if hasHFStyleKeys {
                guard key.hasPrefix("encoder.") else {
                    return false
                }
                guard !key.contains(".cluster_size"),
                      !key.contains(".embed_avg"),
                      !key.contains(".inited") else {
                    return false
                }
                guard key.starts(with: "decoder.") == false else {
                    return false
                }
            } else {
                guard key != "position_embedding.inv_freq" else {
                    return false
                }
                guard !key.contains(".cluster_size"),
                      !key.contains(".embed_avg"),
                      !key.contains(".inited") else {
                    return false
                }
            }

            let normalizedKey = hasHFStyleKeys ? key : mlxExportKeyToSwiftKey(key)
            if normalizedKey.hasPrefix(quantizerPrefix) {
                let suffix = normalizedKey.dropFirst(quantizerPrefix.count)
                let indexText = suffix.prefix { $0.isNumber }
                guard let index = Int(indexText), index < activeQuantizers else {
                    return false
                }
            }
            return true
        }.reduce(into: [String: MLXArray]()) { partialResult, item in
            let rawKey = item.key
            let value = item.value
            let key = hasHFStyleKeys ? rawKey : mlxExportKeyToSwiftKey(rawKey)
            let normalizedValue: MLXArray
            if value.ndim == 3, value.shape[1] > value.shape[2] {
                normalizedValue = value.transposed(0, 2, 1)
            } else {
                normalizedValue = value
            }
            partialResult[key] = normalizedValue
        }
    }

    private static func mlxExportKeyToSwiftKey(_ key: String) -> String {
        var mapped = "encoder." + key
        if mapped.hasPrefix("encoder.down_sample.") {
            mapped = mapped.replacingOccurrences(of: "encoder.down_sample.", with: "encoder.down_sample_layer.0.")
        }
        mapped = mapped.replacingOccurrences(of: ".codebook.", with: "._codebook.")
        return mapped
    }
}
