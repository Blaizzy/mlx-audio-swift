import Foundation
import MLX
import MLXNN
import MLXFast

private let geluConst: Float = 0.7978846
private let geluConst2: Float = 0.044715

class AlbertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    @ModuleInfo var dropout: MLXNN.Dropout

    init(config: PLBertConfig, vocabSize: Int) {
        _wordEmbeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: vocabSize, dimensions: config.embeddingSize), key: "word_embeddings")
        _positionEmbeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: config.embeddingSize), key: "position_embeddings")
        _tokenTypeEmbeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.typeVocabSize, dimensions: config.embeddingSize), key: "token_type_embeddings")
        _layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.embeddingSize, eps: config.layerNormEps), key: "LayerNorm")
        _dropout = ModuleInfo(wrappedValue: MLXNN.Dropout(p: config.hiddenDropoutProb))
    }

    func callAsFunction(_ inputIds: MLXArray, tokenTypeIds: MLXArray? = nil) -> MLXArray {
        let seqLength = inputIds.shape[1]
        let positionIds = MLXArray(0..<Int32(seqLength)).reshaped([1, seqLength])
        let typeIds = tokenTypeIds ?? MLXArray.zeros(like: inputIds)

        let wordEmb = wordEmbeddings(inputIds)
        let posEmb = positionEmbeddings(positionIds)
        let typeEmb = tokenTypeEmbeddings(typeIds)

        var embeddings = wordEmb + posEmb + typeEmb
        embeddings = layerNorm(embeddings)
        embeddings = dropout(embeddings)
        return embeddings
    }
}

class AlbertSelfAttention: Module {
    let numAttentionHeads: Int
    let attentionHeadSize: Int
    let allHeadSize: Int

    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    @ModuleInfo var dropout: MLXNN.Dropout

    init(config: PLBertConfig) {
        numAttentionHeads = config.numAttentionHeads
        attentionHeadSize = config.hiddenSize / config.numAttentionHeads
        allHeadSize = numAttentionHeads * attentionHeadSize

        _query = ModuleInfo(wrappedValue: Linear(config.hiddenSize, allHeadSize))
        _key = ModuleInfo(wrappedValue: Linear(config.hiddenSize, allHeadSize))
        _value = ModuleInfo(wrappedValue: Linear(config.hiddenSize, allHeadSize))
        _dense = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize))
        _layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "LayerNorm")
        _dropout = ModuleInfo(wrappedValue: MLXNN.Dropout(p: config.attentionProbsDropoutProb))
    }

    private func transposeForScores(_ x: MLXArray) -> MLXArray {
        let newShape = Array(x.shape.dropLast()) + [numAttentionHeads, attentionHeadSize]
        return x.reshaped(newShape).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let q = transposeForScores(query(hiddenStates))
        let k = transposeForScores(key(hiddenStates))
        let v = transposeForScores(value(hiddenStates))

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2))
        scores = scores / Float(attentionHeadSize).squareRoot()
        if let mask = attentionMask {
            scores = scores + mask
        }

        var probs = softmax(scores, axis: -1)
        probs = dropout(probs)

        var context = MLX.matmul(probs, v)
        context = context.transposed(0, 2, 1, 3)
        let newShape = Array(context.shape.dropLast(2)) + [allHeadSize]
        context = context.reshaped(newShape)

        let projected = dense(context)
        return layerNorm(projected + hiddenStates)
    }
}

class AlbertLayer: Module {
    @ModuleInfo var attention: AlbertSelfAttention
    @ModuleInfo(key: "full_layer_layer_norm") var fullLayerNorm: LayerNorm
    @ModuleInfo var ffn: Linear
    @ModuleInfo(key: "ffn_output") var ffnOutput: Linear

    init(config: PLBertConfig) {
        _attention = ModuleInfo(wrappedValue: AlbertSelfAttention(config: config))
        _fullLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "full_layer_layer_norm")
        _ffn = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize))
        _ffnOutput = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.hiddenSize), key: "ffn_output")
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
        let ffnOut = ffChunk(attentionOutput)
        return fullLayerNorm(ffnOut + attentionOutput)
    }

    private func ffChunk(_ x: MLXArray) -> MLXArray {
        let h = ffn(x)
        let activated = 0.5 * h * (1.0 + MLX.tanh(geluConst * (h + geluConst2 * (h ** 3))))
        return ffnOutput(activated)
    }
}

class AlbertLayerGroup: Module {
    @ModuleInfo(key: "albert_layers") var albertLayers: [AlbertLayer]

    init(config: PLBertConfig) {
        _albertLayers = ModuleInfo(wrappedValue: (0..<config.innerGroupNum).map { _ in AlbertLayer(config: config) }, key: "albert_layers")
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var x = hiddenStates
        for layer in albertLayers {
            x = layer(x, attentionMask: attentionMask)
        }
        return x
    }
}

class AlbertEncoder: Module {
    let config: PLBertConfig
    @ModuleInfo(key: "embedding_hidden_mapping_in") var embeddingMapping: Linear
    @ModuleInfo(key: "albert_layer_groups") var albertLayerGroups: [AlbertLayerGroup]

    init(config: PLBertConfig) {
        self.config = config
        _embeddingMapping = ModuleInfo(wrappedValue: Linear(config.embeddingSize, config.hiddenSize), key: "embedding_hidden_mapping_in")
        _albertLayerGroups = ModuleInfo(wrappedValue: (0..<config.numHiddenGroups).map { _ in AlbertLayerGroup(config: config) }, key: "albert_layer_groups")
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var x = embeddingMapping(hiddenStates)
        for i in 0..<config.numHiddenLayers {
            let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)
            x = albertLayerGroups[groupIdx](x, attentionMask: attentionMask)
        }
        return x
    }
}

class Albert: Module {
    let config: PLBertConfig
    @ModuleInfo var embeddings: AlbertEmbeddings
    @ModuleInfo var encoder: AlbertEncoder
    @ModuleInfo var pooler: Linear

    init(config: PLBertConfig, vocabSize: Int) {
        self.config = config
        _embeddings = ModuleInfo(wrappedValue: AlbertEmbeddings(config: config, vocabSize: vocabSize))
        _encoder = ModuleInfo(wrappedValue: AlbertEncoder(config: config))
        _pooler = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize))
    }

    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let embeddingOutput = embeddings(inputIds)

        var mask = attentionMask
        if let m = mask {
            let expanded = m.reshaped([m.shape[0], 1, 1, m.shape[m.ndim - 1]])
            mask = (1.0 - expanded) * Float(-10000.0)
        }

        let encoderOutput = encoder(embeddingOutput, attentionMask: mask)
        let pooledOutput = tanh(pooler(encoderOutput[0..., 0, 0...]))

        return (encoderOutput, pooledOutput)
    }
}
