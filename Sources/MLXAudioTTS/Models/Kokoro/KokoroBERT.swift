import Foundation
import MLX
import MLXNN

// MARK: - Albert Configuration

struct KokoroAlbertArgs {
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let embeddingSize: Int
    let innerGroupNum: Int
    let numHiddenGroups: Int
    let layerNormEps: Float
    let vocabSize: Int

    init(
        numHiddenLayers: Int, numAttentionHeads: Int,
        hiddenSize: Int, intermediateSize: Int, vocabSize: Int,
        embeddingSize: Int = 128, innerGroupNum: Int = 1,
        numHiddenGroups: Int = 1, layerNormEps: Float = 1e-12
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.innerGroupNum = innerGroupNum
        self.numHiddenGroups = numHiddenGroups
        self.layerNormEps = layerNormEps
    }
}

// MARK: - Embeddings

class AlbertEmbeddings {
    let wordEmbeddings: Embedding
    let positionEmbeddings: Embedding
    let tokenTypeEmbeddings: Embedding
    let layerNorm: LayerNorm

    init(weights: [String: MLXArray], config: KokoroAlbertArgs) {
        wordEmbeddings = Embedding(weight: weights["bert.embeddings.word_embeddings.weight"]!)
        positionEmbeddings = Embedding(weight: weights["bert.embeddings.position_embeddings.weight"]!)
        tokenTypeEmbeddings = Embedding(weight: weights["bert.embeddings.token_type_embeddings.weight"]!)
        layerNorm = LayerNorm(dimensions: config.embeddingSize, eps: config.layerNormEps)

        let lnW = weights["bert.embeddings.LayerNorm.weight"]!
        let lnB = weights["bert.embeddings.LayerNorm.bias"]!
        for i in 0..<config.embeddingSize {
            layerNorm.weight![i] = lnW[i]
            layerNorm.bias![i] = lnB[i]
        }
    }

    func callAsFunction(_ inputIds: MLXArray, tokenTypeIds: MLXArray? = nil) -> MLXArray {
        let seqLength = inputIds.shape[1]
        let posIds = MLX.expandedDimensions(MLXArray(0..<seqLength), axes: [0])
        let ttIds = tokenTypeIds ?? MLXArray.zeros(like: inputIds)
        let embeddings = wordEmbeddings(inputIds) + positionEmbeddings(posIds) + tokenTypeEmbeddings(ttIds)
        return layerNorm(embeddings)
    }
}

// MARK: - Self Attention

class AlbertSelfAttention {
    let numAttentionHeads: Int
    let attentionHeadSize: Int
    let allHeadSize: Int
    let query: Linear
    let key: Linear
    let value: Linear
    let dense: Linear
    let layerNorm: LayerNorm

    init(weights: [String: MLXArray], config: KokoroAlbertArgs, layerNum: Int, innerGroupNum: Int) {
        numAttentionHeads = config.numAttentionHeads
        attentionHeadSize = config.hiddenSize / config.numAttentionHeads
        allHeadSize = numAttentionHeads * attentionHeadSize

        let prefix = "bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum)"
        query = Linear(weight: weights["\(prefix).attention.query.weight"]!,
                       bias: weights["\(prefix).attention.query.bias"]!)
        key = Linear(weight: weights["\(prefix).attention.key.weight"]!,
                     bias: weights["\(prefix).attention.key.bias"]!)
        value = Linear(weight: weights["\(prefix).attention.value.weight"]!,
                       bias: weights["\(prefix).attention.value.bias"])
        dense = Linear(weight: weights["\(prefix).attention.dense.weight"]!,
                       bias: weights["\(prefix).attention.dense.bias"]!)
        layerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        let lnW = weights["\(prefix).attention.LayerNorm.weight"]!
        let lnB = weights["\(prefix).attention.LayerNorm.bias"]!
        for i in 0..<config.hiddenSize {
            layerNorm.weight![i] = lnW[i]
            layerNorm.bias![i] = lnB[i]
        }
    }

    private func transposeForScores(_ x: MLXArray) -> MLXArray {
        var newShape = Array(x.shape.dropLast())
        newShape.append(numAttentionHeads)
        newShape.append(attentionHeadSize)
        return x.reshaped(newShape).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let q = transposeForScores(query(hiddenStates))
        let k = transposeForScores(key(hiddenStates))
        let v = transposeForScores(value(hiddenStates))

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) / sqrt(Float(attentionHeadSize))
        if let mask = attentionMask { scores = scores + mask }
        let probs = MLX.softmax(scores, axis: -1)

        var context = MLX.matmul(probs, v).transposed(0, 2, 1, 3)
        var newShape = Array(context.shape.dropLast(2))
        newShape.append(allHeadSize)
        context = context.reshaped(newShape)
        return layerNorm(dense(context) + hiddenStates)
    }
}

// MARK: - Albert Layer

class AlbertLayer {
    let attention: AlbertSelfAttention
    let fullLayerLayerNorm: LayerNorm
    let ffn: Linear
    let ffnOutput: Linear

    init(weights: [String: MLXArray], config: KokoroAlbertArgs, layerNum: Int, innerGroupNum: Int) {
        let prefix = "bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum)"
        attention = AlbertSelfAttention(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum)
        ffn = Linear(weight: weights["\(prefix).ffn.weight"]!, bias: weights["\(prefix).ffn.bias"]!)
        ffnOutput = Linear(weight: weights["\(prefix).ffn_output.weight"]!, bias: weights["\(prefix).ffn_output.bias"]!)
        fullLayerLayerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        let lnW = weights["\(prefix).full_layer_layer_norm.weight"]!
        let lnB = weights["\(prefix).full_layer_layer_norm.bias"]!
        for i in 0..<config.hiddenSize {
            fullLayerLayerNorm.weight![i] = lnW[i]
            fullLayerLayerNorm.bias![i] = lnB[i]
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let attnOut = attention(hiddenStates, attentionMask: attentionMask)
        let ffnOut = ffnOutput(MLXNN.gelu(ffn(attnOut)))
        return fullLayerLayerNorm(ffnOut + attnOut)
    }
}

// MARK: - Albert Layer Group

class AlbertLayerGroup {
    let layers: [AlbertLayer]

    init(config: KokoroAlbertArgs, layerNum: Int, weights: [String: MLXArray]) {
        layers = (0..<config.innerGroupNum).map {
            AlbertLayer(weights: weights, config: config, layerNum: layerNum, innerGroupNum: $0)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        layers.reduce(hiddenStates) { $1($0, attentionMask: attentionMask) }
    }
}

// MARK: - Albert Encoder

class AlbertEncoder {
    let config: KokoroAlbertArgs
    let embeddingHiddenMappingIn: Linear
    let layerGroups: [AlbertLayerGroup]

    init(weights: [String: MLXArray], config: KokoroAlbertArgs) {
        self.config = config
        embeddingHiddenMappingIn = Linear(
            weight: weights["bert.encoder.embedding_hidden_mapping_in.weight"]!,
            bias: weights["bert.encoder.embedding_hidden_mapping_in.bias"]!
        )
        layerGroups = (0..<config.numHiddenGroups).map {
            AlbertLayerGroup(config: config, layerNum: $0, weights: weights)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var output = embeddingHiddenMappingIn(hiddenStates)
        for i in 0..<config.numHiddenLayers {
            let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)
            output = layerGroups[groupIdx](output, attentionMask: attentionMask)
        }
        return output
    }
}

// MARK: - Custom Albert (Top-level BERT)

class KokoroBERT {
    let config: KokoroAlbertArgs
    let embeddings: AlbertEmbeddings
    let encoder: AlbertEncoder
    let pooler: Linear

    init(weights: [String: MLXArray], config: KokoroAlbertArgs) {
        self.config = config
        embeddings = AlbertEmbeddings(weights: weights, config: config)
        encoder = AlbertEncoder(weights: weights, config: config)
        pooler = Linear(weight: weights["bert.pooler.weight"]!, bias: weights["bert.pooler.bias"]!)
    }

    func callAsFunction(
        _ inputIds: MLXArray, attentionMask: MLXArray? = nil
    ) -> (sequenceOutput: MLXArray, pooledOutput: MLXArray) {
        let embOutput = embeddings(inputIds)
        var maskProcessed: MLXArray?
        if let mask = attentionMask {
            let shape = mask.shape
            maskProcessed = mask.reshaped([shape[0], 1, 1, shape[1]])
            maskProcessed = (1.0 - maskProcessed!) * -10000.0
        }
        let seqOutput = encoder(embOutput, attentionMask: maskProcessed)
        let pooledOutput = MLX.tanh(pooler(seqOutput[0..., 0, 0...]))
        return (seqOutput, pooledOutput)
    }
}
