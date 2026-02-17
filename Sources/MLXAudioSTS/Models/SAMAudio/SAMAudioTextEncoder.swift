import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

private struct T5ModelConfig: Codable {
    var vocabSize: Int
    var dModel: Int
    var dKV: Int
    var dFF: Int
    var numLayers: Int
    var numHeads: Int
    var relativeAttentionNumBuckets: Int
    var relativeAttentionMaxDistance: Int
    var dropoutRate: Float
    var layerNormEpsilon: Float
    var isGatedAct: Bool
    var denseActFn: String

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case dKV = "d_kv"
        case dFF = "d_ff"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case dropoutRate = "dropout_rate"
        case layerNormEpsilon = "layer_norm_epsilon"
        case isGatedAct = "is_gated_act"
        case denseActFn = "dense_act_fn"
    }

    init(
        vocabSize: Int = 32128,
        dModel: Int = 768,
        dKV: Int = 64,
        dFF: Int = 3072,
        numLayers: Int = 12,
        numHeads: Int = 12,
        relativeAttentionNumBuckets: Int = 32,
        relativeAttentionMaxDistance: Int = 128,
        dropoutRate: Float = 0.1,
        layerNormEpsilon: Float = 1e-6,
        isGatedAct: Bool = true,
        denseActFn: String = "gelu_new"
    ) {
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.dKV = dKV
        self.dFF = dFF
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.relativeAttentionNumBuckets = relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
        self.dropoutRate = dropoutRate
        self.layerNormEpsilon = layerNormEpsilon
        self.isGatedAct = isGatedAct
        self.denseActFn = denseActFn
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32128
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 768
        dKV = try c.decodeIfPresent(Int.self, forKey: .dKV) ?? 64
        dFF = try c.decodeIfPresent(Int.self, forKey: .dFF) ?? 3072
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 12
        relativeAttentionNumBuckets = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionNumBuckets) ?? 32
        relativeAttentionMaxDistance = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionMaxDistance) ?? 128
        dropoutRate = try c.decodeIfPresent(Float.self, forKey: .dropoutRate) ?? 0.1
        layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-6
        isGatedAct = try c.decodeIfPresent(Bool.self, forKey: .isGatedAct) ?? true
        denseActFn = try c.decodeIfPresent(String.self, forKey: .denseActFn) ?? "gelu_new"
    }
}

private final class T5LayerNorm: Module {
    let eps: Float
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(hiddenSize: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let variance = mean(hiddenStates.asType(.float32) * hiddenStates.asType(.float32), axis: -1, keepDims: true)
        let normed = hiddenStates * rsqrt(variance + MLXArray(eps))
        return weight * normed
    }
}

private final class T5DenseActDense: Module {
    let actFn: String
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: T5ModelConfig) {
        self.actFn = config.denseActFn
        self._wi.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wo.wrappedValue = Linear(config.dFF, config.dModel, bias: false)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hs = wi(hiddenStates)
        if actFn == "relu" {
            hs = relu(hs)
        } else {
            hs = gelu(hs)
        }
        return wo(hs)
    }
}

private final class T5DenseGatedActDense: Module {
    let actFn: String
    @ModuleInfo(key: "wi_0") var wi0: Linear
    @ModuleInfo(key: "wi_1") var wi1: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: T5ModelConfig) {
        self.actFn = config.denseActFn
        self._wi0.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wi1.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wo.wrappedValue = Linear(config.dFF, config.dModel, bias: false)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenGated = wi0(hiddenStates)
        if actFn == "relu" {
            hiddenGated = relu(hiddenGated)
        } else {
            hiddenGated = gelu(hiddenGated)
        }
        let hiddenLinear = wi1(hiddenStates)
        return wo(hiddenGated * hiddenLinear)
    }
}

private final class T5LayerFF: Module {
    @ModuleInfo(key: "DenseReluDense") var denseReluDense: Module
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm

    init(config: T5ModelConfig) {
        if config.isGatedAct {
            self._denseReluDense.wrappedValue = T5DenseGatedActDense(config: config)
        } else {
            self._denseReluDense.wrappedValue = T5DenseActDense(config: config)
        }
        self._layerNorm.wrappedValue = T5LayerNorm(hiddenSize: config.dModel, eps: config.layerNormEpsilon)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let forwarded = layerNorm(hiddenStates)
        let out: MLXArray
        if let gated = denseReluDense as? T5DenseGatedActDense {
            out = gated(forwarded)
        } else if let dense = denseReluDense as? T5DenseActDense {
            out = dense(forwarded)
        } else {
            out = forwarded
        }
        return hiddenStates + out
    }
}

private final class T5Attention: Module {
    let hasRelativeAttentionBias: Bool
    let relativeAttentionNumBuckets: Int
    let relativeAttentionMaxDistance: Int
    let dModel: Int
    let keyValueProjDim: Int
    let nHeads: Int
    let innerDim: Int

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?

    init(config: T5ModelConfig, hasRelativeAttentionBias: Bool = false) {
        self.hasRelativeAttentionBias = hasRelativeAttentionBias
        self.relativeAttentionNumBuckets = config.relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = config.relativeAttentionMaxDistance
        self.dModel = config.dModel
        self.keyValueProjDim = config.dKV
        self.nHeads = config.numHeads
        self.innerDim = nHeads * keyValueProjDim

        self._q.wrappedValue = Linear(dModel, innerDim, bias: false)
        self._k.wrappedValue = Linear(dModel, innerDim, bias: false)
        self._v.wrappedValue = Linear(dModel, innerDim, bias: false)
        self._o.wrappedValue = Linear(innerDim, dModel, bias: false)
        self._relativeAttentionBias.wrappedValue = hasRelativeAttentionBias
            ? Embedding(embeddingCount: relativeAttentionNumBuckets, dimensions: nHeads)
            : nil
    }

    private func relativePositionBucket(
        relativePosition: Int,
        bidirectional: Bool = true,
        numBuckets: Int,
        maxDistance: Int
    ) -> Int {
        var rp = relativePosition
        var buckets = numBuckets
        var relativeBuckets = 0

        if bidirectional {
            buckets /= 2
            if rp > 0 {
                relativeBuckets += buckets
            }
            rp = abs(rp)
        } else {
            rp = max(-rp, 0)
        }

        let maxExact = buckets / 2
        if rp < maxExact {
            return relativeBuckets + rp
        }

        let rpFloat = Float(rp)
        let maxExactFloat = Float(maxExact)
        let maxDistanceFloat = Float(maxDistance)
        let bucketFloat = maxExactFloat + Foundation.log(rpFloat / maxExactFloat)
            / Foundation.log(maxDistanceFloat / maxExactFloat)
            * Float(buckets - maxExact)
        let clamped = min(Int(bucketFloat), buckets - 1)
        return relativeBuckets + clamped
    }

    private func computeBias(queryLength: Int, keyLength: Int) -> MLXArray {
        guard let relativeAttentionBias else {
            return MLXArray.zeros([1, nHeads, queryLength, keyLength])
        }

        var buckets: [Int] = []
        buckets.reserveCapacity(queryLength * keyLength)
        for qPos in 0..<queryLength {
            for kPos in 0..<keyLength {
                let rp = kPos - qPos
                let b = relativePositionBucket(
                    relativePosition: rp,
                    bidirectional: true,
                    numBuckets: relativeAttentionNumBuckets,
                    maxDistance: relativeAttentionMaxDistance
                )
                buckets.append(b)
            }
        }

        let bucketArray = MLXArray(buckets, [queryLength, keyLength]).asType(.int32)
        let values = relativeAttentionBias(bucketArray) // (Q, K, H)
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0) // (1, H, Q, K)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        positionBias: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let batchSize = hiddenStates.shape[0]
        let seqLength = hiddenStates.shape[1]

        var queryStates = q(hiddenStates)
        var keyStates = k(hiddenStates)
        var valueStates = v(hiddenStates)

        queryStates = queryStates.reshaped([batchSize, seqLength, nHeads, keyValueProjDim]).transposed(0, 2, 1, 3)
        keyStates = keyStates.reshaped([batchSize, seqLength, nHeads, keyValueProjDim]).transposed(0, 2, 1, 3)
        valueStates = valueStates.reshaped([batchSize, seqLength, nHeads, keyValueProjDim]).transposed(0, 2, 1, 3)

        var scores = matmul(queryStates, keyStates.transposed(0, 1, 3, 2))

        let computedBias: MLXArray
        if let positionBias {
            computedBias = positionBias
        } else if hasRelativeAttentionBias {
            computedBias = computeBias(queryLength: seqLength, keyLength: seqLength)
        } else {
            computedBias = MLXArray.zeros([1, nHeads, seqLength, seqLength])
        }

        scores = scores + computedBias
        if let mask {
            scores = scores + mask
        }

        let attnWeights = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        var attnOutput = matmul(attnWeights, valueStates)
        attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped([batchSize, seqLength, innerDim])
        attnOutput = o(attnOutput)

        return (attnOutput, computedBias)
    }
}

private final class T5LayerSelfAttention: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: T5Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm

    init(config: T5ModelConfig, hasRelativeAttentionBias: Bool = false) {
        self._selfAttention.wrappedValue = T5Attention(
            config: config,
            hasRelativeAttentionBias: hasRelativeAttentionBias
        )
        self._layerNorm.wrappedValue = T5LayerNorm(hiddenSize: config.dModel, eps: config.layerNormEpsilon)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        positionBias: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let normed = layerNorm(hiddenStates)
        let (attnOutput, bias) = selfAttention(normed, mask: attentionMask, positionBias: positionBias)
        return (hiddenStates + attnOutput, bias)
    }
}

private final class T5Block: Module {
    @ModuleInfo(key: "self_attention") var selfAttention: T5LayerSelfAttention
    @ModuleInfo(key: "ff") var ff: T5LayerFF

    init(config: T5ModelConfig, hasRelativeAttentionBias: Bool = false) {
        self._selfAttention.wrappedValue = T5LayerSelfAttention(
            config: config,
            hasRelativeAttentionBias: hasRelativeAttentionBias
        )
        self._ff.wrappedValue = T5LayerFF(config: config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        positionBias: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (selfAttended, bias) = selfAttention(
            hiddenStates,
            attentionMask: attentionMask,
            positionBias: positionBias
        )
        let ffOut = ff(selfAttended)
        return (ffOut, bias)
    }
}

private final class T5Stack: Module {
    let config: T5ModelConfig
    private var embedTokens: Embedding?

    @ModuleInfo(key: "block") var block: [T5Block]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: T5LayerNorm

    init(config: T5ModelConfig) {
        self.config = config
        self._block.wrappedValue = (0..<config.numLayers).map { i in
            T5Block(config: config, hasRelativeAttentionBias: i == 0)
        }
        self._finalLayerNorm.wrappedValue = T5LayerNorm(
            hiddenSize: config.dModel,
            eps: config.layerNormEpsilon
        )
    }

    func setInputEmbeddings(_ embeddings: Embedding) {
        self.embedTokens = embeddings
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        let embeds: MLXArray
        if let inputsEmbeds {
            embeds = inputsEmbeds
        } else if let inputIDs, let embedTokens {
            embeds = embedTokens(inputIDs)
        } else {
            fatalError("Must provide inputIDs or inputsEmbeds")
        }

        var extendedAttentionMask: MLXArray? = nil
        if let attentionMask {
            var m = attentionMask.asType(.float32).expandedDimensions(axis: 1).expandedDimensions(axis: 1)
            m = (MLXArray(1.0) - m) * MLXArray(-1e9)
            extendedAttentionMask = m
        }

        var hiddenStates = embeds
        var positionBias: MLXArray? = nil
        for layer in block {
            let (out, bias) = layer(
                hiddenStates,
                attentionMask: extendedAttentionMask,
                positionBias: positionBias
            )
            hiddenStates = out
            positionBias = bias
        }

        hiddenStates = finalLayerNorm(hiddenStates)
        return hiddenStates
    }
}

private final class T5Encoder: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: T5Stack

    init(config: T5ModelConfig) {
        let sharedEmbedding = Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel)
        let encoderStack = T5Stack(config: config)
        encoderStack.setInputEmbeddings(sharedEmbedding)

        self._shared.wrappedValue = sharedEmbedding
        self._encoder.wrappedValue = encoderStack
        super.init()
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        encoder(inputIDs: inputIDs, attentionMask: attentionMask)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("decoder.") || key.hasPrefix("lm_head.") {
                continue
            }

            var mapped = key
            if key == "encoder.embed_tokens.weight" {
                mapped = "shared.weight"
            }

            // Map Python reference layer naming to the Swift module names in this file.
            mapped = mapped.replacingOccurrences(of: ".layer.0.", with: ".self_attention.")
            mapped = mapped.replacingOccurrences(of: ".layer.1.", with: ".ff.")

            sanitized[mapped] = value
        }

        return sanitized
    }

    static func loadConfig(from modelFolder: URL) throws -> T5ModelConfig {
        let configURL = modelFolder.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(T5ModelConfig.self, from: configData)
    }

    static func loadWeights(from modelFolder: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: modelFolder, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }
        return weights
    }

    static func fromPretrained(modelFolder: URL) throws -> T5Encoder {
        let config = try loadConfig(from: modelFolder)
        let model = T5Encoder(config: config)

        let rawWeights = try loadWeights(from: modelFolder)
        let sanitized = sanitize(weights: rawWeights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }
}

public final class T5TextEncoder {
    public let config: T5EncoderConfig

    private var model: T5Encoder?
    private var tokenizer: Tokenizer?

    public init(config: T5EncoderConfig) {
        self.config = config
    }

    private static func resolveModelFolder(_ pathOrRepo: String) async throws -> URL {
        let fm = FileManager.default
        if fm.fileExists(atPath: pathOrRepo) {
            return URL(fileURLWithPath: pathOrRepo)
        }

        let hub = HubApi()
        let repo = Hub.Repo(id: pathOrRepo)
        return try await hub.snapshot(
            from: repo,
            matching: ["*.json", "*.safetensors", "*.model", "tokenizer*"]
        )
    }

    private func ensureLoaded() async throws {
        if model != nil, tokenizer != nil {
            return
        }

        let modelFolder = try await Self.resolveModelFolder(config.name)
        tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        model = try T5Encoder.fromPretrained(modelFolder: modelFolder)
    }

    static func buildBatchTokenTensors(
        tokenIDs: [[Int]],
        padTokenID: Int = 0,
        maxLength: Int? = nil,
        padMode: String = "longest"
    ) -> (inputIDs: MLXArray, attentionMask: MLXArray) {
        let targetLength: Int
        if let maxLength {
            targetLength = maxLength
        } else if padMode == "longest" {
            targetLength = tokenIDs.map(\.count).max() ?? 0
        } else {
            targetLength = tokenIDs.map(\.count).max() ?? 0
        }

        let batch = tokenIDs.count
        var idsFlat: [Int] = []
        var maskFlat: [Int] = []
        idsFlat.reserveCapacity(batch * targetLength)
        maskFlat.reserveCapacity(batch * targetLength)

        for var ids in tokenIDs {
            if ids.count > targetLength {
                ids = Array(ids.prefix(targetLength))
            }
            let valid = ids.count
            if valid < targetLength {
                ids += Array(repeating: padTokenID, count: targetLength - valid)
            }

            idsFlat.append(contentsOf: ids)
            maskFlat.append(contentsOf: (0..<targetLength).map { $0 < valid ? 1 : 0 })
        }

        let inputIDs = MLXArray(idsFlat, [batch, targetLength]).asType(.int32)
        let attentionMask = MLXArray(maskFlat, [batch, targetLength]).asType(.bool)
        return (inputIDs, attentionMask)
    }

    public func encode(_ texts: [String]) async throws -> (features: MLXArray, attentionMask: MLXArray) {
        try await ensureLoaded()
        guard let model, let tokenizer else {
            throw SAMAudioError.notImplemented("T5TextEncoder failed to initialize")
        }

        let tokenized = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
        let (inputIDs, attentionMask) = Self.buildBatchTokenTensors(
            tokenIDs: tokenized,
            padTokenID: 0,
            maxLength: config.maxLength,
            padMode: config.padMode
        )

        let features = model(inputIDs: inputIDs, attentionMask: attentionMask)
        return (features, attentionMask)
    }
}
