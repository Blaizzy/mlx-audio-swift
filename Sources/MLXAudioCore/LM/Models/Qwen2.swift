import Foundation
import MLX
import MLXNN

class Qwen2Attention: Module {
    let args: Qwen2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: Qwen2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Qwen2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Qwen2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2Attention
    let mlp: Qwen2MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen2Configuration) {
        _attention.wrappedValue = Qwen2Attention(args)
        self.mlp = Qwen2MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?

    fileprivate let layers: [Qwen2TransformerBlock]
    let norm: RMSNorm?
    private let slidingWindow: Int?
    private let slidingLayerStart: Int

    public init(
        _ args: Qwen2Configuration, includeEmbedding: Bool = true, includeNorm: Bool = true,
        slidingWindow: Int? = nil, slidingLayerStart: Int = 0
    ) {
        self.slidingWindow = slidingWindow
        self.slidingLayerStart = slidingLayerStart
        precondition(args.vocabularySize > 0)

        if includeEmbedding {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        }

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen2TransformerBlock(args)
            }
        self.norm = includeNorm ? RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps) : nil
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        callAsFunction(inputEmbeddings: embed(inputs), cache: cache)
    }

    /// Look up tokens when this stack includes an embedding table.
    public func embed(_ tokens: MLXArray) -> MLXArray {
        guard let embedTokens else {
            preconditionFailure("This Qwen2 stack accepts embeddings only")
        }
        return embedTokens(tokens)
    }

    public func callAsFunction(inputEmbeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        callAsFunction(
            inputEmbeddings: inputEmbeddings, cache: cache,
            mask: createAttentionMask(h: inputEmbeddings, cache: cache?.first))
    }

    /// Run an embedded prompt with an explicit attention mask, including full attention.
    public func callAsFunction(
        inputEmbeddings: MLXArray, cache: [KVCache]? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = inputEmbeddings

        for (i, layer) in layers.enumerated() {
            let layerMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let slidingWindow, i >= slidingLayerStart {
                layerMask = .array(createCausalMask(
                    n: h.dim(1), offset: cache?[i].offset ?? 0, windowSize: slidingWindow))
            } else {
                layerMask = mask
            }
            h = layer(h, mask: layerMask, cache: cache?[i])
        }

        return norm?(h) ?? h
    }
}

public class Qwen2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen2ModelInner
    let configuration: Qwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen2ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        project(model(inputs, cache: cache))
    }

    /// Look up text embeddings before merging modality-specific features.
    public func embedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embed(tokens)
    }

    /// Decode an embedded prompt using the same layers and cache as text input.
    public func callAsFunction(inputEmbeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        project(model(inputEmbeddings: inputEmbeddings, cache: cache))
    }

    private func project(_ hidden: MLXArray) -> MLXArray {
        var out = hidden
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens!.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        // Remove unused precomputed rotary freqs
        return weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

public struct Qwen2Configuration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var kvHeads: Int
    public var ropeTheta: Float = 1_000_000
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings = false

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(
        hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int,
        rmsNormEps: Float, vocabularySize: Int, kvHeads: Int, ropeTheta: Float = 1_000_000,
        ropeTraditional: Bool = false, ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen2Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen2Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen2Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen2Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen2Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: Qwen2Configuration.CodingKeys.ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen2Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - LoRA

extension Qwen2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
