// Copyright © 2025 Apple Inc. Adapted from mlx-lm (MIT).

import Foundation
import MLX
import MLXFast
import MLXNN

public struct NemotronHConfiguration: Decodable, Sendable {
    public var hiddenSize = 2688
    public var vocabularySize = 131072
    public var intermediateSize = 1856
    public var attentionHeads = 32
    public var kvHeads = 2
    public var headDim = 128
    public var mambaHeads = 64
    public var mambaHeadDim = 64
    public var stateSize = 128
    public var convolutionKernel = 4
    public var groups = 8
    public var epsilon: Float = 1e-5
    public var attentionBias = false
    public var mambaBias = false
    public var convolutionBias = true
    public var mlpBias = false
    public var experts = 128
    public var selectedExperts = 6
    public var expertGroups = 1
    public var selectedGroups = 1
    public var sharedExperts = 1
    public var expertIntermediateSize = 1856
    public var sharedIntermediateSize = 3712
    public var latentSize: Int?
    public var normalizeExpertWeights = true
    public var routedScale: Float = 2.5
    public var minimumTimeStep: Float = 0
    public var maximumTimeStep: Float = .infinity
    public var blockTypes: [String] = []

    public init() {}

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        func v<T: Decodable>(_ key: String, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: Key(key)) ?? fallback
        }
        hiddenSize = try v("hidden_size", hiddenSize)
        vocabularySize = try v("vocab_size", vocabularySize)
        intermediateSize = try v("intermediate_size", intermediateSize)
        attentionHeads = try v("num_attention_heads", attentionHeads)
        kvHeads = try v("num_key_value_heads", kvHeads)
        headDim = try v("head_dim", headDim)
        mambaHeads = try v("mamba_num_heads", mambaHeads)
        mambaHeadDim = try v("mamba_head_dim", mambaHeadDim)
        stateSize = try v("ssm_state_size", stateSize)
        convolutionKernel = try v("conv_kernel", convolutionKernel)
        groups = try v("n_groups", groups)
        epsilon = try v("layer_norm_epsilon", epsilon)
        attentionBias = try v("attention_bias", attentionBias)
        mambaBias = try v("mamba_proj_bias", mambaBias)
        convolutionBias = try v("use_conv_bias", convolutionBias)
        mlpBias = try v("mlp_bias", mlpBias)
        experts = try v("n_routed_experts", experts)
        selectedExperts = try v("num_experts_per_tok", selectedExperts)
        expertGroups = try v("n_group", expertGroups)
        selectedGroups = try v("topk_group", selectedGroups)
        sharedExperts = try v("n_shared_experts", sharedExperts)
        expertIntermediateSize = try v("moe_intermediate_size", intermediateSize)
        sharedIntermediateSize = try v(
            "moe_shared_expert_intermediate_size", sharedIntermediateSize)
        latentSize = try c.decodeIfPresent(Int.self, forKey: Key("moe_latent_size"))
        normalizeExpertWeights = try v("norm_topk_prob", normalizeExpertWeights)
        routedScale = try v("routed_scaling_factor", routedScale)
        blockTypes = try v("hybrid_override_pattern", [String]())
        if blockTypes.isEmpty { blockTypes = try v("layers_block_type", [String]()) }
        let names = [
            "mamba": "M", "linear_attention": "M", "attention": "*",
            "full_attention": "*", "moe": "E", "mlp": "-",
        ]
        blockTypes = blockTypes.map { names[$0] ?? $0 }
        if c.contains(Key("time_step_limit")), !(try c.decodeNil(forKey: Key("time_step_limit"))) {
            var limits = try c.nestedUnkeyedContainer(forKey: Key("time_step_limit"))
            func number(_ container: inout UnkeyedDecodingContainer) throws -> Float {
                if let value = try? container.decode(Float.self) { return value }
                let tagged = try container.decode([String: String].self)
                guard let name = tagged["__float__"], let value = Float(name) else {
                    throw DecodingError.dataCorruptedError(
                        in: container, debugDescription: "Invalid time-step limit")
                }
                return value
            }
            minimumTimeStep = try number(&limits)
            maximumTimeStep = try number(&limits)
        }
    }

    public func validate() throws {
        guard hiddenSize > 0, vocabularySize > 0, intermediateSize > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0, headDim > 0,
            mambaHeads > 0, mambaHeadDim > 0, groups > 0, mambaHeads % groups == 0,
            stateSize > 0, convolutionKernel > 0, epsilon > 0,
            experts > 0, selectedExperts > 0, selectedExperts <= experts,
            expertGroups > 0, experts % expertGroups == 0,
            selectedGroups > 0, selectedGroups <= expertGroups,
            selectedExperts <= selectedGroups * (experts / expertGroups),
            expertGroups == 1 || experts / expertGroups >= 2,
            sharedExperts >= 0, expertIntermediateSize > 0, sharedIntermediateSize > 0,
            latentSize == nil || latentSize! > 0,
            routedScale.isFinite, minimumTimeStep >= 0, maximumTimeStep > minimumTimeStep,
            !blockTypes.isEmpty, blockTypes.allSatisfy({ ["M", "*", "E", "-"].contains($0) })
        else {
            throw NSError(
                domain: "NemotronH", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid Nemotron-H configuration"])
        }
    }
}

/// Per-recording convolution, recurrent-state and attention caches.
public final class NemotronHCache {
    var convolution: [MLXArray?]
    var recurrent: [MLXArray?]
    var attention: [KVCache?]

    init(_ config: NemotronHConfiguration) {
        convolution = Array(repeating: nil, count: config.blockTypes.count)
        recurrent = Array(repeating: nil, count: config.blockTypes.count)
        attention = config.blockTypes.map { $0 == "*" ? KVCacheSimple() : nil }
    }

    public var state: [MLXArray] {
        convolution.compactMap { $0 } + recurrent.compactMap { $0 }
            + attention.compactMap { $0 }.flatMap { $0.state }
    }
}

final class NemotronHGatedNorm: Module {
    let weight: MLXArray
    let groupSize: Int
    let epsilon: Float
    init(width: Int, groupSize: Int, epsilon: Float) {
        weight = .ones([width])
        self.groupSize = groupSize
        self.epsilon = epsilon
    }
    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let gated = x.asType(.float32) * silu(gate.asType(.float32))
        let groups = gated.reshaped(Array(gated.shape.dropLast()) + [-1, groupSize])
        let normalized = MLXFast.rmsNorm(groups, weight: .ones([groupSize]), eps: epsilon).reshaped(
            x.shape)
        return (normalized * weight).asType(x.dtype)
    }
}

final class NemotronHMambaMixer: Module {
    @ModuleInfo(key: "in_proj") var input: Linear
    let conv1d: Conv1d
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    let D: MLXArray
    let norm: NemotronHGatedNorm
    @ModuleInfo(key: "out_proj") var output: Linear
    let config: NemotronHConfiguration

    init(_ c: NemotronHConfiguration) {
        config = c
        let width = c.mambaHeads * c.mambaHeadDim
        let convWidth = width + 2 * c.groups * c.stateSize
        _input.wrappedValue = Linear(
            c.hiddenSize, width + convWidth + c.mambaHeads, bias: c.mambaBias)
        conv1d = Conv1d(
            inputChannels: convWidth, outputChannels: convWidth,
            kernelSize: c.convolutionKernel, groups: convWidth, bias: c.convolutionBias)
        _aLog.wrappedValue = log(MLXArray(1 ... c.mambaHeads).asType(.float32))
        _dtBias.wrappedValue = .ones([c.mambaHeads])
        D = .ones([c.mambaHeads])
        norm = NemotronHGatedNorm(width: width, groupSize: width / c.groups, epsilon: c.epsilon)
        _output.wrappedValue = Linear(width, c.hiddenSize, bias: c.mambaBias)
    }

    func callAsFunction(_ x: MLXArray, cache: NemotronHCache?, layer: Int) -> MLXArray {
        let c = config
        let width = c.mambaHeads * c.mambaHeadDim
        let convWidth = width + 2 * c.groups * c.stateSize
        let projected = input(x)
        let gate = projected[0..., 0..., ..<width]
        let convInput = projected[0..., 0..., width ..< (width + convWidth)]
        let dt = projected[0..., 0..., (width + convWidth)...]
        let previous =
            cache?.convolution[layer]
            ?? MLXArray.zeros([x.dim(0), c.convolutionKernel - 1, convWidth], dtype: x.dtype)
        let padded = concatenated([previous, convInput], axis: 1)
        if let cache {
            cache.convolution[layer] =
                padded[0..., (padded.dim(1) - c.convolutionKernel + 1)..., 0...]
        }
        let convolved = silu(conv1d(padded))
        let signal = convolved[0..., 0..., ..<width].reshaped(
            x.dim(0), x.dim(1), c.mambaHeads, c.mambaHeadDim)
        let b = convolved[0..., 0..., width ..< (width + c.groups * c.stateSize)]
            .reshaped(x.dim(0), x.dim(1), c.groups, c.stateSize)
        let cc = convolved[0..., 0..., (width + c.groups * c.stateSize)...]
            .reshaped(x.dim(0), x.dim(1), c.groups, c.stateSize)
        // NeMo retains dynamics, skip accumulation and recurrent memory in FP32.
        let scan = Mamba2Scan.callAsFunction(
            x: signal.asType(.float32), aLog: aLog, b: b.asType(.float32),
            c: cc.asType(.float32), d: D, dt: dt, dtBias: dtBias,
            state: cache?.recurrent[layer], minimumTimeStep: c.minimumTimeStep,
            maximumTimeStep: c.maximumTimeStep)
        cache?.recurrent[layer] = scan.state
        let y = scan.output.reshaped(x.dim(0), x.dim(1), width).asType(convolved.dtype)
        return output(norm(y, gate: gate))
    }
}

final class NemotronHAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    let config: NemotronHConfiguration
    init(_ c: NemotronHConfiguration) {
        config = c
        _q.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: c.attentionBias)
        _k.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: c.attentionBias)
        _v.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: c.attentionBias)
        _o.wrappedValue = Linear(c.attentionHeads * c.headDim, c.hiddenSize, bias: c.attentionBias)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let c = config
        let batch = x.dim(0)
        let length = x.dim(1)
        let queries = q(x).reshaped(batch, length, c.attentionHeads, c.headDim).transposed(
            0, 2, 1, 3)
        let keys = k(x).reshaped(batch, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
        let values = v(x).reshaped(batch, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
        // The hybrid decoder's attention blocks do not apply rotary embeddings.
        let mask = createAttentionMask(h: x, cache: cache)
        let y = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: pow(Float(c.headDim), -0.5), mask: mask)
        return o(y.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class NemotronHMLP: Module {
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: NemotronHConfiguration, intermediate: Int? = nil) {
        let size = intermediate ?? c.intermediateSize
        _up.wrappedValue = Linear(c.hiddenSize, size, bias: c.mlpBias)
        _down.wrappedValue = Linear(size, c.hiddenSize, bias: c.mlpBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = relu(up(x))
        return down(activated * activated)
    }
}

final class NemotronHRouter: Module {
    let weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
    let config: NemotronHConfiguration
    init(_ c: NemotronHConfiguration) {
        config = c
        weight = .zeros([c.experts, c.hiddenSize])
        _correction.wrappedValue = .zeros([c.experts])
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let c = config
        let original = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).T))
        var selection = original + correction
        if c.expertGroups > 1 && c.selectedGroups < c.expertGroups {
            let grouped = selection.reshaped(
                Array(selection.shape.dropLast()) + [c.expertGroups, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)
            let dropped = argPartition(
                groupScores, kth: c.expertGroups - c.selectedGroups - 1, axis: -1)
            let indices = dropped[.ellipsis, ..<(c.expertGroups - c.selectedGroups)]
            let masked = putAlong(groupScores, indices, values: MLXArray(-Float.infinity), axis: -1)
            selection = MLX.where(
                masked.expandedDimensions(axis: -1) .== -Float.infinity,
                MLXArray(Float(0)), grouped
            ).reshaped(selection.shape)
        }
        let indices = argPartition(-selection, kth: c.selectedExperts - 1, axis: -1)[
            .ellipsis, ..<c.selectedExperts]
        var scores = takeAlong(original, indices, axis: -1)
        if c.selectedExperts > 1 && c.normalizeExpertWeights {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, scores * c.routedScale)
    }
}

final class NemotronHExperts: Module {
    @ModuleInfo var fc1: ExpertLinear
    @ModuleInfo var fc2: ExpertLinear
    init(_ c: NemotronHConfiguration) {
        let width = c.latentSize ?? c.hiddenSize
        _fc1.wrappedValue = ExpertLinear(
            input: width, output: c.expertIntermediateSize, experts: c.experts)
        _fc2.wrappedValue = ExpertLinear(
            input: c.expertIntermediateSize, output: width, experts: c.experts)
    }
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = x.expandedDimensions(axes: [-2, -3])
        var selected = indices
        let useSorted = indices.size >= 64
        var inverse: MLXArray?
        if useSorted {
            let flat = indices.reshaped(-1)
            let order = argSort(flat)
            inverse = argSort(order)
            input = input.reshaped(-1, 1, x.dim(-1))[order.floorDivide(indices.dim(-1))]
            selected = flat[order]
        }
        let activated = relu(fc1(input, indices: selected, sorted: useSorted))
        var y = fc2(activated * activated, indices: selected, sorted: useSorted)
        if let inverse { y = y[inverse].reshaped(indices.shape + [1, x.dim(-1)]) }
        return y.squeezed(axis: -2)
    }
}

final class NemotronHMoE: Module {
    @ModuleInfo(key: "switch_mlp") var experts: NemotronHExperts
    let gate: NemotronHRouter
    @ModuleInfo(key: "shared_experts") var shared: NemotronHMLP?
    @ModuleInfo(key: "fc1_latent_proj") var latentInput: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var latentOutput: Linear?
    init(_ c: NemotronHConfiguration) {
        _experts.wrappedValue = NemotronHExperts(c)
        gate = NemotronHRouter(c)
        _shared.wrappedValue =
            c.sharedExperts > 0 ? NemotronHMLP(c, intermediate: c.sharedIntermediateSize) : nil
        _latentInput.wrappedValue = c.latentSize.map { Linear(c.hiddenSize, $0, bias: c.mlpBias) }
        _latentOutput.wrappedValue = c.latentSize.map { Linear($0, c.hiddenSize, bias: c.mlpBias) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        let expertInput = latentInput?(x) ?? x
        let selected = experts(expertInput, indices: indices)
        var y = (selected * scores.expandedDimensions(axis: -1)).sum(axis: -2).asType(
            selected.dtype)
        if let latentOutput { y = latentOutput(y) }
        if let shared { y = y + shared(x) }
        return y
    }
}

final class NemotronHBlock: Module {
    let norm: RMSNorm
    let mixer: Module
    let type: String
    init(_ c: NemotronHConfiguration, type: String) {
        self.type = type
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.epsilon)
        switch type {
        case "M": mixer = NemotronHMambaMixer(c)
        case "*": mixer = NemotronHAttention(c)
        case "E": mixer = NemotronHMoE(c)
        default: mixer = NemotronHMLP(c)
        }
    }
    func callAsFunction(_ x: MLXArray, cache: NemotronHCache?, layer: Int) -> MLXArray {
        let normalized = norm(x)
        let y: MLXArray
        switch mixer {
        case let m as NemotronHMambaMixer: y = m(normalized, cache: cache, layer: layer)
        case let m as NemotronHAttention: y = m(normalized, cache: cache?.attention[layer])
        case let m as NemotronHMoE: y = m(normalized)
        default: y = (mixer as! NemotronHMLP)(normalized)
        }
        return x + y
    }
}

public final class NemotronHBackbone: Module {
    @ModuleInfo public var embeddings: Embedding
    let layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var norm: RMSNorm
    init(_ c: NemotronHConfiguration) {
        _embeddings.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = c.blockTypes.map { NemotronHBlock(c, type: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.epsilon)
    }
    public func callAsFunction(
        _ inputs: MLXArray? = nil, embeddings: MLXArray? = nil,
        cache: NemotronHCache? = nil
    ) -> MLXArray {
        var x = embeddings ?? self.embeddings(inputs!)
        for (i, layer) in layers.enumerated() { x = layer(x, cache: cache, layer: i) }
        return norm(x)
    }
}

public final class NemotronHLanguageModel: Module {
    public let backbone: NemotronHBackbone
    @ModuleInfo(key: "lm_head") public var head: Linear
    public let configuration: NemotronHConfiguration
    public init(_ config: NemotronHConfiguration) {
        configuration = config
        backbone = NemotronHBackbone(config)
        _head.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }
    public func makeCache() -> NemotronHCache { NemotronHCache(configuration) }
    public func callAsFunction(
        _ inputs: MLXArray? = nil, embeddings: MLXArray? = nil,
        cache: NemotronHCache? = nil
    ) -> MLXArray {
        head(backbone(inputs, embeddings: embeddings, cache: cache))
    }
}
