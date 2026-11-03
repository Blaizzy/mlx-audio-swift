import MLX
import MLXNN
import MLXAudioCore

final class Maya2AudioInput: Module {
    let codebookSize: Int
    let table: Embedding
    @ParameterInfo(key: "aeos_in") var aeosIn: MLXArray
    let norm: RMSNorm

    init(_ c: Maya2Configuration) {
        codebookSize = c.codebookSize
        table = Embedding(embeddingCount: c.numCodebooks * c.codebookSize, dimensions: c.backbone.hiddenSize)
        _aeosIn.wrappedValue = .zeros([c.backbone.hiddenSize])
        norm = RMSNorm(dimensions: c.backbone.hiddenSize, eps: Float.ulpOfOne)
    }
    func frames(_ codes: MLXArray) -> MLXArray {
        let q = codes.dim(-1)
        return norm(sum(table(codes + MLXArray(0..<q) * codebookSize), axis: -2) / Float(q).squareRoot())
    }
    func aeos() -> MLXArray { norm(aeosIn) }
}

final class Maya2SpeakerAnchor: Module {
    let fc1: Linear
    let fc2: Linear
    @ParameterInfo(key: "null_vec") var nullVec: MLXArray

    init(_ c: Maya2Configuration) {
        fc1 = Linear(c.speakerDim, c.speakerAnchorHidden)
        fc2 = Linear(c.speakerAnchorHidden, c.backbone.hiddenSize)
        _nullVec.wrappedValue = .zeros([c.speakerDim])
    }
    func callAsFunction(_ refs: MLXArray? = nil) -> MLXArray {
        let x = refs.map { mean($0.asType(.float32), axis: 1) } ?? nullVec.expandedDimensions(axis: 0)
        return fc2(gelu(fc1(x.asType(nullVec.dtype))))
    }
}

final class Maya2LanguageModel: Module {
    let model: Qwen3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    init(_ c: Qwen3Configuration) {
        model = Qwen3ModelInner(c)
        _lmHead.wrappedValue = c.tieWordEmbeddings ? nil : Linear(c.hiddenSize, c.vocabularySize, bias: false)
    }
}

final class Maya2Backbone: Module {
    let lm: Maya2LanguageModel
    @ModuleInfo(key: "cb0_head") var cb0Head: Linear
    init(_ c: Maya2Configuration) {
        lm = Maya2LanguageModel(c.backbone)
        _cb0Head.wrappedValue = Linear(c.backbone.hiddenSize, c.codebookSize + 1, bias: false)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> (MLXArray, MLXArray) {
        let h = lm.model(inputEmbeddings: embeddings, cache: cache)
        return (h, cb0Head(h))
    }
}

final class Maya2DepthSelfBlock: Module {
    let n1: LayerNorm
    let n2: LayerNorm
    let qkv: Linear
    let o: Linear
    let f1: Linear
    let f2: Linear
    let heads: Int
    init(_ c: Maya2DepthConfiguration) {
        let d = c.hiddenSize
        heads = c.numAttentionHeads
        n1 = LayerNorm(dimensions: d); n2 = LayerNorm(dimensions: d)
        qkv = Linear(d, 3 * d, bias: false); o = Linear(d, d, bias: false)
        f1 = Linear(d, c.intermediateSize, bias: false); f2 = Linear(c.intermediateSize, d, bias: false)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), d = x.dim(2)
        let qkv = split(qkv(n1(x)), parts: 3, axis: -1).map {
            $0.reshaped(b, t, heads, -1).transposed(0, 2, 1, 3)
        }
        let a = attentionWithCacheUpdate(queries: qkv[0], keys: qkv[1], values: qkv[2],
            cache: cache, scale: 1 / Float(d / heads).squareRoot(), mask: t > 1 ? .causal : .none)
        let h = x + o(a.transposed(0, 2, 1, 3).reshaped(b, t, d))
        return h + f2(gelu(f1(n2(h))))
    }
}

final class Maya2DepthCrossBlock: Module {
    let n: LayerNorm
    let q: Linear
    let kv: Linear
    let o: Linear
    let heads: Int
    init(_ c: Maya2DepthConfiguration) {
        let d = c.hiddenSize
        heads = c.numAttentionHeads; n = LayerNorm(dimensions: d)
        q = Linear(d, d, bias: false); kv = Linear(d, 2 * d, bias: false); o = Linear(d, d, bias: false)
    }
    func callAsFunction(_ x: MLXArray, memory: MLXArray) -> MLXArray {
        if memory.dim(1) == 1 { return x + o(split(kv(memory), parts: 2, axis: -1)[1]) }
        let b = x.dim(0), t = x.dim(1), d = x.dim(2), s = memory.dim(1)
        let query = q(n(x)).reshaped(b, t, heads, -1).transposed(0, 2, 1, 3)
        let pair = split(kv(memory), parts: 2, axis: -1).map { $0.reshaped(b, s, heads, -1).transposed(0, 2, 1, 3) }
        let a = MLXFast.scaledDotProductAttention(queries: query, keys: pair[0], values: pair[1], scale: 1 / Float(d / heads).squareRoot(), mask: .none)
        return x + o(a.transposed(0, 2, 1, 3).reshaped(b, t, d))
    }
}

final class Maya2DepthDecoder: Module {
    static let groupOfLevel = [-1] + Array(0..<8) + Array(repeating: 8, count: 8)
        + Array(repeating: 9, count: 8) + Array(repeating: 10, count: 7)
    let numCodebooks: Int
    let codebookSize: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "code_lo") var codeLo: Embedding
    @ModuleInfo(key: "code_up") var codeUp: Linear
    @ModuleInfo(key: "level_emb") var levelEmb: Embedding
    let groups: [[Maya2DepthSelfBlock]]
    let cross: [Maya2DepthCrossBlock]
    let norm: LayerNorm
    let heads: [Linear]
    @ModuleInfo(key: "spk_proj_campp") var spkProjCampp: Linear
    @ModuleInfo(key: "spk_proj_ecapa") var spkProjEcapa: Linear
    @ParameterInfo(key: "null_mem") var nullMem: MLXArray

    init(_ c: Maya2Configuration) {
        let d = c.depth.hiddenSize
        numCodebooks = c.numCodebooks; codebookSize = c.codebookSize
        _inProj.wrappedValue = Linear(c.backbone.hiddenSize, d, bias: false)
        _codeLo.wrappedValue = Embedding(embeddingCount: (c.numCodebooks - 1) * c.codebookSize, dimensions: c.depth.embeddingRank)
        _codeUp.wrappedValue = Linear(c.depth.embeddingRank, d, bias: false)
        _levelEmb.wrappedValue = Embedding(embeddingCount: c.numCodebooks, dimensions: d)
        groups = (0...Self.groupOfLevel[c.numCodebooks - 1]).map { _ in
            (0..<c.depth.numHiddenLayers).map { _ in Maya2DepthSelfBlock(c.depth) }
        }
        cross = (0..<c.depth.numHiddenLayers).map { _ in Maya2DepthCrossBlock(c.depth) }
        norm = LayerNorm(dimensions: d)
        heads = (1..<c.numCodebooks).map { _ in Linear(d, c.codebookSize, bias: false) }
        _spkProjCampp.wrappedValue = Linear(1024, d, bias: false)
        _spkProjEcapa.wrappedValue = Linear(c.speakerDim, d, bias: false)
        _nullMem.wrappedValue = .zeros([1, 1, d])
    }
    func speakerMemory(_ refs: MLXArray? = nil) -> MLXArray {
        refs.map { spkProjEcapa($0.asType(nullMem.dtype)) } ?? nullMem
    }
    func prefix(hidden: MLXArray, codes: MLXArray, level: Int) -> MLXArray {
        let slots = MLXArray(0..<level)
        let h = inProj(hidden.asType(nullMem.dtype)).expandedDimensions(axis: 1)
        let e = codeUp(codeLo(codes[0..., ..<level] + slots * codebookSize))
        return concatenated([h + levelEmb.weight[..<1], e + levelEmb(slots + 1)], axis: 1)
    }
    func run(_ x: MLXArray, group: Int, memory: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = x
        for i in cross.indices { h = cross[i](groups[group][i](h, cache: cache?[i]), memory: memory) }
        return norm(h[0..., -1, 0...])
    }
    func callAsFunction(hidden: MLXArray, codes: MLXArray, memory: MLXArray? = nil) -> [MLXArray] {
        let mem = memory ?? speakerMemory()
        return (1..<codes.dim(1)).map { level in
            heads[level - 1](run(prefix(hidden: hidden, codes: codes, level: level),
                group: Self.groupOfLevel[level], memory: mem))
        }
    }
    func generate(hidden: MLXArray, cb0: MLXArray, memory: MLXArray, temperature: Float, topK: Int, sampler providedSampler: (any LogitSampler)? = nil) -> MLXArray {
        var codes = [cb0], previous = -1
        var cache: [KVCache] = []
        let sampler = providedSampler ?? GenerateParameters(temperature: temperature, topK: min(topK, codebookSize)).sampler()
        for level in 1..<numCodebooks {
            let group = Self.groupOfLevel[level]
            let x: MLXArray
            if group != previous {
                cache = cross.map { _ in KVCacheSimple() }
                x = prefix(hidden: hidden, codes: stacked(codes, axis: 1), level: level)
            } else {
                let code = codes.last! + (level - 1) * codebookSize
                x = codeUp(codeLo(code)).expandedDimensions(axis: 1) + levelEmb.weight[level]
            }
            let h = run(x, group: group, memory: memory, cache: cache)
            codes.append(sampler.sample(logits: heads[level - 1](h).asType(.float32)))
            previous = group
        }
        return stacked(codes, axis: 1)
    }
}
