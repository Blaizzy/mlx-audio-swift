// Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXNN
import MLXAudioCore

/// Published tensors retain their source layouts. Quantized GPT Conv1D matrices
/// are already transposed by the publisher; floating-point Conv1D matrices are not.
class Confucius4Weights {
    let weights: [String: MLXArray]
    let groupSize: Int
    let bits: Int
    init(_ weights: [String: MLXArray], groupSize: Int = 64, bits: Int = 8) {
        self.weights = weights; self.groupSize = groupSize; self.bits = bits
    }
    subscript(_ key: String) -> MLXArray { weights[key]! }
    func linear(_ x: MLXArray, _ p: String, bias: Bool = true, convLayout: Bool = false) -> MLXArray {
        let y: MLXArray
        if let scales = weights[p + ".scales"] {
            y = quantizedMM(x, self[p + ".weight"], scales: scales, biases: weights[p + ".biases"],
                            transpose: true, groupSize: groupSize, bits: bits)
        } else {
            let w = self[p + ".weight"]
            y = matmul(x, convLayout ? w : w.T)
        }
        return bias ? y + self[p + ".bias"] : y
    }
    func norm(_ x: MLXArray, _ p: String, eps: Float = 1e-5) -> MLXArray {
        // Match the reference's FP32 population variance, including its ordering.
        let mean = x.mean(axis: -1, keepDims: true)
        let delta = x - mean
        return delta * rsqrt((delta * delta).mean(axis: -1, keepDims: true) + eps) * self[p + ".weight"] + self[p + ".bias"]
    }
    func conv(_ x: MLXArray, _ p: String, padding: Int = 0, dilation: Int = 1,
              groups: Int = 1, bias: Bool = true) -> MLXArray {
        let y = conv1d(x, self[p + ".weight"].transposed(0, 2, 1), padding: padding, dilation: dilation, groups: groups)
        return bias ? y + self[p + ".bias"] : y
    }
    func reflectConv(_ x: MLXArray, _ p: String, dilation: Int = 1) -> MLXArray {
        let padding = dilation * (self[p + ".weight"].shape[2] - 1) / 2
        return conv(Self.reflect(x, padding), p, dilation: dilation)
    }
    static func reflect(_ x: MLXArray, _ p: Int) -> MLXArray {
        guard p > 0 else { return x }
        let length = x.shape[1]
        let left = take(x, MLXArray(Array((1...p).reversed()).map(Int32.init)), axis: 1)
        let right = take(x, MLXArray((0..<p).map { Int32(length - 2 - $0) }), axis: 1)
        return concatenated([left, x, right], axis: 1)
    }
}

/// The 17 Conformer blocks needed for hidden_states[17], including W2V-BERT's
/// clipped relative-key attention and causal depthwise convolution.
final class Confucius4W2VBert: Confucius4Weights {
    let heads: Int
    let layers: Int
    init(_ weights: [String: MLXArray], heads: Int = 16, layers: Int = 17,
         groupSize: Int = 64, bits: Int = 8) {
        self.heads = heads; self.layers = layers
        super.init(weights, groupSize: groupSize, bits: bits)
    }
    func feedForward(_ x: MLXArray, _ p: String) -> MLXArray {
        linear(silu(linear(x, p + ".intermediate_dense")), p + ".output_dense")
    }
    func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], width = x.shape[2], d = width / heads
        func project(_ suffix: String) -> MLXArray {
            linear(x, p + suffix).reshaped([b, t, heads, d]).transposed(0, 2, 1, 3)
        }
        let q = project(".linear_q"), k = project(".linear_k"), v = project(".linear_v")
        let distance = clip(MLX.arange(t)[.newAxis, 0...] - MLX.arange(t)[0..., .newAxis], min: -64, max: 8) + 64
        let pe = take(self[p + ".distance_embedding.weight"], distance, axis: 0)
        let relative = einsum("bhld,lrd->bhlr", q, pe)
        let scores = matmul(q, k.swappedAxes(-1, -2)) / Float(sqrt(Double(d))) + relative / Float(sqrt(Double(d)))
        let out = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped([b, t, width])
        return linear(out, p + ".linear_out")
    }
    func convolution(_ x: MLXArray, _ p: String) -> MLXArray {
        let width = x.shape[2]
        var h = norm(x, p + ".layer_norm")
        h = matmul(h, self[p + ".pointwise_conv1.weight"][0..., 0..., 0].T)
        let pair = split(h, parts: 2, axis: -1)
        h = pair[0] * sigmoid(pair[1])
        let kernel = self[p + ".depthwise_conv.weight"].shape[2]
        h = concatenated([MLXArray.zeros([x.shape[0], kernel - 1, width], dtype: h.dtype), h], axis: 1)
        h = conv(h, p + ".depthwise_conv", groups: width, bias: false)
        h = silu(norm(h, p + ".depthwise_layer_norm"))
        return matmul(h, self[p + ".pointwise_conv2.weight"][0..., 0..., 0].T)
    }
    func layer(_ input: MLXArray, _ index: Int) -> MLXArray {
        let p = "encoder.layers.\(index)."
        var x = input + 0.5 * feedForward(norm(input, p + "ffn1_layer_norm"), p + "ffn1")
        x = x + attention(norm(x, p + "self_attn_layer_norm"), p + "self_attn")
        x = x + convolution(x, p + "conv_module")
        x = x + 0.5 * feedForward(norm(x, p + "ffn2_layer_norm"), p + "ffn2")
        return norm(x, p + "final_layer_norm")
    }
    func callAsFunction(_ features: MLXArray) throws -> MLXArray {
        var x = linear(norm(features, "feature_projection.layer_norm"), "feature_projection.projection")
        for i in 0..<layers { try Task.checkCancellation(); x = layer(x, i); eval(x) }
        return x
    }
}

/// Text projection and ECAPA-TDNN semantic reference pooling.
final class Confucius4Prefix: Confucius4Weights {
    func text(_ ids: MLXArray) -> MLXArray {
        var e = take(self["text_projector.embed.weight"], ids, axis: 0)
        e = linear(silu(linear(e, "text_projector.text_projection_fc1")), "text_projector.text_projection_fc2")
        return e + self["text_position_embedding.embedding.weight"][0..<ids.shape[1]][.newAxis]
    }
    func tdnn(_ x: MLXArray, _ p: String, dilation: Int = 1) -> MLXArray {
        relu(reflectConv(x, p + ".conv", dilation: dilation))
    }
    func res2net(_ x: MLXArray, _ p: String, dilation: Int) -> MLXArray {
        let pieces = split(x, parts: 8, axis: -1)
        var output = [pieces[0]], previous = pieces[0]
        for i in 1..<8 {
            previous = tdnn(i == 1 ? pieces[i] : pieces[i] + previous, p + ".blocks.\(i - 1)", dilation: dilation)
            output.append(previous)
        }
        return concatenated(output, axis: -1)
    }
    func seRes2net(_ x: MLXArray, _ p: String, dilation: Int) -> MLXArray {
        var h = tdnn(x, p + ".tdnn1")
        h = tdnn(res2net(h, p + ".res2net_block", dilation: dilation), p + ".tdnn2")
        let summary = relu(reflectConv(h.mean(axis: 1, keepDims: true), p + ".se_block.conv1"))
        h = h * sigmoid(reflectConv(summary, p + ".se_block.conv2"))
        return h + x
    }
    func stats(_ x: MLXArray, _ w: MLXArray) -> (MLXArray, MLXArray) {
        let mean = (w * x).sum(axis: 1)
        let delta = x - mean.expandedDimensions(axis: 1)
        return (mean, sqrt(maximum((w * (delta * delta)).sum(axis: 1), 1e-12)))
    }
    func condition(_ input: MLXArray) -> MLXArray {
        var x = tdnn(input, "speaker_encoder.blocks.0")
        var features: [MLXArray] = []
        for i in 1...3 { x = seRes2net(x, "speaker_encoder.blocks.\(i)", dilation: i + 1); features.append(x) }
        x = tdnn(concatenated(features, axis: -1), "speaker_encoder.mfa")
        let (mean, std) = stats(x, (MLXArray.ones([x.shape[0], x.shape[1], 1]) / Float(x.shape[1])))
        let attIn = concatenated([x, broadcast(mean.expandedDimensions(axis: 1), to: x.shape), broadcast(std.expandedDimensions(axis: 1), to: x.shape)], axis: -1)
        let att = softmax(reflectConv(tanh(tdnn(attIn, "speaker_encoder.asp.tdnn")), "speaker_encoder.asp.conv"), axis: 1)
        let (weightedMean, weightedStd) = stats(x, att)
        return reflectConv(concatenated([weightedMean, weightedStd], axis: -1).expandedDimensions(axis: 1), "speaker_encoder.fc")
    }
}

final class Confucius4T2S: Confucius4Weights {
    let heads: Int
    let layers: Int
    let bos: Int
    let eos: Int
    init(_ weights: [String: MLXArray], heads: Int = 20, layers: Int = 24, bos: Int = 8192, eos: Int = 8193,
         groupSize: Int = 64, bits: Int = 8) {
        self.heads = heads; self.layers = layers; self.bos = bos; self.eos = eos
        super.init(weights, groupSize: groupSize, bits: bits)
    }
    func semantic(_ ids: MLXArray) -> MLXArray {
        take(self["semantic_embedding.weight"], ids, axis: 0)
            + self["semantic_position_embedding.embedding.weight"][0..<ids.shape[1]][.newAxis]
    }
    func token(_ token: Int, position: Int) -> MLXArray {
        (self["semantic_embedding.weight"][token] + self["semantic_position_embedding.embedding.weight"][position]).reshaped([1, 1, -1])
    }
    func transformer(_ input: MLXArray, cache: [(MLXArray, MLXArray)]? = nil) throws -> (MLXArray, [(MLXArray, MLXArray)]) {
        let b = input.shape[0], t = input.shape[1], width = input.shape[2], d = width / heads
        let mask: MLXArray? = cache == nil && t > 1 ? triu((MLXArray.ones([t, t]) * Float(-1e9)), k: 1) : nil
        var x = input, newCache: [(MLXArray, MLXArray)] = []
        for i in 0..<layers {
            try Task.checkCancellation()
            let p = "transformer.h.\(i)."
            let qkv = split(linear(norm(x, p + "ln_1"), p + "attn.c_attn", convLayout: true), parts: 3, axis: -1)
            func head(_ y: MLXArray) -> MLXArray { y.reshaped([b, t, heads, d]).transposed(0, 2, 1, 3) }
            let q = head(qkv[0])
            var k = head(qkv[1]), v = head(qkv[2])
            if let cache { k = concatenated([cache[i].0, k], axis: 2); v = concatenated([cache[i].1, v], axis: 2) }
            newCache.append((k, v))
            var scores = matmul(q, k.swappedAxes(-1, -2)) * Float(1 / sqrt(Double(d)))
            if let mask { scores = scores + mask }
            let output = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped([b, t, width])
            x = x + linear(output, p + "attn.c_proj", convLayout: true)
            let h = linear(norm(x, p + "ln_2"), p + "mlp.c_fc", convLayout: true)
            let gelu = 0.5 * h * (1 + tanh(Float(sqrt(2 / Double.pi)) * (h + 0.044715 * pow(h, 3))))
            x = x + linear(gelu, p + "mlp.c_proj", convLayout: true)
        }
        return (norm(x, "transformer.ln_f"), newCache)
    }
    func head(_ x: MLXArray) -> MLXArray { linear(norm(x, "final_norm"), "semantic_head") }
    /// CPU Float64 filtering follows the reference's top-p exclusion of the
    /// crossing token. The draw is supplied explicitly for deterministic tests.
    static func sample(_ input: [Float], generated: [Int], temperature: Float, topK: Int, topP: Float,
                       repetitionPenalty: Float, uniform: Double) -> Int {
        var logits = input.map(Double.init)
        for id in Set(generated) where id >= 0 && id < logits.count {
            logits[id] = logits[id] > 0 ? logits[id] / Double(repetitionPenalty) : logits[id] * Double(repetitionPenalty)
        }
        if temperature <= 0 { return logits.indices.max { logits[$0] < logits[$1] }! }
        logits = logits.map { $0 / Double(temperature) }
        let order = logits.indices.sorted { logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1] }
        if topK > 0 && topK < logits.count {
            let threshold = logits[order[topK - 1]]
            for i in logits.indices where logits[i] < threshold { logits[i] = -.infinity }
        }
        let largest = logits.max()!
        let probabilities = logits.map { exp($0 - largest) }
        let sum = probabilities.reduce(0, +)
        var cumulative = 0.0
        for (rank, i) in order.enumerated() {
            cumulative += probabilities[i] / sum
            if rank > 0 && cumulative > Double(topP) { logits[i] = -.infinity }
        }
        let final = logits.map { exp($0 - largest) }, total = final.reduce(0, +)
        let point = min(max(uniform, 0), 1 - Double.ulpOfOne) * total
        var cdf = 0.0
        for i in final.indices { cdf += final[i]; if point < cdf { return i } }
        return order[0]
    }
    func generate(condition: MLXArray, text: MLXArray, parameters p: GenerateParameters) throws -> (MLXArray, MLXArray) {
        let prefix = concatenated([condition, text], axis: 1)
        var (hidden, cache) = try transformer(concatenated([prefix, token(bos, position: 0)], axis: 1))
        var logits = head(hidden[0..., (hidden.shape[1] - 1)...])[0, 0]
        var generated: [Int] = []
        var rng = Confucius4Random(seed: p.seed ?? 0)
        let limit = min(p.maxTokens ?? 512, self["semantic_position_embedding.embedding.weight"].shape[0] - 1)
        for position in 1...limit {
            try Task.checkCancellation(); eval(logits)
            let id = Self.sample(logits.asArray(Float.self), generated: generated, temperature: p.temperature,
                                 topK: p.topK, topP: p.topP, repetitionPenalty: p.repetitionPenalty ?? 10, uniform: rng.next())
            generated.append(id)
            if id == eos { break }
            (hidden, cache) = try transformer(token(id, position: position), cache: cache)
            logits = head(hidden)[0, 0]
            eval(cache.flatMap { [$0.0, $0.1] })
        }
        // Reference always drops the final emitted item, even at the token cap.
        guard generated.count >= 2 else { throw AudioGenerationError.generationFailed("Confucius4 generated no semantic frames") }
        let codes = Array(generated.dropLast())
        guard codes.allSatisfy({ $0 >= 0 && $0 < bos }) else {
            throw AudioGenerationError.generationFailed("Confucius4 emitted an invalid semantic code")
        }
        let allIDs = MLXArray(([bos] + generated).map(Int32.init)).reshaped([1, -1])
        let (full, _) = try transformer(concatenated([prefix, semantic(allIDs)], axis: 1))
        let latent = full[0..., prefix.shape[1]..<(full.shape[1] - 2), 0...]
        eval(latent)
        return (MLXArray(codes.map(Int32.init)).reshaped([1, -1]), latent)
    }
}

/// Local deterministic random stream. Sampling filters and greedy decoding are
/// identical across languages; stochastic seeds are reproducible within Swift.
struct Confucius4Random {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var z = state; z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return Double((z ^ (z >> 31)) >> 11) / 9007199254740992
    }
}
