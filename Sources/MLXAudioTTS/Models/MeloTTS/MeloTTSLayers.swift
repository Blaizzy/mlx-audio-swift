// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MLXAudioCore

/// Inference tensors retain the published MLX layout, including depthwise and transposed kernels.
struct MeloLayers {
    let weights: [String: MLXArray]
    func w(_ key: String) -> MLXArray { weights[key]! }
    func linear(_ x: MLXArray, _ path: String) -> MLXArray {
        let weight = w(path + ".weight").T
        if let bias = weights[path + ".bias"] { return addMM(bias, x, weight) }
        return matmul(x, weight)
    }
    func conv(_ x: MLXArray, _ path: String, padding: Int = 0, dilation: Int = 1, groups: Int = 1) -> MLXArray {
        let y = conv1d(x.swappedAxes(1, 2), w(path + ".conv.weight"), padding: padding, dilation: dilation, groups: groups)
        return (weights[path + ".conv.bias"].map { y + $0 } ?? y).swappedAxes(1, 2)
    }
    func norm(_ x: MLXArray, _ path: String, eps: Float = 1e-5) -> MLXArray {
        MLXFast.layerNorm(x, weight: w(path + ".weight"), bias: w(path + ".bias"), eps: eps)
    }
    func channelNorm(_ x: MLXArray, _ path: String) -> MLXArray {
        norm(x.swappedAxes(1, 2), path).swappedAxes(1, 2)
    }
    func attention(_ x: MLXArray, _ path: String, heads: Int) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), t = x.dim(2), d = c / heads
        func project(_ name: String) -> MLXArray {
            conv(x, path + ".conv_" + name).reshaped(b, heads, d, t).swappedAxes(2, 3)
        }
        let q = project("q") / sqrt(Float(d)), k = project("k"), v = project("v")
        func relative(_ name: String) -> MLXArray {
            let value = w(path + ".emb_rel_" + name)
            let window = (value.dim(1) - 1) / 2
            let pad = max(t - window - 1, 0), start = max(window + 1 - t, 0)
            let extended = padded(value, widths: [.init(0), .init(pad), .init(0)])
            return extended[0, start..<(start + 2 * t - 1), 0...]
        }
        var scores = matmul(q, k.swappedAxes(2, 3))
        var rel = matmul(q, relative("k").T)
        rel = padded(rel, widths: [.init(0), .init(0), .init(0), .init((0, 1))])
        rel = rel.reshaped(b, heads, 2 * t * t)
        rel = padded(rel, widths: [.init(0), .init(0), .init((0, t - 1))])
        rel = rel.reshaped(b, heads, t + 1, 2 * t - 1)[0..., 0..., 0..<t, (t - 1)...]
        scores = scores + rel
        let probability = softmax(scores, axis: -1)
        var relativeWeights = padded(probability, widths: [.init(0), .init(0), .init(0), .init((0, t - 1))])
        relativeWeights = relativeWeights.reshaped(b, heads, t * t + t * (t - 1))
        relativeWeights = padded(relativeWeights, widths: [.init(0), .init(0), .init((t, 0))])
        relativeWeights = relativeWeights.reshaped(b, heads, t, 2 * t)[0..., 0..., 0..., 1...]
        let output = matmul(probability, v) + matmul(relativeWeights, relative("v"))
        return conv(output.swappedAxes(2, 3).reshaped(b, c, t), path + ".conv_o")
    }
    func encoder(_ input: MLXArray, _ path: String, layers: Int, heads: Int, kernel: Int, g: MLXArray?) -> MLXArray {
        var x = input
        for i in 0..<layers {
            if i == min(2, layers), let g, weights[path + ".spk_emb_linear.weight"] != nil {
                x = x + linear(g.swappedAxes(1, 2), path + ".spk_emb_linear").swappedAxes(1, 2)
            }
            x = channelNorm(x + attention(x, path + ".attn_layers.\(i)", heads: heads), path + ".norm_layers_1.\(i)")
            let h = relu(conv(x, path + ".ffn_layers.\(i).conv_1", padding: (kernel - 1) / 2))
            x = channelNorm(x + conv(h, path + ".ffn_layers.\(i).conv_2", padding: (kernel - 1) / 2), path + ".norm_layers_2.\(i)")
        }
        return x
    }
    func bert(_ ids: MLXArray, layers: Int = 10, heads: Int = 12) -> MLXArray {
        var x = w("embeddings.word_embeddings.weight").take(ids, axis: 0)
        x = x + w("embeddings.position_embeddings.weight")[0..<ids.dim(1)].expandedDimensions(axis: 0)
        x = norm(x + w("embeddings.token_type_embeddings.weight")[0], "embeddings.norm", eps: 1e-12)
        for i in 0..<layers {
            let p = "encoder.layers.\(i)", b = x.dim(0), t = x.dim(1), c = x.dim(2), d = c / heads
            func project(_ n: String) -> MLXArray { linear(x, p + ".attention." + n + "_proj").reshaped(b, t, heads, d).swappedAxes(1, 2) }
            let q = project("query"), k = project("key"), v = project("value")
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(d)), mask: nil)
            let h = attended.swappedAxes(1, 2).reshaped(b, t, c)
            x = norm(x + linear(h, p + ".attention.out_proj"), p + ".ln1")
            x = norm(x + linear(gelu(linear(x, p + ".linear1")), p + ".linear2"), p + ".ln2")
        }
        return x
    }
    func duration(_ input: MLXArray, g: MLXArray) -> MLXArray {
        let x = input + conv(g, "dp.cond")
        let h = channelNorm(relu(conv(x, "dp.conv_1", padding: 1)), "dp.norm_1")
        return conv(channelNorm(relu(conv(h, "dp.conv_2", padding: 1)), "dp.norm_2"), "dp.proj")
    }
    func dds(_ input: MLXArray, _ p: String, g: MLXArray? = nil) -> MLXArray {
        var x = g.map { input + $0 } ?? input
        for i in 0..<3 {
            let dilation = Int(pow(3.0, Double(i)))
            var y = conv(x, p + ".convs_sep.\(i)", padding: dilation, dilation: dilation, groups: x.dim(1))
            y = gelu(channelNorm(y, p + ".norms_1.\(i)"))
            y = gelu(channelNorm(conv(y, p + ".convs_1x1.\(i)"), p + ".norms_2.\(i)"))
            x = x + y
        }
        return x
    }
    func stochasticDuration(_ input: MLXArray, g: MLXArray, noise: Float, initialNoise: MLXArray? = nil) -> MLXArray {
        let x = conv(dds(conv(input, "sdp.pre") + conv(g, "sdp.cond"), "sdp.convs"), "sdp.proj")
        var z = initialNoise ?? MLXRandom.normal([input.dim(0), 2, input.dim(2)]) * noise
        for i in (0..<9).reversed() {
            let p = "sdp.flows.\(i)"
            if i == 0 { z = (z - w(p + ".m")) * exp(-w(p + ".logs")) }
            else if i % 2 == 0 { z = z.take(MLXArray([Int32(1), 0]), axis: 1) }
            else {
                let z0 = z[0..., 0..<1, 0...], z1 = z[0..., 1..<2, 0...]
                let h = conv(dds(conv(z0, p + ".pre"), p + ".convs", g: x), p + ".proj")
                    .reshaped(input.dim(0), 1, 29, input.dim(2)).swappedAxes(2, 3)
                let transformed = Self.inverseSpline(z1, widths: h[0..., 0..., 0..., 0..<10] / sqrt(Float(192)),
                    heights: h[0..., 0..., 0..., 10..<20] / sqrt(Float(192)), derivatives: h[0..., 0..., 0..., 20...])
                z = concatenated([z0, transformed], axis: 1)
            }
        }
        // Match the reference log(exp(z0)) clipping, including its lower saturation.
        return log(maximum(exp(z[0..., 0..<1, 0...]), 1e-5))
    }
    static func inverseSpline(_ input: MLXArray, widths rawW: MLXArray, heights rawH: MLXArray, derivatives rawD: MLXArray, inverse: Bool = true) -> MLXArray {
        let n = rawW.dim(-1), bound: Float = 5
        func bins(_ raw: MLXArray) -> (MLXArray, MLXArray) {
            let values = (0.001 + (1 - 0.001 * Float(n)) * softmax(raw, axis: -1)) * (2 * bound)
            let zero = MLXArray.zeros(Array(raw.shape.dropLast()) + [1])
            let cumulative = concatenated([zero, cumsum(values, axis: -1)], axis: -1) - bound
            let edges = concatenated([zero - bound, cumulative[.ellipsis, 1..<n], zero + bound], axis: -1)
            return (edges, edges[.ellipsis, 1...] - edges[.ellipsis, ..<n])
        }
        let (cw, widths) = bins(rawW), (ch, heights) = bins(rawH)
        let one = MLXArray.ones(Array(rawD.shape.dropLast()) + [1])
        let derivatives = 0.001 + softplus(concatenated([one, rawD, one], axis: -1))
        let idx = clip(sum((input.expandedDimensions(axis: -1) .>= ((inverse ? ch : cw) + 1e-6)).asType(.int32), axis: -1) - 1, min: 0, max: n - 1).expandedDimensions(axis: -1)
        func gather(_ p: MLXArray) -> MLXArray { takeAlong(p, idx, axis: -1).squeezed(axis: -1) }
        let x0 = gather(cw), width = gather(widths), y0 = gather(ch), height = gather(heights)
        let d = height / width, d0 = gather(derivatives), d1 = gather(derivatives[.ellipsis, 1...])
        let output: MLXArray
        if inverse {
            // Standard rational-quadratic inverse: both coefficients depend on the input's offset.
            let offset = input - y0
            let a = offset * (d0 + d1 - 2 * d) + height * (d - d0)
            let b = height * d0 - offset * (d0 + d1 - 2 * d)
            let c = -d * offset
            let root = 2 * c / (-b - sqrt(maximum(b * b - 4 * a * c, 0)))
            output = root * width + x0
        } else {
            let theta = (input - x0) / width, mixed = theta * (1 - theta)
            output = y0 + height * (d * theta * theta + d0 * mixed) / (d + (d0 + d1 - 2 * d) * mixed)
        }
        return which((input .>= -bound) .&& (input .<= bound), output, input)
    }
    func flow(_ input: MLXArray, g: MLXArray, config: MeloTTSConfiguration) -> MLXArray {
        var z = input
        for i in (0..<8).reversed() {
            if i % 2 == 1 { z = z.take(MLXArray(Array((0..<config.interChannels).reversed()).map(Int32.init)), axis: 1) }
            else {
                let p = "flow_layers.\(i)", half = config.interChannels / 2
                let z0 = z[0..., ..<half, 0...]
                let h = encoder(conv(z0, p + ".pre"), p + ".enc", layers: config.flowLayers, heads: config.heads, kernel: 5, g: g)
                z = concatenated([z0, z[0..., half..., 0...] - conv(h, p + ".post")], axis: 1)
            }
        }
        return z
    }
    func decode(_ input: MLXArray, g: MLXArray, config c: MeloTTSConfiguration) -> MLXArray {
        var x = conv(input, "dec.conv_pre", padding: 3) + conv(g, "dec.cond")
        for i in c.upsampleRates.indices {
            let p = "dec.ups.\(i).conv_t"
            x = convTransposed1d(leakyRelu(x, negativeSlope: 0.1).swappedAxes(1, 2), w(p + ".weight"), stride: c.upsampleRates[i], padding: (c.upsampleKernels[i] - c.upsampleRates[i]) / 2)
            x = (x + w(p + ".bias")).swappedAxes(1, 2)
            var outputs: [MLXArray] = []
            for j in c.resblockKernels.indices {
                var h = x
                let p = "dec.resblocks.\(i * c.resblockKernels.count + j)", kernel = c.resblockKernels[j]
                for (k, dilation) in c.resblockDilations[j].enumerated() {
                    let y = conv(leakyRelu(h, negativeSlope: 0.1), p + ".convs1.\(k)", padding: (kernel - 1) * dilation / 2, dilation: dilation)
                    h = h + conv(leakyRelu(y, negativeSlope: 0.1), p + ".convs2.\(k)", padding: (kernel - 1) / 2)
                }
                outputs.append(h)
            }
            x = outputs.dropFirst().reduce(outputs[0], +) / Float(outputs.count)
            eval(x)
        }
        return tanh(conv(leakyRelu(x, negativeSlope: 0.1), "dec.conv_post", padding: 3))
    }
}
