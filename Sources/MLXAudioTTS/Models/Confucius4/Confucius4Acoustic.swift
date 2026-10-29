// Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXNN

/// Semantic-to-acoustic conditional flow matching, with a U-shaped DiT and a
/// gated WaveNet output network. All tensors use the published folded layout.
final class Confucius4Acoustic: Confucius4Weights {
    let heads: Int
    let depth: Int
    let waveLayers: Int
    var frequencies: MLXArray
    init(_ weights: [String: MLXArray], heads: Int = 8, depth: Int = 13, waveLayers: Int = 8) {
        self.heads = heads; self.depth = depth; self.waveLayers = waveLayers
        frequencies = weights["decoder.estimator.freqs_cis"]!
        super.init(weights)
    }
    func timestep(_ t: MLXArray, _ name: String) -> MLXArray {
        let p = "decoder.estimator." + name
        let half = self[p + ".time_mlp.0.weight"].shape[1] / 2
        let frequencies = exp(MLX.arange(half).asType(.float32) * Float(-log(10000.0) / Double(half)))
        let angles = 1000 * t[0..., .newAxis] * frequencies[.newAxis]
        return linear(silu(linear(concatenated([cos(angles), sin(angles)], axis: -1), p + ".time_mlp.0")), p + ".time_mlp.2")
    }
    func rope(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], h = x.shape[2], half = x.shape[3] / 2
        if t > frequencies.shape[0] {
            // Recover the published angular frequencies without recomputing its
            // original rows; long reference clips retain identical positions.
            let theta = atan2(frequencies[1, 0..., 1], frequencies[1, 0..., 0])
            let angles = MLX.arange(frequencies.shape[0], t).asType(.float32)[0..., .newAxis] * theta[.newAxis]
            frequencies = concatenated([frequencies, stacked([cos(angles), sin(angles)], axis: -1)], axis: 0)
        }
        let paired = x.reshaped([b, t, h, half, 2])
        let c = frequencies[0..<t, 0..., 0].reshaped([1, t, 1, half])
        let s = frequencies[0..<t, 0..., 1].reshaped([1, t, 1, half])
        let real = paired[0..., 0..., 0..., 0..., 0], imaginary = paired[0..., 0..., 0..., 0..., 1]
        return stacked([real * c - imaginary * s, imaginary * c + real * s], axis: -1).reshaped(x.shape)
    }
    func adaptiveNorm(_ x: MLXArray, _ condition: MLXArray, _ p: String) -> MLXArray {
        let pair = split(linear(condition, p + ".modulation"), parts: 2, axis: -1)
        let normalized = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + 1e-5) * self[p + ".norm.weight"]
        return normalized * pair[0].expandedDimensions(axis: 1) + pair[1].expandedDimensions(axis: 1)
    }
    func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], width = x.shape[2], hd = width / heads
        let pieces = split(linear(x, p + ".wqkv", bias: false), parts: 3, axis: -1)
        let q = rope(pieces[0].reshaped([b, t, heads, hd])).transposed(0, 2, 1, 3)
        let k = rope(pieces[1].reshaped([b, t, heads, hd])).transposed(0, 2, 1, 3)
        let v = pieces[2].reshaped([b, t, heads, hd]).transposed(0, 2, 1, 3)
        let scores = matmul(q, k.swappedAxes(-1, -2)) * Float(1 / sqrt(Double(hd)))
        let out = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped([b, t, width])
        return linear(out, p + ".wo", bias: false)
    }
    func block(_ input: MLXArray, condition: MLXArray, index: Int, skip: MLXArray?) -> MLXArray {
        let p = "decoder.estimator.transformer_blocks.\(index)."
        var x = input
        if let skip { x = linear(concatenated([x, skip], axis: -1), p + "skip_in_linear") }
        x = x + attention(adaptiveNorm(x, condition, p + "attention_norm"), p + "attention")
        let h = adaptiveNorm(x, condition, p + "ffn_norm")
        let ff = silu(linear(h, p + "feed_forward.w1", bias: false)) * linear(h, p + "feed_forward.w3", bias: false)
        return x + linear(ff, p + "feed_forward.w2", bias: false)
    }
    func wavenet(_ input: MLXArray, condition: MLXArray) -> MLXArray {
        let p = "decoder.estimator.wavenet."
        let width = input.shape[2]
        let projected = conv(condition.expandedDimensions(axis: 1), p + "cond_layer.conv")
        var x = input, output = MLXArray.zeros(input.shape, dtype: input.dtype)
        for i in 0..<waveLayers {
            let h = conv(x, p + "in_layers.\(i).conv", padding: 2)
                + projected[0..., 0..., (i * 2 * width)..<((i + 1) * 2 * width)]
            let activation = tanh(h[0..., 0..., 0..<width]) * sigmoid(h[0..., 0..., width...])
            let residual = conv(activation, p + "res_skip_layers.\(i).conv")
            if i + 1 < waveLayers {
                x = x + residual[0..., 0..., 0..<width]
                output = output + residual[0..., 0..., width...]
            } else { output = output + residual }
        }
        return output
    }
    func finalLayer(_ x: MLXArray, _ condition: MLXArray) -> MLXArray {
        let p = "decoder.estimator.final_layer."
        let pieces = split(linear(silu(condition), p + "adaLN_modulation.1"), parts: 2, axis: -1)
        let delta = x - x.mean(axis: -1, keepDims: true)
        let normalized = delta * rsqrt((delta * delta).mean(axis: -1, keepDims: true) + 1e-6)
        return linear(normalized * (1 + pieces[1].expandedDimensions(axis: 1)) + pieces[0].expandedDimensions(axis: 1), p + "linear")
    }
    func forward(_ input: MLXArray, mu: MLXArray, time: MLXArray, speaker: MLXArray, prompt: MLXArray) throws -> MLXArray {
        let p = "decoder.estimator."
        let x = input.transposed(0, 2, 1), b = x.shape[0], t = x.shape[1]
        let t1 = timestep(time, "t_embedder")
        let muProjected = linear(mu, p + "input_embed.mu_projection")
        let speakers = broadcast(speaker.expandedDimensions(axis: 1), to: [b, t, speaker.shape[1]])
        var h = linear(concatenated([x, prompt.transposed(0, 2, 1), muProjected, speakers], axis: -1), p + "input_embed.proj")
        var skips: [MLXArray] = []
        for i in 0..<depth {
            try Task.checkCancellation()
            h = block(h, condition: t1, index: i, skip: i > depth / 2 ? skips.removeLast() : nil)
            if i < depth / 2 { skips.append(h) }
        }
        var residual = adaptiveNorm(h, t1, p + "transformer_norm")
        residual = linear(concatenated([residual, x], axis: -1), p + "skip_linear")
        var output = linear(residual, p + "conv1")
        output = wavenet(output, condition: timestep(time, "t_embedder2")) + linear(residual, p + "res_projection")
        output = finalLayer(output, t1)
        return conv(output, p + "conv2").transposed(0, 2, 1)
    }
    func groupNorm(_ x: MLXArray, _ p: String) -> MLXArray {
        let delta = x - x.mean(axes: [1, 2], keepDims: true)
        return delta * rsqrt((delta * delta).mean(axes: [1, 2], keepDims: true) + 1e-5) * self[p + ".weight"] + self[p + ".bias"]
    }
    func buildMu(codes: MLXArray, latent: MLXArray, referenceFrames: Int) -> MLXArray {
        let embedding = take(self["input_embedding.embedding.weight"], codes, axis: 0)
        let semantic = conv(embedding, "input_embedding.out_project")
        let condition = linear(concatenated([latent, semantic], axis: -1), "encoder_proj")
        var x = linear(condition, "length_regulator.content_in_proj")
        let frames = codes.shape[1], outputLength = Int(Double(frames) * 1.72)
        let indices = minimum((MLX.arange(outputLength).asType(.float32) * (Float(frames) / Float(outputLength))).asType(.int32), frames - 1)
        x = take(x, indices, axis: 1)
        for (ci, gi) in [(0, 1), (3, 4), (6, 7), (9, 10)] {
            x = groupNorm(conv(x, "length_regulator.model.\(ci)", padding: 1), "length_regulator.model.\(gi)")
            x = x * tanh(logAddExp(x, MLXArray.zeros(x.shape, dtype: x.dtype)))
        }
        x = conv(x, "length_regulator.model.12")
        let prompt = broadcast(self["prompt_cond"], to: [1, referenceFrames, x.shape[2]])
        return concatenated([prompt, x], axis: 1)
    }
    func solve(noise: MLXArray, prompt: MLXArray, mu: MLXArray, speaker: MLXArray,
               steps: Int, guidance: Float) throws -> MLXArray {
        let ref = prompt.shape[2], total = noise.shape[2], mel = noise.shape[1]
        let zeros = MLXArray.zeros([1, mel, ref], dtype: noise.dtype)
        let promptFull = concatenated([prompt, MLXArray.zeros([1, mel, total - ref], dtype: prompt.dtype)], axis: 2)
        var x = concatenated([zeros, noise[0..., 0..., ref...]], axis: 2)
        let span = MLX.linspace(0, 1, count: steps + 1).asArray(Float.self)
        var currentTime = Double(span[0]), dt = Double(span[1] - span[0])
        for step in 0..<steps {
            try Task.checkCancellation()
            let time = MLXArray([Float(currentTime), Float(currentTime)])
            let derivative = try forward(concatenated([x, x], axis: 0),
                mu: concatenated([mu, MLXArray.zeros(mu.shape, dtype: mu.dtype)], axis: 0), time: time,
                speaker: concatenated([speaker, MLXArray.zeros(speaker.shape, dtype: speaker.dtype)], axis: 0),
                prompt: concatenated([promptFull, MLXArray.zeros(x.shape, dtype: x.dtype)], axis: 0))
            let guided = (1 + guidance) * derivative[0..<1] - guidance * derivative[1..<2]
            x = x + guided * Float(dt)
            currentTime += dt
            if step + 1 < steps { dt = Double(span[step + 2]) - currentTime }
            x = concatenated([zeros, x[0..., 0..., ref...]], axis: 2)
            eval(x)
        }
        return x
    }
}
