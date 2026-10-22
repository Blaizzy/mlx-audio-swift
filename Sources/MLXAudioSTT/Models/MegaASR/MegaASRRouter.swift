import Foundation
import MLX
import MLXAudioCore
import MLXNN

/// Router features have no Whisper-style dynamic-range clamp.
public enum MegaASRFrontend {
    public static func features(_ audio: MLXArray) -> MLXArray {
        // Python constructs the periodic Hann in double precision, then casts to fp32.
        let window = MLXArray(
            (0..<400).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / 400))) })
        let spectrum = stft(
            audio: audio.asType(.float32), window: window, nFft: 400, hopLength: 160)
        let filters = melFilters(
            sampleRate: 16000, nFft: 400, nMels: 80, norm: "slaney", melScale: .slaney)
        return (log10(maximum(matmul(abs(spectrum).square(), filters), 1e-10)) + 4) / 4
    }
}

final class MegaRouterBatchNorm: Module {
    let weight: MLXArray
    let bias: MLXArray
    @ParameterInfo(key: "running_mean") var mean: MLXArray
    @ParameterInfo(key: "running_var") var variance: MLXArray
    init(_ dimensions: Int) {
        weight = .ones([dimensions])
        bias = .zeros([dimensions])
        _mean.wrappedValue = .zeros([dimensions])
        _variance.wrappedValue = .ones([dimensions])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - mean) / sqrt(variance + 1e-5) * weight + bias
    }
}

final class MegaRouterFrontend: Module {
    let conv1: Conv1d
    let bn1: MegaRouterBatchNorm
    let conv2: Conv1d
    let bn2: MegaRouterBatchNorm
    init(hidden: Int, dimensions: Int) {
        conv1 = Conv1d(
            inputChannels: 80, outputChannels: hidden, kernelSize: 3, stride: 2, padding: 1)
        bn1 = MegaRouterBatchNorm(hidden)
        conv2 = Conv1d(
            inputChannels: hidden, outputChannels: dimensions, kernelSize: 3, stride: 2, padding: 1)
        bn2 = MegaRouterBatchNorm(dimensions)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(bn2(conv2(gelu(bn1(conv1(x)))))) }
}

final class MegaRouterPositions: Module {
    let pe: MLXArray
    init(dimensions: Int, maxLength: Int) { pe = .zeros([1, maxLength, dimensions]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + pe[0..., 0..<x.dim(1), 0...] }
}

final class MegaRouterAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    let heads: Int
    let dimensions: Int
    init(dimensions: Int, heads: Int) {
        self.heads = heads
        self.dimensions = dimensions
        _q.wrappedValue = Linear(dimensions, dimensions)
        _k.wrappedValue = Linear(dimensions, dimensions)
        _v.wrappedValue = Linear(dimensions, dimensions)
        _out.wrappedValue = Linear(dimensions, dimensions)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let d = dimensions / heads
        let query = q(x).reshaped(b, t, heads, d).transposed(0, 2, 1, 3)
        let key = k(x).reshaped(b, t, heads, d).transposed(0, 2, 3, 1)
        let value = v(x).reshaped(b, t, heads, d).transposed(0, 2, 1, 3)
        return out(
            matmul(softmax(matmul(query * pow(Float(d), -0.5), key), axis: -1), value)
                .transposed(0, 2, 1, 3).reshaped(b, t, dimensions))
    }
}

final class MegaRouterTransformerLayer: Module {
    let norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: MegaRouterAttention
    let norm2: LayerNorm
    let linear1: Linear
    let linear2: Linear
    init(dimensions: Int, heads: Int, feedforward: Int) {
        norm1 = LayerNorm(dimensions: dimensions)
        norm2 = LayerNorm(dimensions: dimensions)
        _attention.wrappedValue = MegaRouterAttention(dimensions: dimensions, heads: heads)
        linear1 = Linear(dimensions, feedforward)
        linear2 = Linear(feedforward, dimensions)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + linear2(gelu(linear1(norm2(h))))
    }
}

final class MegaRouterTransformer: Module {
    let layers: [MegaRouterTransformerLayer]
    let norm: LayerNorm
    init(dimensions: Int, heads: Int, feedforward: Int, count: Int) {
        layers = (0..<count).map { _ in
            MegaRouterTransformerLayer(
                dimensions: dimensions, heads: heads, feedforward: feedforward)
        }
        norm = LayerNorm(dimensions: dimensions)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return norm(h)
    }
}

final class MegaRouterPooling: Module {
    let query: Linear
    init(_ dimensions: Int) { query = Linear(dimensions, 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        sum(
            softmax(query(x).squeezed(axis: -1), axis: -1).expandedDimensions(axis: -1) * x, axis: 1
        )
    }
}

final class MegaRouterClassifier: Module {
    let fc1: Linear
    let fc2: Linear
    init(dimensions: Int, hidden: Int) {
        fc1 = Linear(dimensions, hidden)
        fc2 = Linear(hidden, 2)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

public struct MegaASRRoute: Sendable {
    public let degradedProbability: Float
    public var usesLoRA: Bool { degradedProbability >= 0.5 }
}

/// Native audio-quality router using a prepared Mega-ASR sidecar.
public final class MegaASRAudioQualityRouter: Module {
    let frontend: MegaRouterFrontend
    @ModuleInfo(key: "pos_encoder") var positions: MegaRouterPositions
    let transformer: MegaRouterTransformer
    let pooling: MegaRouterPooling
    let classifier: MegaRouterClassifier
    public let maxFrames: Int

    init(
        dimensions: Int, heads: Int, feedforward: Int, layers: Int, frontendHidden: Int,
        classifierHidden: Int, maxFrames: Int
    ) {
        self.maxFrames = maxFrames
        frontend = MegaRouterFrontend(hidden: frontendHidden, dimensions: dimensions)
        _positions.wrappedValue = MegaRouterPositions(dimensions: dimensions, maxLength: maxFrames)
        transformer = MegaRouterTransformer(
            dimensions: dimensions, heads: heads, feedforward: feedforward, count: layers)
        pooling = MegaRouterPooling(dimensions)
        classifier = MegaRouterClassifier(dimensions: dimensions, hidden: classifierHidden)
    }

    static func normalizedWeights(_ raw: [String: MLXArray]) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (name, value) in raw {
            if name.hasSuffix("num_batches_tracked") { continue }
            var key = name
            for (old, new) in [
                ("frontend.conv.0.", "frontend.conv1."), ("frontend.conv.1.", "frontend.bn1."),
                ("frontend.conv.4.", "frontend.conv2."), ("frontend.conv.5.", "frontend.bn2."),
                ("classifier.0.", "classifier.fc1."), ("classifier.3.", "classifier.fc2."),
            ] {
                key = key.replacingOccurrences(of: old, with: new)
            }
            if key.hasSuffix(".self_attn.in_proj_weight")
                || key.hasSuffix(".self_attn.in_proj_bias")
            {
                guard value.ndim > 0, value.dim(0) % 3 == 0 else {
                    throw STTError.invalidInput("Malformed router QKV tensor: \(name)")
                }
                let bias = key.hasSuffix("bias")
                let stem = key.replacingOccurrences(
                    of: bias ? "in_proj_bias" : "in_proj_weight", with: "")
                for (index, part) in split(value, parts: 3, axis: 0).enumerated() {
                    weights[
                        stem + ["q_proj", "k_proj", "v_proj"][index] + (bias ? ".bias" : ".weight")] =
                        part
                }
            } else {
                weights[key] = value
            }
        }
        return weights
    }

    static func fromWeights(_ raw: [String: MLXArray], heads: Int = 4) throws
        -> MegaASRAudioQualityRouter
    {
        guard let norm = raw["transformer.norm.weight"], norm.ndim == 1,
            let ff = raw["transformer.layers.0.linear1.weight"], ff.ndim == 2,
            let conv = raw["frontend.conv.0.weight"], conv.ndim == 3, conv.dim(1) == 3,
            conv.dim(2) == 80,
            let classifier = raw["classifier.0.weight"], classifier.ndim == 2,
            let positions = raw["pos_encoder.pe"], positions.ndim == 3, positions.dim(0) == 1,
            positions.dim(1) > 0, positions.dim(2) == norm.size,
            heads > 0, norm.size > 0, norm.size % heads == 0
        else {
            throw STTError.invalidInput("Malformed or unsupported Mega-ASR router sidecar")
        }
        let indices = Set(
            raw.keys.compactMap { key -> Int? in
                let parts = key.split(separator: ".")
                return parts.count > 2 && parts[0] == "transformer" && parts[1] == "layers"
                    ? Int(parts[2]) : nil
            })
        guard !indices.isEmpty, indices == Set(0..<indices.count) else {
            throw STTError.invalidInput("Router Transformer layers must be contiguous")
        }
        let model = MegaASRAudioQualityRouter(
            dimensions: norm.size, heads: heads, feedforward: ff.dim(0),
            layers: indices.count, frontendHidden: conv.dim(0), classifierHidden: classifier.dim(0),
            maxFrames: positions.dim(1))
        try model.update(parameters: .unflattened(normalizedWeights(raw)), verify: .all)
        model.train(false)
        eval(model)
        return model
    }

    public func logits(audio: MLXArray) -> MLXArray {
        precondition(audio.ndim == 1 || audio.ndim == 2, "Expected mono or time-major stereo PCM")
        let waveform = (audio.ndim == 2 ? audio.mean(axis: -1) : audio).asType(.float32)
        precondition(waveform.size > 200, "Mega-ASR routing needs at least 201 audio samples")
        let melFrames = 1 + waveform.size / 160
        let frameCount = ((melFrames + 1) / 2 + 1) / 2
        precondition(
            frameCount <= maxFrames,
            "Audio exceeds the router context (\(maxFrames) subsampled frames); force a base/robust route or split the recording")
        let mel = MegaASRFrontend.features(waveform).expandedDimensions(axis: 0)
        return classifier(pooling(transformer(positions(frontend(mel))))).squeezed(axis: 0)
    }

    public func route(audio: MLXArray) -> MegaASRRoute {
        MegaASRRoute(
            degradedProbability: softmax(logits(audio: audio), axis: -1)[1].item(Float.self))
    }
}
