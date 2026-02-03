import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCodecs

@inline(__always)
private func pocketModulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    return x * (1 + scale) + shift
}

public final class PocketRMSNorm: Module, UnaryLayer {
    public let eps: Float
    public var alpha: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.alpha = MLXArray.ones([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let mean = MLX.mean(x32, axis: -1, keepDims: true)
        let diff = x32 - mean
        let n = max(x32.shape.last ?? 1, 1)
        let denom = MLXArray(Float(max(n - 1, 1)))
        let varr = MLX.sum(diff * diff, axis: -1, keepDims: true) / denom
        let inv = MLXArray(1.0) / MLX.sqrt(varr + MLXArray(eps))
        let y = x32 * inv * alpha.asType(.float32)
        return y.asType(x.dtype)
    }
}

public final class PocketLayerNorm: Module {
    public let eps: Float
    public var weight: MLXArray?
    public var bias: MLXArray?

    public init(channels: Int, eps: Float = 1e-6, elementwiseAffine: Bool = true) {
        self.eps = eps
        if elementwiseAffine {
            self.weight = MLXArray.ones([channels])
            self.bias = MLXArray.zeros([channels])
        } else {
            self.weight = nil
            self.bias = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

public final class PocketTimestepEmbedder: Module {
    public let frequencyEmbeddingSize: Int
    public let freqs: MLXArray
    @ModuleInfo(key: "mlp") public var mlp: [Module]

    public init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256, maxPeriod: Double = 10000.0) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        let half = frequencyEmbeddingSize / 2
        let indices = MLXArray(stride(from: 0, to: half, by: 1)).asType(.float32)
        let scale = -Float(log(maxPeriod)) / Float(half)
        self.freqs = MLX.exp(indices * MLXArray(scale))
        self._mlp = ModuleInfo(wrappedValue: [
            Linear(frequencyEmbeddingSize, hiddenSize, bias: true),
            SiLU(),
            Linear(hiddenSize, hiddenSize, bias: true),
            PocketRMSNorm(dim: hiddenSize)
        ])
        super.init()
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        var tt = t
        if tt.ndim == 1 {
            tt = tt.expandedDimensions(axis: 1)
        }
        let args = tt.asType(.float32) * freqs.reshaped([1, freqs.shape[0]])
        let embedding = concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        var out = embedding
        for layer in mlp {
            out = (layer as! UnaryLayer).callAsFunction(out)
        }
        return out
    }
}

public final class PocketResBlock: Module {
    public final class MLP: Module {
        @ModuleInfo(key: "0") public var linear1: Linear
        @ModuleInfo(key: "2") public var linear2: Linear
        public let act1: SiLU

        public init(channels: Int) {
            self._linear1 = ModuleInfo(wrappedValue: Linear(channels, channels, bias: true))
            self._linear2 = ModuleInfo(wrappedValue: Linear(channels, channels, bias: true))
            self.act1 = SiLU()
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var out = linear1(x)
            out = act1(out)
            return linear2(out)
        }
    }

    @ModuleInfo(key: "in_ln") public var in_ln: PocketLayerNorm
    @ModuleInfo(key: "mlp") public var mlp: [Module]
    @ModuleInfo(key: "adaLN_modulation") public var adaLN_modulation: [Module]

    public init(channels: Int) {
        self._in_ln = ModuleInfo(wrappedValue: PocketLayerNorm(channels: channels, eps: 1e-6))
        self._mlp = ModuleInfo(wrappedValue: [
            Linear(channels, channels, bias: true),
            SiLU(),
            Linear(channels, channels, bias: true)
        ])
        self._adaLN_modulation = ModuleInfo(wrappedValue: [
            SiLU(),
            Linear(channels, 3 * channels, bias: true)
        ])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        var modulation = y
        for layer in adaLN_modulation {
            modulation = (layer as! UnaryLayer).callAsFunction(modulation)
        }
        let splits = split(modulation, indices: [y.shape[1], 2 * y.shape[1]], axis: 1)
        let shift = splits[0]
        let scale = splits[1]
        let gate = splits[2]

        var h = pocketModulate(in_ln(x), shift: shift, scale: scale)
        for layer in mlp {
            h = (layer as! UnaryLayer).callAsFunction(h)
        }
        return x + gate * h
    }
}

public final class PocketFinalLayer: Module {
    @ModuleInfo(key: "norm_final") public var norm_final: PocketLayerNorm
    @ModuleInfo(key: "linear") public var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") public var adaLN_modulation: [Module]

    public init(modelChannels: Int, outChannels: Int) {
        self._norm_final = ModuleInfo(wrappedValue: PocketLayerNorm(channels: modelChannels, eps: 1e-6, elementwiseAffine: false))
        self._linear = ModuleInfo(wrappedValue: Linear(modelChannels, outChannels, bias: true))
        self._adaLN_modulation = ModuleInfo(wrappedValue: [
            SiLU(),
            Linear(modelChannels, 2 * modelChannels, bias: true)
        ])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        var modulation = c
        for layer in adaLN_modulation {
            modulation = (layer as! UnaryLayer).callAsFunction(modulation)
        }
        let parts = split(modulation, indices: [c.shape[1]], axis: 1)
        let shift = parts[0]
        let scale = parts[1]
        let h = pocketModulate(norm_final(x), shift: shift, scale: scale)
        return linear(h)
    }
}

public final class SimpleMLPAdaLN: Module {
    public let in_channels: Int
    public let model_channels: Int
    public let out_channels: Int
    public let num_res_blocks: Int
    public let num_time_conds: Int

    @ModuleInfo(key: "time_embed") public var time_embed: [PocketTimestepEmbedder]
    @ModuleInfo(key: "cond_embed") public var cond_embed: Linear
    @ModuleInfo(key: "input_proj") public var input_proj: Linear
    @ModuleInfo(key: "res_blocks") public var res_blocks: [PocketResBlock]
    @ModuleInfo(key: "final_layer") public var final_layer: PocketFinalLayer

    public init(
        inChannels: Int,
        modelChannels: Int,
        outChannels: Int,
        condChannels: Int,
        numResBlocks: Int,
        numTimeConds: Int = 2
    ) {
        self.in_channels = inChannels
        self.model_channels = modelChannels
        self.out_channels = outChannels
        self.num_res_blocks = numResBlocks
        self.num_time_conds = numTimeConds

        precondition(numTimeConds != 1, "num_time_conds must be != 1 for AdaLN conditioning")

        self._time_embed = ModuleInfo(wrappedValue: (0..<numTimeConds).map { _ in
            PocketTimestepEmbedder(hiddenSize: modelChannels)
        })
        self._cond_embed = ModuleInfo(wrappedValue: Linear(condChannels, modelChannels, bias: true))
        self._input_proj = ModuleInfo(wrappedValue: Linear(inChannels, modelChannels, bias: true))
        self._res_blocks = ModuleInfo(wrappedValue: (0..<numResBlocks).map { _ in
            PocketResBlock(channels: modelChannels)
        })
        self._final_layer = ModuleInfo(wrappedValue: PocketFinalLayer(modelChannels: modelChannels, outChannels: outChannels))
        super.init()
    }

    public func callAsFunction(_ c: MLXArray, _ s: MLXArray, _ t: MLXArray, _ x: MLXArray) -> MLXArray {
        let ts = [s, t]
        var xx = input_proj(x)
        var tCombined = MLXArray.zeros([xx.shape[0], model_channels])
        for (idx, embed) in time_embed.enumerated() {
            tCombined = tCombined + embed(ts[idx])
        }
        tCombined = tCombined / MLXArray(Float(num_time_conds))
        let cc = cond_embed(c)
        let y = tCombined + cc
        for block in res_blocks {
            xx = block(xx, y)
        }
        return final_layer(xx, y)
    }
}
