import Foundation
@preconcurrency import MLX
import MLXNN

func dramaBoxSinusoidalTimestepEmbedding(
    _ timesteps: MLXArray,
    embeddingDim: Int = 256,
    flipSinToCos: Bool = true,
    downscaleFreqShift: Float = 0,
    maxPeriod: Int = 10_000
) -> MLXArray {
    let halfDim = embeddingDim / 2
    var exponent = -log(Float(maxPeriod)) * MLXArray(Array(0..<halfDim).map { Float($0) })
    exponent = exponent / (Float(halfDim) - downscaleFreqShift)
    let freqs = MLX.exp(exponent)
    var emb = timesteps.asType(.float32).expandedDimensions(axis: -1) * freqs
    emb = concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)
    if flipSinToCos {
        let first = emb[.ellipsis, halfDim...]
        let second = emb[.ellipsis, 0..<halfDim]
        emb = concatenated([first, second], axis: -1)
    }
    return emb
}

final class DramaBoxTimestepMLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(hidden: Int) {
        self._linear1.wrappedValue = Linear(256, hidden, bias: true)
        self._linear2.wrappedValue = Linear(hidden, hidden, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.silu(linear1(x)))
    }
}

final class DramaBoxPixArtTimestepEmbedder: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: DramaBoxTimestepMLP

    init(hidden: Int) {
        self._timestepEmbedder.wrappedValue = DramaBoxTimestepMLP(hidden: hidden)
        super.init()
    }

    func callAsFunction(_ timesteps: MLXArray, dtype: DType) -> MLXArray {
        let proj = dramaBoxSinusoidalTimestepEmbedding(timesteps)
        return timestepEmbedder(proj.asType(dtype))
    }
}

final class DramaBoxAdaLayerNormSingle: Module {
    @ModuleInfo var emb: DramaBoxPixArtTimestepEmbedder
    @ModuleInfo var linear: Linear

    init(hidden: Int, coeff: Int) {
        self._emb.wrappedValue = DramaBoxPixArtTimestepEmbedder(hidden: hidden)
        self._linear.wrappedValue = Linear(hidden, coeff * hidden, bias: true)
        super.init()
    }

    func callAsFunction(_ timesteps: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let tEmb = emb(timesteps, dtype: dtype)
        return (linear(MLXNN.silu(tEmb)), tEmb)
    }
}
