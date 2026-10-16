import Foundation
import MLX
import MLXNN

/// Replaces the layers selected by `filter` with quantized layers whose arrays
/// are lazy zero placeholders of the shapes the checkpoint fills in.
///
/// `quantize(model:filter:)` would instead run `MLX.quantized` on each random
/// init weight. The three outputs of that one lazy op are siblings that keep
/// each other alive, and `update(parameters:)` overwrites them in place, which
/// bypasses the sibling release in `mlx::core::array::~array`
/// (ml-explore/mlx#4710): the unevaluated graph of every quantized layer
/// outlived the model, on every load. The checkpoint replaces all three arrays
/// anyway, so nothing is lost by starting from placeholders.
public func quantizeWithPlaceholderArrays(
    model: Module,
    filter: (String, Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
) {
    quantize(
        model: model, filter: filter,
        apply: placeholderQuantizedLayer(replacing:groupSize:bits:mode:))
}

/// The `quantize(model:groupSize:bits:filter:)` form of the same replacement.
public func quantizeWithPlaceholderArrays(
    model: Module,
    groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine,
    filter: (String, Module) -> Bool = { _, _ in true }
) {
    quantize(
        model: model, groupSize: groupSize, bits: bits, mode: mode, filter: filter,
        apply: placeholderQuantizedLayer(replacing:groupSize:bits:mode:))
}

private func placeholderQuantizedLayer(
    replacing layer: Module, groupSize: Int, bits: Int, mode: QuantizationMode
) -> Module? {
    // Already-quantized layers stay untouched, as with `quantizeSingle`. The
    // block-scaled modes pack their scales differently from affine; no
    // checkpoint here uses them yet, so they keep the library's own path.
    guard !(layer is Quantized), mode == .affine else {
        return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
    }
    switch layer {
    case let linear as Linear:
        let (outputDimensions, inputDimensions) = linear.shape
        let arrays = PlaceholderQuantizedArrays(
            rows: outputDimensions, columns: inputDimensions, groupSize: groupSize, bits: bits)
        let quantized = QuantizedLinear(
            weight: arrays.weight, bias: linear.bias, scales: arrays.scales, biases: arrays.biases,
            groupSize: groupSize, bits: bits, mode: mode)
        quantized.freeze()
        return quantized
    case let embedding as Embedding:
        let (embeddingCount, dimensions) = embedding.shape
        let arrays = PlaceholderQuantizedArrays(
            rows: embeddingCount, columns: dimensions, groupSize: groupSize, bits: bits)
        let quantized = QuantizedEmbedding(
            weight: arrays.weight, scales: arrays.scales, biases: arrays.biases,
            groupSize: groupSize, bits: bits, mode: mode)
        quantized.freeze()
        return quantized
    default:
        return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Zero arrays with the shapes affine `MLX.quantized` produces for a
/// `[rows, columns]` matrix.
struct PlaceholderQuantizedArrays {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray

    init(rows: Int, columns: Int, groupSize: Int, bits: Int) {
        let groupCount = columns / groupSize
        weight = MLXArray.zeros([rows, columns * bits / 32], dtype: .uint32)
        scales = MLXArray.zeros([rows, groupCount], dtype: .bfloat16)
        biases = MLXArray.zeros([rows, groupCount], dtype: .bfloat16)
    }
}
