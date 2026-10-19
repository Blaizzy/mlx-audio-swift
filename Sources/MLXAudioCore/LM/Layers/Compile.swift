// Adapted from mlx-vlm-swift 28194b61: Layers/CompiledTrace.swift.
// See LICENSES/mlx-vlm-swift.txt.

// Copyright © 2026 Apple Inc.

import MLX

public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}
