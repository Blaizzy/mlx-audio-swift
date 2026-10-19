// Copyright © 2026 Apple Inc.

import MLX

public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}
