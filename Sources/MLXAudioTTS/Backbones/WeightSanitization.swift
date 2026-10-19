// Copyright © 2024 Apple Inc.

import MLX

/// Removes checkpoint tensors owned by an `lm_head` module when the model uses its token
/// embedding as the output projection instead.
///
/// Quantized linear layers carry parameters in addition to `weight` (for example `scales`
/// and `biases`). Filtering by the module path keeps those parameters from being loaded into
/// the absent head. Matching a complete path component also supports weights that have already
/// been namespaced by a wrapper model without affecting similarly named modules.
package func filterLMHeadWeights(
    from weights: [String: MLXArray], tiedWordEmbeddings: Bool
) -> [String: MLXArray] {
    guard tiedWordEmbeddings else { return weights }

    return weights.filter { key, _ in
        !key.split(separator: ".").contains("lm_head")
    }
}
