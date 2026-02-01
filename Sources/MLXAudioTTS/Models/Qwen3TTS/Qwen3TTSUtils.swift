//
//  Qwen3TTSUtils.swift
//  MLXAudio
//
//  Utility functions for Qwen3-TTS model.
//  Ported from mlx_audio/tts/models/qwen3_tts/talker.py
//

import Foundation
import MLX
import MLXNN

// MARK: - RMS Normalization

/// RMS Layer Normalization.
/// Normalizes the input by its root mean square value.
///
/// - Parameters:
///   - x: Input tensor of any shape
///   - weight: Learnable scale parameter
///   - eps: Small constant for numerical stability (default: 1e-6)
/// - Returns: Normalized tensor with same shape as input
public func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
    // Cast to float32 for numerical stability
    let xFloat = x.asType(.float32)
    let variance = mean(pow(xFloat, 2), axis: -1, keepDims: true)
    let xNormed = xFloat * rsqrt(variance + eps)
    return (weight * xNormed).asType(x.dtype)
}

// MARK: - Rotary Position Embedding Utilities

/// Rotates half the hidden dimensions of the input.
/// Used in rotary position embeddings.
///
/// - Parameter x: Input tensor with shape [..., dim]
/// - Returns: Tensor with second half negated and swapped with first half
public func rotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.shape[x.ndim - 1] / 2
    let x1 = x[.ellipsis, ..<halfDim]
    let x2 = x[.ellipsis, halfDim...]
    return concatenated([-x2, x1], axis: -1)
}

/// Applies Rotary Position Embedding to query and key tensors.
///
/// - Parameters:
///   - q: Query tensor [batch, num_heads, seq_len, head_dim]
///   - k: Key tensor [batch, num_heads, seq_len, head_dim]
///   - cos: Cosine embeddings [batch, seq_len, head_dim]
///   - sin: Sine embeddings [batch, seq_len, head_dim]
/// - Returns: Tuple of (rotated query, rotated key)
public func applyRotaryPosEmb(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    // cos, sin: [batch, seq_len, head_dim]
    // Expand for heads dimension: [batch, 1, seq_len, head_dim]
    let cosExpanded = expandedDimensions(cos, axis: 1)
    let sinExpanded = expandedDimensions(sin, axis: 1)

    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)
    return (qEmbed, kEmbed)
}

/// Applies Multimodal RoPE to query and key tensors.
///
/// The interleaved MRoPE combination is done in TalkerRotaryEmbedding,
/// so here we just apply the combined cos/sin.
///
/// - Parameters:
///   - q: Query tensor [batch, num_heads, seq_len, head_dim]
///   - k: Key tensor [batch, num_heads, seq_len, head_dim]
///   - cos: Cosine embeddings [batch, seq_len, head_dim]
///   - sin: Sine embeddings [batch, seq_len, head_dim]
///   - unsqueezeDim: Dimension to unsqueeze for broadcasting (default: 1)
/// - Returns: Tuple of (rotated query, rotated key)
public func applyMultimodalRotaryPosEmb(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    unsqueezeDim: Int = 1
) -> (MLXArray, MLXArray) {
    let cosExpanded = expandedDimensions(cos, axis: unsqueezeDim)
    let sinExpanded = expandedDimensions(sin, axis: unsqueezeDim)

    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)

    return (qEmbed, kEmbed)
}

// MARK: - Activation Functions

// Note: silu, gelu, relu are provided by MLXNN
// We provide a helper function to get activation by name

/// Get activation function by name
public func getActivation(_ name: String) -> (MLXArray) -> MLXArray {
    switch name.lowercased() {
    case "silu", "swish":
        return MLXNN.silu
    case "gelu":
        return MLXNN.gelu
    case "relu":
        return MLXNN.relu
    default:
        fatalError("Unknown activation function: \(name)")
    }
}

// MARK: - Padding Utilities

/// Reflect pad along the sequence dimension (axis 1 for NLC format).
///
/// - Parameters:
///   - x: Input tensor [batch, seq_len, channels] (NLC format)
///   - pad: Number of elements to pad on each side
/// - Returns: Padded tensor [batch, seq_len + 2*pad, channels]
public func reflectPad1d(_ x: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return x }

    let seqLen = x.shape[1]

    // Handle edge case where pad >= seqLen
    let actualPad = min(pad, seqLen - 1)

    // For NLC format [batch, seq, channels], we need to reflect along axis 1
    // Left reflection: indices 1 to actualPad (reversed)
    // We extract [1:actualPad+1] and reverse using negative stride
    var leftSlices: [MLXArray] = []
    for i in (1...actualPad).reversed() {
        leftSlices.append(x[0..., i...i, 0...])
    }
    let leftReflect = concatenated(leftSlices, axis: 1)

    // Right reflection: indices seqLen-actualPad-1 to seqLen-2 (reversed)
    var rightSlices: [MLXArray] = []
    for i in ((seqLen - actualPad - 1)..<(seqLen - 1)).reversed() {
        rightSlices.append(x[0..., i...i, 0...])
    }
    let rightReflect = concatenated(rightSlices, axis: 1)

    return concatenated([leftReflect, x, rightReflect], axis: 1)
}

/// Constant pad along the time dimension for NCL format.
///
/// - Parameters:
///   - x: Input tensor [batch, channels, time] (NCL format)
///   - padLeft: Padding on left side
///   - padRight: Padding on right side
///   - value: Pad value (default: 0)
/// - Returns: Padded tensor
public func constantPad1dNCL(_ x: MLXArray, padLeft: Int, padRight: Int, value: Float = 0) -> MLXArray {
    guard padLeft > 0 || padRight > 0 else { return x }

    let batch = x.shape[0]
    let channels = x.shape[1]

    var result = x
    if padLeft > 0 {
        let leftPad = MLXArray.full([batch, channels, padLeft], values: MLXArray(value)).asType(x.dtype)
        result = concatenated([leftPad, result], axis: 2)
    }
    if padRight > 0 {
        let rightPad = MLXArray.full([batch, channels, padRight], values: MLXArray(value)).asType(x.dtype)
        result = concatenated([result, rightPad], axis: 2)
    }
    return result
}

// MARK: - Masking Utilities

/// Create a causal attention mask.
///
/// - Parameters:
///   - seqLen: Sequence length
///   - dtype: Data type for the mask
/// - Returns: Causal mask [1, 1, seqLen, seqLen] where True indicates positions to mask
public func createCausalMask(seqLen: Int, dtype: DType = .float32) -> MLXArray {
    // Create lower triangular mask
    let mask = MLXArray.tri(seqLen, m: seqLen, k: 0, type: Float.self)
    // Convert to attention mask format (1 for keep, -inf for mask)
    let attentionMask = MLX.where(mask .== 0, MLXArray(-Float.infinity), MLXArray(0.0))
    // Add batch and head dimensions
    return attentionMask.reshaped([1, 1, seqLen, seqLen]).asType(dtype)
}

/// Create an attention mask from sequence lengths.
///
/// - Parameters:
///   - lengths: Tensor of sequence lengths [batch]
///   - maxLen: Maximum sequence length
///   - dtype: Data type for the mask
/// - Returns: Mask [batch, maxLen] where True indicates valid positions
public func createLengthMask(lengths: MLXArray, maxLen: Int, dtype: DType = .float32) -> MLXArray {
    let positions = MLXArray(Array(0..<maxLen)).reshaped([1, maxLen])
    let expandedLengths = lengths.reshaped([-1, 1])
    return (positions .< expandedLengths).asType(dtype)
}

// MARK: - Sampling Utilities

/// Apply temperature to logits.
public func applyTemperature(_ logits: MLXArray, temperature: Float) -> MLXArray {
    guard temperature > 0 else {
        fatalError("Temperature must be positive")
    }
    return logits / temperature
}

/// Apply top-k filtering to logits.
public func topKFilter(_ logits: MLXArray, k: Int) -> MLXArray {
    guard k > 0 else { return logits }

    // Get the k-th largest value
    let sorted = sorted(logits, axis: -1)
    let vocabSize = logits.shape[logits.ndim - 1]
    let kthValue = sorted[.ellipsis, vocabSize - k]

    // Mask logits below the threshold
    let mask = logits .>= kthValue
    return MLX.where(mask, logits, MLXArray(-Float.infinity))
}

/// Apply top-p (nucleus) filtering to logits.
/// Uses descending sort (highest probability first) and keeps tokens until cumulative prob exceeds p.
public func topPFilter(_ logits: MLXArray, p: Float) -> MLXArray {
    guard p < 1.0 else { return logits }

    let vocabSize = logits.shape[logits.ndim - 1]

    // Sort logits in descending order (highest first)
    // MLX sort is ascending, so we negate, sort, then negate back
    let sortedLogitsDesc = -sorted(-logits, axis: -1)
    let sortedProbs = softmax(sortedLogitsDesc, axis: -1)
    let cumulativeProbs = cumsum(sortedProbs, axis: -1)

    // Create mask: tokens where cumulative prob > p should be masked out
    let sortedMask = cumulativeProbs .> p
    // Shift right by 1 to keep at least one token (like Python)
    let zeros = MLXArray.zeros([sortedMask.shape[0], 1]).asType(.bool)
    let shiftedMask = concatenated([zeros, sortedMask[0..., 0..<(vocabSize - 1)]], axis: -1)

    // Get threshold: the logit value at the cutoff position
    // Find the first position where cumulative prob exceeds p
    let keepCount = sum(shiftedMask .== false, axis: -1, keepDims: true).asType(.int32)
    // Get the minimum logit value that should be kept (index = keepCount - 1, but clamp to 0)
    let thresholdIdx = maximum(keepCount - 1, MLXArray(Int32(0)))
    let threshold = takeAlong(sortedLogitsDesc, thresholdIdx, axis: -1)

    // Mask logits below threshold
    return MLX.where(logits .>= threshold, logits, MLXArray(-Float.infinity))
}

/// Apply repetition penalty to logits.
/// Note: This is a simplified version that creates a penalty mask.
/// For production use, consider a more efficient implementation.
public func applyRepetitionPenalty(
    _ logits: MLXArray,
    generatedTokens: [Int],
    penalty: Float
) -> MLXArray {
    guard penalty != 1.0 && !generatedTokens.isEmpty else { return logits }

    let vocabSize = logits.shape[logits.ndim - 1]
    let uniqueTokens = Array(Set(generatedTokens))

    // Create a mask for tokens that need penalty
    var maskArray = [Float](repeating: 1.0, count: vocabSize)
    for token in uniqueTokens {
        if token < vocabSize {
            maskArray[token] = 0.0
        }
    }
    let mask = MLXArray(maskArray).reshaped([1, vocabSize])

    // Apply penalty: divide positive logits, multiply negative logits
    let penaltyFactor = MLXArray(penalty)
    let penalizedPositive = logits / penaltyFactor
    let penalizedNegative = logits * penaltyFactor

    let penalizedLogits = MLX.where(logits .> 0, penalizedPositive, penalizedNegative)

    // Blend original and penalized based on mask
    return MLX.where(mask .== 1, logits, penalizedLogits)
}

/// Sample from logits using multinomial sampling.
public func sampleFromLogits(
    _ logits: MLXArray,
    temperature: Float = 1.0,
    topK: Int = 0,
    topP: Float = 1.0
) -> MLXArray {
    var processedLogits = logits

    // Apply temperature
    if temperature != 1.0 {
        processedLogits = applyTemperature(processedLogits, temperature: temperature)
    }

    // Apply top-k
    if topK > 0 {
        processedLogits = topKFilter(processedLogits, k: topK)
    }

    // Apply top-p
    if topP < 1.0 {
        processedLogits = topPFilter(processedLogits, p: topP)
    }

    // Sample (categorical expects unnormalized logits)
    return categorical(processedLogits)
}

// MARK: - Weight Loading Utilities

/// Extension providing path-based weight loading for MLX modules.
///
/// This approach bypasses problematic nested structure conversion by
/// directly finding each parameter in the module tree and setting it.
extension Module {
    /// Set weights by traversing the module tree using dot-separated paths.
    ///
    /// - Parameter weights: Dictionary mapping dot-separated paths to weight arrays
    public func setWeightsByPath(_ weights: [String: MLXArray]) {
        var loadedCount = 0
        var failedPaths: [String] = []

        for (path, value) in weights {
            let parts = path.split(separator: ".").map(String.init)
            let success = setWeightRecursive(parts: parts, value: value, in: self, fullPath: path)
            if success {
                loadedCount += 1
            } else {
                failedPaths.append(path)
            }
        }

        // Check for critical weight loading failures
        if !failedPaths.isEmpty {
            let totalWeights = loadedCount + failedPaths.count
            let failureRate = Float(failedPaths.count) / Float(totalWeights)
            if failureRate > 0.1 {
                // More than 10% of weights failed - this is serious
                let message = failedPaths.count < 20
                    ? "Failed to load weights: \(failedPaths.joined(separator: ", "))"
                    : "Failed to load \(failedPaths.count) weights (first 5: \(failedPaths.prefix(5).joined(separator: ", ")))"
                print("Warning: \(message)")
            }
        }
    }

    /// Recursively set a weight by navigating through the module tree.
    ///
    /// - Parameters:
    ///   - parts: Remaining path components
    ///   - value: Weight array to set
    ///   - module: Current module being traversed
    ///   - fullPath: Original full path for debugging
    /// - Returns: true if weight was successfully set
    @discardableResult
    private func setWeightRecursive(parts: [String], value: MLXArray, in module: Module, fullPath: String = "") -> Bool {
        guard !parts.isEmpty else { return false }

        let key = parts[0]
        let remaining = Array(parts.dropFirst())
        let items = module.items()

        guard let item = items[key] else {
            return false
        }

        if remaining.isEmpty {
            // Leaf node - set the parameter
            if case .value(.parameters(let param)) = item {
                param._updateInternal(value)
                return true
            }
            return false
        }

        // Navigate deeper based on item type
        switch item {
        case .value(.module(let childModule)):
            return setWeightRecursive(parts: remaining, value: value, in: childModule, fullPath: fullPath)

        case .array(let array):
            guard let index = Int(remaining[0]), index < array.count else {
                return false
            }
            let subRemaining = Array(remaining.dropFirst())
            let arrayItem = array[index]

            switch arrayItem {
            case .value(.module(let childModule)):
                if subRemaining.isEmpty {
                    return false
                }
                return setWeightRecursive(parts: subRemaining, value: value, in: childModule, fullPath: fullPath)

            case .value(.parameters(let param)):
                if subRemaining.isEmpty {
                    param._updateInternal(value)
                    return true
                }
                return false

            default:
                return navigateNestedItem(item: arrayItem, parts: subRemaining, value: value, fullPath: fullPath)
            }

        case .dictionary(let dict):
            let dictKey = remaining[0]
            guard let dictItem = dict[dictKey] else { return false }
            let subRemaining = Array(remaining.dropFirst())

            switch dictItem {
            case .value(.module(let childModule)):
                if subRemaining.isEmpty {
                    return false
                }
                return setWeightRecursive(parts: subRemaining, value: value, in: childModule, fullPath: fullPath)

            case .value(.parameters(let param)):
                if subRemaining.isEmpty {
                    param._updateInternal(value)
                    return true
                }
                return false

            default:
                return navigateNestedItem(item: dictItem, parts: subRemaining, value: value, fullPath: fullPath)
            }

        default:
            return false
        }
    }

    /// Navigate through nested ModuleItem structures (arrays/dicts within arrays/dicts).
    @discardableResult
    private func navigateNestedItem(item: ModuleItem, parts: [String], value: MLXArray, fullPath: String = "") -> Bool {
        guard !parts.isEmpty else { return false }

        switch item {
        case .value(.module(let module)):
            return setWeightRecursive(parts: parts, value: value, in: module, fullPath: fullPath)

        case .value(.parameters(let param)):
            if parts.isEmpty {
                param._updateInternal(value)
                return true
            }
            return false

        case .array(let array):
            guard let index = Int(parts[0]), index < array.count else {
                return false
            }
            return navigateNestedItem(item: array[index], parts: Array(parts.dropFirst()), value: value, fullPath: fullPath)

        case .dictionary(let dict):
            guard let subItem = dict[parts[0]] else {
                return false
            }
            return navigateNestedItem(item: subItem, parts: Array(parts.dropFirst()), value: value, fullPath: fullPath)

        default:
            return false
        }
    }
}

