//
//  TTSUtils.swift
//  MLXAudioTTS
//
//  Shared utility functions for TTS models.
//

import Foundation
import MLX
import MLXNN

// MARK: - Activation Functions

/// Get activation function by name.
///
/// - Parameter name: Activation name ("silu", "swish", "gelu", "relu")
/// - Returns: Activation function
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
