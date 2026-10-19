// Copyright © 2024, 2025, 2026 Apple Inc.

import MLXNN

public struct LoRAModelMetadata: Sendable, Equatable {
    /// Number of model layers that support LoRA adapters.
    public let layerCount: Int

    /// Default module paths, relative to each LoRA layer, that receive adapters.
    public let defaultKeys: [String]

    public init(layerCount: Int, defaultKeys: [String]) {
        self.layerCount = layerCount
        self.defaultKeys = defaultKeys
    }
}

public protocol LoRAModel {

    /// Return the layers to apply LoRA adapters to.
    ///
    /// Typically, this includes all transformer layers.
    /// Must be defined explicitly since we can't unify it across all models.
    var loraLayers: [Module] { get }

    /// Default layer keys to apply LoRA adapters to.
    ///
    /// Used when not specified in `adapter_config.json`.
    /// Otherwise, keys from the config are applied.
    var loraDefaultKeys: [String] { get }
}

extension LoRAModel {

    /// Metadata for configuring LoRA without inspecting checkpoint weight names.
    public var loraMetadata: LoRAModelMetadata {
        LoRAModelMetadata(
            layerCount: loraLayers.count,
            defaultKeys: loraDefaultKeys.sorted()
        )
    }

    /// By default we apply LoRA to all Linear layers.
    /// This is aligned with `mlx-lm` Python logic.
    public var loraDefaultKeys: [String] {
        let namedModules = loraLayers.flatMap { $0.namedModules() }
        let linearKeys = namedModules.compactMap { key, module in
            if module is Linear {
                return key
            } else {
                return nil
            }
        }
        let unique = Set(linearKeys)
        return Array(unique)
    }
}
