import Foundation
import MLX
import MLXNN

public enum WeightLoader {

    private static let sharedReplacements: [(String, String)] = [
        (".block.", ".layers."),
        (".k.", ".key_proj."),
        (".o.", ".out_proj."),
        (".q.", ".query_proj."),
        (".v.", ".value_proj."),
        ("shared.", "wte."),
        ("lm_head.", "lm_head.linear."),
        (".layer.0.layer_norm.", ".ln1."),
        (".layer.1.layer_norm.", ".ln2."),
        (".layer.2.layer_norm.", ".ln3."),
        (".final_layer_norm.", ".ln."),
        (
            "layers.0.layer.0.SelfAttention.relative_attention_bias.",
            "relative_attention_bias.embeddings."
        ),
    ]

    private static let encoderReplacements: [(String, String)] = [
        (".layer.0.SelfAttention.", ".attention."),
        (".layer.1.DenseReluDense.", ".dense."),
    ]

    private static let decoderReplacements: [(String, String)] = [
        (".layer.0.SelfAttention.", ".self_attention."),
        (".layer.1.EncDecAttention.", ".cross_attention."),
        (".layer.2.DenseReluDense.", ".dense."),
    ]

    private static let ignoredPatterns: [String] = [
        ".cross_attention.relative_attention_bias."
    ]

    static func sanitizeKey(_ key: String) -> String? {
        var key = key

        for (from, to) in sharedReplacements {
            key = key.replacingOccurrences(of: from, with: to)
        }

        if key.hasPrefix("encoder.") {
            for (from, to) in encoderReplacements {
                key = key.replacingOccurrences(of: from, with: to)
            }
        } else if key.hasPrefix("decoder.") {
            for (from, to) in decoderReplacements {
                key = key.replacingOccurrences(of: from, with: to)
            }
        }

        if ignoredPatterns.contains(where: { key.contains($0) }) { return nil }
        return key
    }

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        result.reserveCapacity(weights.count)
        for (key, value) in weights {
            if let sanitized = sanitizeKey(key) {
                result[sanitized] = value
            }
        }
        return result
    }

    public static func load(from directory: URL) throws -> T5ForConditionalGeneration {
        let config = try T5Config.load(from: directory)
        let model = T5ForConditionalGeneration(config: config)

        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let rawWeights = try loadArrays(url: weightsURL)
        let weights = sanitize(rawWeights)

        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: .noUnusedKeys)

        model.freeze()
        model.train(false)

        return model
    }
}
