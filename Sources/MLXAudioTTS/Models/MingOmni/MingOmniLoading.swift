import Foundation
import MLX
import MLXAudioCore

extension MingOmniModel {
    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = mingSanitizeSpeechWeights(weights).filter { key, _ in
            ["model.", "audio.", "flowloss.", "linear_proj_audio.", "spk_head.", "stop_head."].contains(where: key.hasPrefix)
                && !key.contains(".audio_gate.") && !key.contains(".image_gate.") && !key.hasSuffix("rotary_emb.inv_freq")
        }
        let c = configuration.text
        for layer in c.firstKDenseReplace..<c.numHiddenLayers {
            let stem = "model.model.layers.\(layer).mlp."
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let names = (0..<c.numExperts).map { stem + "experts.\($0).\(projection).\(suffix)" }
                    guard names.contains(where: { result[$0] != nil }) else { continue }
                    guard names.allSatisfy({ result[$0] != nil }) else {
                        throw AudioGenerationError.invalidInput("Incomplete Ming Omni experts at \(stem)\(projection).\(suffix)")
                    }
                    result[stem + "switch_mlp.\(projection).\(suffix)"] = stacked(names.map { result.removeValue(forKey: $0)! })
                }
            }
            for suffix in ["weight", "scales", "biases"] {
                if let value = result.removeValue(forKey: stem + "gate." + suffix) {
                    result[stem + "gate.gate_proj." + suffix] = value
                }
            }
        }
        return result
    }
}
