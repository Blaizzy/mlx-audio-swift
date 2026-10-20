import Foundation
import MLX

// PyTorch Sequential indices used by the official Gradium checkpoints.
func pocketTTSSEANetNames(prefix: String, config: PocketTTSSeanetConfig, encoder: Bool) -> [String: String] {
    var names = ["\(prefix).model.0": "\(prefix).init_conv1d"]
    var index = 1
    for layer in config.ratios.indices {
        let destination = "\(prefix).layers.\(layer)"
        if !encoder {
            names["\(prefix).model.\(index + 1)"] = "\(destination).upsample"
            index += 2
        }
        for residual in 0..<config.nResidualLayers {
            for block in 0..<2 {
                names["\(prefix).model.\(index).block.\(2 * block + 1)"] = "\(destination).residuals.\(residual).block.\(block)"
            }
            index += 1
        }
        if encoder {
            names["\(prefix).model.\(index + 1)"] = "\(destination).downsample"
            index += 2
        }
    }
    names["\(prefix).model.\(index + 1)"] = "\(prefix).final_conv1d"
    return names
}

extension PocketTTSModel {
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let weights = weights.filter {
            !$0.key.hasSuffix(".learnt_padding") && !["speaker_wavs", "emb", "audio_prompt"].contains($0.key)
        }
        guard weights.keys.contains(where: { $0.contains(".decoder.model.") || $0.contains(".condition_provider.") }) else {
            return weights
        }

        var names = pocketTTSSEANetNames(prefix: "mimi.encoder", config: config.mimi.seanet, encoder: true)
        names.merge(pocketTTSSEANetNames(prefix: "mimi.decoder", config: config.mimi.seanet, encoder: false)) { _, new in new }
        let prefixes = names.sorted { $0.key.count > $1.key.count }
        let targetKeys = Set(parameters().flattened().map(\.0))
        var mapped: [String: MLXArray] = [:]
        for (source, tensor) in weights {
            if [".w_s_t.", ".quantizer.vq.", ".quantizer.logvar_proj."].contains(where: source.contains) { continue }
            var key = source.replacingOccurrences(
                of: "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight", with: "speaker_proj_weight")
            key = key.replacingOccurrences(of: "flow_lm.speaker_proj_weight", with: "speaker_proj_weight")
            key = key.replacingOccurrences(of: "flow_lm.condition_provider.conditioners.transcript_in_segment.", with: "flow_lm.conditioner.")
            for name in ["num_speakers", "duration_delta"] {
                key = key.replacingOccurrences(of: "flow_lm.condition_provider.conditioners.\(name).", with: "flow_lm.\(name).")
            }
            key = key.replacingOccurrences(of: "flow_lm.backbone.", with: "flow_lm.transformer.")
                .replacingOccurrences(of: "flow_lm.flow.", with: "flow_lm.flow_net.")
                .replacingOccurrences(of: "mimi.model.", with: "mimi.")
            for (prefix, destination) in prefixes where key.hasPrefix(prefix + ".") {
                key = destination + key.dropFirst(prefix.count)
                break
            }
            key = key.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
                .replacingOccurrences(
                    of: #"(\.(?:encoder|decoder)_transformer\.transformer\.layers\.\d+)\.(linear[12])\."#,
                    with: "$1.gating.$2.", options: .regularExpression)
            if !targetKeys.contains(key) {
                key = key.replacingOccurrences(
                    of: #"\.(conv|convtr)\.(weight|bias)$"#, with: ".$1.$1.$2", options: .regularExpression)
            }
            var value = tensor
            if value.ndim == 3 && key.hasSuffix(".weight") {
                if key.contains(".convtr.") {
                    value = key.hasPrefix("mimi.upsample.") ? value.transposed(0, 2, 1) : value.transposed(1, 2, 0)
                } else {
                    value = value.transposed(0, 2, 1)
                }
            }
            mapped[key] = value
        }
        return mapped
    }
}
