import Foundation
import MLX
import MLXAudioCore
import MLXAudioCodecs

/// Validate required component structure before constructing lazy inference
/// graphs, so incomplete snapshots report a useful error at load time.
enum Confucius4Checkpoint {
    static func validate(w2v: [String: MLXArray], t2s: [String: MLXArray], s2a: [String: MLXArray], config: Confucius4Config) throws {
        var required = Set<String>()
        func pair(_ p: String) { required.insert(p + ".weight"); required.insert(p + ".bias") }
        func weight(_ p: String) { required.insert(p + ".weight") }
        func check(_ values: [String: MLXArray], _ component: String) throws {
            let missing = required.subtracting(values.keys).sorted()
            guard missing.isEmpty else { throw AudioGenerationError.invalidInput("Incomplete Confucius4 \(component): missing \(missing.prefix(5).joined(separator: ", "))") }
            for (key, value) in values where key.hasSuffix(".scales") {
                let base = String(key.dropLast(7))
                guard let packed = values[base + ".weight"], let biases = values[base + ".biases"],
                      packed.ndim == 2, value.ndim == 2, packed.dtype == .uint32, value.shape == biases.shape,
                      value.shape[0] == packed.shape[0],
                      packed.shape[1] * (32 / config.quantBits) == value.shape[1] * config.quantGroupSize else {
                    throw AudioGenerationError.invalidInput("Invalid Confucius4 quantized tensor \(base)")
                }
            }
            required.removeAll()
        }
        pair("feature_projection.layer_norm"); pair("feature_projection.projection")
        for i in 0..<17 {
            let p = "encoder.layers.\(i)."
            for n in ["ffn1_layer_norm", "self_attn_layer_norm", "ffn2_layer_norm", "final_layer_norm"] { pair(p + n) }
            for n in ["ffn1", "ffn2"] { pair(p + n + ".intermediate_dense"); pair(p + n + ".output_dense") }
            for n in ["linear_q", "linear_k", "linear_v", "linear_out"] { pair(p + "self_attn." + n) }
            weight(p + "self_attn.distance_embedding")
            for n in ["layer_norm", "depthwise_layer_norm"] { pair(p + "conv_module." + n) }
            for n in ["pointwise_conv1", "pointwise_conv2", "depthwise_conv"] { weight(p + "conv_module." + n) }
        }
        try check(w2v, "W2V-BERT")
        guard w2v["feature_projection.projection.weight"]?.shape == [1024, 160] else { throw AudioGenerationError.invalidInput("Unsupported Confucius4 W2V-BERT dimensions") }
        for n in ["semantic_embedding", "semantic_position_embedding.embedding", "text_projector.embed", "text_position_embedding.embedding"] { weight(n) }
        for n in ["final_norm", "semantic_head", "transformer.ln_f", "text_projector.text_projection_fc1", "text_projector.text_projection_fc2"] { pair(n) }
        for i in 0..<24 {
            let p = "transformer.h.\(i)."
            for n in ["ln_1", "ln_2", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"] { pair(p + n) }
        }
        pair("speaker_encoder.blocks.0.conv")
        for i in 1...3 {
            let p = "speaker_encoder.blocks.\(i)."
            for n in ["tdnn1.conv", "tdnn2.conv", "se_block.conv1", "se_block.conv2"] { pair(p + n) }
            for j in 0..<7 { pair(p + "res2net_block.blocks.\(j).conv") }
        }
        for n in ["mfa.conv", "asp.tdnn.conv", "asp.conv", "fc"] { pair("speaker_encoder." + n) }
        try check(t2s, "T2S")
        guard t2s["semantic_embedding.weight"]?.shape == [8194, 1280], t2s["text_projector.embed.weight"]?.shape == [32000, 4096] else {
            throw AudioGenerationError.invalidInput("Unsupported Confucius4 semantic/text dimensions")
        }
        weight("input_embedding.embedding"); pair("input_embedding.out_project"); pair("encoder_proj"); pair("length_regulator.content_in_proj")
        required.insert("prompt_cond")
        for i in [0, 1, 3, 4, 6, 7, 9, 10, 12] { pair("length_regulator.model.\(i)") }
        let p = "decoder.estimator."
        required.insert(p + "freqs_cis")
        for n in ["t_embedder", "t_embedder2"] { pair(p + n + ".time_mlp.0"); pair(p + n + ".time_mlp.2") }
        for n in ["input_embed.mu_projection", "input_embed.proj", "skip_linear", "conv1", "conv2", "res_projection",
                  "final_layer.adaLN_modulation.1", "final_layer.linear", "wavenet.cond_layer.conv"] { pair(p + n) }
        for n in ["transformer_norm"] { weight(p + n + ".norm"); pair(p + n + ".modulation") }
        for i in 0..<13 {
            let q = p + "transformer_blocks.\(i)."
            for n in ["attention_norm", "ffn_norm"] { weight(q + n + ".norm"); pair(q + n + ".modulation") }
            for n in ["attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"] { weight(q + n) }
            if i > 6 { pair(q + "skip_in_linear") }
        }
        for i in 0..<8 { pair(p + "wavenet.in_layers.\(i).conv"); pair(p + "wavenet.res_skip_layers.\(i).conv") }
        try check(s2a, "S2A")
        guard s2a[p + "freqs_cis"]?.shape == [4096, 32, 2], s2a[p + "conv2.weight"]?.shape == [80, 512, 1] else {
            throw AudioGenerationError.invalidInput("Unsupported Confucius4 acoustic dimensions")
        }
    }
    /// Reconstitute weight-norm parameters losslessly from published folded
    /// weights. Filters and SnakeBeta parameters retain the shared vocoder API.
    static func vocoder(_ values: [String: MLXArray], model: BigVGAN) throws -> [String: MLXArray] {
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var mapped: [String: MLXArray] = [:]
        for (key, parameter) in expected {
            var sourceKey = key.replacingOccurrences(of: "activationPost.", with: "activation_post.")
                .replacingOccurrences(of: "convPre.", with: "conv_pre.")
                .replacingOccurrences(of: "convPost.", with: "conv_post.")
            if sourceKey.hasPrefix("ups.") { sourceKey = sourceKey.replacingOccurrences(of: ".conv.", with: ".0.") }
            let value: MLXArray
            if key.hasSuffix("weight_v") || key.hasSuffix("weight_g") {
                let source = String(sourceKey.dropLast(8)) + "weight"
                guard let weight = values[source], weight.ndim == 3 else { throw AudioGenerationError.invalidInput("Missing Confucius4 vocoder \(source)") }
                let w = key.hasPrefix("ups.") ? weight.transposed(1, 2, 0) : weight.transposed(0, 2, 1)
                value = key.hasSuffix("weight_v") ? w : sqrt((w * w).sum(axes: key.hasPrefix("ups.") ? [0, 1] : [1, 2], keepDims: true))
            } else {
                guard let original = values[sourceKey] else { throw AudioGenerationError.invalidInput("Missing Confucius4 vocoder \(key)") }
                value = original.shape == parameter.shape ? original : original.reshaped(parameter.shape)
            }
            guard value.shape == parameter.shape else { throw AudioGenerationError.invalidInput("Invalid Confucius4 vocoder shape for \(key)") }
            mapped[key] = value
        }
        return mapped
    }
}
