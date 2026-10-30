import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore

extension VoxtralTTSModel {
    static func canonicalPath(_ raw: String) -> String {
        var key = raw
        for prefix in ["language_model.model.model.", "language_model.model."] {
            if key.hasPrefix(prefix) { key = "backbone." + key.dropFirst(prefix.count); break }
        }
        if key.hasPrefix("layers.") || key == "norm.weight" { key = "backbone." + key }
        for original in ["mm_audio_embeddings.tok_embeddings.weight", "tok_embeddings.weight"] {
            if key == original { key = "backbone.embed_tokens.weight" }
        }
        for prefix in ["mm_audio_embeddings.audio_codebook_embeddings.embeddings.", "audio_codebook_embeddings.embeddings."] {
            if key.hasPrefix(prefix) { key = "audio_embeddings." + key.dropFirst(prefix.count) }
        }
        if key.hasPrefix("backbone.layers.") {
            for (source,target) in [("self_attn.","attention."), ("mlp.","feed_forward."),
                ("attention.q_proj.","attention.wq."), ("attention.k_proj.","attention.wk."),
                ("attention.v_proj.","attention.wv."), ("attention.o_proj.","attention.wo."),
                ("feed_forward.gate_proj.","feed_forward.w1."), ("feed_forward.down_proj.","feed_forward.w2."),
                ("feed_forward.up_proj.","feed_forward.w3."), ("input_layernorm.","attention_norm."),
                ("post_attention_layernorm.","ffn_norm.")] {
                key = key.replacingOccurrences(of: source, with: target)
            }
        }
        key = key.replacingOccurrences(of: "audio_tokenizer.quantizer.semantic_codebook.", with: "audio_tokenizer.")
        if key.hasPrefix("audio_tokenizer.decoder_blocks.") {
            let parts = key.components(separatedBy: ".")
            if parts.count >= 4, let index = Int(parts[2]) {
                let suffix = parts.dropFirst(3).joined(separator: ".")
                key = "audio_tokenizer.stages.\(index / 2)." + (index % 2 == 0 ? suffix : "transformer." + suffix)
            }
        }
        key = key.replacingOccurrences(of: "audio_tokenizer.output_proj.conv.", with: "audio_tokenizer.output_proj.")
        return key
    }
    func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (raw,value) in weights {
            if raw == "output.weight" || raw.hasPrefix("language_model.model.lm_head.") { continue }
            if raw.hasSuffix("parametrizations.weight.original1") { continue }
            var key = Self.canonicalPath(raw), tensor = value
            if raw.hasSuffix("parametrizations.weight.original0") {
                let directionKey = String(raw.dropLast("original0".count)) + "original1"
                guard let direction = weights[directionKey], direction.ndim == 3, value.shape == [direction.dim(0),1,1] else {
                    throw AudioGenerationError.invalidInput("Incomplete Voxtral weight normalization: \(raw)")
                }
                tensor = value * direction / sqrt((direction * direction).sum(axes: [1,2], keepDims: true) + 1e-12)
                key = String(key.dropLast("parametrizations.weight.original0".count)) + "weight"
            }
            if let existing = result[key] {
                guard existing.shape == tensor.shape, existing.dtype == tensor.dtype,
                      all(existing .== tensor).item(Bool.self) else {
                    throw AudioGenerationError.invalidInput("Conflicting Voxtral checkpoint aliases: \(key)")
                }
            } else { result[key] = tensor }
        }
        return result
    }
    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> VoxtralTTSModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["voice_embedding/*.safetensors"], cache: cache)
        return try await fromModelDirectory(directory)
    }
    public static func fromModelDirectory(_ directory: URL) async throws -> VoxtralTTSModel {
        let c = try JSONDecoder().decode(VoxtralTTSConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try c.validate()
        let processor = try VoxtralTTSTextProcessor(directory: directory)
        guard processor.special["<s>"] == c.bosID, processor.special["[AUDIO]"] == c.audioID,
              processor.special["[BEGIN_AUDIO]"] == c.beginAudioID else {
            throw AudioGenerationError.invalidInput("Voxtral tokenizer and model control token IDs disagree")
        }
        let model = VoxtralTTSModel(c)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            for (key,value) in try MLX.loadArrays(url: file) {
                guard weights[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate Voxtral tensor: \(key)") }
                weights[key] = value
            }
        }
        guard !weights.isEmpty else { throw AudioGenerationError.modelNotInitialized("No Voxtral safetensors weights at \(directory.path)") }
        let sanitized = try model.sanitize(weights)
        if let q = c.quantization {
            quantize(model: model) { path, _ in
                guard sanitized["\(path).scales"] != nil else { return nil }
                return q.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        model.textProcessor = processor
        let voicesDirectory = directory.appendingPathComponent("voice_embedding")
        for name in processor.voiceLengths.keys {
            let file = voicesDirectory.appendingPathComponent(name + ".safetensors")
            if FileManager.default.fileExists(atPath: file.path) { model.registerVoice(name, file: file) }
        }
        guard !model.availableVoices.isEmpty else { throw AudioGenerationError.modelNotInitialized("Voxtral requires voice_embedding/*.safetensors") }
        return model
    }
}
