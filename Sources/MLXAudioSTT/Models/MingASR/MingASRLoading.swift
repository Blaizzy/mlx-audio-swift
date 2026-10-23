import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

extension MingASRModel {
    static func canonicalWeightPath(_ key: String) -> String {
        let components = key.split(separator: ".").map(String.init)
        if components.count >= 2, components[0] == "linear_proj_audio",
            let index = Int(components[1])
        {
            let module = index % 2 == 0 ? "convolutions.\(index / 2)" : "projection"
            return (["linear_proj_audio", module] + components.dropFirst(2)).joined(separator: ".")
        }
        guard key.hasPrefix("model.audio.") else { return key }
        var path = key.replacingOccurrences(
            of: "model.audio.positional_embedding", with: "model.audio.embed_positions.weight"
        )
        .replacingOccurrences(of: "model.audio.blocks.", with: "model.audio.layers.")
        .replacingOccurrences(of: "model.audio.ln_post.", with: "model.audio.layer_norm.")
        for (from, to) in [
            (".attn.query.", ".self_attn.q_proj."), (".attn.key.", ".self_attn.k_proj."),
            (".attn.value.", ".self_attn.v_proj."), (".attn.out.", ".self_attn.out_proj."),
            (".attn_ln.", ".self_attn_layer_norm."), (".mlp_ln.", ".final_layer_norm."),
            (".mlp.0.", ".fc1."), (".mlp.2.", ".fc2."),
        ] {
            path = path.replacingOccurrences(of: from, with: to)
        }
        return path
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let shapes = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        var result: [String: MLXArray] = [:]
        for (key, var value) in weights {
            if key.hasPrefix("linear_proj.") || key.hasPrefix("model.visual.")
                || key.hasSuffix("rotary_emb.inv_freq")
            {
                continue
            }
            let parts = key.split(separator: ".")
            if parts.count > 2, parts[0] == "model", parts[1] == "layers",
                let layer = Int(parts[2]), layer >= config.textConfig.numHiddenLayers
            {
                continue
            }
            let path = Self.canonicalWeightPath(key)
            if key.hasSuffix("_conv1d.weight"), value.ndim == 3, value.dim(-1) != 1 {
                value = value.transposed(0, 2, 1)
            } else if key.hasPrefix("model.audio.conv") || key.hasPrefix("linear_proj_audio."),
                key.hasSuffix(".weight"), value.ndim == 3, value.shape != shapes[path]
            {
                value = value.transposed(0, 2, 1)
            }
            result[path] = value
        }
        let c = config.textConfig
        for layer in 0 ..< c.numHiddenLayers where layer >= c.firstKDenseReplace {
            let stem = "model.layers.\(layer).mlp."
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard result[stem + "experts.0.\(projection).\(suffix)"] != nil else {
                        continue
                    }
                    let keys = (0 ..< c.numExperts).map {
                        stem + "experts.\($0).\(projection).\(suffix)"
                    }
                    guard keys.allSatisfy({ result[$0] != nil }) else {
                        throw STTError.modelNotInitialized(
                            "Incomplete Ming routed experts for \(stem)\(projection)")
                    }
                    result[stem + "switch_mlp.\(projection).\(suffix)"] = stacked(
                        keys.map { result.removeValue(forKey: $0)! })
                }
            }
        }
        if c.tieWordEmbeddings { result = result.filter { !$0.key.hasPrefix("lm_head.") } }
        return result
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> MingASRModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> MingASRModel {
        var config = try JSONDecoder().decode(
            MingASRConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        if var q = config.perLayerQuantization {
            q.perLayerQuantization = Dictionary(
                uniqueKeysWithValues: q.perLayerQuantization.map {
                    (canonicalWeightPath($0.key), $0.value)
                })
            config.perLayerQuantization = q
        }
        let model = MingASRModel(config)
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            struct Index: Decodable { let weight_map: [String: String] }
            let names = Set(
                try JSONDecoder().decode(Index.self, from: Data(contentsOf: index)).weight_map
                    .values
            ).sorted()
            guard
                names.allSatisfy({
                    !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors")
                })
            else { throw STTError.modelNotInitialized("Invalid Ming weight index") }
            files = names.map { directory.appendingPathComponent($0) }
        } else {
            files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil
            )
            .filter { $0.pathExtension == "safetensors" }.sorted {
                $0.lastPathComponent < $1.lastPathComponent
            }
        }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No Ming safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = try model.sanitize(weights: weights)
        if let q = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights[path + ".scales"] == nil ? nil : q.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await loadSTTTokenizer(
            from: directory, tokenizerClass: "PreTrainedTokenizerFast")
        var markers: (String, String)?
        for (start, pad, end) in [
            ("<audio>", "<audioPatch>", "</audio>"),
            ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"),
        ] {
            if model.tokenizer?.convertTokenToId(pad) == config.audioTokenId {
                markers = (start, end)
                break
            }
        }
        guard let markers else {
            throw STTError.modelNotInitialized(
                "Ming tokenizer audio token does not match the checkpoint")
        }
        model.audioMarkers = markers
        model.train(false)
        eval(model)
        return model
    }
}
