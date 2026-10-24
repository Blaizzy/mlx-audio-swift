import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

extension MossMusicModel {
    static func canonicalWeightPath(_ path: String) -> String {
        let components = path.split(separator: ".")
        if components.count >= 5, components[0] == "audio_encoder", components[1] == "layers",
            Int(components[2]) != nil,
            ["q_proj", "k_proj", "v_proj", "out_proj"].contains(String(components[3]))
        {
            return components.prefix(3).joined(separator: ".") + ".self_attn."
                + components.dropFirst(3).joined(separator: ".")
        }
        return path
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, var value) in weights {
            if key == "audio_encoder.embed_positions.inv_timescales"
                || key.hasSuffix("rotary_emb.inv_freq")
            {
                continue
            }
            if key.hasPrefix("audio_encoder.conv"), key.hasSuffix(".weight"), value.ndim == 4,
                value.dim(1) != 3
            {
                value = value.transposed(0, 2, 3, 1)
            }
            result[canonicalWeightPath(key)] = value
        }
        return result
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> MossMusicModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> MossMusicModel {
        var config = try JSONDecoder().decode(
            MossMusicConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        if var q = config.perLayerQuantization {
            q.perLayerQuantization = Dictionary(
                uniqueKeysWithValues: q.perLayerQuantization.map {
                    (canonicalWeightPath($0.key), $0.value)
                })
            config.perLayerQuantization = q
        }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            struct WeightIndex: Decodable { let weight_map: [String: String] }
            let names = Set(
                try JSONDecoder().decode(WeightIndex.self, from: Data(contentsOf: index)).weight_map
                    .values
            ).sorted()
            guard
                names.allSatisfy({
                    !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors")
                })
            else { throw STTError.modelNotInitialized("Invalid MOSS-Music weight index") }
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
            throw STTError.modelNotInitialized("No MOSS-Music safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights)
        let model = MossMusicModel(config)
        if let q = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights[path + ".scales"] == nil ? nil : q.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await loadMossMusicTokenizer(from: directory)
        model.train(false)
        eval(model)
        return model
    }
}
