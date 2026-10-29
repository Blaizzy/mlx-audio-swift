import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

extension Zonos2Model {
    static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = weights
        for (key, value) in weights
        where key.hasSuffix(".attention.wkv.weight") || key.hasSuffix(".feed_forward.w_in.weight") {
            if value.ndim == 3 {
                guard value.dim(0) == 2 else {
                    throw Zonos2Error.invalidConfiguration("Invalid chunked projection \(key)")
                }
                result[key] = value.reshaped(value.dim(0) * value.dim(1), value.dim(2))
            }
        }
        return result
    }
    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> Zonos2Model
    {
        guard let id = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }
        let folder = try await ModelUtils.resolveOrDownloadModel(
            repoID: id, requiredExtension: ".safetensors",
            additionalMatchingPatterns: [
                "speaker_encoder/*.json", "speaker_encoder/*.safetensors", "dac/*.json",
                "dac/*.safetensors",
            ], cache: cache)
        return try await fromModelDirectory(folder, cache: cache)
    }
    public static func fromModelDirectory(_ folder: URL, cache: HubCache = .default) async throws
        -> Zonos2Model
    {
        let data = try Data(contentsOf: folder.appendingPathComponent("config.json"))
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Zonos2Error.invalidConfiguration("config.json must be an object")
        }
        if let nested = json.removeValue(forKey: "model") as? [String: Any] {
            json.merge(nested) { _, rhs in rhs }
        }
        let config = try JSONDecoder().decode(
            Zonos2Configuration.self, from: JSONSerialization.data(withJSONObject: json))
        let model = try Zonos2Model(config: config)
        let indexURL = folder.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            struct Index: Decodable { let weight_map: [String: String] }
            let names = Set(
                try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL)).weight_map.values
            ).sorted()
            guard !names.isEmpty,
                names.allSatisfy({ !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors") }
                )
            else {
                throw Zonos2Error.invalidConfiguration("Invalid weight index")
            }
            files = names.map { folder.appendingPathComponent($0) }
        } else {
            files = try FileManager.default.contentsOfDirectory(
                at: folder, includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "safetensors" }.sorted {
                $0.lastPathComponent < $1.lastPathComponent
            }
        }
        guard !files.isEmpty else {
            throw Zonos2Error.missingComponent("ZONOS2 safetensors are missing")
        }
        var weights: [String: MLXArray] = [:]
        for file in files {
            let shard = try MLX.loadArrays(url: file)
            guard Set(weights.keys).isDisjoint(with: shard.keys) else {
                throw Zonos2Error.invalidConfiguration("Duplicate tensors in checkpoint shards")
            }
            weights.merge(shard) { _, new in new }
        }
        weights = try sanitize(weights)
        if let q = (json["quantization"] ?? json["quantization_config"]) as? [String: Any] {
            var invalidMode: String?
            quantize(model: model) { path, _ in
                guard weights[path + ".scales"] != nil else { return nil }
                let specific = q[path] as? [String: Any] ?? [:]
                let mode = specific["mode"] as? String ?? q["mode"] as? String ?? "affine"
                if mode != "affine" {
                    invalidMode = mode
                    return nil
                }
                return (
                    specific["group_size"] as? Int ?? q["group_size"] as? Int ?? 64,
                    specific["bits"] as? Int ?? q["bits"] as? Int ?? 4, .affine
                )
            }
            if let invalidMode {
                throw Zonos2Error.invalidConfiguration(
                    "Unsupported ZONOS2 quantization mode \(invalidMode)")
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.directory = folder
        model.hubCache = cache
        model.train(false)
        // Lazy mapped shard tensors replace initial parameters before materializing the model.
        eval(model.parameters())
        return model
    }
}
