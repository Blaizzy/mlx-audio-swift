import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Tokenizers

extension Maya2Model {
    static func configuration(from directory: URL) throws -> Maya2Configuration {
        let root = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let raw = try JSONSerialization.jsonObject(with: root) as? [String: Any] ?? [:]
        var c = try JSONDecoder().decode(Maya2Configuration.self, from: root)
        if raw["backbone_config"] == nil {
            let file = directory.appendingPathComponent("backbone/config.json")
            var config = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] ?? [:]
            if let rope = config["rope_parameters"] as? [String: Any], let theta = rope["rope_theta"] {
                config["rope_theta"] = theta
            }
            c.backbone = try JSONDecoder().decode(Qwen3Configuration.self, from: JSONSerialization.data(withJSONObject: config))
            let sidecar = directory.appendingPathComponent("maya3_config.json")
            if FileManager.default.fileExists(atPath: sidecar.path),
               let data = try JSONSerialization.jsonObject(with: Data(contentsOf: sidecar)) as? [String: Any],
               let model = data["model"] as? [String: Any] {
                let anchor = model["speaker_anchor"] as? [String: Any] ?? [:]
                c.speakerAnchorEnabled = anchor["enabled"] as? Bool ?? false
                c.speakerAnchorHidden = anchor["hidden"] as? Int ?? 512
                c.speakerDim = anchor["input_dim"] as? Int ?? 192
            }
        }
        try c.validate()
        return c
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (raw, value) in weights {
            let key = raw.hasPrefix("model.") ? String(raw.dropFirst(6)) : raw
            if key == "backbone.lm.lm_head.weight", configuration.backbone.tieWordEmbeddings { continue }
            result[key] = key.hasPrefix("depth.") && value.dtype.isFloatingPoint ? value.asType(.float16) : value
        }
        return result
    }

    public static func fromPretrained(_ repository: String, cache: HubCache = .default) async throws -> Maya2Model {
        guard let id = Repo.ID(rawValue: repository) else { throw TTSModelError.invalidRepositoryID(repository) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: id, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["backbone/*.json", "backbone/*.model", "voices/*.json"], cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }

    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default) async throws -> Maya2Model {
        let config = try configuration(from: directory)
        let model = Maya2Model(config)
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            struct Index: Decodable { let weight_map: [String: String] }
            let names = Set(try JSONDecoder().decode(Index.self, from: Data(contentsOf: index)).weight_map.values)
            guard names.allSatisfy({ $0 == URL(fileURLWithPath: $0).lastPathComponent && $0.hasSuffix(".safetensors") }) else {
                throw AudioGenerationError.invalidInput("Invalid Maya-2 shard index")
            }
            files = names.sorted().map { directory.appendingPathComponent($0) }
        } else { files = [directory.appendingPathComponent("model.safetensors")] }
        var weights: [String: MLXArray] = [:]
        for file in files {
            for (key, value) in try loadArrays(url: file) {
                guard weights[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate Maya-2 tensor: \(key)") }
                weights[key] = value
            }
        }
        weights = model.sanitize(weights: weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                guard path.hasPrefix("backbone.lm."), weights["\(path).scales"] != nil else { return nil }
                return q.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false); eval(model)
        model._tokenizer = try await AutoTokenizer.from(modelFolder: directory.appendingPathComponent("backbone"))
        for lang in languages {
            guard let id = model._tokenizer?.convertTokenToId("<|lang:\(lang)|>"), id >= 0, id < config.backbone.vocabularySize else {
                throw AudioGenerationError.invalidInput("Missing Maya-2 language control token: \(lang)")
            }
        }
        model._cache = cache
        model._directory = directory
        let localCodec = directory.appendingPathComponent("mimi")
        if FileManager.default.fileExists(atPath: localCodec.path) {
            model._codec = try Mimi.fromModelDirectory(localCodec, filename: config.codecFilename)
        } else {
            model._codec = try await Mimi.fromPretrained(repoId: config.codecRepo, filename: config.codecFilename,
                cache: cache, progressHandler: { _ in })
        }
        return model
    }
}
