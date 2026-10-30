import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

extension VibeVoiceTTSModel {
    func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let shapes = Dictionary(uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        var result: [String: MLXArray] = [:]
        for (raw, value) in weights {
            let key = canonicalVibeVoiceSynthesisPath(raw)
            if key.hasSuffix("rotary_emb.inv_freq") { continue }
            guard result[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate VibeVoice tensor: \(key)") }
            var tensor = value
            if tensor.ndim == 3, tensor.shape != shapes[key] {
                tensor = key.contains("convtr") ? tensor.transposed(1, 2, 0) : tensor.swappedAxes(1, 2)
            }
            result[key] = tensor
        }
        // Safetensors may serialize both aliases of a tied matrix. Accept only an exact alias.
        if configuration.decoder.tieWordEmbeddings, result["lm_head.weight"] != nil {
            for suffix in ["weight", "scales", "biases"] {
                if let head = result["lm_head.\(suffix)"] {
                    guard let embedding = result["language_model.embed_tokens.\(suffix)"],
                          head.shape == embedding.shape, head.dtype == embedding.dtype,
                          all(head .== embedding).item(Bool.self) else {
                        throw AudioGenerationError.invalidInput("VibeVoice tied LM head differs from input embeddings")
                    }
                    result.removeValue(forKey: "lm_head.\(suffix)")
                }
            }
        }
        return result
    }
    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> VibeVoiceTTSModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors", cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }
    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default) async throws -> VibeVoiceTTSModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(VibeVoiceTTSConfiguration.self, from: data)
        try configuration.validate()
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if let q = json?["quantization_config"] as? [String: Any], q["quant_method"] as? String == "bitsandbytes" {
            throw AudioGenerationError.invalidInput("VibeVoice requires original or MLX safetensors; bitsandbytes weights are unsupported")
        }
        let model = VibeVoiceTTSModel(configuration)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            for (key, tensor) in try MLX.loadArrays(url: file) {
                guard weights[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate VibeVoice checkpoint tensor: \(key)") }
                weights[key] = tensor
            }
        }
        let sanitized = try model.sanitize(weights)
        if let q = configuration.quantization {
            quantize(model: model) { path, _ in sanitized["\(path).scales"] == nil ? nil : q.quantization(layer: path)?.asTuple }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        guard isFinite(model.speechScalingFactor).all().item(Bool.self),
              isFinite(model.speechBiasFactor).all().item(Bool.self),
              all(model.speechScalingFactor .!= 0).item(Bool.self) else {
            throw AudioGenerationError.invalidInput("Invalid VibeVoice speech scaling factors")
        }
        let tokenizerDir: URL
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path) {
            tokenizerDir = directory
        } else if FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer/tokenizer.json").path) {
            tokenizerDir = directory.appendingPathComponent("tokenizer")
        } else {
            let processorData = try Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
            let processor = try JSONSerialization.jsonObject(with: processorData) as? [String: Any]
            guard let name = processor?["language_model_pretrained_name"] as? String, let id = Repo.ID(rawValue: name) else {
                throw AudioGenerationError.modelNotInitialized("VibeVoice checkpoint has no tokenizer or language_model_pretrained_name")
            }
            let client = HubClient(cache: cache)
            let info = try await client.getModel(id, revision: "main", filesMetadata: false)
            guard let revision = info.sha else { throw AudioGenerationError.modelNotInitialized("Could not resolve VibeVoice tokenizer revision") }
            tokenizerDir = cache.cacheDirectory.appendingPathComponent("mlx-audio/tokenizers/" + name.replacingOccurrences(of: "/", with: "_") + "/" + revision)
            _ = try await client.downloadSnapshot(of: id, kind: .model, to: tokenizerDir, revision: revision,
                matching: ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        for (token, id) in [("<|vision_start|>", speechStartID), ("<|vision_end|>", speechEndID), ("<|vision_pad|>", speechDiffusionID)] {
            guard tokenizer.convertTokenToId(token) == id else { throw AudioGenerationError.invalidInput("Incompatible VibeVoice speech tokenizer: \(token)") }
        }
        guard let eos = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<|endoftext|>"), eos >= 0,
              eos < configuration.decoder.vocabularySize else { throw AudioGenerationError.invalidInput("VibeVoice tokenizer has no valid EOS") }
        model.eosTokenID = eos
        model.controlTokens = [speechStartID, speechEndID, speechDiffusionID, eos]
        if let bos = tokenizer.bosTokenId, bos >= 0, !model.controlTokens.contains(bos) { model.controlTokens.append(bos) }
        model.tokenizer = tokenizer
        return model
    }
}
