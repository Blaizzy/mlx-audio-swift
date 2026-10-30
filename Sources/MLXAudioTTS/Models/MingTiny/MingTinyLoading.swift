import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

extension MingTinyModel {
    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = mingSanitizeSpeechWeights(weights)
        for key in Array(result.keys) {
            let parts = key.split(separator: ".")
            if parts.count > 3, parts[0] == "model", parts[1] == "model", parts[2] == "layers",
               let index = Int(parts[3]), index >= config.text.numHiddenLayers {
                result.removeValue(forKey: key)
            } else if key.hasSuffix("rotary_emb.inv_freq") {
                result.removeValue(forKey: key)
            } else if key.hasSuffix("_conv1d.weight"), let weight = result[key], weight.ndim == 3, weight.dim(-1) != 1 {
                result[key] = weight.transposed(0, 2, 1)
            }
        }
        let c = config.text
        for layer in c.firstKDenseReplace..<c.numHiddenLayers {
            let stem = "model.model.layers.\(layer).mlp."
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let keys = (0..<c.numExperts).map { stem + "experts.\($0).\(projection).\(suffix)" }
                    guard result[keys[0]] != nil else { continue }
                    guard keys.allSatisfy({ result[$0] != nil }) else {
                        throw MingTinyError.invalidConfiguration("Incomplete routed experts for \(stem)\(projection)")
                    }
                    result[stem + "switch_mlp.\(projection).\(suffix)"] = stacked(keys.map { result.removeValue(forKey: $0)! })
                }
            }
        }
        return result
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> MingTinyModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo,
            requiredExtension: ".safetensors", additionalMatchingPatterns: ["speaker_encoder/*.safetensors"], cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> MingTinyModel {
        let config = try JSONDecoder().decode(MingTinyConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = try MingTinyModel(config: config)
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            struct Index: Decodable { let weight_map: [String: String] }
            let names = Set(try JSONDecoder().decode(Index.self, from: Data(contentsOf: index)).weight_map.values).sorted()
            guard names.allSatisfy({ !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors") }) else {
                throw MingTinyError.invalidConfiguration("Invalid Ming Tiny weight index")
            }
            files = names.map { directory.appendingPathComponent($0) }
        } else {
            files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        guard !files.isEmpty else { throw MingTinyError.missingComponent("No Ming Tiny safetensors found") }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = try model.sanitize(weights: weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                guard path.hasPrefix("model."), !path.hasSuffix(".gate"), weights[path + ".scales"] != nil else { return nil }
                return (q.groupSize, q.bits, q.mode)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let s = config.speech
        for (token, expected) in [("<audioPatch>", s.audioPatchTokenId), ("<audio>", s.audioStartTokenId),
            ("</audio>", s.audioEndTokenId), ("<spk>", s.speakerStartTokenId), ("</spk>", s.speakerEndTokenId)] {
            guard model.tokenizer?.encode(text: token, addSpecialTokens: false) == [expected] else {
                throw MingTinyError.invalidConfiguration("Tokenizer \(token) does not match the checkpoint")
            }
        }
        guard model.tokenizer?.eosTokenId == s.promptEosTokenId else {
            throw MingTinyError.invalidConfiguration("Tokenizer prompt EOS does not match Ming Tiny")
        }
        model.directory = directory
        model.train(false)
        eval(model)
        return model
    }
}
