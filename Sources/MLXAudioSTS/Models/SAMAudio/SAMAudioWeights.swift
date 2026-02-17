import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

private struct LSTMBiasKey {
    let base: String
    let biasType: String
    let layer: String
}

private func parseLSTMBiasKey(_ key: String) -> LSTMBiasKey? {
    guard let markerRange = key.range(of: ".lstm.bias_", options: .backwards) else {
        return nil
    }
    let prefix = String(key[..<markerRange.lowerBound])
    let remainder = String(key[markerRange.upperBound...])
    guard let split = remainder.range(of: "_l", options: .backwards) else {
        return nil
    }

    let biasType = String(remainder[..<split.lowerBound])
    let layer = String(remainder[split.upperBound...])
    guard (biasType == "ih" || biasType == "hh"), !layer.isEmpty, layer.allSatisfy(\.isNumber) else {
        return nil
    }

    return LSTMBiasKey(base: "\(prefix).lstm", biasType: biasType, layer: layer)
}

private func mapResidualUnit(_ name: String, prefix: String) -> String {
    guard name.hasPrefix(prefix) else {
        return name
    }

    let suffix = String(name.dropFirst(prefix.count))
    if suffix.hasPrefix("block.0.") {
        return prefix + "act1." + String(suffix.dropFirst(8))
    } else if suffix.hasPrefix("block.1.") {
        return prefix + "conv1." + String(suffix.dropFirst(8))
    } else if suffix.hasPrefix("block.2.") {
        return prefix + "act2." + String(suffix.dropFirst(8))
    } else if suffix.hasPrefix("block.3.") {
        return prefix + "conv2." + String(suffix.dropFirst(8))
    }

    return name
}

private func mapDecoderResidualUnit(_ name: String, prefix: String) -> String {
    guard name.hasPrefix(prefix) else {
        return name
    }

    let suffix = String(name.dropFirst(prefix.count))
    if suffix.hasPrefix("block.0.") {
        return prefix + "act1." + String(suffix.dropFirst(8))
    } else if suffix.hasPrefix("block.1.") {
        return prefix + "conv1." + String(suffix.dropFirst(8))
    } else if suffix.hasPrefix("block.2.") {
        return prefix + "act2." + String(suffix.dropFirst(8))
    } else if suffix.hasPrefix("block.3.") {
        return prefix + "conv2." + String(suffix.dropFirst(8))
    }

    return name
}

extension SAMAudio {
    public static let defaultRepo = "mlx-community/sam-audio-large-fp16"

    static func convertWeightName(_ name: String) -> String {
        var result = name

        if result.hasPrefix("audio_codec.encoder.block.0.") {
            result = result.replacingOccurrences(
                of: "audio_codec.encoder.block.0.",
                with: "audio_codec.encoder.conv_in."
            )
        }

        for encIdx in 1...4 {
            let blockIdx = encIdx - 1

            for resIdx in 0...2 {
                let resName = "res\(resIdx + 1)"
                let oldPrefix = "audio_codec.encoder.block.\(encIdx).block.\(resIdx)."
                let newPrefix = "audio_codec.encoder.blocks.\(blockIdx).\(resName)."
                if result.hasPrefix(oldPrefix) {
                    result = result.replacingOccurrences(of: oldPrefix, with: newPrefix)
                    result = mapResidualUnit(result, prefix: newPrefix)
                    break
                }
            }

            let snakePrefix = "audio_codec.encoder.block.\(encIdx).block.3."
            if result.hasPrefix(snakePrefix) {
                result = result.replacingOccurrences(
                    of: snakePrefix,
                    with: "audio_codec.encoder.blocks.\(blockIdx).snake."
                )
            }

            let convPrefix = "audio_codec.encoder.block.\(encIdx).block.4."
            if result.hasPrefix(convPrefix) {
                result = result.replacingOccurrences(
                    of: convPrefix,
                    with: "audio_codec.encoder.blocks.\(blockIdx).conv."
                )
            }
        }

        if result.hasPrefix("audio_codec.encoder.block.5.") {
            result = result.replacingOccurrences(
                of: "audio_codec.encoder.block.5.",
                with: "audio_codec.encoder.snake_out."
            )
        }
        if result.hasPrefix("audio_codec.encoder.block.6.") {
            result = result.replacingOccurrences(
                of: "audio_codec.encoder.block.6.",
                with: "audio_codec.encoder.conv_out."
            )
        }

        if result.hasPrefix("audio_codec.decoder.model.0.") {
            result = result.replacingOccurrences(
                of: "audio_codec.decoder.model.0.",
                with: "audio_codec.decoder.conv_in."
            )
        }

        for decIdx in 1...4 {
            let blockIdx = decIdx - 1
            for blockNum in [0, 1, 3, 4, 5, 6, 7, 8, 11] {
                let oldPrefix = "audio_codec.decoder.model.\(decIdx).block.\(blockNum)."
                let newPrefix = "audio_codec.decoder.blocks.\(blockIdx).block_\(blockNum)."
                if result.hasPrefix(oldPrefix) {
                    result = result.replacingOccurrences(of: oldPrefix, with: newPrefix)
                    if [4, 5, 6, 7, 8].contains(blockNum) {
                        result = mapDecoderResidualUnit(result, prefix: newPrefix)
                    }
                    break
                }
            }
        }

        if result.hasPrefix("audio_codec.decoder.wm_model.encoder_block.pre.0.") {
            result = result.replacingOccurrences(
                of: "audio_codec.decoder.wm_model.encoder_block.pre.0.",
                with: "audio_codec.decoder.snake_out."
            )
        }
        if result.hasPrefix("audio_codec.decoder.wm_model.encoder_block.pre.1.") {
            result = result.replacingOccurrences(
                of: "audio_codec.decoder.wm_model.encoder_block.pre.1.",
                with: "audio_codec.decoder.conv_out."
            )
        }

        for block in ["encoder_block", "decoder_block"] {
            for prefix in ["pre", "post"] {
                for idx in 0..<4 {
                    let old = ".\(block).\(prefix).\(idx)."
                    let new = ".\(block).\(prefix)_\(idx)."
                    if result.contains(old) {
                        result = result.replacingOccurrences(of: old, with: new)
                    }
                }
            }
        }

        for (marker, replacement) in [
            ("weight_ih_l", "Wx"),
            ("weight_hh_l", "Wh"),
            ("combined_bias_l", "bias"),
        ] {
            let token = ".lstm.\(marker)"
            guard let range = result.range(of: token, options: .backwards) else {
                continue
            }
            let layer = String(result[range.upperBound...])
            if !layer.isEmpty, layer.allSatisfy(\.isNumber) {
                let prefix = String(result[..<range.lowerBound])
                result = "\(prefix).lstm.layers.\(layer).\(replacement)"
            }
            break
        }

        if result.hasPrefix("audio_codec.quantizer.in_proj.") {
            result = result.replacingOccurrences(
                of: "audio_codec.quantizer.in_proj.",
                with: "audio_codec.quantizer_in_proj."
            )
        }
        if result.hasPrefix("audio_codec.quantizer.out_proj.") {
            result = result.replacingOccurrences(
                of: "audio_codec.quantizer.out_proj.",
                with: "audio_codec.quantizer_out_proj."
            )
        }

        return result
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let dropPrefixes = [
            "text_encoder.",
            "span_predictor.",
            "visual_ranker.",
            "text_ranker.",
            "vision_encoder.",
            "align_masked_video.",
        ]

        var lstmBiases: [String: [String: MLXArray]] = [:]
        var keysToRemove = Set<String>()

        for (key, value) in weights {
            if dropPrefixes.contains(where: { key.hasPrefix($0) }) || key.contains("wm_rates") {
                keysToRemove.insert(key)
                continue
            }

            if let parsed = parseLSTMBiasKey(key) {
                let bucketKey = "\(parsed.base)|\(parsed.layer)"
                var bucket = lstmBiases[bucketKey, default: [:]]
                bucket[parsed.biasType] = value
                lstmBiases[bucketKey] = bucket
                keysToRemove.insert(key)
            }
        }

        var sanitized: [String: MLXArray] = [:]

        for (bucketKey, bucket) in lstmBiases {
            guard let ih = bucket["ih"], let hh = bucket["hh"] else {
                continue
            }
            let parts = bucketKey.split(separator: "|", maxSplits: 1).map(String.init)
            guard parts.count == 2 else {
                continue
            }
            let combinedKey = convertWeightName("\(parts[0]).combined_bias_l\(parts[1])")
            sanitized[combinedKey] = ih + hh
        }

        for (key, value) in weights where !keysToRemove.contains(key) {
            sanitized[convertWeightName(key)] = value
        }

        return sanitized
    }

    static func convertWeightShape(_ value: MLXArray, targetShape: [Int]) -> MLXArray? {
        let sourceShape = value.shape
        if sourceShape == targetShape {
            return value
        }

        if sourceShape.count == 2, sourceShape == Array(targetShape.reversed()) {
            return value.transposed(1, 0)
        }

        if sourceShape.count == 3, targetShape.count == 3 {
            if sourceShape[0] == targetShape[0], sourceShape[1] == targetShape[2], sourceShape[2] == targetShape[1] {
                return value.transposed(0, 2, 1)
            }

            if sourceShape[0] == targetShape[2], sourceShape[1] == targetShape[0], sourceShape[2] == targetShape[1] {
                return value.transposed(1, 2, 0)
            }

            if sourceShape[1] == 1, sourceShape[2] == 1, sourceShape[0] == targetShape[2] {
                return value.transposed(1, 2, 0)
            }
        }

        return nil
    }

    @discardableResult
    public func loadConvertedWeights(_ rawWeights: [String: MLXArray], strict: Bool = false) throws -> SAMAudio {
        let sanitized = Self.sanitize(weights: rawWeights)
        let modelParams = Dictionary(uniqueKeysWithValues: parameters().flattened())

        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(sanitized.count)
        var loadedKeys = Set<String>()

        for (key, value) in sanitized {
            guard let target = modelParams[key] else {
                continue
            }
            guard let converted = Self.convertWeightShape(value, targetShape: target.shape) else {
                continue
            }
            updates.append((key, converted))
            loadedKeys.insert(key)
        }

        guard !updates.isEmpty else {
            throw SAMAudioError.noCompatibleWeights
        }

        try update(parameters: ModuleParameters.unflattened(updates), verify: .noUnusedKeys)
        eval(parameters())

        if strict {
            let missing = modelParams.keys.filter { !loadedKeys.contains($0) && !$0.contains("wm_model") }
            if !missing.isEmpty {
                throw SAMAudioError.missingModelWeights(missing.count)
            }
        }

        return self
    }

    private static func loadAllSafetensors(from modelDir: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir,
            includingPropertiesForKeys: nil
        )
        let safetensors = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !safetensors.isEmpty else {
            throw SAMAudioError.modelFilesNotFound(modelDir.path)
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensors {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }
        return weights
    }

    public static func fromPretrained(
        _ modelPath: String = defaultRepo,
        hfToken: String? = nil,
        strict: Bool = false
    ) async throws -> SAMAudio {
        let fm = FileManager.default
        let modelDir: URL

        if fm.fileExists(atPath: modelPath) {
            modelDir = URL(fileURLWithPath: modelPath)
        } else {
            guard let repoID = Repo.ID(rawValue: modelPath) else {
                throw SAMAudioError.invalidRepoID(modelPath)
            }
            modelDir = try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: "safetensors",
                hfToken: hfToken
            )
        }

        let configURL = modelDir.appendingPathComponent("config.json")
        let config: SAMAudioConfig
        if fm.fileExists(atPath: configURL.path), let configData = try? Data(contentsOf: configURL) {
            config = (try? JSONDecoder().decode(SAMAudioConfig.self, from: configData)) ?? SAMAudioConfig()
        } else {
            config = SAMAudioConfig()
        }

        let model = SAMAudio(config: config)
        let weights = try loadAllSafetensors(from: modelDir)
        try model.loadConvertedWeights(weights, strict: strict)
        return model
    }
}
