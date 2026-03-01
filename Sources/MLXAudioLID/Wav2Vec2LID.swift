import Foundation
import MLX
import MLXNN
import MLXAudioCore
import HuggingFace

public class Wav2Vec2ForSequenceClassification: Module {
    public let config: Wav2Vec2LIDConfig
    public let id2label: [Int: String]

    @ModuleInfo(key: "feature_extractor") var featureExtractor: Wav2Vec2FeatureExtractor
    @ModuleInfo(key: "feature_projection") var featureProjection: Wav2Vec2FeatureProjection
    @ModuleInfo var encoder: Wav2Vec2Encoder
    @ModuleInfo var projector: Linear
    @ModuleInfo var classifier: Linear

    public init(config: Wav2Vec2LIDConfig) {
        self.config = config

        var labels: [Int: String] = [:]
        if let mapping = config.id2label {
            for (key, value) in mapping {
                if let idx = Int(key) {
                    labels[idx] = value
                }
            }
        }
        self.id2label = labels

        let featureOutputDim = config.convDim.last ?? 512
        _featureExtractor.wrappedValue = Wav2Vec2FeatureExtractor(config: config)
        _featureProjection.wrappedValue = Wav2Vec2FeatureProjection(
            inputDim: featureOutputDim, outputDim: config.hiddenSize
        )
        _encoder.wrappedValue = Wav2Vec2Encoder(config: config)
        _projector.wrappedValue = Linear(config.hiddenSize, config.classifierProjSize)
        let outputLabels = config.id2label?.count ?? config.numLabels ?? 256
        _classifier.wrappedValue = Linear(config.classifierProjSize, outputLabels)
    }

    public func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        var x = expandedDimensions(waveform, axis: -1)
        x = featureExtractor(x)
        x = featureProjection(x)
        x = encoder(x)
        x = mean(x, axis: 1)
        x = projector(x)
        return classifier(x)
    }

    // MARK: - Prediction

    public func predict(waveform: MLXArray, topK: Int = 5) -> LIDOutput {
        let m = mean(waveform)
        let s = sqrt(mean((waveform - m) * (waveform - m)))
        let normalized = (waveform - m) / (s + 1e-7)

        let input = expandedDimensions(normalized, axis: 0)
        let logits = self.callAsFunction(input)
        let probs = softmax(logits, axis: -1)

        let probsFlat = probs.squeezed(axis: 0)
        let topIndices = argSort(probsFlat, axis: -1)

        let numLabels = probsFlat.dim(0)
        let k = min(topK, numLabels)
        var topLanguages: [LanguagePrediction] = []

        for i in 0..<k {
            let idx = topIndices[numLabels - 1 - i].item(Int.self)
            let conf = probsFlat[idx].item(Float.self)
            let lang = id2label[idx] ?? "unknown_\(idx)"
            topLanguages.append(LanguagePrediction(language: lang, confidence: conf))
        }

        let best = topLanguages.first ?? LanguagePrediction(language: "unknown", confidence: 0)
        return LIDOutput(
            language: best.language,
            confidence: best.confidence,
            topLanguages: topLanguages
        )
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var weightG: MLXArray?
        var weightV: MLXArray?

        for (key, var value) in weights {
            if key.contains("masked_spec_embed") { continue }
            if key.contains("adapter_layer") { continue }

            var newKey: String
            if key.hasPrefix("projector.") || key.hasPrefix("classifier.") {
                newKey = key
            } else if key.hasPrefix("wav2vec2.") {
                newKey = String(key.dropFirst("wav2vec2.".count))
            } else {
                continue
            }

            if newKey == "encoder.pos_conv_embed.conv.weight_g" {
                weightG = value
                continue
            }
            if newKey == "encoder.pos_conv_embed.conv.weight_v" {
                weightV = value
                continue
            }

            if newKey.hasSuffix(".conv.weight") && value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }

            sanitized[newKey] = value
        }

        if let g = weightG, let v = weightV {
            let norm = sqrt(sum(v * v, axes: [0, 1], keepDims: true) + 1e-12)
            var fullWeight = g * v / norm
            fullWeight = fullWeight.transposed(0, 2, 1)
            sanitized["encoder.pos_conv_embed.conv.weight"] = fullWeight
        }

        return sanitized
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelName: String, hfToken: String? = nil
    ) async throws -> Wav2Vec2ForSequenceClassification {
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: Repo.ID(rawValue: modelName)!,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        let configPath = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configPath.path) else {
            throw LIDError.configNotFound
        }
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Wav2Vec2LIDConfig.self, from: configData)

        let model = Wav2Vec2ForSequenceClassification(config: config)
        model.train(false)

        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
        guard !safetensorFiles.isEmpty else {
            throw LIDError.weightsNotFound
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let fileWeights = try loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitized = Self.sanitize(weights: weights)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys
        )
        eval(model)

        return model
    }
}
