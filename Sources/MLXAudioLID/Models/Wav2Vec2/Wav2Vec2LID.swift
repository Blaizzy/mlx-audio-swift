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

    /// Forward pass: raw waveform → logits over language classes.
    /// - Parameter waveform: Raw audio waveform `(batch, time)` at 16 kHz
    /// - Returns: Logits `(batch, numLabels)` — apply `softmax` for probabilities
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

    /// Run language identification on a 16 kHz mono audio waveform.
    /// Audio is automatically normalized (zero-mean, unit-variance) before inference.
    /// - Parameters:
    ///   - waveform: 1-D audio samples at 16 kHz
    ///   - topK: Number of top language predictions to return (default: 5)
    /// - Returns: `LIDOutput` with top predicted language and confidence scores
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

    /// Remap HuggingFace weight keys to MLX model structure.
    /// Handles `wav2vec2.*` prefix stripping, conv weight transposition,
    /// and positional conv weight-norm decomposition (`weight_g` + `weight_v` → `weight`).
    /// - Parameter weights: Raw weights loaded from `.safetensors` files
    /// - Returns: Sanitized weight dictionary ready for `model.update(parameters:)`
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

    /// Download and load a pretrained MMS-LID model from Hugging Face.
    /// Uses `HF_TOKEN` environment variable or Info.plist for authentication.
    /// - Parameter modelName: Hugging Face repository ID (e.g. `"facebook/mms-lid-256"`)
    /// - Returns: A loaded and evaluated `Wav2Vec2ForSequenceClassification` model
    public static func fromPretrained(
        _ modelName: String
    ) async throws -> Wav2Vec2ForSequenceClassification {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelName) else {
            throw LIDError.invalidRepoID(modelName)
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LIDError.configNotFound
        }
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Wav2Vec2LIDConfig.self, from: configData)
        let model = Wav2Vec2ForSequenceClassification(config: config)

        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !safetensorFiles.isEmpty else {
            throw LIDError.weightsNotFound
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitized = Self.sanitize(weights: weights)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all]
        )
        eval(model)

        return model
    }
}
