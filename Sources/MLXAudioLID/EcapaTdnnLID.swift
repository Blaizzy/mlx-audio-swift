import Foundation
import MLX
import MLXNN
import MLXAudioCore
import HuggingFace

/// ECAPA-TDNN model for spoken language identification (107 languages).
///
/// Based on the SpeechBrain `speechbrain/lang-id-voxlingua107-ecapa` model,
/// trained on VoxLingua107 dataset. Input is raw 16 kHz mono audio; output is
/// a probability distribution over 107 languages.
public class EcapaTdnn: Module {
    public let config: EcapaTdnnConfig
    public let id2label: [Int: String]

    @ModuleInfo(key: "embedding_model") var embeddingModel: EcapaTdnnEmbedding
    @ModuleInfo var classifier: EcapaClassifier

    public init(config: EcapaTdnnConfig) {
        self.config = config

        var labels: [Int: String] = [:]
        if let mapping = config.id2label {
            for (key, value) in mapping {
                if let idx = Int(key) {
                    let lang = value.components(separatedBy: ":").first?
                        .trimmingCharacters(in: .whitespaces) ?? value
                    labels[idx] = lang
                }
            }
        }
        self.id2label = labels

        _embeddingModel.wrappedValue = EcapaTdnnEmbedding(config: config)
        _classifier.wrappedValue = EcapaClassifier(config: config)
    }

    /// Forward pass: mel features → log-probabilities over language classes.
    /// - Parameter melFeatures: `[batch, time, nMels]` mel spectrogram
    /// - Returns: Log-probabilities `[batch, numClasses]`
    public func callAsFunction(_ melFeatures: MLXArray) -> MLXArray {
        let embeddings = embeddingModel(melFeatures)
        return classifier(embeddings)
    }

    // MARK: - Prediction

    /// Run language identification on a 16 kHz mono audio waveform.
    /// Computes SpeechBrain-compatible mel spectrogram internally.
    /// - Parameters:
    ///   - waveform: 1-D audio samples at 16 kHz
    ///   - topK: Number of top language predictions to return (default: 5)
    /// - Returns: `LIDOutput` with top predicted language and confidence scores
    public func predict(waveform: MLXArray, topK: Int = 5) -> LIDOutput {
        let mel = EcapaMelSpectrogram.compute(audio: waveform)
        let logProbs = self.callAsFunction(mel)
        let probs = exp(logProbs)

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

    /// Remap SpeechBrain checkpoint keys to MLX model structure.
    ///
    /// Handles:
    /// - Dropping `num_batches_tracked` keys
    /// - Remapping top-level block indices: `blocks.0.` → `blocks.block0.`
    /// - Flattening SpeechBrain double-nesting: `.conv.conv.` → `.conv.`
    /// - SE block conv wrappers, ASP BN norm, FC conv flattening
    /// - Parameter weights: Raw weights loaded from `.safetensors` files
    /// - Returns: Sanitized weight dictionary ready for `model.update(parameters:)`
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("num_batches_tracked") { continue }

            var k = key

            // Remap top-level block indices (NOT res2net_block.blocks which is a real array)
            k = k.replacingOccurrences(of: "embedding_model.blocks.0.", with: "embedding_model.blocks.block0.")
            k = k.replacingOccurrences(of: "embedding_model.blocks.1.", with: "embedding_model.blocks.block1.")
            k = k.replacingOccurrences(of: "embedding_model.blocks.2.", with: "embedding_model.blocks.block2.")
            k = k.replacingOccurrences(of: "embedding_model.blocks.3.", with: "embedding_model.blocks.block3.")

            // Flatten SpeechBrain double-nesting
            k = k.replacingOccurrences(of: ".conv.conv.", with: ".conv.")
            k = k.replacingOccurrences(of: ".norm.norm.", with: ".norm.")

            // SE block Conv1d wrappers
            k = k.replacingOccurrences(of: ".se_block.conv1.conv.", with: ".se_block.conv1.")
            k = k.replacingOccurrences(of: ".se_block.conv2.conv.", with: ".se_block.conv2.")

            // ASP BN
            k = k.replacingOccurrences(of: ".asp_bn.norm.", with: ".asp_bn.")

            // FC conv
            k = k.replacingOccurrences(of: ".fc.conv.", with: ".fc.")

            sanitized[k] = value
        }

        return sanitized
    }

    // MARK: - Loading

    /// Download and load a pretrained ECAPA-TDNN model from Hugging Face.
    /// Uses `HF_TOKEN` environment variable or Info.plist for authentication.
    /// - Parameter modelName: Hugging Face repository ID (e.g. `"beshkenadze/lang-id-voxlingua107-ecapa-mlx"`)
    /// - Returns: A loaded and evaluated `EcapaTdnn` model
    public static func fromPretrained(
        _ modelName: String
    ) async throws -> EcapaTdnn {
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

        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(EcapaTdnnConfig.self, from: configData)

        guard config.id2label != nil else {
            throw LIDError.noLabels
        }

        let model = EcapaTdnn(config: config)

        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
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
