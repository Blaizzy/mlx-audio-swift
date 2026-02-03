import Foundation
import HuggingFace
import MLXAudioCore

public enum TTSModelUtilsError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepositoryID(String)
    case unsupportedModelType(String?)

    public var errorDescription: String? {
        description
    }

    public var description: String {
        switch self {
        case .invalidRepositoryID(let modelRepo):
            return "Invalid repository ID: \(modelRepo)"
        case .unsupportedModelType(let modelType):
            return "Unsupported model type: \(String(describing: modelType))"
        }
    }
}

public enum TTSModelUtils {
    public static func loadModel(
        modelRepo: String,
        hfToken: String? = nil
    ) async throws -> SpeechGenerationModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelUtilsError.invalidRepositoryID(modelRepo)
        }

        let modelType = try await ModelUtils.resolveModelType(repoID: repoID, hfToken: hfToken)
        return try await loadModel(modelRepo: modelRepo, modelType: modelType)
    }

    public static func loadModel(
        modelRepo: String,
        modelType: String?
    ) async throws -> SpeechGenerationModel {
        let resolvedType = normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
        guard let resolvedType else {
            throw TTSModelUtilsError.unsupportedModelType(modelType)
        }

        switch resolvedType {
        case "qwen3_tts", "qwen3", "qwen":
            return try await Qwen3Model.fromPretrained(modelRepo)
        case "llama_tts", "llama3_tts", "llama3", "llama", "orpheus", "orpheus_tts":
            return try await LlamaTTSModel.fromPretrained(modelRepo)
        case "csm", "sesame":
            return try await MarvisTTSModel.fromPretrained(modelRepo)
        case "soprano_tts", "soprano":
            return try await SopranoModel.fromPretrained(modelRepo)
        case "pocket_tts":
            return try await PocketTTSModel.fromPretrained(modelRepo)
        default:
            throw TTSModelUtilsError.unsupportedModelType(modelType ?? resolvedType)
        }
    }

    private static func normalizedModelType(_ modelType: String?) -> String? {
        guard let modelType else { return nil }
        let trimmed = modelType.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed.lowercased()
    }

    private static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        if lower.contains("qwen") {
            return "qwen3_tts"
        }
        if lower.contains("soprano") {
            return "soprano"
        }
        if lower.contains("llama") || lower.contains("orpheus") {
            return "llama_tts"
        }
        if lower.contains("csm") || lower.contains("sesame") {
            return "csm"
        }
        if lower.contains("pocket_tts") {
            return "pocket_tts"
        }
        return nil
    }
}
