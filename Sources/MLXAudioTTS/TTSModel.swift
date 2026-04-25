import Foundation
import HuggingFace
import MLXAudioCore

public enum TTSModelError: Error, LocalizedError, CustomStringConvertible {
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

public enum TTS {
    public static func loadModel(
        modelRepo: String,
        textProcessor: TextProcessor? = nil,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> SpeechGenerationModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }

        let modelType = try await ModelUtils.resolveModelType(
            repoID: repoID,
            hfToken: hfToken,
            cache: cache
        )
        return try await loadModel(modelRepo: modelRepo, modelType: modelType, textProcessor: textProcessor, cache: cache)
    }

    public static func loadModel(
        modelRepo: String,
        modelType: String?,
        textProcessor: TextProcessor? = nil,
        cache: HubCache = .default
    ) async throws -> SpeechGenerationModel {
        let resolvedType = normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
        guard let resolvedType else {
            throw TTSModelError.unsupportedModelType(modelType)
        }

        switch resolvedType {
        case "moss_tts_nano":
            return try await MossTTSNanoModel.fromPretrained(modelRepo, cache: cache)
        case "echo_tts", "echo":
            return try await EchoTTSModel.fromPretrained(modelRepo, cache: cache)
        case "qwen3_tts":
            return try await Qwen3TTSModel.fromPretrained(modelRepo, cache: cache)
        case "qwen3", "qwen":
            return try await Qwen3Model.fromPretrained(modelRepo, cache: cache)
        case "fish_speech", "fish_qwen3_omni":
            return try await FishSpeechModel.fromPretrained(modelRepo, cache: cache)
        case "llama_tts", "llama3_tts", "llama3", "llama", "orpheus", "orpheus_tts":
            return try await LlamaTTSModel.fromPretrained(modelRepo, cache: cache)
        case "csm", "sesame":
            return try await MarvisTTSModel.fromPretrained(modelRepo, cache: cache)
        case "soprano_tts", "soprano":
            return try await SopranoModel.fromPretrained(modelRepo, cache: cache)
        case "pocket_tts":
            return try await PocketTTSModel.fromPretrained(modelRepo, cache: cache)
        case "chatterbox", "chatterbox_tts", "chatterbox_turbo":
            return try await ChatterboxModel.fromPretrained(modelRepo)
        case "kitten_tts", "kitten":
            return try await KittenTTSModel.fromPretrained(modelRepo, textProcessor: textProcessor ?? MisakiTextProcessor(), cache: cache)
        case "kokoro", "kokoro_tts":
            let processor = textProcessor ?? KokoroMultilingualProcessor()
            return try await KokoroModel.fromPretrained(modelRepo, textProcessor: processor, cache: cache)
        default:
            throw TTSModelError.unsupportedModelType(modelType ?? resolvedType)
        }
    }

    private static func normalizedModelType(_ modelType: String?) -> String? {
        guard let modelType else { return nil }
        let trimmed = modelType.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed.lowercased()
    }

    static func resolveModelType(modelRepo: String, modelType: String? = nil) -> String? {
        normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
    }

    private static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        if lower.contains("qwen3_tts") {
            return "qwen3_tts"
        }
        if lower.contains("fish_qwen3_omni") {
            return "fish_qwen3_omni"
        }
        if lower.contains("fish-audio") || lower.contains("fish_audio")
            || lower.contains("fish-speech") || lower.contains("fish_speech")
        {
            return "fish_speech"
        }
        if lower.contains("echo") {
            return "echo_tts"
        }
        if lower.contains("moss") && lower.contains("tts") {
            return "moss_tts_nano"
        }
        if lower.contains("qwen3") || lower.contains("qwen") {
            return "qwen3"
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
        if lower.contains("chatterbox") {
            return "chatterbox"
        }
        if lower.contains("kitten") {
            return "kitten_tts"
        }
        if lower.contains("kokoro") {
            return "kokoro"
        }
        return nil
    }
}

@available(*, deprecated, renamed: "TTSModelError")
public typealias TTSModelUtilsError = TTSModelError

@available(*, deprecated, renamed: "TTS")
public typealias TTSModelUtils = TTS
