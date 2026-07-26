// MLXAudioSTT - Speech-to-Text module

import Foundation
import HuggingFace
import MLXAudioCore

public enum STTModelError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepositoryID(String)
    case unsupportedModelType(String?)

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalidRepositoryID(let modelRepo):
            return "Invalid repository ID: \(modelRepo)"
        case .unsupportedModelType(let modelType):
            return "Unsupported STT model type: \(String(describing: modelType))"
        }
    }
}

public enum STT {
    public static func loadModel(
        modelRepo: String,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> any STTGenerationModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }

        let modelType = try await ModelUtils.resolveModelType(
            repoID: repoID,
            hfToken: hfToken,
            cache: cache
        )
        return try await loadModel(modelRepo: modelRepo, modelType: modelType, cache: cache)
    }

    public static func loadModel(
        modelRepo: String,
        modelType: String?,
        cache: HubCache = .default
    ) async throws -> any STTGenerationModel {
        let resolved = normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
        guard let resolved else {
            throw STTModelError.unsupportedModelType(modelType)
        }

        switch resolved {
        case "moss_transcribe_diarize":
            return try await MossTranscribeDiarizeModel.fromPretrained(modelRepo, cache: cache)
        case "qwen3_asr":
            return try await Qwen3ASRModel.fromPretrained(modelRepo, cache: cache)
        case "glmasr", "glm":
            return try await GLMASRModel.fromPretrained(modelRepo, cache: cache)
        case "voxtral", "voxtral_realtime":
            return try await VoxtralRealtimeModel.fromPretrained(modelRepo)
        case "cohere_asr", "cohere":
            return try await CohereTranscribeModel.fromPretrained(modelRepo)
        case "parakeet":
            return try await ParakeetModel.fromPretrained(modelRepo, cache: cache)
        case "canary":
            return try await CanaryModel.fromPretrained(modelRepo)
        case "wav2vec", "wav2vec2", "mms":
            return try await Wav2Vec2CTCModel.fromPretrained(modelRepo)
        case "lasr", "lasr_ctc":
            return try await LasrCTCModel.fromPretrained(modelRepo)
        case "moonshine":
            return try await MoonshineModel.fromPretrained(modelRepo)
        case "nemotron", "nemotron_asr":
            return try await NemotronASRModel.fromPretrained(modelRepo, cache: cache)
        case "fireredasr2", "firered", "fire_red":
            return try await FireRedASR2Model.fromPretrained(modelRepo, cache: cache)
        case "sensevoice":
            return try await SenseVoiceModel.fromPretrained(modelRepo, cache: cache)
        case "whisper":
            return try await WhisperModel.fromPretrained(modelRepo, cache: cache)
        case "granite_speech":
            return try await GraniteSpeechModel.fromPretrained(modelRepo, cache: cache)
        default:
            throw STTModelError.unsupportedModelType(resolved)
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
        if lower.contains("moss-transcribe-diarize") || lower.contains("moss_transcribe_diarize") {
            return "moss_transcribe_diarize"
        }
        if lower.contains("forcedalign") || lower.contains("forced-align") {
            return nil
        }
        if lower.contains("qwen3-asr") || lower.contains("qwen3_asr") {
            return "qwen3_asr"
        }
        if lower.contains("glmasr") || lower.contains("glm-asr") {
            return "glmasr"
        }
        if lower.contains("voxtral") {
            return "voxtral_realtime"
        }
        if lower.contains("cohere") {
            return "cohere_asr"
        }
        if lower.contains("parakeet") {
            return "parakeet"
        }
        if lower.contains("canary") {
            return "canary"
        }
        if lower.contains("wav2vec") || lower.contains("wav2vec2") || lower.contains("/mms-")
            || lower.contains("mms_") || lower.contains("mms-") {
            return "wav2vec2"
        }
        if lower.contains("lasr") {
            return "lasr_ctc"
        }
        if lower.contains("moonshine") {
            return "moonshine"
        }
        if lower.contains("nemotron") {
            return "nemotron_asr"
        }
        if lower.contains("firered") || lower.contains("fire-red") {
            return "fireredasr2"
        }
        if lower.contains("sensevoice") {
            return "sensevoice"
        }
        if lower.contains("whisper") {
            return "whisper"
        }
        if lower.contains("granite") {
            return "granite_speech"
        }
        return nil
    }
}
