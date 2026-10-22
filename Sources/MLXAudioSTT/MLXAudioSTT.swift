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
        if localDirectory(modelRepo) != nil {
            return try await loadModel(modelRepo: modelRepo, modelType: nil, cache: cache)
        }

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
        let directory = localDirectory(modelRepo)
        var resolved = normalizedModelType(modelType)
        if resolved == nil, let directory {
            let config = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
            let configuredType = (config?["model_type"] as? String)
                ?? (config?["architecture"] as? String)
                ?? (config?["model_version"] as? String)
            resolved = normalizedModelType(configuredType)
        }
        guard let resolved = resolved ?? inferModelType(from: modelRepo) else {
            throw STTModelError.unsupportedModelType(modelType)
        }

        switch resolved {
        case "higgs_audio_3":
            return try await HiggsAudio3Model.fromPretrained(directory?.path ?? modelRepo, cache: cache)
        case "phonon2", "phonon_2":
            return try await Phonon2Model.fromPretrained(modelRepo, cache: cache)
        case "moss_transcribe_diarize":
            if let directory { return try await MossTranscribeDiarizeModel.fromModelDirectory(directory) }
            return try await MossTranscribeDiarizeModel.fromPretrained(modelRepo, cache: cache)
        case "mega_asr":
            if let directory { return try await MegaASRModel.fromModelDirectory(directory) }
            return try await MegaASRModel.fromPretrained(modelRepo, cache: cache)
        case "qwen3_asr":
            if let directory { return try await Qwen3ASRModel.fromModelDirectory(directory) }
            return try await Qwen3ASRModel.fromPretrained(modelRepo, cache: cache)
        case "glmasr", "glm":
            if let directory { return try await GLMASRModel.fromModelDirectory(directory) }
            return try await GLMASRModel.fromPretrained(modelRepo, cache: cache)
        case "voxtral", "voxtral_realtime":
            if let directory { return try VoxtralRealtimeModel.fromDirectory(directory) }
            return try await VoxtralRealtimeModel.fromPretrained(modelRepo, cache: cache)
        case "cohere_asr", "cohere":
            if let directory { return try CohereTranscribeModel.fromDirectory(directory) }
            return try await CohereTranscribeModel.fromPretrained(modelRepo, cache: cache)
        case "parakeet", "parakeet_tdt":
            if let directory { return try ParakeetModel.fromDirectory(directory) }
            return try await ParakeetModel.fromPretrained(modelRepo, cache: cache)
        case "canary":
            if let directory { return try await CanaryModel.fromModelDirectory(directory) }
            return try await CanaryModel.fromPretrained(modelRepo, cache: cache)
        case "wav2vec", "wav2vec2", "mms":
            if let directory { return try Wav2Vec2CTCModel.fromModelDirectory(directory) }
            return try await Wav2Vec2CTCModel.fromPretrained(modelRepo, cache: cache)
        case "lasr", "lasr_ctc":
            if let directory { return try LasrCTCModel.fromModelDirectory(directory) }
            return try await LasrCTCModel.fromPretrained(modelRepo, cache: cache)
        case "moonshine":
            if let directory { return try await MoonshineModel.fromModelDirectory(directory) }
            return try await MoonshineModel.fromPretrained(modelRepo, cache: cache)
        case "nemotron", "nemotron_asr":
            if let directory { return try NemotronASRModel.fromDirectory(directory) }
            return try await NemotronASRModel.fromPretrained(modelRepo, cache: cache)
        case "fireredasr2", "firered", "fire_red":
            if let directory { return try FireRedASR2Model.fromDirectory(directory) }
            return try await FireRedASR2Model.fromPretrained(modelRepo, cache: cache)
        case "fun_asr_nano":
            if let directory { return try await FunASRNanoModel.fromModelDirectory(directory) }
            return try await FunASRNanoModel.fromPretrained(modelRepo, cache: cache)
        case "sensevoice":
            if let directory { return try SenseVoiceModel.fromDirectory(directory) }
            return try await SenseVoiceModel.fromPretrained(modelRepo, cache: cache)
        case "whisper":
            if let directory { return try await WhisperModel.fromDirectory(directory, cache: cache) }
            return try await WhisperModel.fromPretrained(modelRepo, cache: cache)
        case "granite_speech5_ctc":
            return try await GraniteSpeech5CTCModel.fromPretrained(modelRepo, cache: cache)
        case "granite_speech":
            if let directory { return try await GraniteSpeechModel.fromModelDirectory(directory) }
            return try await GraniteSpeechModel.fromPretrained(modelRepo, cache: cache)
        default:
            throw STTModelError.unsupportedModelType(resolved)
        }
    }

    private static func localDirectory(_ path: String) -> URL? {
        let expanded = (path as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }
        return URL(fileURLWithPath: expanded, isDirectory: true).resolvingSymlinksInPath()
    }

    private static func normalizedModelType(_ modelType: String?) -> String? {
        guard let modelType else { return nil }
        let trimmed = modelType.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed.lowercased()
    }

    private static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        if lower.contains("higgs-audio-v3-stt") || lower.contains("higgs_audio_3") {
            return "higgs_audio_3"
        }
        if lower.contains("phonon-2") || lower.contains("phonon2") || lower.contains("phonon_2") {
            return "phonon2"
        }
        if lower.contains("moss-transcribe-diarize") || lower.contains("moss_transcribe_diarize") {
            return "moss_transcribe_diarize"
        }
        if lower.contains("forcedalign") || lower.contains("forced-align") {
            return nil
        }
        if lower.contains("mega-asr") || lower.contains("mega_asr") {
            return "mega_asr"
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
        if lower.contains("fun-asr-nano") || lower.contains("fun_asr_nano") {
            return "fun_asr_nano"
        }
        if lower.contains("sensevoice") {
            return "sensevoice"
        }
        if lower.contains("whisper") {
            return "whisper"
        }
        if lower.contains("turboctc") || lower.contains("granite_speech5_ctc") {
            return "granite_speech5_ctc"
        }
        if lower.contains("granite") {
            return "granite_speech"
        }
        return nil
    }
}
