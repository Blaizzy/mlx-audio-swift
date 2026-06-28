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
    private enum ModelSource {
        case repository(String, cache: HubCache)
        case localDirectory(URL, repoHint: String)

        var fallbackName: String {
            switch self {
            case .repository(let modelRepo, _):
                modelRepo
            case .localDirectory(let modelDir, let repoHint):
                repoHint.isEmpty ? modelDir.path : repoHint
            }
        }
    }

    public static func loadModel(
        modelRepo: String,
        textProcessor: TextProcessor? = nil,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> SpeechGenerationModel {
        if let modelDir = localModelDirectory(modelRepo) {
            let modelType = try localModelType(modelDir) ?? inferModelType(from: modelRepo)
            return try await loadResolvedModel(
                modelType: modelType,
                source: .localDirectory(modelDir, repoHint: modelRepo),
                textProcessor: textProcessor
            )
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }

        let modelType = try await ModelUtils.resolveModelType(
            repoID: repoID,
            hfToken: hfToken,
            cache: cache
        )
        return try await loadResolvedModel(
            modelType: modelType,
            source: .repository(modelRepo, cache: cache),
            textProcessor: textProcessor
        )
    }

    public static func loadModel(
        modelRepo: String,
        modelType: String?,
        textProcessor: TextProcessor? = nil,
        cache: HubCache = .default
    ) async throws -> SpeechGenerationModel {
        if let modelDir = localModelDirectory(modelRepo) {
            let localType = try localModelType(modelDir)
            let resolvedModelType = normalizedModelType(modelType)
                ?? localType
                ?? inferModelType(from: modelRepo)
            return try await loadResolvedModel(
                modelType: resolvedModelType,
                source: .localDirectory(modelDir, repoHint: modelRepo),
                textProcessor: textProcessor
            )
        }

        return try await loadResolvedModel(
            modelType: modelType,
            source: .repository(modelRepo, cache: cache),
            textProcessor: textProcessor
        )
    }

    private static func loadResolvedModel(
        modelType: String?,
        source: ModelSource,
        textProcessor: TextProcessor?
    ) async throws -> SpeechGenerationModel {
        let resolvedType = normalizedModelType(modelType) ?? inferModelType(from: source.fallbackName)
        guard let resolvedType else {
            throw TTSModelError.unsupportedModelType(modelType)
        }

        switch resolvedType {
        case "moss_tts_nano":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await MossTTSNanoModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await MossTTSNanoModel.fromModelDirectory(modelDir) }
            )
        case "moss_tts", "moss_tts_delay", "moss_tts_local":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await MossTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await MossTTSModel.fromModelDirectory(modelDir) }
            )
        case "echo_tts", "echo":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await EchoTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await EchoTTSModel.fromModelDirectory(modelDir) }
            )
        case "irodori_tts", "irodori":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await IrodoriTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await IrodoriTTSModel.fromModelDirectory(modelDir) }
            )
        case "qwen3_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await Qwen3TTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await Qwen3TTSModel.fromModelDirectory(modelDir) }
            )
        case "qwen3", "qwen":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await Qwen3Model.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await Qwen3Model.fromModelDirectory(modelDir) }
            )
        case "fish_speech", "fish_qwen3_omni":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await FishSpeechModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await FishSpeechModel.fromModelDirectory(modelDir) }
            )
        case "llama_tts", "llama3_tts", "llama3", "llama", "orpheus", "orpheus_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await LlamaTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await LlamaTTSModel.fromModelDirectory(modelDir) }
            )
        case "csm", "sesame":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await MarvisTTSModel.fromPretrained($0, cache: $1) }
            )
        case "soprano_tts", "soprano":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await SopranoModel.fromPretrained($0, cache: $1) },
                local: { modelDir, repoHint in try await SopranoModel.fromModelDirectory(modelDir, repo: repoHint) }
            )
        case "pocket_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await PocketTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await PocketTTSModel.fromModelDirectory(modelDir) }
            )
        case "chatterbox", "chatterbox_tts", "chatterbox_turbo":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { modelRepo, _ in try await ChatterboxModel.fromPretrained(modelRepo) },
                local: { modelDir, _ in try await ChatterboxModel.fromModelDirectory(modelDir, hfToken: nil) }
            )
        case "kitten_tts", "kitten":
            let processor = textProcessor ?? MisakiTextProcessor()
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await KittenTTSModel.fromPretrained($0, textProcessor: processor, cache: $1) },
                local: { modelDir, _ in try await KittenTTSModel.fromModelDirectory(modelDir, textProcessor: processor) }
            )
        case "kokoro", "kokoro_tts":
            let processor = textProcessor ?? KokoroMultilingualProcessor()
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await KokoroModel.fromPretrained($0, textProcessor: processor, cache: $1) },
                local: { modelDir, _ in try await KokoroModel.fromModelDirectory(modelDir, textProcessor: processor) }
            )
        case "omnivoice":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await OmniVoiceModel.fromPretrained($0, cache: $1) }
            )
        default:
            throw TTSModelError.unsupportedModelType(resolvedType)
        }
    }

    private static func load<Model: SpeechGenerationModel>(
        _ source: ModelSource,
        modelType: String,
        pretrained: (String, HubCache) async throws -> Model,
        local: ((URL, String) async throws -> Model)? = nil
    ) async throws -> SpeechGenerationModel {
        switch source {
        case .repository(let modelRepo, let cache):
            return try await pretrained(modelRepo, cache)
        case .localDirectory(let modelDir, let repoHint):
            guard let local else {
                throw TTSModelError.unsupportedModelType("\(modelType) from local directory")
            }
            return try await local(modelDir, repoHint)
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

    private static func localModelDirectory(_ path: String) -> URL? {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path)
        else {
            return nil
        }
        return url
    }

    private static func localModelType(_ modelDir: URL) throws -> String? {
        let data = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return (config["model_type"] as? String)
            ?? (config["architecture"] as? String)
            ?? (config["model_version"] as? String)
    }

    private static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        // Repo names are hyphenated (e.g. "Irodori-TTS-600M-…"); match the bare name.
        if lower.contains("irodori") {
            return "irodori_tts"
        }
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
            if lower.contains("nano") {
                return "moss_tts_nano"
            }
            if lower.contains("local-transformer") || lower.contains("local_transformer") {
                return "moss_tts_local"
            }
            return "moss_tts_delay"
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
        if lower.contains("omnivoice") {
            return "omnivoice"
        }
        return nil
    }
}

@available(*, deprecated, renamed: "TTSModelError")
public typealias TTSModelUtilsError = TTSModelError

@available(*, deprecated, renamed: "TTS")
public typealias TTSModelUtils = TTS
