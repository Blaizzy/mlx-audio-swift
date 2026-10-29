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
        case localDirectory(URL, repoHint: String, cache: HubCache)

        /// Where text processors keep their own downloads.
        var cache: HubCache {
            switch self {
            case .repository(_, let cache): cache
            case .localDirectory(_, _, let cache): cache
            }
        }

        var fallbackName: String {
            switch self {
            case .repository(let modelRepo, _):
                modelRepo
            case .localDirectory(let modelDir, let repoHint, _):
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
                source: .localDirectory(modelDir, repoHint: modelRepo, cache: cache),
                textProcessor: textProcessor,
                hfToken: hfToken
            )
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }

        let modelType: String?
        let cachedNestedConfig = cache.cacheDirectory.appendingPathComponent(
            "mlx-audio/" + modelRepo.replacingOccurrences(of: "/", with: "_") + "/model/config.json")
        if let data = try? Data(contentsOf: cachedNestedConfig), let nestedType = try nestedModelType(data) {
            modelType = nestedType
        } else {
            do {
                modelType = try await ModelUtils.resolveModelType(repoID: repoID, hfToken: hfToken, cache: cache)
            } catch {
                // Some complete packages keep metadata beside their language model.
                guard case let HTTPClientError.responseError(response, _) = error,
                      response.statusCode == 404 else { throw error }
                let client: HubClient
                if let hfToken, !hfToken.isEmpty {
                    client = HubClient(host: HubClient.defaultHost, bearerToken: hfToken, cache: cache)
                } else { client = HubClient(cache: cache) }
                let url = try await client.downloadFile(at: "model/config.json", from: repoID)
                guard let type = try nestedModelType(Data(contentsOf: url)) else { throw error }
                modelType = type
            }
        }
        return try await loadResolvedModel(
            modelType: modelType,
            source: .repository(modelRepo, cache: cache),
            textProcessor: textProcessor,
            hfToken: hfToken
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
                source: .localDirectory(modelDir, repoHint: modelRepo, cache: cache),
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
        textProcessor: TextProcessor?,
        hfToken: String? = nil
    ) async throws -> SpeechGenerationModel {
        var resolvedType = normalizedModelType(modelType) ?? inferModelType(from: source.fallbackName)
        if resolvedType == "qwen2", source.fallbackName.lowercased().contains("spark") {
            resolvedType = "spark"
        }
        if ["qwen3", "qwen", "llama", "qwen2"].contains(resolvedType ?? "") {
            let directory: URL
            switch source {
            case .localDirectory(let modelDir, _, _): directory = modelDir
            case .repository(let repo, let cache):
                guard let id = Repo.ID(rawValue: repo) else { throw TTSModelError.invalidRepositoryID(repo) }
                directory = try await ModelUtils.resolveOrDownloadModel(
                    repoID: id, requiredExtension: "safetensors", hfToken: hfToken, cache: cache)
            }
            if try localModelType(directory) == "tada" { resolvedType = "tada" }
            else if try isOuteTTS(directory) || source.fallbackName.lowercased().contains("outetts") {
                resolvedType = "outetts"
            }
        }
        guard let resolvedType else {
            throw TTSModelError.unsupportedModelType(modelType)
        }

        // Keep cases alphabetized by their first model type.
        switch resolvedType {
        case "arktts", "audio8":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await Audio8Model.fromPretrained($0, cache: $1) },
                local: { directory, _ in try await Audio8Model.fromModelDirectory(directory) }
            )
        case "bailingmm", "ming_omni_tts":
            throw TTSModelError.unsupportedModelType("Ming Bailing MoE backbone; this loader supports the dense Qwen2.5 variants only")
        case "bark":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await BarkModel.fromPretrained($0, cache: $1) },
                local: { folder, _ in try await BarkModel.fromModelDirectory(folder, cache: source.cache) }
            )
        case "breeze", "breeze_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await BreezeTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await BreezeTTSModel.fromModelDirectory(modelDir) }
            )
        case "chatterbox", "chatterbox_tts", "chatterbox_turbo":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { modelRepo, _ in try await ChatterboxModel.fromPretrained(modelRepo) },
                local: { modelDir, _ in try await ChatterboxModel.fromModelDirectory(modelDir, hfToken: nil) }
            )
        case "confucius4", "confucius4_tts":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await Confucius4Model.fromPretrained($0, cache: $1) },
                local: { directory, _ in try await Confucius4Model.fromModelDirectory(directory) })
        case "csm", "sesame":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await MarvisTTSModel.fromPretrained($0, cache: $1) }
            )
        case "dense", "ming_dense":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await MingDenseModel.fromPretrained($0, cache: $1) },
                local: { directory, _ in try await MingDenseModel.fromModelDirectory(directory) })
        case "dia":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await DiaTTSModel.fromPretrained($0, cache: $1) }
            )
        case "echo_tts", "echo":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await EchoTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await EchoTTSModel.fromModelDirectory(modelDir) }
            )
        case "fish_speech", "fish_qwen3_omni":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await FishSpeechModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await FishSpeechModel.fromModelDirectory(modelDir) }
            )
        case "higgs_audio", "higgs_audio_v2":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await HiggsV2Model.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await HiggsV2Model.fromModelDirectory(modelDir, cache: source.cache) }
            )
        case "higgs_audio_v3", "higgs_multimodal_qwen3":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await HiggsAudio3TTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await HiggsAudio3TTSModel.fromModelDirectory(modelDir) }
            )
        case "indextts", "index_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await IndexTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await IndexTTSModel.fromModelDirectory(modelDir) }
            )
        case "irodori_tts", "irodori":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await IrodoriTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await IrodoriTTSModel.fromModelDirectory(modelDir) }
            )
        case "kitten_tts", "kitten":
            let processor = textProcessor ?? MisakiTextProcessor(cache: source.cache)
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await KittenTTSModel.fromPretrained($0, textProcessor: processor, cache: $1) },
                local: { modelDir, _ in try await KittenTTSModel.fromModelDirectory(modelDir, textProcessor: processor) }
            )
        case "kokoro", "kokoro_tts":
            let processor = textProcessor ?? KokoroMultilingualProcessor(cache: source.cache)
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await KokoroModel.fromPretrained($0, textProcessor: processor, cache: $1) },
                local: { modelDir, _ in try await KokoroModel.fromModelDirectory(modelDir, textProcessor: processor) }
            )
        case "kugelaudio":
            return try await load(source, modelType: resolvedType,
                pretrained: { try await KugelAudioModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await KugelAudioModel.fromModelDirectory(modelDir) })
        case "lingv3_tts", "ming_tiny":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await MingTinyModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await MingTinyModel.fromModelDirectory(modelDir) }
            )
        case "llama_tts", "llama3_tts", "llama3", "llama", "orpheus", "orpheus_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await LlamaTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await LlamaTTSModel.fromModelDirectory(modelDir) }
            )
        case "longcat_audiodit", "audiodit":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await LongCatModel.fromPretrained($0, cache: $1) },
                local: { directory, _ in try await LongCatModel.fromModelDirectory(directory, cache: source.cache) }
            )
        case "melotts", "melo_tts":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await MeloTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await MeloTTSModel.fromModelDirectory(modelDir, cache: source.cache) }
            )
        case "moss_tts", "moss_tts_delay", "moss_tts_local":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await MossTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await MossTTSModel.fromModelDirectory(modelDir) }
            )
        case "moss_tts_nano":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await MossTTSNanoModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await MossTTSNanoModel.fromModelDirectory(modelDir) }
            )
        case "omnivoice":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await OmniVoiceModel.fromPretrained($0, cache: $1) }
            )
        case "outetts", "oute_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await OuteTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await OuteTTSModel.fromModelDirectory(modelDir, cache: source.cache) }
            )
        case "pocket_tts", "phonon":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await PocketTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await PocketTTSModel.fromModelDirectory(modelDir) }
            )
        case "qwen3", "qwen":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await Qwen3Model.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await Qwen3Model.fromModelDirectory(modelDir) }
            )
        case "qwen3_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await Qwen3TTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await Qwen3TTSModel.fromModelDirectory(modelDir) }
            )
        case "rumik_oss", "rumik":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await RumikModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await RumikModel.fromModelDirectory(modelDir) }
            )
        case "soprano_tts", "soprano":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await SopranoModel.fromPretrained($0, cache: $1) },
                local: { modelDir, repoHint in try await SopranoModel.fromModelDirectory(modelDir, repo: repoHint) }
            )
        case "spark", "spark_tts":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await SparkModel.fromPretrained($0, cache: $1) }
            )
        case "tada", "tada_tts":
            return try await load(source, modelType: resolvedType,
                pretrained: { try await TADAModel.fromPretrained($0, cache: $1, hfToken: hfToken) },
                local: { directory, _ in try await TADAModel.fromModelDirectory(directory, cache: source.cache, hfToken: hfToken) })
        case "vibevoice":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await VibeVoiceTTSModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await VibeVoiceTTSModel.fromModelDirectory(modelDir, cache: source.cache) }
            )
        case "vibevoice_streaming":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await VibeVoiceRealtimeModel.fromPretrained($0, cache: $1) },
                local: { directory, _ in try await VibeVoiceRealtimeModel.fromModelDirectory(directory) }
            )
        case "voxcpm", "vox_cpm", "voxcpm1", "voxcpm1.5":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await VoxCPMModel.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await VoxCPMModel.fromModelDirectory(modelDir) }
            )
        case "voxcpm2", "vox_cpm2":
            return try await load(
                source,
                modelType: resolvedType,
                pretrained: { try await VoxCPM2Model.fromPretrained($0, cache: $1) },
                local: { modelDir, _ in try await VoxCPM2Model.fromModelDirectory(modelDir, hfToken: nil) }
            )
        case "voxtral_tts":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await VoxtralTTSModel.fromPretrained($0, cache: $1) },
                local: { directory, _ in try await VoxtralTTSModel.fromModelDirectory(directory) }
            )
        case "zonos2":
            return try await load(
                source, modelType: resolvedType,
                pretrained: { try await Zonos2Model.fromPretrained($0, cache: $1) },
                local: { folder, _ in try await Zonos2Model.fromModelDirectory(folder, cache: source.cache) }
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
        case .localDirectory(let modelDir, let repoHint, _):
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
        if modelRepo.lowercased().contains("outetts"),
           modelType == nil || ["qwen3", "qwen2", "llama", "qwen"].contains(normalizedModelType(modelType) ?? "") {
            return "outetts"
        }
        return normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
    }

    static func localModelDirectory(_ path: String) -> URL? {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              (FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) || FileManager.default.fileExists(atPath: url.appendingPathComponent("model/config.json").path))
        else {
            return nil
        }
        return url
    }

    static func localModelType(_ modelDir: URL) throws -> String? {
        let nested = modelDir.appendingPathComponent("model/config.json")
        if FileManager.default.fileExists(atPath: nested.path),
           let type = try nestedModelType(Data(contentsOf: nested)) { return type }
        let data = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let type = try nestedModelType(data) { return type }
        if try isOuteTTS(modelDir) { return "outetts" }
        return (config["model_type"] as? String)
            ?? (config["architecture"] as? String)
            ?? (config["model_version"] as? String)
            ?? (config["flow_lm"] != nil && config["mimi"] != nil ? "pocket_tts" : nil)
    }

    static func nestedModelType(_ data: Data) throws -> String? {
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let architectures = object?["architectures"] as? [String] ?? []
        return architectures.contains("TadaForCausalLM") ? "tada" : nil
    }

    private static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        if lower.contains("zonos2") { return "zonos2" }
        if lower.contains("ming-3.0-tts-tiny") { return "lingv3_tts" }
        if lower.contains("ming"), lower.contains("tts"), lower.contains("0.5b") {
            return "dense"
        }
        if lower.contains("tada") { return "tada" }
        if lower.contains("confucius4") { return "confucius4" }
        if lower.contains("melotts") { return "melotts" }
        if lower.contains("kugelaudio") { return "kugelaudio" }
        if lower.contains("vibevoice") && lower.contains("realtime") {
            return "vibevoice_streaming"
        }
        if lower.contains("longcat") && lower.contains("audiodit") { return "longcat_audiodit" }
        if lower.contains("audio8") || lower.contains("arktts") { return "arktts" }
        if lower.contains("outetts") { return "outetts" }
        if lower.contains("breeze") && lower.contains("tts") {
            return "breeze"
        }
        // Repo names are hyphenated (e.g. "Irodori-TTS-600M-…"); match the bare name.
        if lower.contains("rumik") { return "rumik_oss" }
        if lower.contains("spark") {
            return "spark"
        }
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
        if lower.contains("bark") { return "bark" }
        if lower.contains("soprano") {
            return "soprano"
        }
        if lower.contains("llama") || lower.contains("orpheus") {
            return "llama_tts"
        }
        if lower.contains("csm") || lower.contains("sesame") {
            return "csm"
        }
        if lower.contains("pocket_tts") || lower.contains("pocket-tts") || lower.contains("phonon") {
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
        if lower.contains("voxtral") && lower.contains("tts") { return "voxtral_tts" }
        if lower.contains("voxcpm2") || lower.contains("vox_cpm2") {
            return "voxcpm2"
        }
        if lower.contains("voxcpm") || lower.contains("vox_cpm") {
            return "voxcpm"
        }
        if lower.contains("omnivoice") {
            return "omnivoice"
        }
        if lower.contains("indextts") || lower.contains("index-tts") || lower.contains("index_tts") {
            return "indextts"
        }
        if lower.contains("nari-labs") || lower.contains("nari_labs")
            || lower.contains("dia-1.6b") || lower.contains("dia-tts") || lower.contains("dia_tts")
        {
            return "dia"
        }
        return nil
    }

    /// Oute checkpoints advertise their language backbone as model_type. Their
    /// paired audio-token vocabulary identifies them even in renamed snapshots.
    static func isOuteTTS(_ directory: URL) throws -> Bool {
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
        if (config?["model_type"] as? String) == "outetts" { return true }
        if ["tokenizer_name", "_name_or_path"].contains(where: {
            (config?[$0] as? String)?.lowercased().contains("outetts") == true
        }) { return true }
        for filename in ["added_tokens.json", "tokenizer_config.json"] {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent(filename)),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            let tokens: Set<String>
            if filename == "added_tokens.json" {
                tokens = Set(object.keys)
            } else {
                let decoder = object["added_tokens_decoder"] as? [String: [String: Any]] ?? [:]
                tokens = Set(decoder.values.compactMap { $0["content"] as? String })
            }
            if ["<|c1_0|>", "<|c2_0|>", "<|word_start|>", "<|features|>"].allSatisfy(tokens.contains) {
                return true
            }
        }
        return false
    }
}

@available(*, deprecated, renamed: "TTSModelError")
public typealias TTSModelUtilsError = TTSModelError

@available(*, deprecated, renamed: "TTS")
public typealias TTSModelUtils = TTS
