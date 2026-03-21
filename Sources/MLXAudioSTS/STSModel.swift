import Foundation
import HuggingFace
import MLX
import MLXAudioCore

// MARK: - STSModel Protocol

public protocol STSModel: AnyObject {
    var sampleRate: Int { get }
}

// MARK: - Loaded Model Container

public enum LoadedSTSModel {
    case samAudio(SAMAudio)
    case lfmAudio(LFM2AudioModel)
    case mossFormer2SE(MossFormer2SEModel)
    case deepFilterNet(DeepFilterNetModel)

    public var model: any STSModel {
        switch self {
        case .samAudio(let m): return m
        case .lfmAudio(let m): return m
        case .mossFormer2SE(let m): return m
        case .deepFilterNet(let m): return m
        }
    }

    public var sampleRate: Int { model.sampleRate }
}

// MARK: - Errors

public enum STSModelError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepositoryID(String)
    case unsupportedModelType(String?)

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalidRepositoryID(let repo):
            return "Invalid repository ID: \(repo)"
        case .unsupportedModelType(let modelType):
            return "Unsupported STS model type: \(String(describing: modelType))"
        }
    }
}

// MARK: - Factory

public enum STS {

    public static func loadModel(
        modelRepo: String,
        hfToken: String? = nil,
        strict: Bool = false,
        cache: HubCache = .default
    ) async throws -> LoadedSTSModel {
        // Local directory path support (primarily for enhancement models).
        let localURL = URL(fileURLWithPath: modelRepo).standardizedFileURL
        if FileManager.default.fileExists(atPath: localURL.path) {
            let modelType = try resolveLocalModelType(localURL)
            return try await loadModel(
                modelRepo: modelRepo,
                modelType: modelType,
                hfToken: hfToken,
                strict: strict,
                cache: cache
            )
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw STSModelError.invalidRepositoryID(modelRepo)
        }

        let modelType = try await ModelUtils.resolveModelType(
            repoID: repoID,
            hfToken: hfToken,
            cache: cache
        )
        return try await loadModel(
            modelRepo: modelRepo,
            modelType: modelType,
            hfToken: hfToken,
            strict: strict,
            cache: cache
        )
    }

    public static func loadModel(
        modelRepo: String,
        modelType: String?,
        hfToken: String? = nil,
        strict: Bool = false,
        cache: HubCache = .default
    ) async throws -> LoadedSTSModel {
        let resolved = normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
        guard let resolved else {
            throw STSModelError.unsupportedModelType(modelType)
        }

        switch resolved {
        case "lfm_audio", "lfm", "lfm2", "lfm2_audio":
            let model = try await LFM2AudioModel.fromPretrained(modelRepo, cache: cache)
            return .lfmAudio(model)

        case "sam_audio", "sam", "samaudio":
            let model = try await SAMAudio.fromPretrained(
                modelRepo,
                hfToken: hfToken,
                strict: strict,
                cache: cache
            )
            return .samAudio(model)

        case "mossformer2_se", "mossformer2", "mossformer":
            let model = try await MossFormer2SEModel.fromPretrained(modelRepo, cache: cache)
            return .mossFormer2SE(model)

        case "deepfilternet", "deepfilternet2", "deepfilternet3", "dfn":
            let model = try await DeepFilterNetModel.fromPretrained(
                modelRepo,
                hfToken: hfToken,
                cache: cache
            )
            return .deepFilterNet(model)

        default:
            throw STSModelError.unsupportedModelType(resolved)
        }
    }

    // MARK: - Private

    private static func normalizedModelType(_ modelType: String?) -> String? {
        guard let modelType else { return nil }
        let trimmed = modelType.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed.lowercased()
    }

    private static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        if lower.contains("lfm") {
            return "lfm_audio"
        }
        if lower.contains("mossformer") {
            return "mossformer2_se"
        }
        if lower.contains("deepfilter") || lower.contains("dfn") {
            return "deepfilternet3"
        }
        if lower.contains("sam") || lower.contains("source-separation") {
            return "sam_audio"
        }
        return nil
    }

    private static func resolveLocalModelType(_ path: URL) throws -> String? {
        let configURL: URL
        if path.hasDirectoryPath {
            configURL = path.appendingPathComponent("config.json")
        } else if path.lastPathComponent == "config.json" {
            configURL = path
        } else {
            configURL = path.deletingLastPathComponent().appendingPathComponent("config.json")
        }
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            return inferModelType(from: path.path)
        }
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return inferModelType(from: path.path)
        }
        if let modelType = json["model_type"] as? String {
            return modelType.lowercased()
        }
        if let arch = json["architecture"] as? String {
            return arch.lowercased()
        }
        if let version = json["model_version"] as? String {
            return version.lowercased()
        }
        return inferModelType(from: path.path)
    }
}
