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

    public var model: any STSModel {
        switch self {
        case .samAudio(let m): return m
        case .lfmAudio(let m): return m
        case .mossFormer2SE(let m): return m
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
        strict: Bool = false
    ) async throws -> LoadedSTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw STSModelError.invalidRepositoryID(modelRepo)
        }

        let modelType = try await ModelUtils.resolveModelType(repoID: repoID, hfToken: hfToken)
        return try await loadModel(
            modelRepo: modelRepo,
            modelType: modelType,
            hfToken: hfToken,
            strict: strict
        )
    }

    public static func loadModel(
        modelRepo: String,
        modelType: String?,
        hfToken: String? = nil,
        strict: Bool = false
    ) async throws -> LoadedSTSModel {
        let resolved = normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
        guard let resolved else {
            throw STSModelError.unsupportedModelType(modelType)
        }

        switch resolved {
        case "lfm_audio", "lfm", "lfm2", "lfm2_audio":
            let model = try await LFM2AudioModel.fromPretrained(modelRepo)
            return .lfmAudio(model)

        case "sam_audio", "sam", "samaudio":
            let model = try await SAMAudio.fromPretrained(modelRepo, hfToken: hfToken, strict: strict)
            return .samAudio(model)

        case "mossformer2_se", "mossformer2", "mossformer":
            let model = try await MossFormer2SEModel.fromPretrained(modelRepo)
            return .mossFormer2SE(model)

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
        if lower.contains("sam") || lower.contains("source-separation") {
            return "sam_audio"
        }
        return nil
    }
}
