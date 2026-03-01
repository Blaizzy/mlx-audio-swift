import Foundation

public struct LanguagePrediction: Sendable {
    public let language: String
    public let confidence: Float
}

public struct LIDOutput: Sendable {
    public let language: String
    public let confidence: Float
    public let topLanguages: [LanguagePrediction]
}

public enum LIDError: Error, LocalizedError, Sendable {
    case invalidRepoID(String)
    case configNotFound
    case weightsNotFound
    case noLabels

    public var errorDescription: String? {
        switch self {
        case .invalidRepoID(let id): "Invalid HuggingFace repository ID: \(id)"
        case .configNotFound: "config.json not found in model directory"
        case .weightsNotFound: "No .safetensors files found in model directory"
        case .noLabels: "No id2label mapping found in config"
        }
    }
}
