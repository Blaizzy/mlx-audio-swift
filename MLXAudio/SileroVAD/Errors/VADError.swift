import Foundation
import MLX

public enum VADError: LocalizedError, Sendable {
    // Initialization
    case weightsNotFound(path: String)
    case weightsCorrupted(reason: String)
    case modelInitializationFailed(description: String)

    // Audio format
    case invalidSampleRate(expected: Int, got: Int)
    case invalidChunkSize(expected: Int, got: Int)
    case invalidAudioShape(expected: String, got: [Int])
    case invalidDtype(expected: DType, got: DType)
    case audioOutOfRange(min: Float, max: Float)

    // Runtime
    case processingFailed(reason: String)
    case stateCorrupted

    public var errorDescription: String? {
        switch self {
        case .weightsNotFound(let path):
            return "Model weights not found at path: \(path)"
        case .weightsCorrupted(let reason):
            return "Model weights are corrupted: \(reason)"
        case .modelInitializationFailed(let description):
            return "Failed to initialize VAD model: \(description)"
        case .invalidSampleRate(let expected, let got):
            return "Invalid sample rate: expected \(expected) Hz, got \(got) Hz"
        case .invalidChunkSize(let expected, let got):
            return "Invalid chunk size: expected \(expected) samples, got \(got) samples"
        case .invalidAudioShape(let expected, let got):
            return "Invalid audio shape: expected \(expected), got \(got)"
        case .invalidDtype(let expected, let got):
            return "Invalid dtype: expected \(expected), got \(got)"
        case .audioOutOfRange(let min, let max):
            return "Audio samples out of range [-1.0, 1.0]: found values in [\(min), \(max)]"
        case .processingFailed(let reason):
            return "Audio processing failed: \(reason)"
        case .stateCorrupted:
            return "VAD internal state is corrupted"
        }
    }
}

extension VADError {
    public static func modelInitializationFailed(underlying: Error) -> VADError {
        .modelInitializationFailed(description: underlying.localizedDescription)
    }
}
