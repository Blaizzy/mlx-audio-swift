import Foundation

public enum WhisperError: LocalizedError, Sendable {
    case modelNotFound(String)
    case modelDownloadFailed(URL, underlying: Error)
    case invalidModelFormat(String)
    case invalidAudioFormat(expected: String, got: String)
    case audioTooShort(minSeconds: Double)
    case sampleRateMismatch(expected: Int, got: Int)
    case encodingFailed(String)
    case decodingFailed(String)
    case cancelled
    case timeout(TimeInterval)
    case tokenizerLoadFailed(String)
    case insufficientMemory(required: Int, available: Int)
    case quantizedModelNotAvailable(WhisperModel, WhisperQuantization)
    case modelNotReady
    case loadingFailed(underlying: Error)
    case loadingTimeout

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .modelDownloadFailed(let url, let error):
            return "Failed to download model from \(url): \(error.localizedDescription)"
        case .invalidModelFormat(let reason):
            return "Invalid model format: \(reason)"
        case .invalidAudioFormat(let expected, let got):
            return "Invalid audio format. Expected \(expected), got \(got)"
        case .audioTooShort(let min):
            return "Audio too short. Minimum duration: \(min) seconds"
        case .sampleRateMismatch(let expected, let got):
            return "Sample rate mismatch. Expected \(expected) Hz, got \(got) Hz"
        case .encodingFailed(let reason):
            return "Audio encoding failed: \(reason)"
        case .decodingFailed(let reason):
            return "Decoding failed: \(reason)"
        case .cancelled:
            return "Transcription was cancelled"
        case .timeout(let duration):
            return "Transcription timed out after \(duration) seconds"
        case .tokenizerLoadFailed(let reason):
            return "Failed to load tokenizer: \(reason)"
        case .insufficientMemory(let required, let available):
            return "Insufficient GPU memory. Required: \(required) MB, Available: \(available) MB"
        case .quantizedModelNotAvailable(let model, let quant):
            return "Quantized model \(model) with \(quant) not available on HuggingFace"
        case .modelNotReady:
            return "Model not ready. Call waitUntilReady() first"
        case .loadingFailed(let error):
            return "Model loading failed: \(error.localizedDescription)"
        case .loadingTimeout:
            return "Model loading timed out"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .modelNotFound:
            return "Check network connection and try again"
        case .modelDownloadFailed:
            return "Verify URL and retry, or use a local model path"
        case .sampleRateMismatch(let expected, _):
            return "Resample audio to \(expected) Hz before transcription"
        case .audioTooShort(let min):
            return "Provide audio at least \(min) seconds long"
        case .timeout:
            return "Try smaller audio chunks or increase timeout"
        case .insufficientMemory:
            return "Use a smaller model (tiny/base) or free GPU memory"
        case .quantizedModelNotAvailable:
            return "Try a different quantization level or use float16"
        case .modelNotReady:
            return "Call await session.waitUntilReady() before transcribing"
        case .loadingFailed:
            return "Check network connection and try again"
        case .loadingTimeout:
            return "Increase timeout or check system resources"
        default:
            return nil
        }
    }
}
