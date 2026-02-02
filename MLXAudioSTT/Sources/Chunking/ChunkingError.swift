import Foundation

/// Errors from chunking and long audio processing
public enum ChunkingError: Error, Sendable {
    // VAD errors
    case vadFailed(underlying: Error)
    case vadModelLoadFailed(String)

    // Chunk processing errors
    case chunkTranscriptionFailed(chunkIndex: Int, timeRange: ClosedRange<TimeInterval>, underlying: Error)
    case chunkTimeout(chunkIndex: Int, timeRange: ClosedRange<TimeInterval>)

    // Resource errors
    case resourceExhausted(ResourceType)
    case totalTimeoutExceeded(processedDuration: TimeInterval, totalDuration: TimeInterval)

    // Input validation
    case audioTooShort(minimum: TimeInterval, actual: TimeInterval)
    case invalidSampleRate(expected: Int, got: Int)

    // Cancellation
    case cancelled(partialResult: PartialTranscriptionResult?)

    public enum ResourceType: Sendable, Equatable {
        case memory(requestedMB: Int, availableMB: Int)
        case concurrency(requested: Int, limit: Int)
    }
}

extension ChunkingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .vadFailed(let underlying):
            return "VAD processing failed: \(underlying.localizedDescription)"
        case .vadModelLoadFailed(let message):
            return "Failed to load VAD model: \(message)"
        case .chunkTranscriptionFailed(let index, let timeRange, let underlying):
            return "Chunk \(index) (\(String(format: "%.1f", timeRange.lowerBound))-\(String(format: "%.1f", timeRange.upperBound))s) transcription failed: \(underlying.localizedDescription)"
        case .chunkTimeout(let index, let timeRange):
            return "Chunk \(index) (\(String(format: "%.1f", timeRange.lowerBound))-\(String(format: "%.1f", timeRange.upperBound))s) timed out"
        case .resourceExhausted(let resourceType):
            switch resourceType {
            case .memory(let requested, let available):
                return "Memory exhausted: requested \(requested)MB, available \(available)MB"
            case .concurrency(let requested, let limit):
                return "Concurrency limit exceeded: requested \(requested), limit \(limit)"
            }
        case .totalTimeoutExceeded(let processed, let total):
            return "Total timeout exceeded: processed \(String(format: "%.1f", processed))s of \(String(format: "%.1f", total))s"
        case .audioTooShort(let minimum, let actual):
            return "Audio too short: minimum \(String(format: "%.1f", minimum))s, actual \(String(format: "%.1f", actual))s"
        case .invalidSampleRate(let expected, let got):
            return "Invalid sample rate: expected \(expected), got \(got)"
        case .cancelled(let partial):
            if let partial = partial {
                return "Cancelled with partial result: \(partial.completedChunks)/\(partial.totalChunks) chunks (\(String(format: "%.1f", partial.processedDuration))s processed)"
            }
            return "Cancelled"
        }
    }
}

extension ChunkingError: Equatable {
    public static func == (lhs: ChunkingError, rhs: ChunkingError) -> Bool {
        switch (lhs, rhs) {
        case (.vadModelLoadFailed(let l), .vadModelLoadFailed(let r)):
            return l == r
        case (.chunkTimeout(let li, let lr), .chunkTimeout(let ri, let rr)):
            return li == ri && lr == rr
        case (.resourceExhausted(let l), .resourceExhausted(let r)):
            return l == r
        case (.totalTimeoutExceeded(let lp, let lt), .totalTimeoutExceeded(let rp, let rt)):
            return lp == rp && lt == rt
        case (.audioTooShort(let lm, let la), .audioTooShort(let rm, let ra)):
            return lm == rm && la == ra
        case (.invalidSampleRate(let le, let lg), .invalidSampleRate(let re, let rg)):
            return le == re && lg == rg
        default:
            return false
        }
    }
}
