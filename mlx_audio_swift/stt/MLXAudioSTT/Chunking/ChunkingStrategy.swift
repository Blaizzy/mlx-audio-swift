import Foundation
import MLX

/// Protocol for long audio chunking strategies
public protocol ChunkingStrategy: Sendable {
    /// Process long audio and yield results as chunks complete
    func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error>

    /// Strategy identifier for logging/debugging
    var name: String { get }

    /// Transcription mode (affects context handling)
    var transcriptionMode: TranscriptionMode { get }
}

/// Error thrown when an operation exceeds its timeout
public struct TimeoutError: Error, Sendable {
    public let timeout: TimeInterval

    public init(timeout: TimeInterval) {
        self.timeout = timeout
    }
}

extension TimeoutError: LocalizedError {
    public var errorDescription: String? {
        "Operation timed out after \(String(format: "%.1f", timeout)) seconds"
    }
}

/// Execute an async operation with a timeout
public func withTimeout<T: Sendable>(
    _ timeout: TimeInterval,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }

        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            throw TimeoutError(timeout: timeout)
        }

        guard let result = try await group.next() else {
            throw TimeoutError(timeout: timeout)
        }

        group.cancelAll()
        return result
    }
}
