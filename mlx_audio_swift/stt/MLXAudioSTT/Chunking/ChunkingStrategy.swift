import Foundation
import MLX

/// Streaming result from chunking strategy - includes both partial and chunk-complete results
public struct ChunkStreamingResult: Sendable {
    public let text: String
    public let timestamp: ClosedRange<TimeInterval>
    public let isPartial: Bool
    public let isChunkFinal: Bool
    public let chunkIndex: Int

    public init(
        text: String,
        timestamp: ClosedRange<TimeInterval>,
        isPartial: Bool,
        isChunkFinal: Bool,
        chunkIndex: Int
    ) {
        self.text = text
        self.timestamp = timestamp
        self.isPartial = isPartial
        self.isChunkFinal = isChunkFinal
        self.chunkIndex = chunkIndex
    }
}

/// Protocol for long audio chunking strategies
public protocol ChunkingStrategy: Sendable {
    /// Process long audio and yield results as chunks complete (blocking per chunk)
    func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error>

    /// Process long audio with streaming - yields partial results as tokens are decoded
    func processStreaming(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkStreamingResult, Error>

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
