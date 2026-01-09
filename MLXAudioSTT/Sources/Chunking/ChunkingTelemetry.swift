import Foundation

/// Protocol for observability and debugging of chunking operations
public protocol ChunkingTelemetry: Sendable {
    func strategyStarted(_ strategyName: String, audioDuration: TimeInterval)
    func chunkStarted(index: Int, timeRange: ClosedRange<TimeInterval>)
    func chunkCompleted(index: Int, duration: TimeInterval, text: String)
    func chunkFailed(index: Int, error: Error)
    func vadSegmentsDetected(count: Int, totalSpeechDuration: TimeInterval)
    func strategyCompleted(totalChunks: Int, totalDuration: TimeInterval)
    func error(_ error: Error)
}

/// Default implementation that logs to console
public final class ConsoleTelemetry: ChunkingTelemetry, @unchecked Sendable {
    public init() {}

    public func strategyStarted(_ strategyName: String, audioDuration: TimeInterval) {
        print("[Chunking] Strategy '\(strategyName)' started for \(String(format: "%.1f", audioDuration))s audio")
    }

    public func chunkStarted(index: Int, timeRange: ClosedRange<TimeInterval>) {
        print("[Chunking] Chunk \(index) started: \(String(format: "%.1f", timeRange.lowerBound))-\(String(format: "%.1f", timeRange.upperBound))s")
    }

    public func chunkCompleted(index: Int, duration: TimeInterval, text: String) {
        let preview = text.count > 50 ? String(text.prefix(50)) + "..." : text
        print("[Chunking] Chunk \(index) completed in \(String(format: "%.2f", duration))s: \"\(preview)\"")
    }

    public func chunkFailed(index: Int, error: Error) {
        print("[Chunking] Chunk \(index) failed: \(error.localizedDescription)")
    }

    public func vadSegmentsDetected(count: Int, totalSpeechDuration: TimeInterval) {
        print("[Chunking] VAD detected \(count) segments, total speech: \(String(format: "%.1f", totalSpeechDuration))s")
    }

    public func strategyCompleted(totalChunks: Int, totalDuration: TimeInterval) {
        print("[Chunking] Strategy completed: \(totalChunks) chunks in \(String(format: "%.2f", totalDuration))s")
    }

    public func error(_ error: Error) {
        print("[Chunking] Error: \(error.localizedDescription)")
    }
}

/// Silent telemetry that does nothing (for production/testing)
public final class NoOpTelemetry: ChunkingTelemetry, @unchecked Sendable {
    public init() {}

    public func strategyStarted(_ strategyName: String, audioDuration: TimeInterval) {}
    public func chunkStarted(index: Int, timeRange: ClosedRange<TimeInterval>) {}
    public func chunkCompleted(index: Int, duration: TimeInterval, text: String) {}
    public func chunkFailed(index: Int, error: Error) {}
    public func vadSegmentsDetected(count: Int, totalSpeechDuration: TimeInterval) {}
    public func strategyCompleted(totalChunks: Int, totalDuration: TimeInterval) {}
    public func error(_ error: Error) {}
}
