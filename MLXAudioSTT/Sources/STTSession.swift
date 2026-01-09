import Foundation
import MLX

public protocol STTSession: Sendable {
    func transcribe(
        _ audio: MLXArray,
        sampleRate: Int
    ) -> AsyncThrowingStream<StreamingResult, Error>

    func transcribe(
        _ audio: MLXArray,
        sampleRate: Int
    ) async throws -> String
}
