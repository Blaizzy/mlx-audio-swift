import Foundation

public struct TranscriptionOptions: Sendable {
    public var language: String?
    public var task: TranscriptionTask
    public var timeout: TimeInterval
    public var temperature: Float

    public enum TranscriptionTask: String, Sendable {
        case transcribe
        case translate
    }

    public init(
        language: String? = nil,
        task: TranscriptionTask = .transcribe,
        timeout: TimeInterval = 30.0,
        temperature: Float = 0.0
    ) {
        self.language = language
        self.task = task
        self.timeout = timeout
        self.temperature = temperature
    }

    public static let `default` = TranscriptionOptions()
}
