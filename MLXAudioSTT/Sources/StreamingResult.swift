import Foundation

public struct StreamingResult: Sendable {
    public let text: String
    public let isFinal: Bool
    public let timestamp: ClosedRange<TimeInterval>

    public init(text: String, isFinal: Bool, timestamp: ClosedRange<TimeInterval>) {
        self.text = text
        self.isFinal = isFinal
        self.timestamp = timestamp
    }
}
