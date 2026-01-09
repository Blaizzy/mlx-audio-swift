import Foundation

public struct StreamingConfig: Sendable {
    public var frameThreshold: Int
    public var minChunkDuration: TimeInterval
    public var emitPartial: Bool

    public init(
        frameThreshold: Int = 25,
        minChunkDuration: TimeInterval = 0.5,
        emitPartial: Bool = true
    ) {
        self.frameThreshold = frameThreshold
        self.minChunkDuration = minChunkDuration
        self.emitPartial = emitPartial
    }

    public static let `default` = StreamingConfig()
}
