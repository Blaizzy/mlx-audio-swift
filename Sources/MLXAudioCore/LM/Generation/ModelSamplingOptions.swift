public struct ModelSamplingOptions: Sendable {
    public struct Penalty: Sendable {
        public var value: Float
        public var contextSize: Int
        public init(_ value: Float, contextSize: Int = 20) {
            self.value = value
            self.contextSize = contextSize
        }
    }
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var seed: UInt64?
    /// Masked after penalties and before sampling; invalid IDs or a fully masked vocabulary throw.
    public var suppressedTokenIDs: Set<Int>
    public var repetition: Penalty?
    public var presence: Penalty?
    public var frequency: Penalty?
    public init(
        temperature: Float = 0.6, topP: Float = 1, topK: Int = 0, minP: Float = 0,
        seed: UInt64? = nil, suppressedTokenIDs: Set<Int> = [], repetition: Penalty? = nil,
        presence: Penalty? = nil, frequency: Penalty? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.suppressedTokenIDs = suppressedTokenIDs
        self.repetition = repetition
        self.presence = presence
        self.frequency = frequency
    }
}
