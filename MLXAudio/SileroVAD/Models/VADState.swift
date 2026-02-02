import MLX

/// LSTM state container for VAD streaming.
/// - Note: Not thread-safe. Each VADIterator should own its own state.
///   For concurrent streams, create separate VADIterator instances.
public struct VADState: @unchecked Sendable {
    public var hidden: MLXArray
    public var cell: MLXArray

    public init(hidden: MLXArray, cell: MLXArray) {
        self.hidden = hidden
        self.cell = cell
    }

    public static func initial(hiddenSize: Int = 128) -> VADState {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        return VADState(
            hidden: MLXArray.zeros([1, hiddenSize]),
            cell: MLXArray.zeros([1, hiddenSize])
        )
    }

    public mutating func reset() {
        let defaultHiddenSize = 128
        guard hidden.shape.count >= 2, hidden.shape[1] > 0 else {
            hidden = MLXArray.zeros([1, defaultHiddenSize])
            cell = MLXArray.zeros([1, defaultHiddenSize])
            return
        }
        let hiddenSize = hidden.shape[1]
        hidden = MLXArray.zeros([1, hiddenSize])
        cell = MLXArray.zeros([1, hiddenSize])
    }
}
