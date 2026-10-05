import Foundation
import MLX

// MARK: - Diarization Segment

public struct DiarizationSegment: Sendable {
    public let start: Float
    public let end: Float
    public let speaker: Int

    public init(start: Float, end: Float, speaker: Int) {
        self.start = start
        self.end = end
        self.speaker = speaker
    }
}

// MARK: - Diarization Output

/// Owned diarization results. Async producers transfer the result with `sending`.
public struct DiarizationOutput {
    public let segments: [DiarizationSegment]
    public let speakerProbs: MLXArray?
    public let numSpeakers: Int
    public let totalTime: Double

    public init(
        segments: [DiarizationSegment],
        speakerProbs: MLXArray? = nil,
        numSpeakers: Int = 0,
        totalTime: Double = 0.0
    ) {
        self.segments = segments
        self.speakerProbs = speakerProbs
        self.numSpeakers = numSpeakers
        self.totalTime = totalTime
    }

    /// Format output as RTTM text.
    public var text: String {
        var lines = [String]()
        for seg in segments {
            let duration = seg.end - seg.start
            lines.append(
                "SPEAKER audio 1 \(String(format: "%.3f", seg.start)) \(String(format: "%.3f", duration)) <NA> <NA> speaker_\(seg.speaker) <NA> <NA>"
            )
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Streaming State

/// A single-use continuation for Sortformer streaming inference.
///
/// Pass this handle to `feed` or `streamingStep`, then save the returned handle
/// for the next chunk. Copies share the same consumption status; they are not
/// independent checkpoints. Scalar metadata remains readable after consumption.
public struct StreamingState: Sendable {
    /// The handle has already been used to continue inference.
    public enum ConsumptionError: Error, Equatable {
        case alreadyConsumed
    }

    public let framesProcessed: Int
    public let spkcacheLen: Int
    public let fifoLen: Int

    private let storage: Storage

    init(_ state: sending SortformerStreamingState) {
        framesProcessed = state.framesProcessed
        spkcacheLen = state.spkcacheLen
        fifoLen = state.fifoLen
        storage = Storage(state)
    }

    func take() throws -> sending SortformerStreamingState {
        try storage.take()
    }

    // Only this box crosses isolation boundaries. Its initializer requires
    // exclusive ownership, and its lock lets exactly one consumer take the
    // tensors. No tensor reference is exposed through the public handle.
    private final class Storage: @unchecked Sendable {
        private let lock = NSLock()
        private var state: SortformerStreamingState?

        init(_ state: sending SortformerStreamingState) {
            self.state = state
        }

        func take() throws -> sending SortformerStreamingState {
            lock.lock()
            defer { lock.unlock() }
            guard let state else { throw ConsumptionError.alreadyConsumed }
            self.state = nil
            return state
        }
    }
}

/// Tensor state used only within one inference operation.
struct SortformerStreamingState {
    var spkcache: MLXArray        // (1, cache_frames, emb_dim)
    var spkcachePreds: MLXArray   // (1, cache_frames, n_spk)
    var fifo: MLXArray            // (1, fifo_frames, emb_dim)
    var fifoPreds: MLXArray       // (1, fifo_frames, n_spk)
    var framesProcessed: Int
    var meanSilEmb: MLXArray      // (1, emb_dim) running mean silence embedding
    var nSilFrames: MLXArray      // (1,) count of silence frames seen

    var spkcacheLen: Int { spkcache.dim(1) }
    var fifoLen: Int { fifo.dim(1) }
}
