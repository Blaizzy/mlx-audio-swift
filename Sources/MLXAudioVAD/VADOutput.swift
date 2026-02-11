import Foundation
@preconcurrency import MLX

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

public struct DiarizationOutput: Sendable {
    public let segments: [DiarizationSegment]
    public let speakerProbs: MLXArray?
    public let numSpeakers: Int
    public let totalTime: Double
    public var state: StreamingState?

    public init(
        segments: [DiarizationSegment],
        speakerProbs: MLXArray? = nil,
        numSpeakers: Int = 0,
        totalTime: Double = 0.0,
        state: StreamingState? = nil
    ) {
        self.segments = segments
        self.speakerProbs = speakerProbs
        self.numSpeakers = numSpeakers
        self.totalTime = totalTime
        self.state = state
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

public struct StreamingState: Sendable {
    public var spkcache: MLXArray        // (1, cache_frames, emb_dim)
    public var spkcachePreds: MLXArray   // (1, cache_frames, n_spk)
    public var fifo: MLXArray            // (1, fifo_frames, emb_dim)
    public var fifoPreds: MLXArray       // (1, fifo_frames, n_spk)
    public var framesProcessed: Int      // total diarization frames emitted
    public var meanSilEmb: MLXArray      // (1, emb_dim) running mean silence embedding
    public var nSilFrames: MLXArray      // (1,) count of silence frames seen

    public var spkcacheLen: Int { spkcache.dim(1) }
    public var fifoLen: Int { fifo.dim(1) }

    public init(
        spkcache: MLXArray,
        spkcachePreds: MLXArray,
        fifo: MLXArray,
        fifoPreds: MLXArray,
        framesProcessed: Int,
        meanSilEmb: MLXArray,
        nSilFrames: MLXArray
    ) {
        self.spkcache = spkcache
        self.spkcachePreds = spkcachePreds
        self.fifo = fifo
        self.fifoPreds = fifoPreds
        self.framesProcessed = framesProcessed
        self.meanSilEmb = meanSilEmb
        self.nSilFrames = nSilFrames
    }
}
