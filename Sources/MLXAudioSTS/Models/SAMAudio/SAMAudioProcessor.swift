import Foundation
import MLX
import MLXAudioCore

public enum SAMAudioProcessorError: Error, LocalizedError {
    case mismatchedBatchCounts
    case emptyAudioBatch

    public var errorDescription: String? {
        switch self {
        case .mismatchedBatchCounts:
            return "Descriptions, audio inputs, and anchors must have matching batch sizes."
        case .emptyAudioBatch:
            return "Audio batch is empty."
        }
    }
}

public enum SAMAudioProcessorAudioInput {
    case file(String)
    case array(MLXArray)
}

/// Input processor for SAMAudio prompts and waveforms.
public final class SAMAudioProcessor {
    public static let anchorDict: [String: Int] = [
        "<null>": 0,
        "+": 1,
        "-": 2,
        "<pad>": 3,
    ]

    public let audioHopLength: Int
    public let audioSamplingRate: Int

    public init(audioHopLength: Int, audioSamplingRate: Int = 48_000) {
        self.audioHopLength = audioHopLength
        self.audioSamplingRate = audioSamplingRate
    }

    public convenience init(config: SAMAudioConfig) {
        self.init(
            audioHopLength: config.audioCodec.hopLength,
            audioSamplingRate: config.audioCodec.sampleRate
        )
    }

    public func wavToFeatureIdx(_ wavIdx: Int) -> Int {
        Int(ceil(Float(wavIdx) / Float(audioHopLength)))
    }

    public func featureToWavIdx(_ featureIdx: Int) -> Int {
        featureIdx * audioHopLength
    }

    public static func maskFromSizes(_ sizes: MLXArray) -> MLXArray {
        let maxLen = Int(sizes.max().item(Int32.self))
        let batchSize = sizes.shape[0]
        let positions = MLXArray(Array(0..<maxLen), [1, maxLen])
        let expandedPositions = MLX.broadcast(positions, to: [batchSize, maxLen])
        let expandedSizes = sizes.expandedDimensions(axis: 1)
        return expandedPositions .< expandedSizes
    }

    private static func toMono(_ wav: MLXArray) -> MLXArray {
        if wav.ndim == 1 {
            return wav
        }
        if wav.ndim == 2 {
            if wav.shape[0] <= 2 {
                return wav.mean(axis: 0)
            }
            if wav.shape[1] <= 2 {
                return wav.mean(axis: 1)
            }
        }
        return wav.reshaped([-1])
    }

    private static func linearResample(_ wav: MLXArray, from sourceRate: Int, to targetRate: Int) -> MLXArray {
        if sourceRate == targetRate {
            return wav
        }

        let src = wav.asArray(Float.self)
        if src.count <= 1 {
            return wav
        }

        let newCount = max(1, Int(round(Double(src.count) * Double(targetRate) / Double(sourceRate))))
        if newCount == src.count {
            return wav
        }

        var out = [Float]()
        out.reserveCapacity(newCount)

        let denom = max(newCount - 1, 1)
        let srcMax = Float(src.count - 1)
        for i in 0..<newCount {
            let pos = Float(i) * srcMax / Float(denom)
            let lo = Int(floor(pos))
            let hi = min(lo + 1, src.count - 1)
            let frac = pos - Float(lo)
            out.append(src[lo] * (1 - frac) + src[hi] * frac)
        }
        return MLXArray(out)
    }

    public func batchAudio(_ audios: [SAMAudioProcessorAudioInput]) throws -> (MLXArray, MLXArray) {
        guard !audios.isEmpty else {
            throw SAMAudioProcessorError.emptyAudioBatch
        }

        var wavs: [MLXArray] = []
        wavs.reserveCapacity(audios.count)

        for audio in audios {
            let wav: MLXArray
            switch audio {
            case .array(let arr):
                wav = Self.toMono(arr)
            case .file(let path):
                let url = path.hasPrefix("/") ? URL(fileURLWithPath: path) :
                    URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(path)
                let (sampleRate, loaded) = try loadAudioArray(from: url)
                let mono = Self.toMono(loaded)
                wav = Self.linearResample(mono, from: sampleRate, to: audioSamplingRate)
            }
            wavs.append(wav)
        }

        let sizes = MLXArray(wavs.map { $0.shape[0] })
        let maxLen = Int(sizes.max().item(Int32.self))

        var batched: [MLXArray] = []
        batched.reserveCapacity(wavs.count)
        for wav in wavs {
            if wav.shape[0] < maxLen {
                let pad = maxLen - wav.shape[0]
                batched.append(MLX.padded(wav, widths: [.init((0, pad))]))
            } else {
                batched.append(wav)
            }
        }

        let stacked = MLX.stacked(batched, axis: 0).expandedDimensions(axis: 1) // (B, 1, T)
        return (stacked, sizes)
    }

    public func processAnchors(
        _ anchors: [[SAMAudioAnchor]]?,
        audioPadMask: MLXArray,
        batchSize: Int
    ) -> (anchorIDs: MLXArray, anchorAlignment: MLXArray) {
        let seqLen = audioPadMask.shape[1]

        let nullToken = Self.anchorDict["<null>"] ?? 0
        let padToken = Self.anchorDict["<pad>"] ?? 3

        if anchors == nil {
            let nullCol = MLXArray(Array(repeating: nullToken, count: batchSize), [batchSize, 1]).asType(.int32)
            let padCol = MLXArray(Array(repeating: padToken, count: batchSize), [batchSize, 1]).asType(.int32)
            let anchorIDs = MLX.concatenated([nullCol, padCol], axis: 1)

            var alignment = MLXArray.zeros([batchSize, seqLen]).asType(.int32)
            alignment = MLX.where(audioPadMask, alignment, MLXArray.ones([batchSize, seqLen]).asType(.int32))
            return (anchorIDs, alignment)
        }

        var alignment2D = Array(repeating: Array(repeating: 0, count: seqLen), count: batchSize)
        let padMask = audioPadMask.asArray(Bool.self)
        for b in 0..<batchSize {
            for t in 0..<seqLen {
                if !padMask[b * seqLen + t] {
                    alignment2D[b][t] = 1 // index of <pad> in anchorIDs row
                }
            }
        }

        var allIDs: [[Int]] = []
        for i in 0..<batchSize {
            let anchorList = anchors?[i] ?? []
            var current = [nullToken, padToken]
            for anchor in anchorList {
                let startIdx = wavToFeatureIdx(Int(anchor.startTime * Float(audioSamplingRate)))
                let endIdx = wavToFeatureIdx(Int(anchor.endTime * Float(audioSamplingRate)))
                let anchorIdx = current.count
                if startIdx < seqLen {
                    let upper = min(endIdx, seqLen)
                    if startIdx < upper {
                        for t in startIdx..<upper {
                            alignment2D[i][t] = anchorIdx
                        }
                    }
                }
                current.append(Self.anchorDict[anchor.token] ?? nullToken)
            }
            allIDs.append(current)
        }

        let maxAnchors = allIDs.map(\.count).max() ?? 2
        let paddedIDs = allIDs.map { row in
            row + Array(repeating: padToken, count: maxAnchors - row.count)
        }

        let anchorIDs = MLXArray(paddedIDs.flatMap { $0 }, [batchSize, maxAnchors]).asType(.int32)
        let anchorAlignment = MLXArray(alignment2D.flatMap { $0 }, [batchSize, seqLen]).asType(.int32)
        return (anchorIDs, anchorAlignment)
    }

    public func process(
        descriptions: [String],
        audios: [SAMAudioProcessorAudioInput],
        anchors: [[SAMAudioAnchor]]? = nil
    ) throws -> SAMAudioBatch {
        guard descriptions.count == audios.count else {
            throw SAMAudioProcessorError.mismatchedBatchCounts
        }
        if let anchors, anchors.count != descriptions.count {
            throw SAMAudioProcessorError.mismatchedBatchCounts
        }

        let (batchedAudios, wavSizes) = try batchAudio(audios)
        let wavSizesInts = wavSizes.asArray(Int32.self).map(Int.init)
        let featureSizes = MLXArray(wavSizesInts.map(wavToFeatureIdx))

        let audioPadMask = Self.maskFromSizes(featureSizes)
        let (anchorIDs, anchorAlignment) = processAnchors(
            anchors,
            audioPadMask: audioPadMask,
            batchSize: descriptions.count
        )

        return SAMAudioBatch(
            audios: batchedAudios,
            sizes: featureSizes,
            wavSizes: wavSizes,
            descriptions: descriptions,
            anchorIDs: anchorIDs,
            anchorAlignment: anchorAlignment,
            audioPadMask: audioPadMask
        )
    }
}
