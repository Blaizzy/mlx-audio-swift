import Foundation
import MLX

/// ODE method used during SAMAudio generation.
public enum SAMAudioODEMethod: String, Codable, Sendable {
    case midpoint
    case euler
}

/// ODE solver options for SAMAudio inference.
public struct SAMAudioODEOptions: Codable, Sendable {
    public var method: SAMAudioODEMethod
    public var stepSize: Float

    enum CodingKeys: String, CodingKey {
        case method
        case stepSize = "step_size"
    }

    public init(method: SAMAudioODEMethod = .midpoint, stepSize: Float = 2.0 / 32.0) {
        self.method = method
        self.stepSize = stepSize
    }

    public static let `default` = SAMAudioODEOptions()
}

/// Temporal anchor tuple used to hint where a target event appears in audio.
public typealias SAMAudioAnchor = (token: String, startTime: Float, endTime: Float)

/// Preprocessed input batch for SAMAudio inference.
public struct SAMAudioBatch: Sendable {
    public var audios: MaterializedArray
    public var sizes: MaterializedArray?
    public var wavSizes: MaterializedArray?
    public var descriptions: [String]?
    public var anchorIDs: MaterializedArray?
    public var anchorAlignment: MaterializedArray?
    public var audioPadMask: MaterializedArray?

    public init(
        audios: MLXArray,
        sizes: MLXArray? = nil,
        wavSizes: MLXArray? = nil,
        descriptions: [String]? = nil,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil,
        audioPadMask: MLXArray? = nil
    ) {
        let optionalTensors = [sizes, wavSizes, anchorIDs, anchorAlignment, audioPadMask]
        var snapshots = materialize([audios] + optionalTensors.compactMap { $0 }).makeIterator()
        self.audios = snapshots.next()!
        self.sizes = sizes == nil ? nil : snapshots.next()!
        self.wavSizes = wavSizes == nil ? nil : snapshots.next()!
        self.descriptions = descriptions
        self.anchorIDs = anchorIDs == nil ? nil : snapshots.next()!
        self.anchorAlignment = anchorAlignment == nil ? nil : snapshots.next()!
        self.audioPadMask = audioPadMask == nil ? nil : snapshots.next()!
    }
}

/// Full-audio separation output.
public struct SAMAudioSeparationResult: Sendable {
    public var target: [MaterializedArray]
    public var residual: [MaterializedArray]
    public var noise: MaterializedArray?
    public var peakMemoryGB: Float?

    public init(
        target: [MLXArray],
        residual: [MLXArray],
        noise: MLXArray? = nil,
        peakMemoryGB: Float? = nil
    ) {
        var tensors = target + residual
        if let noise { tensors.append(noise) }
        let snapshots = materialize(tensors)
        self.target = Array(snapshots[..<target.count])
        self.residual = Array(snapshots[target.count..<(target.count + residual.count)])
        self.noise = noise == nil ? nil : snapshots.last
        self.peakMemoryGB = peakMemoryGB
    }
}

/// Streaming chunk output.
public struct SAMAudioStreamingChunk: Sendable {
    public var target: MaterializedArray
    public var residual: MaterializedArray
    public var chunkIndex: Int
    public var isLastChunk: Bool
    public var noise: MaterializedArray?
    public var peakMemoryGB: Float?

    public init(
        target: MLXArray,
        residual: MLXArray,
        chunkIndex: Int,
        isLastChunk: Bool,
        noise: MLXArray? = nil,
        peakMemoryGB: Float? = nil
    ) {
        var tensors = [target, residual]
        if let noise { tensors.append(noise) }
        let snapshots = materialize(tensors)
        self.target = snapshots[0]
        self.residual = snapshots[1]
        self.chunkIndex = chunkIndex
        self.isLastChunk = isLastChunk
        self.noise = noise == nil ? nil : snapshots[2]
        self.peakMemoryGB = peakMemoryGB
    }
}
