import Foundation
import MLX

extension MLXArray: @unchecked @retroactive Sendable {}

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
public struct SAMAudioBatch: @unchecked Sendable {
    public var audios: MLXArray
    public var sizes: MLXArray?
    public var wavSizes: MLXArray?
    public var descriptions: [String]?
    public var anchorIDs: MLXArray?
    public var anchorAlignment: MLXArray?
    public var audioPadMask: MLXArray?

    public init(
        audios: MLXArray,
        sizes: MLXArray? = nil,
        wavSizes: MLXArray? = nil,
        descriptions: [String]? = nil,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil,
        audioPadMask: MLXArray? = nil
    ) {
        self.audios = audios
        self.sizes = sizes
        self.wavSizes = wavSizes
        self.descriptions = descriptions
        self.anchorIDs = anchorIDs
        self.anchorAlignment = anchorAlignment
        self.audioPadMask = audioPadMask
    }
}

/// Full-audio separation output.
public struct SAMAudioSeparationResult: @unchecked Sendable {
    public var target: [MLXArray]
    public var residual: [MLXArray]
    public var noise: MLXArray?
    public var peakMemoryGB: Float?

    public init(
        target: [MLXArray],
        residual: [MLXArray],
        noise: MLXArray? = nil,
        peakMemoryGB: Float? = nil
    ) {
        self.target = target
        self.residual = residual
        self.noise = noise
        self.peakMemoryGB = peakMemoryGB
    }
}

/// Streaming chunk output.
public struct SAMAudioStreamingChunk: @unchecked Sendable {
    public var target: MLXArray
    public var residual: MLXArray
    public var chunkIndex: Int
    public var isLastChunk: Bool
    public var noise: MLXArray?
    public var peakMemoryGB: Float?

    public init(
        target: MLXArray,
        residual: MLXArray,
        chunkIndex: Int,
        isLastChunk: Bool,
        noise: MLXArray? = nil,
        peakMemoryGB: Float? = nil
    ) {
        self.target = target
        self.residual = residual
        self.chunkIndex = chunkIndex
        self.isLastChunk = isLastChunk
        self.noise = noise
        self.peakMemoryGB = peakMemoryGB
    }
}
