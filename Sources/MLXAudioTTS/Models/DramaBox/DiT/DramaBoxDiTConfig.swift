import Foundation

public struct DramaBoxDiTConfig: Sendable {
    public var audioInChannels: Int
    public var audioOutChannels: Int
    public var audioNumAttentionHeads: Int
    public var audioAttentionHeadDim: Int
    public var audioCrossAttentionDim: Int
    public var audioPositionalEmbeddingMaxPos: Int
    public var numLayers: Int
    public var normEps: Float
    public var crossAttentionAdaln: Bool
    public var applyGatedAttention: Bool
    public var ropeType: String
    public var positionalEmbeddingTheta: Float
    public var timestepScaleMultiplier: Float
    public var useMiddleIndicesGrid: Bool

    public var audioInnerDim: Int {
        audioNumAttentionHeads * audioAttentionHeadDim
    }

    public init(
        audioInChannels: Int = 128,
        audioOutChannels: Int = 128,
        audioNumAttentionHeads: Int = 32,
        audioAttentionHeadDim: Int = 64,
        audioCrossAttentionDim: Int = 2048,
        audioPositionalEmbeddingMaxPos: Int = 20,
        numLayers: Int = 48,
        normEps: Float = 1e-6,
        crossAttentionAdaln: Bool = true,
        applyGatedAttention: Bool = true,
        ropeType: String = "split",
        positionalEmbeddingTheta: Float = 10_000,
        timestepScaleMultiplier: Float = 1000,
        useMiddleIndicesGrid: Bool = true
    ) {
        self.audioInChannels = audioInChannels
        self.audioOutChannels = audioOutChannels
        self.audioNumAttentionHeads = audioNumAttentionHeads
        self.audioAttentionHeadDim = audioAttentionHeadDim
        self.audioCrossAttentionDim = audioCrossAttentionDim
        self.audioPositionalEmbeddingMaxPos = audioPositionalEmbeddingMaxPos
        self.numLayers = numLayers
        self.normEps = normEps
        self.crossAttentionAdaln = crossAttentionAdaln
        self.applyGatedAttention = applyGatedAttention
        self.ropeType = ropeType
        self.positionalEmbeddingTheta = positionalEmbeddingTheta
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.useMiddleIndicesGrid = useMiddleIndicesGrid
    }
}
