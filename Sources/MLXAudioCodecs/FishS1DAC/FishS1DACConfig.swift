import Foundation

public struct FishS1DACModelArgs {
    public var blockSize: Int = 2048
    public var nLayer: Int = 8
    public var nHead: Int = 8
    public var dim: Int = 512
    public var intermediateSize: Int = 1536
    public var nLocalHeads: Int = -1
    public var headDim: Int = 64
    public var ropeBase: Double = 10_000
    public var normEps: Float = 1e-5
    public var channelsFirst: Bool = true
    public var posEmbedType: String = "rope"

    public init(
        blockSize: Int = 2048,
        nLayer: Int = 8,
        nHead: Int = 8,
        dim: Int = 512,
        intermediateSize: Int = 1536,
        nLocalHeads: Int = -1,
        headDim: Int = 64,
        ropeBase: Double = 10_000,
        normEps: Float = 1e-5,
        channelsFirst: Bool = true,
        posEmbedType: String = "rope"
    ) {
        self.blockSize = blockSize
        self.nLayer = nLayer
        self.nHead = nHead
        self.dim = dim
        self.intermediateSize = intermediateSize
        self.nLocalHeads = nLocalHeads == -1 ? nHead : nLocalHeads
        self.headDim = headDim
        self.ropeBase = ropeBase
        self.normEps = normEps
        self.channelsFirst = channelsFirst
        self.posEmbedType = posEmbedType
    }
}

public struct FishS1DACBuildConfig: Codable {
    public var encoderDim: Int = 64
    public var encoderRates: [Int] = [2, 4, 8, 8]
    public var latentDim: Int = 1024
    public var decoderDim: Int = 1536
    public var decoderRates: [Int] = [8, 8, 4, 2]
    public var nCodebooks: Int = 9
    public var codebookSize: Int = 1024
    public var codebookDim: Int = 8
    public var semanticCodebookSize: Int = 4096
    public var quantizerDropout: Float = 0.5
    public var downsampleFactor: [Int] = [2, 2]
    public var downsampleDims: [Int]? = nil
    public var sampleRate: Int = 44_100
    public var causal: Bool = true
    public var encoderTransformerLayers: [Int] = [0, 0, 0, 4]
    public var decoderTransformerLayers: [Int] = [4, 0, 0, 0]
    public var quantizerTransformerBlockSize: Int = 4096
    public var quantizerTransformerLayers: Int = 8
    public var quantizerTransformerHeads: Int = 16
    public var quantizerTransformerDim: Int = 1024
    public var quantizerTransformerIntermediateSize: Int = 3072
    public var quantizerTransformerHeadDim: Int = 64
    public var quantizerWindowSize: Int = 128
    public var transformerBlockSize: Int = 16_384
    public var transformerHeadDim: Int = 64
    public var transformerNormEps: Float = 1e-5
    public var transformerRopeBase: Double = 10_000

    public init() {}
}
