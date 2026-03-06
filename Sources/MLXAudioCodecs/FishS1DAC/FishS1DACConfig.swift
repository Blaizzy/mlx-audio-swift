import Foundation

public struct FishS1DACConfig: Decodable {
    // Transformer args
    public var blockSize: Int
    public var nLayer: Int
    public var nHead: Int
    public var dim: Int
    public var intermediateSize: Int
    public var nLocalHeads: Int
    public var headDim: Int
    public var ropeBase: Float
    public var normEps: Float

    // Architecture
    public var sampleRate: Int
    public var hopLength: Int
    public var latentDim: Int
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var encoderTransformerLayers: [Int]
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var codebookSize: Int
    public var semanticCodebookSize: Int
    public var codebookDim: Int
    public var nCodebooks: Int
    public var downsampleFactor: [Int]
    public var quantizerWindowSize: Int
    public var quantizerNLayer: Int
    public var quantizerNHead: Int
    public var quantizerDim: Int
    public var quantizerIntermediateSize: Int

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case dim
        case intermediateSize = "intermediate_size"
        case nLocalHeads = "n_local_heads"
        case headDim = "head_dim"
        case ropeBase = "rope_base"
        case normEps = "norm_eps"
        case sampleRate = "sample_rate"
        case hopLength = "hop_length"
        case latentDim = "latent_dim"
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case encoderTransformerLayers = "encoder_transformer_layers"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case codebookSize = "codebook_size"
        case semanticCodebookSize = "semantic_codebook_size"
        case codebookDim = "codebook_dim"
        case nCodebooks = "n_codebooks"
        case downsampleFactor = "downsample_factor"
        case quantizerWindowSize = "quantizer_window_size"
        case quantizerNLayer = "quantizer_n_layer"
        case quantizerNHead = "quantizer_n_head"
        case quantizerDim = "quantizer_dim"
        case quantizerIntermediateSize = "quantizer_intermediate_size"
    }

    public init(from jsonDecoder: any Swift.Decoder) throws {
        let c = try jsonDecoder.container(keyedBy: CodingKeys.self)

        self.blockSize = (try? c.decode(Int.self, forKey: CodingKeys.blockSize)) ?? 2048
        self.nLayer = (try? c.decode(Int.self, forKey: CodingKeys.nLayer)) ?? 8
        self.nHead = (try? c.decode(Int.self, forKey: CodingKeys.nHead)) ?? 8
        self.dim = (try? c.decode(Int.self, forKey: CodingKeys.dim)) ?? 512
        self.intermediateSize = (try? c.decode(Int.self, forKey: CodingKeys.intermediateSize)) ?? 1536
        self.nLocalHeads = (try? c.decode(Int.self, forKey: CodingKeys.nLocalHeads)) ?? -1
        self.headDim = (try? c.decode(Int.self, forKey: CodingKeys.headDim)) ?? 64
        self.ropeBase = (try? c.decode(Float.self, forKey: CodingKeys.ropeBase)) ?? 10000.0
        self.normEps = (try? c.decode(Float.self, forKey: CodingKeys.normEps)) ?? 1e-5
        self.sampleRate = (try? c.decode(Int.self, forKey: CodingKeys.sampleRate)) ?? 44100
        self.hopLength = (try? c.decode(Int.self, forKey: CodingKeys.hopLength)) ?? 512
        self.latentDim = (try? c.decode(Int.self, forKey: CodingKeys.latentDim)) ?? 1024
        self.encoderDim = (try? c.decode(Int.self, forKey: CodingKeys.encoderDim)) ?? 64
        self.encoderRates = (try? c.decode([Int].self, forKey: CodingKeys.encoderRates)) ?? [2, 4, 8, 8]
        self.encoderTransformerLayers = (try? c.decode([Int].self, forKey: CodingKeys.encoderTransformerLayers)) ?? [0, 0, 0, 4]
        self.decoderDim = (try? c.decode(Int.self, forKey: CodingKeys.decoderDim)) ?? 1536
        self.decoderRates = (try? c.decode([Int].self, forKey: CodingKeys.decoderRates)) ?? [8, 8, 4, 2]
        self.codebookSize = (try? c.decode(Int.self, forKey: CodingKeys.codebookSize)) ?? 1024
        self.semanticCodebookSize = (try? c.decode(Int.self, forKey: CodingKeys.semanticCodebookSize)) ?? 4096
        self.codebookDim = (try? c.decode(Int.self, forKey: CodingKeys.codebookDim)) ?? 8
        self.nCodebooks = (try? c.decode(Int.self, forKey: CodingKeys.nCodebooks)) ?? 9
        self.downsampleFactor = (try? c.decode([Int].self, forKey: CodingKeys.downsampleFactor)) ?? [2, 2]
        self.quantizerWindowSize = (try? c.decode(Int.self, forKey: CodingKeys.quantizerWindowSize)) ?? 128
        self.quantizerNLayer = (try? c.decode(Int.self, forKey: CodingKeys.quantizerNLayer)) ?? 8
        self.quantizerNHead = (try? c.decode(Int.self, forKey: CodingKeys.quantizerNHead)) ?? 16
        self.quantizerDim = (try? c.decode(Int.self, forKey: CodingKeys.quantizerDim)) ?? 1024
        self.quantizerIntermediateSize = (try? c.decode(Int.self, forKey: CodingKeys.quantizerIntermediateSize)) ?? 3072

        if self.nLocalHeads == -1 {
            self.nLocalHeads = self.nHead
        }
    }

    public init() {
        blockSize = 2048
        nLayer = 8
        nHead = 8
        dim = 512
        intermediateSize = 1536
        nLocalHeads = 8
        headDim = 64
        ropeBase = 10000.0
        normEps = 1e-5
        sampleRate = 44100
        hopLength = 512
        latentDim = 1024
        encoderDim = 64
        encoderRates = [2, 4, 8, 8]
        encoderTransformerLayers = [0, 0, 0, 4]
        decoderDim = 1536
        decoderRates = [8, 8, 4, 2]
        codebookSize = 1024
        semanticCodebookSize = 4096
        codebookDim = 8
        nCodebooks = 9
        downsampleFactor = [2, 2]
        quantizerWindowSize = 128
        quantizerNLayer = 8
        quantizerNHead = 16
        quantizerDim = 1024
        quantizerIntermediateSize = 3072
    }

    public var frameLength: Int {
        hopLength * downsampleFactor.reduce(1, *)
    }
}
