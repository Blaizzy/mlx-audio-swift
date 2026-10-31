import Foundation
import MLXAudioCore

public struct Maya2DepthConfiguration: Codable, Sendable {
    public var hiddenSize = 1024
    public var numAttentionHeads = 16
    public var intermediateSize = 3072
    public var numHiddenLayers = 4
    public var embeddingRank = 128

    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size", numHiddenLayers = "num_hidden_layers"
        case embeddingRank = "embedding_rank"
    }
    public init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? numAttentionHeads
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
        embeddingRank = try c.decodeIfPresent(Int.self, forKey: .embeddingRank) ?? embeddingRank
    }
}

public struct Maya2Configuration: Decodable, Sendable {
    public var backbone = Qwen3Configuration(hiddenSize: 2048, hiddenLayers: 28,
        intermediateSize: 6144, attentionHeads: 16, kvHeads: 8, headDim: 128,
        vocabularySize: 194548, rmsNormEps: 1e-6, ropeTheta: 1_000_000, tieWordEmbeddings: true)
    public var depth = Maya2DepthConfiguration()
    public var numCodebooks = 32
    public var codebookSize = 2048
    public var sampleRate = 24000
    public var frameRate: Double = 12.5
    public var speakerDim = 192
    public var speakerAnchorHidden = 512
    public var speakerAnchorEnabled = true
    public var codecRepo = "kyutai/moshiko-pytorch-bf16"
    public var codecFilename = "tokenizer-e351c8d8-checkpoint125.safetensors"
    public var speakerEncoderRepo = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
    public var quantization: BaseConfiguration.Quantization?

    public init() {}
    enum CodingKeys: String, CodingKey {
        case backbone = "backbone_config", depth = "depth_config", numCodebooks = "num_codebooks"
        case codebookSize = "codebook_size", sampleRate = "sample_rate", frameRate = "frame_rate"
        case speakerDim = "speaker_dim", speakerAnchorHidden = "speaker_anchor_hidden"
        case speakerAnchorEnabled = "speaker_anchor_enabled", codecRepo = "codec_repo"
        case codecFilename = "codec_filename", speakerEncoderRepo = "speaker_encoder_repo"
        case quantization
    }
    public init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        backbone = try c.decodeIfPresent(Qwen3Configuration.self, forKey: .backbone) ?? backbone
        depth = try c.decodeIfPresent(Maya2DepthConfiguration.self, forKey: .depth) ?? depth
        numCodebooks = try c.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? numCodebooks
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? codebookSize
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        frameRate = try c.decodeIfPresent(Double.self, forKey: .frameRate) ?? frameRate
        speakerDim = try c.decodeIfPresent(Int.self, forKey: .speakerDim) ?? speakerDim
        speakerAnchorHidden = try c.decodeIfPresent(Int.self, forKey: .speakerAnchorHidden) ?? speakerAnchorHidden
        speakerAnchorEnabled = try c.decodeIfPresent(Bool.self, forKey: .speakerAnchorEnabled) ?? speakerAnchorEnabled
        codecRepo = try c.decodeIfPresent(String.self, forKey: .codecRepo) ?? codecRepo
        codecFilename = try c.decodeIfPresent(String.self, forKey: .codecFilename) ?? codecFilename
        speakerEncoderRepo = try c.decodeIfPresent(String.self, forKey: .speakerEncoderRepo) ?? speakerEncoderRepo
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
    }

    func validate() throws {
        let b = backbone, d = depth
        guard (2...32).contains(numCodebooks), codebookSize > 0, sampleRate == 24000,
              frameRate == 12.5, speakerDim > 0, speakerAnchorHidden > 0,
              b.hiddenSize > 0, b.hiddenLayers > 0, b.intermediateSize > 0,
              b.attentionHeads > 0, b.kvHeads > 0, b.headDim > 0,
              b.attentionHeads % b.kvHeads == 0, b.headDim % 2 == 0,
              b.vocabularySize > 0, b.maxPositionEmbeddings > 0,
              d.hiddenSize > 0, d.numAttentionHeads > 0,
              d.hiddenSize % d.numAttentionHeads == 0, d.numHiddenLayers > 0,
              d.intermediateSize > 0, d.embeddingRank > 0,
              codecFilename == URL(fileURLWithPath: codecFilename).lastPathComponent,
              codecFilename.hasSuffix(".safetensors") else {
            throw AudioGenerationError.invalidInput("Invalid Maya-2 architecture or codec configuration")
        }
    }
}

public struct Maya2GenerationOptions: Sendable {
    public enum Conditioning: String, Sendable { case inContextAndSpeaker = "icl_spk", inContext = "icl", speaker = "spk" }
    public var referenceLanguage: String?
    public var conditioning: Conditioning = .inContextAndSpeaker
    public var depthTemperature: Float = 0.7
    public var depthTopK = 64
    public var presencePenalty: Float = 0.8
    public var presenceWindow = 12
    public var minTokens = 2
    public var normalizeText = true
    public init() {}
}
