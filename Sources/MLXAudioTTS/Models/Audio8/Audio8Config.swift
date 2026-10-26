import Foundation
import MLXAudioCore

/// Configuration for the Audio8 attention-backbone checkpoints.
public struct Audio8Config: Codable, Sendable {
    public var modelType: String = "arktts"
    public var slowBackbone: String = "arktts"
    public var vocabSize: Int = 155776
    public var dim: Int = 896
    public var nLayer: Int = 24
    public var nHead: Int = 14
    public var nLocalHeads: Int = 2
    public var headDim: Int = 64
    public var intermediateSize: Int = 4864
    public var maxSeqLen: Int = 2048
    public var ropeBase: Float = 1000000
    public var normEps: Float = 1e-6
    public var attentionQKVBias: Bool = true
    public var attentionQKNorm: Bool = false
    public var attentionOBias: Bool = false
    public var tieWordEmbeddings: Bool = true
    public var codebookSize: Int = 4096
    public var numCodebooks: Int = 10
    public var semanticBeginID: Int = 151678
    public var semanticEndID: Int = 155773
    public var nFastLayer: Int = 4
    public var fastDim: Int = 896
    public var fastNHead: Int = 14
    public var fastNLocalHeads: Int = 2
    public var fastHeadDim: Int = 64
    public var fastIntermediateSize: Int = 4864
    public var fastAttentionQKVBias: Bool = false
    public var fastAttentionQKNorm: Bool = false
    public var fastAttentionOBias: Bool = false
    public var normFastlayerInput: Bool = true
    public var codecFilename: String = "codec.safetensors"
    public var codecSampleRate: Int = 44100
    public var codecFrameSize: Int = 2048
    public var codecPostNLayer: Int = 8
    public var codecPostNHead: Int = 16
    public var codecPostNLocalHeads: Int = 8
    public var codecPostIntermediateSize: Int = 1216
    public var rasWindowSize: Int = 10
    public var rasTemperature: Float = 1
    public var rasTopP: Float = 0.9
    public var eosTokenID: Int = 151645
    public var padTokenID: Int = 151643
    public var quantization: BaseConfiguration.Quantization?

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case slowBackbone = "slow_backbone"
        case vocabSize = "vocab_size"
        case dim = "dim"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case nLocalHeads = "n_local_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxSeqLen = "max_seq_len"
        case ropeBase = "rope_base"
        case normEps = "norm_eps"
        case attentionQKVBias = "attention_qkv_bias"
        case attentionQKNorm = "attention_qk_norm"
        case attentionOBias = "attention_o_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case codebookSize = "codebook_size"
        case numCodebooks = "num_codebooks"
        case semanticBeginID = "semantic_begin_id"
        case semanticEndID = "semantic_end_id"
        case nFastLayer = "n_fast_layer"
        case fastDim = "fast_dim"
        case fastNHead = "fast_n_head"
        case fastNLocalHeads = "fast_n_local_heads"
        case fastHeadDim = "fast_head_dim"
        case fastIntermediateSize = "fast_intermediate_size"
        case fastAttentionQKVBias = "fast_attention_qkv_bias"
        case fastAttentionQKNorm = "fast_attention_qk_norm"
        case fastAttentionOBias = "fast_attention_o_bias"
        case normFastlayerInput = "norm_fastlayer_input"
        case codecFilename = "codec_filename"
        case codecSampleRate = "codec_sample_rate"
        case codecFrameSize = "codec_frame_size"
        case codecPostNLayer = "codec_post_n_layer"
        case codecPostNHead = "codec_post_n_head"
        case codecPostNLocalHeads = "codec_post_n_local_heads"
        case codecPostIntermediateSize = "codec_post_intermediate_size"
        case rasWindowSize = "ras_window_size"
        case rasTemperature = "ras_temperature"
        case rasTopP = "ras_top_p"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
        case quantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "arktts"
        slowBackbone = try c.decodeIfPresent(String.self, forKey: .slowBackbone) ?? "arktts"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 155776
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 896
        nLayer = try c.decodeIfPresent(Int.self, forKey: .nLayer) ?? 24
        nHead = try c.decodeIfPresent(Int.self, forKey: .nHead) ?? 14
        nLocalHeads = try c.decodeIfPresent(Int.self, forKey: .nLocalHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864
        maxSeqLen = try c.decodeIfPresent(Int.self, forKey: .maxSeqLen) ?? 2048
        ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 1000000
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-6
        attentionQKVBias = try c.decodeIfPresent(Bool.self, forKey: .attentionQKVBias) ?? true
        attentionQKNorm = try c.decodeIfPresent(Bool.self, forKey: .attentionQKNorm) ?? false
        attentionOBias = try c.decodeIfPresent(Bool.self, forKey: .attentionOBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 4096
        numCodebooks = try c.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? 10
        semanticBeginID = try c.decodeIfPresent(Int.self, forKey: .semanticBeginID) ?? 151678
        semanticEndID = try c.decodeIfPresent(Int.self, forKey: .semanticEndID) ?? 155773
        nFastLayer = try c.decodeIfPresent(Int.self, forKey: .nFastLayer) ?? 4
        fastDim = try c.decodeIfPresent(Int.self, forKey: .fastDim) ?? 896
        fastNHead = try c.decodeIfPresent(Int.self, forKey: .fastNHead) ?? 14
        fastNLocalHeads = try c.decodeIfPresent(Int.self, forKey: .fastNLocalHeads) ?? 2
        fastHeadDim = try c.decodeIfPresent(Int.self, forKey: .fastHeadDim) ?? 64
        fastIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .fastIntermediateSize) ?? 4864
        fastAttentionQKVBias = try c.decodeIfPresent(Bool.self, forKey: .fastAttentionQKVBias) ?? false
        fastAttentionQKNorm = try c.decodeIfPresent(Bool.self, forKey: .fastAttentionQKNorm) ?? false
        fastAttentionOBias = try c.decodeIfPresent(Bool.self, forKey: .fastAttentionOBias) ?? false
        normFastlayerInput = try c.decodeIfPresent(Bool.self, forKey: .normFastlayerInput) ?? true
        codecFilename = try c.decodeIfPresent(String.self, forKey: .codecFilename) ?? "codec.safetensors"
        codecSampleRate = try c.decodeIfPresent(Int.self, forKey: .codecSampleRate) ?? 44100
        codecFrameSize = try c.decodeIfPresent(Int.self, forKey: .codecFrameSize) ?? 2048
        codecPostNLayer = try c.decodeIfPresent(Int.self, forKey: .codecPostNLayer) ?? 8
        codecPostNHead = try c.decodeIfPresent(Int.self, forKey: .codecPostNHead) ?? 16
        codecPostNLocalHeads = try c.decodeIfPresent(Int.self, forKey: .codecPostNLocalHeads) ?? 8
        codecPostIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .codecPostIntermediateSize) ?? 1216
        rasWindowSize = try c.decodeIfPresent(Int.self, forKey: .rasWindowSize) ?? 10
        rasTemperature = try c.decodeIfPresent(Float.self, forKey: .rasTemperature) ?? 1
        rasTopP = try c.decodeIfPresent(Float.self, forKey: .rasTopP) ?? 0.9
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 151645
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 151643
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
    }

    func validate() throws {
        guard slowBackbone == "arktts" else {
            throw AudioGenerationError.invalidInput("Audio8 slow_backbone \(slowBackbone) is unsupported; the Falcon-H1 0.1B variant requires its own runtime.")
        }
        guard modelType == "arktts", tieWordEmbeddings, dim > 0, fastDim > 0, nLayer > 0, nFastLayer > 0,
              nHead > 0, fastNHead > 0, nLocalHeads > 0, fastNLocalHeads > 0, nHead % nLocalHeads == 0,
              fastNHead % fastNLocalHeads == 0, headDim > 0, headDim % 2 == 0,
              fastHeadDim > 0, fastHeadDim % 2 == 0, maxSeqLen > 1,
              intermediateSize > 0, fastIntermediateSize > 0, normEps > 0, normEps.isFinite,
              codecPostNLayer > 0, codecPostNHead > 0, codecPostNLocalHeads > 0,
              codecPostNHead % codecPostNLocalHeads == 0, codecPostIntermediateSize > 0,
              codebookSize == 4096, numCodebooks == 10,
              semanticBeginID >= 0, semanticEndID - semanticBeginID + 1 == codebookSize,
              semanticEndID < vocabSize, (0..<vocabSize).contains(eosTokenID),
              codecSampleRate == 44100, codecFrameSize == 2048, rasWindowSize > 0 else {
            throw AudioGenerationError.invalidInput("Invalid or unsupported Audio8 attention/codec configuration")
        }
    }
}
