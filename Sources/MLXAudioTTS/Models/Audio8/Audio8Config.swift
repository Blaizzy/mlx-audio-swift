import Foundation
import MLXAudioCore

/// Configuration for the Audio8 attention and Falcon-H1 hybrid checkpoints.
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
    public var hiddenAct: String = "silu"
    public var attentionBias: Bool = false
    public var attentionInMultiplier: Float = 1
    public var attentionOutMultiplier: Float = 1
    public var keyMultiplier: Float = 1
    public var embeddingMultiplier: Float = 1
    public var lmHeadMultiplier: Float = 1
    public var mlpBias: Bool = false
    public var mlpMultipliers: [Float] = [1, 1]
    public var projectorsBias: Bool = false
    public var ssmInMultiplier: Float = 1
    public var ssmMultipliers: [Float] = [1, 1, 1, 1, 1]
    public var ssmOutMultiplier: Float = 1
    public var mambaChunkSize: Int = 128
    public var mambaConvBias: Bool = true
    public var mambaDConv: Int = 4
    public var mambaDHead: Int = 32
    public var mambaDSSM: Int = 768
    public var mambaDState: Int = 64
    public var mambaExpand: Int = 2
    public var mambaNGroups: Int = 1
    public var mambaNHeads: Int = 24
    public var mambaNormBeforeGate: Bool = false
    public var mambaProjBias: Bool = false
    public var mambaRMSNorm: Bool = false
    public var mambaUseMLP: Bool = true
    public var quantization: BaseConfiguration.Quantization?

    public init() {}

    public var usesFalconSlow: Bool { slowBackbone == "falcon_h1" }

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
        case hiddenAct = "hidden_act"
        case attentionBias = "attention_bias"
        case attentionInMultiplier = "attention_in_multiplier"
        case attentionOutMultiplier = "attention_out_multiplier"
        case keyMultiplier = "key_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case lmHeadMultiplier = "lm_head_multiplier"
        case mlpBias = "mlp_bias"
        case mlpMultipliers = "mlp_multipliers"
        case projectorsBias = "projectors_bias"
        case ssmInMultiplier = "ssm_in_multiplier"
        case ssmMultipliers = "ssm_multipliers"
        case ssmOutMultiplier = "ssm_out_multiplier"
        case mambaChunkSize = "mamba_chunk_size"
        case mambaConvBias = "mamba_conv_bias"
        case mambaDConv = "mamba_d_conv"
        case mambaDHead = "mamba_d_head"
        case mambaDSSM = "mamba_d_ssm"
        case mambaDState = "mamba_d_state"
        case mambaExpand = "mamba_expand"
        case mambaNGroups = "mamba_n_groups"
        case mambaNHeads = "mamba_n_heads"
        case mambaNormBeforeGate = "mamba_norm_before_gate"
        case mambaProjBias = "mamba_proj_bias"
        case mambaRMSNorm = "mamba_rms_norm"
        case mambaUseMLP = "mamba_use_mlp"
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
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionInMultiplier = try c.decodeIfPresent(Float.self, forKey: .attentionInMultiplier) ?? 1
        attentionOutMultiplier = try c.decodeIfPresent(Float.self, forKey: .attentionOutMultiplier) ?? 1
        keyMultiplier = try c.decodeIfPresent(Float.self, forKey: .keyMultiplier) ?? 1
        embeddingMultiplier = try c.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1
        lmHeadMultiplier = try c.decodeIfPresent(Float.self, forKey: .lmHeadMultiplier) ?? 1
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        mlpMultipliers = try c.decodeIfPresent([Float].self, forKey: .mlpMultipliers) ?? [1, 1]
        projectorsBias = try c.decodeIfPresent(Bool.self, forKey: .projectorsBias) ?? false
        ssmInMultiplier = try c.decodeIfPresent(Float.self, forKey: .ssmInMultiplier) ?? 1
        ssmMultipliers = try c.decodeIfPresent([Float].self, forKey: .ssmMultipliers) ?? [1, 1, 1, 1, 1]
        ssmOutMultiplier = try c.decodeIfPresent(Float.self, forKey: .ssmOutMultiplier) ?? 1
        mambaChunkSize = try c.decodeIfPresent(Int.self, forKey: .mambaChunkSize) ?? 128
        mambaConvBias = try c.decodeIfPresent(Bool.self, forKey: .mambaConvBias) ?? true
        mambaDConv = try c.decodeIfPresent(Int.self, forKey: .mambaDConv) ?? 4
        mambaDHead = try c.decodeIfPresent(Int.self, forKey: .mambaDHead) ?? 32
        mambaDSSM = try c.decodeIfPresent(Int.self, forKey: .mambaDSSM) ?? 768
        mambaDState = try c.decodeIfPresent(Int.self, forKey: .mambaDState) ?? 64
        mambaExpand = try c.decodeIfPresent(Int.self, forKey: .mambaExpand) ?? 2
        mambaNGroups = try c.decodeIfPresent(Int.self, forKey: .mambaNGroups) ?? 1
        mambaNHeads = try c.decodeIfPresent(Int.self, forKey: .mambaNHeads) ?? 24
        mambaNormBeforeGate = try c.decodeIfPresent(Bool.self, forKey: .mambaNormBeforeGate) ?? false
        mambaProjBias = try c.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        mambaRMSNorm = try c.decodeIfPresent(Bool.self, forKey: .mambaRMSNorm) ?? false
        mambaUseMLP = try c.decodeIfPresent(Bool.self, forKey: .mambaUseMLP) ?? true
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
    }

    func validate() throws {
        guard ["arktts", "falcon_h1"].contains(slowBackbone) else {
            throw AudioGenerationError.invalidInput("Unsupported Audio8 slow_backbone: \(slowBackbone)")
        }
        if usesFalconSlow {
            guard hiddenAct == "silu", mambaDConv > 0, mambaDHead > 0, mambaDState > 0,
                  mambaNHeads > 0, mambaNGroups > 0, mambaNHeads % mambaNGroups == 0,
                  mambaDSSM == mambaNHeads * mambaDHead,
                  embeddingMultiplier.isFinite, embeddingMultiplier > 0,
                  attentionInMultiplier == 1, attentionOutMultiplier == 1, keyMultiplier == 1,
                  ssmInMultiplier == 1, ssmOutMultiplier == 1,
                  mlpMultipliers == [1, 1], ssmMultipliers == [1, 1, 1, 1, 1] else {
                throw AudioGenerationError.invalidInput("Invalid Audio8 Falcon-H1 dimensions or unsupported internal muP multipliers")
            }
        }
        guard modelType == "arktts", (usesFalconSlow || tieWordEmbeddings), dim > 0, fastDim > 0, nLayer > 0, nFastLayer > 0,
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
            throw AudioGenerationError.invalidInput("Invalid or unsupported Audio8 backbone/codec configuration")
        }
    }
}
