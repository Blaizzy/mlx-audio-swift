//
//  SparkConfig.swift
//  MLXAudio
//
//  Configuration for Spark-TTS (SparkAudio): a Qwen2 language-model backbone
//  that emits BiCodec semantic tokens, decoded to audio by the BiCodec codec.
//

import Foundation
import MLXLMCommon

// MARK: - Language model (Qwen2) configuration

/// Qwen2 backbone configuration, decoded from the checkpoint's `config.json`.
public struct SparkConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var sampleRate: Int
    public var headDim: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case sampleRate = "sample_rate"
        case headDim = "head_dim"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 151643
        self.eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 151645
        self.sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        // Qwen2 omits head_dim; derive it from hidden_size / num_attention_heads.
        self.headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)
    }
}

// MARK: - BiCodec configuration

/// BiCodec codec configuration, decoded from the checkpoint's `BiCodec/config.yaml`
/// (the `audio_tokenizer` section). The encoder/prenet/postnet share a Vocos
/// backbone; the decoder is a HiFiGAN-style wave generator; global tokens come
/// from an ECAPA + finite-scalar-quantized speaker encoder.
public struct BiCodecConfiguration: Codable, Sendable {
    public struct MelParams: Codable, Sendable {
        public var sampleRate: Int
        public var nFFT: Int
        public var winLength: Int
        public var hopLength: Int
        public var melFmin: Float
        public var melFmax: Float?
        public var numMels: Int

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case nFFT = "n_fft"
            case winLength = "win_length"
            case hopLength = "hop_length"
            case melFmin = "mel_fmin"
            case melFmax = "mel_fmax"
            case numMels = "num_mels"
        }
    }

    /// Vocos-backbone config shared by `encoder`, `prenet` and `postnet`.
    public struct VocosBackbone: Codable, Sendable {
        public var inputChannels: Int
        public var vocosDim: Int
        public var vocosIntermediateDim: Int
        public var vocosNumLayers: Int
        public var outChannels: Int
        public var conditionDim: Int?
        public var sampleRatios: [Int]?
        public var useTanhAtFinal: Bool?

        enum CodingKeys: String, CodingKey {
            case inputChannels = "input_channels"
            case vocosDim = "vocos_dim"
            case vocosIntermediateDim = "vocos_intermediate_dim"
            case vocosNumLayers = "vocos_num_layers"
            case outChannels = "out_channels"
            case conditionDim = "condition_dim"
            case sampleRatios = "sample_ratios"
            case useTanhAtFinal = "use_tanh_at_final"
        }
    }

    /// HiFiGAN-style wave generator (decoder).
    public struct WaveGenerator: Codable, Sendable {
        public var inputChannel: Int
        public var channels: Int
        public var rates: [Int]
        public var kernelSizes: [Int]

        enum CodingKeys: String, CodingKey {
            case inputChannel = "input_channel"
            case channels
            case rates
            case kernelSizes = "kernel_sizes"
        }
    }

    /// Factorized vector quantizer producing the semantic tokens.
    public struct Quantizer: Codable, Sendable {
        public var inputDim: Int
        public var codebookSize: Int
        public var codebookDim: Int
        public var commitment: Float
        public var useL2Normlize: Bool

        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case codebookSize = "codebook_size"
            case codebookDim = "codebook_dim"
            case commitment
            case useL2Normlize = "use_l2_normlize"
        }
    }

    /// ECAPA + finite-scalar-quantized speaker encoder producing global tokens.
    public struct SpeakerEncoder: Codable, Sendable {
        public var inputDim: Int
        public var outDim: Int
        public var latentDim: Int
        public var tokenNum: Int
        public var fsqLevels: [Int]
        public var fsqNumQuantizers: Int

        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case outDim = "out_dim"
            case latentDim = "latent_dim"
            case tokenNum = "token_num"
            case fsqLevels = "fsq_levels"
            case fsqNumQuantizers = "fsq_num_quantizers"
        }
    }

    public var melParams: MelParams
    public var encoder: VocosBackbone
    public var decoder: WaveGenerator
    public var quantizer: Quantizer
    public var speakerEncoder: SpeakerEncoder
    public var prenet: VocosBackbone
    public var postnet: VocosBackbone

    enum CodingKeys: String, CodingKey {
        case melParams = "mel_params"
        case encoder, decoder, quantizer
        case speakerEncoder = "speaker_encoder"
        case prenet, postnet
    }
}
