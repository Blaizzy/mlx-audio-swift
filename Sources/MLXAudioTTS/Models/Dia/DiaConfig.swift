import Foundation

/// Configuration for Dia (Nari Labs), decoded from the checkpoint's `config.json`.
public struct DiaConfig: Codable, Sendable {
    public struct EncoderConfig: Codable, Sendable {
        public var nLayer: Int
        public var nEmbd: Int
        public var nHidden: Int
        public var nHead: Int
        public var headDim: Int
        public var mlpActivations: [String]
        public var usePreNorm: Bool

        enum CodingKeys: String, CodingKey {
            case nLayer = "n_layer"
            case nEmbd = "n_embd"
            case nHidden = "n_hidden"
            case nHead = "n_head"
            case headDim = "head_dim"
            case mlpActivations = "mlp_activations"
            case usePreNorm = "use_pre_norm"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            nLayer = try c.decode(Int.self, forKey: .nLayer)
            nEmbd = try c.decode(Int.self, forKey: .nEmbd)
            nHidden = try c.decode(Int.self, forKey: .nHidden)
            nHead = try c.decode(Int.self, forKey: .nHead)
            headDim = try c.decode(Int.self, forKey: .headDim)
            mlpActivations = try c.decodeIfPresent([String].self, forKey: .mlpActivations) ?? ["silu", "linear"]
            usePreNorm = try c.decodeIfPresent(Bool.self, forKey: .usePreNorm) ?? false
        }
    }

    public struct DecoderConfig: Codable, Sendable {
        public var nLayer: Int
        public var nEmbd: Int
        public var nHidden: Int
        public var gqaQueryHeads: Int
        public var kvHeads: Int
        public var gqaHeadDim: Int
        public var crossQueryHeads: Int
        public var crossHeadDim: Int
        public var mlpActivations: [String]
        public var usePreNorm: Bool

        enum CodingKeys: String, CodingKey {
            case nLayer = "n_layer"
            case nEmbd = "n_embd"
            case nHidden = "n_hidden"
            case gqaQueryHeads = "gqa_query_heads"
            case kvHeads = "kv_heads"
            case gqaHeadDim = "gqa_head_dim"
            case crossQueryHeads = "cross_query_heads"
            case crossHeadDim = "cross_head_dim"
            case mlpActivations = "mlp_activations"
            case usePreNorm = "use_pre_norm"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            nLayer = try c.decode(Int.self, forKey: .nLayer)
            nEmbd = try c.decode(Int.self, forKey: .nEmbd)
            nHidden = try c.decode(Int.self, forKey: .nHidden)
            gqaQueryHeads = try c.decode(Int.self, forKey: .gqaQueryHeads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            gqaHeadDim = try c.decode(Int.self, forKey: .gqaHeadDim)
            crossQueryHeads = try c.decode(Int.self, forKey: .crossQueryHeads)
            crossHeadDim = try c.decode(Int.self, forKey: .crossHeadDim)
            mlpActivations = try c.decodeIfPresent([String].self, forKey: .mlpActivations) ?? ["silu", "linear"]
            usePreNorm = try c.decodeIfPresent(Bool.self, forKey: .usePreNorm) ?? false
        }
    }

    public struct ModelConfig: Codable, Sendable {
        public var encoder: EncoderConfig
        public var decoder: DecoderConfig
        public var srcVocabSize: Int
        public var tgtVocabSize: Int
        public var normalizationLayerEpsilon: Float
        public var ropeMinTimescale: Int
        public var ropeMaxTimescale: Int
        public var sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case encoder, decoder
            case srcVocabSize = "src_vocab_size"
            case tgtVocabSize = "tgt_vocab_size"
            case normalizationLayerEpsilon = "normalization_layer_epsilon"
            case ropeMinTimescale = "rope_min_timescale"
            case ropeMaxTimescale = "rope_max_timescale"
            case sampleRate = "sample_rate"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            encoder = try c.decode(EncoderConfig.self, forKey: .encoder)
            self.decoder = try c.decode(DecoderConfig.self, forKey: .decoder)
            srcVocabSize = try c.decodeIfPresent(Int.self, forKey: .srcVocabSize) ?? 256
            tgtVocabSize = try c.decodeIfPresent(Int.self, forKey: .tgtVocabSize) ?? 1028
            normalizationLayerEpsilon = try c.decodeIfPresent(Float.self, forKey: .normalizationLayerEpsilon) ?? 1e-5
            ropeMinTimescale = try c.decodeIfPresent(Int.self, forKey: .ropeMinTimescale) ?? 1
            ropeMaxTimescale = try c.decodeIfPresent(Int.self, forKey: .ropeMaxTimescale) ?? 10_000
            sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 44_100
        }
    }

    public struct DataConfig: Codable, Sendable {
        public var textLength: Int
        public var audioLength: Int
        public var channels: Int
        public var textPadValue: Int
        public var audioEosValue: Int
        public var audioPadValue: Int
        public var audioBosValue: Int
        public var delayPattern: [Int]

        enum CodingKeys: String, CodingKey {
            case textLength = "text_length"
            case audioLength = "audio_length"
            case channels
            case textPadValue = "text_pad_value"
            case audioEosValue = "audio_eos_value"
            case audioPadValue = "audio_pad_value"
            case audioBosValue = "audio_bos_value"
            case delayPattern = "delay_pattern"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            textLength = try c.decode(Int.self, forKey: .textLength)
            audioLength = try c.decode(Int.self, forKey: .audioLength)
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 9
            textPadValue = try c.decodeIfPresent(Int.self, forKey: .textPadValue) ?? 0
            audioEosValue = try c.decodeIfPresent(Int.self, forKey: .audioEosValue) ?? 1024
            audioPadValue = try c.decodeIfPresent(Int.self, forKey: .audioPadValue) ?? 1025
            audioBosValue = try c.decodeIfPresent(Int.self, forKey: .audioBosValue) ?? 1026
            delayPattern = try c.decodeIfPresent([Int].self, forKey: .delayPattern) ?? [0, 8, 9, 10, 11, 12, 13, 14, 15]
        }
    }

    public var model: ModelConfig
    public var data: DataConfig
}
