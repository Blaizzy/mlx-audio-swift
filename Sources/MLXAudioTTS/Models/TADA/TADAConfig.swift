import Foundation

/// TADA's Llama backbone and jointly predicted acoustic/duration features.
public struct TADAConfiguration: Decodable, Sendable {
    public var vocabSize = 128256
    public var hiddenSize = 2048
    public var numHiddenLayers = 16
    public var numAttentionHeads = 32
    public var numKeyValueHeads = 8
    public var headDim = 64
    public var intermediateSize = 8192
    public var rmsNormEps: Float = 1e-5
    public struct RopeScaling: Decodable, Sendable {
        public var factor: Float
        public var lowFreqFactor: Float
        public var highFreqFactor: Float
        public var originalContext: Float
        enum CodingKeys: String, CodingKey {
            case factor, lowFreqFactor = "low_freq_factor", highFreqFactor = "high_freq_factor"
            case originalContext = "original_max_position_embeddings"
        }
    }
    public var ropeScaling: RopeScaling?
    public var ropeTheta: Float = 500000
    public var acousticDim = 512
    public var numTimeClasses = 256
    public var shiftAcoustic = 5
    public var headLayers = 6
    public var headFFNRatio: Float = 4
    public var bottleneckDim: Int?
    public var acousticMean: Float = 0
    public var acousticStd: Float = 1.5
    public var sampleRate = 24000
    public var tokenizerRepo = "mlx-community/Llama-3.2-1B-Instruct-bf16"
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        func value<T: Decodable>(_ key: String, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: Key(stringValue: key)!) ?? fallback
        }
        vocabSize = try value("vocab_size", vocabSize)
        hiddenSize = try value("hidden_size", hiddenSize)
        numHiddenLayers = try value("num_hidden_layers", numHiddenLayers)
        numAttentionHeads = try value("num_attention_heads", numAttentionHeads)
        numKeyValueHeads = try value("num_key_value_heads", numKeyValueHeads)
        headDim = try value("head_dim", hiddenSize / max(1, numAttentionHeads))
        intermediateSize = try value("intermediate_size", intermediateSize)
        rmsNormEps = try value("rms_norm_eps", rmsNormEps)
        ropeTheta = try value("rope_theta", ropeTheta)
        ropeScaling = try c.decodeIfPresent(RopeScaling.self, forKey: Key(stringValue: "rope_scaling")!)
        acousticDim = try value("acoustic_dim", acousticDim)
        numTimeClasses = try value("num_time_classes", numTimeClasses)
        shiftAcoustic = try value("shift_acoustic", shiftAcoustic)
        headLayers = try value("head_layers", headLayers)
        headFFNRatio = try value("head_ffn_ratio", headFFNRatio)
        bottleneckDim = try c.decodeIfPresent(Int.self, forKey: Key(stringValue: "bottleneck_dim")!)
        acousticMean = try value("acoustic_mean", acousticMean)
        acousticStd = try value("acoustic_std", acousticStd)
        sampleRate = try value("sample_rate", sampleRate)
        tokenizerRepo = try value("tokenizer_repo", tokenizerRepo)
        guard hiddenSize > 0, numAttentionHeads > 0, numKeyValueHeads > 0,
              numHiddenLayers > 0, acousticDim > 0, numTimeClasses > 1,
              numTimeClasses & (numTimeClasses - 1) == 0, acousticStd > 0 else {
            throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Invalid TADA dimensions"))
        }
    }
    private struct Key: CodingKey {
        var stringValue: String; var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    var timeBits: Int { Int(log2(Double(numTimeClasses))) }
    var latentSize: Int { acousticDim + 2 * timeBits }
}

public struct TADAGenerationOptions: Sendable {
    public enum Schedule: String, Sendable { case constant, linear, cosine, logsnr }
    public var acousticGuidance: Float = 1.6
    public var durationGuidance: Float = 1
    public var guidanceSchedule: Schedule = .cosine
    public var timeSchedule: Schedule = .logsnr
    public var noiseTemperature: Float = 0.9
    public var flowSteps = 20
    public var transitionSteps = 5
    /// Two-pass duration control; values greater than one produce faster speech.
    public var speed: Float?
    public init() {}
}
