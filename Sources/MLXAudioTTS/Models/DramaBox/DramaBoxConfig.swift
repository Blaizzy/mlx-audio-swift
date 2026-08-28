import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon

public let dramaBoxDefaultAudioRepository = "appautomaton/dramabox-tts-3.3b-bf16-mlx"
public let dramaBoxDefaultGemmaRepository = "appautomaton/gemma-3-12b-it-backbone-4bit-mlx"

public let dramaBoxDiTWeightFile = "dramabox-dit-v1.safetensors"
public let dramaBoxAudioComponentsFile = "dramabox-audio-components.safetensors"

/// Warm-server negative prompt. CFG needs named acoustic failure modes;
/// an empty string collapses guidance to a near no-op.
public let dramaBoxDefaultNegativePrompt = """
worst quality, inconsistent, robotic, distorted, noise, static, \
muffled, unclear, unnatural, monotone
"""

public enum DramaBoxError: Error, LocalizedError, Sendable {
    case invalidRepositoryID(String)
    case missingCheckpoint(URL, String)
    case missingGemmaBackbone(String)
    case denoiseRefUnsupported
    case silentReferenceAudio
    case invalidAudioShape([Int])
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id):
            "Invalid repository ID: \(id)"
        case .missingCheckpoint(let dir, let name):
            "Missing DramaBox file '\(name)' in \(dir.path)"
        case .missingGemmaBackbone(let message):
            "DramaBox Gemma 3 backbone: \(message)"
        case .denoiseRefUnsupported:
            "denoiseRef=true is out of scope for v1 (RE-USE / SEMamba). Pass denoiseRef=false and a raw voice reference."
        case .silentReferenceAudio:
            "Reference audio is silent."
        case .invalidAudioShape(let shape):
            "Invalid DramaBox audio shape: \(shape)"
        case .generationFailed(let message):
            message
        }
    }
}

/// DramaBox-only sampling knobs. Autoregressive `GenerateParameters`
/// (`maxTokens`, `temperature`, `topP`, …) are ignored.
public struct DramaBoxGenerateConfig: Sendable {
    public var durationSeconds: Float
    public var cfgScale: Float
    public var stgScale: Float
    /// `nil` means auto: `autoRescaleForCfg(cfgScale)` (0.3 when cfg=2.5).
    public var rescaleScale: Float?
    public var modalityScale: Float
    public var steps: Int
    public var seed: UInt64
    /// v1 throws if true. RE-USE / SEMamba is out of scope.
    public var denoiseRef: Bool
    public var maxPromptLength: Int
    public var referenceDurationSeconds: Float
    public var referenceSampleRate: Int?

    public init(
        durationSeconds: Float = 5.0,
        cfgScale: Float = 2.5,
        stgScale: Float = 1.5,
        rescaleScale: Float? = nil,
        modalityScale: Float = 1.0,
        steps: Int = 30,
        seed: UInt64 = 42,
        denoiseRef: Bool = false,
        maxPromptLength: Int = 1024,
        referenceDurationSeconds: Float = 10.0,
        referenceSampleRate: Int? = nil
    ) {
        self.durationSeconds = durationSeconds
        self.cfgScale = cfgScale
        self.stgScale = stgScale
        self.rescaleScale = rescaleScale
        self.modalityScale = modalityScale
        self.steps = steps
        self.seed = seed
        self.denoiseRef = denoiseRef
        self.maxPromptLength = maxPromptLength
        self.referenceDurationSeconds = referenceDurationSeconds
        self.referenceSampleRate = referenceSampleRate
    }

    public static let `default` = DramaBoxGenerateConfig()

    public var resolvedRescaleScale: Float {
        rescaleScale ?? dramaBoxAutoRescaleForCfg(cfgScale)
    }
}

/// Subset of Gemma 3 text-config used by the headless DramaBox encoder.
public struct DramaBoxGemmaTextConfig: Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeLocalBaseFreq: Float
    public var slidingWindow: Int
    public var slidingWindowPattern: Int
    public var queryPreAttnScalar: Int
    public var hiddenActivation: String
    public var maxPositionEmbeddings: Int
    public var ropeScalingFactor: Float
    public var quantization: BaseConfiguration.Quantization?

    public var attentionScale: Float {
        pow(Float(queryPreAttnScalar), -0.5)
    }

    public var hiddenStateCount: Int {
        numHiddenLayers + 1
    }

    /// Every `slidingWindowPattern`-th layer (1-indexed) is full attention.
    public func layerTypes() -> [String] {
        (0..<numHiddenLayers).map { index in
            (index + 1).isMultiple(of: slidingWindowPattern) ? "full_attention" : "sliding_attention"
        }
    }

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        vocabSize: Int,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000,
        ropeLocalBaseFreq: Float = 10_000,
        slidingWindow: Int = 1024,
        slidingWindowPattern: Int = 6,
        queryPreAttnScalar: Int = 256,
        hiddenActivation: String = "gelu_pytorch_tanh",
        maxPositionEmbeddings: Int = 131_072,
        ropeScalingFactor: Float = 8.0,
        quantization: BaseConfiguration.Quantization? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.queryPreAttnScalar = queryPreAttnScalar
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeScalingFactor = ropeScalingFactor
        self.quantization = quantization
    }

    public static func fromJSONObject(_ payload: [String: Any]) throws -> DramaBoxGemmaTextConfig {
        let text: [String: Any]
        if let nested = payload["text_config"] as? [String: Any] {
            text = nested
        } else {
            text = payload
        }

        func requireInt(_ key: String) throws -> Int {
            if let value = text[key] as? Int { return value }
            if let value = text[key] as? Double { return Int(value) }
            throw DramaBoxError.missingGemmaBackbone("config.json missing \(key)")
        }

        let rope = text["rope_scaling"] as? [String: Any]
        let quantPayload = (payload["quantization"] as? [String: Any])
            ?? (payload["quantization_config"] as? [String: Any])
        let quantization: BaseConfiguration.Quantization?
        if let quantPayload {
            let groupSize = (quantPayload["group_size"] as? Int)
                ?? Int(quantPayload["group_size"] as? Double ?? 64)
            let bits = (quantPayload["bits"] as? Int)
                ?? Int(quantPayload["bits"] as? Double ?? 4)
            quantization = BaseConfiguration.Quantization(groupSize: groupSize, bits: bits)
        } else {
            quantization = nil
        }

        return DramaBoxGemmaTextConfig(
            hiddenSize: try requireInt("hidden_size"),
            intermediateSize: try requireInt("intermediate_size"),
            numHiddenLayers: try requireInt("num_hidden_layers"),
            numAttentionHeads: try requireInt("num_attention_heads"),
            numKeyValueHeads: try requireInt("num_key_value_heads"),
            headDim: try requireInt("head_dim"),
            vocabSize: try requireInt("vocab_size"),
            rmsNormEps: Float(text["rms_norm_eps"] as? Double ?? 1e-6),
            ropeTheta: Float(text["rope_theta"] as? Double ?? 1_000_000),
            ropeLocalBaseFreq: Float(text["rope_local_base_freq"] as? Double ?? 10_000),
            slidingWindow: (text["sliding_window"] as? Int) ?? 1024,
            slidingWindowPattern: (text["sliding_window_pattern"] as? Int) ?? 6,
            queryPreAttnScalar: (text["query_pre_attn_scalar"] as? Int) ?? 256,
            hiddenActivation: text["hidden_activation"] as? String ?? "gelu_pytorch_tanh",
            maxPositionEmbeddings: (text["max_position_embeddings"] as? Int) ?? 131_072,
            ropeScalingFactor: Float(rope?["factor"] as? Double ?? 8.0),
            quantization: quantization
        )
    }

    public static func load(from directory: URL) throws -> DramaBoxGemmaTextConfig {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        guard let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DramaBoxError.missingGemmaBackbone("config.json is not a JSON object")
        }
        return try fromJSONObject(payload)
    }
}

public struct DramaBoxResult: Sendable {
    /// Stereo waveform `[2, T]` float32 in `[-1, 1]`.
    public var waveform: MLXArray
    public var sampleRate: Int
    public var durationSeconds: Float
    public var settings: DramaBoxGenerateConfig

    public init(
        waveform: MLXArray,
        sampleRate: Int = 48_000,
        durationSeconds: Float,
        settings: DramaBoxGenerateConfig
    ) {
        self.waveform = waveform
        self.sampleRate = sampleRate
        self.durationSeconds = durationSeconds
        self.settings = settings
    }
}

func dramaBoxAutoRescaleForCfg(_ cfgScale: Float) -> Float {
    if cfgScale <= 2.0 { return 0.0 }
    if cfgScale <= 3.0 { return 0.6 * (cfgScale - 2.0) }
    if cfgScale <= 4.0 { return 0.6 + 0.2 * (cfgScale - 3.0) }
    if cfgScale <= 8.0 { return 0.8 }
    return min(1.0, 0.8 + 0.1 * (cfgScale - 8.0))
}

func dramaBoxFinfoMax(_ dtype: DType) -> Float {
    switch dtype {
    case .float16:
        65504
    case .bfloat16:
        // Finite bfloat16 max. `Float.greatestFiniteMagnitude` rounds to Inf in bf16,
        // and a fully-masked attention row of -Inf makes softmax NaN.
        3.38953139e38
    default:
        Float.greatestFiniteMagnitude
    }
}

func dramaBoxFinfoMin(_ dtype: DType) -> Float {
    -dramaBoxFinfoMax(dtype)
}
