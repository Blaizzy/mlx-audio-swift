import Foundation

public enum MiMoAudioTokenizerError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case missingDirectory(URL)
    case missingWeights(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): "Invalid MiMo tokenizer configuration: \(message)"
        case .invalidInput(let message): message
        case .missingDirectory(let directory): "MiMo tokenizer directory not found: \(directory.path)"
        case .missingWeights(let directory): "No MiMo tokenizer safetensors found in \(directory.path)"
        }
    }
}

/// Configuration of the offline MiMo waveform tokenizer, including its vocoder.
public struct MiMoAudioTokenizerConfig: Codable, Sendable {
    public var modelType = "mimo_audio_tokenizer"
    public var dModel = 1280
    public var nMels = 128
    public var samplingRate = 24_000
    public var nFFT = 960
    public var hopLength = 240
    public var windowSize = 960
    public var fMin: Float = 0
    public var fMax: Float?
    public var kernelSize = 3
    public var strideSize = 2
    public var avgPooler = 2
    public var encoderLayers = 32
    public var encoderSkipLayerID: Int? = 3
    public var encoderAttentionHeads = 20
    public var encoderFFNDim = 5120
    public var encoderCausal = false
    public var encoderAttentionWindow = [-1, -1]
    public var decoderLayers = 32
    public var decoderAttentionHeads = 20
    public var decoderFFNDim = 5120
    public var decoderKernelSize = 3
    public var decoderStrideSize = 2
    public var decoderCausal = true
    public var decoderAttentionWindow = [-1, -1]
    public var vocoderDim = 256
    public var vocoderIntermediateDim = 1024
    public var vocoderNumLayers = 16
    public var vocoderAttentionHeads = 16
    public var vocoderAttentionWindow = [40, 10]
    public var vocoderPadding = "same"
    public var numQuantizers = 20
    public var codebookSize = [1024, 1024] + Array(repeating: 128, count: 18)
    public var ropeTheta: Float = 10_000
    public var ropeType = "default"
    public var normalization = "LayerNorm"
    public var activationFunction = "gelu"
    public var weightsFormat = "pytorch"

    public init() {}

    public var downsampleRate: Int { hopLength * strideSize * avgPooler }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", dModel = "d_model", nMels = "n_mels"
        case samplingRate = "sampling_rate", nFFT = "nfft", hopLength = "hop_length"
        case windowSize = "window_size", fMin = "fmin", fMax = "fmax"
        case kernelSize = "kernel_size", strideSize = "stride_size", avgPooler = "avg_pooler"
        case encoderLayers = "encoder_layers", encoderSkipLayerID = "encoder_skip_layer_id"
        case encoderAttentionHeads = "encoder_attention_heads", encoderFFNDim = "encoder_ffn_dim"
        case encoderCausal = "encoder_causal", encoderAttentionWindow = "encoder_attn_window_size"
        case decoderLayers = "decoder_layers", decoderAttentionHeads = "decoder_attention_heads"
        case decoderFFNDim = "decoder_ffn_dim", decoderKernelSize = "decoder_kernel_size"
        case decoderStrideSize = "decoder_stride_size", decoderCausal = "decoder_causal"
        case decoderAttentionWindow = "decoder_attn_window_size", vocoderDim = "vocoder_dim"
        case vocoderIntermediateDim = "vocoder_intermediate_dim", vocoderNumLayers = "vocoder_num_layers"
        case vocoderAttentionHeads = "vocoder_attention_heads", vocoderAttentionWindow = "vocoder_attn_window_size"
        case vocoderPadding = "vocoder_padding", numQuantizers = "num_quantizers", codebookSize = "codebook_size"
        case ropeTheta = "rope_theta", ropeType = "rope_type", normalization = "ln_type"
        case activationFunction = "activation_function", weightsFormat = "weights_format"
    }

    public init(from decoder: Swift.Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? dModel
        nMels = try c.decodeIfPresent(Int.self, forKey: .nMels) ?? nMels
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? samplingRate
        nFFT = try c.decodeIfPresent(Int.self, forKey: .nFFT) ?? nFFT
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? hopLength
        windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? windowSize
        fMin = try c.decodeIfPresent(Float.self, forKey: .fMin) ?? fMin
        fMax = try c.decodeIfPresent(Float.self, forKey: .fMax)
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? kernelSize
        strideSize = try c.decodeIfPresent(Int.self, forKey: .strideSize) ?? strideSize
        avgPooler = try c.decodeIfPresent(Int.self, forKey: .avgPooler) ?? avgPooler
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? encoderLayers
        if c.contains(.encoderSkipLayerID) {
            encoderSkipLayerID = try c.decodeIfPresent(Int.self, forKey: .encoderSkipLayerID)
        }
        encoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? encoderAttentionHeads
        encoderFFNDim = try c.decodeIfPresent(Int.self, forKey: .encoderFFNDim) ?? encoderFFNDim
        encoderCausal = try c.decodeIfPresent(Bool.self, forKey: .encoderCausal) ?? encoderCausal
        encoderAttentionWindow = try c.decodeIfPresent([Int].self, forKey: .encoderAttentionWindow) ?? encoderAttentionWindow
        decoderLayers = try c.decodeIfPresent(Int.self, forKey: .decoderLayers) ?? decoderLayers
        decoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .decoderAttentionHeads) ?? decoderAttentionHeads
        decoderFFNDim = try c.decodeIfPresent(Int.self, forKey: .decoderFFNDim) ?? decoderFFNDim
        decoderKernelSize = try c.decodeIfPresent(Int.self, forKey: .decoderKernelSize) ?? decoderKernelSize
        decoderStrideSize = try c.decodeIfPresent(Int.self, forKey: .decoderStrideSize) ?? decoderStrideSize
        decoderCausal = try c.decodeIfPresent(Bool.self, forKey: .decoderCausal) ?? decoderCausal
        decoderAttentionWindow = try c.decodeIfPresent([Int].self, forKey: .decoderAttentionWindow) ?? decoderAttentionWindow
        vocoderDim = try c.decodeIfPresent(Int.self, forKey: .vocoderDim) ?? vocoderDim
        vocoderIntermediateDim = try c.decodeIfPresent(Int.self, forKey: .vocoderIntermediateDim) ?? vocoderIntermediateDim
        vocoderNumLayers = try c.decodeIfPresent(Int.self, forKey: .vocoderNumLayers) ?? vocoderNumLayers
        vocoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .vocoderAttentionHeads) ?? vocoderAttentionHeads
        vocoderAttentionWindow = try c.decodeIfPresent([Int].self, forKey: .vocoderAttentionWindow) ?? vocoderAttentionWindow
        vocoderPadding = try c.decodeIfPresent(String.self, forKey: .vocoderPadding) ?? vocoderPadding
        numQuantizers = try c.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? numQuantizers
        codebookSize = try c.decodeIfPresent([Int].self, forKey: .codebookSize) ?? codebookSize
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        ropeType = try c.decodeIfPresent(String.self, forKey: .ropeType) ?? ropeType
        normalization = try c.decodeIfPresent(String.self, forKey: .normalization) ?? normalization
        activationFunction = try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? activationFunction
        weightsFormat = try c.decodeIfPresent(String.self, forKey: .weightsFormat) ?? weightsFormat
        try validate()
    }

    public func validate() throws {
        func require(_ condition: Bool, _ message: String) throws {
            if !condition { throw MiMoAudioTokenizerError.invalidConfiguration(message) }
        }
        try require(samplingRate > 0 && nMels > 0 && nFFT > 0 && nFFT % 2 == 0,
                    "sample rate, mel count, and even FFT size must be positive")
        try require(hopLength > 0 && hopLength <= nFFT && windowSize > 1 && windowSize <= nFFT
                    && (nFFT - hopLength) % 2 == 0, "invalid STFT window/hop or same-padding geometry")
        try require(kernelSize > 0 && strideSize > 0 && avgPooler > 0
                    && decoderKernelSize > 0 && decoderStrideSize > 0, "convolution sizes must be positive")
        try require(encoderLayers >= 0 && decoderLayers >= 0 && vocoderNumLayers >= 0
                    && encoderFFNDim > 0 && decoderFFNDim > 0 && vocoderIntermediateDim > 0,
                    "invalid Transformer dimensions")
        try require(numQuantizers > 0 && codebookSize.count == numQuantizers
                    && codebookSize.allSatisfy { $0 > 0 }, "codebook_size must specify every RVQ codebook")
        try require(ropeType == "default" && activationFunction == "gelu" && vocoderPadding == "same",
                    "requires default RoPE, GELU, and same-padding ISTFT")
        try require(["LayerNorm", "RMSNorm"].contains(normalization), "unsupported normalization")
        try require(["pytorch", "mlx"].contains(weightsFormat), "weights_format must be pytorch or mlx")
        try require(ropeTheta.isFinite && ropeTheta > 0 && fMin.isFinite && fMin >= 0
                    && (fMax ?? Float(samplingRate) / 2).isFinite
                    && (fMax ?? Float(samplingRate) / 2) > fMin
                    && (fMax ?? Float(samplingRate) / 2) <= Float(samplingRate) / 2,
                    "invalid RoPE base or mel frequency range")
        for (dim, heads) in [(dModel, encoderAttentionHeads), (dModel, decoderAttentionHeads),
                             (vocoderDim, vocoderAttentionHeads)] {
            try require(dim > 0 && heads > 0 && dim % heads == 0 && (dim / heads) % 2 == 0,
                        "attention dimensions must have an even head size")
        }
        for window in [encoderAttentionWindow, decoderAttentionWindow, vocoderAttentionWindow] {
            try require(window.count == 2 && window.allSatisfy { $0 >= -1 }, "attention windows require [left, right]")
        }
    }
}
