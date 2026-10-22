import Foundation

public struct FunASRNanoAdaptorConfig: Decodable {
    public var downsampleRate: Int = 1
    public var ffnDim: Int = 2048
    public var llmDim: Int = 1024
    public var encoderDim: Int = 512
    public var numLayers: Int = 2
    public var attentionHeads: Int = 8

    public init(downsampleRate: Int = 1, ffnDim: Int = 2048, llmDim: Int = 1024,
                encoderDim: Int = 512, numLayers: Int = 2, attentionHeads: Int = 8) {
        self.downsampleRate = downsampleRate
        self.ffnDim = ffnDim
        self.llmDim = llmDim
        self.encoderDim = encoderDim
        self.numLayers = numLayers
        self.attentionHeads = attentionHeads
    }

    enum CodingKeys: String, CodingKey {
        case downsampleRate = "downsample_rate", ffnDim = "ffn_dim", llmDim = "llm_dim"
        case encoderDim = "encoder_dim", numLayers = "n_layer", attentionHeads = "attention_heads"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        downsampleRate = try c.decodeIfPresent(Int.self, forKey: .downsampleRate) ?? 1
        ffnDim = try c.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 2048
        llmDim = try c.decodeIfPresent(Int.self, forKey: .llmDim) ?? 1024
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 512
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 2
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
    }
}

public struct FunASRNanoConfig: Decodable {
    public var modelType: String
    public var inputSize: Int
    public var tokenizerPath: String
    public var frontend: SenseVoiceFrontendConfig
    public var audioEncoder: SenseVoiceEncoderConfig
    public var audioAdaptor: FunASRNanoAdaptorConfig
    public var textConfig: Qwen3TextConfig
    public var defaultMaxTokens: Int

    public init(inputSize: Int = 560, tokenizerPath: String = "Qwen3-0.6B",
                frontend: SenseVoiceFrontendConfig = .init(),
                audioEncoder: SenseVoiceEncoderConfig = .init(),
                audioAdaptor: FunASRNanoAdaptorConfig = .init(),
                textConfig: Qwen3TextConfig = .init(), defaultMaxTokens: Int = 512) {
        self.modelType = "fun_asr_nano"
        self.inputSize = inputSize
        self.tokenizerPath = tokenizerPath
        self.frontend = frontend
        self.audioEncoder = audioEncoder
        self.audioAdaptor = audioAdaptor
        self.textConfig = textConfig
        self.defaultMaxTokens = defaultMaxTokens
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", inputSize = "input_size"
        case tokenizerPath = "qwen_tokenizer_path", frontend = "frontend_conf"
        case audioEncoder = "audio_encoder_conf", audioAdaptor = "audio_adaptor_conf"
        case textConfig = "text_config", llmConfig = "llm_config", defaultMaxTokens = "default_max_tokens"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "fun_asr_nano"
        inputSize = try c.decodeIfPresent(Int.self, forKey: .inputSize) ?? 560
        tokenizerPath = try c.decodeIfPresent(String.self, forKey: .tokenizerPath) ?? "Qwen3-0.6B"
        frontend = try c.decodeIfPresent(SenseVoiceFrontendConfig.self, forKey: .frontend) ?? .init()
        audioEncoder = try c.decodeIfPresent(SenseVoiceEncoderConfig.self, forKey: .audioEncoder) ?? .init()
        audioAdaptor = try c.decodeIfPresent(FunASRNanoAdaptorConfig.self, forKey: .audioAdaptor) ?? .init()
        textConfig = try c.decodeIfPresent(Qwen3TextConfig.self, forKey: .textConfig)
            ?? c.decodeIfPresent(Qwen3TextConfig.self, forKey: .llmConfig) ?? .init()
        defaultMaxTokens = try c.decodeIfPresent(Int.self, forKey: .defaultMaxTokens) ?? 512
    }
}
