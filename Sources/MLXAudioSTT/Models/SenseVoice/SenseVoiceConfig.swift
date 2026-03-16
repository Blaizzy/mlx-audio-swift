import Foundation

public struct SenseVoiceEncoderConfig: Decodable, Sendable {
    public let outputSize: Int
    public let attentionHeads: Int
    public let linearUnits: Int
    public let numBlocks: Int
    public let tpBlocks: Int
    public let dropoutRate: Float
    public let positionalDropoutRate: Float
    public let attentionDropoutRate: Float
    public let kernelSize: Int
    public let sanmShift: Int
    public let normalizeBefore: Bool

    public init(
        outputSize: Int = 512,
        attentionHeads: Int = 4,
        linearUnits: Int = 2048,
        numBlocks: Int = 50,
        tpBlocks: Int = 20,
        dropoutRate: Float = 0.1,
        positionalDropoutRate: Float = 0.1,
        attentionDropoutRate: Float = 0.1,
        kernelSize: Int = 11,
        sanmShift: Int = 0,
        normalizeBefore: Bool = true
    ) {
        self.outputSize = outputSize
        self.attentionHeads = attentionHeads
        self.linearUnits = linearUnits
        self.numBlocks = numBlocks
        self.tpBlocks = tpBlocks
        self.dropoutRate = dropoutRate
        self.positionalDropoutRate = positionalDropoutRate
        self.attentionDropoutRate = attentionDropoutRate
        self.kernelSize = kernelSize
        self.sanmShift = sanmShift
        self.normalizeBefore = normalizeBefore
    }

    enum CodingKeys: String, CodingKey {
        case outputSize = "output_size"
        case attentionHeads = "attention_heads"
        case linearUnits = "linear_units"
        case numBlocks = "num_blocks"
        case tpBlocks = "tp_blocks"
        case dropoutRate = "dropout_rate"
        case positionalDropoutRate = "positional_dropout_rate"
        case attentionDropoutRate = "attention_dropout_rate"
        case kernelSize = "kernel_size"
        case sanmShift = "sanm_shift"
        case sanmShfit = "sanm_shfit"
        case normalizeBefore = "normalize_before"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        outputSize = try container.decodeIfPresent(Int.self, forKey: .outputSize) ?? 512
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        linearUnits = try container.decodeIfPresent(Int.self, forKey: .linearUnits) ?? 2048
        numBlocks = try container.decodeIfPresent(Int.self, forKey: .numBlocks) ?? 50
        tpBlocks = try container.decodeIfPresent(Int.self, forKey: .tpBlocks) ?? 20
        dropoutRate = try container.decodeIfPresent(Float.self, forKey: .dropoutRate) ?? 0.1
        positionalDropoutRate = try container.decodeIfPresent(Float.self, forKey: .positionalDropoutRate) ?? 0.1
        attentionDropoutRate = try container.decodeIfPresent(Float.self, forKey: .attentionDropoutRate) ?? 0.1
        kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 11
        sanmShift = try container.decodeIfPresent(Int.self, forKey: .sanmShift)
            ?? container.decodeIfPresent(Int.self, forKey: .sanmShfit)
            ?? 0
        normalizeBefore = try container.decodeIfPresent(Bool.self, forKey: .normalizeBefore) ?? true
    }
}

public struct SenseVoiceFrontendConfig: Decodable, Sendable {
    public let fs: Int
    public let window: String
    public let nMels: Int
    public let frameLength: Int
    public let frameShift: Int
    public let lfrM: Int
    public let lfrN: Int

    public init(
        fs: Int = 16000,
        window: String = "hamming",
        nMels: Int = 80,
        frameLength: Int = 25,
        frameShift: Int = 10,
        lfrM: Int = 7,
        lfrN: Int = 6
    ) {
        self.fs = fs
        self.window = window
        self.nMels = nMels
        self.frameLength = frameLength
        self.frameShift = frameShift
        self.lfrM = lfrM
        self.lfrN = lfrN
    }

    enum CodingKeys: String, CodingKey {
        case fs
        case window
        case nMels = "n_mels"
        case frameLength = "frame_length"
        case frameShift = "frame_shift"
        case lfrM = "lfr_m"
        case lfrN = "lfr_n"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        fs = try container.decodeIfPresent(Int.self, forKey: .fs) ?? 16000
        window = try container.decodeIfPresent(String.self, forKey: .window) ?? "hamming"
        nMels = try container.decodeIfPresent(Int.self, forKey: .nMels) ?? 80
        frameLength = try container.decodeIfPresent(Int.self, forKey: .frameLength) ?? 25
        frameShift = try container.decodeIfPresent(Int.self, forKey: .frameShift) ?? 10
        lfrM = try container.decodeIfPresent(Int.self, forKey: .lfrM) ?? 7
        lfrN = try container.decodeIfPresent(Int.self, forKey: .lfrN) ?? 6
    }
}

public struct SenseVoiceConfig: Decodable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let inputSize: Int
    public let encoderConf: SenseVoiceEncoderConfig
    public let frontendConf: SenseVoiceFrontendConfig
    public let cmvnMeans: [Float]?
    public let cmvnIstd: [Float]?

    public init(
        modelType: String = "sensevoice",
        vocabSize: Int = 25055,
        inputSize: Int = 560,
        encoderConf: SenseVoiceEncoderConfig = SenseVoiceEncoderConfig(),
        frontendConf: SenseVoiceFrontendConfig = SenseVoiceFrontendConfig(),
        cmvnMeans: [Float]? = nil,
        cmvnIstd: [Float]? = nil
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.inputSize = inputSize
        self.encoderConf = encoderConf
        self.frontendConf = frontendConf
        self.cmvnMeans = cmvnMeans
        self.cmvnIstd = cmvnIstd
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case inputSize = "input_size"
        case encoderConf = "encoder_conf"
        case frontendConf = "frontend_conf"
        case cmvnMeans = "cmvn_means"
        case cmvnIstd = "cmvn_istd"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "sensevoice"
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 25055
        inputSize = try container.decodeIfPresent(Int.self, forKey: .inputSize) ?? 560
        encoderConf = try container.decodeIfPresent(SenseVoiceEncoderConfig.self, forKey: .encoderConf)
            ?? SenseVoiceEncoderConfig()
        frontendConf = try container.decodeIfPresent(SenseVoiceFrontendConfig.self, forKey: .frontendConf)
            ?? SenseVoiceFrontendConfig()
        cmvnMeans = try container.decodeIfPresent([Float].self, forKey: .cmvnMeans)
        cmvnIstd = try container.decodeIfPresent([Float].self, forKey: .cmvnIstd)
    }
}
