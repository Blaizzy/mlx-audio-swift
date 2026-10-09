import Foundation

public struct IndexTTSConformerConfig: Codable, Sendable {
    public let inputSize: Int
    public let outputSize: Int
    public let numBlocks: Int
    public let linearUnits: Int
    public let attentionHeads: Int
    public let posEncLayerType: String
    public let inputLayer: String
    public let cnnModuleKernel: Int
    public let posEmbMaxLen: Int
    public let causalDownsampling: Bool
    public let useBias: Bool
    public let xscaling: Bool
    public let macaronStyle: Bool
    public let perceiverMult: Int

    public init(
        inputSize: Int = 100,
        outputSize: Int = 256,
        numBlocks: Int = 6,
        linearUnits: Int = 2048,
        attentionHeads: Int = 4,
        posEncLayerType: String = "rel_pos",
        inputLayer: String = "conv2d",
        cnnModuleKernel: Int = 15,
        posEmbMaxLen: Int = 2048,
        causalDownsampling: Bool = false,
        useBias: Bool = true,
        xscaling: Bool = true,
        macaronStyle: Bool = false,
        perceiverMult: Int = 2
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numBlocks = numBlocks
        self.linearUnits = linearUnits
        self.attentionHeads = attentionHeads
        self.posEncLayerType = posEncLayerType
        self.inputLayer = inputLayer
        self.cnnModuleKernel = cnnModuleKernel
        self.posEmbMaxLen = posEmbMaxLen
        self.causalDownsampling = causalDownsampling
        self.useBias = useBias
        self.xscaling = xscaling
        self.macaronStyle = macaronStyle
        self.perceiverMult = perceiverMult
    }

    enum CodingKeys: String, CodingKey {
        case inputSize = "input_size"
        case outputSize = "output_size"
        case numBlocks = "num_blocks"
        case linearUnits = "linear_units"
        case attentionHeads = "attention_heads"
        case posEncLayerType = "pos_enc_layer_type"
        case inputLayer = "input_layer"
        case cnnModuleKernel = "cnn_module_kernel"
        case posEmbMaxLen = "pos_emb_max_len"
        case causalDownsampling = "causal_downsampling"
        case useBias = "use_bias"
        case xscaling
        case macaronStyle = "macaron_style"
        case perceiverMult = "perceiver_mult"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            inputSize: try c.decodeIfPresent(Int.self, forKey: .inputSize) ?? 100,
            outputSize: try c.decodeIfPresent(Int.self, forKey: .outputSize) ?? 256,
            numBlocks: try c.decodeIfPresent(Int.self, forKey: .numBlocks) ?? 6,
            linearUnits: try c.decodeIfPresent(Int.self, forKey: .linearUnits) ?? 2048,
            attentionHeads: try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4,
            posEncLayerType: try c.decodeIfPresent(String.self, forKey: .posEncLayerType) ?? "rel_pos",
            inputLayer: try c.decodeIfPresent(String.self, forKey: .inputLayer) ?? "conv2d",
            cnnModuleKernel: try c.decodeIfPresent(Int.self, forKey: .cnnModuleKernel) ?? 15,
            posEmbMaxLen: try c.decodeIfPresent(Int.self, forKey: .posEmbMaxLen) ?? 2048,
            causalDownsampling: try c.decodeIfPresent(Bool.self, forKey: .causalDownsampling) ?? false,
            useBias: try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? true,
            xscaling: try c.decodeIfPresent(Bool.self, forKey: .xscaling) ?? true,
            macaronStyle: try c.decodeIfPresent(Bool.self, forKey: .macaronStyle) ?? false,
            perceiverMult: try c.decodeIfPresent(Int.self, forKey: .perceiverMult) ?? 2
        )
    }
}

public struct IndexTTSGPTConfig: Codable, Sendable {
    public let modelDim: Int
    public let heads: Int
    public let layers: Int
    public let maxMelTokens: Int
    public let maxTextTokens: Int
    public let numberTextTokens: Int
    public let numberMelCodes: Int
    public let startMelToken: Int
    public let stopMelToken: Int
    public let startTextToken: Int
    public let stopTextToken: Int
    public let useMelCodesAsInput: Bool
    public let melLengthCompression: Int
    public let conditionType: String
    public let conditionModule: IndexTTSConformerConfig
    public let maxConditioningInputs: Int
    public let conditionNumLatent: Int

    public init(
        modelDim: Int,
        heads: Int,
        layers: Int,
        maxMelTokens: Int,
        maxTextTokens: Int,
        numberTextTokens: Int,
        numberMelCodes: Int,
        startMelToken: Int,
        stopMelToken: Int,
        startTextToken: Int,
        stopTextToken: Int,
        useMelCodesAsInput: Bool,
        melLengthCompression: Int,
        conditionType: String,
        conditionModule: IndexTTSConformerConfig,
        maxConditioningInputs: Int = 1,
        conditionNumLatent: Int = 32
    ) {
        self.modelDim = modelDim
        self.heads = heads
        self.layers = layers
        self.maxMelTokens = maxMelTokens
        self.maxTextTokens = maxTextTokens
        self.numberTextTokens = numberTextTokens
        self.numberMelCodes = numberMelCodes
        self.startMelToken = startMelToken
        self.stopMelToken = stopMelToken
        self.startTextToken = startTextToken
        self.stopTextToken = stopTextToken
        self.useMelCodesAsInput = useMelCodesAsInput
        self.melLengthCompression = melLengthCompression
        self.conditionType = conditionType
        self.conditionModule = conditionModule
        self.maxConditioningInputs = maxConditioningInputs
        self.conditionNumLatent = conditionNumLatent
    }

    enum CodingKeys: String, CodingKey {
        case modelDim = "model_dim"
        case heads
        case layers
        case maxMelTokens = "max_mel_tokens"
        case maxTextTokens = "max_text_tokens"
        case numberTextTokens = "number_text_tokens"
        case numberMelCodes = "number_mel_codes"
        case startMelToken = "start_mel_token"
        case stopMelToken = "stop_mel_token"
        case startTextToken = "start_text_token"
        case stopTextToken = "stop_text_token"
        case useMelCodesAsInput = "use_mel_codes_as_input"
        case melLengthCompression = "mel_length_compression"
        case conditionType = "condition_type"
        case conditionModule = "condition_module"
        case maxConditioningInputs = "max_conditioning_inputs"
        case conditionNumLatent = "condition_num_latent"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelDim: try c.decode(Int.self, forKey: .modelDim),
            heads: try c.decode(Int.self, forKey: .heads),
            layers: try c.decode(Int.self, forKey: .layers),
            maxMelTokens: try c.decode(Int.self, forKey: .maxMelTokens),
            maxTextTokens: try c.decode(Int.self, forKey: .maxTextTokens),
            numberTextTokens: try c.decode(Int.self, forKey: .numberTextTokens),
            numberMelCodes: try c.decode(Int.self, forKey: .numberMelCodes),
            startMelToken: try c.decode(Int.self, forKey: .startMelToken),
            stopMelToken: try c.decode(Int.self, forKey: .stopMelToken),
            startTextToken: try c.decode(Int.self, forKey: .startTextToken),
            stopTextToken: try c.decode(Int.self, forKey: .stopTextToken),
            useMelCodesAsInput: try c.decodeIfPresent(Bool.self, forKey: .useMelCodesAsInput) ?? true,
            melLengthCompression: try c.decodeIfPresent(Int.self, forKey: .melLengthCompression) ?? 1024,
            conditionType: try c.decodeIfPresent(String.self, forKey: .conditionType) ?? "conformer_perceiver",
            conditionModule: try c.decodeIfPresent(IndexTTSConformerConfig.self, forKey: .conditionModule) ?? IndexTTSConformerConfig(),
            maxConditioningInputs: try c.decodeIfPresent(Int.self, forKey: .maxConditioningInputs) ?? 1,
            conditionNumLatent: try c.decodeIfPresent(Int.self, forKey: .conditionNumLatent) ?? 32
        )
    }

    public static func tinyForTests() -> IndexTTSGPTConfig {
        IndexTTSGPTConfig(
            modelDim: 8,
            heads: 2,
            layers: 1,
            maxMelTokens: 8,
            maxTextTokens: 8,
            numberTextTokens: 16,
            numberMelCodes: 8,
            startMelToken: 6,
            stopMelToken: 7,
            startTextToken: 14,
            stopTextToken: 15,
            useMelCodesAsInput: true,
            melLengthCompression: 2,
            conditionType: "conformer_perceiver",
            conditionModule: IndexTTSConformerConfig(
                inputSize: 4,
                outputSize: 8,
                numBlocks: 1,
                linearUnits: 16,
                attentionHeads: 2,
                perceiverMult: 2
            ),
            maxConditioningInputs: 1,
            conditionNumLatent: 2
        )
    }
}

public struct IndexTTSBigVGANConditioningConfig: Codable, Sendable {
    public let numMels: Int
    public let upsampleRates: [Int]
    public let upsampleKernelSizes: [Int]
    public let upsampleInitialChannel: Int
    public let resblock: String
    public let resblockKernelSizes: [Int]
    public let resblockDilationSizes: [[Int]]
    public let activation: String
    public let snakeLogscale: Bool
    public let useBiasAtFinal: Bool
    public let useTanhAtFinal: Bool
    public let gptDim: Int
    public let speakerEmbeddingDim: Int
    public let condDVectorInEachUpsamplingLayer: Bool

    public init(
        numMels: Int = 100,
        upsampleRates: [Int] = [8, 8, 2, 2],
        upsampleKernelSizes: [Int] = [16, 16, 4, 4],
        upsampleInitialChannel: Int = 512,
        resblock: String = "1",
        resblockKernelSizes: [Int] = [3, 7, 11],
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        activation: String = "snakebeta",
        snakeLogscale: Bool = true,
        useBiasAtFinal: Bool = true,
        useTanhAtFinal: Bool = true,
        gptDim: Int = 1,
        speakerEmbeddingDim: Int = 1,
        condDVectorInEachUpsamplingLayer: Bool = true
    ) {
        self.numMels = numMels
        self.upsampleRates = upsampleRates
        self.upsampleKernelSizes = upsampleKernelSizes
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblock = resblock
        self.resblockKernelSizes = resblockKernelSizes
        self.resblockDilationSizes = resblockDilationSizes
        self.activation = activation
        self.snakeLogscale = snakeLogscale
        self.useBiasAtFinal = useBiasAtFinal
        self.useTanhAtFinal = useTanhAtFinal
        self.gptDim = gptDim
        self.speakerEmbeddingDim = speakerEmbeddingDim
        self.condDVectorInEachUpsamplingLayer = condDVectorInEachUpsamplingLayer
    }

    enum CodingKeys: String, CodingKey {
        case numMels = "num_mels"
        case upsampleRates = "upsample_rates"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case upsampleInitialChannel = "upsample_initial_channel"
        case resblock
        case resblockKernelSizes = "resblock_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case activation
        case snakeLogscale = "snake_logscale"
        case useBiasAtFinal = "use_bias_at_final"
        case useTanhAtFinal = "use_tanh_at_final"
        case gptDim = "gpt_dim"
        case speakerEmbeddingDim = "speaker_embedding_dim"
        case condDVectorInEachUpsamplingLayer = "cond_d_vector_in_each_upsampling_layer"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            numMels: try c.decodeIfPresent(Int.self, forKey: .numMels) ?? 100,
            upsampleRates: try c.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 8, 2, 2],
            upsampleKernelSizes: try c.decodeIfPresent([Int].self, forKey: .upsampleKernelSizes) ?? [16, 16, 4, 4],
            upsampleInitialChannel: try c.decodeIfPresent(Int.self, forKey: .upsampleInitialChannel) ?? 512,
            resblock: try c.decodeIfPresent(String.self, forKey: .resblock) ?? "1",
            resblockKernelSizes: try c.decodeIfPresent([Int].self, forKey: .resblockKernelSizes) ?? [3, 7, 11],
            resblockDilationSizes: try c.decodeIfPresent([[Int]].self, forKey: .resblockDilationSizes) ?? [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation: try c.decodeIfPresent(String.self, forKey: .activation) ?? "snakebeta",
            snakeLogscale: try c.decodeIfPresent(Bool.self, forKey: .snakeLogscale) ?? true,
            useBiasAtFinal: try c.decodeIfPresent(Bool.self, forKey: .useBiasAtFinal) ?? true,
            useTanhAtFinal: try c.decodeIfPresent(Bool.self, forKey: .useTanhAtFinal) ?? true,
            gptDim: try c.decodeIfPresent(Int.self, forKey: .gptDim) ?? 1,
            speakerEmbeddingDim: try c.decodeIfPresent(Int.self, forKey: .speakerEmbeddingDim) ?? 1,
            condDVectorInEachUpsamplingLayer: try c.decodeIfPresent(Bool.self, forKey: .condDVectorInEachUpsamplingLayer) ?? true
        )
    }
}

public struct IndexTTSConfig: Codable, Sendable {
    public let modelType: String
    public let bigvgan: IndexTTSBigVGANConditioningConfig
    public let gpt: IndexTTSGPTConfig
    public let tokenizerName: String
    public let sampleRate: Int

    public init(
        modelType: String = "indextts",
        bigvgan: IndexTTSBigVGANConditioningConfig,
        gpt: IndexTTSGPTConfig,
        tokenizerName: String,
        sampleRate: Int = 24_000
    ) {
        self.modelType = modelType
        self.bigvgan = bigvgan
        self.gpt = gpt
        self.tokenizerName = tokenizerName
        self.sampleRate = sampleRate
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case bigvgan
        case gpt
        case tokenizerName = "tokenizer_name"
        case sampleRate = "sample_rate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try c.decodeIfPresent(String.self, forKey: .modelType) ?? "indextts",
            bigvgan: try c.decodeIfPresent(IndexTTSBigVGANConditioningConfig.self, forKey: .bigvgan)
                ?? IndexTTSBigVGANConditioningConfig(),
            gpt: try c.decode(IndexTTSGPTConfig.self, forKey: .gpt),
            tokenizerName: try c.decodeIfPresent(String.self, forKey: .tokenizerName) ?? "",
            sampleRate: try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
        )
    }

    public static func tinyForTests() -> IndexTTSConfig {
        let gpt = IndexTTSGPTConfig.tinyForTests()
        return IndexTTSConfig(
            bigvgan: IndexTTSBigVGANConditioningConfig(
                numMels: 4,
                upsampleRates: [2],
                upsampleKernelSizes: [4],
                upsampleInitialChannel: 8,
                gptDim: gpt.modelDim,
                speakerEmbeddingDim: 4
            ),
            gpt: gpt,
            tokenizerName: "",
            sampleRate: 24_000
        )
    }
}
