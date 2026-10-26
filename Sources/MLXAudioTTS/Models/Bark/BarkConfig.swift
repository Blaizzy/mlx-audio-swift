import Foundation
import MLXAudioCore

public struct BarkStageConfig: Codable, Sendable {
    public var blockSize: Int = 1024
    public var inputVocabSize: Int
    public var outputVocabSize: Int
    public var layers: Int = 12
    public var heads: Int = 12
    public var width: Int = 768
    public var bias: Bool = false
    public var codebooks: Int = 8
    public var givenCodebooks: Int = 1

    public init(inputVocabSize: Int, outputVocabSize: Int, blockSize: Int = 1024,
                layers: Int = 12, heads: Int = 12, width: Int = 768,
                bias: Bool = false, codebooks: Int = 8, givenCodebooks: Int = 1) {
        self.inputVocabSize = inputVocabSize; self.outputVocabSize = outputVocabSize
        self.blockSize = blockSize; self.layers = layers; self.heads = heads
        self.width = width; self.bias = bias; self.codebooks = codebooks
        self.givenCodebooks = givenCodebooks
    }

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size", inputVocabSize = "input_vocab_size"
        case outputVocabSize = "output_vocab_size", layers = "n_layer", heads = "n_head"
        case width = "n_embd", bias, codebooks = "n_codes_total", givenCodebooks = "n_codes_given"
        case numLayers = "num_layers", numHeads = "num_heads", hiddenSize = "hidden_size"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputVocabSize = try c.decode(Int.self, forKey: .inputVocabSize)
        outputVocabSize = try c.decode(Int.self, forKey: .outputVocabSize)
        blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 1024
        layers = try c.decodeIfPresent(Int.self, forKey: .layers)
            ?? c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
        heads = try c.decodeIfPresent(Int.self, forKey: .heads)
            ?? c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 12
        width = try c.decodeIfPresent(Int.self, forKey: .width)
            ?? c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        bias = try c.decodeIfPresent(Bool.self, forKey: .bias) ?? false
        codebooks = try c.decodeIfPresent(Int.self, forKey: .codebooks) ?? 8
        givenCodebooks = try c.decodeIfPresent(Int.self, forKey: .givenCodebooks) ?? 1
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(blockSize, forKey: .blockSize); try c.encode(inputVocabSize, forKey: .inputVocabSize)
        try c.encode(outputVocabSize, forKey: .outputVocabSize); try c.encode(layers, forKey: .layers)
        try c.encode(heads, forKey: .heads); try c.encode(width, forKey: .width)
        try c.encode(bias, forKey: .bias); try c.encode(codebooks, forKey: .codebooks)
        try c.encode(givenCodebooks, forKey: .givenCodebooks)
    }
}

public struct BarkConfig: Decodable, Sendable {
    public let semantic: BarkStageConfig
    public let coarse: BarkStageConfig
    public let fine: BarkStageConfig
    public let codecPath: String
    public let sampleRate: Int
    public let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case semantic = "semantic_config", coarse = "coarse_acoustics_config"
        case fine = "fine_acoustics_config", codecPath = "codec_path", sampleRate = "sample_rate"
        case quantization
    }
    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        semantic = try c.decode(BarkStageConfig.self, forKey: .semantic)
        coarse = try c.decode(BarkStageConfig.self, forKey: .coarse)
        fine = try c.decode(BarkStageConfig.self, forKey: .fine)
        codecPath = try c.decodeIfPresent(String.self, forKey: .codecPath)
            ?? "mlx-community/encodec-24khz-float32"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
    }
}

/// Stage controls. `GenerateParameters.maxTokens` bounds semantic generation (at 49.9 Hz).
public struct BarkGenerationOptions: Sendable {
    public var maxCoarseHistory = 60
    public var coarseWindow = 60
    public var allowEarlyStop = true
    public var useFineHistory = false
    public var fineTemperature: Float? = nil
    public init() {}
}
