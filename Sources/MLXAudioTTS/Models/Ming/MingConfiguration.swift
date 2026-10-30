import Foundation
import MLXAudioCore

public struct MingTransformerConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var depth: Int = 8
    public var numHeads: Int = 16
    public var mlpRatio: Float = 4
    public var patchSize: Int = 1
    public var historyPatchSize: Int = 1

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", depth, numHeads = "num_heads", mlpRatio = "mlp_ratio"
        case patchSize = "patch_size", historyPatchSize = "history_patch_size"
    }
    public init(hiddenSize: Int = 1024, depth: Int = 8, numHeads: Int = 16,
                mlpRatio: Float = 4, patchSize: Int = 1, historyPatchSize: Int = 1) {
        self.hiddenSize = hiddenSize; self.depth = depth; self.numHeads = numHeads
        self.mlpRatio = mlpRatio; self.patchSize = patchSize; self.historyPatchSize = historyPatchSize
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 8
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 1
        historyPatchSize = try c.decodeIfPresent(Int.self, forKey: .historyPatchSize) ?? patchSize
    }
}

public struct MingAudioBackboneConfiguration: Codable, Sendable {
    public var backbone: Qwen2Configuration
    public var inputDim: Int = 882
    public var outputDim: Int = 882
    public var hopSize: Int = 882
    public var latentDim: Int = 64
    public var useSlidingWindow: Bool = true
    public var slidingWindow: Int = 64
    public var maxWindowLayers: Int = 0
    enum CodingKeys: String, CodingKey {
        case backbone, inputDim = "input_dim", outputDim = "output_dim"
        case hopSize = "hop_size", latentDim = "latent_dim"
    }
    enum BackboneKeys: String, CodingKey {
        case useSlidingWindow = "use_sliding_window", slidingWindow = "sliding_window"
        case maxWindowLayers = "max_window_layers"
    }
    public init(backbone: Qwen2Configuration, inputDim: Int = 882, outputDim: Int = 882,
                hopSize: Int = 882, latentDim: Int = 64, slidingWindow: Int = 64) {
        self.backbone = backbone; self.inputDim = inputDim; self.outputDim = outputDim
        self.hopSize = hopSize; self.latentDim = latentDim; self.slidingWindow = slidingWindow
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        backbone = try c.decode(Qwen2Configuration.self, forKey: .backbone)
        inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? 882
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 882
        hopSize = try c.decodeIfPresent(Int.self, forKey: .hopSize) ?? inputDim
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64
        let b = try c.nestedContainer(keyedBy: BackboneKeys.self, forKey: .backbone)
        useSlidingWindow = try b.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
        slidingWindow = try b.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 64
        maxWindowLayers = try b.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? backbone.hiddenLayers
    }
    public func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(backbone, forKey: .backbone)
        try c.encode(inputDim, forKey: .inputDim); try c.encode(outputDim, forKey: .outputDim)
        try c.encode(hopSize, forKey: .hopSize); try c.encode(latentDim, forKey: .latentDim)
        var b = c.nestedContainer(keyedBy: BackboneKeys.self, forKey: .backbone)
        try b.encode(useSlidingWindow, forKey: .useSlidingWindow)
        try b.encode(slidingWindow, forKey: .slidingWindow)
        try b.encode(maxWindowLayers, forKey: .maxWindowLayers)
    }
}

public struct MingAudioVAEConfiguration: Codable, Sendable {
    public var sampleRate: Int = 44100
    public var patchSize: Int = 4
    public var encoder: MingAudioBackboneConfiguration
    public var decoder: MingAudioBackboneConfiguration
    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate", patchSize = "patch_size"
        case encoder = "enc_kwargs", decoder = "dec_kwargs"
    }
    public init(encoder: MingAudioBackboneConfiguration, decoder: MingAudioBackboneConfiguration,
                sampleRate: Int = 44100, patchSize: Int = 4) {
        self.encoder = encoder; self.decoder = decoder; self.sampleRate = sampleRate; self.patchSize = patchSize
    }
}

public struct MingDenseConfiguration: Decodable, Sendable {
    public var modelType: String
    public var text: Qwen2Configuration
    public var audio: MingAudioVAEConfiguration
    public var flow: MingTransformerConfiguration
    public var aggregator: MingTransformerConfiguration
    public var quantization: BaseConfiguration.Quantization?
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", text = "llm_config", textConfig = "text_config"
        case audio = "audio_tokenizer_config", flow = "ditar_config", aggregator = "aggregator_config"
        case quantization
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "dense"
        // The older 16.8B Bailing MoE speech model is a separate architecture.
        let raw = try c.superDecoder(forKey: c.contains(.text) ? .text : .textConfig)
        let t = try raw.container(keyedBy: MingBackboneKindKeys.self)
        if try t.decodeIfPresent(Int.self, forKey: .numExperts) != nil {
            throw DecodingError.dataCorrupted(.init(codingPath: c.codingPath,
                debugDescription: "Ming dense supports Qwen2.5 only; the Bailing MoE backbone is unsupported."))
        }
        text = try Qwen2Configuration(from: raw)
        audio = try c.decode(MingAudioVAEConfiguration.self, forKey: .audio)
        flow = try c.decode(MingTransformerConfiguration.self, forKey: .flow)
        aggregator = try c.decode(MingTransformerConfiguration.self, forKey: .aggregator)
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
    }
    public var isMing3: Bool { flow.patchSize == 1 && flow.historyPatchSize == 1 }
}
private enum MingBackboneKindKeys: String, CodingKey { case numExperts = "num_experts" }
