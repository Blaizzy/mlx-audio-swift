import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

public struct HiggsAudioTokenizerConfig: Codable, Sendable {
    public var modelType: String
    public var sampleRate: Int
    public var codebookSize: Int
    public var codebookDim: Int
    public var downsampleFactor: Int
    public var dacSampleRate: Int
    public var dacNumCodebooks: Int
    public var dacEncoderRatios: [Int]
    public var dacEncoderHidden: Int
    public var dacDecoderHidden: Int

    public init(
        modelType: String = "higgs_audio_v2_tokenizer",
        sampleRate: Int = 24_000,
        codebookSize: Int = 1_024,
        codebookDim: Int = 64,
        downsampleFactor: Int = 320,
        dacSampleRate: Int = 24_000,
        dacNumCodebooks: Int = 8,
        dacEncoderRatios: [Int] = [8, 5, 4, 2, 3],
        dacEncoderHidden: Int = 64,
        dacDecoderHidden: Int = 1_024
    ) {
        self.modelType = modelType
        self.sampleRate = sampleRate
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.downsampleFactor = downsampleFactor
        self.dacSampleRate = dacSampleRate
        self.dacNumCodebooks = dacNumCodebooks
        self.dacEncoderRatios = dacEncoderRatios
        self.dacEncoderHidden = dacEncoderHidden
        self.dacDecoderHidden = dacDecoderHidden
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case downsampleFactor = "downsample_factor"
        case dacSampleRate = "dac_sample_rate"
        case dacNumCodebooks = "dac_num_codebooks"
        case dacEncoderRatios = "dac_encoder_ratios"
        case dacEncoderHidden = "dac_encoder_hidden"
        case dacDecoderHidden = "dac_decoder_hidden"
    }
}

public enum HiggsAudioTokenizerError: Error, LocalizedError {
    case missingConfig(URL)
    case missingWeights(URL)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            "Missing Higgs audio tokenizer config at \(url.path)"
        case .missingWeights(let url):
            "Missing Higgs audio tokenizer weights at \(url.path)"
        case .invalidInput(let message):
            message
        }
    }
}

private func higgsSnake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let sinValue = sin(alpha * x)
    return x + (sinValue * sinValue) / (alpha + MLXArray(Float(1e-9)))
}

private final class HiggsSnake1d: Module {
    @ModuleInfo var alpha: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        higgsSnake(x, alpha: alpha)
    }
}

private final class HiggsConv1d: Module {
    let stride: Int
    let padding: Int
    let dilation: Int

    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1) {
        self.stride = stride
        self.padding = padding == 0 ? (kernelSize - stride) * dilation / 2 : padding
        self.dilation = dilation
        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        _weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outChannels, kernelSize, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv1d(x, weight, stride: stride, padding: padding, dilation: dilation) + bias
    }
}

private final class HiggsConvTranspose1d: Module {
    let stride: Int
    let padding: Int
    let expectedStride: Int

    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.stride = stride
        self.padding = padding
        self.expectedStride = stride
        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        _weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outChannels, kernelSize, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let expected = x.dim(1) * expectedStride
        var y = convTransposed1d(x, weight, stride: stride, padding: padding) + bias
        if y.dim(1) > expected {
            y = y[0..., 0..<expected, 0...]
        }
        return y
    }
}

private final class HiggsResidualUnit: Module {
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d
    @ModuleInfo(key: "snake2") var snake2: HiggsSnake1d
    @ModuleInfo(key: "conv2") var conv2: HiggsConv1d

    init(dim: Int, dilation: Int = 1) {
        _snake1.wrappedValue = HiggsSnake1d(channels: dim)
        _conv1.wrappedValue = HiggsConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, dilation: dilation)
        _snake2.wrappedValue = HiggsSnake1d(channels: dim)
        _conv2.wrappedValue = HiggsConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = snake1(x)
        y = conv1(y)
        y = snake2(y)
        y = conv2(y)
        let pad = (x.dim(1) - y.dim(1)) / 2
        let residual = pad > 0 ? x[0..., pad..<(x.dim(1) - pad), 0...] : x
        return residual + y
    }
}

private final class HiggsAcousticEncoderBlock: Module {
    @ModuleInfo(key: "res_unit1") var resUnit1: HiggsResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: HiggsResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: HiggsResidualUnit
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d

    init(inDim: Int, outDim: Int, stride: Int) {
        _resUnit1.wrappedValue = HiggsResidualUnit(dim: inDim, dilation: 1)
        _resUnit2.wrappedValue = HiggsResidualUnit(dim: inDim, dilation: 3)
        _resUnit3.wrappedValue = HiggsResidualUnit(dim: inDim, dilation: 9)
        _snake1.wrappedValue = HiggsSnake1d(channels: inDim)
        _conv1.wrappedValue = HiggsConv1d(inChannels: inDim, outChannels: outDim, kernelSize: 2 * stride, stride: stride, padding: Int(ceil(Float(stride) / 2.0)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = resUnit1(x)
        y = resUnit2(y)
        y = resUnit3(y)
        y = snake1(y)
        return conv1(y)
    }
}

private final class HiggsAcousticDecoderBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv_t1") var convT1: HiggsConvTranspose1d
    @ModuleInfo(key: "res_unit1") var resUnit1: HiggsResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: HiggsResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: HiggsResidualUnit

    init(inDim: Int, outDim: Int, stride: Int) {
        _snake1.wrappedValue = HiggsSnake1d(channels: inDim)
        _convT1.wrappedValue = HiggsConvTranspose1d(
            inChannels: inDim,
            outChannels: outDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: stride / 2
        )
        _resUnit1.wrappedValue = HiggsResidualUnit(dim: outDim, dilation: 1)
        _resUnit2.wrappedValue = HiggsResidualUnit(dim: outDim, dilation: 3)
        _resUnit3.wrappedValue = HiggsResidualUnit(dim: outDim, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = snake1(x)
        y = convT1(y)
        y = resUnit1(y)
        y = resUnit2(y)
        y = resUnit3(y)
        return y
    }
}

private final class HiggsAcousticEncoder: Module {
    private static let strides = [8, 5, 4, 2, 3]
    private static let channels = [64, 128, 256, 512, 1_024, 2_048]

    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d
    @ModuleInfo(key: "block") var block: [HiggsAcousticEncoderBlock]
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv2") var conv2: HiggsConv1d

    override init() {
        _conv1.wrappedValue = HiggsConv1d(inChannels: 1, outChannels: 64, kernelSize: 7, padding: 3)
        _block.wrappedValue = Self.strides.enumerated().map { index, stride in
            HiggsAcousticEncoderBlock(
                inDim: Self.channels[index],
                outDim: Self.channels[index + 1],
                stride: stride
            )
        }
        _snake1.wrappedValue = HiggsSnake1d(channels: 2_048)
        _conv2.wrappedValue = HiggsConv1d(inChannels: 2_048, outChannels: 256, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1(x)
        for block in block {
            y = block(y)
        }
        y = snake1(y)
        return conv2(y)
    }
}

private final class HiggsAcousticDecoder: Module {
    private static let strides = [8, 5, 4, 2, 3]
    private static let inChannels = [1_024, 512, 256, 128, 64]
    private static let outChannels = [512, 256, 128, 64, 32]

    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d
    @ModuleInfo(key: "block") var block: [HiggsAcousticDecoderBlock]
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv2") var conv2: HiggsConv1d

    override init() {
        _conv1.wrappedValue = HiggsConv1d(inChannels: 256, outChannels: 1_024, kernelSize: 7, padding: 3)
        _block.wrappedValue = Self.strides.enumerated().map { index, stride in
            HiggsAcousticDecoderBlock(
                inDim: Self.inChannels[index],
                outDim: Self.outChannels[index],
                stride: stride
            )
        }
        _snake1.wrappedValue = HiggsSnake1d(channels: 32)
        _conv2.wrappedValue = HiggsConv1d(inChannels: 32, outChannels: 1, kernelSize: 7, padding: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1(x)
        for block in block {
            y = block(y)
        }
        y = snake1(y)
        return conv2(y)
    }
}

private final class HiggsVectorQuantizer: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo var codebook: Embedding
    @ModuleInfo(key: "project_out") var projectOut: Linear

    init(latentDim: Int = 1_024, codebookSize: Int = 1_024, codebookDim: Int = 64) {
        _projectIn.wrappedValue = Linear(latentDim, codebookDim)
        _codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
        _projectOut.wrappedValue = Linear(codebookDim, latentDim)
    }

    func decodeCodes(_ codes: MLXArray) -> MLXArray {
        projectOut(codebook(codes))
    }

    func encode(_ z: MLXArray) -> MLXArray {
        let zq = projectIn(z)
        let distances = sum(zq * zq, axis: -1, keepDims: true)
            + sum(codebook.weight * codebook.weight, axis: -1)
            - 2 * matmul(zq, codebook.weight.T)
        return argMin(distances, axis: -1).asType(.int32)
    }
}

private final class HiggsResidualVectorQuantizer: Module {
    @ModuleInfo var quantizers: [HiggsVectorQuantizer]

    init(nCodebooks: Int = 8, latentDim: Int = 1_024, codebookSize: Int = 1_024, codebookDim: Int = 64) {
        _quantizers.wrappedValue = (0..<nCodebooks).map { _ in
            HiggsVectorQuantizer(latentDim: latentDim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let nCodebooks = quantizers.count
        var sum: MLXArray?
        for index in 0..<nCodebooks {
            let decoded = quantizers[index].decodeCodes(codes[0..., 0..., index])
            sum = sum == nil ? decoded : sum! + decoded
        }
        return sum ?? MLXArray.zeros([codes.dim(0), codes.dim(1), 1_024], type: Float.self)
    }

    func encode(_ z: MLXArray) -> MLXArray {
        var residual = z
        var tokens: [MLXArray] = []
        for quantizer in quantizers {
            let indices = quantizer.encode(residual)
            tokens.append(indices)
            residual = residual - quantizer.decodeCodes(indices)
        }
        return stacked(tokens, axis: -1).asType(.int32)
    }
}

public final class HiggsAudioTokenizer: Module {
    public static let defaultCodecPrefix = "tied.embedding.modality_embeddings.0.model."

    public let config: HiggsAudioTokenizerConfig

    @ModuleInfo(key: "acoustic_encoder") private var acousticEncoder: HiggsAcousticEncoder
    @ModuleInfo private var quantizer: HiggsResidualVectorQuantizer
    @ModuleInfo(key: "acoustic_decoder") private var acousticDecoder: HiggsAcousticDecoder
    @ModuleInfo var fc2: Linear

    public init(config: HiggsAudioTokenizerConfig = HiggsAudioTokenizerConfig()) {
        self.config = config
        _acousticEncoder.wrappedValue = HiggsAcousticEncoder()
        _quantizer.wrappedValue = HiggsResidualVectorQuantizer(
            nCodebooks: config.dacNumCodebooks,
            codebookSize: config.codebookSize,
            codebookDim: config.codebookDim
        )
        _acousticDecoder.wrappedValue = HiggsAcousticDecoder()
        _fc2.wrappedValue = Linear(1_024, 256)
    }

    public func decode(_ tokens: MLXArray) -> MLXArray {
        let squeeze = tokens.ndim == 2
        let batched = squeeze ? tokens.expandedDimensions(axis: 0) : tokens
        var z = quantizer.decode(batched.asType(.int32))
        z = fc2(z)
        let waveform = acousticDecoder(z)
        return squeeze ? waveform[0, 0..., 0] : waveform
    }

    public func encodeAcoustic(_ waveform: MLXArray) -> MLXArray {
        let batched = waveform.ndim == 2 ? waveform.expandedDimensions(axis: -1) : waveform
        let features = acousticEncoder(batched.asType(.float32))
        return quantizer.encode(features)
    }

    public static func fromPretrained(
        _ source: String,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> HiggsAudioTokenizer {
        guard let repoID = Repo.ID(rawValue: source) else {
            return try fromModelDirectory(URL(fileURLWithPath: NSString(string: source).expandingTildeInPath))
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.json", "*.index.json"],
            hfToken: hfToken,
            cache: cache
        )
        return try fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ directory: URL) throws -> HiggsAudioTokenizer {
        let modelDir = resolveTokenizerDirectory(directory)
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw HiggsAudioTokenizerError.missingConfig(configURL)
        }
        let config = try JSONDecoder().decode(HiggsAudioTokenizerConfig.self, from: Data(contentsOf: configURL))
        let model = HiggsAudioTokenizer(config: config)
        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw HiggsAudioTokenizerError.missingWeights(weightsURL)
        }
        let weights = try loadArrays(url: weightsURL)
        try model.update(parameters: ModuleParameters.unflattened(sanitize(weights: weights)), verify: .noUnusedKeys)
        eval(model)
        return model
    }

    private static func resolveTokenizerDirectory(_ directory: URL) -> URL {
        let nested = directory.appendingPathComponent("audio_tokenizer")
        if FileManager.default.fileExists(atPath: nested.appendingPathComponent("config.json").path) {
            return nested
        }
        return directory
    }

    public static func sanitize(weights: [String: MLXArray], prefix: String = "") -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (rawKey, rawValue) in weights {
            var key = rawKey
            if !prefix.isEmpty {
                guard key.hasPrefix(prefix) else { continue }
                key = String(key.dropFirst(prefix.count))
            }
            if key == "semantic_model.masked_spec_embed" { continue }
            if key.hasPrefix("decoder_semantic.") || key.hasPrefix("fc1.") { continue }
            if key.hasSuffix(".embed_avg") || key.hasSuffix(".cluster_size") || key.hasSuffix(".inited") { continue }
            guard key.hasPrefix("acoustic_encoder.")
                || key.hasPrefix("acoustic_decoder.")
                || key.hasPrefix("quantizer.")
                || key.hasPrefix("fc2.")
            else {
                continue
            }

            var value = rawValue
            if key.hasSuffix(".codebook.embed") {
                key = String(key.dropLast("embed".count)) + "weight"
            }

            if key.hasSuffix(".alpha"), value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            } else if key.contains(".conv_t"), key.hasSuffix(".weight"), value.ndim == 3 {
                value = value.transposed(1, 2, 0)
            } else if key.hasSuffix(".weight"), value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }

            result[key] = value
        }
        return result
    }
}
