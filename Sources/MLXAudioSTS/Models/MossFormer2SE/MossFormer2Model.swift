import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

final class MossFormerM: Module {
    @ModuleInfo(key: "mossformerM") var mossformerM: MossFormerBlock_GFSMN
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(
        numBlocks: Int,
        dModel: Int,
        causal: Bool = false,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        attnDropout: Float = 0.1
    ) {
        self._mossformerM.wrappedValue = MossFormerBlock_GFSMN(
            dim: dModel,
            depth: numBlocks,
            groupSize: groupSize,
            queryKeyDim: queryKeyDim,
            expansionFactor: expansionFactor,
            causal: causal,
            attnDropout: attnDropout
        )
        self._norm.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-8)
    }

    func callAsFunction(_ src: MLXArray) -> MLXArray {
        norm(mossformerM(src))
    }
}

final class ComputationBlock: Module {
    let skipAroundIntra: Bool

    @ModuleInfo(key: "intra_mdl") var intraMdl: MossFormerM
    @ModuleInfo(key: "intra_norm") var intraNorm: LayerNorm?

    init(
        numBlocks: Int,
        outChannels: Int,
        norm: String = "ln",
        skipAroundIntra: Bool = true,
        useMossformer2 _: Bool = false
    ) {
        self.skipAroundIntra = skipAroundIntra
        self._intraMdl.wrappedValue = MossFormerM(numBlocks: numBlocks, dModel: outChannels)

        if norm == "ln" {
            self._intraNorm.wrappedValue = LayerNorm(dimensions: outChannels, eps: 1e-8)
        } else {
            self._intraNorm.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var intra = x.transposed(0, 2, 1)
        intra = intraMdl(intra)
        if let intraNorm {
            intra = intraNorm(intra)
        }
        intra = intra.transposed(0, 2, 1)

        if skipAroundIntra {
            intra = intra + x
        }
        return intra
    }
}

final class MossFormerMaskNet: Module {
    let useGlobalPosEnc: Bool
    let numSpks: Int

    @ModuleInfo(key: "norm") var norm: GlobalLayerNorm
    @ModuleInfo(key: "conv1d_encoder") var conv1dEncoder: Conv1d
    @ModuleInfo(key: "pos_enc") var posEnc: ScaledSinuEmbedding?
    @ModuleInfo(key: "mdl") var mdl: ComputationBlock
    @ModuleInfo(key: "conv1d_out") var conv1dOut: Conv1d
    @ModuleInfo(key: "conv1_decoder") var conv1Decoder: Conv1d
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "output") var output: Conv1d
    @ModuleInfo(key: "output_gate") var outputGate: Conv1d

    init(
        inChannels: Int = 180,
        outChannels: Int = 512,
        outChannelsFinal: Int = 961,
        numBlocks: Int = 24,
        norm _: String = "gln",
        numSpks: Int = 2,
        skipAroundIntra: Bool = true,
        useGlobalPosEnc: Bool = true,
        maxLength _: Int = 20_000
    ) {
        self.useGlobalPosEnc = useGlobalPosEnc
        self.numSpks = numSpks

        self._norm.wrappedValue = GlobalLayerNorm(dim: inChannels, shape: 3)
        self._conv1dEncoder.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: false
        )

        if useGlobalPosEnc {
            self._posEnc.wrappedValue = ScaledSinuEmbedding(dim: outChannels)
        } else {
            self._posEnc.wrappedValue = nil
        }

        self._mdl.wrappedValue = ComputationBlock(
            numBlocks: numBlocks,
            outChannels: outChannels,
            norm: "ln",
            skipAroundIntra: skipAroundIntra,
            useMossformer2: false
        )
        self._conv1dOut.wrappedValue = Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannels * numSpks,
            kernelSize: 1,
            bias: true
        )
        self._conv1Decoder.wrappedValue = Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannelsFinal,
            kernelSize: 1,
            bias: false
        )
        self._prelu.wrappedValue = PReLU()
        self._output.wrappedValue = Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: true
        )
        self._outputGate.wrappedValue = Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: true
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = norm(input)

        x = x.transposed(0, 2, 1)
        x = conv1dEncoder(x)
        x = x.transposed(0, 2, 1)

        if useGlobalPosEnc, let posEnc {
            let base = x
            let xNlc = x.transposed(0, 2, 1)
            var emb = posEnc(xNlc)

            if emb.ndim == 2 {
                emb = MLX.broadcast(emb.expandedDimensions(axis: 0), to: [xNlc.shape[0], emb.shape[0], emb.shape[1]])
            }

            emb = emb.transposed(0, 2, 1)
            x = base + emb
        }

        x = mdl(x)
        x = prelu(x)

        x = x.transposed(0, 2, 1)
        x = conv1dOut(x)
        x = x.transposed(0, 2, 1)

        let batch = x.shape[0]
        let seqLen = x.shape[2]
        x = x.reshaped(batch * numSpks, -1, seqLen)

        x = x.transposed(0, 2, 1)
        let outputVal = tanh(output(x))
        let gateVal = sigmoid(outputGate(x))
        x = outputVal * gateVal

        x = conv1Decoder(x)
        x = x.transposed(0, 2, 1)

        let nBins = x.shape[1]
        let nFrames = x.shape[2]
        x = x.reshaped(batch, numSpks, nBins, nFrames)
        x = relu(x)
        x = x.transposed(1, 0, 2, 3)

        return x[0].transposed(0, 2, 1)
    }
}

final class TestNet: Module {
    @ModuleInfo(key: "mossformer") var mossformer: MossFormerMaskNet

    init(config: MossFormer2SEConfig) {
        self._mossformer.wrappedValue = MossFormerMaskNet(
            inChannels: config.inChannels,
            outChannels: config.outChannels,
            outChannelsFinal: config.outChannelsFinal,
            numBlocks: config.numBlocks
        )
    }

    func callAsFunction(_ input: MLXArray) -> [MLXArray] {
        let x = input.transposed(0, 2, 1)
        let mask = stopGradient(mossformer(x))
        return [mask]
    }
}

public final class MossFormer2SE: Module {
    @ModuleInfo(key: "model") var model: TestNet

    public init(config: MossFormer2SEConfig) {
        self._model.wrappedValue = TestNet(config: config)
    }

    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        model(x)
    }
}

public enum MossFormer2SEError: Error, LocalizedError {
    case invalidRepoID(String)
    case invalidAudioShape([Int])
    case missingMask
    case missingSafetensors(URL)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepoID(let repoID):
            return "Invalid repository ID: \(repoID)"
        case .invalidAudioShape(let shape):
            return "Expected a 1D waveform, got shape \(shape)"
        case .missingMask:
            return "Model did not return a mask"
        case .missingSafetensors(let directory):
            return "No .safetensors files found in \(directory.path)"
        case .duplicateWeightKey(let key):
            return "Duplicate normalized weight key detected while loading local safetensors: \(key)"
        }
    }
}

public final class MossFormer2SEModel {
    public static let defaultRepo = "starkdmi/MossFormer2-SE-fp16"

    public let model: MossFormer2SE
    public let config: MossFormer2SEConfig

    public init(model: MossFormer2SE, config: MossFormer2SEConfig) {
        self.model = model
        self.config = config
    }

    private static func normalizedWeightKey(_ rawKey: String) -> String {
        var key = rawKey

        if key.hasPrefix("module.") {
            key = String(key.dropFirst("module.".count))
        }

        if key.hasPrefix("mossformer.") {
            key = "model." + key
        }

        return key
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (rawKey, value) in weights {
            let key = normalizedWeightKey(rawKey)
            sanitized[key] = value
        }

        return sanitized
    }

    private enum ConfigFallbackPolicy {
        case fallbackOnReadError
        case fallbackOnlyWhenMissing
    }

    private enum DuplicateKeyPolicy {
        case overwrite
        case fail
    }

    private static func loadConfig(from directory: URL, fallbackPolicy: ConfigFallbackPolicy) throws -> MossFormer2SEConfig {
        let configURL = directory.appendingPathComponent("config.json")

        switch fallbackPolicy {
        case .fallbackOnReadError:
            guard let configData = try? Data(contentsOf: configURL) else {
                return MossFormer2SEConfig()
            }
            return try JSONDecoder().decode(MossFormer2SEConfig.self, from: configData)
        case .fallbackOnlyWhenMissing:
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                return MossFormer2SEConfig()
            }
            let configData = try Data(contentsOf: configURL)
            return try JSONDecoder().decode(MossFormer2SEConfig.self, from: configData)
        }
    }

    private static func loadWeights(from directory: URL, duplicateKeyPolicy: DuplicateKeyPolicy) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        var normalizedKeys = Set<String>()
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !files.isEmpty else {
            throw MossFormer2SEError.missingSafetensors(directory)
        }

        for file in files {
            let fileWeights = try MLX.loadArrays(url: file)
            switch duplicateKeyPolicy {
            case .overwrite:
                weights.merge(fileWeights) { _, new in new }
            case .fail:
                for (key, value) in fileWeights {
                    let normalized = normalizedWeightKey(key)
                    if normalizedKeys.contains(normalized) {
                        throw MossFormer2SEError.duplicateWeightKey(normalized)
                    }
                    normalizedKeys.insert(normalized)
                    weights[key] = value
                }
            }
        }

        return weights
    }

    private static func buildModel(config: MossFormer2SEConfig, weights: [String: MLXArray]) throws -> MossFormer2SEModel {
        let model = MossFormer2SE(config: config)
        let sanitizedWeights = sanitize(weights: weights)

        if let quantization = config.quantizationConfig {
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) { path, _ in
                sanitizedWeights["\(path).scales"] != nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])
        eval(model)
        return MossFormer2SEModel(model: model, config: config)
    }

    public static func fromPretrained(_ modelPath: String = defaultRepo) async throws -> MossFormer2SEModel {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw MossFormer2SEError.invalidRepoID(modelPath)
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(repoID: repoID, requiredExtension: "safetensors")
        let config = try loadConfig(from: modelDir, fallbackPolicy: .fallbackOnReadError)
        let weights = try loadWeights(from: modelDir, duplicateKeyPolicy: .overwrite)
        return try buildModel(config: config, weights: weights)
    }

    public static func fromLocal(_ directory: URL) throws -> MossFormer2SEModel {
        let config = try loadConfig(from: directory, fallbackPolicy: .fallbackOnlyWhenMissing)
        let weights = try loadWeights(from: directory, duplicateKeyPolicy: .fail)
        return try buildModel(config: config, weights: weights)
    }

    public func enhance(_ audioInput: MLXArray, dither: Float = 0.0) throws -> MLXArray {
        guard audioInput.ndim == 1 else {
            throw MossFormer2SEError.invalidAudioShape(audioInput.shape)
        }

        let audio = audioInput.asType(.float32)
        let kaldiInt16Scale = MLXArray(Float(32768.0))
        let kaldiAudio = audio * kaldiInt16Scale
        let lowerType = config.winType.lowercased()
        let window: MLXArray
        if lowerType.contains("hann") {
            window = MossFormer2DSP.hannWindow(size: config.winLen, periodic: false)
        } else {
            window = MossFormer2DSP.hammingWindow(size: config.winLen, periodic: false)
        }

        let fbank = MossFormer2DSP.computeFbankKaldi(
            audio: kaldiAudio,
            sampleRate: config.sampleRate,
            winLen: config.winLen,
            winInc: config.winInc,
            numMels: config.numMels,
            winType: config.winType,
            preemphasis: config.preemphasis,
            dither: dither,
            removeDCOffset: true,
            roundToPowerOfTwo: true,
            lowFreq: 20.0
        )
        let fbankT = fbank.transposed(1, 0)
        let deltaT = MossFormer2DSP.computeDeltasKaldi(fbankT, winLength: 5)
        let deltaDeltaT = MossFormer2DSP.computeDeltasKaldi(deltaT, winLength: 5)
        let delta = deltaT.transposed(1, 0)
        let deltaDelta = deltaDeltaT.transposed(1, 0)
        let features = MLX.concatenated([fbank, delta, deltaDelta], axis: -1)

        let batchedFeatures = features.expandedDimensions(axis: 0)
        guard let mask = model(batchedFeatures).first else {
            throw MossFormer2SEError.missingMask
        }

        var stftComplex = MossFormer2DSP.stft(
            audio: kaldiAudio,
            fftLen: config.fftLen,
            hopLength: config.winInc,
            winLen: config.winLen,
            window: window,
            center: false
        )

        var mask2d = mask[0]
        if stftComplex.ndim == 3 {
            stftComplex = stftComplex[0]
        }

        let frames = min(stftComplex.shape[0], mask2d.shape[0])
        let bins = min(stftComplex.shape[1], mask2d.shape[1])
        stftComplex = stftComplex[0..<frames, 0..<bins]
        mask2d = mask2d[0..<frames, 0..<bins]

        let enhancedComplex = stftComplex * mask2d
        let enhancedReal = enhancedComplex.realPart().transposed(1, 0).expandedDimensions(axis: 0)
        let enhancedImag = enhancedComplex.imaginaryPart().transposed(1, 0).expandedDimensions(axis: 0)

        let enhanced = MossFormer2DSP.istft(
            real: enhancedReal,
            imag: enhancedImag,
            fftLen: config.fftLen,
            hopLength: config.winInc,
            winLen: config.winLen,
            window: window,
            center: false,
            audioLength: kaldiAudio.shape[0]
        )
        return enhanced / kaldiInt16Scale
    }
}
