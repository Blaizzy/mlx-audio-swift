import Accelerate
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

public enum DeepFilterNetError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepoID(String)
    case modelPathNotFound(String)
    case missingConfig(URL)
    case missingWeights(URL)
    case missingWeightKey(String)
    case invalidAudioShape([Int])
    case streamingNotSupportedForModelVersion(String)

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalidRepoID(let value):
            return "Invalid Hugging Face model repo ID: \(value)"
        case .modelPathNotFound(let path):
            return "Model path not found: \(path)"
        case .missingConfig(let directory):
            return "Missing config.json in model directory: \(directory.path)"
        case .missingWeights(let directory):
            return "Missing .safetensors weights in model directory: \(directory.path)"
        case .missingWeightKey(let key):
            return "Missing DeepFilterNet weight key: \(key)"
        case .invalidAudioShape(let shape):
            return "Expected mono 1D audio array, got shape: \(shape)"
        case .streamingNotSupportedForModelVersion(let version):
            return "Streaming is not supported for model version \(version). Use offline enhancement instead."
        }
    }
}

public struct DeepFilterNetStreamingConfig: Sendable {
    public var padEndFrames: Int
    public var compensateDelay: Bool
    public var enableStageSkipping: Bool
    public var minDbThresh: Float
    public var maxDbErbThresh: Float
    public var maxDbDfThresh: Float
    public var enableProfiling: Bool
    public var profilingForceEvalPerStage: Bool
    public var materializeEveryHops: Int

    public init(
        padEndFrames: Int = 3,
        compensateDelay: Bool = true,
        enableStageSkipping: Bool = false,
        minDbThresh: Float = -10.0,
        maxDbErbThresh: Float = 30.0,
        maxDbDfThresh: Float = 20.0,
        enableProfiling: Bool = false,
        profilingForceEvalPerStage: Bool = false,
        materializeEveryHops: Int = 512
    ) {
        self.padEndFrames = padEndFrames
        self.compensateDelay = compensateDelay
        self.enableStageSkipping = enableStageSkipping
        self.minDbThresh = minDbThresh
        self.maxDbErbThresh = maxDbErbThresh
        self.maxDbDfThresh = maxDbDfThresh
        self.enableProfiling = enableProfiling
        self.profilingForceEvalPerStage = profilingForceEvalPerStage
        self.materializeEveryHops = materializeEveryHops
    }
}

public struct DeepFilterNetStreamingChunk: @unchecked Sendable {
    public let audio: MLXArray
    public let chunkIndex: Int
    public let isLastChunk: Bool
}

// MARK: - Model

public final class DeepFilterNetModel: STSModel {
    public static let defaultRepo = "iky1e/DeepFilterNet3-MLX"

    public let config: DeepFilterNetConfig
    public let modelDirectory: URL
    public let modelVersion: String
    public var isV1: Bool { modelVersion.lowercased() == "deepfilternet" }
    public var supportsStreaming: Bool { !isV1 }
    public var sampleRate: Int { config.sampleRate }

    // Internal visibility for cross-file access by extensions and DeepFilterNetStreamer.
    let weights: [String: MLXArray]
    let erbFB: MLXArray
    let erbInvFB: MLXArray
    let erbBandWidths: [Int]
    let vorbisWindowArray: MLXArray
    let wnorm: Float
    let normAlphaValue: Float
    let inferenceDType: DType
    let bnScale: [String: MLXArray]
    let bnBias: [String: MLXArray]
    let conv2dWeightsOHWI: [String: MLXArray]
    let convTransposeDenseWeights: [String: MLXArray]
    let convTransposeGroupWeights: [String: [MLXArray]]
    let gruTransposedWeights: [String: MLXArray]
    let j: MLXArray = MLXArray(real: Float(0.0), imaginary: Float(1.0))

    struct V1GroupedLinearPack {
        let weightGIO: MLXArray  // [G, I, O]
        let biasGO: MLXArray  // [G, O]
        let groups: Int
        let inputPerGroup: Int
        let outputPerGroup: Int
    }

    struct V1GroupedGRULayerPack {
        let weightIHGI3H: MLXArray  // [G, I, 3H]
        let weightHHGH3H: MLXArray  // [G, H, 3H]
        let biasIHG3H: MLXArray  // [G, 3H]
        let biasHHG3H: MLXArray  // [G, 3H]
        let inputPerGroup: Int
        let hiddenPerGroup: Int
    }

    struct V1GroupedGRUPack {
        let groups: Int
        let layers: [V1GroupedGRULayerPack]
    }

    let v1GroupedLinearPacks: [String: V1GroupedLinearPack]
    let v1GroupedGRUPacks: [String: V1GroupedGRUPack]

    private init(
        config: DeepFilterNetConfig,
        modelDirectory: URL,
        weights: [String: MLXArray]
    ) throws {
        self.config = config
        self.modelDirectory = modelDirectory
        self.modelVersion = config.modelVersion
        self.weights = weights

        guard let erbInvFB = weights["mask.erb_inv_fb"] else {
            throw DeepFilterNetError.missingWeightKey("mask.erb_inv_fb")
        }
        self.erbFB = weights["erb_fb"] ?? MLXArray.zeros([1, 1], type: Float.self)
        self.erbInvFB = erbInvFB
        let widthsFromConfig = config.erbWidths
        if let widthsFromConfig, widthsFromConfig.reduce(0, +) == config.freqBins {
            self.erbBandWidths = widthsFromConfig
        } else {
            self.erbBandWidths = Self.libdfErbBandWidths(
                sampleRate: config.sampleRate,
                fftSize: config.fftSize,
                nbBands: config.nbErb,
                minNbFreqs: max(1, config.minNbErbFreqs)
            )
        }
        self.vorbisWindowArray = Self.vorbisWindow(size: config.fftSize)
        self.wnorm = 1.0 / Float(config.fftSize * config.fftSize) * Float(2 * config.hopSize)
        self.normAlphaValue = Self.computeNormAlpha(hopSize: config.hopSize, sampleRate: config.sampleRate)
        self.inferenceDType =
            weights["enc.erb_conv0.1.weight"]?.dtype
            ?? weights["enc.erb_conv0.sconv.weight"]?.dtype
            ?? weights.values.first?.dtype
            ?? .float32
        let (bnScale, bnBias) = Self.buildBatchNormAffine(weights: weights)
        self.bnScale = bnScale
        self.bnBias = bnBias
        self.conv2dWeightsOHWI = Self.buildConv2dWeightCache(weights: weights)
        self.convTransposeDenseWeights = Self.buildDenseTransposeWeights(
            weights: weights,
            groups: max(1, config.convCh)
        )
        self.convTransposeGroupWeights = Self.buildGroupedTransposeWeights(
            weights: weights,
            groups: max(1, config.convCh)
        )
        self.gruTransposedWeights = Self.buildGRUTransposedWeightCache(weights: weights)
        self.v1GroupedLinearPacks = Self.buildV1GroupedLinearPacks(
            weights: weights,
            groups: max(1, config.linearGroups)
        )
        self.v1GroupedGRUPacks = Self.buildV1GroupedGRUPacks(
            weights: weights,
            groups: max(1, config.gruGroups),
            prefixes: [
                "enc.emb_gru.grus",
                "clc_dec.clc_gru.grus",
            ]
        )
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelPathOrRepo: String = defaultRepo,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> DeepFilterNetModel {
        let local = URL(fileURLWithPath: modelPathOrRepo).standardizedFileURL
        if FileManager.default.fileExists(atPath: local.path) {
            if local.hasDirectoryPath {
                return try fromLocal(local)
            }
            return try fromLocal(local.deletingLastPathComponent())
        }

        guard let repoID = Repo.ID(rawValue: modelPathOrRepo) else {
            throw DeepFilterNetError.invalidRepoID(modelPathOrRepo)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try fromLocal(modelDir)
    }

    public static func fromLocal(_ directory: URL) throws -> DeepFilterNetModel {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DeepFilterNetError.missingConfig(directory)
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let configData = try Data(contentsOf: configURL)
        var config = try decoder.decode(DeepFilterNetConfig.self, from: configData)
        if config.modelVersion.isEmpty {
            config.modelVersion = "DeepFilterNet3"
        }

        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard let weightsURL = files.first(where: { $0.lastPathComponent == "model.safetensors" }) ?? files.first else {
            throw DeepFilterNetError.missingWeights(directory)
        }

        let weights = try MLX.loadArrays(url: weightsURL)
        return try DeepFilterNetModel(config: config, modelDirectory: directory, weights: weights)
    }

    // MARK: - Public API

    public func enhance(_ audioInput: MLXArray) throws -> MLXArray {
        guard audioInput.ndim == 1 else {
            throw DeepFilterNetError.invalidAudioShape(audioInput.shape)
        }

        let x = audioInput.asType(.float32)
        let origLen = x.shape[0]
        let padded = MLX.concatenated([
            MLXArray.zeros([config.hopSize], type: Float.self),
            x,
            MLXArray.zeros([config.fftSize], type: Float.self),
        ], axis: 0)

        let specComplex = MossFormer2DSP.stft(
            audio: padded,
            fftLen: config.fftSize,
            hopLength: config.hopSize,
            winLen: config.fftSize,
            window: vorbisWindowArray,
            center: false
        )
        let spec = specComplex * MLXArray(wnorm)
        let specRe = spec.realPart()
        let specIm = spec.imaginaryPart()

        let specMagSq = specRe.square() + specIm.square()
        let erb = erbEnergies(specMagSq)
        let erbDB = MLXArray(Float(10.0)) * (erb + MLXArray(Float(1e-10))).log10()
        let featErb2D = isV1 ? bandMeanNormExact(erbDB) : bandMeanNorm(erbDB)

        let dfRe = specRe[0..., 0..<config.nbDf]
        let dfIm = specIm[0..., 0..<config.nbDf]
        let (dfFeatRe, dfFeatIm) = isV1
            ? bandUnitNormExact(real: dfRe, imag: dfIm)
            : bandUnitNorm(real: dfRe, imag: dfIm)

        let featErb = featErb2D.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let featDf = MLX.stacked([dfFeatRe, dfFeatIm], axis: -1)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let specIn = MLX.stacked([specRe, specIm], axis: -1)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)

        let forwardOut: (MLXArray, MLXArray, MLXArray, MLXArray)
        if isV1 {
            forwardOut = try forwardV1(
                spec: specIn.asType(inferenceDType),
                featErb: featErb.asType(inferenceDType),
                featSpec5D: featDf.asType(inferenceDType)
            )
        } else {
            forwardOut = try forward(
                spec: specIn.asType(inferenceDType),
                featErb: featErb.asType(inferenceDType),
                featSpec5D: featDf.asType(inferenceDType)
            )
        }
        let specEnhanced = forwardOut.0

        var enhTF2 = specEnhanced
            .squeezed(axis: 0)
            .squeezed(axis: 0)
        if enhTF2.ndim == 4, enhTF2.shape[0] == 1 {
            enhTF2 = enhTF2.squeezed(axis: 0)
        }
        var enh = enhTF2[0..., 0..., 0] + j * enhTF2[0..., 0..., 1]
        enh = enh / MLXArray(wnorm)

        var enhReal2D = enh.realPart().squeezed()
        var enhImag2D = enh.imaginaryPart().squeezed()
        if enhReal2D.ndim != 2 || enhImag2D.ndim != 2 {
            let t = spec.shape[2]
            let f = config.freqBins
            enhReal2D = enhReal2D.reshaped([t, f])
            enhImag2D = enhImag2D.reshaped([t, f])
        }
        let enhReal = enhReal2D.transposed(1, 0).expandedDimensions(axis: 0)
        let enhImag = enhImag2D.transposed(1, 0).expandedDimensions(axis: 0)

        var audioOut = MossFormer2DSP.istft(
            real: enhReal,
            imag: enhImag,
            fftLen: config.fftSize,
            hopLength: config.hopSize,
            winLen: config.fftSize,
            window: vorbisWindowArray,
            center: false,
            audioLength: origLen + config.hopSize + config.fftSize
        )

        let delay = config.fftSize - config.hopSize
        let end = min(delay + origLen, audioOut.shape[0])
        audioOut = audioOut[delay..<end]
        return MLX.clip(audioOut, min: -1.0, max: 1.0)
    }

    public func createStreamer(
        config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()
    ) -> DeepFilterNetStreamer {
        precondition(
            supportsStreaming,
            "DeepFilterNet v1 streaming is not supported in Swift yet. Use enhance(_:) for offline."
        )
        return DeepFilterNetStreamer(model: self, config: config)
    }

    public func enhanceStreaming(
        _ audioInput: MLXArray,
        chunkSamples: Int? = nil,
        config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()
    ) throws -> MLXArray {
        guard supportsStreaming else {
            throw DeepFilterNetError.streamingNotSupportedForModelVersion(modelVersion)
        }
        guard audioInput.ndim == 1 else {
            throw DeepFilterNetError.invalidAudioShape(audioInput.shape)
        }
        let samples = audioInput.asType(.float32)
        if samples.shape[0] == 0 {
            return MLXArray.zeros([0], type: Float.self)
        }

        let streamer = createStreamer(config: config)
        let frameChunk = max(self.config.hopSize, chunkSamples ?? self.config.hopSize)
        var outputChunks = [MLXArray]()
        outputChunks.reserveCapacity(max(1, samples.shape[0] / frameChunk))

        var start = 0
        while start < samples.shape[0] {
            let end = min(start + frameChunk, samples.shape[0])
            let chunk = samples[start..<end]
            let out = try streamer.processChunk(chunk)
            if out.shape[0] > 0 {
                outputChunks.append(out)
            }
            start = end
        }
        let tail = try streamer.flushMLX()
        if tail.shape[0] > 0 {
            outputChunks.append(tail)
        }
        if outputChunks.isEmpty {
            return MLXArray.zeros([0], type: Float.self)
        }
        return MLX.clip(MLX.concatenated(outputChunks, axis: 0), min: -1.0, max: 1.0)
    }

    public func enhanceStreaming(
        _ audioInput: MLXArray,
        chunkSamples: Int? = nil,
        config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()
    ) -> AsyncThrowingStream<DeepFilterNetStreamingChunk, Error> {
        AsyncThrowingStream { continuation in
            do {
                guard supportsStreaming else {
                    throw DeepFilterNetError.streamingNotSupportedForModelVersion(modelVersion)
                }
                guard audioInput.ndim == 1 else {
                    throw DeepFilterNetError.invalidAudioShape(audioInput.shape)
                }
                let samples = audioInput.asType(.float32)
                let streamer = createStreamer(config: config)
                let frameChunk = max(self.config.hopSize, chunkSamples ?? self.config.hopSize)

                var chunkIndex = 0
                var start = 0
                while start < samples.shape[0] {
                    let end = min(start + frameChunk, samples.shape[0])
                    let chunk = samples[start..<end]
                    let out = try streamer.processChunk(chunk)
                    if out.shape[0] > 0 {
                        continuation.yield(
                            DeepFilterNetStreamingChunk(
                                audio: out,
                                chunkIndex: chunkIndex,
                                isLastChunk: false
                            )
                        )
                        chunkIndex += 1
                    }
                    start = end
                }

                let tail = try streamer.flushMLX()
                if tail.shape[0] > 0 {
                    continuation.yield(
                        DeepFilterNetStreamingChunk(
                            audio: tail,
                            chunkIndex: chunkIndex,
                            isLastChunk: true
                        )
                    )
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}
