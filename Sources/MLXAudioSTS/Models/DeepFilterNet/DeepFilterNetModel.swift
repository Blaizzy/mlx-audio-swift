import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

/// Errors thrown by DeepFilterNet model loading and inference.
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

/// Configuration for DeepFilterNet streaming inference.
public struct DeepFilterNetStreamingConfig: Sendable {
    /// Number of zero-padded frames appended when flushing to drain the pipeline.
    public var padEndFrames: Int
    /// Whether to trim the algorithmic delay (fftSize - hopSize samples) from the output.
    public var compensateDelay: Bool
    /// Enable LSNR-based stage skipping to reduce computation on clean frames.
    public var enableStageSkipping: Bool
    /// LSNR threshold (dB) below which input is treated as noise-only.
    public var minDbThresh: Float
    /// LSNR threshold (dB) above which ERB gains are skipped (clean speech).
    public var maxDbErbThresh: Float
    /// LSNR threshold (dB) above which deep filtering is skipped.
    public var maxDbDfThresh: Float
    /// Enable per-stage profiling output via ``DeepFilterNetStreamer/profilingSummary()``.
    public var enableProfiling: Bool
    /// Force `eval()` after each pipeline stage to isolate per-stage GPU time.
    public var profilingForceEvalPerStage: Bool
    /// How many hops between forced `eval()` calls to bound lazy graph growth.
    public var materializeEveryHops: Int

    /// Creates a streaming configuration with the given parameters.
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

/// A chunk of enhanced audio produced by streaming inference.
public struct DeepFilterNetStreamingChunk: @unchecked Sendable {
    /// Enhanced audio samples for this chunk.
    public let audio: MLXArray
    /// Sequential index of this chunk (0-based).
    public let chunkIndex: Int
    /// Whether this is the final chunk in the stream.
    public let isLastChunk: Bool
}

// MARK: - Model

/// DeepFilterNet speech enhancement model.
///
/// Removes background noise from speech audio using a dual-pathway (ERB + deep filtering)
/// encoder-decoder architecture. Supports offline batch enhancement and low-latency
/// hop-by-hop streaming.
///
/// ```swift
/// let model = try await DeepFilterNetModel.fromPretrained()
/// let enhanced = try model.enhance(noisyAudio)
/// ```
public final class DeepFilterNetModel: STSModel, @unchecked Sendable {
    /// Default HuggingFace repo containing DeepFilterNet v1/v2/v3 weights.
    public static let defaultRepo = "mlx-community/DeepFilterNet-mlx"

    /// Default subfolder within the repo (v3 is recommended).
    public static let defaultSubfolder = "v3"

    /// Model configuration decoded from `config.json`.
    public let config: DeepFilterNetConfig
    /// Local directory containing model weights and configuration.
    public let modelDirectory: URL
    /// Model version string (e.g. "DeepFilterNet3").
    public let modelVersion: String
    /// Whether this is a DeepFilterNet V1 model.
    public var isV1: Bool { modelVersion.lowercased() == "deepfilternet" }
    /// Whether this model supports streaming mode. V2/V3 only.
    public var supportsStreaming: Bool { !isV1 }
    /// Audio sample rate expected by this model (typically 48kHz).
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

    /// Loads a DeepFilterNet model from a local path or HuggingFace repo.
    ///
    /// If `modelPathOrRepo` is a local directory, loads directly. Otherwise, downloads
    /// from HuggingFace Hub. The default repo contains v1/v2/v3 subfolders — use
    /// `subfolder` to select the version.
    ///
    /// - Parameters:
    ///   - modelPathOrRepo: Local directory path or HuggingFace repo ID.
    ///   - subfolder: Subfolder within the repo (e.g. `"v1"`, `"v2"`, `"v3"`).
    ///     Ignored for local paths that already contain `config.json`.
    ///   - hfToken: Optional HuggingFace API token for private repos.
    ///   - cache: HuggingFace cache configuration.
    /// - Returns: A loaded model ready for inference.
    public static func fromPretrained(
        _ modelPathOrRepo: String = defaultRepo,
        subfolder: String? = defaultSubfolder,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> DeepFilterNetModel {
        let local = URL(fileURLWithPath: modelPathOrRepo).standardizedFileURL
        if FileManager.default.fileExists(atPath: local.path) {
            if local.hasDirectoryPath {
                return try fromLocal(local, subfolder: subfolder)
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
        return try fromLocal(modelDir, subfolder: subfolder)
    }

    /// Loads a DeepFilterNet model from a local directory.
    ///
    /// The directory must contain `config.json` and at least one `.safetensors` file,
    /// either at the top level or within the specified `subfolder`.
    ///
    /// - Parameters:
    ///   - directory: Path to the model directory.
    ///   - subfolder: Optional subfolder (e.g. `"v3"`). Used when the directory
    ///     contains version subfolders rather than model files directly.
    /// - Returns: A loaded model ready for inference.
    public static func fromLocal(_ directory: URL, subfolder: String? = nil) throws -> DeepFilterNetModel {
        // If the directory itself contains config.json, use it directly.
        // Otherwise, try the subfolder.
        var modelDir = directory
        if !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("config.json").path),
           let subfolder, !subfolder.isEmpty
        {
            modelDir = directory.appendingPathComponent(subfolder)
        }

        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DeepFilterNetError.missingConfig(modelDir)
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let configData = try Data(contentsOf: configURL)
        var config = try decoder.decode(DeepFilterNetConfig.self, from: configData)
        if config.modelVersion.isEmpty {
            config.modelVersion = "DeepFilterNet3"
        }

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard let weightsURL = files.first(where: { $0.lastPathComponent == "model.safetensors" }) ?? files.first else {
            throw DeepFilterNetError.missingWeights(modelDir)
        }

        let weights = try MLX.loadArrays(url: weightsURL)
        return try DeepFilterNetModel(config: config, modelDirectory: modelDir, weights: weights)
    }

    // MARK: - Public API

    /// Enhances speech audio by removing background noise (offline/batch mode).
    ///
    /// Processes the entire audio in one pass. For real-time use, see
    /// ``createStreamer(config:)`` or ``enhanceStreaming(_:chunkSamples:config:)-3u1fv``.
    ///
    /// - Parameter audioInput: Mono audio as a 1D `MLXArray` of float samples in `[-1, 1]`.
    /// - Returns: Enhanced audio with the same length and sample rate as the input.
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

    /// Creates a stateful streamer for hop-by-hop streaming enhancement.
    ///
    /// The streamer processes audio incrementally, maintaining internal state across calls.
    /// Each call to ``DeepFilterNetStreamer/processChunk(_:isLast:)-9gh0v`` accepts any number
    /// of samples and returns enhanced audio as it becomes available.
    ///
    /// - Parameter config: Streaming configuration. Defaults to standard settings.
    /// - Returns: A new streamer instance. Call ``DeepFilterNetStreamer/reset()`` to reuse.
    public func createStreamer(
        config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()
    ) -> DeepFilterNetStreamer {
        precondition(
            supportsStreaming,
            "DeepFilterNet v1 streaming is not supported in Swift yet. Use enhance(_:) for offline."
        )
        return DeepFilterNetStreamer(model: self, config: config)
    }

    /// Enhances speech audio using streaming mode, returning the full result.
    ///
    /// Convenience method that internally creates a streamer, processes all chunks,
    /// and returns the concatenated enhanced audio.
    ///
    /// - Parameters:
    ///   - audioInput: Mono audio as a 1D `MLXArray`.
    ///   - chunkSamples: Chunk size in samples. Defaults to one hop (480 samples = 10ms).
    ///   - config: Streaming configuration.
    /// - Returns: Enhanced audio with the same sample rate as the input.
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

    /// Enhances speech audio using streaming mode, yielding chunks as they are produced.
    ///
    /// - Parameters:
    ///   - audioInput: Mono audio as a 1D `MLXArray`.
    ///   - chunkSamples: Chunk size in samples. Defaults to one hop (480 samples = 10ms).
    ///   - config: Streaming configuration.
    /// - Returns: An async stream of ``DeepFilterNetStreamingChunk`` values.
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
