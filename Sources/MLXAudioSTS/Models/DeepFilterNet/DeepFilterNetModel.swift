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
    public var materializeEveryHops: Int

    public init(
        padEndFrames: Int = 3,
        compensateDelay: Bool = true,
        enableStageSkipping: Bool = false,
        minDbThresh: Float = -10.0,
        maxDbErbThresh: Float = 30.0,
        maxDbDfThresh: Float = 20.0,
        enableProfiling: Bool = false,
        materializeEveryHops: Int = 32
    ) {
        self.padEndFrames = padEndFrames
        self.compensateDelay = compensateDelay
        self.enableStageSkipping = enableStageSkipping
        self.minDbThresh = minDbThresh
        self.maxDbErbThresh = maxDbErbThresh
        self.maxDbDfThresh = maxDbDfThresh
        self.enableProfiling = enableProfiling
        self.materializeEveryHops = materializeEveryHops
    }
}

public struct DeepFilterNetStreamingChunk: @unchecked Sendable {
    public let audio: MLXArray
    public let chunkIndex: Int
    public let isLastChunk: Bool
}

public final class DeepFilterNetModel: STSModel {
    public static let defaultRepo = "kylehowells/DeepFilterNet3-MLX"

    public let config: DeepFilterNetConfig
    public let modelDirectory: URL
    public let modelVersion: String
    public var sampleRate: Int { config.sampleRate }

    private let weights: [String: MLXArray]
    private let erbFB: MLXArray
    private let erbInvFB: MLXArray
    private let erbBandWidths: [Int]
    private let vorbisWindow: MLXArray
    private let wnorm: Float
    private let normAlphaValue: Float
    private let inferenceDType: DType
    private let j: MLXArray = MLXArray(real: Float(0.0), imaginary: Float(1.0))

    private init(
        config: DeepFilterNetConfig,
        modelDirectory: URL,
        weights: [String: MLXArray]
    ) throws {
        self.config = config
        self.modelDirectory = modelDirectory
        self.modelVersion = config.modelVersion
        self.weights = weights

        guard let erbFB = weights["erb_fb"], let erbInvFB = weights["mask.erb_inv_fb"] else {
            throw DeepFilterNetError.missingWeightKey("erb_fb / mask.erb_inv_fb")
        }
        self.erbFB = erbFB
        self.erbInvFB = erbInvFB
        let widthsFromConfig = config.erbWidths
        if let widthsFromConfig, widthsFromConfig.reduce(0, +) == config.freqBins {
            self.erbBandWidths = widthsFromConfig
        } else {
            self.erbBandWidths = Self.libdfErbBandWidths(
                sampleRate: config.sampleRate,
                fftSize: config.fftSize,
                nbBands: config.nbErb,
                minNbFreqs: 1
            )
        }
        self.vorbisWindow = Self.vorbisWindow(size: config.fftSize)
        self.wnorm = 1.0 / Float(config.fftSize * config.fftSize) * Float(2 * config.hopSize)
        self.normAlphaValue = Self.computeNormAlpha(hopSize: config.hopSize, sampleRate: config.sampleRate)
        self.inferenceDType = weights["enc.erb_conv0.1.weight"]?.dtype ?? .float32
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
            window: vorbisWindow,
            center: false
        )
        let spec = specComplex * MLXArray(wnorm)
        let specRe = spec.realPart()
        let specIm = spec.imaginaryPart()

        let specMagSq = specRe.square() + specIm.square()
        let erb = erbEnergies(specMagSq)
        let erbDB = MLXArray(Float(10.0)) * (erb + MLXArray(Float(1e-10))).log10()
        let featErb2D = bandMeanNorm(erbDB)

        let dfRe = specRe[0..., 0..<config.nbDf]
        let dfIm = specIm[0..., 0..<config.nbDf]
        let (dfFeatRe, dfFeatIm) = bandUnitNorm(real: dfRe, imag: dfIm)

        let featErb = featErb2D.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let featDf = MLX.stacked([dfFeatRe, dfFeatIm], axis: -1)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let specIn = MLX.stacked([specRe, specIm], axis: -1)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)

        let (specEnhanced, _, _, _) = try forward(
            spec: specIn.asType(inferenceDType),
            featErb: featErb.asType(inferenceDType),
            featSpec5D: featDf.asType(inferenceDType)
        )
        var enh = specEnhanced[0, 0, 0..., 0..., 0] + j * specEnhanced[0, 0, 0..., 0..., 1]
        enh = enh / MLXArray(wnorm)

        let enhReal = enh.realPart().transposed(1, 0).expandedDimensions(axis: 0)
        let enhImag = enh.imaginaryPart().transposed(1, 0).expandedDimensions(axis: 0)

        var audioOut = MossFormer2DSP.istft(
            real: enhReal,
            imag: enhImag,
            fftLen: config.fftSize,
            hopLength: config.hopSize,
            winLen: config.fftSize,
            window: vorbisWindow,
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
        DeepFilterNetStreamer(model: self, config: config)
    }

    public func enhanceStreaming(
        _ audioInput: MLXArray,
        chunkSamples: Int? = nil,
        config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()
    ) throws -> MLXArray {
        guard audioInput.ndim == 1 else {
            throw DeepFilterNetError.invalidAudioShape(audioInput.shape)
        }
        let samples = audioInput.asType(.float32)
        if samples.shape[0] == 0 {
            return MLXArray.zeros([0], type: Float.self)
        }

        let streamer = createStreamer(config: config)
        // Default to true low-latency chunking: one hop (10ms at 48kHz).
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
                guard audioInput.ndim == 1 else {
                    throw DeepFilterNetError.invalidAudioShape(audioInput.shape)
                }
                let samples = audioInput.asType(.float32)
                let streamer = createStreamer(config: config)
                // Default to true low-latency chunking: one hop (10ms at 48kHz).
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

    // MARK: - Streaming

    public final class DeepFilterNetStreamer {
        private let model: DeepFilterNetModel
        public let config: DeepFilterNetStreamingConfig

        private let fftSize: Int
        private let hopSize: Int
        private let freqBins: Int
        private let nbDf: Int
        private let nbErb: Int
        private let dfOrder: Int
        private let dfLookahead: Int
        private let convLookahead: Int

        private let alphaArray: MLXArray
        private let oneMinusAlphaArray: MLXArray
        private let fftScaleArray: MLXArray
        private let vorbisWindow: MLXArray
        private let wnormArray: MLXArray
        private let inferenceDType: DType
        private let epsEnergy = MLXArray(Float(1e-10))
        private let epsNorm = MLXArray(Float(1e-12))
        private let tenArray = MLXArray(Float(10.0))
        private let fortyArray = MLXArray(Float(40.0))
        private let zeroSpecFrame: MLXArray
        private let zeroMaskFrame: MLXArray
        private let analysisMemCount: Int
        private let synthMemCount: Int
        private let erbFBFrame: MLXArray?
        private let lsnrWeight: MLXArray
        private let lsnrBias: MLXArray
        private let lsnrScale: MLXArray
        private let lsnrOffset: MLXArray

        private var pendingSamples = MLXArray.zeros([0], type: Float.self)
        private var analysisMem: MLXArray
        private var synthMem: MLXArray
        private var erbState: MLXArray
        private var dfState: MLXArray

        private var specQueue: [MLXArray] = []
        private var specPast: [MLXArray] = []
        private var frameCount = 0

        private var encErb0In: [MLXArray] = []
        private var encDf0In: [MLXArray] = []
        private var dfConvpIn: [MLXArray] = []

        private var encEmbState: [MLXArray]?
        private var erbDecState: [MLXArray]?
        private var dfDecState: [MLXArray]?

        private var delayDropped = 0
        private var hopsSinceMaterialize = 0
        private let enableProfiling: Bool
        private var profHopCount = 0
        private var profAnalysisSeconds = 0.0
        private var profFeaturesSeconds = 0.0
        private var profInferSeconds = 0.0
        private var profSynthesisSeconds = 0.0
        private var profMaterializeSeconds = 0.0

        public init(model: DeepFilterNetModel, config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()) {
            self.model = model
            self.config = config
            self.enableProfiling = config.enableProfiling

            self.fftSize = model.config.fftSize
            self.hopSize = model.config.hopSize
            self.freqBins = model.config.freqBins
            self.nbDf = model.config.nbDf
            self.nbErb = model.config.nbErb
            self.dfOrder = model.config.dfOrder
            self.dfLookahead = model.config.dfLookahead
            self.convLookahead = model.config.convLookahead

            let alpha = model.normAlpha()
            self.alphaArray = MLXArray(alpha)
            self.oneMinusAlphaArray = MLXArray(Float(1.0) - alpha)
            self.fftScaleArray = MLXArray(Float(model.config.fftSize))
            self.vorbisWindow = model.vorbisWindow.asType(.float32)
            self.wnormArray = MLXArray(model.wnorm)
            self.inferenceDType = model.inferenceDType
            self.analysisMemCount = max(0, model.config.fftSize - model.config.hopSize)
            self.synthMemCount = max(0, model.config.fftSize - model.config.hopSize)
            self.zeroSpecFrame = MLXArray.zeros([model.config.freqBins, 2], type: Float.self)
            self.zeroMaskFrame = MLXArray.zeros([1, 1, 1, model.config.nbErb], type: Float.self)
            if model.erbFB.shape.count == 2,
               model.erbFB.shape[0] == model.config.freqBins,
               model.erbFB.shape[1] == model.config.nbErb
            {
                self.erbFBFrame = model.erbFB.asType(.float32)
            } else {
                self.erbFBFrame = nil
            }
            self.lsnrWeight = (try? model.w("enc.lsnr_fc.0.weight")) ?? MLXArray.zeros([1, model.config.embHiddenDim], type: Float.self)
            self.lsnrBias = (try? model.w("enc.lsnr_fc.0.bias")) ?? MLXArray.zeros([1], type: Float.self)
            self.lsnrScale = MLXArray(Float(model.config.lsnrMax - model.config.lsnrMin))
            self.lsnrOffset = MLXArray(Float(model.config.lsnrMin))

            self.analysisMem = MLXArray.zeros([analysisMemCount], type: Float.self)
            self.synthMem = MLXArray.zeros([synthMemCount], type: Float.self)
            self.erbState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: model.config.nbErb))
            self.dfState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: model.config.nbDf))
        }

        public func reset() {
            pendingSamples = MLXArray.zeros([0], type: Float.self)
            analysisMem = MLXArray.zeros([analysisMemCount], type: Float.self)
            synthMem = MLXArray.zeros([synthMemCount], type: Float.self)
            erbState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: nbErb))
            dfState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: nbDf))
            specQueue.removeAll(keepingCapacity: true)
            specPast.removeAll(keepingCapacity: true)
            frameCount = 0
            encErb0In.removeAll(keepingCapacity: true)
            encDf0In.removeAll(keepingCapacity: true)
            dfConvpIn.removeAll(keepingCapacity: true)
            encEmbState = nil
            erbDecState = nil
            dfDecState = nil
            delayDropped = 0
            hopsSinceMaterialize = 0
            profHopCount = 0
            profAnalysisSeconds = 0.0
            profFeaturesSeconds = 0.0
            profInferSeconds = 0.0
            profSynthesisSeconds = 0.0
            profMaterializeSeconds = 0.0
        }

        public func processChunk(_ chunk: MLXArray, isLast: Bool = false) throws -> MLXArray {
            guard chunk.ndim == 1 else {
                throw DeepFilterNetError.invalidAudioShape(chunk.shape)
            }
            let chunkF32 = chunk.asType(.float32)
            if chunkF32.shape[0] > 0 {
                if pendingSamples.shape[0] == 0 {
                    pendingSamples = chunkF32
                } else {
                    pendingSamples = MLX.concatenated([pendingSamples, chunkF32], axis: 0)
                }
            }

            var outs = [MLXArray]()
            while pendingSamples.shape[0] >= hopSize {
                let hop = pendingSamples[0..<hopSize]
                pendingSamples = pendingSamples[hopSize..<pendingSamples.shape[0]]
                if let out = try processHop(hop) {
                    outs.append(out)
                }
            }

            if isLast {
                if config.padEndFrames > 0 {
                    let pad = MLXArray.zeros([config.padEndFrames * hopSize], type: Float.self)
                    if pendingSamples.shape[0] == 0 {
                        pendingSamples = pad
                    } else {
                        pendingSamples = MLX.concatenated([pendingSamples, pad], axis: 0)
                    }
                }
                while pendingSamples.shape[0] >= hopSize {
                    let hop = pendingSamples[0..<hopSize]
                    pendingSamples = pendingSamples[hopSize..<pendingSamples.shape[0]]
                    if let out = try processHop(hop) {
                        outs.append(out)
                    }
                }
            }

            var y: MLXArray
            if outs.isEmpty {
                y = MLXArray.zeros([0], type: Float.self)
            } else if outs.count == 1, let first = outs.first {
                y = first
            } else {
                y = MLX.concatenated(outs, axis: 0)
            }

            if config.compensateDelay {
                let totalDelay = fftSize - hopSize
                if delayDropped < totalDelay {
                    let toDrop = min(totalDelay - delayDropped, y.shape[0])
                    if toDrop > 0 {
                        y = y[toDrop..<y.shape[0]]
                        delayDropped += toDrop
                    }
                }
            }

            return y
        }

        public func processChunk(_ chunk: [Float], isLast: Bool = false) throws -> [Float] {
            guard !chunk.isEmpty || isLast else { return [] }
            let y = try processChunk(MLXArray(chunk), isLast: isLast)
            if y.shape[0] == 0 {
                return []
            }
            return y.asArray(Float.self)
        }

        public func flush() throws -> [Float] {
            try processChunk([], isLast: true)
        }

        public func flushMLX() throws -> MLXArray {
            try processChunk(MLXArray.zeros([0], type: Float.self), isLast: true)
        }

        public func profilingSummary() -> String? {
            guard enableProfiling else { return nil }
            let hops = max(profHopCount, 1)
            let total = profAnalysisSeconds + profFeaturesSeconds + profInferSeconds + profSynthesisSeconds + profMaterializeSeconds
            let perHopMs = (total / Double(hops)) * 1000.0
            func pct(_ v: Double) -> Double {
                guard total > 0 else { return 0.0 }
                return (v / total) * 100.0
            }
            return String(
                format:
                    """
                    Stream profile: hops=%d total=%.3fs perHop=%.3fms
                      analysis:    %.3fs (%.1f%%)
                      features:    %.3fs (%.1f%%)
                      infer:       %.3fs (%.1f%%)
                      synthesis:   %.3fs (%.1f%%)
                      materialize: %.3fs (%.1f%%)
                    """,
                profHopCount,
                total,
                perHopMs,
                profAnalysisSeconds, pct(profAnalysisSeconds),
                profFeaturesSeconds, pct(profFeaturesSeconds),
                profInferSeconds, pct(profInferSeconds),
                profSynthesisSeconds, pct(profSynthesisSeconds),
                profMaterializeSeconds, pct(profMaterializeSeconds)
            )
        }

        private func processHop(_ hopTD: MLXArray) throws -> MLXArray? {
            let tAnalysis0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let spec = analysisFrame(hopTD)
            if enableProfiling {
                profAnalysisSeconds += CFAbsoluteTimeGetCurrent() - tAnalysis0
            }

            let tFeatures0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let (featErb, featDf) = featuresFrame(spec)
            if enableProfiling {
                profFeaturesSeconds += CFAbsoluteTimeGetCurrent() - tFeatures0
            }
            specQueue.append(spec)
            frameCount += 1

            if frameCount <= convLookahead {
                return nil
            }

            let specT = specQueue.removeFirst()
            let tInfer0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let specEnhanced = try inferFrame(spec: specT, featErb: featErb, featDf: featDf)
            if enableProfiling {
                profInferSeconds += CFAbsoluteTimeGetCurrent() - tInfer0
            }

            let tSynth0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let out = synthesisFrame(specEnhanced.asType(.float32))
            if enableProfiling {
                profSynthesisSeconds += CFAbsoluteTimeGetCurrent() - tSynth0
            }
            hopsSinceMaterialize += 1
            if config.materializeEveryHops > 0, hopsSinceMaterialize >= config.materializeEveryHops {
                let tMat0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
                materializeStreamingState(output: out)
                if enableProfiling {
                    profMaterializeSeconds += CFAbsoluteTimeGetCurrent() - tMat0
                }
                hopsSinceMaterialize = 0
            }
            if enableProfiling {
                profHopCount += 1
            }
            return out
        }

        private func analysisFrame(_ hopTD: MLXArray) -> MLXArray {
            let frame = analysisMemCount > 0
                ? MLX.concatenated([analysisMem, hopTD], axis: 0)
                : hopTD
            let frameWin = frame * vorbisWindow
            let specComplex = MLXFFT.rfft(frameWin, axis: 0) * wnormArray
            let spec = MLX.stacked([specComplex.realPart(), specComplex.imaginaryPart()], axis: -1)
            updateAnalysisMemory(with: hopTD)
            return spec
        }

        private func synthesisFrame(_ specNorm: MLXArray) -> MLXArray {
            let complex = specNorm[0..., 0] + model.j * specNorm[0..., 1]
            var time = MLXFFT.irfft(complex, axis: 0)
            time = time * fftScaleArray
            time = time * vorbisWindow

            let out = time[0..<hopSize] + synthMem[0..<hopSize]
            updateSynthesisMemory(with: time)
            return out
        }

        private func updateAnalysisMemory(with hop: MLXArray) {
            guard analysisMemCount > 0 else { return }
            if analysisMemCount > hopSize {
                let split = analysisMemCount - hopSize
                let rotated = MLX.concatenated([
                    analysisMem[hopSize..<analysisMemCount],
                    analysisMem[0..<hopSize],
                ], axis: 0)
                analysisMem = MLX.concatenated([rotated[0..<split], hop], axis: 0)
            } else {
                analysisMem = hop[(hopSize - analysisMemCount)..<hopSize]
            }
        }

        private func updateSynthesisMemory(with time: MLXArray) {
            guard synthMemCount > 0 else { return }
            let xSecond = time[hopSize..<fftSize]
            if synthMemCount > hopSize {
                let split = synthMemCount - hopSize
                let rotated = MLX.concatenated([
                    synthMem[hopSize..<synthMemCount],
                    synthMem[0..<hopSize],
                ], axis: 0)
                let sFirst = rotated[0..<split] + xSecond[0..<split]
                let sSecond = xSecond[split..<(split + hopSize)]
                synthMem = MLX.concatenated([sFirst, sSecond], axis: 0)
            } else {
                synthMem = xSecond[0..<synthMemCount]
            }
        }

        private func featuresFrame(_ spec: MLXArray) -> (MLXArray, MLXArray) {
            let re = spec[0..., 0]
            let im = spec[0..., 1]
            let magSq = re.square() + im.square()

            let erb: MLXArray
            if let erbFBFrame {
                erb = MLX.matmul(magSq.expandedDimensions(axis: 0), erbFBFrame).squeezed()
            } else {
                var erbBands = [MLXArray]()
                erbBands.reserveCapacity(nbErb)
                var start = 0
                for width in model.erbBandWidths {
                    let stop = min(start + width, freqBins)
                    if stop > start {
                        erbBands.append(MLX.mean(magSq[start..<stop], axis: 0))
                    } else {
                        erbBands.append(MLXArray.zeros([1], type: Float.self).squeezed())
                    }
                    start = stop
                }
                erb = MLX.stacked(erbBands, axis: 0)
            }
            let erbDB = tenArray * MLX.log10(erb + epsEnergy)
            erbState = erbDB * oneMinusAlphaArray + erbState * alphaArray
            let featErb = (erbDB - erbState) / fortyArray

            let dfRe = re[0..<nbDf]
            let dfIm = im[0..<nbDf]
            let mag = MLX.sqrt(dfRe.square() + dfIm.square())
            dfState = mag * oneMinusAlphaArray + dfState * alphaArray
            let denom = MLX.sqrt(MLX.maximum(dfState, epsNorm))
            let featDfRe = dfRe / denom
            let featDfIm = dfIm / denom

            let featErbMX = featErb
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            var featDfMX = MLX.stacked([featDfRe, featDfIm], axis: -1)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            featDfMX = featDfMX.transposed(0, 3, 1, 2)
            return (featErbMX, featDfMX)
        }

        private func inferFrame(
            spec: MLXArray,
            featErb: MLXArray,
            featDf: MLXArray
        ) throws -> MLXArray {
            let specMX = spec
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .asType(inferenceDType)
            let featErbMX = featErb.asType(inferenceDType)
            let featDfMX = featDf.asType(inferenceDType)
            appendWithLimit(&encErb0In, featErbMX, maxLen: 3)
            appendWithLimit(&encDf0In, featDfMX, maxLen: 3)
            appendWithLimit(&specPast, spec, maxLen: dfOrder)

            let e0 = try applyConvLast(inputs: encErb0In, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
            let e1 = try applyConvLast(inputs: [e0], prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
            let e2 = try applyConvLast(inputs: [e1], prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
            let e3 = try applyConvLast(inputs: [e2], prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

            let c0 = try applyConvLast(inputs: encDf0In, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
            let c1 = try applyConvLast(inputs: [c0], prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

            var cemb = c1.transposed(0, 2, 3, 1).reshaped([1, 1, -1])
            cemb = relu(model.groupedLinear(cemb, weight: try model.w("enc.df_fc_emb.0.weight")))

            var emb = e3.transposed(0, 2, 3, 1).reshaped([1, 1, -1])
            emb = model.config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)

            emb = try squeezedGRUStep(
                emb,
                prefix: "enc.emb_gru",
                hiddenSize: model.config.embHiddenDim,
                linearOut: true,
                state: &encEmbState
            )

            let applyGains: Bool
            let applyGainZeros: Bool
            let applyDf: Bool
            if config.enableStageSkipping {
                let lsnr = sigmoid(model.linear(emb, weight: lsnrWeight, bias: lsnrBias)) * lsnrScale + lsnrOffset
                let lsnrValue = lsnr.asArray(Float.self).first ?? Float(model.config.lsnrMin)
                (applyGains, applyGainZeros, applyDf) = applyStages(lsnr: lsnrValue)
            } else {
                (applyGains, applyGainZeros, applyDf) = (true, false, true)
            }

            let mask: MLXArray
            if applyGains {
                mask = try erbDecoderStep(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
            } else if applyGainZeros {
                mask = zeroMaskFrame.asType(inferenceDType)
            } else {
                return specMX[0, 0, 0, 0..., 0...]
            }
            let specMasked = applyMask(spec: specMX, mask: mask)
            if !applyDf {
                return specMasked[0, 0, 0, 0..., 0...]
            }

            var dfCoefs = try dfDecoderStep(emb: emb, c0: c0)
            dfCoefs = dfCoefs.reshaped([1, 1, nbDf, dfOrder, 2]).transposed(0, 3, 1, 2, 4)

            let specEnhanced = try deepFilterAssign(spec: specMX, specMasked: specMasked, dfCoefs: dfCoefs, currentSpec: spec)
            return specEnhanced[0, 0, 0, 0..., 0...]
        }

        private func applyStages(lsnr: Float) -> (Bool, Bool, Bool) {
            if lsnr < config.minDbThresh {
                // Only noise detected: apply zero ERB mask and skip DF.
                return (false, true, false)
            }
            if lsnr > config.maxDbErbThresh {
                // Clean speech detected: skip ERB and DF.
                return (false, false, false)
            }
            if lsnr > config.maxDbDfThresh {
                // Mild noise: apply ERB gains only.
                return (true, false, false)
            }
            // Regular noisy speech: apply both stages.
            return (true, false, true)
        }

        private func applyConvLast(
            inputs: [MLXArray],
            prefix: String,
            main: Int,
            pointwise: Int?,
            bn: Int,
            fstride: Int
        ) throws -> MLXArray {
            var y = MLX.concatenated(inputs, axis: 2)
            y = try model.conv2dLayer(
                y,
                weight: model.w("\(prefix).\(main).weight"),
                bias: nil,
                fstride: fstride,
                lookahead: 0
            )
            if let pointwise {
                y = try model.conv2dLayer(
                    y,
                    weight: model.w("\(prefix).\(pointwise).weight"),
                    bias: nil,
                    fstride: 1,
                    lookahead: 0
                )
            }
            y = try model.batchNorm(y, prefix: "\(prefix).\(bn)")
            y = relu(y)
            let t = y.shape[2]
            return y[0..., 0..., (t - 1)..<t, 0...]
        }

        private func squeezedGRUStep(
            _ x: MLXArray,
            prefix: String,
            hiddenSize: Int,
            linearOut: Bool,
            state: inout [MLXArray]?
        ) throws -> MLXArray {
            var y = relu(model.groupedLinear(x, weight: try model.w("\(prefix).linear_in.0.weight")))
            var nextState = [MLXArray]()
            var layer = 0

            while model.weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
                let prevState: MLXArray
                if let state, layer < state.count {
                    prevState = state[layer]
                } else {
                    prevState = MLXArray.zeros([y.shape[0], hiddenSize], type: Float.self)
                }
                let h = try gruLayerStep(y, prefix: "\(prefix).gru", layer: layer, hiddenSize: hiddenSize, prevState: prevState)
                nextState.append(h)
                y = h.expandedDimensions(axis: 1)
                layer += 1
            }

            state = nextState
            if linearOut, model.weights["\(prefix).linear_out.0.weight"] != nil {
                y = relu(model.groupedLinear(y, weight: try model.w("\(prefix).linear_out.0.weight")))
            }
            return y
        }

        private func gruLayerStep(
            _ x: MLXArray,
            prefix: String,
            layer: Int,
            hiddenSize: Int,
            prevState: MLXArray
        ) throws -> MLXArray {
            let wih = try model.w("\(prefix).weight_ih_l\(layer)")
            let whh = try model.w("\(prefix).weight_hh_l\(layer)")
            let bih = try model.w("\(prefix).bias_ih_l\(layer)")
            let bhh = try model.w("\(prefix).bias_hh_l\(layer)")

            let xt = x[0..., 0, 0...]
            let gx = MLX.addMM(bih, xt, wih.transposed())
            let gh = MLX.addMM(bhh, prevState, whh.transposed())

            let xr = gx[0..., 0..<hiddenSize]
            let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
            let xn = gx[0..., (2 * hiddenSize)...]
            let hr = gh[0..., 0..<hiddenSize]
            let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
            let hn = gh[0..., (2 * hiddenSize)...]

            let r = sigmoid(xr + hr)
            let z = sigmoid(xz + hz)
            let n = tanh(xn + r * hn)
            return (MLXArray(Float(1.0)) - z) * n + z * prevState
        }

        private func erbDecoderStep(
            emb: MLXArray,
            e3: MLXArray,
            e2: MLXArray,
            e1: MLXArray,
            e0: MLXArray
        ) throws -> MLXArray {
            var embDec = try squeezedGRUStep(
                emb,
                prefix: "erb_dec.emb_gru",
                hiddenSize: model.config.embHiddenDim,
                linearOut: true,
                state: &erbDecState
            )
            let f8 = e3.shape[3]
            embDec = embDec.reshaped([1, 1, f8, -1]).transposed(0, 3, 1, 2)

            var d3 = relu(try model.applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
            d3 = relu(try model.applyRegularBlock(d3, prefix: "erb_dec.convt3"))
            var d2 = relu(try model.applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
            d2 = relu(try model.applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
            var d1 = relu(try model.applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
            d1 = relu(try model.applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
            let d0 = relu(try model.applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
            let out = try model.applyOutputConv(d0, prefix: "erb_dec.conv0_out")
            return sigmoid(out)
        }

        private func dfDecoderStep(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
            var c = try squeezedGRUStep(
                emb,
                prefix: "df_dec.df_gru",
                hiddenSize: model.config.dfHiddenDim,
                linearOut: false,
                state: &dfDecState
            )
            if model.weights["df_dec.df_skip.weight"] != nil {
                c = c + model.groupedLinear(emb, weight: try model.w("df_dec.df_skip.weight"))
            }

            appendWithLimit(&dfConvpIn, c0, maxLen: model.config.dfPathwayKernelSizeT)
            let c0Seq = MLX.concatenated(dfConvpIn, axis: 2)
            var c0p = try model.conv2dLayer(
                c0Seq,
                weight: model.w("df_dec.df_convp.1.weight"),
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
            c0p = try model.conv2dLayer(
                c0p,
                weight: model.w("df_dec.df_convp.2.weight"),
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
            c0p = relu(try model.batchNorm(c0p, prefix: "df_dec.df_convp.3"))
            let t = c0p.shape[2]
            c0p = c0p[0..., 0..., (t - 1)..<t, 0...]
            c0p = c0p.transposed(0, 2, 3, 1)

            let dfOut = tanh(model.groupedLinear(c, weight: try model.w("df_dec.df_out.0.weight")))
                .reshaped([1, 1, nbDf, dfOrder * 2])
            return dfOut + c0p
        }

        private func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
            let b = mask.shape[0]
            let t = mask.shape[2]
            let e = mask.shape[3]
            let flat = mask.reshaped([b * t, e])
            let gains = MLX.matmul(flat, model.erbInvFB).reshaped([b, 1, t, model.config.freqBins, 1])
            return spec * gains
        }

        private func deepFilterAssign(
            spec: MLXArray,
            specMasked: MLXArray,
            dfCoefs: MLXArray,
            currentSpec: MLXArray
        ) throws -> MLXArray {
            let left = dfOrder - dfLookahead - 1

            let past = Array(specPast.dropLast())
            let needPast = max(0, left - past.count)
            var specWindow = [MLXArray]()
            specWindow.reserveCapacity(dfOrder)
            for _ in 0..<needPast {
                specWindow.append(zeroSpecFrame)
            }
            if left > 0 {
                specWindow.append(contentsOf: past.suffix(left))
            }
            specWindow.append(currentSpec)

            let futureAvailable = min(dfLookahead, specQueue.count)
            if futureAvailable > 0 {
                specWindow.append(contentsOf: specQueue.prefix(futureAvailable))
            }
            for _ in futureAvailable..<dfLookahead {
                specWindow.append(zeroSpecFrame)
            }

            let specHistory = MLX.stacked(specWindow, axis: 0)
            let specLow = specHistory[0..., 0..<nbDf, 0...]
            let coef = dfCoefs[0, 0..., 0, 0..<nbDf, 0...]

            let sr = specLow[0..., 0..., 0]
            let si = specLow[0..., 0..., 1]
            let cr = coef[0..., 0..., 0]
            let ci = coef[0..., 0..., 1]

            let outReal = MLX.sum(sr * cr - si * ci, axis: 0)
            let outImag = MLX.sum(sr * ci + si * cr, axis: 0)

            let low = MLX.stacked([outReal, outImag], axis: -1)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)

            if model.config.encConcat {
                let high = specMasked[0..., 0..., 0..., nbDf..., 0...]
                return MLX.concatenated([low, high], axis: 3)
            }

            let highUnmasked = spec[0..., 0..., 0..., nbDf..., 0...]
            let specDf = MLX.concatenated([low, highUnmasked], axis: 3)
            let lowAssigned = specDf[0..., 0..., 0..., 0..<nbDf, 0...]
            let highMasked = specMasked[0..., 0..., 0..., nbDf..., 0...]
            return MLX.concatenated([lowAssigned, highMasked], axis: 3)
        }

        private func appendWithLimit<T>(_ buffer: inout [T], _ value: T, maxLen: Int) {
            buffer.append(value)
            if buffer.count > maxLen {
                buffer.removeFirst(buffer.count - maxLen)
            }
        }

        // Materialize recurrent tensors each hop so MLX does not keep growing a long lazy graph.
        private func materializeStreamingState(output: MLXArray) {
            eval(output, analysisMem, synthMem, erbState, dfState)
            if let encEmbState {
                for x in encEmbState { eval(x) }
            }
            if let erbDecState {
                for x in erbDecState { eval(x) }
            }
            if let dfDecState {
                for x in dfDecState { eval(x) }
            }
        }

        private static func linspace(start: Float, end: Float, count: Int) -> [Float] {
            guard count > 1 else { return [start] }
            let step = (end - start) / Float(count - 1)
            return (0..<count).map { start + Float($0) * step }
        }
    }

    // MARK: - Network Forward

    private func forward(
        spec: MLXArray,
        featErb: MLXArray,
        featSpec5D: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let featSpec = featSpec5D
            .squeezed(axis: 1)
            .transposed(0, 3, 1, 2)

        let featErbShift = applyLookahead(feature: featErb, lookahead: config.convLookahead)
        let featSpecShift = applyLookahead(feature: featSpec, lookahead: config.convLookahead)

        let (e0, e1, e2, e3, emb, c0, lsnr) = try encode(featErb: featErbShift, featSpec: featSpecShift)

        let mask = try decodeErb(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        let specMasked = applyMask(spec: spec, mask: mask)

        let dfCoefs = try decodeDf(emb: emb, c0: c0)
        let b = dfCoefs.shape[0]
        let t = dfCoefs.shape[1]
        let dfCoefs5 = dfCoefs
            .reshaped([b, t, config.nbDf, config.dfOrder, 2])
            .transposed(0, 3, 1, 2, 4)

        let specEnhanced: MLXArray
        if config.encConcat {
            specEnhanced = deepFilter(spec: specMasked, coefs: dfCoefs5)
        } else {
            let specDf = deepFilter(spec: spec, coefs: dfCoefs5)
            let low = specDf[0..., 0..., 0..., 0..<config.nbDf, 0...]
            let high = specMasked[0..., 0..., 0..., config.nbDf..., 0...]
            specEnhanced = MLX.concatenated([low, high], axis: 3)
        }

        return (specEnhanced, mask, lsnr, dfCoefs5)
    }

    private func encode(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyEncoderConv(featErb, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
        let e1 = try applyEncoderConv(e0, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e2 = try applyEncoderConv(e1, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e3 = try applyEncoderConv(e2, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

        let c0 = try applyEncoderConv(featSpec, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
        let c1 = try applyEncoderConv(c0, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

        let b = c1.shape[0]
        let t = c1.shape[2]
        var cemb = c1.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        cemb = relu(groupedLinear(cemb, weight: try w("enc.df_fc_emb.0.weight")))

        var emb = e3.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        emb = config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)

        emb = try squeezedGRU(
            emb,
            prefix: "enc.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let lsnr = sigmoid(linear(
            emb,
            weight: try w("enc.lsnr_fc.0.weight"),
            bias: try w("enc.lsnr_fc.0.bias")
        )) * MLXArray(Float(config.lsnrMax - config.lsnrMin)) + MLXArray(Float(config.lsnrMin))

        return (e0, e1, e2, e3, emb, c0, lsnr)
    }

    private func decodeErb(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        var embDec = try squeezedGRU(
            emb,
            prefix: "erb_dec.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let b = embDec.shape[0]
        let t = embDec.shape[1]
        let f8 = e3.shape[3]
        embDec = embDec.reshaped([b, t, f8, -1]).transposed(0, 3, 1, 2)

        var d3 = relu(try applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
        // Matches MLX/PyTorch DF decoder: convt3 is a regular conv block, not transposed.
        d3 = relu(try applyRegularBlock(d3, prefix: "erb_dec.convt3"))
        var d2 = relu(try applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
        d2 = relu(try applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
        var d1 = relu(try applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
        d1 = relu(try applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
        let d0 = relu(try applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
        let out = try applyOutputConv(d0, prefix: "erb_dec.conv0_out")
        return sigmoid(out)
    }

    private func decodeDf(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
        var c = try squeezedGRU(
            emb,
            prefix: "df_dec.df_gru",
            hiddenSize: config.dfHiddenDim,
            linearOut: false
        )

        if weights["df_dec.df_skip.weight"] != nil {
            c = c + groupedLinear(emb, weight: try w("df_dec.df_skip.weight"))
        }

        var c0p = try conv2dLayer(
            c0,
            weight: try w("df_dec.df_convp.1.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = try conv2dLayer(
            c0p,
            weight: try w("df_dec.df_convp.2.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = relu(try batchNorm(c0p, prefix: "df_dec.df_convp.3"))
        c0p = c0p.transposed(0, 2, 3, 1)

        let b = c.shape[0]
        let t = c.shape[1]
        let dfOut = tanh(groupedLinear(c, weight: try w("df_dec.df_out.0.weight")))
            .reshaped([b, t, config.nbDf, config.dfOrder * 2])

        return dfOut + c0p
    }

    private func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
        let b = mask.shape[0]
        let t = mask.shape[2]
        let e = mask.shape[3]
        let flat = mask.reshaped([b * t, e])
        let gains = MLX.matmul(flat, erbInvFB).reshaped([b, 1, t, config.freqBins, 1])
        return spec * gains
    }

    private func deepFilter(spec: MLXArray, coefs: MLXArray) -> MLXArray {
        let t = spec.shape[2]
        let padLeft = config.dfOrder - 1 - config.dfLookahead
        let padRight = config.dfLookahead

        let specLow = spec[0..., 0, 0..., 0..<config.nbDf, 0...]
        let padded = MLX.padded(
            specLow,
            widths: [
                .init(0),
                .init((padLeft, padRight)),
                .init(0),
                .init(0),
            ],
            mode: .constant
        )

        var outFrames = [MLXArray]()
        outFrames.reserveCapacity(t)
        for i in 0..<t {
            let window = padded[0..., i..<(i + config.dfOrder), 0..., 0...]
            let coef = coefs[0..., 0..., i, 0..., 0...]
            let sr = window[0..., 0..., 0..., 0]
            let si = window[0..., 0..., 0..., 1]
            let cr = coef[0..., 0..., 0..., 0]
            let ci = coef[0..., 0..., 0..., 1]

            let outR = MLX.sum(sr * cr - si * ci, axis: 1)
            let outI = MLX.sum(sr * ci + si * cr, axis: 1)
            outFrames.append(MLX.stacked([outR, outI], axis: -1))
        }

        // Stack over time to get [B, 1, T, F_df, 2], matching spec layout.
        let low = MLX.stacked(outFrames, axis: 1).expandedDimensions(axis: 1)
        let high = spec[0..., 0..., 0..., config.nbDf..., 0...]
        return MLX.concatenated([low, high], axis: 3)
    }

    // MARK: - Layer Helpers

    private func applyEncoderConv(
        _ x: MLXArray,
        prefix: String,
        main: Int,
        pointwise: Int?,
        bn: Int,
        fstride: Int
    ) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).\(main).weight"),
            bias: nil,
            fstride: fstride,
            lookahead: 0
        )
        if let pointwise {
            y = try conv2dLayer(
                y,
                weight: w("\(prefix).\(pointwise).weight"),
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }
        y = try batchNorm(y, prefix: "\(prefix).\(bn)")
        return relu(y)
    }

    private func applyPathwayConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return relu(y)
    }

    private func applyTransposeBlock(_ x: MLXArray, prefix: String, fstride: Int) throws -> MLXArray {
        var y = try convTranspose2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            fstride: fstride,
            groups: config.convCh
        )
        y = try conv2dLayer(
            y,
            weight: w("\(prefix).1.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    private func applyRegularBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try conv2dLayer(
            y,
            weight: w("\(prefix).1.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    private func applyOutputConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return y
    }

    private func conv2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let inPerGroup = weight.shape[1]
        let inChannels = xBCHW.shape[1]
        let groups = max(1, inChannels / max(1, inPerGroup))

        let rawLeft = kT - 1 - lookahead
        let timeCrop = max(0, -rawLeft)
        let timePadLeft = max(0, rawLeft)
        let timePadRight = max(0, lookahead)
        let freqPad = kF / 2

        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        if timeCrop > 0, x.shape[1] > timeCrop {
            x = x[0..., timeCrop..., 0..., 0...]
        }
        x = MLX.padded(
            x,
            widths: [
                .init(0),
                .init((timePadLeft, timePadRight)),
                .init((freqPad, freqPad)),
                .init(0),
            ],
            mode: .constant
        )

        let wOHWI = weight.transposed(0, 2, 3, 1)
        var y = try groupedConv2d(input: x, weight: wOHWI, strideW: fstride, groups: groups)
        if let bias {
            y = y + bias.reshaped([1, 1, 1, bias.shape[0]])
        }
        return y.transposed(0, 3, 1, 2)
    }

    private func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))

        if groups <= 1 {
            let w = weight.transposed(1, 2, 3, 0)
            x = MLX.convTransposed2d(
                x,
                w,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }

        let inPerGroup = max(1, x.shape[3] / groups)
        let outPerGroup = weight.shape[1]
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)

        for g in 0..<groups {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]

            let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
            let wT = wg.transposed(1, 2, 3, 0)  // [out_pg, kT, kF, in_pg]
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }

        let y = MLX.concatenated(ys, axis: 3)
        _ = outPerGroup  // keep shape intent explicit
        return y.transposed(0, 3, 1, 2)
    }

    private func groupedConv2d(
        input: MLXArray,
        weight: MLXArray,
        strideW: Int,
        groups: Int
    ) throws -> MLXArray {
        MLX.conv2d(input, weight, stride: [1, strideW], padding: [0, 0], groups: groups)
    }

    private func batchNorm(_ x: MLXArray, prefix: String) throws -> MLXArray {
        let gamma = try w("\(prefix).weight")
        let beta = try w("\(prefix).bias")
        let mean = try w("\(prefix).running_mean")
        let variance = try w("\(prefix).running_var")

        let gammaR = gamma.reshaped([1, gamma.shape[0], 1, 1])
        let betaR = beta.reshaped([1, beta.shape[0], 1, 1])
        let meanR = mean.reshaped([1, mean.shape[0], 1, 1])
        let varianceR = variance.reshaped([1, variance.shape[0], 1, 1])
        return ((x - meanR) / MLX.sqrt(varianceR + MLXArray(Float(1e-5)))) * gammaR + betaR
    }

    private func squeezedGRU(
        _ x: MLXArray,
        prefix: String,
        hiddenSize: Int,
        linearOut: Bool
    ) throws -> MLXArray {
        var y = relu(groupedLinear(x, weight: try w("\(prefix).linear_in.0.weight")))

        var layer = 0
        while weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
            y = try pytorchGRULayer(y, prefix: "\(prefix).gru", layer: layer, hiddenSize: hiddenSize)
            layer += 1
        }

        if linearOut, weights["\(prefix).linear_out.0.weight"] != nil {
            y = relu(groupedLinear(y, weight: try w("\(prefix).linear_out.0.weight")))
        }
        return y
    }

    private func groupedLinear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let groups = weight.shape[0]
        let ws = weight.shape[1]
        let hs = weight.shape[2]
        let b = x.shape[0]
        let t = x.shape[1]
        let reshaped = x.reshaped([b, t, groups, ws])
        let out = MLX.einsum("btgi,gih->btgh", reshaped, weight)
        return out.reshaped([b, t, groups * hs])
    }

    private func pytorchGRULayer(
        _ x: MLXArray,
        prefix: String,
        layer: Int,
        hiddenSize: Int
    ) throws -> MLXArray {
        let wih = try w("\(prefix).weight_ih_l\(layer)")
        let whh = try w("\(prefix).weight_hh_l\(layer)")
        let bih = try w("\(prefix).bias_ih_l\(layer)")
        let bhh = try w("\(prefix).bias_hh_l\(layer)")

        let b = x.shape[0]
        let t = x.shape[1]

        var h = MLXArray.zeros([b, hiddenSize], type: Float.self)
        var states = [MLXArray]()
        states.reserveCapacity(t)

        for i in 0..<t {
            let xt = x[0..., i, 0...]
            let gx = MLX.addMM(bih, xt, wih.transposed())
            let gh = MLX.addMM(bhh, h, whh.transposed())

            let xr = gx[0..., 0..<hiddenSize]
            let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
            let xn = gx[0..., (2 * hiddenSize)...]
            let hr = gh[0..., 0..<hiddenSize]
            let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
            let hn = gh[0..., (2 * hiddenSize)...]

            let r = sigmoid(xr + hr)
            let z = sigmoid(xz + hz)
            let n = tanh(xn + r * hn)
            h = (MLXArray(Float(1.0)) - z) * n + z * h
            states.append(h)
        }
        return MLX.stacked(states, axis: 1)
    }

    private func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let x2 = x.reshaped([b * t, x.shape[2]])
        var y = MLX.matmul(x2, weight.transposed())
        y = y + bias
        return y.reshaped([b, t, weight.shape[0]])
    }

    // MARK: - Feature Helpers

    private func bandMeanNorm(_ x: MLXArray) -> MLXArray {
        let frames = x.shape[0]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a
        let time = MLXArray.arange(frames).asType(.float32)
        let powers = MLX.pow(MLXArray(a), time) // [T]
        let invPowers = MLXArray(Float(1.0)) / powers

        let scaled = x * invPowers.expandedDimensions(axis: 1)
        let accum = cumsum(scaled, axis: 0)

        let initState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: x.shape[1]))
            .expandedDimensions(axis: 0)
        let state = powers.expandedDimensions(axis: 1) * (initState + MLXArray(oneMinusA) * accum)
        return (x - state) / MLXArray(Float(40.0))
    }

    private func bandUnitNorm(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let frames = real.shape[0]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a
        let time = MLXArray.arange(frames).asType(.float32)
        let powers = MLX.pow(MLXArray(a), time) // [T]
        let invPowers = MLXArray(Float(1.0)) / powers

        let mag = MLX.sqrt(real.square() + imag.square())
        let scaled = mag * invPowers.expandedDimensions(axis: 1)
        let accum = cumsum(scaled, axis: 0)

        let initState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: real.shape[1]))
            .expandedDimensions(axis: 0)
        let state = powers.expandedDimensions(axis: 1) * (initState + MLXArray(oneMinusA) * accum)
        let denom = MLX.sqrt(MLX.maximum(state, MLXArray(Float(1e-12))))
        return (real / denom, imag / denom)
    }

    private func normAlpha() -> Float {
        normAlphaValue
    }

    private static func computeNormAlpha(hopSize: Int, sampleRate: Int) -> Float {
        let aRaw = exp(-Float(hopSize) / Float(sampleRate))
        var precision = 3
        var a: Float = 1.0
        while a >= 1.0 {
            let scale = powf(10, Float(precision))
            a = (aRaw * scale).rounded() / scale
            precision += 1
        }
        return a
    }

    private func applyLookahead(feature: MLXArray, lookahead: Int) -> MLXArray {
        guard lookahead > 0 else { return feature }
        let t = feature.shape[2]
        guard t > lookahead else { return feature }
        let shifted = feature[0..., 0..., lookahead..<t, 0...]
        let pad = MLXArray.zeros([feature.shape[0], feature.shape[1], lookahead, feature.shape[3]], type: Float.self)
        return MLX.concatenated([shifted, pad], axis: 2)
    }

    // MARK: - Utility

    private func erbEnergies(_ specMagSq: MLXArray) -> MLXArray {
        var bands = [MLXArray]()
        bands.reserveCapacity(erbBandWidths.count)
        var start = 0
        for width in erbBandWidths {
            let stop = min(start + width, config.freqBins)
            if stop > start {
                bands.append(MLX.mean(specMagSq[0..., start..<stop], axis: 1))
            } else {
                bands.append(MLXArray.zeros([specMagSq.shape[0]], type: Float.self))
            }
            start = stop
        }
        return MLX.stacked(bands, axis: 1)
    }

    private static func libdfFreqToErb(_ freqHz: Float) -> Float {
        9.265 * log1p(freqHz / (24.7 * 9.265))
    }

    private static func libdfErbToFreq(_ erb: Float) -> Float {
        24.7 * 9.265 * (exp(erb / 9.265) - 1.0)
    }

    private static func libdfErbBandWidths(
        sampleRate: Int,
        fftSize: Int,
        nbBands: Int,
        minNbFreqs: Int
    ) -> [Int] {
        guard sampleRate > 0, fftSize > 0, nbBands > 0 else { return [] }

        let nyq = sampleRate / 2
        let freqWidth = Float(sampleRate) / Float(fftSize)
        let erbLow = libdfFreqToErb(0)
        let erbHigh = libdfFreqToErb(Float(nyq))
        let step = (erbHigh - erbLow) / Float(nbBands)

        var widths = [Int](repeating: 0, count: nbBands)
        var prevFreq = 0
        var freqOver = 0
        let minBins = max(1, minNbFreqs)

        for i in 1...nbBands {
            let f = libdfErbToFreq(erbLow + Float(i) * step)
            let fb = Int((f / freqWidth).rounded())
            var nbFreqs = fb - prevFreq - freqOver
            if nbFreqs < minBins {
                freqOver = minBins - nbFreqs
                nbFreqs = minBins
            } else {
                freqOver = 0
            }
            widths[i - 1] = max(1, nbFreqs)
            prevFreq = fb
        }

        widths[nbBands - 1] += 1  // fft_size/2 + 1 bins
        let target = fftSize / 2 + 1
        let total = widths.reduce(0, +)
        if total > target {
            widths[nbBands - 1] -= (total - target)
        } else if total < target {
            widths[nbBands - 1] += (target - total)
        }
        return widths
    }

    private static func vorbisWindow(size: Int) -> MLXArray {
        let half = max(1, size / 2)
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            let inner = sin(0.5 * Float.pi * (Float(i) + 0.5) / Float(half))
            window[i] = sin(0.5 * Float.pi * inner * inner)
        }
        return MLXArray(window)
    }

    private static func linspace(start: Float, end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    private func w(_ key: String) throws -> MLXArray {
        guard let value = weights[key] else {
            throw DeepFilterNetError.missingWeightKey(key)
        }
        return value
    }
}
