import Foundation
import MLX
import MLXNN

/// Stateful streaming speech enhancer for DeepFilterNet.
///
/// Processes audio incrementally, one hop (10ms at 48kHz) at a time. Maintains
/// internal recurrent state (GRU hidden states, analysis/synthesis overlap buffers)
/// across calls.
///
/// Create via ``DeepFilterNetModel/createStreamer(config:)``.
///
/// ```swift
/// let streamer = model.createStreamer()
/// let enhanced = try streamer.processChunk(audioChunk)
/// let tail = try streamer.flush()
/// ```
public final class DeepFilterNetStreamer {
    private struct StreamGRULayer {
        let wihT: MLXArray
        let whhT: MLXArray
        let bih: MLXArray
        let bhh: MLXArray
    }

    private struct StreamGRU {
        let linearInWeight: MLXArray
        let layers: [StreamGRULayer]
        let linearOutWeight: MLXArray?
    }

    private struct StaticRing {
        private(set) var values: [MLXArray]
        private(set) var totalWritten: Int = 0

        var capacity: Int { values.count }
        var count: Int { min(totalWritten, capacity) }
        var oldestAbsoluteIndex: Int { max(0, totalWritten - capacity) }

        init(capacity: Int, initial: MLXArray) {
            precondition(capacity > 0, "StaticRing capacity must be > 0")
            values = Array(repeating: initial, count: capacity)
        }

        mutating func reset() {
            totalWritten = 0
        }

        mutating func push(_ value: MLXArray) {
            values[totalWritten % capacity] = value
            totalWritten += 1
        }

        func get(absoluteIndex: Int) -> MLXArray? {
            guard absoluteIndex >= oldestAbsoluteIndex, absoluteIndex < totalWritten else {
                return nil
            }
            return values[absoluteIndex % capacity]
        }

        func orderedLast(_ n: Int) -> [MLXArray] {
            let k = min(max(0, n), totalWritten)
            guard k > 0 else { return [] }
            let start = totalWritten - k
            return (start..<(start + k)).compactMap { get(absoluteIndex: $0) }
        }
    }

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
    private let zeroSpecLowFrame: MLXArray
    private let zeroMaskFrame: MLXArray
    private let zeroEncErbFrame: MLXArray
    private let zeroEncDfFrame: MLXArray
    private let zeroDfConvpFrame: MLXArray
    private let specRingCapacity: Int
    private let dfConvpKernelSizeT: Int
    private let dfSpecLeft: Int
    private let analysisMemCount: Int
    private let synthMemCount: Int
    private let erbFBFrame: MLXArray?
    private let lsnrWeight: MLXArray
    private let lsnrBias: MLXArray
    private let lsnrScale: MLXArray
    private let lsnrOffset: MLXArray
    private let encDfFcEmbWeight: MLXArray
    private let dfDecSkipWeight: MLXArray?
    private let dfDecOutWeight: MLXArray
    private let encEmbGRU: StreamGRU
    private let erbDecEmbGRU: StreamGRU
    private let dfDecGRU: StreamGRU

    private var pendingSamples = MLXArray.zeros([0], type: Float.self)
    private var analysisMem: MLXArray
    private var synthMem: MLXArray
    private var erbState: MLXArray
    private var dfState: MLXArray

    private var specRing: StaticRing
    private var encErbHistory: MLXArray
    private var encDfHistory: MLXArray
    private var dfConvpHistory: MLXArray
    private var dfSpecHistory: MLXArray
    private var dfSpecHistoryInitialized = false

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
    private var profInferEncodeSeconds = 0.0
    private var profInferEmbSeconds = 0.0
    private var profInferErbSeconds = 0.0
    private var profInferDfSeconds = 0.0
    private var profSynthesisSeconds = 0.0
    private var profMaterializeSeconds = 0.0
    private let profilingForceEvalPerStage: Bool

    /// Creates a new streamer for the given model.
    ///
    /// Prefer using ``DeepFilterNetModel/createStreamer(config:)`` which validates
    /// that the model supports streaming.
    public init(model: DeepFilterNetModel, config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()) {
        self.model = model
        self.config = config
        self.enableProfiling = config.enableProfiling
        self.profilingForceEvalPerStage = config.profilingForceEvalPerStage

        self.fftSize = model.config.fftSize
        self.hopSize = model.config.hopSize
        self.freqBins = model.config.freqBins
        self.nbDf = model.config.nbDf
        self.nbErb = model.config.nbErb
        self.dfOrder = model.config.dfOrder
        self.dfLookahead = model.config.dfLookahead
        self.convLookahead = model.config.convLookahead

        let alpha = model.normAlphaValue
        self.alphaArray = MLXArray(alpha)
        self.oneMinusAlphaArray = MLXArray(Float(1.0) - alpha)
        self.fftScaleArray = MLXArray(Float(model.config.fftSize))
        self.vorbisWindow = model.vorbisWindowArray.asType(.float32)
        self.wnormArray = MLXArray(model.wnorm)
        self.inferenceDType = model.inferenceDType
        self.analysisMemCount = max(0, model.config.fftSize - model.config.hopSize)
        self.synthMemCount = max(0, model.config.fftSize - model.config.hopSize)
        self.zeroSpecFrame = MLXArray.zeros([model.config.freqBins, 2], type: Float.self)
        self.zeroSpecLowFrame = zeroSpecFrame[0..<model.config.nbDf, 0...]
        self.zeroMaskFrame = MLXArray.zeros([1, 1, 1, model.config.nbErb], type: Float.self)
        self.zeroEncErbFrame = MLXArray.zeros([1, 1, 1, model.config.nbErb], type: Float.self).asType(model.inferenceDType)
        self.zeroEncDfFrame = MLXArray.zeros([1, 2, 1, model.config.nbDf], type: Float.self).asType(model.inferenceDType)
        self.zeroDfConvpFrame = MLXArray.zeros([1, model.config.convCh, 1, model.config.nbDf], type: Float.self).asType(model.inferenceDType)
        self.dfConvpKernelSizeT = max(1, model.config.dfPathwayKernelSizeT)
        let leftHistory = max(0, model.config.dfOrder - model.config.dfLookahead - 1)
        self.dfSpecLeft = leftHistory
        self.specRingCapacity = max(8, leftHistory + model.config.convLookahead + model.config.dfLookahead + 4)
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
        self.encDfFcEmbWeight = Self.requireWeight(model, key: "enc.df_fc_emb.0.weight")
        self.dfDecSkipWeight = model.weights["df_dec.df_skip.weight"]
        self.dfDecOutWeight = Self.requireWeight(model, key: "df_dec.df_out.0.weight")
        self.encEmbGRU = Self.buildStreamGRU(model: model, prefix: "enc.emb_gru")
        self.erbDecEmbGRU = Self.buildStreamGRU(model: model, prefix: "erb_dec.emb_gru")
        self.dfDecGRU = Self.buildStreamGRU(model: model, prefix: "df_dec.df_gru")

        self.analysisMem = MLXArray.zeros([analysisMemCount], type: Float.self)
        self.synthMem = MLXArray.zeros([synthMemCount], type: Float.self)
        self.erbState = MLXArray(DeepFilterNetModel.linspace(start: -60.0, end: -90.0, count: model.config.nbErb))
        self.dfState = MLXArray(DeepFilterNetModel.linspace(start: 0.001, end: 0.0001, count: model.config.nbDf))
        self.specRing = StaticRing(capacity: specRingCapacity, initial: zeroSpecFrame)
        self.encErbHistory = MLX.repeated(zeroEncErbFrame, count: 3, axis: 2)
        self.encDfHistory = MLX.repeated(zeroEncDfFrame, count: 3, axis: 2)
        self.dfConvpHistory = MLX.repeated(zeroDfConvpFrame, count: dfConvpKernelSizeT, axis: 2)
        self.dfSpecHistory = MLX.repeated(zeroSpecLowFrame.expandedDimensions(axis: 0), count: model.config.dfOrder, axis: 0)
    }

    private static func requireWeight(_ model: DeepFilterNetModel, key: String) -> MLXArray {
        guard let weight = model.weights[key] else {
            preconditionFailure("Missing required DeepFilterNet weight: \(key)")
        }
        return weight
    }

    private static func buildStreamGRU(model: DeepFilterNetModel, prefix: String) -> StreamGRU {
        let linearIn = requireWeight(model, key: "\(prefix).linear_in.0.weight")
        let linearOut = model.weights["\(prefix).linear_out.0.weight"]

        var layers = [StreamGRULayer]()
        var layer = 0
        while model.weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
            let wihKey = "\(prefix).gru.weight_ih_l\(layer)"
            let whhKey = "\(prefix).gru.weight_hh_l\(layer)"
            let wihT = model.gruTransposedWeights[wihKey] ?? requireWeight(model, key: wihKey).transposed()
            let whhT = model.gruTransposedWeights[whhKey] ?? requireWeight(model, key: whhKey).transposed()
            let bih = requireWeight(model, key: "\(prefix).gru.bias_ih_l\(layer)")
            let bhh = requireWeight(model, key: "\(prefix).gru.bias_hh_l\(layer)")
            layers.append(StreamGRULayer(wihT: wihT, whhT: whhT, bih: bih, bhh: bhh))
            layer += 1
        }

        return StreamGRU(linearInWeight: linearIn, layers: layers, linearOutWeight: linearOut)
    }

    // MARK: - Public API

    /// Resets all internal state, allowing the streamer to be reused for a new audio stream.
    public func reset() {
        pendingSamples = MLXArray.zeros([0], type: Float.self)
        analysisMem = MLXArray.zeros([analysisMemCount], type: Float.self)
        synthMem = MLXArray.zeros([synthMemCount], type: Float.self)
        erbState = MLXArray(DeepFilterNetModel.linspace(start: -60.0, end: -90.0, count: nbErb))
        dfState = MLXArray(DeepFilterNetModel.linspace(start: 0.001, end: 0.0001, count: nbDf))
        specRing.reset()
        encErbHistory = MLX.repeated(zeroEncErbFrame, count: 3, axis: 2)
        encDfHistory = MLX.repeated(zeroEncDfFrame, count: 3, axis: 2)
        dfConvpHistory = MLX.repeated(zeroDfConvpFrame, count: dfConvpKernelSizeT, axis: 2)
        dfSpecHistory = MLX.repeated(zeroSpecLowFrame.expandedDimensions(axis: 0), count: dfOrder, axis: 0)
        dfSpecHistoryInitialized = false
        encEmbState = nil
        erbDecState = nil
        dfDecState = nil
        delayDropped = 0
        hopsSinceMaterialize = 0
        profHopCount = 0
        profAnalysisSeconds = 0.0
        profFeaturesSeconds = 0.0
        profInferSeconds = 0.0
        profInferEncodeSeconds = 0.0
        profInferEmbSeconds = 0.0
        profInferErbSeconds = 0.0
        profInferDfSeconds = 0.0
        profSynthesisSeconds = 0.0
        profMaterializeSeconds = 0.0
    }

    /// Processes a chunk of audio and returns any enhanced samples ready for output.
    ///
    /// Internally buffers input and processes hop-by-hop (480 samples at 48kHz).
    /// May return zero samples if the pipeline hasn't produced output yet.
    ///
    /// - Parameters:
    ///   - chunk: Input audio samples as a 1D `MLXArray`.
    ///   - isLast: Set `true` to flush remaining samples after the final chunk.
    /// - Returns: Enhanced audio samples (may be empty if buffering).
    public func processChunk(_ chunk: MLXArray, isLast: Bool = false) throws -> MLXArray {
        guard chunk.ndim == 1 else {
            throw DeepFilterNetError.invalidAudioShape(chunk.shape)
        }
        let chunkF32 = chunk.asType(.float32)
        if !isLast, pendingSamples.shape[0] == 0, chunkF32.shape[0] == hopSize {
            if var out = try processHop(chunkF32) {
                if config.compensateDelay {
                    let totalDelay = fftSize - hopSize
                    if delayDropped < totalDelay {
                        let toDrop = min(totalDelay - delayDropped, out.shape[0])
                        if toDrop > 0 {
                            out = out[toDrop..<out.shape[0]]
                            delayDropped += toDrop
                        }
                    }
                }
                return out
            }
            return MLXArray.zeros([0], type: Float.self)
        }
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

    /// Processes a chunk of audio samples and returns enhanced output as a `[Float]` array.
    ///
    /// Convenience overload that accepts and returns Swift arrays.
    public func processChunk(_ chunk: [Float], isLast: Bool = false) throws -> [Float] {
        guard !chunk.isEmpty || isLast else { return [] }
        let y = try processChunk(MLXArray(chunk), isLast: isLast)
        if y.shape[0] == 0 {
            return []
        }
        return y.asArray(Float.self)
    }

    /// Flushes remaining buffered samples and returns any final enhanced output.
    public func flush() throws -> [Float] {
        try processChunk([], isLast: true)
    }

    /// Flushes remaining buffered samples and returns any final enhanced output as an `MLXArray`.
    public func flushMLX() throws -> MLXArray {
        try processChunk(MLXArray.zeros([0], type: Float.self), isLast: true)
    }

    /// Returns a formatted profiling summary, or `nil` if profiling is disabled.
    ///
    /// Requires ``DeepFilterNetStreamingConfig/enableProfiling`` to be `true`.
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
                    infer.enc: %.3fs (%.1f%% infer)
                    infer.emb: %.3fs (%.1f%% infer)
                    infer.erb: %.3fs (%.1f%% infer)
                    infer.df:  %.3fs (%.1f%% infer)
                  synthesis:   %.3fs (%.1f%%)
                  materialize: %.3fs (%.1f%%)
                """,
            profHopCount,
            total,
            perHopMs,
            profAnalysisSeconds, pct(profAnalysisSeconds),
            profFeaturesSeconds, pct(profFeaturesSeconds),
            profInferSeconds, pct(profInferSeconds),
            profInferEncodeSeconds, profInferSeconds > 0 ? (profInferEncodeSeconds / profInferSeconds) * 100.0 : 0.0,
            profInferEmbSeconds, profInferSeconds > 0 ? (profInferEmbSeconds / profInferSeconds) * 100.0 : 0.0,
            profInferErbSeconds, profInferSeconds > 0 ? (profInferErbSeconds / profInferSeconds) * 100.0 : 0.0,
            profInferDfSeconds, profInferSeconds > 0 ? (profInferDfSeconds / profInferSeconds) * 100.0 : 0.0,
            profSynthesisSeconds, pct(profSynthesisSeconds),
            profMaterializeSeconds, pct(profMaterializeSeconds)
        )
    }

    // MARK: - Per-Hop Processing

    private func processHop(_ hopTD: MLXArray) throws -> MLXArray? {
        let tAnalysis0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        let spec = analysisFrame(hopTD)
        if enableProfiling, profilingForceEvalPerStage {
            eval(spec)
        }
        if enableProfiling {
            profAnalysisSeconds += CFAbsoluteTimeGetCurrent() - tAnalysis0
        }

        let tFeatures0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        let (featErb, featDf) = featuresFrame(spec)
        if enableProfiling, profilingForceEvalPerStage {
            eval(featErb, featDf)
        }
        if enableProfiling {
            profFeaturesSeconds += CFAbsoluteTimeGetCurrent() - tFeatures0
        }
        specRing.push(spec)
        encErbHistory = appendHistoryFrame(encErbHistory, frame: featErb.asType(inferenceDType))
        encDfHistory = appendHistoryFrame(encDfHistory, frame: featDf.asType(inferenceDType))

        if specRing.totalWritten <= convLookahead {
            return nil
        }
        let targetFrameIndex = specRing.totalWritten - 1 - convLookahead
        guard let specT = specRing.get(absoluteIndex: targetFrameIndex) else {
            return nil
        }
        updateDfSpecHistory(targetFrameIndex: targetFrameIndex)
        let tInfer0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        let specEnhanced = try inferFrame(spec: specT, targetFrameIndex: targetFrameIndex)
        if enableProfiling, profilingForceEvalPerStage {
            eval(specEnhanced)
        }
        if enableProfiling {
            profInferSeconds += CFAbsoluteTimeGetCurrent() - tInfer0
        }

        let tSynth0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        let out = synthesisFrame(specEnhanced.asType(.float32))
        if enableProfiling, profilingForceEvalPerStage {
            eval(out)
        }
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

    // MARK: - Analysis / Synthesis

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

    // MARK: - Features

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

    // MARK: - Inference

    private func inferFrame(
        spec: MLXArray,
        targetFrameIndex: Int
    ) throws -> MLXArray {
        let tEncode0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        let specMX = spec
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
            .asType(inferenceDType)

        let encErbSeq = encErbHistory
        let encDfSeq = encDfHistory

        let e0 = try applyConvLast(input: encErbSeq, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
        let e1 = try applyConvLast(input: e0, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e2 = try applyConvLast(input: e1, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e3 = try applyConvLast(input: e2, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

        let c0 = try applyConvLast(input: encDfSeq, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
        let c1 = try applyConvLast(input: c0, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

        var cemb = c1.transposed(0, 2, 3, 1).reshaped([1, 1, -1])
        cemb = relu(model.groupedLinear(cemb, weight: encDfFcEmbWeight))

        var emb = e3.transposed(0, 2, 3, 1).reshaped([1, 1, -1])
        emb = model.config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)
        if enableProfiling, profilingForceEvalPerStage {
            eval(e3, c1, emb)
        }
        if enableProfiling {
            profInferEncodeSeconds += CFAbsoluteTimeGetCurrent() - tEncode0
        }

        let tEmb0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        emb = squeezedGRUStep(
            emb,
            gru: encEmbGRU,
            hiddenSize: model.config.embHiddenDim,
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
        if enableProfiling, profilingForceEvalPerStage {
            eval(emb)
        }
        if enableProfiling {
            profInferEmbSeconds += CFAbsoluteTimeGetCurrent() - tEmb0
        }

        let mask: MLXArray
        let tErb0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        if applyGains {
            mask = try erbDecoderStep(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        } else if applyGainZeros {
            mask = zeroMaskFrame.asType(inferenceDType)
        } else {
            return specMX[0, 0, 0, 0..., 0...]
        }
        let specMasked = model.applyMask(spec: specMX, mask: mask)
        if enableProfiling, profilingForceEvalPerStage {
            eval(mask, specMasked)
        }
        if enableProfiling {
            profInferErbSeconds += CFAbsoluteTimeGetCurrent() - tErb0
        }
        if !applyDf {
            return specMasked[0, 0, 0, 0..., 0...]
        }

        let tDf0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
        var dfCoefs = try dfDecoderStep(emb: emb, c0: c0)
        dfCoefs = dfCoefs.reshaped([1, 1, nbDf, dfOrder, 2]).transposed(0, 3, 1, 2, 4)

        let specEnhanced = try deepFilterAssign(
            spec: specMX,
            specMasked: specMasked,
            dfCoefs: dfCoefs,
            targetFrameIndex: targetFrameIndex
        )
        if enableProfiling, profilingForceEvalPerStage {
            eval(dfCoefs, specEnhanced)
        }
        if enableProfiling {
            profInferDfSeconds += CFAbsoluteTimeGetCurrent() - tDf0
        }
        return specEnhanced[0, 0, 0, 0..., 0...]
    }

    private func applyStages(lsnr: Float) -> (Bool, Bool, Bool) {
        if lsnr < config.minDbThresh {
            return (false, true, false)
        }
        if lsnr > config.maxDbErbThresh {
            return (false, false, false)
        }
        if lsnr > config.maxDbDfThresh {
            return (true, false, false)
        }
        return (true, false, true)
    }

    private func applyConvLast(
        input: MLXArray,
        prefix: String,
        main: Int,
        pointwise: Int?,
        bn: Int,
        fstride: Int
    ) throws -> MLXArray {
        var y = input
        y = try model.conv2dLayer(
            y,
            weightKey: "\(prefix).\(main).weight",
            bias: nil,
            fstride: fstride,
            lookahead: 0
        )
        if let pointwise {
            y = try model.conv2dLayer(
                y,
                weightKey: "\(prefix).\(pointwise).weight",
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
        gru: StreamGRU,
        hiddenSize: Int,
        state: inout [MLXArray]?
    ) -> MLXArray {
        var y = relu(model.groupedLinear(x, weight: gru.linearInWeight))
        var nextState = [MLXArray]()
        nextState.reserveCapacity(gru.layers.count)
        for (layer, layerDef) in gru.layers.enumerated() {
            let prevState: MLXArray
            if let state, layer < state.count {
                prevState = state[layer]
            } else {
                prevState = MLXArray.zeros([y.shape[0], hiddenSize], type: Float.self)
            }
            let h = gruLayerStep(y, layer: layerDef, hiddenSize: hiddenSize, prevState: prevState)
            nextState.append(h)
            y = h.expandedDimensions(axis: 1)
        }

        state = nextState
        if let linearOut = gru.linearOutWeight {
            y = relu(model.groupedLinear(y, weight: linearOut))
        }
        return y
    }

    private func gruLayerStep(
        _ x: MLXArray,
        layer: StreamGRULayer,
        hiddenSize: Int,
        prevState: MLXArray
    ) -> MLXArray {
        let xt = x[0..., 0, 0...]
        let gx = MLX.addMM(layer.bih, xt, layer.wihT)
        let gh = MLX.addMM(layer.bhh, prevState, layer.whhT)

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

    // MARK: - Decoders

    private func erbDecoderStep(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        var embDec = squeezedGRUStep(
            emb,
            gru: erbDecEmbGRU,
            hiddenSize: model.config.embHiddenDim,
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
        var c = squeezedGRUStep(
            emb,
            gru: dfDecGRU,
            hiddenSize: model.config.dfHiddenDim,
            state: &dfDecState
        )
        if let dfDecSkipWeight {
            c = c + model.groupedLinear(emb, weight: dfDecSkipWeight)
        }

        dfConvpHistory = appendHistoryFrame(dfConvpHistory, frame: c0)
        let c0Seq = dfConvpHistory
        var c0p = try model.conv2dLayer(
            c0Seq,
            weightKey: "df_dec.df_convp.1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = try model.conv2dLayer(
            c0p,
            weightKey: "df_dec.df_convp.2.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = relu(try model.batchNorm(c0p, prefix: "df_dec.df_convp.3"))
        let t = c0p.shape[2]
        c0p = c0p[0..., 0..., (t - 1)..<t, 0...]
        c0p = c0p.transposed(0, 2, 3, 1)

        let dfOut = tanh(model.groupedLinear(c, weight: dfDecOutWeight))
            .reshaped([1, 1, nbDf, dfOrder * 2])
        return dfOut + c0p
    }

    // MARK: - Deep Filter & History

    private func deepFilterAssign(
        spec: MLXArray,
        specMasked: MLXArray,
        dfCoefs: MLXArray,
        targetFrameIndex _: Int
    ) throws -> MLXArray {
        let specLow = dfSpecHistory
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

    private func appendHistoryFrame(_ history: MLXArray, frame: MLXArray) -> MLXArray {
        let t = history.shape[2]
        guard t > 1 else { return frame }
        return MLX.concatenated([history[0..., 0..., 1..<t, 0...], frame], axis: 2)
    }

    private func updateDfSpecHistory(targetFrameIndex: Int) {
        if !dfSpecHistoryInitialized {
            var frames = [MLXArray]()
            frames.reserveCapacity(dfOrder)
            for k in 0..<dfOrder {
                let absoluteIndex = targetFrameIndex - dfSpecLeft + k
                if let frame = specRing.get(absoluteIndex: absoluteIndex) {
                    frames.append(frame[0..<nbDf, 0...])
                } else {
                    frames.append(zeroSpecLowFrame)
                }
            }
            dfSpecHistory = MLX.stacked(frames, axis: 0)
            dfSpecHistoryInitialized = true
            return
        }
        let newRightIndex = targetFrameIndex - dfSpecLeft + dfOrder - 1
        let nextLow: MLXArray
        if let frame = specRing.get(absoluteIndex: newRightIndex) {
            nextLow = frame[0..<nbDf, 0...]
        } else {
            nextLow = zeroSpecLowFrame
        }
        dfSpecHistory = MLX.concatenated([dfSpecHistory[1..<dfOrder, 0..., 0...], nextLow.expandedDimensions(axis: 0)], axis: 0)
    }

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
}
