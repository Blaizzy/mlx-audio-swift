import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

public struct FSMNVADEncoderConfig: Codable, Sendable {
    public var inputDim: Int
    public var inputAffineDim: Int
    public var fsmnLayers: Int
    public var linearDim: Int
    public var projDim: Int
    public var lorder: Int
    public var rorder: Int
    public var lstride: Int
    public var rstride: Int
    public var outputAffineDim: Int
    public var outputDim: Int

    public init(
        inputDim: Int = 400,
        inputAffineDim: Int = 140,
        fsmnLayers: Int = 4,
        linearDim: Int = 250,
        projDim: Int = 128,
        lorder: Int = 20,
        rorder: Int = 0,
        lstride: Int = 1,
        rstride: Int = 0,
        outputAffineDim: Int = 140,
        outputDim: Int = 248
    ) {
        self.inputDim = inputDim
        self.inputAffineDim = inputAffineDim
        self.fsmnLayers = fsmnLayers
        self.linearDim = linearDim
        self.projDim = projDim
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.outputAffineDim = outputAffineDim
        self.outputDim = outputDim
    }

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case inputAffineDim = "input_affine_dim"
        case fsmnLayers = "fsmn_layers"
        case linearDim = "linear_dim"
        case projDim = "proj_dim"
        case lorder
        case rorder
        case lstride
        case rstride
        case outputAffineDim = "output_affine_dim"
        case outputDim = "output_dim"
    }
}

public struct FSMNVADConfig: Codable, Sendable {
    public var modelType: String
    public var architecture: String
    public var encoder: FSMNVADEncoderConfig
    public var sampleRate: Int
    public var nMels: Int
    public var frameLength: Int
    public var frameShift: Int
    public var lfrM: Int
    public var lfrN: Int
    public var maxEndSilenceTime: Int
    public var maxStartSilenceTime: Int
    public var windowSizeMs: Int
    public var silToSpeechTimeThres: Int
    public var speechToSilTimeThres: Int
    public var speechNoiseThres: Float
    public var silPdfIds: [Int]
    public var frameInMs: Int

    public init(
        modelType: String = "fsmn",
        architecture: String = "fsmn_vad",
        encoder: FSMNVADEncoderConfig = FSMNVADEncoderConfig(),
        sampleRate: Int = 16_000,
        nMels: Int = 80,
        frameLength: Int = 25,
        frameShift: Int = 10,
        lfrM: Int = 5,
        lfrN: Int = 1,
        maxEndSilenceTime: Int = 800,
        maxStartSilenceTime: Int = 3_000,
        windowSizeMs: Int = 200,
        silToSpeechTimeThres: Int = 150,
        speechToSilTimeThres: Int = 150,
        speechNoiseThres: Float = 0.6,
        silPdfIds: [Int] = [0],
        frameInMs: Int = 10
    ) {
        self.modelType = modelType
        self.architecture = architecture
        self.encoder = encoder
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.frameLength = frameLength
        self.frameShift = frameShift
        self.lfrM = lfrM
        self.lfrN = lfrN
        self.maxEndSilenceTime = maxEndSilenceTime
        self.maxStartSilenceTime = maxStartSilenceTime
        self.windowSizeMs = windowSizeMs
        self.silToSpeechTimeThres = silToSpeechTimeThres
        self.speechToSilTimeThres = speechToSilTimeThres
        self.speechNoiseThres = speechNoiseThres
        self.silPdfIds = silPdfIds
        self.frameInMs = frameInMs
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture
        case encoder
        case sampleRate = "sample_rate"
        case nMels = "n_mels"
        case frameLength = "frame_length"
        case frameShift = "frame_shift"
        case lfrM = "lfr_m"
        case lfrN = "lfr_n"
        case maxEndSilenceTime = "max_end_silence_time"
        case maxStartSilenceTime = "max_start_silence_time"
        case windowSizeMs = "window_size_ms"
        case silToSpeechTimeThres = "sil_to_speech_time_thres"
        case speechToSilTimeThres = "speech_to_sil_time_thres"
        case speechNoiseThres = "speech_noise_thres"
        case silPdfIds = "sil_pdf_ids"
        case frameInMs = "frame_in_ms"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "fsmn"
        self.architecture = try container.decodeIfPresent(String.self, forKey: .architecture) ?? "fsmn_vad"
        self.encoder = try container.decodeIfPresent(FSMNVADEncoderConfig.self, forKey: .encoder) ?? FSMNVADEncoderConfig()
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000
        self.nMels = try container.decodeIfPresent(Int.self, forKey: .nMels) ?? 80
        self.frameLength = try container.decodeIfPresent(Int.self, forKey: .frameLength) ?? 25
        self.frameShift = try container.decodeIfPresent(Int.self, forKey: .frameShift) ?? 10
        self.lfrM = try container.decodeIfPresent(Int.self, forKey: .lfrM) ?? 5
        self.lfrN = try container.decodeIfPresent(Int.self, forKey: .lfrN) ?? 1
        self.maxEndSilenceTime = try container.decodeIfPresent(Int.self, forKey: .maxEndSilenceTime) ?? 800
        self.maxStartSilenceTime = try container.decodeIfPresent(Int.self, forKey: .maxStartSilenceTime) ?? 3_000
        self.windowSizeMs = try container.decodeIfPresent(Int.self, forKey: .windowSizeMs) ?? 200
        self.silToSpeechTimeThres = try container.decodeIfPresent(Int.self, forKey: .silToSpeechTimeThres) ?? 150
        self.speechToSilTimeThres = try container.decodeIfPresent(Int.self, forKey: .speechToSilTimeThres) ?? 150
        self.speechNoiseThres = try container.decodeIfPresent(Float.self, forKey: .speechNoiseThres) ?? 0.6
        self.silPdfIds = try container.decodeIfPresent([Int].self, forKey: .silPdfIds) ?? [0]
        self.frameInMs = try container.decodeIfPresent(Int.self, forKey: .frameInMs) ?? 10
    }
}

private final class FSMNMemoryBlock: Module {
    let padLeft: Int
    @ModuleInfo(key: "conv_left") var convLeft: Conv1d

    init(projDim: Int, lorder: Int, lstride: Int = 1) {
        self.padLeft = (lorder - 1) * lstride
        _convLeft.wrappedValue = Conv1d(
            inputChannels: projDim,
            outputChannels: projDim,
            kernelSize: lorder,
            stride: 1,
            padding: 0,
            groups: projDim,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.concatenated([
            MLXArray.zeros([x.dim(0), padLeft, x.dim(2)], dtype: x.dtype),
            x,
        ], axis: 1)
        return x + convLeft(padded)
    }
}

private final class FSMNLayer: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo(key: "fsmn_block") var fsmnBlock: FSMNMemoryBlock
    @ModuleInfo var affine: Linear

    init(linearDim: Int, projDim: Int, lorder: Int, lstride: Int = 1) {
        _linear.wrappedValue = Linear(linearDim, projDim, bias: false)
        _fsmnBlock.wrappedValue = FSMNMemoryBlock(projDim: projDim, lorder: lorder, lstride: lstride)
        _affine.wrappedValue = Linear(projDim, linearDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(affine(fsmnBlock(linear(x))))
    }
}

public final class FSMNVADEncoder: Module {
    public let config: FSMNVADEncoderConfig

    @ModuleInfo(key: "in_linear1") var inLinear1: Linear
    @ModuleInfo(key: "in_linear2") var inLinear2: Linear
    @ModuleInfo fileprivate var fsmn: [FSMNLayer]
    @ModuleInfo(key: "out_linear1") var outLinear1: Linear
    @ModuleInfo(key: "out_linear2") var outLinear2: Linear

    public init(config: FSMNVADEncoderConfig = FSMNVADEncoderConfig()) {
        self.config = config
        _inLinear1.wrappedValue = Linear(config.inputDim, config.inputAffineDim)
        _inLinear2.wrappedValue = Linear(config.inputAffineDim, config.linearDim)
        _fsmn.wrappedValue = (0..<config.fsmnLayers).map { _ in
            FSMNLayer(
                linearDim: config.linearDim,
                projDim: config.projDim,
                lorder: config.lorder,
                lstride: config.lstride
            )
        }
        _outLinear1.wrappedValue = Linear(config.linearDim, config.outputAffineDim)
        _outLinear2.wrappedValue = Linear(config.outputAffineDim, config.outputDim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = inLinear1(x)
        hidden = relu(inLinear2(hidden))
        for layer in fsmn {
            hidden = layer(hidden)
        }
        hidden = outLinear1(hidden)
        hidden = outLinear2(hidden)
        return softmax(hidden, axis: -1)
    }
}

private enum FSMNVADState {
    case startPointNotDetected
    case inSpeechSegment
    case endPointDetected
}

private enum FSMNVADFrameState {
    case invalid
    case speech
    case silence
}

private enum FSMNVADAudioChangeState {
    case speechToSpeech
    case speechToSilence
    case silenceToSilence
    case silenceToSpeech
    case invalid
}

private final class FSMNVADSpeechBuffer {
    var startMs = 0
    var endMs = 0
    var containSegStartPoint = false
    var containSegEndPoint = false

    func reset() {
        startMs = 0
        endMs = 0
        containSegStartPoint = false
        containSegEndPoint = false
    }
}

private final class FSMNVADWindowDetector {
    let winSizeFrame: Int
    let silToSpeechFrameCountThreshold: Int
    let speechToSilFrameCountThreshold: Int
    var curWinPos = 0
    var winSum = 0
    var winState: [Int]
    var previousFrameState: FSMNVADFrameState = .silence

    init(windowSizeMs: Int, silToSpeechTime: Int, speechToSilTime: Int, frameSizeMs: Int) {
        self.winSizeFrame = windowSizeMs / frameSizeMs
        self.silToSpeechFrameCountThreshold = silToSpeechTime / frameSizeMs
        self.speechToSilFrameCountThreshold = speechToSilTime / frameSizeMs
        self.winState = Array(repeating: 0, count: max(winSizeFrame, 1))
    }

    func reset() {
        curWinPos = 0
        winSum = 0
        winState = Array(repeating: 0, count: max(winSizeFrame, 1))
        previousFrameState = .silence
    }

    func detectOneFrame(_ frameState: FSMNVADFrameState) -> FSMNVADAudioChangeState {
        let current = frameState == .speech ? 1 : 0
        winSum -= winState[curWinPos]
        winSum += current
        winState[curWinPos] = current
        curWinPos = (curWinPos + 1) % winState.count

        if previousFrameState == .silence && winSum >= silToSpeechFrameCountThreshold {
            previousFrameState = .speech
            return .silenceToSpeech
        }
        if previousFrameState == .speech && winSum <= speechToSilFrameCountThreshold {
            previousFrameState = .silence
            return .speechToSilence
        }
        if previousFrameState == .silence { return .silenceToSilence }
        if previousFrameState == .speech { return .speechToSpeech }
        return .invalid
    }
}

private final class FSMNVADPostprocessStats {
    var dataBufStartFrame = 0
    var frameCount = 0
    var latestConfirmedSpeechFrame = 0
    var latestConfirmedSilenceFrame = -1
    var continuousSilenceFrameCount = 0
    var vadStateMachine: FSMNVADState = .startPointNotDetected
    var confirmedStartFrame = -1
    var confirmedEndFrame = -1
    var numberEndTimeDetected = 0
    var silFrame = 0
    let silPdfIds: [Int]
    var noiseAverageDecibel: Float = -100.0
    var preEndSilenceDetected = false
    var outputDataBuf: [FSMNVADSpeechBuffer] = []
    var outputDataBufOffset = 0
    let maxEndSilFrameCountThreshold: Int
    let speechNoiseThreshold: Float
    var scores: [[Float]] = []
    var maxTimeOut = false
    var decibel: [Float] = []
    var dataBuf: [Float] = []
    var dataBufAll: [Float] = []
    var lastDropFrames = 0

    init(silPdfIds: [Int], maxEndSilFrameCountThreshold: Int, speechNoiseThreshold: Float) {
        self.silPdfIds = silPdfIds
        self.maxEndSilFrameCountThreshold = maxEndSilFrameCountThreshold
        self.speechNoiseThreshold = speechNoiseThreshold
    }
}

private final class FSMNVADPostprocess {
    let config: FSMNVADConfig
    let windowDetector: FSMNVADWindowDetector
    let stats: FSMNVADPostprocessStats

    init(config: FSMNVADConfig) {
        self.config = config
        self.windowDetector = FSMNVADWindowDetector(
            windowSizeMs: config.windowSizeMs,
            silToSpeechTime: config.silToSpeechTimeThres,
            speechToSilTime: config.speechToSilTimeThres,
            frameSizeMs: config.frameInMs
        )
        self.stats = FSMNVADPostprocessStats(
            silPdfIds: config.silPdfIds,
            maxEndSilFrameCountThreshold: config.maxEndSilenceTime - config.speechToSilTimeThres,
            speechNoiseThreshold: config.speechNoiseThres
        )
    }

    private var frameShiftSamples: Int {
        config.frameInMs * config.sampleRate / 1000
    }

    private func computeDecibel(_ waveform: [Float]) {
        let frameSampleLength = config.frameLength * config.sampleRate / 1000
        let shift = frameShiftSamples
        if stats.dataBufAll.isEmpty {
            stats.dataBufAll = waveform
            stats.dataBuf = waveform
        } else {
            stats.dataBufAll.append(contentsOf: waveform)
        }
        guard waveform.count >= frameSampleLength else { return }
        var start = 0
        while start + frameSampleLength <= waveform.count {
            var energy: Float = 0
            for sample in waveform[start..<(start + frameSampleLength)] {
                energy += sample * sample
            }
            stats.decibel.append(10.0 * log10(energy + 1e-6))
            start += shift
        }
    }

    private func computeScores(_ scores: [[Float]]) {
        stats.frameCount += scores.count
        stats.scores.append(contentsOf: scores)
    }

    private func latencyFrameCountAtStartPoint() -> Int {
        var vadLatency = windowDetector.winSizeFrame
        vadLatency += config.windowSizeMs / config.frameInMs
        return vadLatency
    }

    private func popDataBufTillFrame(_ frameIndex: Int) {
        while stats.dataBufStartFrame < frameIndex {
            if stats.dataBuf.count >= frameShiftSamples {
                stats.dataBufStartFrame += 1
                let start = max(0, (stats.dataBufStartFrame - stats.lastDropFrames) * frameShiftSamples)
                stats.dataBuf = start < stats.dataBufAll.count ? Array(stats.dataBufAll[start...]) : []
            } else {
                break
            }
        }
    }

    private func popDataToOutputBuffer(
        startFrame: Int,
        frameCount: Int,
        firstFrameIsStartPoint: Bool,
        lastFrameIsEndPoint: Bool
    ) {
        popDataBufTillFrame(startFrame)
        if stats.outputDataBuf.isEmpty || firstFrameIsStartPoint {
            let buffer = FSMNVADSpeechBuffer()
            buffer.reset()
            buffer.startMs = startFrame * config.frameInMs
            buffer.endMs = buffer.startMs
            stats.outputDataBuf.append(buffer)
        }
        guard let current = stats.outputDataBuf.last else { return }
        stats.dataBufStartFrame += frameCount
        current.endMs = (startFrame + frameCount) * config.frameInMs
        if firstFrameIsStartPoint {
            current.containSegStartPoint = true
        }
        if lastFrameIsEndPoint {
            current.containSegEndPoint = true
        }
    }

    private func onSilenceDetected(_ validFrame: Int) {
        stats.latestConfirmedSilenceFrame = validFrame
        if stats.vadStateMachine == .startPointNotDetected {
            popDataBufTillFrame(validFrame)
        }
    }

    private func onVoiceDetected(_ validFrame: Int) {
        stats.latestConfirmedSpeechFrame = validFrame
        popDataToOutputBuffer(
            startFrame: validFrame,
            frameCount: 1,
            firstFrameIsStartPoint: false,
            lastFrameIsEndPoint: false
        )
    }

    private func onVoiceStart(_ startFrame: Int, fakeResult: Bool = false) {
        if stats.confirmedStartFrame == -1 {
            stats.confirmedStartFrame = startFrame
        }
        if !fakeResult && stats.vadStateMachine == .startPointNotDetected {
            popDataToOutputBuffer(
                startFrame: stats.confirmedStartFrame,
                frameCount: 1,
                firstFrameIsStartPoint: true,
                lastFrameIsEndPoint: false
            )
        }
    }

    private func onVoiceEnd(_ endFrame: Int, fakeResult: Bool, isLastFrame: Bool) {
        if stats.latestConfirmedSpeechFrame + 1 < endFrame {
            for frame in (stats.latestConfirmedSpeechFrame + 1)..<endFrame {
                onVoiceDetected(frame)
            }
        }
        if stats.confirmedEndFrame == -1 {
            stats.confirmedEndFrame = endFrame
        }
        if !fakeResult {
            stats.silFrame = 0
            popDataToOutputBuffer(
                startFrame: stats.confirmedEndFrame,
                frameCount: 1,
                firstFrameIsStartPoint: false,
                lastFrameIsEndPoint: true
            )
        }
        stats.numberEndTimeDetected += 1
        _ = isLastFrame
    }

    private func maybeOnVoiceEndIfLastFrame(_ isFinalFrame: Bool, currentFrameIndex: Int) {
        if isFinalFrame {
            onVoiceEnd(currentFrameIndex, fakeResult: false, isLastFrame: true)
            stats.vadStateMachine = .endPointDetected
        }
    }

    private func resetDetection() {
        stats.continuousSilenceFrameCount = 0
        stats.latestConfirmedSpeechFrame = 0
        stats.latestConfirmedSilenceFrame = -1
        stats.confirmedStartFrame = -1
        stats.confirmedEndFrame = -1
        stats.vadStateMachine = .startPointNotDetected
        windowDetector.reset()
        stats.silFrame = 0
        if let last = stats.outputDataBuf.last, last.containSegEndPoint {
            let dropFrames = last.endMs / config.frameInMs
            let realDropFrames = dropFrames - stats.lastDropFrames
            stats.lastDropFrames = dropFrames
            let sampleDrop = realDropFrames * frameShiftSamples
            stats.dataBufAll = sampleDrop < stats.dataBufAll.count ? Array(stats.dataBufAll[sampleDrop...]) : []
            stats.decibel = realDropFrames < stats.decibel.count ? Array(stats.decibel[realDropFrames...]) : []
            stats.scores = realDropFrames < stats.scores.count ? Array(stats.scores[realDropFrames...]) : []
        }
    }

    private func frameState(at index: Int) -> FSMNVADFrameState {
        if index < 0 || index >= stats.decibel.count || index >= stats.scores.count {
            return .silence
        }
        let currentDecibel = stats.decibel[index]
        let currentSNR = currentDecibel - stats.noiseAverageDecibel
        if currentDecibel < -100.0 {
            return .silence
        }

        var sumScore: Float = 0
        var noiseProb: Float = log(1e-7)
        if !stats.silPdfIds.isEmpty {
            if stats.silPdfIds.count > 1 {
                for id in stats.silPdfIds where id < stats.scores[index].count {
                    sumScore += stats.scores[index][id]
                }
            } else if let id = stats.silPdfIds.first, id < stats.scores[index].count {
                sumScore = stats.scores[index][id]
            }
            sumScore = max(min(sumScore, 1.0 - 1e-7), 1e-7)
            noiseProb = log(sumScore)
            sumScore = 1.0 - sumScore
        }

        let speechProb = log(sumScore)
        if exp(speechProb) >= exp(noiseProb) + stats.speechNoiseThreshold {
            if currentSNR >= -100.0 && currentDecibel >= -100.0 {
                return .speech
            }
            return .silence
        }

        if stats.noiseAverageDecibel < -99.9 {
            stats.noiseAverageDecibel = currentDecibel
        } else {
            stats.noiseAverageDecibel = (
                currentDecibel + stats.noiseAverageDecibel * Float(100 - 1)
            ) / Float(100)
        }
        return .silence
    }

    private func detectOneFrame(
        _ currentFrameState: FSMNVADFrameState,
        currentFrameIndex: Int,
        isFinalFrame: Bool
    ) {
        let stateChange = windowDetector.detectOneFrame(currentFrameState == .speech ? .speech : .silence)
        let frameShiftMs = config.frameInMs

        switch stateChange {
        case .silenceToSpeech:
            stats.continuousSilenceFrameCount = 0
            stats.preEndSilenceDetected = false
            if stats.vadStateMachine == .startPointNotDetected {
                let startFrame = max(
                    stats.dataBufStartFrame,
                    currentFrameIndex - latencyFrameCountAtStartPoint()
                )
                onVoiceStart(startFrame)
                stats.vadStateMachine = .inSpeechSegment
                if startFrame + 1 <= currentFrameIndex {
                    for frame in (startFrame + 1)...currentFrameIndex {
                        onVoiceDetected(frame)
                    }
                }
            } else if stats.vadStateMachine == .inSpeechSegment {
                if stats.latestConfirmedSpeechFrame + 1 < currentFrameIndex {
                    for frame in (stats.latestConfirmedSpeechFrame + 1)..<currentFrameIndex {
                        onVoiceDetected(frame)
                    }
                }
                if currentFrameIndex - stats.confirmedStartFrame + 1 > 60_000 / frameShiftMs {
                    onVoiceEnd(currentFrameIndex, fakeResult: false, isLastFrame: false)
                    stats.vadStateMachine = .endPointDetected
                } else if !isFinalFrame {
                    onVoiceDetected(currentFrameIndex)
                } else {
                    maybeOnVoiceEndIfLastFrame(isFinalFrame, currentFrameIndex: currentFrameIndex)
                }
            }

        case .speechToSilence:
            stats.continuousSilenceFrameCount = 0
            if stats.vadStateMachine == .inSpeechSegment {
                if currentFrameIndex - stats.confirmedStartFrame + 1 > 60_000 / frameShiftMs {
                    onVoiceEnd(currentFrameIndex, fakeResult: false, isLastFrame: false)
                    stats.vadStateMachine = .endPointDetected
                } else if !isFinalFrame {
                    onVoiceDetected(currentFrameIndex)
                } else {
                    maybeOnVoiceEndIfLastFrame(isFinalFrame, currentFrameIndex: currentFrameIndex)
                }
            }

        case .speechToSpeech:
            stats.continuousSilenceFrameCount = 0
            if stats.vadStateMachine == .inSpeechSegment {
                if currentFrameIndex - stats.confirmedStartFrame + 1 > 60_000 / frameShiftMs {
                    stats.maxTimeOut = true
                    onVoiceEnd(currentFrameIndex, fakeResult: false, isLastFrame: false)
                    stats.vadStateMachine = .endPointDetected
                } else if !isFinalFrame {
                    onVoiceDetected(currentFrameIndex)
                } else {
                    maybeOnVoiceEndIfLastFrame(isFinalFrame, currentFrameIndex: currentFrameIndex)
                }
            }

        case .silenceToSilence:
            stats.continuousSilenceFrameCount += 1
            if stats.vadStateMachine == .startPointNotDetected {
                if isFinalFrame && stats.numberEndTimeDetected == 0 {
                    if stats.latestConfirmedSilenceFrame + 1 < currentFrameIndex {
                        for frame in (stats.latestConfirmedSilenceFrame + 1)..<currentFrameIndex {
                            onSilenceDetected(frame)
                        }
                    }
                    onVoiceStart(0, fakeResult: true)
                    onVoiceEnd(0, fakeResult: true, isLastFrame: false)
                    stats.vadStateMachine = .endPointDetected
                } else if currentFrameIndex >= latencyFrameCountAtStartPoint() {
                    onSilenceDetected(currentFrameIndex - latencyFrameCountAtStartPoint())
                }
            } else if stats.vadStateMachine == .inSpeechSegment {
                if stats.continuousSilenceFrameCount * frameShiftMs >= stats.maxEndSilFrameCountThreshold {
                    var lookbackFrame = stats.maxEndSilFrameCountThreshold / frameShiftMs
                    lookbackFrame -= config.windowSizeMs / frameShiftMs / 2
                    lookbackFrame -= 1
                    lookbackFrame = max(0, lookbackFrame)
                    onVoiceEnd(currentFrameIndex - lookbackFrame, fakeResult: false, isLastFrame: false)
                    stats.vadStateMachine = .endPointDetected
                } else if currentFrameIndex - stats.confirmedStartFrame + 1 > 60_000 / frameShiftMs {
                    onVoiceEnd(currentFrameIndex, fakeResult: false, isLastFrame: false)
                    stats.vadStateMachine = .endPointDetected
                } else if stats.continuousSilenceFrameCount <= config.windowSizeMs / frameShiftMs / 2 && !isFinalFrame {
                    onVoiceDetected(currentFrameIndex)
                } else {
                    maybeOnVoiceEndIfLastFrame(isFinalFrame, currentFrameIndex: currentFrameIndex)
                }
            }

        case .invalid:
            break
        }

        if stats.vadStateMachine == .endPointDetected {
            resetDetection()
        }
    }

    private func detectLastFrames() {
        if stats.vadStateMachine == .endPointDetected { return }
        let blockSize = stats.scores.count
        for i in stride(from: blockSize - 1, through: 0, by: -1) {
            let frameIndex = stats.frameCount - 1 - i
            let state = frameState(at: frameIndex - stats.lastDropFrames)
            detectOneFrame(state, currentFrameIndex: frameIndex, isFinalFrame: i == 0)
        }
    }

    func forward(scores: [[Float]], waveform: [Float]) -> [[Int]] {
        computeDecibel(waveform)
        computeScores(scores)
        detectLastFrames()

        var segments: [[Int]] = []
        if !stats.outputDataBuf.isEmpty {
            var index = stats.outputDataBufOffset
            while index < stats.outputDataBuf.count {
                let segment = stats.outputDataBuf[index]
                stats.outputDataBufOffset += 1
                segments.append([segment.startMs, segment.endMs])
                index += 1
            }
        }
        return segments
    }
}

public final class FSMNVAD: Module {
    public static let defaultRepository = "mlx-community/fsmn-vad"

    public let config: FSMNVADConfig
    private var cmvnShift: [Float]?
    private var cmvnScale: [Float]?
    @ModuleInfo public var encoder: FSMNVADEncoder

    public init(config: FSMNVADConfig = FSMNVADConfig()) {
        self.config = config
        _encoder.wrappedValue = FSMNVADEncoder(config: config.encoder)
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        encoder(features)
    }

    public func extractFeatures(_ waveform: MLXArray, sampleRate: Int = 16_000) throws -> MLXArray {
        var samples = waveform.asType(.float32).asArray(Float.self)
        if sampleRate != config.sampleRate {
            samples = try resampleAudio(samples, from: sampleRate, to: config.sampleRate)
        }
        let audio = MLXArray(samples) * Float(1 << 15)
        let winLen = config.sampleRate * config.frameLength / 1000
        let winInc = config.sampleRate * config.frameShift / 1000
        let fbank = Self.computeKaldiFbank(
            audio,
            sampleRate: config.sampleRate,
            winLen: winLen,
            winInc: winInc,
            numMels: config.nMels
        )
        var features = Self.applyLFR(fbank, lfrM: config.lfrM, lfrN: config.lfrN)
        if let cmvnShift, let cmvnScale, cmvnShift.count == features.dim(1), cmvnScale.count == features.dim(1) {
            features = (features + MLXArray(cmvnShift)) * MLXArray(cmvnScale)
        }
        return features
    }

    public func detect(_ waveform: MLXArray, sampleRate: Int = 16_000) throws -> [[Int]] {
        var samples = waveform.asType(.float32).asArray(Float.self)
        if sampleRate != config.sampleRate {
            samples = try resampleAudio(samples, from: sampleRate, to: config.sampleRate)
        }
        let features = try extractFeatures(MLXArray(samples), sampleRate: config.sampleRate)
        let scores = encoder(features.expandedDimensions(axis: 0))
        eval(scores)
        let values = scores.asArray(Float.self)
        let time = scores.dim(1)
        let dim = scores.dim(2)
        var rows: [[Float]] = []
        rows.reserveCapacity(time)
        for t in 0..<time {
            let start = t * dim
            rows.append(Array(values[start..<(start + dim)]))
        }
        return FSMNVADPostprocess(config: config).forward(scores: rows, waveform: samples)
    }

    public static func fromPretrained(
        _ repoId: String = defaultRepository,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> FSMNVAD {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            return try fromModelDirectory(URL(fileURLWithPath: NSString(string: repoId).expandingTildeInPath))
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["cmvn.json", "am.mvn"],
            hfToken: hfToken,
            cache: cache
        )
        return try fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) throws -> FSMNVAD {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(FSMNVADConfig.self, from: Data(contentsOf: configURL))
        let model = FSMNVAD(config: config)
        let weights = try loadArrays(url: modelDir.appendingPathComponent("model.safetensors"))
        try model.encoder.update(parameters: ModuleParameters.unflattened(sanitizeEncoderWeights(weights)), verify: .all)
        try model.loadCMVN(from: modelDir)
        eval(model)
        return model
    }

    private struct CMVNJSON: Decodable {
        let shift: [Float]
        let scale: [Float]
    }

    private func loadCMVN(from modelDir: URL) throws {
        let cmvnURL = modelDir.appendingPathComponent("cmvn.json")
        if FileManager.default.fileExists(atPath: cmvnURL.path) {
            let cmvn = try JSONDecoder().decode(CMVNJSON.self, from: Data(contentsOf: cmvnURL))
            cmvnShift = cmvn.shift
            cmvnScale = cmvn.scale
            return
        }
        let mvnURL = modelDir.appendingPathComponent("am.mvn")
        if FileManager.default.fileExists(atPath: mvnURL.path) {
            let parsed = try Self.parseKaldiCMVN(Data(contentsOf: mvnURL))
            cmvnShift = parsed.shift
            cmvnScale = parsed.scale
        }
    }

    private static func sanitizeEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            let newKey = key.hasPrefix("encoder.") ? String(key.dropFirst("encoder.".count)) : key
            sanitized[newKey] = value
        }
        return sanitized
    }

    private static func computeKaldiFbank(
        _ waveform: MLXArray,
        sampleRate: Int,
        winLen: Int,
        winInc: Int,
        numMels: Int
    ) -> MLXArray {
        let audioLength = waveform.dim(0)
        guard audioLength >= winLen else {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }
        let numFrames = 1 + (audioLength - winLen) / winInc
        var frames: [MLXArray] = []
        frames.reserveCapacity(numFrames)
        for frameIndex in 0..<numFrames {
            let start = frameIndex * winInc
            var frame = waveform[start..<(start + winLen)]
            frame = frame - mean(frame)
            if winLen > 1 {
                let first = frame[0..<1]
                let rest = frame[1..<winLen] - Float(0.97) * frame[0..<(winLen - 1)]
                frame = concatenated([first, rest], axis: 0)
            }
            frames.append(frame)
        }
        var frameTensor = stacked(frames, axis: 0)
        let nFft = nextPowerOfTwo(winLen)
        frameTensor = frameTensor * hammingWindow(size: winLen, periodic: false)
        if nFft > winLen {
            frameTensor = concatenated([
                frameTensor,
                MLXArray.zeros([numFrames, nFft - winLen], type: Float.self),
            ], axis: 1)
        }
        let spectrum = abs(MLXFFT.rfft(frameTensor, n: nFft, axis: 1)).square()
        let melBank = kaldiMelFilterbank(
            numBins: numMels,
            nFft: nFft,
            sampleRate: sampleRate,
            lowFreq: 20.0,
            highFreq: 0.0
        )
        return log(maximum(matmul(spectrum, melBank), MLXArray(Float(1e-8))))
    }

    private static func applyLFR(_ features: MLXArray, lfrM: Int, lfrN: Int) -> MLXArray {
        let time = features.dim(0)
        let dim = features.dim(1)
        guard time > 0 else { return MLXArray.zeros([0, dim * lfrM], type: Float.self) }
        let values = features.asArray(Float.self)
        let leftPad = (lfrM - 1) / 2
        let paddedTime = time + leftPad
        let outputTime = (paddedTime + lfrN - 1) / lfrN
        var output = Array(repeating: Float(0), count: outputTime * dim * lfrM)

        func sourceFrame(_ index: Int, feature: Int) -> Float {
            let sourceIndex: Int
            if index < leftPad {
                sourceIndex = 0
            } else if index - leftPad < time {
                sourceIndex = index - leftPad
            } else {
                sourceIndex = time - 1
            }
            return values[sourceIndex * dim + feature]
        }

        for outFrame in 0..<outputTime {
            let start = outFrame * lfrN
            for stackIndex in 0..<lfrM {
                let frameIndex = start + stackIndex
                for feature in 0..<dim {
                    output[(outFrame * lfrM + stackIndex) * dim + feature] = sourceFrame(frameIndex, feature: feature)
                }
            }
        }
        return MLXArray(output, [outputTime, dim * lfrM])
    }

    private static func parseKaldiCMVN(_ data: Data) throws -> (shift: [Float], scale: [Float]) {
        let text = String(decoding: data, as: UTF8.self)
        func parseBlock(_ marker: String) -> [Float]? {
            guard let markerRange = text.range(of: marker),
                  let open = text[markerRange.upperBound...].firstIndex(of: "["),
                  let close = text[open...].firstIndex(of: "]")
            else { return nil }
            return text[text.index(after: open)..<close]
                .split { $0 == " " || $0 == "\n" || $0 == "\t" }
                .compactMap { Float($0) }
        }
        guard let shift = parseBlock("<AddShift>"),
              let scale = parseBlock("<Rescale>")
        else {
            throw NSError(
                domain: "FSMNVAD",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Could not parse Kaldi CMVN data"]
            )
        }
        return (shift, scale)
    }

    private static func kaldiMelFilterbank(
        numBins: Int,
        nFft: Int,
        sampleRate: Int,
        lowFreq: Float,
        highFreq: Float
    ) -> MLXArray {
        let numFFTbins = nFft / 2
        let nyquist = 0.5 * Float(sampleRate)
        let high = highFreq <= 0 ? highFreq + nyquist : highFreq
        let fftBinWidth = Float(sampleRate) / Float(nFft)
        let melLow = kaldiMel(lowFreq)
        let melHigh = kaldiMel(high)
        let melDelta = (melHigh - melLow) / Float(numBins + 1)

        var values = Array(repeating: Float(0), count: (numFFTbins + 1) * numBins)
        for bin in 0..<numBins {
            let leftMel = melLow + Float(bin) * melDelta
            let centerMel = melLow + Float(bin + 1) * melDelta
            let rightMel = melLow + Float(bin + 2) * melDelta
            for fftBin in 0..<numFFTbins {
                let mel = kaldiMel(fftBinWidth * Float(fftBin))
                let upSlope = (mel - leftMel) / (centerMel - leftMel)
                let downSlope = (rightMel - mel) / (rightMel - centerMel)
                values[fftBin * numBins + bin] = max(0, min(upSlope, downSlope))
            }
        }
        return MLXArray(values, [numFFTbins + 1, numBins])
    }

    private static func kaldiMel(_ frequency: Float) -> Float {
        1127.0 * log(1.0 + frequency / 700.0)
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        guard value > 1 else { return max(value, 1) }
        var n = 1
        while n < value {
            n <<= 1
        }
        return n
    }
}
