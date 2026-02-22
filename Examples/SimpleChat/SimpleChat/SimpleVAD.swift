@preconcurrency import AVFoundation
import os
import Speech

@MainActor
protocol SimpleVADDelegate: AnyObject {
    func didStartSpeaking()
    func didStopSpeaking(transcription: String?)
}

@MainActor
final class SimpleVAD {
    private enum LifecycleState {
        case idle
        case starting
        case ready
        case failed
    }

    weak var delegate: SimpleVADDelegate?
    var hangTime: TimeInterval

    private var analyzer: SpeechAnalyzer?
    private var analyzerInputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var analysisFormat: AVAudioFormat?
    private var analyzerConverter: AVAudioConverter?
    private var transcriberTask: Task<Void, Never>?
    private var lifecycleState: LifecycleState = .idle

    private let transcriptState = TranscriptState()

    init(hangTime: TimeInterval = 2.0) {
        self.hangTime = hangTime

        Task {
            await setupSpeechPipeline()
        }
    }

    deinit {
        analyzerInputContinuation?.finish()
        transcriberTask?.cancel()
    }

    func process(buffer: AVAudioPCMBuffer) {
        guard lifecycleState == .ready, let copied = buffer.deepCopy() else { return }
        processBuffer(copied)
    }

    func reset() {
        Task { [weak self] in
            await self?.transcriptState.reset()
        }
    }

    private func setupSpeechPipeline() async {
        guard lifecycleState == .idle else { return }
        lifecycleState = .starting

        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale.current) else {
            print("Warning: Current locale (\(Locale.current)) is not supported for speech detection.")
            lifecycleState = .failed
            return
        }

        let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
        do {
            try await prepareAssets(for: transcriber)
        } catch {
            print("Error: Unable to prepare on-device transcription: \(error)")
            lifecycleState = .failed
            return
        }

        let detectionOptions = SpeechDetector.DetectionOptions(sensitivityLevel: .medium)
        let detector = SpeechDetector(detectionOptions: detectionOptions, reportResults: false)

        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [detector, transcriber]) else {
            print("Error: Speech detector unavailable until required assets are installed.")
            lifecycleState = .failed
            return
        }

        let inputStream = AsyncStream<AnalyzerInput> { [weak self] continuation in
            Task { @MainActor [weak self] in
                self?.analyzerInputContinuation = continuation
            }
            continuation.onTermination = { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.analyzerInputContinuation = nil
                }
            }
        }

        let analyzer = SpeechAnalyzer(
            modules: [detector, transcriber],
            options: .init(priority: .userInitiated, modelRetention: .lingering)
        )

        self.analyzer = analyzer
        analysisFormat = format
        analyzerConverter = nil
        lifecycleState = .ready

        Task { [weak self] in
            guard let self else { return }
            do {
                try await analyzer.start(inputSequence: inputStream)
            } catch {
                print("Error: Speech analyzer failed: \(error)")
                lifecycleState = .failed
            }
        }

        transcriberTask?.cancel()
        transcriberTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await result in transcriber.results {
                    let update = await transcriptState.recordResult(result, now: CACurrentMediaTime())
                    if update.didStart {
                        delegate?.didStartSpeaking()
                        print("Did start listening.")
                    }
                    guard case let .stop(transcription, reason, idleDuration) = update.stopDecision else { continue }
                    logStop(reason: reason, idleDuration: idleDuration)
                    delegate?.didStopSpeaking(transcription: transcription)
                }
            } catch {
                print("Error: Transcriber results failed: \(error)")
            }
        }
    }

    private func prepareAssets(for transcriber: SpeechTranscriber) async throws {
        if let installationRequest = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await installationRequest.downloadAndInstall()
        }
    }

    private func processBuffer(_ buffer: AVAudioPCMBuffer) {
        enqueueForSpeechDetection(buffer)

        let now = CACurrentMediaTime()
        let timeout = hangTime
        Task { [weak self] in
            guard let self else { return }
            let stopDecision = await transcriptState.timeoutDecision(now: now, timeout: timeout)
            guard case let .stop(transcription, reason, idleDuration) = stopDecision else { return }
            logStop(reason: reason, idleDuration: idleDuration)
            delegate?.didStopSpeaking(transcription: transcription)
        }
    }

    private func enqueueForSpeechDetection(_ buffer: AVAudioPCMBuffer) {
        guard let continuation = analyzerInputContinuation else { return }
        guard let converted = convertBufferIfNeeded(buffer) else { return }
        continuation.yield(AnalyzerInput(buffer: converted))
    }

    private func convertBufferIfNeeded(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let analysisFormat else { return buffer }
        if formatsMatch(buffer.format, analysisFormat) { return buffer }

        if analyzerConverter == nil ||
            !formatsMatch(analyzerConverter?.inputFormat, buffer.format) ||
            !formatsMatch(analyzerConverter?.outputFormat, analysisFormat) {
            analyzerConverter = AVAudioConverter(from: buffer.format, to: analysisFormat)
        }

        guard let converter = analyzerConverter else {
            print("Error: Unable to create audio converter for speech detection.")
            return nil
        }

        let ratio = analysisFormat.sampleRate / buffer.format.sampleRate
        let capacity = max(AVAudioFrameCount(Double(buffer.frameLength) * ratio + 1), 1)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: analysisFormat, frameCapacity: capacity) else {
            return nil
        }

        var error: NSError?
        let inputState = OSAllocatedUnfairLock(initialState: false)
        converter.convert(to: outBuffer, error: &error) { _, outStatus in
            let shouldProvideInput = inputState.withLock { didProvideInput in
                if didProvideInput {
                    return false
                }
                didProvideInput = true
                return true
            }
            if !shouldProvideInput {
                outStatus.pointee = .noDataNow
                return nil
            }
            outStatus.pointee = .haveData
            return buffer
        }

        if let error {
            print("Error: Audio conversion failed for speech detection: \(error)")
            return nil
        }
        return outBuffer.frameLength > 0 ? outBuffer : nil
    }

    private func logStop(reason: TranscriptState.StopReason, idleDuration: TimeInterval) {
        switch reason {
        case .finalizedSegment:
            print("Did stop listening for finalized segment.")
        case .timeout:
            print("Did stop listening after \(idleDuration)s without transcription updates.")
        }
    }

    private func formatsMatch(_ lhs: AVAudioFormat?, _ rhs: AVAudioFormat?) -> Bool {
        guard let lhs, let rhs else { return false }
        return lhs.sampleRate == rhs.sampleRate &&
            lhs.channelCount == rhs.channelCount &&
            lhs.commonFormat == rhs.commonFormat &&
            lhs.isInterleaved == rhs.isInterleaved
    }

}

private actor TranscriptState {
    enum StopReason: Sendable {
        case finalizedSegment
        case timeout
    }

    enum StopDecision: Sendable {
        case none
        case stop(transcription: String?, reason: StopReason, idleDuration: TimeInterval)
    }

    struct Update: Sendable {
        let didStart: Bool
        let stopDecision: StopDecision

        static let none = Update(didStart: false, stopDecision: .none)
    }

    private var isListening = false
    private var lastSpeechTime: TimeInterval?
    private var finalizedTranscript = ""
    private var restartBoundary: TimeInterval?
    private var latestFinalizationTime: TimeInterval = 0
    private var segmentMaxRangeEnd: TimeInterval = 0
    private let finalizationEpsilon: TimeInterval = 0.001

    func reset() {
        isListening = false
        lastSpeechTime = nil
        finalizedTranscript = ""
        restartBoundary = nil
        latestFinalizationTime = 0
        segmentMaxRangeEnd = 0
    }

    func recordResult(_ result: SpeechTranscriber.Result, now: TimeInterval) -> Update {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        let rangeStart = seconds(result.range.start)
        let rangeEnd = seconds(result.range.end)
        let resultsFinalization = seconds(result.resultsFinalizationTime)
        let isFinalized = CMTimeCompare(result.resultsFinalizationTime, result.range.end) >= 0

        var didStartNow = false

        latestFinalizationTime = max(latestFinalizationTime, resultsFinalization)

        if !isListening, let restartBoundary, rangeStart < restartBoundary {
            return .none
        }

        if !isListening {
            restartBoundary = nil
            isListening = true
            finalizedTranscript = ""
            segmentMaxRangeEnd = 0
            didStartNow = true
        }
        lastSpeechTime = now
        segmentMaxRangeEnd = max(segmentMaxRangeEnd, rangeEnd)

        if isFinalized {
            if rangeEnd + finalizationEpsilon < segmentMaxRangeEnd {
                segmentMaxRangeEnd = max(rangeEnd, resultsFinalization)
            }
            mergeFinalizedText(text)
        }

        if latestFinalizationTime + finalizationEpsilon >= segmentMaxRangeEnd, segmentMaxRangeEnd > 0 {
            return Update(
                didStart: didStartNow,
                stopDecision: finishListening(reason: .finalizedSegment, idleDuration: 0, fallbackTranscription: text)
            )
        }

        return Update(didStart: didStartNow, stopDecision: .none)
    }

    func timeoutDecision(now: TimeInterval, timeout: TimeInterval) -> StopDecision {
        guard isListening, let lastSpeechTime else { return .none }

        let idleDuration = now - lastSpeechTime
        guard idleDuration > timeout else { return .none }

        return finishListening(reason: .timeout, idleDuration: idleDuration, fallbackTranscription: nil)
    }

    private func mergeFinalizedText(_ text: String) {
        guard !text.isEmpty else { return }

        if finalizedTranscript.isEmpty {
            finalizedTranscript = text
            return
        }
        if text == finalizedTranscript || finalizedTranscript.hasSuffix(text) {
            return
        }
        if text.hasPrefix(finalizedTranscript) {
            finalizedTranscript = text
            return
        }
        finalizedTranscript += " " + text
    }

    private func seconds(_ time: CMTime) -> TimeInterval {
        let value = CMTimeGetSeconds(time)
        return value.isFinite ? value : 0
    }

    private func finishListening(reason: StopReason, idleDuration: TimeInterval, fallbackTranscription: String?) -> StopDecision {
        let finalized = finalizedTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = fallbackTranscription?.trimmingCharacters(in: .whitespacesAndNewlines)
        let transcription: String? = if !finalized.isEmpty {
            finalized
        } else if let fallback, !fallback.isEmpty {
            fallback
        } else {
            nil
        }

        let boundary = max(latestFinalizationTime, segmentMaxRangeEnd)

        isListening = false
        lastSpeechTime = nil
        finalizedTranscript = ""
        segmentMaxRangeEnd = 0
        restartBoundary = boundary

        return .stop(transcription: transcription, reason: reason, idleDuration: idleDuration)
    }
}

// MARK: - AVAudioPCMBuffer Helpers

private extension AVAudioPCMBuffer {
    func deepCopy() -> AVAudioPCMBuffer? {
        guard let copied = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength) else {
            return nil
        }
        copied.frameLength = frameLength

        guard let src = floatChannelData, let dst = copied.floatChannelData else {
            return nil
        }

        let channelCount = Int(format.channelCount)
        let bytes = Int(frameLength) * MemoryLayout<Float>.size
        for channel in 0 ..< channelCount {
            memcpy(dst[channel], src[channel], bytes)
        }
        return copied
    }
}
