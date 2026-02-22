@preconcurrency import AVFoundation
import MLX
import MLXAudioCore
import MLXAudioVAD
import os
import Speech

actor SemanticVAD {
    enum Event: Sendable {
        case started
        case stopped(transcription: String?)
    }

    private enum LifecycleState {
        case idle
        case starting
        case ready
        case failed
    }

    var hangTime: TimeInterval

    private var analyzer: SpeechAnalyzer?
    private var analyzerInputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var analysisFormat: AVAudioFormat?
    private var analyzerConverter: AVAudioConverter?
    private var transcriberTask: Task<Void, Never>?
    private var smartTurnLoadTask: Task<Void, Never>?
    private var lifecycleState: LifecycleState = .idle
    private var isListening = false
    private var lastSpeechTime: TimeInterval?
    private var hasRunSmartTurnEndpointCheck = false
    private var utteranceSamples: [Float] = []
    private var utteranceSampleRate: Int?
    private let detectionRMSThreshold: Float
    private let smartTurnRepoID: String
    private let smartTurnThreshold: Float?
    private var smartTurnModel: SmartTurnModel?

    private let transcriptState = TranscriptState()

    init(
        hangTime: TimeInterval = 0.8,
        detectionRMSThreshold: Float = 0.01,
        smartTurnRepoID: String = "mlx-community/smart-turn-v3",
        smartTurnThreshold: Float? = nil
    ) {
        self.hangTime = hangTime
        self.detectionRMSThreshold = detectionRMSThreshold
        self.smartTurnRepoID = smartTurnRepoID
        self.smartTurnThreshold = smartTurnThreshold

        Task {
            await setupSpeechPipeline()
        }
    }

    deinit {
        analyzerInputContinuation?.finish()
        transcriberTask?.cancel()
        smartTurnLoadTask?.cancel()
    }

    func process(chunk: AudioChunk) async -> Event? {
        guard lifecycleState == .ready else { return nil }
        guard let buffer = AVAudioPCMBuffer.makeFrom(chunk: chunk) else { return nil }
        return await processBuffer(buffer)
    }

    func reset() async {
        isListening = false
        lastSpeechTime = nil
        hasRunSmartTurnEndpointCheck = false
        utteranceSamples.removeAll(keepingCapacity: true)
        utteranceSampleRate = nil
        await transcriptState.reset()
    }

    private func setupSpeechPipeline() async {
        guard lifecycleState == .idle else { return }
        lifecycleState = .starting

        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale.current) else {
            print("Warning: Current locale (\(Locale.current)) is not supported for speech transcription.")
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

        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            print("Error: Speech transcriber unavailable until required assets are installed.")
            lifecycleState = .failed
            return
        }

        let inputStream = AsyncStream<AnalyzerInput> { continuation in
            analyzerInputContinuation = continuation
            continuation.onTermination = { _ in
                Task {
                    await self.clearAnalyzerContinuation()
                }
            }
        }

        let analyzer = SpeechAnalyzer(
            modules: [transcriber],
            options: .init(priority: .userInitiated, modelRetention: .lingering)
        )

        self.analyzer = analyzer
        analysisFormat = format
        analyzerConverter = nil
        lifecycleState = .ready

        Task {
            do {
                try await analyzer.start(inputSequence: inputStream)
            } catch {
                print("Error: Speech analyzer failed: \(error)")
                lifecycleState = .failed
            }
        }

        transcriberTask?.cancel()
        transcriberTask = Task {
            do {
                for try await result in transcriber.results {
                    await transcriptState.recordResult(result)
                }
            } catch {
                print("Error: Transcriber results failed: \(error)")
            }
        }

        startSmartTurnLoadIfNeeded()
    }

    private func clearAnalyzerContinuation() {
        analyzerInputContinuation = nil
    }

    private func prepareAssets(for transcriber: SpeechTranscriber) async throws {
        if let installationRequest = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await installationRequest.downloadAndInstall()
        }
    }

    private func startSmartTurnLoadIfNeeded() {
        guard smartTurnModel == nil, smartTurnLoadTask == nil else { return }

        let repoID = smartTurnRepoID
        smartTurnLoadTask = Task {
            do {
                let model = try await SmartTurnModel.fromPretrained(repoID)
                self.installSmartTurnModel(model)
            } catch {
                self.handleSmartTurnLoadFailure(error)
            }
        }
    }

    private func installSmartTurnModel(_ model: SmartTurnModel) {
        smartTurnModel = model
        smartTurnLoadTask = nil
        print("Loaded SmartTurn endpoint model from \(smartTurnRepoID).")
    }

    private func handleSmartTurnLoadFailure(_ error: Error) {
        smartTurnLoadTask = nil
        print("Warning: Failed to load SmartTurn endpoint model (\(smartTurnRepoID)): \(error)")
    }

    private func processBuffer(_ buffer: AVAudioPCMBuffer) async -> Event? {
        enqueueForSpeechTranscription(buffer)

        let now = CACurrentMediaTime()
        let isSpeechFrame = buffer.rmsLevel() >= detectionRMSThreshold

        if isSpeechFrame {
            if !isListening {
                isListening = true
                hasRunSmartTurnEndpointCheck = false
                utteranceSamples.removeAll(keepingCapacity: true)
                utteranceSampleRate = nil
                await transcriptState.beginUtterance()
                print("Did start listening (RMS VAD).")
                lastSpeechTime = now
                appendUtteranceSamples(from: buffer)
                return .started
            }
            appendUtteranceSamples(from: buffer)
            lastSpeechTime = now
            return nil
        }

        guard isListening, let lastSpeechTime else { return nil }
        appendUtteranceSamples(from: buffer)
        let finalizedTranscript = await transcriptState.currentFinalizedTranscript()

        if !hasRunSmartTurnEndpointCheck, finalizedTranscript != nil {
            hasRunSmartTurnEndpointCheck = true
            if smartTurnDetectedEndpoint() {
                let transcription = await consumeUtteranceTranscription()
                print("SmartTurn detected endpoint, short-circuiting hang time after \(now - lastSpeechTime) seconds.")
                return .stopped(transcription: transcription)
            }
        }

        let idleDuration = now - lastSpeechTime
        guard idleDuration > hangTime else { return nil }

        let transcription = await consumeUtteranceTranscription()
        print("Did stop listening after \(idleDuration)s below RMS threshold.")
        return .stopped(transcription: transcription)
    }

    private func consumeUtteranceTranscription() async -> String? {
        isListening = false
        lastSpeechTime = nil
        hasRunSmartTurnEndpointCheck = false
        utteranceSamples.removeAll(keepingCapacity: true)
        utteranceSampleRate = nil
        return await transcriptState.consumeTranscript()
    }

    private func appendUtteranceSamples(from buffer: AVAudioPCMBuffer) {
        let frameLength = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameLength > 0, channelCount > 0 else { return }
        guard let channelData = buffer.floatChannelData else { return }
        let sampleRate = Int(buffer.format.sampleRate.rounded())
        guard sampleRate > 0 else { return }

        let monoSamples: [Float]
        if channelCount == 1 {
            monoSamples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        } else if buffer.format.isInterleaved {
            let interleaved = UnsafeBufferPointer(start: channelData[0], count: frameLength * channelCount)
            var downmixed = [Float](repeating: 0, count: frameLength)
            let gain = 1.0 / Float(channelCount)
            for frameIdx in 0 ..< frameLength {
                var sum: Float = 0
                let base = frameIdx * channelCount
                for channel in 0 ..< channelCount {
                    sum += interleaved[base + channel]
                }
                downmixed[frameIdx] = sum * gain
            }
            monoSamples = downmixed
        } else {
            var downmixed = [Float](repeating: 0, count: frameLength)
            let gain = 1.0 / Float(channelCount)
            for channel in 0 ..< channelCount {
                let channelPtr = UnsafeBufferPointer(start: channelData[channel], count: frameLength)
                for frameIdx in 0 ..< frameLength {
                    downmixed[frameIdx] += channelPtr[frameIdx] * gain
                }
            }
            monoSamples = downmixed
        }

        guard !monoSamples.isEmpty else { return }
        if let utteranceSampleRate {
            if utteranceSampleRate == sampleRate {
                utteranceSamples.append(contentsOf: monoSamples)
            } else {
                do {
                    let resampled = try resampleAudio(monoSamples, from: sampleRate, to: utteranceSampleRate)
                    utteranceSamples.append(contentsOf: resampled)
                } catch {
                    print("Warning: Failed to resample utterance chunk from \(sampleRate)Hz to \(utteranceSampleRate)Hz: \(error)")
                }
            }
        } else {
            utteranceSampleRate = sampleRate
            utteranceSamples.append(contentsOf: monoSamples)
        }
    }

    private func smartTurnDetectedEndpoint() -> Bool {
        guard let smartTurnModel else { return false }
        guard !utteranceSamples.isEmpty else { return false }
        let sourceRate = utteranceSampleRate ?? 16000

        do {
            let resampledSamples: [Float] = if sourceRate == 16000 {
                utteranceSamples
            } else {
                try resampleAudio(utteranceSamples, from: sourceRate, to: 16000)
            }

            guard !resampledSamples.isEmpty else { return false }
            let audio = MLXArray(resampledSamples)
            let endpoint = try smartTurnModel.predictEndpoint(
                audio,
                sampleRate: 16000,
                threshold: smartTurnThreshold
            )
            print("SmartTurn endpoint prediction=\(endpoint.prediction) probability=\(endpoint.probability)")
            return endpoint.prediction == 1
        } catch {
            print("Warning: SmartTurn endpoint detection failed: \(error)")
            return false
        }
    }

    private func enqueueForSpeechTranscription(_ buffer: AVAudioPCMBuffer) {
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
            print("Error: Unable to create audio converter for speech transcription.")
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
            print("Error: Audio conversion failed for speech transcription: \(error)")
            return nil
        }
        return outBuffer.frameLength > 0 ? outBuffer : nil
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
    private var finalizedTranscript = ""
    private var latestHypothesis: String?

    func reset() {
        finalizedTranscript = ""
        latestHypothesis = nil
    }

    func beginUtterance() {
        finalizedTranscript = ""
        latestHypothesis = nil
    }

    func recordResult(_ result: SpeechTranscriber.Result) {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let isFinalized = CMTimeCompare(result.resultsFinalizationTime, result.range.end) >= 0

        if isFinalized {
            mergeFinalizedText(text)
            latestHypothesis = nil
        } else {
            latestHypothesis = text
        }
    }

    func consumeTranscript() -> String? {
        let transcription = currentTranscript()

        finalizedTranscript = ""
        latestHypothesis = nil
        return transcription
    }

    func currentTranscript() -> String? {
        let finalized = finalizedTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let hypothesis = latestHypothesis?.trimmingCharacters(in: .whitespacesAndNewlines)
        if !finalized.isEmpty {
            return finalized
        }
        if let hypothesis, !hypothesis.isEmpty {
            return hypothesis
        }
        return nil
    }

    func currentFinalizedTranscript() -> String? {
        let finalized = finalizedTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        return finalized.isEmpty ? nil : finalized
    }

    private func mergeFinalizedText(_ text: String) {
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
}

// MARK: - AVAudioPCMBuffer Helpers

private extension AVAudioPCMBuffer {
    static func makeFrom(chunk: AudioChunk) -> AVAudioPCMBuffer? {
        guard chunk.channelCount > 0, chunk.frameLength > 0 else { return nil }
        guard chunk.samples.count == chunk.frameLength * chunk.channelCount else { return nil }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: chunk.sampleRate,
            channels: AVAudioChannelCount(chunk.channelCount),
            interleaved: chunk.isInterleaved
        ) else {
            return nil
        }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(chunk.frameLength)
        ) else {
            return nil
        }

        buffer.frameLength = AVAudioFrameCount(chunk.frameLength)

        guard let destination = buffer.floatChannelData else { return nil }

        if chunk.isInterleaved {
            _ = chunk.samples.withUnsafeBufferPointer { source in
                memcpy(
                    destination[0],
                    source.baseAddress!,
                    chunk.samples.count * MemoryLayout<Float>.size
                )
            }
        } else {
            for channel in 0 ..< chunk.channelCount {
                let sourceOffset = channel * chunk.frameLength
                _ = chunk.samples.withUnsafeBufferPointer { source in
                    memcpy(
                        destination[channel],
                        source.baseAddress!.advanced(by: sourceOffset),
                        chunk.frameLength * MemoryLayout<Float>.size
                    )
                }
            }
        }

        return buffer
    }

    func rmsLevel() -> Float {
        guard format.commonFormat == .pcmFormatFloat32 else {
            assertionFailure("SemanticVAD only supports .pcmFormatFloat32.")
            return 0
        }

        let frameCount = Int(frameLength)
        let channelCount = Int(format.channelCount)
        guard frameCount > 0, channelCount > 0 else { return 0 }
        guard let data = floatChannelData else { return 0 }

        var sumSquares = 0.0
        var sampleCount = 0

        if format.isInterleaved {
            let totalSamples = frameCount * channelCount
            for idx in 0 ..< totalSamples {
                let sample = Double(data[0][idx])
                sumSquares += sample * sample
            }
            sampleCount = totalSamples
        } else {
            for channel in 0 ..< channelCount {
                for frame in 0 ..< frameCount {
                    let sample = Double(data[channel][frame])
                    sumSquares += sample * sample
                }
            }
            sampleCount = frameCount * channelCount
        }

        guard sampleCount > 0 else { return 0 }
        return Float(sqrt(sumSquares / Double(sampleCount)))
    }
}
