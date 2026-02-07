@preconcurrency import AVFoundation
import os

protocol AudioEngineDelegate: AnyObject {
    func audioCaptureEngine(_ engine: AudioEngine, didReceive buffer: AVAudioPCMBuffer)
    func audioCaptureEngine(_ engine: AudioEngine, isSpeakingDidChange speaking: Bool)
}

final class AudioEngine {
    weak var delegate: AudioEngineDelegate?

    private(set) var isSpeaking = false

    var isMicrophoneMuted: Bool {
        get { engine.inputNode.isVoiceProcessingInputMuted }
        set { engine.inputNode.isVoiceProcessingInputMuted = newValue }
    }

    private let engine = AVAudioEngine()
    private let streamingPlayer = AVAudioPlayerNode()
    private var configurationChangeObserver: Task<Void, Never>?

    private var currentSpeakingTask: Task<Void, Error>?
    private var firstBufferQueued = false
    private var queuedBuffers = 0
    private var streamFinished = false
    private var pendingData = PendingDataBuffer()

    private let inputBufferSize: AVAudioFrameCount
    private lazy var streamingInputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!

    private lazy var requiredBytesForChunk: Int = {
        let f = streamingInputFormat
        let bytesPerFrame = Int(f.streamDescription.pointee.mBytesPerFrame)
        return bytesPerFrame * Int(f.sampleRate) * 1
    }()

    init(inputBufferSize: AVAudioFrameCount) {
        self.inputBufferSize = inputBufferSize
        engine.attach(streamingPlayer)
    }

    func setup() throws {
        precondition(engine.isRunning == false, "Audio engine must be stopped before setup.")

        if configurationChangeObserver == nil {
            configurationChangeObserver = Task { [weak self] in
                guard let self else { return }

                for await _ in NotificationCenter.default.notifications(named: .AVAudioEngineConfigurationChange) {
                    engineConfigurationChanged()
                }
            }
        }

        let input = engine.inputNode
//        try input.setVoiceProcessingEnabled(true)

        let output = engine.outputNode
//        try output.setVoiceProcessingEnabled(true)

        engine.connect(streamingPlayer, to: output, format: nil)

        let tapHandler: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { [weak self] buf, _ in
            Task { @MainActor [weak self] in
                self?.processInputBuffer(buf)
            }
        }
        input.installTap(onBus: 0, bufferSize: inputBufferSize, format: nil, block: tapHandler)

        engine.prepare()
    }

    func start() throws {
        guard !engine.isRunning else { return }
        try engine.start()
        print("Started audio engine.")
    }

    func stop() {
        resetStreamingState()
        if engine.isRunning { engine.stop() }
    }

    func speak(samplesStream: AsyncThrowingStream<[Float], any Error>) {
        resetStreamingState()

        currentSpeakingTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await stream(samplesStream: samplesStream)
            } catch is CancellationError {
                // no-op
            } catch {
                resetStreamingState()
            }
        }
    }

    func speak(samples: [Float]) {
        let stream = AsyncThrowingStream<[Float], any Error> { continuation in
            if !samples.isEmpty {
                continuation.yield(samples)
            }
            continuation.finish()
        }
        speak(samplesStream: stream)
    }

    func endSpeaking() {
        resetStreamingState()
    }

    private func engineConfigurationChanged() {
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                print("Failed to start audio engine after configuration change: \(error)")
            }
        }
    }

    private func resetStreamingState() {
        streamingPlayer.stop()
        isSpeaking = false

        currentSpeakingTask?.cancel()
        currentSpeakingTask = nil

        Task { await pendingData.reset() }
        firstBufferQueued = false
        queuedBuffers = 0
        streamFinished = false

        print("Resetting streaming state...")
    }

    private func stream(samplesStream: AsyncThrowingStream<[Float], any Error>) async throws {
        let inputFormat = streamingInputFormat
        let outputFormat = engine.outputNode.inputFormat(forBus: 0)

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw NSError(domain: "AudioEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create converter"])
        }

        @inline(__always)
        func dataFromFloats(_ floats: [Float]) -> Data {
            guard !floats.isEmpty else { return Data() }
            return floats.withUnsafeBufferPointer { Data(buffer: $0) }
        }

        for try await batch in samplesStream {
            if !batch.isEmpty {
                await pendingData.append(dataFromFloats(batch))
            }
            while let chunk = await pendingData.extractChunk(ofSize: requiredBytesForChunk) {
                try convertAndQueue(chunk: chunk, inputFormat: inputFormat, converter: converter)
            }
        }

        let leftover = await pendingData.flushRemaining()
        if !leftover.isEmpty {
            try convertAndQueue(chunk: leftover, inputFormat: inputFormat, converter: converter)
        }
        try flushConverter(converter, inputFormat: inputFormat)
        streamFinished = true
    }

    private func convertAndQueue(chunk: Data, inputFormat: AVAudioFormat, converter: AVAudioConverter) throws {
        var remaining = chunk
        while let buf = try convertOnce(&remaining, inputFormat: inputFormat, converter: converter, endOfStream: false) {
            enqueue(buf)
            if remaining.isEmpty { break }
        }
    }

    private func flushConverter(_ converter: AVAudioConverter, inputFormat: AVAudioFormat) throws {
        var dummy = Data()
        while let buf = try convertOnce(&dummy, inputFormat: inputFormat, converter: converter, endOfStream: true) {
            enqueue(buf)
        }
    }

    private func convertOnce(
        _ pending: inout Data,
        inputFormat: AVAudioFormat,
        converter: AVAudioConverter,
        endOfStream: Bool
    ) throws -> AVAudioPCMBuffer? {
        let bytesPerFrame = Int(inputFormat.streamDescription.pointee.mBytesPerFrame)
        let framesInPending = pending.count / bytesPerFrame

        if framesInPending == 0, !endOfStream { return nil }

        let srcBuffer: AVAudioPCMBuffer? = {
            guard framesInPending > 0 else { return nil }
            let b = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(framesInPending))!
            b.frameLength = AVAudioFrameCount(framesInPending)
            _ = pending.withUnsafeBytes { raw in
                memcpy(b.floatChannelData![0], raw.baseAddress!, framesInPending * bytesPerFrame)
            }
            return b
        }()

        let ratio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let dstCapacity = AVAudioFrameCount(Double(framesInPending) * ratio) + 512
        let dstBuffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: max(dstCapacity, 512))!

        var error: NSError?
        let didConsumeInput = OSAllocatedUnfairLock(initialState: false)

        _ = converter.convert(to: dstBuffer, error: &error) { _, outStatus in
            if let src = srcBuffer {
                let shouldProvideInput = didConsumeInput.withLock { consumed in
                    if consumed {
                        return false
                    }
                    consumed = true
                    return true
                }
                if shouldProvideInput {
                    outStatus.pointee = .haveData
                    return src
                }
            }
            if endOfStream {
                outStatus.pointee = .endOfStream
                return nil
            } else {
                outStatus.pointee = .noDataNow
                return nil
            }
        }

        if let error { throw error }
        if didConsumeInput.withLock({ $0 }) {
            pending.removeAll()
        }

        return dstBuffer.frameLength > 0 ? dstBuffer : nil
    }

    private func enqueue(_ buffer: AVAudioPCMBuffer) {
        queuedBuffers += 1

        let completion: @Sendable (AVAudioPlayerNodeCompletionCallbackType) -> Void = { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.handleBufferConsumed() }
        }
        streamingPlayer.scheduleBuffer(buffer, completionCallbackType: .dataConsumed, completionHandler: completion)

        if !firstBufferQueued {
            firstBufferQueued = true
            streamingPlayer.play()
            if !isSpeaking {
                isSpeaking = true
                delegate?.audioCaptureEngine(self, isSpeakingDidChange: true)
            }
            print("Starting to speak...")
        }
    }

    private func handleBufferConsumed() {
        queuedBuffers -= 1
        if streamFinished, queuedBuffers == 0 {
            isSpeaking = false
            delegate?.audioCaptureEngine(self, isSpeakingDidChange: false)
            print("Finished speaking.")
        }
    }

    private func processInputBuffer(_ buffer: AVAudioPCMBuffer) {
        guard !isMicrophoneMuted else { return }
        delegate?.audioCaptureEngine(self, didReceive: buffer)
    }
}

// MARK: -

private actor PendingDataBuffer {
    private var data = Data()

    func append(_ chunk: Data) { data.append(chunk) }

    func extractChunk(ofSize size: Int) -> Data? {
        guard data.count >= size else { return nil }
        let chunk = data.prefix(size)
        data.removeFirst(size)
        return Data(chunk)
    }

    func flushRemaining() -> Data {
        defer { data.removeAll() }
        return data
    }

    func reset() { data.removeAll(keepingCapacity: true) }
}
