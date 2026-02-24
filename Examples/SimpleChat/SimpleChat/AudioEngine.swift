@preconcurrency import AVFoundation
import MLXAudioCore
import os

struct AudioChunk: Sendable {
    let samples: [Float]
    let frameLength: Int
    let sampleRate: Double
    let channelCount: Int
    let isInterleaved: Bool
}

@MainActor
protocol AudioEngineDelegate: AnyObject {
    func audioCaptureEngine(_ engine: AudioEngine, isSpeakingDidChange speaking: Bool)
}

@MainActor
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
    private let speakingGate = BooleanGate(initialValue: false)
    private let capturedChunksStream: AsyncStream<AudioChunk>
    private let capturedChunksContinuation: AsyncStream<AudioChunk>.Continuation

    private let inputBufferSize: AVAudioFrameCount

    var capturedChunks: AsyncStream<AudioChunk> {
        capturedChunksStream
    }

    init(inputBufferSize: AVAudioFrameCount) {
        self.inputBufferSize = inputBufferSize
        let stream = AsyncStream.makeStream(
            of: AudioChunk.self,
            bufferingPolicy: .bufferingNewest(8)
        )
        self.capturedChunksStream = stream.stream
        self.capturedChunksContinuation = stream.continuation
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
#if os(iOS)
        try input.setVoiceProcessingEnabled(true)
#endif

        let output = engine.outputNode
#if os(iOS)
        try output.setVoiceProcessingEnabled(true)
#endif

        engine.connect(streamingPlayer, to: output, format: nil)

        let inputMuted: @Sendable () -> Bool = { [weak input] in
            input?.isVoiceProcessingInputMuted ?? true
        }
        let speakingGate = speakingGate
        let continuation = capturedChunksContinuation
        let tapHandler: AVAudioNodeTapBlock = { buf, _ in
            guard !inputMuted() else { return }
            guard !speakingGate.get() else { return }
            guard let chunk = buf.asAudioChunk() else { return }
            continuation.yield(chunk)
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

    func speak(buffersStream: AsyncThrowingStream<AVAudioPCMBuffer, any Error>) {
        resetStreamingState()

        currentSpeakingTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await stream(buffersStream: buffersStream)
            } catch is CancellationError {
                // no-op
            } catch {
                resetStreamingState()
            }
        }
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
        speakingGate.set(false)

        currentSpeakingTask?.cancel()
        currentSpeakingTask = nil

        firstBufferQueued = false
        queuedBuffers = 0
        streamFinished = false

        print("Resetting streaming state...")
    }

    private func stream(buffersStream: AsyncThrowingStream<AVAudioPCMBuffer, any Error>) async throws {
        let converter = PCMStreamConverter(outputFormat: engine.outputNode.inputFormat(forBus: 0))

        for try await buffer in buffersStream {
            let convertedBuffers = try converter.push(buffer)
            for convertedBuffer in convertedBuffers {
                enqueue(convertedBuffer)
            }
        }

        let trailingBuffers = try converter.finish()
        for trailingBuffer in trailingBuffers {
            enqueue(trailingBuffer)
        }

        streamFinished = true
    }

    private func enqueue(_ buffer: AVAudioPCMBuffer) {
        queuedBuffers += 1

        let completion: @Sendable (AVAudioPlayerNodeCompletionCallbackType) -> Void = { [weak self] _ in
            Task { @MainActor in
                self?.handleBufferConsumed()
            }
        }
        streamingPlayer.scheduleBuffer(buffer, completionCallbackType: .dataConsumed, completionHandler: completion)

        if !firstBufferQueued {
            firstBufferQueued = true
            streamingPlayer.play()
            if !isSpeaking {
                isSpeaking = true
                speakingGate.set(true)
                delegate?.audioCaptureEngine(self, isSpeakingDidChange: true)
            }
            print("Starting to speak...")
        }
    }

    private func handleBufferConsumed() {
        queuedBuffers -= 1
        if streamFinished, queuedBuffers == 0 {
            isSpeaking = false
            speakingGate.set(false)
            delegate?.audioCaptureEngine(self, isSpeakingDidChange: false)
            print("Finished speaking.")
        }
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

private final class BooleanGate: @unchecked Sendable {
    private let lock: OSAllocatedUnfairLock<Bool>

    init(initialValue: Bool) {
        self.lock = OSAllocatedUnfairLock(initialState: initialValue)
    }

    func get() -> Bool {
        lock.withLock { $0 }
    }

    func set(_ value: Bool) {
        lock.withLock { $0 = value }
    }
}

private extension AVAudioPCMBuffer {
    func asAudioChunk() -> AudioChunk? {
        guard format.commonFormat == .pcmFormatFloat32 else {
            assertionFailure("AudioEngine input tap only supports .pcmFormatFloat32.")
            return nil
        }
        let frameCount = Int(frameLength)
        let channelCount = Int(format.channelCount)
        guard frameCount > 0, channelCount > 0 else { return nil }
        guard let source = floatChannelData else { return nil }

        let sampleCount = frameCount * channelCount
        var samples = [Float](repeating: 0, count: sampleCount)

        if format.isInterleaved {
            _ = samples.withUnsafeMutableBufferPointer { destination in
                memcpy(destination.baseAddress!, source[0], sampleCount * MemoryLayout<Float>.size)
            }
        } else {
            for channel in 0 ..< channelCount {
                let destinationOffset = channel * frameCount
                _ = samples.withUnsafeMutableBufferPointer { destination in
                    memcpy(
                        destination.baseAddress!.advanced(by: destinationOffset),
                        source[channel],
                        frameCount * MemoryLayout<Float>.size
                    )
                }
            }
        }

        return AudioChunk(
            samples: samples,
            frameLength: frameCount,
            sampleRate: format.sampleRate,
            channelCount: channelCount,
            isInterleaved: format.isInterleaved
        )
    }
}
