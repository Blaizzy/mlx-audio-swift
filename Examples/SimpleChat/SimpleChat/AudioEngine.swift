@preconcurrency import AVFoundation
import MLXAudioCore

@MainActor
protocol AudioEngineDelegate: AnyObject {
    func audioCaptureEngine(_ engine: AudioEngine, didReceive buffer: AVAudioPCMBuffer)
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

    private let inputBufferSize: AVAudioFrameCount

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
#if os(iOS)
       try input.setVoiceProcessingEnabled(true)
#endif

        let output = engine.outputNode
#if os(iOS)
       try output.setVoiceProcessingEnabled(true)
#endif

        engine.connect(streamingPlayer, to: output, format: nil)

        let tapHandler: (AVAudioPCMBuffer, AVAudioTime) -> Void = { [weak self] buf, _ in
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
