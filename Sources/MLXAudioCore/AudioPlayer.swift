import Foundation
import AVFoundation
import Combine

@MainActor
public class AudioPlayer: NSObject, ObservableObject {
    // Published properties for UI binding
    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var isSpeaking: Bool = false
    @Published public private(set) var currentTime: TimeInterval = 0
    @Published public private(set) var duration: TimeInterval = 0
    @Published public private(set) var currentAudioURL: URL?

    /// Optional callback for speaking-state transitions during playback/streaming.
    public var onSpeakingStateChanged: ((Bool) -> Void)?
    /// Optional callback fired when a streaming source has finished and all queued audio is consumed.
    public var onDidFinishStreaming: (() -> Void)?

    private var player: AVAudioPlayer?
    private var timer: Timer?

    // Streaming playback components
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var streamingFormat: AVAudioFormat?
    private var isStreaming: Bool = false
    private var scheduledFrames: Int = 0
    private var queuedBuffers: Int = 0
    private var streamFinished: Bool = false
    private var streamingTask: Task<Void, Never>?
    private var configurationChangeObserver: Task<Void, Never>?

    public override init() {
        super.init()
    }

    @MainActor deinit {
        stop()
    }

    // MARK: - Playback Control

    public func loadAudio(from url: URL) {
        do {
            // Stop any existing playback
            stop()

            // Setup audio session for iOS
            #if os(iOS)
            AudioSessionManager.shared.setupAudioSession()
            #endif

            // Create new player
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.prepareToPlay()

            // Update state
            currentAudioURL = url
            duration = player?.duration ?? 0
            currentTime = 0

        } catch {
            print("Failed to load audio: \(error.localizedDescription)")
            currentAudioURL = nil
            duration = 0
            currentTime = 0
        }
    }

    public func play() {
        guard let player = player else { return }

        player.play()
        isPlaying = true
        setSpeaking(true)
        startTimer()
    }

    public func pause() {
        if isStreaming {
            playerNode?.pause()
        } else {
            player?.pause()
        }
        isPlaying = false
        setSpeaking(false)
        stopTimer()
    }

    public func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            if isStreaming {
                playerNode?.play()
                isPlaying = true
                setSpeaking(queuedBuffers > 0)
                startTimer()
            } else {
                play()
            }
        }
    }

    public func stop() {
        if isStreaming {
            stopStreaming()
        } else {
            player?.stop()
            isPlaying = false
            setSpeaking(false)
            stopTimer()
            currentTime = 0
        }
    }

    public func seek(to time: TimeInterval) {
        guard let player = player else { return }
        player.currentTime = max(0, min(time, duration))
        currentTime = player.currentTime
    }

    // MARK: - Streaming Playback

    /// Start streaming playback - call this before scheduling chunks
    public func startStreaming(sampleRate: Double) {
        stop()

        // Setup audio session for iOS
        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        streamingFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
        scheduledFrames = 0
        queuedBuffers = 0
        streamFinished = false
        duration = 0

        do {
            try startStreamingEngine(connectionFormat: streamingFormat)
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    /// Play a stream of PCM buffers, converting source format to the active output format.
    public func play(stream: AsyncThrowingStream<AVAudioPCMBuffer, Error>) {
        stop()

        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        do {
            try startStreamingEngine(connectionFormat: nil)
        } catch {
            print("Failed to start audio engine: \(error)")
            return
        }

        guard let engine = audioEngine else { return }
        let converter = PCMStreamConverter(outputFormat: engine.outputNode.inputFormat(forBus: 0))

        streamingTask = Task { @MainActor [weak self] in
            guard let self else { return }

            do {
                for try await inputBuffer in stream {
                    let convertedBuffers = try converter.push(inputBuffer)
                    for buffer in convertedBuffers {
                        scheduleStreamingBuffer(buffer)
                    }
                }

                let trailingBuffers = try converter.finish()
                for trailingBuffer in trailingBuffers {
                    scheduleStreamingBuffer(trailingBuffer)
                }

                streamFinished = true
                finishStreamIfDrained()
            } catch is CancellationError {
                // no-op
            } catch {
                print("Streaming playback failed: \(error)")
                stopStreaming()
            }
        }
    }

    /// Mark an open chunk stream as complete so completion can fire once queued audio drains.
    public func finishStreamingInput() {
        streamFinished = true
        finishStreamIfDrained()
    }

    /// Schedule audio samples for streaming playback
    public func scheduleAudioChunk(_ samples: [Float], withCrossfade: Bool = true) {
        guard isStreaming,
              let format = streamingFormat else { return }

        var processedSamples = samples

        // Apply fade-in to first chunk, crossfade to subsequent chunks
        if scheduledFrames == 0 {
            // Fade in the first chunk (10ms)
            let fadeInSamples = min(Int(format.sampleRate * 0.01), samples.count)
            for i in 0..<fadeInSamples {
                let factor = Float(i) / Float(fadeInSamples)
                processedSamples[i] *= factor
            }
        } else if withCrossfade {
            // Crossfade: fade in at the start (20ms)
            let crossfadeSamples = min(Int(format.sampleRate * 0.02), samples.count)
            for i in 0..<crossfadeSamples {
                let factor = Float(i) / Float(crossfadeSamples)
                processedSamples[i] *= factor
            }
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(processedSamples.count)) else {
            return
        }

        buffer.frameLength = AVAudioFrameCount(processedSamples.count)

        if let channelData = buffer.floatChannelData {
            processedSamples.withUnsafeBufferPointer { src in
                channelData[0].update(from: src.baseAddress!, count: processedSamples.count)
            }
        }

        scheduleStreamingBuffer(buffer)
    }

    /// Stop streaming and clean up
    public func stopStreaming() {
        streamingTask?.cancel()
        streamingTask = nil

        configurationChangeObserver?.cancel()
        configurationChangeObserver = nil

        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        streamingFormat = nil

        isStreaming = false
        isPlaying = false
        setSpeaking(false)

        scheduledFrames = 0
        queuedBuffers = 0
        streamFinished = false

        currentTime = 0
        duration = 0
        stopTimer()
    }

    /// Check if currently in streaming mode
    public var isStreamingMode: Bool {
        return isStreaming
    }

    // MARK: - Timer Management

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let player = self?.player else { return }
                self?.currentTime = player.currentTime
            }
        }
        timer?.tolerance = 0.05
    }

    private func startStreamingTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let node = self?.playerNode,
                      let nodeTime = node.lastRenderTime,
                      let playerTime = node.playerTime(forNodeTime: nodeTime) else { return }
                let time = Double(playerTime.sampleTime) / playerTime.sampleRate
                self?.currentTime = time
            }
        }
        timer?.tolerance = 0.05
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func startStreamingEngine(connectionFormat: AVAudioFormat?) throws {
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        guard let engine = audioEngine, let node = playerNode else { return }

        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: connectionFormat)
        engine.prepare()
        try engine.start()
        node.play()

        isStreaming = true
        isPlaying = true
        setSpeaking(false)
        scheduledFrames = 0
        queuedBuffers = 0
        streamFinished = false
        currentTime = 0
        duration = 0

        startStreamingTimer()
        observeEngineConfigurationChanges()
    }

    private func observeEngineConfigurationChanges() {
        guard configurationChangeObserver == nil else { return }
        configurationChangeObserver = Task { @MainActor [weak self] in
            guard let self else { return }

            for await _ in NotificationCenter.default.notifications(named: .AVAudioEngineConfigurationChange) {
                guard isStreaming, let engine = audioEngine else { continue }
                if !engine.isRunning {
                    do {
                        try engine.start()
                        if isPlaying {
                            playerNode?.play()
                        }
                    } catch {
                        print("Failed to restart audio engine after configuration change: \(error)")
                    }
                }
            }
        }
    }

    private func scheduleStreamingBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let node = playerNode else { return }

        queuedBuffers += 1
        scheduledFrames += Int(buffer.frameLength)
        duration = Double(scheduledFrames) / buffer.format.sampleRate

        let completion: @Sendable (AVAudioPlayerNodeCompletionCallbackType) -> Void = { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleScheduledBufferConsumed()
            }
        }
        node.scheduleBuffer(buffer, completionCallbackType: .dataConsumed, completionHandler: completion)

        if !isSpeaking {
            setSpeaking(true)
        }
    }

    private func handleScheduledBufferConsumed() {
        queuedBuffers = max(queuedBuffers - 1, 0)
        finishStreamIfDrained()
    }

    private func finishStreamIfDrained() {
        guard streamFinished, queuedBuffers == 0 else { return }
        isPlaying = false
        setSpeaking(false)
        stopTimer()
        onDidFinishStreaming?()
    }

    private func setSpeaking(_ speaking: Bool) {
        guard isSpeaking != speaking else { return }
        isSpeaking = speaking
        onSpeakingStateChanged?(speaking)
    }
}

// MARK: - AVAudioPlayerDelegate

@available(*, deprecated, renamed: "AudioPlayer", message: "Use AudioPlayer instead.")
public typealias AudioPlayerManager = AudioPlayer

extension AudioPlayer: @MainActor AVAudioPlayerDelegate {
    public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        setSpeaking(false)
        stopTimer()
        currentTime = 0
    }

    public func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("Audio decode error: \(error?.localizedDescription ?? "unknown")")
        isPlaying = false
        setSpeaking(false)
        stopTimer()
    }
}
