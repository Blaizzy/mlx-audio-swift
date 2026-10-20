@preconcurrency import AVFoundation
import Foundation

@MainActor
final class SpeechPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var playbackID = UUID()
    private var pendingBuffers = 0

    init() {
        engine.attach(player)
    }

    func play(_ stream: AsyncThrowingStream<[Float], Error>) async throws {
        stop()
        let id = playbackID
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try session.setActive(true)
        #endif
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false
        ) else { throw TranslationError.emptyAudio }
        engine.disconnectNodeOutput(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.prepare()
        do {
            try engine.start()
            player.play()
            var receivedAudio = false
            for try await samples in stream {
                try Task.checkCancellation()
                guard !samples.isEmpty else { continue }
                while pendingBuffers >= 8 {
                    try await Task.sleep(for: .milliseconds(15))
                }
                try Task.checkCancellation()
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
                      let channel = buffer.floatChannelData?[0] else { throw TranslationError.emptyAudio }
                buffer.frameLength = buffer.frameCapacity
                samples.withUnsafeBufferPointer { source in
                    channel.update(from: source.baseAddress!, count: source.count)
                }
                receivedAudio = true
                pendingBuffers += 1
                player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                    Task { @MainActor in
                        guard let self, self.playbackID == id else { return }
                        self.pendingBuffers = max(0, self.pendingBuffers - 1)
                    }
                }
            }
            try Task.checkCancellation()
            guard receivedAudio else { throw TranslationError.emptyAudio }
            while pendingBuffers > 0 {
                try await Task.sleep(for: .milliseconds(15))
            }
            stop()
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        playbackID = UUID()
        player.stop()
        engine.stop()
        pendingBuffers = 0
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }
}
