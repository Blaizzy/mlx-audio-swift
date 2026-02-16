import Foundation
import AVFoundation
import MLX

/// Manages continuous audio capture via AVAudioEngine with a ring buffer.
/// Audio is captured at 16kHz mono, accumulated in a thread-safe buffer,
/// and can be sliced for transcription without stopping recording.
@MainActor
@Observable
class AudioRecorderManager: NSObject {
    var isRecording = false
    var recordingDuration: TimeInterval = 0
    var audioLevel: Float = 0

    private var timer: Timer?
    private var recordingStartTime: Date?

    /// The underlying capture engine (non-isolated, runs on GCD main queue)
    private let capture = AudioCaptureEngine()

    func startRecording() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true)
        #endif

        try capture.start()

        isRecording = true
        recordingStartTime = Date()
        recordingDuration = 0
        audioLevel = 0

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                if let start = self.recordingStartTime {
                    self.recordingDuration = Date().timeIntervalSince(start)
                }
                self.audioLevel = min(self.capture.currentLevel * 5, 1.0)
            }
        }
    }

    /// Get audio from `startSample` to the current end of the buffer.
    /// Returns (audio MLXArray, sampleCount at end) so caller can track position.
    func getAudio(from startSample: Int) -> (MLXArray, Int)? {
        capture.getAudio(from: startSample)
    }

    /// Total number of samples captured so far.
    var sampleCount: Int {
        capture.sampleCount
    }

    func stopRecording() -> MLXArray? {
        guard isRecording else { return nil }

        timer?.invalidate()
        timer = nil
        isRecording = false
        audioLevel = 0
        recordingStartTime = nil

        capture.stop()

        // Return all captured audio
        guard let (audio, _) = capture.getAudio(from: 0) else { return nil }
        capture.reset()
        return audio
    }

    func cancelRecording() {
        guard isRecording else { return }

        timer?.invalidate()
        timer = nil
        isRecording = false
        audioLevel = 0
        recordingStartTime = nil

        capture.stop()
        capture.reset()
    }
}

// MARK: - Audio Capture Engine

/// Non-@MainActor engine that captures audio via AVAudioEngine.installTap.
/// Thread-safe: tap callback writes on audio thread, reads happen on main thread.
final class AudioCaptureEngine: @unchecked Sendable {
    private var engine: AVAudioEngine?
    private let lock = NSLock()
    private var samples: [Float] = []
    private var _currentLevel: Float = 0

    let targetSampleRate: Double = 16000

    var currentLevel: Float {
        lock.lock()
        defer { lock.unlock() }
        return _currentLevel
    }

    var sampleCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return samples.count
    }

    func start() throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let nativeFormat = inputNode.outputFormat(forBus: 0)

        let sampleRate = targetSampleRate
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "AudioCapture", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create target audio format"])
        }

        let converter: AVAudioConverter?
        if nativeFormat.sampleRate != sampleRate || nativeFormat.channelCount != 1 {
            converter = AVAudioConverter(from: nativeFormat, to: targetFormat)
        } else {
            converter = nil
        }

        let nativeSampleRate = nativeFormat.sampleRate

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: nativeFormat) {
            [weak self] buffer, _ in
            guard let self else { return }

            let floats: [Float]
            if let converter {
                let frameCapacity = AVAudioFrameCount(
                    Double(buffer.frameLength) * sampleRate / nativeSampleRate
                )
                guard let converted = AVAudioPCMBuffer(
                    pcmFormat: targetFormat, frameCapacity: frameCapacity
                ) else { return }

                var error: NSError?
                var consumed = false
                converter.convert(to: converted, error: &error) { _, outStatus in
                    if consumed {
                        outStatus.pointee = .noDataNow
                        return nil
                    }
                    consumed = true
                    outStatus.pointee = .haveData
                    return buffer
                }
                if error != nil { return }

                floats = Array(UnsafeBufferPointer(
                    start: converted.floatChannelData![0],
                    count: Int(converted.frameLength)
                ))
            } else {
                floats = Array(UnsafeBufferPointer(
                    start: buffer.floatChannelData![0],
                    count: Int(buffer.frameLength)
                ))
            }

            let rms = sqrt(floats.reduce(0) { $0 + $1 * $1 } / max(Float(floats.count), 1))

            self.lock.lock()
            self.samples.append(contentsOf: floats)
            self._currentLevel = rms
            self.lock.unlock()
        }

        try engine.start()
        self.engine = engine
    }

    func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
    }

    /// Get audio from a sample offset to the end. Returns nil if no samples available.
    func getAudio(from startSample: Int) -> (MLXArray, Int)? {
        lock.lock()
        let count = samples.count
        guard startSample < count else {
            lock.unlock()
            return nil
        }
        let slice = Array(samples[startSample...])
        lock.unlock()

        guard !slice.isEmpty else { return nil }
        return (MLXArray(slice), count)
    }

    func reset() {
        lock.lock()
        samples.removeAll()
        _currentLevel = 0
        lock.unlock()
    }
}
