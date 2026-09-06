// Live microphone capture and speaker playback on Linux via system GStreamer.
//
// Apple platforms use AVFoundation (see AudioPlayer). These GStreamer-backed types provide the
// equivalent live audio I/O on Linux, where AVAudioEngine is unavailable. Gated on
// `canImport(GStreamer)` so non-Linux builds are unaffected.
#if canImport(GStreamer)
import Foundation
import GStreamer
import MLX

/// Plays mono float PCM to the default speaker on Linux via GStreamer
/// (`appsrc → audioconvert → audioresample → autoaudiosink`, managed by `AudioSink`).
///
/// Requires a working audio output device and the GStreamer base/good plugins at runtime.
public final class AudioSpeaker: @unchecked Sendable {
    private let sink: AudioSink
    public let sampleRate: Int
    public let channels: Int

    public init(sampleRate: Int, channels: Int = 1, deviceIndex: Int = 0) throws {
        self.sampleRate = sampleRate
        self.channels = channels
        self.sink = try AudioSink.speaker(deviceIndex: deviceIndex)
            .withSampleRate(sampleRate)
            .withChannels(channels)
            .withFormat(.f32le)
            .build()
    }

    /// Queue a block of float samples for playback. For `channels > 1`, `samples` must be interleaved.
    public func play(_ samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        let bytes = samples.withUnsafeBytes { [UInt8]($0) }
        try await sink.play(data: bytes)
    }

    /// Queue a 1-D float32 MLX tensor for playback.
    public func play(_ samples: MLXArray) async throws {
        let f32 = samples.dtype == .float32 ? samples : samples.asType(.float32)
        guard f32.size > 0 else { return }
        try await play(f32.asArray(Float.self))
    }

    /// Play a stream of sample chunks (e.g. from streaming TTS) as they arrive.
    public func play(stream: AsyncThrowingStream<[Float], Error>) async throws {
        for try await chunk in stream {
            try await play(chunk)
        }
        sink.finish()
    }

    /// Signal end-of-stream; playback drains any queued audio.
    public func finish() {
        sink.finish()
    }

    /// Stop playback and tear down the pipeline.
    public func stop() async {
        await sink.stop()
    }
}

/// Captures mono float PCM from the default microphone on Linux via GStreamer
/// (managed by `AudioSource`). Requires a working capture device at runtime.
public final class AudioMicrophone: @unchecked Sendable {
    private let source: AudioSource
    public let sampleRate: Int
    public let channels: Int

    public init(sampleRate: Int, channels: Int = 1, deviceIndex: Int = 0) throws {
        self.sampleRate = sampleRate
        self.channels = channels
        self.source = try AudioSource.microphone(deviceIndex: deviceIndex)
            .withSampleRate(sampleRate)
            .withChannels(channels)
            .withFormat(.f32le)
            .build()
    }

    /// A stream of captured audio as mono float chunks. The stream ends when capture stops.
    public func samples() -> AsyncStream<[Float]> {
        let buffers = source.buffers()
        return AsyncStream { continuation in
            let task = Task {
                for await buffer in buffers {
                    let span = buffer.bytes
                    let stride = MemoryLayout<Float>.stride
                    let count = span.byteCount / stride
                    guard count > 0 else { continue }
                    var chunk = [Float]()
                    chunk.reserveCapacity(count)
                    for i in 0 ..< count {
                        chunk.append(span.unsafeLoadUnaligned(fromByteOffset: i * stride, as: Float.self))
                    }
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Stop capture and tear down the pipeline.
    public func stop() async {
        await source.stop()
    }
}
#endif
