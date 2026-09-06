// Arbitrary-format audio decoding via system GStreamer, used on Linux where the pure-Swift
// WAV reader only covers WAV. Decodes any format GStreamer's `decodebin` supports (mp3, flac,
// ogg, m4a, …) to mono float PCM. Gated on `canImport(GStreamer)` so builds without the
// dependency (e.g. Apple platforms, which use AVFoundation) are unaffected.
#if canImport(GStreamer)
import Foundation
import Dispatch // DispatchSemaphore — Foundation doesn't reliably re-export Dispatch on Linux.
import GStreamer

enum GStreamerAudioDecoder {
    enum DecodeError: Error, CustomStringConvertible {
        case pipelineError(String)
        case noSamplesProduced(String)

        var description: String {
            switch self {
            case .pipelineError(let message): return "GStreamer decode failed: \(message)"
            case .noSamplesProduced(let path): return "GStreamer produced no audio samples for \(path)"
            }
        }
    }

    // Reference type so the decode tasks and the caller share one instance; the DispatchSemaphore
    // establishes the happens-before needed to read it safely after the tasks finish.
    private final class Collector: @unchecked Sendable {
        var samples: [Float] = []
        var sampleRate: Int = 0
        var error: Error?
    }

    /// Decode `url` to interleaved mono `Float` samples.
    /// - Parameter targetSampleRate: if non-nil, GStreamer resamples to it; otherwise the source
    ///   rate is used and returned.
    static func decode(url: URL, targetSampleRate: Int?) throws -> (sampleRate: Int, samples: [Float]) {
        var caps = "audio/x-raw,format=F32LE,layout=interleaved,channels=1"
        if let targetSampleRate {
            caps += ",rate=\(targetSampleRate)"
        }

        // gst_parse_launch treats backslashes and quotes specially inside the quoted location.
        let escapedPath = url.path
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        let description =
            "filesrc location=\"\(escapedPath)\" ! decodebin ! audioconvert ! audioresample ! \(caps) ! appsink name=sink sync=false"

        let pipeline = try Pipeline(description)
        let sink = try pipeline.audioBufferSink(named: "sink")
        let collector = Collector()
        let finished = DispatchSemaphore(value: 0)

        try pipeline.play()

        // Accumulate decoded buffers; the stream ends when the appsink reaches EOS.
        let buffersTask = Task.detached {
            for await buffer in sink.buffers() {
                if collector.sampleRate == 0, buffer.sampleRate > 0 {
                    collector.sampleRate = buffer.sampleRate
                }
                let span = buffer.bytes
                let stride = MemoryLayout<Float>.stride
                let floatCount = span.byteCount / stride
                guard floatCount > 0 else { continue }
                collector.samples.reserveCapacity(collector.samples.count + floatCount)
                for i in 0 ..< floatCount {
                    collector.samples.append(span.unsafeLoadUnaligned(fromByteOffset: i * stride, as: Float.self))
                }
            }
            finished.signal()
        }

        // A pipeline error never reaches appsink EOS, so watch the bus to avoid hanging.
        let errorTask = Task.detached {
            for await message in pipeline.bus.messages(filter: [.error]) {
                if case .error(let text, let debug) = message {
                    collector.error = DecodeError.pipelineError(debug.map { "\(text) — \($0)" } ?? text)
                    buffersTask.cancel()
                    finished.signal()
                    return
                }
            }
        }

        finished.wait()
        errorTask.cancel()
        buffersTask.cancel()
        pipeline.stop()

        if let error = collector.error {
            throw error
        }
        let rate = targetSampleRate ?? collector.sampleRate
        guard rate > 0 else {
            throw DecodeError.noSamplesProduced(url.path)
        }
        return (rate, collector.samples)
    }
}
#endif
