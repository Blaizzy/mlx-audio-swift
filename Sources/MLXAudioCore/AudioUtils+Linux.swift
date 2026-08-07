// Linux (and any non-AVFoundation platform) implementation of MLXAudioCore's audio file
// I/O and resampling. Mirrors the public API of the AVFoundation implementation in
// `AudioUtils.swift`, backed by the pure-Swift `WAV` codec and `SincResampler`.
//
// Scope vs. the Apple path: input decoding is WAV-only here (AVFoundation decoded any
// container). Arbitrary-format decoding on Linux is a follow-up (GStreamer decodebin).
#if !canImport(AVFoundation)
// Full Foundation (not FoundationEssentials) — StreamingWAVWriter uses FileHandle, and
// loadAudioArray uses Data(contentsOf:), which live in the full module.
import Foundation
import MLX

public class AudioUtils {
    public enum AudioUtilsErrors: Error, LocalizedError {
        case cannotCreateAVAudioFormat
        case cannotCreateAudioBuffer
        case cannotReadFloatChannelData
        case invalidSampleRate(Int)
        case resamplingFailed

        public var errorDescription: String? {
            switch self {
            case .cannotCreateAVAudioFormat:
                "Failed to create audio format."
            case .cannotCreateAudioBuffer:
                "Failed to create audio buffer."
            case .cannotReadFloatChannelData:
                "Failed to access float channel data."
            case .invalidSampleRate(let sampleRate):
                "Sample rate must be positive, got \(sampleRate)."
            case .resamplingFailed:
                "Audio resampling failed."
            }
        }
    }

    private init() {}

    public static func writeWavFile(samples: [Float], sampleRate: Int, fileURL: URL) throws {
        try writeWavFile(samples: samples, sampleRate: Double(sampleRate), fileURL: fileURL)
    }

    public static func writeWavFile(samples: [Float], sampleRate: Double, fileURL: URL) throws {
        let bytes = WAV.encodeFloat32(samples: samples, sampleRate: Int(sampleRate.rounded()))
        try Data(bytes).write(to: fileURL)
    }
}

/// Load audio and return the sample rate and audio data.
///
/// WAV files are decoded by the pure-Swift `WAV` reader (no external dependencies). Any other
/// format is decoded via GStreamer when available (Linux). Without GStreamer, non-WAV input
/// surfaces a clear `WAV` decode error.
public func loadAudioArray(from url: URL, sampleRate: Int? = nil) throws -> (Int, MLXArray) {
    if let targetSampleRate = sampleRate, targetSampleRate <= 0 {
        throw AudioUtils.AudioUtilsErrors.invalidSampleRate(targetSampleRate)
    }

    if isWAVFile(url) {
        let data = try Data(contentsOf: url)
        let decoded = try WAV.decode([UInt8](data))
        let sourceSampleRate = decoded.sampleRate
        let targetSampleRate = sampleRate ?? sourceSampleRate
        if targetSampleRate == sourceSampleRate {
            return (sourceSampleRate, MLXArray(decoded.samples))
        }
        let resampled = try resampleAudio(decoded.samples, from: sourceSampleRate, to: targetSampleRate)
        return (targetSampleRate, MLXArray(resampled))
    }

    #if canImport(GStreamer)
    let (rate, samples) = try GStreamerAudioDecoder.decode(url: url, targetSampleRate: sampleRate)
    return (rate, MLXArray(samples))
    #else
    // Not WAV and no GStreamer backend: report a precise error rather than misparsing.
    let data = try Data(contentsOf: url)
    _ = try WAV.decode([UInt8](data))
    throw WAV.WAVError.notRIFF
    #endif
}

/// Peek the first 12 bytes to detect a RIFF/WAVE container without reading the whole file.
private func isWAVFile(_ url: URL) -> Bool {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
    defer { try? handle.close() }
    guard let header = try? handle.read(upToCount: 12), header.count >= 12 else { return false }
    let bytes = [UInt8](header)
    return bytes[0] == 0x52 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x46 // "RIFF"
        && bytes[8] == 0x57 && bytes[9] == 0x41 && bytes[10] == 0x56 && bytes[11] == 0x45 // "WAVE"
}

/// Save audio data to a WAV file.
func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let samples = audio.asArray(Float.self)
    try AudioUtils.writeWavFile(samples: samples, sampleRate: sampleRate, fileURL: url)
}

/// Resample audio to a target sample rate.
public func resampleAudio(
    _ samples: [Float],
    from sourceSampleRate: Int,
    to targetSampleRate: Int
) throws -> [Float] {
    if samples.isEmpty || sourceSampleRate == targetSampleRate {
        return samples
    }
    guard sourceSampleRate > 0, targetSampleRate > 0 else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }
    return SincResampler.resample(samples, from: sourceSampleRate, to: targetSampleRate)
}

/// Resample audio to a target sample rate.
public func resampleAudio(
    _ samples: MLXArray,
    from sourceSampleRate: Int,
    to targetSampleRate: Int
) throws -> MLXArray {
    let input = samples.asArray(Float.self)
    let resampled = try resampleAudio(input, from: sourceSampleRate, to: targetSampleRate)
    return MLXArray(resampled)
}

/// A streaming WAV writer that appends audio chunks incrementally to a file.
///
/// Writes a placeholder RIFF header up front and patches the RIFF/data chunk sizes in
/// `finalize()`, so memory use stays flat regardless of total audio length.
public final class StreamingWAVWriter {
    private let url: URL
    private let sampleRate: Int
    private var handle: FileHandle?
    public private(set) var framesWritten: Int = 0

    public init(url: URL, sampleRate: Double) throws {
        self.url = url
        self.sampleRate = Int(sampleRate.rounded())

        // Header with zeroed sizes; patched on finalize().
        var header = [UInt8]()
        header.append(contentsOf: Array("RIFF".utf8))
        WAV.appendLE(&header, UInt32(0))
        header.append(contentsOf: Array("WAVE".utf8))
        header.append(contentsOf: Array("fmt ".utf8))
        WAV.appendLE(&header, UInt32(16))
        WAV.appendLE(&header, UInt16(3)) // IEEE float
        WAV.appendLE(&header, UInt16(1)) // channels
        WAV.appendLE(&header, UInt32(self.sampleRate))
        WAV.appendLE(&header, UInt32(self.sampleRate * 4)) // byteRate (mono float32)
        WAV.appendLE(&header, UInt16(4)) // blockAlign
        WAV.appendLE(&header, UInt16(32)) // bitsPerSample
        header.append(contentsOf: Array("data".utf8))
        WAV.appendLE(&header, UInt32(0))

        try Data(header).write(to: url)
        self.handle = try FileHandle(forWritingTo: url)
        try self.handle?.seekToEnd()
    }

    /// Write a chunk of audio samples to the file.
    public func writeChunk(_ samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        var bytes = [UInt8]()
        bytes.reserveCapacity(samples.count * 4)
        for sample in samples { WAV.appendLE(&bytes, sample.bitPattern) }
        try writeRawBytes(bytes, sampleCount: samples.count)
    }

    /// Write a chunk of audio samples from an MLX array directly. Expects a 1D float32 tensor.
    public func writeChunk(_ samples: MLXArray) throws {
        let f32 = samples.dtype == .float32 ? samples : samples.asType(.float32)
        guard f32.size > 0 else { return }
        try writeChunk(f32.asArray(Float.self))
    }

    /// Write a chunk of float32 samples from a byte buffer.
    public func writeChunkData(_ samplesData: Data, sampleCount: Int) throws {
        guard sampleCount >= 0 else {
            throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
        }
        guard sampleCount > 0 else { return }
        let expectedBytes = sampleCount * 4
        guard samplesData.count >= expectedBytes else {
            throw AudioUtils.AudioUtilsErrors.cannotReadFloatChannelData
        }
        try writeRawBytes([UInt8](samplesData.prefix(expectedBytes)), sampleCount: sampleCount)
    }

    private func writeRawBytes(_ bytes: [UInt8], sampleCount: Int) throws {
        guard let handle else {
            throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
        }
        try handle.write(contentsOf: Data(bytes))
        framesWritten += sampleCount
    }

    /// Finalize the WAV file (patch header sizes) and return the URL.
    public func finalize() -> URL {
        guard let handle else { return url }
        let dataBytes = framesWritten * 4
        try? handle.synchronize()
        // Patch RIFF chunk size (offset 4) and data chunk size (offset 40).
        var riffSize = [UInt8](); WAV.appendLE(&riffSize, UInt32(36 + dataBytes))
        var dataSize = [UInt8](); WAV.appendLE(&dataSize, UInt32(dataBytes))
        try? handle.seek(toOffset: 4)
        try? handle.write(contentsOf: Data(riffSize))
        try? handle.seek(toOffset: 40)
        try? handle.write(contentsOf: Data(dataSize))
        try? handle.close()
        self.handle = nil
        return url
    }
}
#endif
