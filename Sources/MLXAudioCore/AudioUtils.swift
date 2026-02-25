import AVFoundation
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
                "Failed to create AVAudioFormat."
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

    public static func writeWavFile(samples: [Float], sampleRate: Double, fileURL: URL) throws {
        let frameCount = AVAudioFrameCount(samples.count)

        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
        else {
            throw AudioUtilsErrors.cannotCreateAVAudioFormat
        }

        buffer.frameLength = frameCount
        let channelData = buffer.floatChannelData![0]
        for i in 0 ..< Int(frameCount) {
            channelData[i] = samples[i]
        }

        let audioFile = try AVAudioFile(
            forWriting: fileURL,
            settings: format.settings,
            commonFormat: format.commonFormat,
            interleaved: format.isInterleaved
        )

        try audioFile.write(from: buffer)
    }
}

/// Load audio from a file and return the sample rate and audio data.
public func loadAudioArray(from url: URL, sampleRate: Int? = nil) throws -> (Int, MLXArray) {
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
    }

    try audioFile.read(into: buffer)

    guard let floatChannelData = buffer.floatChannelData else {
        throw AudioUtils.AudioUtilsErrors.cannotReadFloatChannelData
    }

    let sourceSampleRate = Int(format.sampleRate)
    let samples = Array(UnsafeBufferPointer(start: floatChannelData[0], count: Int(buffer.frameLength)))
    let targetSampleRate = sampleRate ?? sourceSampleRate

    if targetSampleRate <= 0 {
        throw AudioUtils.AudioUtilsErrors.invalidSampleRate(targetSampleRate)
    }

    if targetSampleRate == sourceSampleRate {
        return (sourceSampleRate, MLXArray(samples))
    }

    let resampled = try resampleAudio(
        samples,
        from: sourceSampleRate,
        to: targetSampleRate
    )
    return (targetSampleRate, MLXArray(resampled))
}

/// Save audio data to a WAV file.
func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let samples = audio.asArray(Float.self)

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)

    let frameCount = AVAudioFrameCount(samples.count)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
    }

    buffer.frameLength = frameCount

    if let channelData = buffer.floatChannelData {
        for i in 0 ..< samples.count {
            channelData[0][i] = samples[i]
        }
    }

    try audioFile.write(from: buffer)
}

private final class AudioConverterInputProvider: @unchecked Sendable {
    let inputBuffer: AVAudioPCMBuffer
    var consumedInput = false

    init(inputBuffer: AVAudioPCMBuffer) {
        self.inputBuffer = inputBuffer
    }
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
    guard sourceSampleRate > 0 else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }
    guard targetSampleRate > 0 else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    guard let inputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sourceSampleRate),
        channels: 1,
        interleaved: false
    ), let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(targetSampleRate),
        channels: 1,
        interleaved: false
    ) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    let inputFrameCount = AVAudioFrameCount(samples.count)
    guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }
    inputBuffer.frameLength = inputFrameCount
    samples.withUnsafeBufferPointer { ptr in
        guard let src = ptr.baseAddress else { return }
        memcpy(
            inputBuffer.floatChannelData![0],
            src,
            samples.count * MemoryLayout<Float>.size
        )
    }

    let ratio = Double(targetSampleRate) / Double(sourceSampleRate)
    let estimatedFrames = max(1, Int(ceil(Double(samples.count) * ratio)) + 64)
    guard let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: AVAudioFrameCount(estimatedFrames)
    ) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    let provider = AudioConverterInputProvider(inputBuffer: inputBuffer)
    var conversionError: NSError?
    let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
        if provider.consumedInput {
            outStatus.pointee = .endOfStream
            return nil
        }
        provider.consumedInput = true
        outStatus.pointee = .haveData
        return provider.inputBuffer
    }

    if let conversionError {
        _ = conversionError
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    guard status == .haveData || status == .endOfStream || status == .inputRanDry else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    let outCount = Int(outputBuffer.frameLength)
    guard let outData = outputBuffer.floatChannelData?[0], outCount > 0 else {
        return []
    }
    return Array(UnsafeBufferPointer(start: outData, count: outCount))
}

/// Resample audio to a target sample rate.
public func resampleAudio(
    _ samples: MLXArray,
    from sourceSampleRate: Int,
    to targetSampleRate: Int
) throws -> MLXArray {
    let input = samples.asArray(Float.self)
    let resampled = try resampleAudio(
        input,
        from: sourceSampleRate,
        to: targetSampleRate
    )
    return MLXArray(resampled)
}

/// A streaming WAV writer that allows writing audio chunks incrementally to a file.
/// This is more memory efficient than collecting all audio in memory before writing.
public class StreamingWAVWriter {
    private let url: URL
    private let sampleRate: Double
    private var audioFile: AVAudioFile?
    private let format: AVAudioFormat
    public private(set) var framesWritten: Int = 0

    public init(url: URL, sampleRate: Double) throws {
        self.url = url
        self.sampleRate = sampleRate

        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1) else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"]
            )
        }
        self.format = format
        self.audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
    }

    /// Write a chunk of audio samples to the file.
    public func writeChunk(_ samples: [Float]) throws {
        guard let audioFile else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Audio file not initialized or already finalized"]
            )
        }

        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"]
            )
        }

        buffer.frameLength = frameCount

        if let channelData = buffer.floatChannelData {
            for i in 0 ..< samples.count {
                channelData[0][i] = samples[i]
            }
        }

        try audioFile.write(from: buffer)
        framesWritten += samples.count
    }

    /// Finalize the WAV file and return the URL.
    /// After calling this method, no more chunks can be written.
    public func finalize() -> URL {
        audioFile = nil // Close the file by releasing the reference
        return url
    }
}
