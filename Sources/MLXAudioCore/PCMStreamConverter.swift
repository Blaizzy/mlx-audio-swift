@preconcurrency import AVFoundation
import os

public enum PCMStreamConverterError: Error, LocalizedError {
    case converterCreationFailed
    case conversionFailed
    case flushFailed
    case outputBufferCreationFailed

    public var errorDescription: String? {
        switch self {
        case .converterCreationFailed:
            return "Unable to create audio converter."
        case .conversionFailed:
            return "Audio conversion failed."
        case .flushFailed:
            return "Audio converter flush failed."
        case .outputBufferCreationFailed:
            return "Unable to create output audio buffer."
        }
    }
}

public final class PCMStreamConverter {
    private let outputFormat: AVAudioFormat
    private var converter: AVAudioConverter?
    private var converterInputFormat: AVAudioFormat?

    public init(outputFormat: AVAudioFormat) {
        self.outputFormat = outputFormat
    }

    public func push(_ inputBuffer: AVAudioPCMBuffer) throws -> [AVAudioPCMBuffer] {
        guard inputBuffer.frameLength > 0 else { return [] }

        var outputBuffers: [AVAudioPCMBuffer] = []
        if converter == nil || !isEquivalentAudioFormat(converterInputFormat, inputBuffer.format) {
            if let converter {
                outputBuffers.append(contentsOf: try flush(converter))
            }
            converter = try makeConverter(from: inputBuffer.format)
            converterInputFormat = inputBuffer.format
        }

        if let converter {
            outputBuffers.append(contentsOf: try convert(inputBuffer, using: converter))
        }

        return outputBuffers
    }

    public func finish() throws -> [AVAudioPCMBuffer] {
        guard let converter else { return [] }
        return try flush(converter)
    }
}

private extension PCMStreamConverter {
    func makeConverter(from inputFormat: AVAudioFormat) throws -> AVAudioConverter {
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw PCMStreamConverterError.converterCreationFailed
        }
        return converter
    }

    func convert(_ inputBuffer: AVAudioPCMBuffer, using converter: AVAudioConverter) throws -> [AVAudioPCMBuffer] {
        let ratio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio) + 512
        let didProvideInput = OSAllocatedUnfairLock(initialState: false)
        var outputBuffers: [AVAudioPCMBuffer] = []

        while true {
            let dstBuffer = try makeOutputBuffer(converter: converter, frameCapacity: max(outputCapacity, 512))
            var error: NSError?
            let status = converter.convert(to: dstBuffer, error: &error) { _, outStatus in
                let shouldProvideInput = didProvideInput.withLock { didProvide in
                    if didProvide {
                        return false
                    }
                    didProvide = true
                    return true
                }

                if shouldProvideInput {
                    outStatus.pointee = .haveData
                    return inputBuffer
                } else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
            }

            if let error { throw error }
            if dstBuffer.frameLength > 0 {
                outputBuffers.append(dstBuffer)
            }

            switch status {
            case .haveData:
                continue
            case .inputRanDry, .endOfStream:
                return outputBuffers
            case .error:
                throw PCMStreamConverterError.conversionFailed
            @unknown default:
                return outputBuffers
            }
        }
    }

    func flush(_ converter: AVAudioConverter) throws -> [AVAudioPCMBuffer] {
        var outputBuffers: [AVAudioPCMBuffer] = []

        while true {
            let dstBuffer = try makeOutputBuffer(converter: converter, frameCapacity: 4096)
            var error: NSError?
            let status = converter.convert(to: dstBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .endOfStream
                return nil
            }

            if let error { throw error }
            if dstBuffer.frameLength > 0 {
                outputBuffers.append(dstBuffer)
            }

            switch status {
            case .haveData:
                continue
            case .inputRanDry, .endOfStream:
                return outputBuffers
            case .error:
                throw PCMStreamConverterError.flushFailed
            @unknown default:
                return outputBuffers
            }
        }
    }

    func makeOutputBuffer(converter: AVAudioConverter, frameCapacity: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: frameCapacity) else {
            throw PCMStreamConverterError.outputBufferCreationFailed
        }
        return buffer
    }

    func isEquivalentAudioFormat(_ lhs: AVAudioFormat?, _ rhs: AVAudioFormat) -> Bool {
        guard let lhs else { return false }
        return lhs.commonFormat == rhs.commonFormat &&
            lhs.sampleRate == rhs.sampleRate &&
            lhs.channelCount == rhs.channelCount &&
            lhs.isInterleaved == rhs.isInterleaved
    }
}
