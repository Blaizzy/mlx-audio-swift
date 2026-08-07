#if canImport(FoundationEssentials)
import FoundationEssentials
#else
import Foundation
#endif

/// Write interleaved float32 samples to a 32-bit IEEE-float WAV file. Pure Swift, available on
/// all platforms (no AVFoundation dependency). For `channels > 1`, `samples` must be interleaved.
public func writeWAVFloat32(_ samples: [Float], channels: Int = 1, sampleRate: Int, to url: URL) throws {
    let bytes = WAV.encodeFloat32(samples: samples, sampleRate: sampleRate, channels: max(1, channels))
    try Data(bytes).write(to: url)
}

/// Pure-Swift WAV (RIFF/WAVE) encode/decode.
///
/// This is the platform-neutral audio container backing `MLXAudioCore`'s file I/O on
/// platforms without AVFoundation (Linux). It supports the PCM subset the audio models
/// need: integer PCM (16/24/32-bit) and IEEE float (32/64-bit), any channel count and
/// sample rate. Decoding returns a single channel of `Float` samples (channel 0), matching
/// the AVFoundation code path which reads `floatChannelData[0]`.
enum WAV {
    enum WAVError: Error, CustomStringConvertible {
        case notRIFF
        case notWAVE
        case missingFormatChunk
        case missingDataChunk
        case unsupportedFormat(audioFormat: Int, bitsPerSample: Int)
        case truncated

        var description: String {
            switch self {
            case .notRIFF: return "Not a RIFF file."
            case .notWAVE: return "Not a WAVE file."
            case .missingFormatChunk: return "WAV is missing its 'fmt ' chunk."
            case .missingDataChunk: return "WAV is missing its 'data' chunk."
            case .unsupportedFormat(let f, let b): return "Unsupported WAV format (audioFormat=\(f), bitsPerSample=\(b))."
            case .truncated: return "WAV data is truncated."
            }
        }
    }

    // MARK: - Encoding

    /// Encode mono float32 samples as a 32-bit IEEE-float WAV file.
    static func encodeFloat32(samples: [Float], sampleRate: Int, channels: Int = 1) -> [UInt8] {
        let bitsPerSample = 32
        let audioFormat: UInt16 = 3 // IEEE float
        let dataBytes = samples.count * MemoryLayout<Float>.size
        var out = [UInt8]()
        out.reserveCapacity(44 + dataBytes)
        appendHeader(
            into: &out,
            audioFormat: audioFormat,
            channels: channels,
            sampleRate: sampleRate,
            bitsPerSample: bitsPerSample,
            dataBytes: dataBytes
        )
        for sample in samples {
            appendLE(&out, sample.bitPattern)
        }
        return out
    }

    /// Header layout shared by encoders. Writes everything up to and including the
    /// `data` chunk descriptor (44 bytes for a canonical PCM/float header).
    private static func appendHeader(
        into out: inout [UInt8],
        audioFormat: UInt16,
        channels: Int,
        sampleRate: Int,
        bitsPerSample: Int,
        dataBytes: Int
    ) {
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8

        out.append(contentsOf: Array("RIFF".utf8))
        appendLE(&out, UInt32(36 + dataBytes))
        out.append(contentsOf: Array("WAVE".utf8))

        out.append(contentsOf: Array("fmt ".utf8))
        appendLE(&out, UInt32(16))
        appendLE(&out, audioFormat)
        appendLE(&out, UInt16(channels))
        appendLE(&out, UInt32(sampleRate))
        appendLE(&out, UInt32(byteRate))
        appendLE(&out, UInt16(blockAlign))
        appendLE(&out, UInt16(bitsPerSample))

        out.append(contentsOf: Array("data".utf8))
        appendLE(&out, UInt32(dataBytes))
    }

    // MARK: - Decoding

    struct Decoded {
        var sampleRate: Int
        var channels: Int
        /// Channel 0 samples as Float in [-1, 1] (integer PCM is normalised).
        var samples: [Float]
    }

    static func decode(_ bytes: [UInt8]) throws -> Decoded {
        guard bytes.count >= 12 else { throw WAVError.truncated }
        guard readFourCC(bytes, 0) == "RIFF" else { throw WAVError.notRIFF }
        guard readFourCC(bytes, 8) == "WAVE" else { throw WAVError.notWAVE }

        var offset = 12
        var audioFormat = 0
        var channels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var dataRange: Range<Int>?

        while offset + 8 <= bytes.count {
            let chunkID = readFourCC(bytes, offset)
            let chunkSize = Int(readLE32(bytes, offset + 4))
            let body = offset + 8
            guard body + chunkSize <= bytes.count else {
                // Some encoders round-trip the final chunk size loosely; clamp.
                if chunkID == "data", body <= bytes.count {
                    dataRange = body ..< bytes.count
                    break
                }
                throw WAVError.truncated
            }
            switch chunkID {
            case "fmt ":
                audioFormat = Int(readLE16(bytes, body))
                channels = Int(readLE16(bytes, body + 2))
                sampleRate = Int(readLE32(bytes, body + 4))
                bitsPerSample = Int(readLE16(bytes, body + 14))
            case "data":
                dataRange = body ..< (body + chunkSize)
            default:
                break
            }
            // Chunks are word-aligned (padded to even size).
            offset = body + chunkSize + (chunkSize & 1)
        }

        guard channels > 0, sampleRate > 0, bitsPerSample > 0 else { throw WAVError.missingFormatChunk }
        guard let dataRange else { throw WAVError.missingDataChunk }

        let samples = try decodeChannel0(
            bytes: bytes,
            range: dataRange,
            audioFormat: audioFormat,
            channels: channels,
            bitsPerSample: bitsPerSample
        )
        return Decoded(sampleRate: sampleRate, channels: channels, samples: samples)
    }

    private static func decodeChannel0(
        bytes: [UInt8],
        range: Range<Int>,
        audioFormat: Int,
        channels: Int,
        bitsPerSample: Int
    ) throws -> [Float] {
        let bytesPerSample = bitsPerSample / 8
        let frameStride = bytesPerSample * channels
        guard frameStride > 0 else { throw WAVError.unsupportedFormat(audioFormat: audioFormat, bitsPerSample: bitsPerSample) }
        let frameCount = range.count / frameStride
        var out = [Float]()
        out.reserveCapacity(frameCount)

        var p = range.lowerBound
        // audioFormat 1 = integer PCM, 3 = IEEE float. (0xFFFE WAVE_FORMAT_EXTENSIBLE is
        // treated by bit depth.)
        switch (audioFormat, bitsPerSample) {
        case (3, 32):
            for _ in 0 ..< frameCount {
                out.append(Float(bitPattern: readLE32(bytes, p)))
                p += frameStride
            }
        case (3, 64):
            for _ in 0 ..< frameCount {
                out.append(Float(Double(bitPattern: readLE64(bytes, p))))
                p += frameStride
            }
        case (1, 16), (0xFFFE, 16):
            let scale = Float(1.0 / 32768.0)
            for _ in 0 ..< frameCount {
                let raw = Int16(bitPattern: readLE16(bytes, p))
                out.append(Float(raw) * scale)
                p += frameStride
            }
        case (1, 24), (0xFFFE, 24):
            let scale = Float(1.0 / 8388608.0)
            for _ in 0 ..< frameCount {
                var v = Int32(bytes[p]) | (Int32(bytes[p + 1]) << 8) | (Int32(bytes[p + 2]) << 16)
                if v & 0x800000 != 0 { v |= Int32(bitPattern: 0xFF00_0000) } // sign-extend
                out.append(Float(v) * scale)
                p += frameStride
            }
        case (1, 32), (0xFFFE, 32):
            let scale = Float(1.0 / 2147483648.0)
            for _ in 0 ..< frameCount {
                let raw = Int32(bitPattern: readLE32(bytes, p))
                out.append(Float(raw) * scale)
                p += frameStride
            }
        default:
            throw WAVError.unsupportedFormat(audioFormat: audioFormat, bitsPerSample: bitsPerSample)
        }
        return out
    }

    // MARK: - Little-endian helpers

    static func appendLE(_ out: inout [UInt8], _ value: UInt16) {
        out.append(UInt8(value & 0xFF))
        out.append(UInt8((value >> 8) & 0xFF))
    }

    static func appendLE(_ out: inout [UInt8], _ value: UInt32) {
        out.append(UInt8(value & 0xFF))
        out.append(UInt8((value >> 8) & 0xFF))
        out.append(UInt8((value >> 16) & 0xFF))
        out.append(UInt8((value >> 24) & 0xFF))
    }

    private static func readFourCC(_ bytes: [UInt8], _ offset: Int) -> String {
        String(decoding: bytes[offset ..< offset + 4], as: UTF8.self)
    }

    private static func readLE16(_ bytes: [UInt8], _ offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    private static func readLE32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
    }

    private static func readLE64(_ bytes: [UInt8], _ offset: Int) -> UInt64 {
        var v: UInt64 = 0
        for i in 0 ..< 8 { v |= UInt64(bytes[offset + i]) << (i * 8) }
        return v
    }
}
