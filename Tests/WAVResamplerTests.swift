import Testing
import Foundation
@testable import MLXAudioCore

struct WAVResamplerTests {
    @Test func float32RoundTrip() throws {
        let samples: [Float] = [0, 0.5, -0.5, 1.0, -1.0, 0.123, -0.987]
        let bytes = WAV.encodeFloat32(samples: samples, sampleRate: 24000)
        let decoded = try WAV.decode(bytes)
        #expect(decoded.sampleRate == 24000)
        #expect(decoded.channels == 1)
        #expect(decoded.samples.count == samples.count)
        for (a, b) in zip(decoded.samples, samples) {
            #expect(abs(a - b) < 1e-6)
        }
    }

    @Test func float32HeaderIsCanonical() {
        let bytes = WAV.encodeFloat32(samples: [0, 0], sampleRate: 16000)
        #expect(String(decoding: bytes[0..<4], as: UTF8.self) == "RIFF")
        #expect(String(decoding: bytes[8..<12], as: UTF8.self) == "WAVE")
        // audioFormat (offset 20) == 3 (IEEE float)
        #expect(bytes[20] == 3 && bytes[21] == 0)
        // data chunk = 2 samples * 4 bytes = 8
        #expect(bytes[40] == 8 && bytes[41] == 0)
    }

    @Test func int16Decode() throws {
        // Build a minimal 16-bit PCM mono WAV with two frames: max positive and max negative.
        var b = [UInt8]()
        b.append(contentsOf: Array("RIFF".utf8)); WAV.appendLE(&b, UInt32(36 + 4)); b.append(contentsOf: Array("WAVE".utf8))
        b.append(contentsOf: Array("fmt ".utf8)); WAV.appendLE(&b, UInt32(16))
        WAV.appendLE(&b, UInt16(1)); WAV.appendLE(&b, UInt16(1)); WAV.appendLE(&b, UInt32(8000))
        WAV.appendLE(&b, UInt32(16000)); WAV.appendLE(&b, UInt16(2)); WAV.appendLE(&b, UInt16(16))
        b.append(contentsOf: Array("data".utf8)); WAV.appendLE(&b, UInt32(4))
        WAV.appendLE(&b, UInt16(bitPattern: 32767)) // ~ +1.0
        WAV.appendLE(&b, UInt16(bitPattern: -32768)) // -1.0
        let decoded = try WAV.decode(b)
        #expect(decoded.sampleRate == 8000)
        #expect(decoded.samples.count == 2)
        #expect(abs(decoded.samples[0] - 0.99997) < 1e-3)
        #expect(abs(decoded.samples[1] - (-1.0)) < 1e-6)
    }

    @Test func decodeRejectsNonRIFF() {
        #expect(throws: WAV.WAVError.self) {
            _ = try WAV.decode(Array("NOPExxxxWAVE".utf8))
        }
    }

    @Test func resampleSameRateIsIdentity() {
        let s: [Float] = [1, 2, 3, 4, 5]
        #expect(SincResampler.resample(s, from: 16000, to: 16000) == s)
    }

    @Test func resampleOutputLengthMatchesRatio() {
        let s = (0..<16000).map { Float(sin(2.0 * Double.pi * 440.0 * Double($0) / 16000.0)) }
        let up = SincResampler.resample(s, from: 16000, to: 24000)
        // Expect ~1.5x length.
        #expect(abs(Double(up.count) - Double(s.count) * 1.5) <= 2)
        let down = SincResampler.resample(s, from: 16000, to: 8000)
        #expect(abs(Double(down.count) - Double(s.count) * 0.5) <= 2)
        #expect(!up.contains { $0.isNaN })
        #expect(!down.contains { $0.isNaN })
    }

    @Test func resamplePreservesSineAmplitude() {
        // A 300 Hz tone well below both Nyquists should survive resampling with ~unit amplitude.
        let src = 16000, dst = 24000
        let tone = (0..<src).map { Float(sin(2.0 * Double.pi * 300.0 * Double($0) / Double(src))) }
        let out = SincResampler.resample(tone, from: src, to: dst)
        // Peak of the middle region (avoid filter edge transients).
        let mid = out[(out.count / 4)..<(3 * out.count / 4)]
        let peak = mid.map { abs($0) }.max() ?? 0
        #expect(peak > 0.9 && peak < 1.1)
    }
}
