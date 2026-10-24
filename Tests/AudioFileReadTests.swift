import Foundation
import AVFoundation
import Testing
import MLX
@testable import MLXAudioCore

struct AudioFileReadTests {
    @Test func loadsEveryFrameFromFloatWAV() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let count = 96_000
        let values = (0..<count).map { Float(sin(Double($0) * 0.07)) * 0.1 }
        try saveAudioArray(MLXArray(values), sampleRate: 24_000, to: url)
        let (rate, loaded) = try loadAudioArray(from: url)
        #expect(rate == 24_000)
        #expect(loaded.size == count)
        let actual = loaded.asArray(Float.self)
        #expect(actual.first == values.first)
        #expect(actual.last == values.last)
    }
}
