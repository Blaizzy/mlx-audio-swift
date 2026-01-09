import Testing
import MLX
@testable import MLXAudioSTT

struct AudioUtilsTests {
    // MARK: - padOrTrim Tests

    @Test func padOrTrimPadsShortAudioWithZeros() {
        // Audio shorter than target with zero strategy should be zero-padded
        let shortAudio = MLXArray.ones([8000])  // 0.5 seconds
        let result = AudioUtils.padOrTrim(shortAudio, length: 16000, strategy: .zero)

        #expect(result.shape == [16000])
        // First 8000 should be 1.0, rest should be 0.0
        let first = result[0].item(Float.self)
        let last = result[15999].item(Float.self)
        #expect(first == 1.0)
        #expect(last == 0.0)
    }

    @Test func padOrTrimPadsShortAudioWithRepeat() {
        // Audio shorter than target with repeat strategy should repeat content
        let shortAudio = MLXArray.ones([8000])  // 0.5 seconds
        let result = AudioUtils.padOrTrim(shortAudio, length: 16000, strategy: .repeat)

        #expect(result.shape == [16000])
        // All values should be 1.0 (repeated from original)
        let first = result[0].item(Float.self)
        let last = result[15999].item(Float.self)
        #expect(first == 1.0)
        #expect(last == 1.0)  // Should be repeated content, not zeros
    }

    @Test func padOrTrimDefaultsToRepeatStrategy() {
        // Default strategy should be repeat (to avoid Whisper silence detection)
        let shortAudio = MLXArray.ones([8000])
        let result = AudioUtils.padOrTrim(shortAudio, length: 16000)

        #expect(result.shape == [16000])
        // Last value should be 1.0 (repeated), not 0.0 (zero-padded)
        let last = result[15999].item(Float.self)
        #expect(last == 1.0)
    }

    @Test func padOrTrimTrimsLongAudio() {
        // Audio longer than target should be trimmed
        let longAudio = MLXArray.ones([32000])  // 2 seconds
        let result = AudioUtils.padOrTrim(longAudio, length: 16000)

        #expect(result.shape == [16000])
        // All values should be 1.0 (from original)
        let first = result[0].item(Float.self)
        let last = result[15999].item(Float.self)
        #expect(first == 1.0)
        #expect(last == 1.0)
    }

    @Test func padOrTrimKeepsExactLength() {
        // Audio exactly at target should be unchanged
        let exactAudio = MLXArray.ones([16000])
        let result = AudioUtils.padOrTrim(exactAudio, length: 16000)

        #expect(result.shape == [16000])
    }

    @Test func padOrTrimDefaultsTo30Seconds() {
        // Without explicit length, should use 480000 (30s at 16kHz)
        let shortAudio = MLXArray.zeros([1000])
        let result = AudioUtils.padOrTrim(shortAudio)

        #expect(result.shape == [AudioConstants.nSamples])
    }
}
