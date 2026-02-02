import XCTest
@testable import MLXAudio

final class VADConfigTests: XCTestCase {

    // MARK: - Default Values

    func testDefaultConfig() {
        let config = VADConfig.default
        XCTAssertEqual(config.threshold, 0.5)
        XCTAssertEqual(config.minSpeechDurationMs, 250)
        XCTAssertEqual(config.minSilenceDurationMs, 100)
        XCTAssertEqual(config.speechPadMs, 30)
    }

    func testSensitivePreset() {
        let config = VADConfig.sensitive
        XCTAssertEqual(config.threshold, 0.35)
        XCTAssertEqual(config.minSpeechDurationMs, 200)
        XCTAssertEqual(config.minSilenceDurationMs, 150)
        XCTAssertEqual(config.speechPadMs, 50)
    }

    func testStrictPreset() {
        let config = VADConfig.strict
        XCTAssertEqual(config.threshold, 0.65)
        XCTAssertEqual(config.minSpeechDurationMs, 300)
        XCTAssertEqual(config.minSilenceDurationMs, 80)
        XCTAssertEqual(config.speechPadMs, 20)
    }

    func testConversationPreset() {
        let config = VADConfig.conversation
        XCTAssertEqual(config.threshold, 0.5)
        XCTAssertEqual(config.minSpeechDurationMs, 200)
        XCTAssertEqual(config.minSilenceDurationMs, 50)
        XCTAssertEqual(config.speechPadMs, 30)
    }

    // MARK: - Custom Config

    func testCustomConfig() {
        let config = VADConfig(
            threshold: 0.7,
            minSpeechDurationMs: 500,
            minSilenceDurationMs: 200,
            speechPadMs: 100
        )
        XCTAssertEqual(config.threshold, 0.7)
        XCTAssertEqual(config.minSpeechDurationMs, 500)
        XCTAssertEqual(config.minSilenceDurationMs, 200)
        XCTAssertEqual(config.speechPadMs, 100)
    }

    func testConfigEquatable() {
        let config1 = VADConfig.default
        let config2 = VADConfig(threshold: 0.5, minSpeechDurationMs: 250, minSilenceDurationMs: 100, speechPadMs: 30)
        XCTAssertEqual(config1, config2)
    }

    // MARK: - Audio Format Constants

    func testAudioFormatConstants() {
        XCTAssertEqual(VADAudioFormat.sampleRate, 16000)
        XCTAssertEqual(VADAudioFormat.chunkSamples, 512)
        XCTAssertEqual(VADAudioFormat.valueRange, -1.0...1.0)
    }

    func testChunkDurationCalculation() {
        let expectedDuration = Double(512) / Double(16000)
        XCTAssertEqual(VADAudioFormat.chunkDuration, expectedDuration, accuracy: 0.0001)
        XCTAssertEqual(VADAudioFormat.chunkDuration, 0.032, accuracy: 0.001)
    }
}
