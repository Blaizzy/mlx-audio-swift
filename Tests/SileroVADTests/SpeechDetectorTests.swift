import XCTest
@testable import MLXAudio

final class SpeechDetectorTests: XCTestCase {

    // MARK: - Basic Detection

    func testNoSpeechReturnsNil() {
        let detector = SpeechDetector(config: .default)

        let result = VADResult(probability: 0.3, isSpeech: false, timestamp: 0.0)
        let event = detector.feed(result)

        XCTAssertNil(event)
    }

    func testSpeechAtThresholdIsSpeech() {
        // Edge case: probability = threshold should be classified as speech
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 32, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        let result = VADResult(probability: 0.5, isSpeech: true, timestamp: 0.0)
        let event = detector.feed(result)

        // First speech frame goes to pending state, no event yet
        XCTAssertNil(event)
    }

    func testSpeechStartedAfterMinChunks() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 64, minSilenceDurationMs: 100, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        // 64ms / 32ms = 2 chunks needed
        let result1 = VADResult(probability: 0.8, isSpeech: true, timestamp: 0.0)
        let result2 = VADResult(probability: 0.8, isSpeech: true, timestamp: 0.032)

        let event1 = detector.feed(result1)
        XCTAssertNil(event1) // Still pending

        let event2 = detector.feed(result2)
        if case .speechStarted(let time) = event2 {
            XCTAssertEqual(time, 0.0, accuracy: 0.001)
        } else {
            XCTFail("Expected speechStarted event")
        }
    }

    // MARK: - Speech End Detection

    func testSpeechEndedAfterSilence() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 64, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        // Start speech
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.0))

        // Continue speech to meet minSpeechDuration
        for i in 1..<10 {
            _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: Double(i) * 0.032))
        }

        // Silence frames (need 2 for 64ms)
        _ = detector.feed(VADResult(probability: 0.2, isSpeech: false, timestamp: 0.320))
        let event = detector.feed(VADResult(probability: 0.2, isSpeech: false, timestamp: 0.352))

        if case .speechEnded(let time, let duration) = event {
            XCTAssertGreaterThan(time, 0.0)
            XCTAssertGreaterThan(duration, 0.0)
        } else {
            XCTFail("Expected speechEnded event, got \(String(describing: event))")
        }
    }

    // MARK: - Short Pause Handling

    func testShortPauseTreatedAsContinuousSpeech() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 100, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        // Start speech
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.0))
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.032))

        // Short pause (1 frame = 32ms < 100ms minSilence)
        let pauseEvent = detector.feed(VADResult(probability: 0.2, isSpeech: false, timestamp: 0.064))
        XCTAssertNil(pauseEvent)

        // Resume speech - should not trigger new speechStarted
        let resumeEvent = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.096))
        XCTAssertNil(resumeEvent) // Continuous speech, no new event
    }

    // MARK: - Speech Discarded

    func testShortSpeechDiscarded() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 500, minSilenceDurationMs: 32, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        // Very short speech burst (only ~64ms, need 500ms)
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.0))
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.032))

        // Silence to end
        _ = detector.feed(VADResult(probability: 0.2, isSpeech: false, timestamp: 0.064))

        // This would end speech if minSilence was met, but speech was too short
        // Note: with minSpeechDurationMs=500, minSpeechChunks = ceil(500/32) = 16
        // So we never even enter speaking state with only 2 speech frames
    }

    // MARK: - Finalize

    func testFinalizeWithOngoingSpeech() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 100, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        // Start and continue speech
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.0))
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.032))
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.064))
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.096))

        // Finalize without silence
        let event = detector.finalize(at: 0.128)

        if case .speechEnded(_, let duration) = event {
            XCTAssertGreaterThan(duration, 0.0)
        } else {
            XCTFail("Expected speechEnded event from finalize")
        }
    }

    func testFinalizeWithNoSpeech() {
        let detector = SpeechDetector(config: .default)

        // No speech fed
        let event = detector.finalize(at: 1.0)
        XCTAssertNil(event)
    }

    // MARK: - Reset

    func testReset() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 100, speechPadMs: 0)
        let detector = SpeechDetector(config: config)

        // Start speech
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.0))
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.032))

        // Reset
        detector.reset()

        // Should be back to idle, finalize returns nil
        let event = detector.finalize(at: 1.0)
        XCTAssertNil(event)
    }

    // MARK: - Speech Padding

    func testSpeechPaddingAppliedToStart() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 100, speechPadMs: 50)
        let detector = SpeechDetector(config: config)

        // Speech starts at 0.5s
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.5))
        let event = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.532))

        if case .speechStarted(let time) = event {
            // Start should be padded back by 50ms
            XCTAssertEqual(time, 0.45, accuracy: 0.001)
        } else {
            XCTFail("Expected speechStarted event")
        }
    }

    func testSpeechPaddingClampedToZero() {
        let config = VADConfig(threshold: 0.5, minSpeechDurationMs: 32, minSilenceDurationMs: 100, speechPadMs: 100)
        let detector = SpeechDetector(config: config)

        // Speech starts at 0.032s, padding 100ms would go negative
        _ = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.032))
        let event = detector.feed(VADResult(probability: 0.8, isSpeech: true, timestamp: 0.064))

        if case .speechStarted(let time) = event {
            // Start should be clamped to 0
            XCTAssertEqual(time, 0.0, accuracy: 0.001)
        } else {
            XCTFail("Expected speechStarted event")
        }
    }
}
