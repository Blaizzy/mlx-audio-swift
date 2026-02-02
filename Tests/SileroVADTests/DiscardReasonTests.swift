import XCTest
@testable import MLXAudio

final class DiscardReasonTests: XCTestCase {

    func testTooShortReason() {
        let reason = DiscardReason.tooShort(duration: 0.15)

        if case .tooShort(let duration) = reason {
            XCTAssertEqual(duration, 0.15, accuracy: 0.001)
        } else {
            XCTFail("Expected tooShort reason")
        }
    }

    func testDiscardReasonEquatable() {
        let reason1 = DiscardReason.tooShort(duration: 0.1)
        let reason2 = DiscardReason.tooShort(duration: 0.1)
        let reason3 = DiscardReason.tooShort(duration: 0.2)

        XCTAssertEqual(reason1, reason2)
        XCTAssertNotEqual(reason1, reason3)
    }

    func testDiscardReasonSendable() {
        let reason = DiscardReason.tooShort(duration: 0.1)
        Task {
            let _ = reason
        }
    }
}

final class SpeechEventTests: XCTestCase {

    func testSpeechStartedEvent() {
        let event = SpeechEvent.speechStarted(at: 1.5)

        if case .speechStarted(let time) = event {
            XCTAssertEqual(time, 1.5)
        } else {
            XCTFail("Expected speechStarted event")
        }
    }

    func testSpeechEndedEvent() {
        let event = SpeechEvent.speechEnded(at: 3.0, duration: 1.5)

        if case .speechEnded(let time, let duration) = event {
            XCTAssertEqual(time, 3.0)
            XCTAssertEqual(duration, 1.5)
        } else {
            XCTFail("Expected speechEnded event")
        }
    }

    func testSpeechDiscardedEvent() {
        let event = SpeechEvent.speechDiscarded(reason: .tooShort(duration: 0.05))

        if case .speechDiscarded(let reason) = event {
            if case .tooShort(let duration) = reason {
                XCTAssertEqual(duration, 0.05, accuracy: 0.001)
            } else {
                XCTFail("Expected tooShort reason")
            }
        } else {
            XCTFail("Expected speechDiscarded event")
        }
    }

    func testSpeechEventEquatable() {
        let event1 = SpeechEvent.speechStarted(at: 1.0)
        let event2 = SpeechEvent.speechStarted(at: 1.0)
        let event3 = SpeechEvent.speechStarted(at: 2.0)
        let event4 = SpeechEvent.speechEnded(at: 1.0, duration: 0.5)

        XCTAssertEqual(event1, event2)
        XCTAssertNotEqual(event1, event3)
        XCTAssertNotEqual(event1, event4)
    }

    func testSpeechEventSendable() {
        let event = SpeechEvent.speechStarted(at: 1.0)
        Task {
            let _ = event
        }
    }
}
