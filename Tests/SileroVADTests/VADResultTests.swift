import XCTest
@testable import MLXAudio

final class VADResultTests: XCTestCase {

    func testResultCreation() {
        let result = VADResult(probability: 0.75, isSpeech: true, timestamp: 1.5)
        XCTAssertEqual(result.probability, 0.75)
        XCTAssertTrue(result.isSpeech)
        XCTAssertEqual(result.timestamp, 1.5)
    }

    func testResultEquatable() {
        let result1 = VADResult(probability: 0.5, isSpeech: true, timestamp: 0.0)
        let result2 = VADResult(probability: 0.5, isSpeech: true, timestamp: 0.0)
        let result3 = VADResult(probability: 0.6, isSpeech: true, timestamp: 0.0)

        XCTAssertEqual(result1, result2)
        XCTAssertNotEqual(result1, result3)
    }

    func testResultHashable() {
        let result1 = VADResult(probability: 0.5, isSpeech: true, timestamp: 0.0)
        let result2 = VADResult(probability: 0.5, isSpeech: true, timestamp: 0.0)

        var set = Set<VADResult>()
        set.insert(result1)
        set.insert(result2)

        XCTAssertEqual(set.count, 1)
    }

    func testResultSendable() {
        let result = VADResult(probability: 0.5, isSpeech: true, timestamp: 0.0)

        Task {
            let _ = result
        }
    }
}
