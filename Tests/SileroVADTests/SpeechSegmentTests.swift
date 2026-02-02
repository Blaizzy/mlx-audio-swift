import XCTest
@testable import MLXAudio

final class SpeechSegmentTests: XCTestCase {

    func testSegmentCreation() {
        let segment = SpeechSegment(start: 1.0, end: 2.5)
        XCTAssertEqual(segment.start, 1.0)
        XCTAssertEqual(segment.end, 2.5)
    }

    func testSegmentDuration() {
        let segment = SpeechSegment(start: 1.0, end: 3.5)
        XCTAssertEqual(segment.duration, 2.5, accuracy: 0.001)
    }

    func testSegmentZeroDuration() {
        let segment = SpeechSegment(start: 1.0, end: 1.0)
        XCTAssertEqual(segment.duration, 0.0)
    }

    func testSegmentEquatable() {
        let segment1 = SpeechSegment(start: 0.0, end: 1.0)
        let segment2 = SpeechSegment(start: 0.0, end: 1.0)
        let segment3 = SpeechSegment(start: 0.5, end: 1.0)

        XCTAssertEqual(segment1, segment2)
        XCTAssertNotEqual(segment1, segment3)
    }

    func testSegmentComparable() {
        let segment1 = SpeechSegment(start: 0.0, end: 1.0)
        let segment2 = SpeechSegment(start: 1.0, end: 2.0)
        let segment3 = SpeechSegment(start: 0.5, end: 1.5)

        XCTAssertTrue(segment1 < segment2)
        XCTAssertTrue(segment1 < segment3)
        XCTAssertTrue(segment3 < segment2)
    }

    func testSegmentSorting() {
        let segments = [
            SpeechSegment(start: 2.0, end: 3.0),
            SpeechSegment(start: 0.0, end: 1.0),
            SpeechSegment(start: 1.0, end: 2.0)
        ]

        let sorted = segments.sorted()

        XCTAssertEqual(sorted[0].start, 0.0)
        XCTAssertEqual(sorted[1].start, 1.0)
        XCTAssertEqual(sorted[2].start, 2.0)
    }

    func testSegmentHashable() {
        let segment1 = SpeechSegment(start: 0.0, end: 1.0)
        let segment2 = SpeechSegment(start: 0.0, end: 1.0)

        var set = Set<SpeechSegment>()
        set.insert(segment1)
        set.insert(segment2)

        XCTAssertEqual(set.count, 1)
    }
}
