import XCTest
import MLX
@testable import MLXAudio

final class VADErrorTests: XCTestCase {

    func testWeightsNotFoundError() {
        let error = VADError.weightsNotFound(path: "/path/to/weights.safetensors")
        XCTAssertTrue(error.errorDescription?.contains("/path/to/weights.safetensors") ?? false)
    }

    func testWeightsCorruptedError() {
        let error = VADError.weightsCorrupted(reason: "Invalid format")
        XCTAssertTrue(error.errorDescription?.contains("Invalid format") ?? false)
    }

    func testInvalidSampleRateError() {
        let error = VADError.invalidSampleRate(expected: 16000, got: 44100)
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("16000"))
        XCTAssertTrue(description.contains("44100"))
    }

    func testInvalidChunkSizeError() {
        let error = VADError.invalidChunkSize(expected: 512, got: 256)
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("512"))
        XCTAssertTrue(description.contains("256"))
    }

    func testInvalidAudioShapeError() {
        let error = VADError.invalidAudioShape(expected: "[512] or [1, 512]", got: [2, 512])
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("[512] or [1, 512]"))
        XCTAssertTrue(description.contains("[2, 512]"))
    }

    func testInvalidDtypeError() {
        let error = VADError.invalidDtype(expected: .float32, got: .int32)
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("float32") || description.contains("Float32"))
    }

    func testAudioOutOfRangeError() {
        let error = VADError.audioOutOfRange(min: -2.5, max: 3.0)
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("-1.0"))
        XCTAssertTrue(description.contains("1.0"))
        XCTAssertTrue(description.contains("-2.5"))
        XCTAssertTrue(description.contains("3.0"))
    }

    func testProcessingFailedError() {
        let error = VADError.processingFailed(reason: "GPU error")
        XCTAssertTrue(error.errorDescription?.contains("GPU error") ?? false)
    }

    func testStateCorruptedError() {
        let error = VADError.stateCorrupted
        XCTAssertNotNil(error.errorDescription)
    }

    func testModelInitializationFailedFromUnderlying() {
        let underlyingError = NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Test error"])
        let error = VADError.modelInitializationFailed(underlying: underlyingError)
        XCTAssertTrue(error.errorDescription?.contains("Test error") ?? false)
    }

    func testErrorIsSendable() {
        let error = VADError.stateCorrupted
        Task {
            let _ = error
        }
    }
}
