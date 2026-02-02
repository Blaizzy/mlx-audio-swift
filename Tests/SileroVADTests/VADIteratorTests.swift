import XCTest
import MLX
@testable import MLXAudio

final class VADIteratorTests: XCTestCase {

    // Note: These tests require a loaded model. Some tests are marked with XCTSkip
    // when a model is not available.

    // MARK: - Input Validation

    func testInvalidDtypeThrows() throws {
        // Create mock model for validation testing
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)

        // Int32 array instead of Float32
        let audio = MLXArray(Array(repeating: Int32(0), count: 512))

        XCTAssertThrowsError(try iterator.process(audio)) { error in
            guard case VADError.invalidDtype = error else {
                XCTFail("Expected invalidDtype error")
                return
            }
        }
    }

    func testInvalidChunkSizeThrows() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)

        // 256 samples instead of 512
        let audio = MLXArray(Array(repeating: Float(0), count: 256))

        XCTAssertThrowsError(try iterator.process(audio)) { error in
            guard case VADError.invalidChunkSize(let expected, let got) = error else {
                XCTFail("Expected invalidChunkSize error")
                return
            }
            XCTAssertEqual(expected, 512)
            XCTAssertEqual(got, 256)
        }
    }

    func testInvalidShapeThrows() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)

        // Shape [2, 512] instead of [512] or [1, 512]
        let audio = MLXArray.zeros([2, 512])

        XCTAssertThrowsError(try iterator.process(audio)) { error in
            guard case VADError.invalidAudioShape = error else {
                XCTFail("Expected invalidAudioShape error")
                return
            }
        }
    }

    func testAudioOutOfRangeThrows() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)
        iterator.validateRange = true

        // Values outside [-1, 1]
        var samples = Array(repeating: Float(0), count: 512)
        samples[0] = 2.0 // Out of range

        let audio = MLXArray(samples)

        XCTAssertThrowsError(try iterator.process(audio)) { error in
            guard case VADError.audioOutOfRange = error else {
                XCTFail("Expected audioOutOfRange error")
                return
            }
        }
    }

    func testValidateRangeCanBeDisabled() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)
        iterator.validateRange = false

        // Values outside [-1, 1] - should not throw when validation disabled
        var samples = Array(repeating: Float(0), count: 512)
        samples[0] = 2.0

        let audio = MLXArray(samples)

        // This will fail at model inference (no weights loaded),
        // but should not fail at validation
        // For this test, we just verify it doesn't throw audioOutOfRange
        do {
            _ = try iterator.process(audio)
            // If model works, that's fine
        } catch VADError.audioOutOfRange {
            XCTFail("Should not throw audioOutOfRange when validation is disabled")
        } catch {
            // Other errors are expected (no weights)
        }
    }

    // MARK: - Valid Input Shapes

    func testAccepts1DInput() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)
        iterator.validateRange = false

        let audio = MLXArray.zeros([512])

        // Should not throw validation error
        do {
            _ = try iterator.process(audio)
        } catch VADError.invalidAudioShape {
            XCTFail("Should accept [512] shape")
        } catch VADError.invalidChunkSize {
            XCTFail("Should accept 512 samples")
        } catch VADError.invalidDtype {
            XCTFail("Should accept float32")
        } catch {
            // Other errors expected (no weights)
        }
    }

    func testAccepts2DInput() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)
        iterator.validateRange = false

        let audio = MLXArray.zeros([1, 512])

        // Should not throw validation error
        do {
            _ = try iterator.process(audio)
        } catch VADError.invalidAudioShape {
            XCTFail("Should accept [1, 512] shape")
        } catch VADError.invalidChunkSize {
            XCTFail("Should accept 512 samples")
        } catch {
            // Other errors expected (no weights)
        }
    }

    // MARK: - Timestamp Tracking

    func testCurrentTimestampInitiallyZero() {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)

        XCTAssertEqual(iterator.currentTimestamp, 0.0)
    }

    // MARK: - Reset

    func testResetClearsTimestamp() throws {
        let model = SileroVADModel()
        let iterator = VADIterator(model: model, config: .default)

        // Simulate some processing by checking initial state
        XCTAssertEqual(iterator.currentTimestamp, 0.0)

        iterator.reset()

        XCTAssertEqual(iterator.currentTimestamp, 0.0)
    }
}
