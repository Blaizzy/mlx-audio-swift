import XCTest
import MLX
@testable import MLXAudio

final class VADStateTests: XCTestCase {

    // MARK: - Initial State

    func testInitialStateDefaultHiddenSize() {
        let state = VADState.initial()
        XCTAssertEqual(state.hidden.shape, [1, 128])
        XCTAssertEqual(state.cell.shape, [1, 128])
    }

    func testInitialStateCustomHiddenSize() {
        let state = VADState.initial(hiddenSize: 64)
        XCTAssertEqual(state.hidden.shape, [1, 64])
        XCTAssertEqual(state.cell.shape, [1, 64])
    }

    func testInitialStateZeros() {
        let state = VADState.initial()
        eval(state.hidden, state.cell)

        let hiddenSum = state.hidden.sum().item(Float.self)
        let cellSum = state.cell.sum().item(Float.self)

        XCTAssertEqual(hiddenSum, 0.0, accuracy: 1e-6)
        XCTAssertEqual(cellSum, 0.0, accuracy: 1e-6)
    }

    // MARK: - Reset

    func testResetPreservesHiddenSize() {
        var state = VADState.initial(hiddenSize: 64)
        state.hidden = MLXArray.ones([1, 64])
        state.cell = MLXArray.ones([1, 64])

        state.reset()

        XCTAssertEqual(state.hidden.shape, [1, 64])
        XCTAssertEqual(state.cell.shape, [1, 64])

        eval(state.hidden, state.cell)
        let hiddenSum = state.hidden.sum().item(Float.self)
        XCTAssertEqual(hiddenSum, 0.0, accuracy: 1e-6)
    }

    func testResetWithMalformedState() {
        var state = VADState(
            hidden: MLXArray.zeros([1]),
            cell: MLXArray.zeros([1])
        )

        state.reset()

        XCTAssertEqual(state.hidden.shape, [1, 128])
        XCTAssertEqual(state.cell.shape, [1, 128])
    }

    // MARK: - Custom Init

    func testCustomInit() {
        let hidden = MLXArray.ones([1, 256])
        let cell = MLXArray.zeros([1, 256])

        let state = VADState(hidden: hidden, cell: cell)

        XCTAssertEqual(state.hidden.shape, [1, 256])
        XCTAssertEqual(state.cell.shape, [1, 256])
    }
}
