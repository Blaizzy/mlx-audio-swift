import Foundation

public enum WhisperProgress: Sendable {
    case downloading(Float)
    case loading(Float)
    case encoding
    case decoding(Float)
}
