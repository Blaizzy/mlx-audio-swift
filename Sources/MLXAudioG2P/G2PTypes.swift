import Foundation

public struct PhonemeUnit: Sendable, Hashable {
    public let symbol: String
    public init(symbol: String) { self.symbol = symbol }
}

public protocol Phonemizing: Sendable {
    func phonemize(_ grapheme: String) throws -> [PhonemeUnit]
}

public enum G2PError: Error, Sendable, Equatable {
    case emptyInput
    case unsupportedLocale(String)
    case phonemizationFailed(token: String, reason: String)
    case alignmentFailed(reason: String)
    case resourceLoadFailed(name: String, reason: String)
}
