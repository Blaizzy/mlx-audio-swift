import Foundation

public struct ByT5Tokenizer: Sendable {
    public static let padTokenId: Int32 = 0
    public static let eosTokenId: Int32 = 1
    public static let unkTokenId: Int32 = 2
    private static let byteOffset: Int32 = 3

    public init() {}

    public func encode(_ text: String) -> [Int32] {
        var ids = [Int32]()
        ids.reserveCapacity(text.utf8.count + 1)
        for byte in text.utf8 {
            ids.append(Int32(byte) + Self.byteOffset)
        }
        ids.append(Self.eosTokenId)
        return ids
    }

    public func decode(_ ids: [Int32]) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(ids.count)
        for id in ids {
            if id == Self.eosTokenId { break }
            if id == Self.padTokenId || id == Self.unkTokenId { continue }
            let byte = id - Self.byteOffset
            if byte >= 0, byte <= 255 {
                bytes.append(UInt8(byte))
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    public func formatInput(_ word: String, language: String) -> String {
        "<\(language)>: \(word)"
    }
}
