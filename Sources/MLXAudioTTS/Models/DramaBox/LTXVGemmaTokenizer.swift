import Foundation
@preconcurrency import MLX
import Tokenizers

/// Plain-text Gemma tokenizer with left padding. DramaBox never applies a
/// chat template (`<start_of_turn>` etc.); the prompt is stripped text.
public final class LTXVGemmaTokenizer: @unchecked Sendable {
    public let padTokenId: Int
    public let eosTokenId: Int
    private let encodeText: @Sendable (String) -> [Int]

    public init(
        padTokenId: Int,
        eosTokenId: Int,
        encodeText: @escaping @Sendable (String) -> [Int]
    ) {
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.encodeText = encodeText
    }

    public convenience init(tokenizer: any Tokenizer, padTokenId: Int, eosTokenId: Int) {
        self.init(
            padTokenId: padTokenId,
            eosTokenId: eosTokenId,
            encodeText: { text in
                tokenizer.encode(text: text, addSpecialTokens: true)
            }
        )
    }

    public static func fromDirectory(_ directory: URL) async throws -> LTXVGemmaTokenizer {
        let tokenizerJSON = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerJSON.path) else {
            throw DramaBoxError.missingGemmaBackbone("tokenizer.json missing at \(directory.path)")
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let (padId, eosId) = resolveSpecialTokenIds(directory: directory, tokenizer: tokenizer)
        return LTXVGemmaTokenizer(tokenizer: tokenizer, padTokenId: padId, eosTokenId: eosId)
    }

    /// Encode one string, left-padded to `maxLength`. Truncates from the
    /// right (keeps the prefix) when the encoded sequence is longer.
    /// Returns `(inputIds, attentionMask)` with shape `[1, maxLength]`, int32.
    public func encode(_ text: String, maxLength: Int = 1024) -> (MLXArray, MLXArray) {
        let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var ids = encodeText(stripped)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        let padCount = maxLength - ids.count
        let padded = Array(repeating: padTokenId, count: padCount) + ids
        let mask = Array(repeating: 0, count: padCount) + Array(repeating: 1, count: ids.count)
        return (
            MLXArray(padded.map { Int32($0) }).reshaped([1, maxLength]),
            MLXArray(mask.map { Int32($0) }).reshaped([1, maxLength])
        )
    }

    public func encodeBatch(_ texts: [String], maxLength: Int = 1024) -> (MLXArray, MLXArray) {
        var idRows: [[Int32]] = []
        var maskRows: [[Int32]] = []
        idRows.reserveCapacity(texts.count)
        maskRows.reserveCapacity(texts.count)
        for text in texts {
            let (ids, mask) = encode(text, maxLength: maxLength)
            idRows.append(ids.asArray(Int32.self))
            maskRows.append(mask.asArray(Int32.self))
        }
        return (
            MLXArray(idRows.flatMap { $0 }).reshaped([texts.count, maxLength]),
            MLXArray(maskRows.flatMap { $0 }).reshaped([texts.count, maxLength])
        )
    }

    static func resolveSpecialTokenIds(
        directory: URL,
        tokenizer: any Tokenizer
    ) -> (pad: Int, eos: Int) {
        var padId = tokenizer.convertTokenToId("<pad>") ?? 0
        var eosId = tokenizer.convertTokenToId("<eos>") ?? tokenizer.eosTokenId ?? 1

        let specialMap = directory.appendingPathComponent("special_tokens_map.json")
        guard let data = try? Data(contentsOf: specialMap),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return (padId, eosId)
        }

        if let content = specialTokenContent(payload["pad_token"]),
           let resolved = tokenizer.convertTokenToId(content)
        {
            padId = resolved
        }
        if let content = specialTokenContent(payload["eos_token"]),
           let resolved = tokenizer.convertTokenToId(content)
        {
            eosId = resolved
        }
        return (padId, eosId)
    }

    static func specialTokenContent(_ value: Any?) -> String? {
        if let string = value as? String {
            return string
        }
        if let dict = value as? [String: Any], let content = dict["content"] as? String {
            return content
        }
        return nil
    }
}
