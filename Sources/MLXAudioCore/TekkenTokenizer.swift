import Foundation

private struct TekkenFile: Decodable {
    struct Config: Decodable {
        let defaultNumSpecialTokens: Int?
        let defaultVocabSize: Int?
        let pattern: String?

        enum CodingKeys: String, CodingKey {
            case defaultNumSpecialTokens = "default_num_special_tokens"
            case defaultVocabSize = "default_vocab_size"
            case pattern
        }
    }

    struct SpecialToken: Decodable {
        let rank: Int?
    }

    struct VocabEntry: Decodable {
        let tokenBytes: String

        enum CodingKeys: String, CodingKey {
            case tokenBytes = "token_bytes"
        }
    }

    let vocab: [VocabEntry]
    let config: Config?
    let specialTokens: [SpecialToken]?

    enum CodingKeys: String, CodingKey {
        case vocab
        case config
        case specialTokens = "special_tokens"
    }
}

/// Byte-level Tekken text tokenization shared by speech recognition and synthesis.
public final class TekkenTokenizer {
    private let vocab: [TekkenFile.VocabEntry]
    public let nSpecial: Int
    private let specialIds: Set<Int>

    private var bytesCache: [Int: [UInt8]] = [:]
    private let pattern: NSRegularExpression?
    private let encodingVocabSize: Int
    private var encodingRanks: [Data: Int]?

    public init(tekkenURL: URL) throws {
        let data = try Data(contentsOf: tekkenURL)
        let parsed = try JSONDecoder().decode(TekkenFile.self, from: data)
        vocab = parsed.vocab
        nSpecial = parsed.config?.defaultNumSpecialTokens ?? 1000
        encodingVocabSize = min(
            vocab.count, (parsed.config?.defaultVocabSize ?? (vocab.count + nSpecial)) - nSpecial)
        pattern = try parsed.config?.pattern.map { try NSRegularExpression(pattern: $0) }
        specialIds = Set((parsed.specialTokens ?? []).compactMap { $0.rank })
    }

    public static func fromModelDirectory(_ modelDir: URL) throws -> TekkenTokenizer {
        let tekkenURL = modelDir.appendingPathComponent("tekken.json")
        guard FileManager.default.fileExists(atPath: tekkenURL.path) else {
            throw NSError(
                domain: "TekkenTokenizer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tekken.json not found at \(modelDir.path)"]
            )
        }
        return try TekkenTokenizer(tekkenURL: tekkenURL)
    }

    /// Encode ordinary text using Tekken's byte BPE; control tokens are inserted by the processor.
    public func encode(_ text: String) -> [Int] {
        if encodingRanks == nil {
            encodingRanks = Dictionary(
                uniqueKeysWithValues: vocab.prefix(encodingVocabSize).enumerated().compactMap {
                    index, entry in
                    Data(base64Encoded: entry.tokenBytes).map { ($0, index) }
                })
        }
        let ranks = encodingRanks!
        let pieces: [String]
        if let pattern {
            pieces = pattern.matches(in: text, range: NSRange(text.startIndex..., in: text))
                .compactMap {
                    Range($0.range, in: text).map { String(text[$0]) }
                }
        } else {
            pieces = [text]
        }
        return pieces.flatMap { piece -> [Int] in
            let bytes = Data(piece.utf8)
            if let rank = ranks[bytes] { return [rank + nSpecial] }
            var parts = bytes.map { Data([$0]) }
            while parts.count > 1 {
                var best: (index: Int, rank: Int)?
                for i in 0 ..< (parts.count - 1) {
                    if let rank = ranks[parts[i] + parts[i + 1]], rank < (best?.rank ?? Int.max) {
                        best = (i, rank)
                    }
                }
                guard let best else { break }
                parts[best.index].append(parts[best.index + 1])
                parts.remove(at: best.index + 1)
            }
            return parts.map { part in
                guard let rank = ranks[part] else {
                    preconditionFailure("Tekken vocabulary is missing a byte token")
                }
                return rank + nSpecial
            }
        }
    }

    public func decode(tokenIds: [Int]) -> String {
        var out: [UInt8] = []
        out.reserveCapacity(tokenIds.count * 2)

        for tokenId in tokenIds {
            guard tokenId >= 0 else { continue }
            if tokenId < nSpecial || specialIds.contains(tokenId) {
                continue
            }
            out.append(contentsOf: tokenBytes(for: tokenId))
        }

        return String(decoding: out, as: UTF8.self)
    }

    /// The bytes `decode(tokenIds:)` emits for one token: empty for special tokens and
    /// ids outside the vocabulary.
    public func tokenBytes(for tokenId: Int) -> [UInt8] {
        if let cached = bytesCache[tokenId] {
            return cached
        }

        guard tokenId >= nSpecial, !specialIds.contains(tokenId) else {
            bytesCache[tokenId] = []
            return []
        }

        let vocabId = tokenId - nSpecial
        guard vocabId >= 0, vocabId < vocab.count else {
            bytesCache[tokenId] = []
            return []
        }

        let entry = vocab[vocabId]
        let decoded = Data(base64Encoded: entry.tokenBytes) ?? Data()
        let bytes = [UInt8](decoded)
        bytesCache[tokenId] = bytes
        return bytes
    }
}
