import Foundation

struct VoxtralRealtimeTekkenFile: Decodable {
    struct Config: Decodable {
        let defaultNumSpecialTokens: Int?

        enum CodingKeys: String, CodingKey {
            case defaultNumSpecialTokens = "default_num_special_tokens"
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

final class VoxtralRealtimeTokenizer {
    let vocab: [VoxtralRealtimeTekkenFile.VocabEntry]
    let nSpecial: Int
    let specialIds: Set<Int>

    private var bytesCache: [Int: [UInt8]] = [:]

    init(tekkenURL: URL) throws {
        let data = try Data(contentsOf: tekkenURL)
        let parsed = try JSONDecoder().decode(VoxtralRealtimeTekkenFile.self, from: data)
        vocab = parsed.vocab
        nSpecial = parsed.config?.defaultNumSpecialTokens ?? 1000
        specialIds = Set((parsed.specialTokens ?? []).compactMap { $0.rank })
    }

    static func fromModelDirectory(_ modelDir: URL) throws -> VoxtralRealtimeTokenizer {
        let tekkenURL = modelDir.appendingPathComponent("tekken.json")
        guard FileManager.default.fileExists(atPath: tekkenURL.path) else {
            throw NSError(
                domain: "VoxtralRealtimeTokenizer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tekken.json not found at \(modelDir.path)"]
            )
        }
        return try VoxtralRealtimeTokenizer(tekkenURL: tekkenURL)
    }

    func decode(tokenIds: [Int]) -> String {
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

    private func tokenBytes(for tokenId: Int) -> [UInt8] {
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
