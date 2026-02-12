//
//  VoxtralRealtimeTokenizer.swift
//  MLXAudioSTT
//

import Foundation

public class TekkenTokenizer {
    let vocab: [[String: Any]]
    let nSpecial: Int
    let specialIds: Set<Int>
    private var bytesCache: [Int: Data] = [:]

    public init(tekkenPath: URL) throws {
        let data = try Data(contentsOf: tekkenPath)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        self.vocab = json["vocab"] as! [[String: Any]]
        let config = json["config"] as? [String: Any] ?? [:]
        self.nSpecial = (config["default_num_special_tokens"] as? Int) ?? 1000
        let specialTokens = json["special_tokens"] as? [[String: Any]] ?? []
        self.specialIds = Set(specialTokens.compactMap { $0["rank"] as? Int })
    }

    func tokenBytes(_ tokenId: Int) -> Data {
        if let cached = bytesCache[tokenId] { return cached }

        if tokenId < 0 || tokenId < nSpecial || specialIds.contains(tokenId) {
            bytesCache[tokenId] = Data()
            return Data()
        }

        let vocabId = tokenId - nSpecial
        guard vocabId >= 0 && vocabId < vocab.count,
            let b64 = vocab[vocabId]["token_bytes"] as? String,
            let decoded = Data(base64Encoded: b64)
        else {
            bytesCache[tokenId] = Data()
            return Data()
        }

        bytesCache[tokenId] = decoded
        return decoded
    }

    public func decode(_ tokenIds: [Int]) -> String {
        var out = Data()
        for tid in tokenIds {
            if tid < nSpecial || specialIds.contains(tid) { continue }
            out.append(tokenBytes(tid))
        }
        return String(data: out, encoding: .utf8)
            ?? String(bytes: out, encoding: .utf8)
            ?? String(decoding: out, as: UTF8.self)
    }

    public static func fromModelPath(_ modelPath: URL) throws -> TekkenTokenizer {
        let tekkenPath = modelPath.appendingPathComponent("tekken.json")
        guard FileManager.default.fileExists(atPath: tekkenPath.path) else {
            throw NSError(
                domain: "TekkenTokenizer", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tekken.json not found at \(modelPath.path)"])
        }
        return try TekkenTokenizer(tekkenPath: tekkenPath)
    }
}
