import Foundation
import MLXAudioCore

struct SenseVoiceTokenizer {
    let tokenizer: UnigramTokenizer?
    let tokenList: [String]?

    init(modelDirectory: URL) throws {
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        let modelFiles = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        let sentencePieceURL = modelFiles
            .filter { $0.pathExtension == "model" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .first
        let tokensURL = modelDirectory.appendingPathComponent("tokens.json")

        if FileManager.default.fileExists(atPath: tokenizerURL.path) {
            self.tokenizer = try UnigramTokenizer.from(tokenizerJSONURL: tokenizerURL)
        } else if let sentencePieceURL {
            self.tokenizer = try UnigramTokenizer.from(sentencePieceModelURL: sentencePieceURL)
        } else {
            self.tokenizer = nil
        }

        if FileManager.default.fileExists(atPath: tokensURL.path) {
            let data = try Data(contentsOf: tokensURL)
            self.tokenList = try JSONDecoder().decode([String].self, from: data)
        } else {
            self.tokenList = nil
        }
    }

    func decode(_ tokenIDs: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(tokenIDs)
        }
        if let tokenList {
            let pieces = tokenIDs.compactMap { tokenID -> String? in
                guard tokenList.indices.contains(tokenID) else { return nil }
                return tokenList[tokenID]
            }
            return pieces.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
        }
        return tokenIDs.map(String.init).joined(separator: " ")
    }
}
