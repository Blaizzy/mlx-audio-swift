import Foundation
import MLX

final class EnglishFallbackNetwork {
    static let unknownTokenId = 3

    private let configuration: BARTConfig
    private let modelWeights: [String: MLXArray]
    private let model: BARTModel
    private let graphemeToToken: [Character: Int]
    private let tokenToPhoneme: [Int: Character]

    private let british: Bool

    enum LoadError: Error, LocalizedError {
        case configNotFound(URL)
        case weightsNotFound(URL)

        var errorDescription: String? {
            switch self {
            case .configNotFound(let url): return "BART G2P config not found at \(url.path)"
            case .weightsNotFound(let url): return "BART G2P weights not found at \(url.path)"
            }
        }
    }

    init(british: Bool, directory: URL) throws {
        guard let config = EnglishFallbackNetwork.loadConfig(british: british, directory: directory) else {
            throw LoadError.configNotFound(directory)
        }
        guard let weights = EnglishFallbackNetwork.loadWeights(british: british, directory: directory) else {
            throw LoadError.weightsNotFound(directory)
        }
        configuration = config
        modelWeights = weights

        self.british = british

        self.model = BARTModel(config: configuration, weights: modelWeights)

        var graphemeDict: [Character: Int] = [:]
        for (index, grapheme) in configuration.graphemeChars.enumerated() {
            graphemeDict[grapheme] = index
        }
        self.graphemeToToken = graphemeDict

        var phonemeDict: [Int: Character] = [:]
        for (index, phoneme) in configuration.phonemeChars.enumerated() {
            phonemeDict[index] = phoneme
        }
        self.tokenToPhoneme = phonemeDict
    }

    private func graphemesToTokens(_ graphemes: String) -> [Int] {
        var tokens: [Int] = [configuration.bosTokenId]

        for char in graphemes {
            if let tokenId = graphemeToToken[char] {
                tokens.append(Int(tokenId))
            } else {
                tokens.append(EnglishFallbackNetwork.unknownTokenId)
            }
        }

        tokens.append(configuration.eosTokenId)
        return tokens
    }

    private func tokensToPhonemes(_ tokens: [Int]) -> String {
        var phonemes = ""

        for token in tokens {
            if token > EnglishFallbackNetwork.unknownTokenId {
                if let phoneme = tokenToPhoneme[Int(token)] {
                    phonemes += String(phoneme)
                }
            }
        }

        return phonemes
    }

    func callAsFunction(_ word: MToken) -> (phoneme: String, rating: Int) {
        let tokenIds = graphemesToTokens(word.text)
        let inputIds = MLXArray(tokenIds).reshaped([1, tokenIds.count])
        let generatedIds = model.generate(inputIds: inputIds)
        let outputText = tokensToPhonemes(generatedIds.asArray(Int.self))

        return (outputText, 1)
    }

    private static func loadConfig(british: Bool, directory: URL) -> BARTConfig? {
        let fileName = "\(british ? "gb" : "us")_bart_config"
        var url = directory.appendingPathComponent("\(fileName).json")
        if !FileManager.default.fileExists(atPath: url.path), british {
            url = directory.appendingPathComponent("us_bart_config.json")
        }
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(BARTConfig.self, from: data) else {
            return nil
        }
        return config
    }

    private static func loadWeights(british: Bool, directory: URL) -> [String: MLXArray]? {
        let fileName = "\(british ? "gb" : "us")_bart"
        var url = directory.appendingPathComponent("\(fileName).safetensors")
        if !FileManager.default.fileExists(atPath: url.path), british {
            url = directory.appendingPathComponent("us_bart.safetensors")
        }
        guard let weights = try? MLX.loadArrays(url: url) else {
            return nil
        }
        return weights
    }
}
