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
    
  init(british: Bool) {
    guard let config = EnglishFallbackNetwork.loadConfig(british: british) else {
      fatalError("Failed to load BART G2P config from bundle resources")
    }
    guard let weights = EnglishFallbackNetwork.loadWeights(british: british) else {
      fatalError("Failed to load BART G2P weights from bundle resources")
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
  
  private static func loadConfig(british: Bool) -> BARTConfig? {
    let fileName = "\(british ? "gb" : "us")_bart_config"
    
    
    guard let url = Bundle.module.url(forResource: fileName, withExtension: "json", subdirectory: "KittenTTS"),
          let data = try? Data(contentsOf: url),
          let config = try? JSONDecoder().decode(BARTConfig.self, from: data) else {
        return nil
    }
    return config
  }
  
  private static func loadWeights(british: Bool) -> [String: MLXArray]? {
    let fileName = "\(british ? "gb" : "us")_bart"
    guard let url = Bundle.module.url(forResource: fileName, withExtension: "safetensors", subdirectory: "KittenTTS"),
          let weights = try? MLX.loadArrays(url: url) else {
      return nil
    }
    return weights
  }
}
