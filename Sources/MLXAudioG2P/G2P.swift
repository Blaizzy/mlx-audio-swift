import Foundation
import MLX
import MLXNN

public class G2P {
    private let model: T5ForConditionalGeneration
    private let tokenizer = ByT5Tokenizer()
    private let maxLength: Int

    public init(modelDirectory: URL, maxLength: Int = 50) throws {
        self.model = try WeightLoader.load(from: modelDirectory)
        self.maxLength = maxLength
    }

    public func convert(_ word: String, language: String) -> String {
        let input = tokenizer.formatInput(word, language: language)
        let inputIds = tokenizer.encode(input)
        let inputTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        let encoderOutput = model.encode(inputTensor)
        let outputIds = greedyDecode(encoderOutput: encoderOutput)
        return tokenizer.decode(outputIds)
    }

    private func greedyDecode(encoderOutput: MLXArray) -> [Int32] {
        var currentToken = MLXArray([Int32(model.config.decoderStartTokenId)])
            .expandedDimensions(axis: 0)
        var cache: [KVCache?]? = nil
        var outputIds = [Int32]()

        for _ in 0 ..< maxLength {
            let (logits, newCaches) = model.decode(
                currentToken,
                encoderOutput: encoderOutput,
                cache: cache
            )

            let nextTokenId = argMax(logits[0..., -1, 0...], axis: -1)
                .item(Int32.self)

            if nextTokenId == model.config.eosTokenId {
                break
            }

            outputIds.append(nextTokenId)
            currentToken = MLXArray([nextTokenId]).expandedDimensions(axis: 0)
            cache = newCaches.map { Optional($0) }
            eval(currentToken, newCaches)
        }

        return outputIds
    }

    public func convert(_ words: [String], language: String) -> [String] {
        words.map { convert($0, language: language) }
    }
}
