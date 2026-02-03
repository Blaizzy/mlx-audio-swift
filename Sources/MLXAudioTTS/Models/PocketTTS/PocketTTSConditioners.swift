import Foundation
@preconcurrency import MLX
import MLXNN
public struct TokenizedText {
    public let tokens: MLXArray
}

public final class SentencePieceTokenizer {
    public let tokenizer: UnigramTokenizer

    public init(nBins: Int, modelFolder: URL) async throws {
        let tokenizerJSON = modelFolder.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerJSON.path),
              let data = try? Data(contentsOf: tokenizerJSON) else {
            throw NSError(
                domain: "PocketTTSConditioners",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing tokenizer.json in \(modelFolder.path)"]
            )
        }
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        self.tokenizer = try UnigramTokenizer(tokenizerJSON: json)
    }

    public func callAsFunction(_ text: String) -> TokenizedText {
        let ids = tokenizer.encodeWithByteFallback(text)
        let arr = MLXArray(ids).expandedDimensions(axis: 0)
        return TokenizedText(tokens: arr)
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encodeWithByteFallback(text)
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(ids)
    }
}

public final class LUTConditioner: Module {
    public let tokenizer: SentencePieceTokenizer
    public let dim: Int
    public let outputDim: Int

    @ModuleInfo(key: "embed") public var embed: Embedding
    @ModuleInfo(key: "output_proj") public var output_proj: Linear?

    public init(nBins: Int, modelFolder: URL, dim: Int, outputDim: Int) async throws {
        self.tokenizer = try await SentencePieceTokenizer(nBins: nBins, modelFolder: modelFolder)
        self.dim = dim
        self.outputDim = outputDim
        self._embed = ModuleInfo(wrappedValue: Embedding(embeddingCount: nBins + 1, dimensions: dim))
        if dim == outputDim {
            self._output_proj = ModuleInfo(wrappedValue: nil)
        } else {
            self._output_proj = ModuleInfo(wrappedValue: Linear(dim, outputDim, bias: false))
        }
        super.init()
    }

    public func prepare(_ text: String) -> TokenizedText {
        tokenizer(text)
    }

    public func callAsFunction(_ inputs: TokenizedText) -> MLXArray {
        var embeds = embed(inputs.tokens)
        if let proj = output_proj {
            embeds = proj(embeds)
        }
        return embeds
    }
}
