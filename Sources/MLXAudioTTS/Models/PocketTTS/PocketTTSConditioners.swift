import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
public struct TokenizedText {
    public let tokens: MLXArray
}

public final class PocketTTSSentencePieceTokenizer {
    public let tokenizer: MLXAudioCore.SentencePieceTokenizer

    public init(nBins: Int, modelFolder: URL, tokenizerPath: String? = nil) async throws {
        let candidates = ([tokenizerPath].compactMap { $0 } + ["tokenizer.model", "tokenizer.json"])
            .filter { !$0.contains("://") }
            .map { modelFolder.appendingPathComponent($0) }
        guard let file = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw NSError(
                domain: "PocketTTSConditioners",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing tokenizer.model or tokenizer.json in \(modelFolder.path)"]
            )
        }
        let data = try Data(contentsOf: file)
        self.tokenizer = try file.pathExtension == "json"
            ? MLXAudioCore.SentencePieceTokenizer(tokenizerJSONData: data)
            : MLXAudioCore.SentencePieceTokenizer(sentencePieceModelData: data)
        guard tokenizer.vocabularySize == nBins else {
            throw NSError(domain: "PocketTTSConditioners", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Tokenizer vocabulary does not match n_bins (\(nBins))"])
        }
    }

    public func callAsFunction(_ text: String) -> TokenizedText {
        let ids = tokenizer.encodeWithByteFallback(text)
        let arr = MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0)
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
    public let tokenizer: PocketTTSSentencePieceTokenizer
    public let dim: Int
    public let outputDim: Int

    @ModuleInfo(key: "embed") public var embed: Embedding
    @ModuleInfo(key: "output_proj") public var output_proj: Linear?

    public init(nBins: Int, modelFolder: URL, dim: Int, outputDim: Int, tokenizerPath: String? = nil) async throws {
        self.tokenizer = try await PocketTTSSentencePieceTokenizer(nBins: nBins, modelFolder: modelFolder, tokenizerPath: tokenizerPath)
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

public final class SpeakerCountConditioner: Module {
    @ModuleInfo(key: "embed") public var embed: Embedding
    @ModuleInfo(key: "output_proj") public var output_proj: Linear

    public init(nBins: Int, dim: Int, outputDim: Int) {
        self._embed = ModuleInfo(wrappedValue: Embedding(embeddingCount: nBins + 1, dimensions: dim))
        self._output_proj = ModuleInfo(wrappedValue: Linear(dim, outputDim, bias: false))
        super.init()
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        output_proj(embed(tokens))
    }
}

public final class ContinuousConditioner: Module {
    public let scaleFactor: Float
    private let _freqs: MLXArray
    @ModuleInfo(key: "output_proj") public var output_proj: Linear?

    public init(dim: Int, outputDim: Int, scaleFactor: Float, maxPeriod: Float = 10000) {
        precondition(dim >= 4 && dim % 2 == 0, "Continuous conditioning requires an even dim >= 4")
        self.scaleFactor = scaleFactor
        self._freqs = MLX.exp(
            -log(maxPeriod) * MLXArray(0..<(dim / 2)).asType(.float32) / Float(dim / 2 - 1))
        self._output_proj = ModuleInfo(wrappedValue: dim == outputDim ? nil : Linear(dim, outputDim, bias: false))
        super.init()
    }

    public func callAsFunction(_ values: MLXArray) -> MLXArray {
        let phase = values.expandedDimensions(axis: -1) * scaleFactor * _freqs
        let embedding = concatenated([MLX.cos(phase), MLX.sin(phase)], axis: -1)
        guard let output_proj else { return embedding }
        return output_proj(embedding.asType(output_proj.weight.dtype))
    }
}
