import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite(.serialized)
struct BarkTests {
    @Test func publishedAndNativeStageConfigNames() throws {
        let json = #"{"input_vocab_size":32,"output_vocab_size":24,"hidden_size":16,"num_heads":2,"num_layers":3}"#
        let config = try JSONDecoder().decode(BarkStageConfig.self, from: Data(json.utf8))
        #expect(config.width == 16 && config.heads == 2 && config.layers == 3)
        let roundtrip = try JSONDecoder().decode(BarkStageConfig.self, from: JSONEncoder().encode(config))
        #expect(roundtrip.width == 16 && roundtrip.outputVocabSize == 24)
    }

    @Test func cachedCausalLogitsMatchFullSequence() {
        let config = BarkStageConfig(inputVocabSize: 32, outputVocabSize: 24,
            blockSize: 16, layers: 2, heads: 2, width: 16)
        let model = BarkGPT(config)
        let cache = model.makeCache()
        _ = model(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]), cache: cache)
        let cached = model(MLXArray([Int32(4)]).reshaped([1, 1]), cache: cache)
        let full = model(MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        #expect(abs(cached - full).max().item(Float.self) < 0.002)
        #expect(cache.allSatisfy { $0.offset == 4 })
    }

    @Test func causalPrefixCannotSeeLaterTokens() {
        let config = BarkStageConfig(inputVocabSize: 32, outputVocabSize: 24,
            blockSize: 16, layers: 2, heads: 2, width: 16)
        let model = BarkGPT(config)
        let a = model.makeCache(); let b = model.makeCache()
        _ = model(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]), cache: a)
        _ = model(MLXArray([Int32(1), 2, 9]).reshaped([1, 3]), cache: b)
        #expect(abs(a[1].state[0][0..., 0..., 0..<2, 0...]
            - b[1].state[0][0..., 0..., 0..<2, 0...]).max().item(Float.self) < 1e-6)
    }

    @Test func fineStagePredictsEveryFrameAndIgnoresFutureCodebooks() {
        let config = BarkStageConfig(inputVocabSize: 16, outputVocabSize: 16,
            blockSize: 8, layers: 1, heads: 2, width: 8)
        let model = BarkFineGPT(config)
        let tokens = MLXArray.zeros([1, 6, 8], dtype: .int32)
        let changed = MLXArray.zeros([1, 6, 8], dtype: .int32)
        changed[0..., 0..., 3...] = MLXArray(Int32(10))
        let first = model(tokens, codebook: 2)
        #expect(first.shape == [1, 6, 16])
        #expect(abs(first - model(changed, codebook: 2)).max().item(Float.self) < 1e-6)
    }

    @Test func quantizedProjectionRoundTrip() throws {
        let config = BarkStageConfig(inputVocabSize: 32, outputVocabSize: 32,
            blockSize: 8, layers: 1, heads: 2, width: 64)
        let model = BarkGPT(config)
        quantize(model: model, groupSize: 32, bits: 4)
        #expect(model.layers[0].mlp.input is QuantizedLinear)
        let restored = BarkGPT(config)
        quantize(model: restored, groupSize: 32, bits: 4)
        try restored.update(parameters: model.parameters(), verify: .all)
        let input = MLXArray([Int32(1), 2]).reshaped([1, 2])
        #expect(abs(model(input) - restored(input)).max().item(Float.self) < 1e-6)
    }

    private func tinyPipeline() throws -> BarkModel {
        let json = #"{"semantic_config":{"input_vocab_size":129600,"output_vocab_size":10048,"n_embd":4,"n_layer":1,"n_head":1},"coarse_acoustics_config":{"input_vocab_size":12096,"output_vocab_size":12096,"n_embd":4,"n_layer":1,"n_head":1},"fine_acoustics_config":{"input_vocab_size":1056,"output_vocab_size":1056,"n_embd":4,"n_layer":1,"n_head":1}}"#
        return BarkModel(try JSONDecoder().decode(BarkConfig.self, from: Data(json.utf8)))
    }

    @Test func semanticContextUsesLatestHistoryAndTruncatesText() throws {
        let model = try tinyPipeline()
        let history = try BarkHistory(semantic: MLXArray(0..<300),
            coarse: .zeros([2, 450], dtype: .int32), fine: .zeros([8, 450], dtype: .int32))
        let input = model.semanticInput(Array(0..<300), history: history)
        #expect(input.count == 513)
        #expect(Array(input.prefix(256)) == (0..<256).map { Int32($0 + 10048) })
        #expect(Array(input[256..<512]) == (44..<300).map { Int32($0) })
        #expect(input.last == 129599)
        let padded = model.semanticInput([7], history: nil)
        #expect(padded[0] == 10055 && padded[1] == 129595)
        #expect(padded[256] == 10000 && padded[511] == 10000)
    }

    @Test func coarseGenerationIncludesThePartialFinalWindow() throws {
        let model = try tinyPipeline()
        model.options.maxCoarseHistory = 60
        let output = try model.generateCoarse([Int32](repeating: 5, count: 21), history: nil,
            parameters: .init(temperature: 0))
        #expect(output.shape == [2, 31])
        #expect(output.min().item(Int.self) >= 0 && output.max().item(Int.self) < 1024)
    }

    @Test func coarseRatePreservesPartialWindows() {
        #expect(BarkGenerationOptions().maxCoarseHistory == 60)
        #expect(!BarkGenerationOptions().useFineHistory)
        #expect(BarkModel.coarseSteps(semanticCount: 1) == 2)
        #expect(BarkModel.coarseSteps(semanticCount: 21) == 62)
        #expect(BarkModel.coarseSteps(semanticCount: 499) == 1500)
    }

    @Test func historyRejectsInvalidCodebooks() {
        #expect(throws: AudioGenerationError.self) {
            try BarkHistory(semantic: MLXArray([Int32(10000)]),
                coarse: .zeros([2, 1], dtype: .int32), fine: .zeros([8, 1], dtype: .int32))
        }
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/bark-small") == "bark")
    }
}
