import Foundation
import MLX
import MLXAudioCodecs
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXAudioTTS

struct DiaGenerationTests {
    private func makeModel(eos: Bool = false) throws -> DiaTTSModel {
        let json = """
        {
          "model": {
            "encoder": {"n_layer": 0, "n_embd": 4, "n_hidden": 8, "n_head": 1, "head_dim": 4},
            "decoder": {"n_layer": 0, "n_embd": 4, "n_hidden": 8, "gqa_query_heads": 1,
                        "cross_query_heads": 1, "kv_heads": 1, "gqa_head_dim": 4, "cross_head_dim": 4}
          },
          "data": {"text_length": 4, "audio_length": 16, "channels": 2, "delay_pattern": [0, 2]}
        }
        """
        let config = try JSONDecoder().decode(DiaConfig.self, from: Data(json.utf8))
        let model = DiaModel(config)
        for embedding in model.decoder.embeddings {
            embedding.update(parameters: ModuleParameters.unflattened([
                "weight": MLXArray.ones(like: embedding.weight)
            ]))
        }
        if eos {
            var weights = [Float](repeating: 0, count: 4 * 2 * 1028)
            for i in 0..<4 {
                for c in 0..<2 { weights[(i * 2 + c) * 1028 + 1024] = 1 }
            }
            model.decoder.logitsDense.weight = MLXArray(weights, [4, 2, 1028])
        }
        let dac = DescriptDAC(config: DescriptDACConfig(
            encoderDim: 2, encoderRates: [2], latentDim: 4, decoderDim: 4,
            decoderRates: [2], nCodebooks: 2, codebookSize: 1024, codebookDim: 2))
        return DiaTTSModel(model: model, dac: dac, config: config)
    }

    @Test func generationIncludesBOSAndEveryPrediction() throws {
        let model = try makeModel()
        let codes = try model.generateCodes("Hi", temperature: 0, topP: 0.95, topK: 35, maxTokens: 5)
        #expect(codes.shape == [2, 6])
        #expect(codes[0].asArray(Int32.self) == [1026, 0, 0, 0, 0, 0])
        #expect(codes[1].asArray(Int32.self) == [1026, 1026, 1026, 0, 0, 0])
        let reverted = try diaRevertCodebookDelay(codes, delayPattern: [0, 2], maxT: 5, channels: 2)
        #expect(reverted.shape == [1, 2, 3])
    }

    @Test func generationFlushesEOSAccordingToEachDelay() throws {
        let model = try makeModel(eos: true)
        let codes = try model.generateCodes("Hi", temperature: 0, topP: 1, topK: 35, maxTokens: 16)
        #expect(codes[0].asArray(Int32.self) == [1026, 1024, 1025, 1025])
        #expect(codes[1].asArray(Int32.self) == [1026, 1026, 1026, 1024])
        #expect(throws: DiaError.self) {
            try diaRevertCodebookDelay(codes, delayPattern: [0, 2], maxT: 16, channels: 2)
        }
    }

    @Test func delayRemovalPreservesFirstAndLastCompleteFrames() throws {
        let codes = MLXArray([Int32(1026), 10, 11, 12, 13, 14,
                             1026, 1026, 1026, 20, 21, 22], [2, 6])
        let reverted = try diaRevertCodebookDelay(codes, delayPattern: [0, 2], maxT: 5, channels: 2)
        #expect(reverted.shape == [1, 2, 3])
        #expect(reverted.reshaped([-1]).asArray(Int32.self) == [10, 11, 12, 20, 21, 22])
    }

    @Test func EOSAndTokenLimitBothBoundAudioFrames() throws {
        let codes = MLXArray([Int32(1026), 10, 11, 12, 1024, 1025, 1025,
                             1026, 1026, 1026, 20, 21, 22, 1024], [2, 7])
        let full = try diaRevertCodebookDelay(codes, delayPattern: [0, 2], maxT: 6, channels: 2)
        #expect(full.reshaped([-1]).asArray(Int32.self) == [10, 11, 12, 20, 21, 22])
        let capped = try diaRevertCodebookDelay(codes, delayPattern: [0, 2], maxT: 4, channels: 2)
        #expect(capped.reshaped([-1]).asArray(Int32.self) == [10, 11, 20, 21])
    }

    @Test func oneFrameWithoutDelayIsRetained() throws {
        let codes = MLXArray([Int32(1026), 17], [1, 2])
        let reverted = try diaRevertCodebookDelay(codes, delayPattern: [0], maxT: 1, channels: 1)
        #expect(reverted.shape == [1, 1, 1])
        #expect(reverted.item(Int32.self) == 17)
    }

    @Test func tooFewTokensThrowInsteadOfDecodingAnEmptyTensor() async throws {
        let model = try makeModel()
        do {
            _ = try await model.generate(
                text: "Hi", voice: nil, refAudio: nil, refText: nil, language: nil,
                generationParameters: GenerateParameters(maxTokens: 1, temperature: 0))
            Issue.record("Expected noAudio for an incomplete delayed frame")
        } catch DiaError.noAudio {
            // Expected: the delayed channel has not produced an audio code yet.
        }
    }

    @Test func topPFiltersEachChannelAndIsInvariantToLogitOffsets() {
        let logits = log(MLXArray([Float(0.6), 0.3, 0.1, 0.1, 0.6, 0.3], [2, 3]))
        let narrow = diaFilterLogits(logits, topP: 0.5, topK: 35)
        let wide = diaFilterLogits(logits, topP: 0.95, topK: 35)
        #expect(isFinite(narrow).asArray(Bool.self) == [true, false, false, false, true, false])
        #expect(isFinite(wide).asArray(Bool.self) == [true, true, true, true, true, true])
        #expect(isFinite(diaFilterLogits(logits + 100, topP: 0.5, topK: 35)).asArray(Bool.self)
            == isFinite(narrow).asArray(Bool.self))
        let sampled = diaSampleChannels(logits, temperature: 1.3, topP: 0.5, topK: 35)
        #expect(sampled.asArray(Int32.self) == [0, 1])
    }

    @Test func topKAndGreedySamplingRemainSupported() {
        let logits = MLXArray([Float(1), 4, 2], [1, 3])
        #expect(isFinite(diaFilterLogits(logits, topP: 1, topK: 2)).asArray(Bool.self)
            == [false, true, true])
        #expect(diaSampleChannels(logits, temperature: 0, topP: 0.01, topK: 1).item(Int32.self) == 1)
    }
}
