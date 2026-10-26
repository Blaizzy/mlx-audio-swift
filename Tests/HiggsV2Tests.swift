import Foundation
import Testing
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
@testable import MLXAudioTTS

@Suite("Higgs Audio v2")
struct HiggsV2Tests {
    private func tinyConfig() -> HiggsV2Configuration {
        HiggsV2Configuration(textConfig: LlamaTTSConfiguration(
            hiddenSize: 16, hiddenLayers: 2, intermediateSize: 32, attentionHeads: 2,
            rmsNormEps: 1e-5, vocabularySize: 32, kvHeads: 1),
            audioNumCodebooks: 3, audioCodebookSize: 8)
    }

    @Test func publishedConfigurationAcceptsIntegerFlagsAndRejectsUnsupportedHeads() throws {
        let json = """
        {"text_config":{"hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,
        "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":0.00001,"vocab_size":32},
        "audio_num_codebooks":3,"audio_codebook_size":8,"audio_dual_ffn_layers":[0,1],
        "use_audio_out_self_attention":0,"audio_decoder_proj_num_layers":0,
        "quantization":{"group_size":64,"bits":8,"class_predicate_skip":["audio_codebook_embeddings"]}}
        """
        let config = try JSONDecoder().decode(HiggsV2Configuration.self, from: Data(json.utf8))
        #expect(config.audioStreamBOS == 8)
        #expect(config.audioStreamEOS == 9)
        #expect(config.quantization?.bits == 8)
        let unsupported = json.replacingOccurrences(of: "\"audio_decoder_proj_num_layers\":0", with: "\"audio_decoder_proj_num_layers\":1")
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(HiggsV2Configuration.self, from: Data(unsupported.utf8))
        }
    }

    @Test func delayRoundTripAndBoundaries() {
        let codes = MLXArray((0..<12).map(Int32.init)).reshaped([3, 4])
        let delayed = higgsV2ApplyDelay(codes, bos: 20, eos: 21)
        #expect(delayed.shape == [3, 6])
        #expect(delayed.asArray(Int32.self) == [0, 1, 2, 3, 21, 21, 20, 4, 5, 6, 7, 21, 20, 20, 8, 9, 10, 11])
        #expect(higgsV2RevertDelay(delayed).asArray(Int32.self) == codes.asArray(Int32.self))
        #expect(higgsV2RevertDelay(MLXArray.zeros([3, 1], dtype: .int32)).shape == [3, 0])
    }

    @Test func audioEmbeddingsSumAcrossCodebooks() throws {
        let model = HiggsV2Model(tinyConfig())
        let weights = MLXArray((0..<480).map { Float($0) / 100 }).reshaped([30, 16])
        try model.audioEmbeddings.update(parameters: ModuleParameters.unflattened(["weight": weights]))
        let codes = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped([3, 2])
        let expected = weights[1] + weights[13] + weights[25]
        let actual = model.audioEmbedding(codes)
        #expect(actual.shape == [2, 16])
        #expect(abs(actual[0] - expected).max().item(Float.self) < 1e-6)
    }

    @Test func incrementalCacheMatchesFullSequenceAndAudioRouting() {
        let model = HiggsV2Model(tinyConfig())
        let input = sin(MLXArray(0..<64).asType(.float32)).reshaped([1, 4, 16])
        let audioMask = MLXArray([false, false, true, true]).reshaped([1, 4])
        let full = model.hiddenStates(input, audioMask: audioMask)
        let cache: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
        let prefix = model.hiddenStates(input[0..., 0..<2], audioMask: audioMask[0..., 0..<2], cache: cache)
        eval(prefix)
        let suffix = model.hiddenStates(input[0..., 2..<4], audioMask: audioMask[0..., 2..<4], cache: cache)
        #expect(abs(full[0..., 2..<4] - suffix).max().item(Float.self) < 1e-4)
        let audioOnly = model.hiddenStates(input, audioMask: MLXArray.ones([1, 4], dtype: .bool), audioOnly: true)
        let routed = model.hiddenStates(input, audioMask: MLXArray.ones([1, 4], dtype: .bool))
        #expect(abs(audioOnly - routed).max().item(Float.self) < 1e-5)
    }

    @Test func quantizedDualFFNLoadRoundTrip() throws {
        let config = HiggsV2Configuration(textConfig: LlamaTTSConfiguration(
            hiddenSize: 64, hiddenLayers: 1, intermediateSize: 64, attentionHeads: 2,
            rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 1), audioNumCodebooks: 2, audioCodebookSize: 8)
        let original = HiggsV2Model(config)
        quantize(model: original, groupSize: 32, bits: 8)
        let parameters = original.parameters()
        #expect(parameters.flattened().contains { $0.0 == "layers.0.audio_mlp.gate_proj.scales" })
        let loaded = HiggsV2Model(config)
        quantize(model: loaded, groupSize: 32, bits: 8)
        try loaded.update(parameters: parameters, verify: .all)
        let input = MLXArray.ones([1, 2, 64])
        let mask = MLXArray([false, true]).reshaped([1, 2])
        let (_, expected) = original(inputsEmbeds: input, audioOutMask: mask)
        let (_, actual) = loaded(inputsEmbeds: input, audioOutMask: mask)
        #expect(abs(actual - expected).max().item(Float.self) < 1e-5)
    }

    @Test func samplerRestrictsEachCodebookAndGreedyIsStable() {
        let logits = MLXArray([Float(0), 1, 2, 3, 4, 3, 2, 1]).reshaped([2, 4])
        let key = MLXRandom.key(1)
        #expect(higgsV2Sample(logits, temperature: 0, topP: 1, topK: 0, key: key).asArray(Int32.self) == [3, 0])
        #expect(higgsV2Sample(logits, temperature: 1, topP: 0.01, topK: 1, key: key).asArray(Int32.self) == [3, 0])
    }

    @Test func streamingFlushPreservesEverySample() {
        var streamer = HiggsV2PCMStreamer(overlap: 4)
        let first = streamer.consume([Float](repeating: 1, count: 12), final: false)
        let second = streamer.consume([Float](repeating: 1, count: 20), final: false)
        let final = streamer.consume([Float](repeating: 1, count: 20), final: true)
        #expect(first.count == 8)
        #expect(second.count == 8)
        #expect(final.count == 4)
        #expect((first + second + final).allSatisfy { abs($0 - 1) < 1e-7 })
    }

}
