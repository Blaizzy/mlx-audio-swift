import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCodecs

@Suite("Higgs audio tokenizer")
struct HiggsAudioSemanticTests {
    @Test func semanticEncoderUsesPublishedKeysAndAveragesHiddenStates() {
        var config = HiggsAudioSemanticConfig()
        config.hiddenSize = 16
        config.numHiddenLayers = 2
        config.numAttentionHeads = 2
        config.intermediateSize = 32
        config.convDim = [8]
        config.convKernel = [3]
        config.convStride = [2]
        let hubert = HiggsAudioHubertModel(config: config)
        let semantic = HiggsAudioSemanticEncoder(config: config)
        let audio = sin(MLXArray(0..<64).asType(.float32)).reshaped([1, 64])
        let mean = hubert.meanHiddenStates(audio)
        #expect(mean.shape == [1, 31, 16])
        #expect(semantic(mean).shape == mean.shape)
        let keys = Set(hubert.parameters().flattened().map(\.0))
        #expect(keys.contains("feature_extractor.conv_layers.0.layer_norm.weight"))
        #expect(keys.contains("encoder.pos_conv_embed.conv.weight_v"))
        #expect(keys.contains("encoder.layers.1.feed_forward.intermediate_dense.weight"))
        let finite = mean.asArray(Float.self).allSatisfy { $0.isFinite }
        #expect(finite)
    }

    @Test func codecConfigDefaultsAndSemanticSanitizer() throws {
        let data = Data("""
        {"model_type":"higgs_audio_v2_tokenizer","semantic_model_config":{"model_type":"hubert"}}
        """.utf8)
        let config = try JSONDecoder().decode(HiggsAudioTokenizerConfig.self, from: data)
        #expect(config.sampleRate == 24000)
        #expect(config.semanticModelConfig?.numHiddenLayers == 12)
        let weight = MLXArray.ones([2, 3, 4])
        let sanitized = HiggsAudioTokenizer.sanitize(weights: [
            "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1": weight,
            "encoder_semantic.conv.weight": weight,
            "semantic_model.masked_spec_embed": MLXArray.ones([2])], includeSemantic: true)
        #expect(sanitized["semantic_model.encoder.pos_conv_embed.conv.weight_v"]?.shape == [2, 3, 4])
        #expect(sanitized["encoder_semantic.conv.weight"]?.shape == [2, 3, 4])
        #expect(sanitized["semantic_model.masked_spec_embed"] == nil)
    }

    @Test func sincResamplingPreservesDCAndDuration() {
        let result = higgsAudioSincResample([Float](repeating: 1, count: 2400), from: 24000, to: 16000)
        #expect(result.count == 1600)
        #expect(result[20..<1580].allSatisfy { abs($0 - 1) < 0.001 })
    }
}
