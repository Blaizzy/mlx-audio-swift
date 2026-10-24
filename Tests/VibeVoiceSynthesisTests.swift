import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite(.serialized)
struct VibeVoiceSynthesisTests {
    @Test func causalDecoderStreamingMatchesWhole() throws {
        var c = VibeVoiceAcousticConfig()
        c.vaeDim = 4; c.encoderNFilters = 4; c.decoderNFilters = 4
        c.encoderRatios = [2, 3]; c.encoderDepths = [1, 1, 1]
        c.layerScaleInitValue = 0.1
        let model = VibeVoiceAcousticTokenizer(c)
        let latent = MLXArray((0..<28).map { sin(Float($0)) }).reshaped(1, 7, 4)
        let whole = model.decode(latent)
        let cache = VibeVoiceConvolutionCache()
        let chunks = [model.decode(latent[0..., 0..<2, 0...], cache: cache),
                      model.decode(latent[0..., 2..<3, 0...], cache: cache),
                      model.decode(latent[0..., 3..<7, 0...], cache: cache)]
        let streaming = concatenated(chunks, axis: 1)
        #expect(whole.shape == [1, 42, 1])
        #expect(abs(whole - streaming).max().item(Float.self) < 1e-4)
        #expect(cache.state.values.allSatisfy { $0.dim(1) <= 6 })
    }
    @Test func configAndCanonicalPaths() throws {
        let c = try JSONDecoder().decode(VibeVoiceAcousticConfig.self, from: Data(#"{"encoder_ratios":[2,3],"encoder_depths":"1-2-3","decoder_depths":[3,2,1]}"#.utf8))
        #expect(c.encoderDepths == [1,2,3])
        #expect(c.decoderDepths == [3,2,1])
        #expect(c.hopLength == 6)
        #expect(canonicalVibeVoiceSynthesisPath("prediction_head.t_embedder.mlp.0") == "prediction_head.t_embedder.mlp.fc1")
        #expect(canonicalVibeVoiceSynthesisPath("prediction_head.final_layer.adaLN_modulation.1") == "prediction_head.final_layer.adaLN_modulation.linear")
        #expect(canonicalVibeVoiceSynthesisPath("model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight") == "acoustic_tokenizer.decoder.upsample_layers.1.convtr.convtr.weight")
        #expect(canonicalVibeVoiceSynthesisPath("prediction_head.layers.0.adaLN_modulation.layers.1.scales") == "prediction_head.layers.0.adaLN_modulation.linear.scales")
    }
    @Test func acousticQuantizedRoundTrip() throws {
        var c = VibeVoiceAcousticConfig()
        c.encoderRatios = [2]; c.encoderDepths = [1,1]
        let source = VibeVoiceAcousticTokenizer(c)
        quantize(model: source, groupSize: 32, bits: 4)
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let restored = VibeVoiceAcousticTokenizer(c)
        quantize(model: restored, groupSize: 32, bits: 4)
        try restored.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let input = MLXArray.ones([1,2,64])
        #expect(abs(source.decode(input) - restored.decode(input)).max().item(Float.self) == 0)
        #expect(weights["decoder.stages.0.blocks.0.ffn.linear1.scales"] != nil)
    }
    @Test func diffusionQuantizedRoundTrip() throws {
        var c = VibeVoiceDiffusionConfig()
        c.hiddenSize = 64; c.latentSize = 64; c.headLayers = 1
        let source = VibeVoiceDiffusionHead(c)
        quantize(model: source, groupSize: 32, bits: 4)
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let restored = VibeVoiceDiffusionHead(c)
        quantize(model: restored, groupSize: 32, bits: 4)
        try restored.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let noisy = MLXArray.ones([1,64])
        let condition = MLXArray.ones([1,64])
        let timestep = MLXArray([Float(10)])
        let a = source(noisy, timesteps: timestep, condition: condition)
        let b = restored(noisy, timesteps: timestep, condition: condition)
        #expect(abs(a - b).max().item(Float.self) == 0)
        #expect(weights["t_embedder.mlp.fc1.scales"] != nil)
        #expect(weights["final_layer.adaLN_modulation.linear.scales"] != nil)
    }
    @Test func solverFinalStepAndReset() {
        let solver = VibeVoiceDPMSolver(trainingSteps: 20)
        solver.setTimesteps(4)
        #expect(solver.timesteps == [19,14,10,5])
        var sample = MLXArray.ones([1,4])
        for _ in solver.timesteps { sample = solver.step(.zeros([1,4]), sample: sample) }
        #expect(isFinite(sample).all().item(Bool.self))
        solver.reset()
        var repeated = MLXArray.ones([1,4])
        for _ in solver.timesteps { repeated = solver.step(.zeros([1,4]), sample: repeated) }
        #expect(abs(sample - repeated).max().item(Float.self) == 0)
    }
    @Test func splitQwenOmitsFinalNorm() {
        let c = Qwen2Configuration(hiddenSize: 8, hiddenLayers: 0, intermediateSize: 16, attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 32, kvHeads: 1)
        let stack = Qwen2ModelInner(c, includeEmbedding: false, includeNorm: false)
        let input = MLXArray.ones([1,2,8]) * 3
        #expect(abs(stack(inputEmbeddings: input) - input).max().item(Float.self) == 0)
        #expect(stack.parameters().flattened().isEmpty)
        let normalized = Qwen2ModelInner(c, includeEmbedding: false)
        #expect(abs(normalized(inputEmbeddings: input) - 1).max().item(Float.self) < 1e-5)
    }
}
