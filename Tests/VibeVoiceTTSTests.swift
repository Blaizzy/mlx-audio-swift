import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite(.serialized)
struct VibeVoiceTTSTests {
    func configuration() -> VibeVoiceTTSConfiguration {
        var acoustic = VibeVoiceAcousticConfig()
        acoustic.vaeDim = 4; acoustic.encoderNFilters = 4; acoustic.decoderNFilters = 4
        acoustic.encoderRatios = [2]; acoustic.encoderDepths = [1, 1]
        var semantic = acoustic; semantic.vaeDim = 8; semantic.fixStd = 0
        var diffusion = VibeVoiceDiffusionConfig(); diffusion.hiddenSize = 8
        diffusion.latentSize = 4; diffusion.headLayers = 1; diffusion.ddpmNumInferenceSteps = 2
        return .init(decoder: .init(hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16,
            attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 151655, kvHeads: 1,
            tieWordEmbeddings: true), acousticTokenizer: acoustic, semanticTokenizer: semantic, diffusion: diffusion)
    }
    @Test func scriptNormalizationAndSpeakerOrder() throws {
        #expect(try VibeVoiceTTSPrompt.parse("Hello\n world") == [.init(speaker: 0, text: " Hello world")])
        let turns = try VibeVoiceTTSPrompt.parse("Speaker 8: Hello\nthere\nSpeaker 3: Hi\nSpeaker 8: Again")
        #expect(turns == [.init(speaker: 1, text: " Hello there"), .init(speaker: 0, text: " Hi"), .init(speaker: 1, text: " Again")])
        #expect(VibeVoiceTTSPrompt.speakerOrder(turns) == [1, 0])
        #expect(throws: Error.self) { try VibeVoiceTTSPrompt.parse(" \n ") }
    }
    @Test func rejectsASRAndIncompatibleFeedback() throws {
        var c = configuration(); try c.validate()
        c.architectures = ["VibeVoiceASRForConditionalGeneration"]
        #expect(throws: Error.self) { try c.validate() }
        c.architectures = nil; c.semanticTokenizer.encoderRatios = [3]
        #expect(throws: Error.self) { try c.validate() }
    }
    @Test func quantizationModulePathsAndTiedHeadValidation() throws {
        let data = Data(#"{"model_type":"vibevoice","decoder_config":{"hidden_size":64,"num_hidden_layers":1,"intermediate_size":128,"num_attention_heads":4,"rms_norm_eps":0.000001,"vocab_size":151655,"num_key_value_heads":2},"acoustic_tokenizer_config":{},"semantic_tokenizer_config":{},"diffusion_head_config":{},"quantization":{"group_size":32,"bits":4,"model.semantic_tokenizer.encoder.stages.0.0.ffn.linear1":{"group_size":32,"bits":8}}}"#.utf8)
        let config = try JSONDecoder().decode(VibeVoiceTTSConfiguration.self, from: data)
        #expect(config.quantization?.quantization(layer: "semantic_tokenizer.encoder.stages.0.blocks.0.ffn.linear1")?.bits == 8)
        let model = VibeVoiceTTSModel(configuration())
        let weights = ["model.language_model.embed_tokens.weight": MLXArray.zeros([2,2]), "lm_head.weight": MLXArray.zeros([2,2])]
        #expect(try model.sanitize(weights)["lm_head.weight"] == nil)
        #expect(throws: Error.self) { try model.sanitize(["language_model.embed_tokens.weight": .zeros([2,2]), "lm_head.weight": .ones([2,2])]) }
        #expect(try model.sanitize(["unexpected_head.weight": .ones([2,2])])["unexpected_head.weight"] != nil)
    }
    @Test func referenceNormalizationAndSamplingContract() throws {
        let model = VibeVoiceTTSModel(configuration())
        let input = MLXArray((0..<32).map { sin(Float($0)) * 0.1 })
        let noise = MLXArray.zeros([1, 16, 4]), scale = MLXArray.zeros([1])
        let a = try model.referenceEmbedding(input, sampleRate: 24000, samplingNoise: noise, scaleNoise: scale)
        let b = try model.referenceEmbedding(stacked([input, input], axis: -1), sampleRate: 24000,
            samplingNoise: noise, scaleNoise: scale)
        #expect(a.shape == [1,16,8])
        #expect(abs(a - b).max().item(Float.self) < 1e-6)
        #expect(throws: Error.self) { try model.referenceEmbedding(.zeros([0]), sampleRate: 24000) }
    }
    @Test func referenceSampleRateConversion() throws {
        let model = VibeVoiceTTSModel(configuration())
        let input = MLXArray((0..<2400).map { sin(Float($0) * 0.1) * 0.1 })
        let embedding = try model.referenceEmbedding(input, sampleRate: 48000, scaleNoise: .zeros([1]))
        #expect(embedding.dim(1) == 600)
        #expect(isFinite(embedding).all().item(Bool.self))
    }
    @Test func semanticEncoderStreamingAndFreshState() {
        let c = configuration().semanticTokenizer
        let encoder = VibeVoiceSynthesisEncoder(c)
        let input = MLXArray((0..<32).map { sin(Float($0) * 0.2) }).reshaped(1,32,1)
        let whole = encoder(input)
        let cache = VibeVoiceConvolutionCache()
        let chunks = [encoder(input[0..., ..<16, 0...], cache: cache), encoder(input[0..., 16..., 0...], cache: cache)]
        #expect(abs(whole - concatenated(chunks, axis: 1)).max().item(Float.self) < 1e-5)
        cache.state.removeAll()
        #expect(abs(whole - encoder(input, cache: cache)).max().item(Float.self) < 1e-5)
    }
    @Test func negativeCacheKeepsOnlyInitialPosition() {
        let cache = KVCacheSimple()
        _ = cache.update(keys: MLXArray((0..<24).map(Float.init)).reshaped(1,2,3,4), values: .ones([1,2,3,4]))
        VibeVoiceTTSModel.resetNegativeCache([cache])
        #expect(cache.offset == 1)
        #expect(cache.state[0].shape == [1,2,1,4])
        #expect(cache.state[0][0,1,0,0].item(Float.self) == 12)
    }
    @Test func tiedQwenProjectionSupportsQuantizedEmbedding() throws {
        let c = Qwen2Configuration(hiddenSize: 64, hiddenLayers: 0, intermediateSize: 128,
            attentionHeads: 4, rmsNormEps: 1e-6, vocabularySize: 32, kvHeads: 2, tieWordEmbeddings: true)
        let model = Qwen2ModelInner(c)
        let input = MLXArray.ones([1,64])
        let original = model.parameters().flattened().first { $0.0 == "embed_tokens.weight" }!.1
        #expect(abs(model.projectToVocabulary(input) - matmul(input, original.T)).max().item(Float.self) == 0)
        quantize(model: model, groupSize: 32, bits: 4)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(weights["embed_tokens.scales"] != nil)
        let restored = Qwen2ModelInner(c); quantize(model: restored, groupSize: 32, bits: 4)
        try restored.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        #expect(abs(model.projectToVocabulary(input) - restored.projectToVocabulary(input)).max().item(Float.self) == 0)
    }
}
