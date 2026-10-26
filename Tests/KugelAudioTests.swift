import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite(.serialized)
struct KugelAudioTests {
    func config() -> KugelAudioConfiguration {
        let decoder = Qwen2Configuration(hiddenSize: 64, hiddenLayers: 1, intermediateSize: 128,
            attentionHeads: 4, rmsNormEps: 1e-6, vocabularySize: 152064, kvHeads: 2)
        var acoustic = VibeVoiceAcousticConfig()
        acoustic.encoderNFilters = 4; acoustic.decoderNFilters = 4
        acoustic.encoderRatios = [2]; acoustic.encoderDepths = [1,1]
        var diffusion = VibeVoiceDiffusionConfig()
        diffusion.hiddenSize = 64; diffusion.latentSize = 64; diffusion.headLayers = 1
        diffusion.ddpmNumSteps = 20; diffusion.ddpmNumInferenceSteps = 4
        return .init(decoder: decoder, acousticTokenizer: acoustic, diffusion: diffusion)
    }
    @Test func promptAndStopping() {
        #expect(TTS.resolveModelType(modelRepo: "kugelaudio/kugelaudio-0-open") == "kugelaudio")
        #expect(TTS.resolveModelType(modelRepo: "local", modelType: " KUGELAUDIO ") == "kugelaudio")
        let ordinary = KugelAudioModel.prompt("  Hello!\n")
        #expect(ordinary.hasSuffix(" Text input:\n Speaker 0: Hello!\n Speech output:\n"))
        #expect(KugelAudioModel.prompt("Speaker 2: Hi").contains("Speaker 2: Hi"))
        #expect(!KugelAudioModel.prompt("Speaker 2: Hi").contains("Speaker 0:"))
        #expect(KugelAudioModel.speechDecision([0,8,4,-1], hasLatents: true).finalLatent)
        #expect(!KugelAudioModel.speechDecision([0,8,3,-1], hasLatents: true).finalLatent)
        #expect(!KugelAudioModel.speechDecision([0,8,4,-1], hasLatents: false).finalLatent)
        #expect(KugelAudioModel.speechDecision([0,0,0,0], hasLatents: false).token == KugelAudioModel.eosID)
    }
    @Test func configAliasesAndQuantizedPaths() throws {
        let json = #"{"model_type":"kugelaudio","acostic_vae_dim":64,"decoder_config":{"hidden_size":64,"num_hidden_layers":1,"intermediate_size":128,"num_attention_heads":4,"rms_norm_eps":0.000001,"vocab_size":152064,"num_key_value_heads":2},"acoustic_tokenizer_config":{},"diffusion_head_config":{"hidden_size":64},"quantization":{"group_size":32,"bits":4,"model.prediction_head.t_embedder.mlp.0":{"group_size":64,"bits":8}}}"#
        let c = try JSONDecoder().decode(KugelAudioConfiguration.self, from: Data(json.utf8))
        #expect(c.acousticVAEDim == 64)
        #expect(c.quantization?.quantization(layer: "prediction_head.t_embedder.mlp.fc1")?.bits == 8)
        #expect(c.quantization?.quantization(layer: "language_model.layers.0.self_attn.q_proj")?.bits == 4)
    }
    @Test func stochasticSolverNoiseAndReset() {
        let c = config().diffusion
        let solver = KugelAudioSDESolver(c, steps: 4)
        #expect(solver.timesteps == [19,14,10,5])
        func run(_ noise: Float) -> MLXArray {
            solver.reset()
            var value = MLXArray.ones([1,4])
            for _ in solver.timesteps { value = solver.step(.zeros([1,4]), sample: value, noise: .ones([1,4]) * noise) }
            return value
        }
        let a = run(1), b = run(0), repeated = run(1)
        #expect(isFinite(a).all().item(Bool.self))
        #expect(abs(a - repeated).max().item(Float.self) == 0)
        #expect(abs(a - b).max().item(Float.self) > 0.001)
    }
    @Test func solverMatchesReferenceValues() {
        let solver = KugelAudioSDESolver(config().diffusion, steps: 4)
        var value = MLXArray([Float(0.1),0.2,0.3,0.4]).reshaped(1,4)
        let prediction = MLXArray([Float(0.2),-0.3,0.4,-0.5]).reshaped(1,4)
        let noise = MLXArray([Float(0.5),0.4,-0.3,-0.2]).reshaped(1,4)
        for _ in solver.timesteps { value = solver.step(prediction, sample: value, noise: noise) }
        let expected = MLXArray([Float(0.37147319),0.95771211,-0.96835387,0.45531994]).reshaped(1,4)
        #expect(abs(value - expected).max().item(Float.self) < 1e-6)
    }
    @Test func sanitizeOriginalConvolutionsAndSkipUnusedEncoders() {
        let model = KugelAudioModel(config())
        let clean = model.sanitize([
            "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight": .ones([8,4,6]),
            "model.acoustic_tokenizer.decoder.stages.0.0.norm.weight": .ones([8]),
            "model.semantic_connector.fc1.weight": .ones([4,4]),
            "model.acoustic_tokenizer.encoder.head.conv.weight": .ones([4,4,3]),
            "model.prediction_head.t_embedder.mlp.layers.0.scales": .ones([64,2]),
            "unexpected.weight": .ones([2,2])])
        #expect(clean["acoustic_tokenizer.decoder.upsample_layers.1.convtr.convtr.weight"]?.shape == [4,6,8])
        #expect(clean["acoustic_tokenizer.decoder.stages.0.blocks.0.norm.weight"] != nil)
        #expect(clean["prediction_head.t_embedder.mlp.fc1.scales"] != nil)
        #expect(clean["unexpected.weight"] != nil)
        #expect(clean.count == 4)
        let native = model.sanitize(["acoustic_tokenizer.decoder.upsample_layers.0.conv.conv.weight": .ones([8,7,64])])
        #expect(native["acoustic_tokenizer.decoder.upsample_layers.0.conv.conv.weight"]?.shape == [8,7,64])
    }
    @Test func connectorQuantizationAndGuidedDiffusion() throws {
        let source = KugelAudioModel(config())
        quantize(model: source.acousticConnector, groupSize: 32, bits: 4)
        let weights = Dictionary(uniqueKeysWithValues: source.acousticConnector.parameters().flattened())
        let restored = VibeVoiceSpeechConnector(input: 64, output: 64)
        quantize(model: restored, groupSize: 32, bits: 4)
        try restored.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        #expect(weights["fc1.scales"] != nil && weights["fc2.scales"] != nil)
        #expect(abs(source.acousticConnector(.ones([1,1,64])) - restored(.ones([1,1,64]))).max().item(Float.self) == 0)
        let noise = (0..<4).map { _ in MLXArray.zeros([1,64]) }
        let solver = KugelAudioSDESolver(config().diffusion, steps: 4)
        let latent = try source.sampleSpeech(condition: .ones([1,64]), negative: .zeros([1,64]),
            options: .init(), solver: solver, initialNoise: .ones([1,64]), stepNoise: noise)
        #expect(latent.shape == [1,64] && isFinite(latent).all().item(Bool.self))
        let plain = try source.sampleSpeech(condition: .ones([1,64]), negative: nil,
            options: .init(guidanceScale: 1), solver: solver, initialNoise: .ones([1,64]), stepNoise: noise)
        #expect(plain.shape == [1,64] && isFinite(plain).all().item(Bool.self))
        #expect(source.acousticTokenizer.decode(latent.expandedDimensions(axis: 1)).shape == [1,2,1])
    }
    @Test func invalidInputFailsBeforeTokenizer() async throws {
        let model = KugelAudioModel(config())
        await #expect(throws: AudioGenerationError.self) { try await model.generate(text: "  ") }
        await #expect(throws: AudioGenerationError.self) {
            try await model.generate(text: "hello", options: .init(diffusionSteps: 0))
        }
        await #expect(throws: AudioGenerationError.self) {
            try await model.generate(text: "hello", voice: nil, refAudio: .zeros([128]), refText: "hello",
                language: nil, generationParameters: .init())
        }
    }
}
