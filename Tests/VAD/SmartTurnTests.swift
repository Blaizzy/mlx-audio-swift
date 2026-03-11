import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioVAD

// MARK: - Smart Turn Config Tests

struct SmartTurnConfigTests {

    @Test func smartTurnConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.modelType == "smart_turn")
        #expect(cfg.architecture == "smart_turn")
        #expect(cfg.dtype == "float32")
        #expect(cfg.encoderConfig.numMelBins == 80)
        #expect(cfg.processorConfig.samplingRate == 16000)
        #expect(cfg.processorConfig.maxAudioSeconds == 8)
    }

    @Test func smartTurnConfigFromDict() throws {
        let json = """
        {
            "dtype": "float16",
            "sample_rate": 22050,
            "max_audio_seconds": 6,
            "threshold": 0.42,
            "encoder_config": {
                "num_mel_bins": 8,
                "max_source_positions": 64,
                "d_model": 16,
                "encoder_attention_heads": 2,
                "encoder_layers": 1,
                "encoder_ffn_dim": 32,
                "k_proj_bias": false
            },
            "processor_config": {
                "sampling_rate": 16000,
                "max_audio_seconds": 8,
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 8,
                "normalize_audio": true,
                "threshold": 0.5
            }
        }
        """
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.dtype == "float16")
        #expect(cfg.sampleRate == 22050)
        #expect(cfg.maxAudioSeconds == 6)
        #expect(abs(cfg.threshold - 0.42) < 1e-6)
        #expect(cfg.encoderConfig.dModel == 16)
        #expect(cfg.processorConfig.nMels == 8)
    }

    @Test func smartTurnSynthesizesProcessorConfig() throws {
        let json = """
        {
            "sample_rate": 24000,
            "max_audio_seconds": 5,
            "threshold": 0.33,
            "encoder_config": { "num_mel_bins": 64 }
        }
        """
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.processorConfig.samplingRate == 24000)
        #expect(cfg.processorConfig.maxAudioSeconds == 5)
        #expect(cfg.processorConfig.nMels == 64)
        #expect(abs(cfg.processorConfig.threshold - 0.33) < 1e-6)
    }
}

// MARK: - Smart Turn Model Tests

private func makeTinySmartTurnConfig(dtype: String = "float32") -> SmartTurnConfig {
    let encoder = SmartTurnEncoderConfig(
        numMelBins: 8,
        maxSourcePositions: 64,
        dModel: 16,
        encoderAttentionHeads: 2,
        encoderLayers: 1,
        encoderFfnDim: 32,
        kProjBias: false
    )
    let processor = SmartTurnProcessorConfig(
        samplingRate: 16000,
        maxAudioSeconds: 8,
        nFft: 400,
        hopLength: 160,
        nMels: 8,
        normalizeAudio: true,
        threshold: 0.5
    )
    return SmartTurnConfig(dtype: dtype, encoderConfig: encoder, processorConfig: processor)
}

private func makeTinySmartTurnModel(dtype: String = "float32") throws -> SmartTurnModel {
    let model = SmartTurnModel(makeTinySmartTurnConfig(dtype: dtype))
    eval(model.parameters())

    if dtype == "float16" {
        let casted = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (key, value.asType(.float16))
            }
        )
        try model.update(parameters: ModuleParameters.unflattened(casted), verify: .noUnusedKeys)
        eval(model.parameters())
    }

    return model
}

struct SmartTurnForwardTests {

    @Test func smartTurnForwardShapeAndRange() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([1, 8, 64], type: Float.self)
        let out = model(input)
        eval(out)

        #expect(out.shape == [1, 1])
        let minVal = out.min().item(Float.self)
        let maxVal = out.max().item(Float.self)
        #expect(minVal >= 0.0)
        #expect(maxVal <= 1.0)
    }

    @Test func smartTurnForwardReturnLogits() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([1, 8, 64], type: Float.self)
        let logits = model(input, returnLogits: true)
        eval(logits)

        #expect(logits.shape == [1, 1])
    }

    @Test func smartTurnForwardBatchDimension() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([2, 8, 64], type: Float.self)
        let out = model(input)
        eval(out)

        #expect(out.shape == [2, 1])
    }

    @Test func smartTurnDTypePropagation() throws {
        let fp32Model = try makeTinySmartTurnModel(dtype: "float32")
        let fp32In = MLXArray.zeros([1, 8, 64], type: Float.self)
        let fp32Out = fp32Model(fp32In)
        eval(fp32Out)
        #expect(fp32Model.modelDType == .float32)
        #expect(fp32Out.dtype == .float32)

        let fp16Model = try makeTinySmartTurnModel(dtype: "float16")
        let fp16In = MLXArray.zeros([1, 8, 64], type: Float.self).asType(.float16)
        let fp16Out = fp16Model(fp16In)
        eval(fp16Out)
        #expect(fp16Model.modelDType == .float16)
        #expect(fp16Out.dtype == .float16)
    }

    @Test func smartTurnPrepareAudioArrayLengths() throws {
        let model = try makeTinySmartTurnModel()
        let maxSamples = model.config.processorConfig.maxAudioSeconds * model.config.processorConfig.samplingRate

        let short = MLXArray.ones([16000], type: Float.self)
        let shortOut = try model.prepareAudioSamples(short, sampleRate: 16000)
        #expect(shortOut.count == maxSamples)

        let long = MLXArray.ones([200000], type: Float.self)
        let longOut = try model.prepareAudioSamples(long, sampleRate: 16000)
        #expect(longOut.count == maxSamples)
    }

    @Test func smartTurnPrepareAudioArrayResample() throws {
        let model = try makeTinySmartTurnModel()
        let maxSamples = model.config.processorConfig.maxAudioSeconds * model.config.processorConfig.samplingRate

        let audio8k = MLXArray.ones([8000], type: Float.self)
        let out = try model.prepareAudioSamples(audio8k, sampleRate: 8000)
        #expect(out.count == maxSamples)
    }

    @Test func smartTurnPrepareInputFeaturesShape() throws {
        let model = try makeTinySmartTurnModel()
        let audio = MLXArray.zeros([16000], type: Float.self)
        let features = try model.prepareInputFeatures(audio, sampleRate: 16000)
        eval(features)

        #expect(features.shape == [8, 800])
    }

    @Test func smartTurnPredictEndpointReturnsOutput() throws {
        let model = try makeTinySmartTurnModel()
        let audio = MLXArray.zeros([16000], type: Float.self)
        let result = try model.predictEndpoint(audio, sampleRate: 16000, threshold: 0.5)

        #expect(result.prediction == 0 || result.prediction == 1)
        #expect(result.probability >= 0.0 && result.probability <= 1.0)
    }
}

// MARK: - Smart Turn Sanitization Tests

struct SmartTurnSanitizeTests {

    @Test func smartTurnSanitizeDropsValConstants() {
        let sanitized = SmartTurnModel.sanitize([
            "val_17": MLXArray.zeros([16, 16], type: Float.self),
            "val_123": MLXArray.zeros([1], type: Float.self)
        ])
        #expect(sanitized.isEmpty)
    }

    @Test func smartTurnSanitizeRemapsPrefixes() {
        let sanitized = SmartTurnModel.sanitize([
            "inner.classifier.0.weight": MLXArray.zeros([16, 16], type: Float.self),
            "inner.pool_attention.2.bias": MLXArray.zeros([1], type: Float.self)
        ])
        #expect(sanitized["classifier_0.weight"] != nil)
        #expect(sanitized["pool_attention_2.bias"] != nil)
    }

    @Test func smartTurnSanitizeConv1dTranspose() {
        let weights: [String: MLXArray] = [
            "encoder.conv1.weight": MLXArray.zeros([16, 8, 3], type: Float.self)
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["encoder.conv1.weight"]?.shape == [16, 3, 8])
    }

    @Test func smartTurnSanitizeFCTransposeHeuristics() {
        let weights: [String: MLXArray] = [
            "encoder.layers.0.fc1.weight": MLXArray.zeros([16, 32], type: Float.self),
            "encoder.layers.0.fc2.weight": MLXArray.zeros([32, 16], type: Float.self),
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["encoder.layers.0.fc1.weight"]?.shape == [32, 16])
        #expect(sanitized["encoder.layers.0.fc2.weight"]?.shape == [16, 32])
    }

    @Test func smartTurnSanitizePoolTransposeHeuristics() {
        let weights: [String: MLXArray] = [
            "pool_attention.0.weight": MLXArray.zeros([16, 256], type: Float.self),
            "pool_attention.2.weight": MLXArray.zeros([256, 1], type: Float.self),
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["pool_attention_0.weight"]?.shape == [256, 16])
        #expect(sanitized["pool_attention_2.weight"]?.shape == [1, 256])
    }
}

// MARK: - Smart Turn Network Tests

struct SmartTurnNetworkTests {

    @Test func smartTurnFromPretrainedEvaluatesConversationalAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network SmartTurn test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_SMARTTURN_REPO"] ?? "mlx-community/smart-turn-v3"
        let model = try await SmartTurnModel.fromPretrained(repo)

        let audioURLTrue = Bundle.module.url(
            forResource: "conversational_a",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (_, audioTrue) = try loadAudioArray(from: audioURLTrue, sampleRate: 16000)
        let resultTrue = try model.predictEndpoint(audioTrue, sampleRate: 16000, threshold: 0.5)
        #expect(resultTrue.prediction == 1)
        #expect(resultTrue.probability >= 0.5 && resultTrue.probability <= 1.0)

        let audioURLFalse = Bundle.module.url(
            forResource: "false-turn",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (_, audioFalse) = try loadAudioArray(from: audioURLFalse, sampleRate: 16000)
        let resultFalse = try model.predictEndpoint(audioFalse, sampleRate: 16000, threshold: 0.5)
        #expect(resultFalse.prediction == 0)
        #expect(resultFalse.probability >= 0.0 && resultFalse.probability < 0.5)
    }
}
