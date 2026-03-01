import Foundation
import Testing
import MLX
import MLXNN
import MLXAudioCore

@testable import MLXAudioLID

// MARK: - Configuration Tests

struct Wav2Vec2LIDConfigTests {

    @Test func configDecodingMmsLid256() throws {
        let json = """
        {
            "hidden_size": 1280,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "intermediate_size": 5120,
            "classifier_proj_size": 1024,
            "num_labels": 256,
            "conv_dim": [512, 512, 512, 512, 512, 512, 512],
            "conv_stride": [5, 2, 2, 2, 2, 2, 2],
            "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
            "num_conv_pos_embeddings": 128,
            "num_conv_pos_embedding_groups": 16,
            "id2label": {"0": "ara", "1": "cmn", "2": "eng"}
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Wav2Vec2LIDConfig.self, from: data)

        #expect(config.hiddenSize == 1280)
        #expect(config.numHiddenLayers == 48)
        #expect(config.numAttentionHeads == 16)
        #expect(config.intermediateSize == 5120)
        #expect(config.classifierProjSize == 1024)
        #expect(config.numLabels == 256)
        #expect(config.convDim.count == 7)
        #expect(config.convKernel.first == 10)
        #expect(config.convStride.first == 5)
        #expect(config.numConvPosEmbeddings == 128)
        #expect(config.numConvPosEmbeddingGroups == 16)
        #expect(config.id2label?["0"] == "ara")
        #expect(config.id2label?["2"] == "eng")
    }

    @Test func configDecodingWithoutId2label() throws {
        let json = """
        {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "classifier_proj_size": 256,
            "num_labels": 4,
            "conv_dim": [512, 512, 512, 512, 512, 512, 512],
            "conv_stride": [5, 2, 2, 2, 2, 2, 2],
            "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
            "num_conv_pos_embeddings": 128,
            "num_conv_pos_embedding_groups": 16
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Wav2Vec2LIDConfig.self, from: data)

        #expect(config.hiddenSize == 768)
        #expect(config.numHiddenLayers == 12)
        #expect(config.numLabels == 4)
        #expect(config.id2label == nil)
    }
}

// MARK: - LIDOutput Tests

struct LIDOutputTests {

    @Test func languagePredictionCreation() {
        let pred = LanguagePrediction(language: "eng", confidence: 0.95)
        #expect(pred.language == "eng")
        #expect(pred.confidence == 0.95)
    }

    @Test func lidOutputCreation() {
        let output = LIDOutput(
            language: "eng",
            confidence: 0.95,
            topLanguages: [
                LanguagePrediction(language: "eng", confidence: 0.95),
                LanguagePrediction(language: "fra", confidence: 0.03),
            ]
        )
        #expect(output.language == "eng")
        #expect(output.confidence == 0.95)
        #expect(output.topLanguages.count == 2)
        #expect(output.topLanguages[1].language == "fra")
    }

    @Test func lidErrorDescriptions() {
        let err1 = LIDError.invalidRepoID("bad/repo")
        #expect(err1.localizedDescription.contains("bad/repo"))

        let err2 = LIDError.configNotFound
        #expect(err2.localizedDescription.contains("config.json"))

        let err3 = LIDError.weightsNotFound
        #expect(err3.localizedDescription.contains("safetensors"))
    }
}

// MARK: - Sanitize Tests

struct Wav2Vec2SanitizeTests {

    @Test func sanitizeStripsPrefixWav2vec2() {
        let weights: [String: MLXArray] = [
            "wav2vec2.feature_projection.layer_norm.weight": MLXArray.ones([512]),
            "wav2vec2.feature_projection.layer_norm.bias": MLXArray.zeros([512]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        #expect(sanitized["feature_projection.layer_norm.weight"] != nil)
        #expect(sanitized["feature_projection.layer_norm.bias"] != nil)
        #expect(sanitized["wav2vec2.feature_projection.layer_norm.weight"] == nil)
    }

    @Test func sanitizeKeepsClassifierProjector() {
        let weights: [String: MLXArray] = [
            "projector.weight": MLXArray.ones([1024, 1280]),
            "projector.bias": MLXArray.zeros([1024]),
            "classifier.weight": MLXArray.ones([256, 1024]),
            "classifier.bias": MLXArray.zeros([256]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        #expect(sanitized["projector.weight"] != nil)
        #expect(sanitized["projector.bias"] != nil)
        #expect(sanitized["classifier.weight"] != nil)
        #expect(sanitized["classifier.bias"] != nil)
    }

    @Test func sanitizeSkipsMaskedSpecEmbed() {
        let weights: [String: MLXArray] = [
            "wav2vec2.masked_spec_embed": MLXArray.ones([1280]),
            "projector.weight": MLXArray.ones([1024, 1280]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        #expect(sanitized.keys.count == 1)
        #expect(sanitized["projector.weight"] != nil)
    }

    @Test func sanitizeSkipsAdapterLayers() {
        let weights: [String: MLXArray] = [
            "wav2vec2.encoder.layers.0.adapter_layer.linear_1.weight": MLXArray.ones([16, 1280]),
            "wav2vec2.encoder.layers.0.adapter_layer.linear_1.bias": MLXArray.zeros([16]),
            "wav2vec2.encoder.layers.0.layer_norm.weight": MLXArray.ones([1280]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        #expect(sanitized.keys.count == 1)
        #expect(sanitized["encoder.layers.0.layer_norm.weight"] != nil)
    }

    @Test func sanitizeTransposesConvWeights() {
        let weights: [String: MLXArray] = [
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight": MLXArray.ones([512, 1, 10]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        let w = sanitized["feature_extractor.conv_layers.0.conv.weight"]!
        eval(w)
        #expect(w.shape == [512, 10, 1])
    }

    @Test func sanitizeDoesNotTransposeLinearWeights() {
        let weights: [String: MLXArray] = [
            "wav2vec2.encoder.layers.0.attention.q_proj.weight": MLXArray.ones([1280, 1280]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        let w = sanitized["encoder.layers.0.attention.q_proj.weight"]!
        eval(w)
        #expect(w.shape == [1280, 1280])
    }

    @Test func sanitizeMergesWeightNorm() {
        let weightG = MLXArray.ones([1, 1, 128])
        let weightV = MLXArray.ones([1280, 80, 128])

        let weights: [String: MLXArray] = [
            "wav2vec2.encoder.pos_conv_embed.conv.weight_g": weightG,
            "wav2vec2.encoder.pos_conv_embed.conv.weight_v": weightV,
            "wav2vec2.encoder.pos_conv_embed.conv.bias": MLXArray.zeros([1280]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)

        #expect(sanitized["encoder.pos_conv_embed.conv.weight_g"] == nil)
        #expect(sanitized["encoder.pos_conv_embed.conv.weight_v"] == nil)

        let mergedWeight = sanitized["encoder.pos_conv_embed.conv.weight"]!
        eval(mergedWeight)
        #expect(mergedWeight.ndim == 3)
        #expect(mergedWeight.shape == [1280, 128, 80])

        #expect(sanitized["encoder.pos_conv_embed.conv.bias"] != nil)
    }

    @Test func sanitizeDropsUnknownPrefixKeys() {
        let weights: [String: MLXArray] = [
            "some_random_key.weight": MLXArray.ones([10]),
        ]

        let sanitized = Wav2Vec2ForSequenceClassification.sanitize(weights: weights)
        #expect(sanitized.isEmpty)
    }
}

// MARK: - Model Initialization Tests

struct Wav2Vec2ModelInitTests {

    static func makeSmallConfig() -> Wav2Vec2LIDConfig {
        let json = """
        {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "classifier_proj_size": 16,
            "num_labels": 4,
            "conv_dim": [16, 16, 16, 16, 16, 16, 16],
            "conv_stride": [5, 2, 2, 2, 2, 2, 2],
            "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
            "num_conv_pos_embeddings": 8,
            "num_conv_pos_embedding_groups": 4,
            "id2label": {"0": "eng", "1": "fra", "2": "deu", "3": "spa"}
        }
        """
        return try! JSONDecoder().decode(
            Wav2Vec2LIDConfig.self, from: json.data(using: .utf8)!
        )
    }

    @Test func modelCreation() {
        let config = Self.makeSmallConfig()
        let model = Wav2Vec2ForSequenceClassification(config: config)

        #expect(model.id2label.count == 4)
        #expect(model.id2label[0] == "eng")
        #expect(model.id2label[3] == "spa")
    }

    @Test func modelForwardPass() {
        let config = Self.makeSmallConfig()
        let model = Wav2Vec2ForSequenceClassification(config: config)

        let waveform = MLXRandom.normal([1, 16000])
        let logits = model(waveform)
        eval(logits)

        #expect(logits.ndim == 2)
        #expect(logits.dim(0) == 1)
        #expect(logits.dim(1) == 4)
    }

    @Test func modelPredictReturnsValidOutput() {
        let config = Self.makeSmallConfig()
        let model = Wav2Vec2ForSequenceClassification(config: config)

        let waveform = MLXRandom.normal([16000])
        let output = model.predict(waveform: waveform, topK: 3)

        #expect(!output.language.isEmpty)
        #expect(output.confidence >= 0 && output.confidence <= 1)
        #expect(output.topLanguages.count == 3)

        var prevConf: Float = 1.0
        for pred in output.topLanguages {
            #expect(pred.confidence <= prevConf)
            prevConf = pred.confidence
        }
    }

    @Test func modelPredictTopKClamped() {
        let config = Self.makeSmallConfig()
        let model = Wav2Vec2ForSequenceClassification(config: config)

        let waveform = MLXRandom.normal([16000])
        let output = model.predict(waveform: waveform, topK: 100)

        #expect(output.topLanguages.count == 4)
    }
}

// MARK: - Integration Test (requires model download)

struct MmsLid256IntegrationTests {

    @Test func loadRealModelAndPredict() async throws {
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (_, audioData) = try MLXAudioCore.loadAudioArray(from: audioURL)

        let model = try await Wav2Vec2ForSequenceClassification.fromPretrained("facebook/mms-lid-256")
        #expect(model.id2label.count == 256)

        let output = model.predict(waveform: audioData, topK: 5)
        #expect(!output.language.isEmpty, "Should detect some language")
        #expect(output.confidence > 0, "Confidence should be positive")
        #expect(output.topLanguages.count == 5, "Should return top-5 languages")
        // Note: intention.wav is very short (~2s), language detection may vary.
        // We verify the model loads, runs, and returns valid structured output.
    }
}
