import Foundation
import Testing
import MLX
import MLXNN
import MLXAudioCore

@testable import MLXAudioLID
@testable import mlx_audio_swift_lid

// MARK: - ECAPA-TDNN Configuration Tests

struct EcapaTdnnConfigTests {

    @Test func configDecodingDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(EcapaTdnnConfig.self, from: data)

        #expect(config.nMels == 60)
        #expect(config.channels == 1024)
        #expect(config.kernelSizes == [5, 3, 3, 3, 1])
        #expect(config.dilations == [1, 2, 3, 4, 1])
        #expect(config.attentionChannels == 128)
        #expect(config.res2netScale == 8)
        #expect(config.seChannels == 128)
        #expect(config.embeddingDim == 256)
        #expect(config.classifierHiddenDim == 512)
        #expect(config.numClasses == 107)
        #expect(config.id2label == nil)
    }

    @Test func configDecodingWithLabels() throws {
        let json = """
        {
            "n_mels": 60,
            "channels": 1024,
            "embedding_dim": 256,
            "id2label": {"0": "en: English", "1": "fr: French", "2": "de: German"}
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(EcapaTdnnConfig.self, from: data)

        #expect(config.nMels == 60)
        #expect(config.channels == 1024)
        #expect(config.numClasses == 3)
        #expect(config.id2label?["0"] == "en: English")
        #expect(config.id2label?["2"] == "de: German")
    }

    @Test func configDirectInit() {
        let config = EcapaTdnnConfig(
            nMels: 40, channels: 512, numClasses: 50
        )
        #expect(config.nMels == 40)
        #expect(config.channels == 512)
        #expect(config.numClasses == 50)
        #expect(config.embeddingDim == 256)
    }
}

// MARK: - ECAPA-TDNN Sanitize Tests

struct EcapaTdnnSanitizeTests {

    @Test func sanitizeDropsNumBatchesTracked() {
        let weights: [String: MLXArray] = [
            "embedding_model.blocks.0.conv.conv.weight": MLXArray.ones([1024, 5, 60]),
            "embedding_model.blocks.0.norm.norm.num_batches_tracked": MLXArray.zeros([1]),
        ]
        let sanitized = EcapaTdnn.sanitize(weights: weights)
        #expect(sanitized.count == 1)
        #expect(sanitized["embedding_model.block0.conv.weight"] != nil)
    }

    @Test func sanitizeRemapsBlockIndices() {
        let weights: [String: MLXArray] = [
            "embedding_model.blocks.0.conv.conv.weight": MLXArray.ones([1024, 5, 60]),
            "embedding_model.blocks.1.tdnn1.conv.conv.weight": MLXArray.ones([1024, 1, 1024]),
            "embedding_model.blocks.2.tdnn1.conv.conv.weight": MLXArray.ones([1024, 1, 1024]),
            "embedding_model.blocks.3.tdnn1.conv.conv.weight": MLXArray.ones([1024, 1, 1024]),
        ]
        let sanitized = EcapaTdnn.sanitize(weights: weights)

        #expect(sanitized["embedding_model.block0.conv.weight"] != nil)
        #expect(sanitized["embedding_model.block1.tdnn1.conv.weight"] != nil)
        #expect(sanitized["embedding_model.block2.tdnn1.conv.weight"] != nil)
        #expect(sanitized["embedding_model.block3.tdnn1.conv.weight"] != nil)
    }

    @Test func sanitizeFlattensDoubleNesting() {
        let weights: [String: MLXArray] = [
            "embedding_model.blocks.0.conv.conv.weight": MLXArray.ones([1024, 5, 60]),
            "embedding_model.blocks.0.conv.conv.bias": MLXArray.zeros([1024]),
            "embedding_model.blocks.0.norm.norm.weight": MLXArray.ones([1024]),
            "embedding_model.blocks.0.norm.norm.bias": MLXArray.zeros([1024]),
        ]
        let sanitized = EcapaTdnn.sanitize(weights: weights)

        #expect(sanitized["embedding_model.block0.conv.weight"] != nil)
        #expect(sanitized["embedding_model.block0.conv.bias"] != nil)
        #expect(sanitized["embedding_model.block0.norm.weight"] != nil)
        #expect(sanitized["embedding_model.block0.norm.bias"] != nil)
    }

    @Test func sanitizeFlattensSEBlockConv() {
        let weights: [String: MLXArray] = [
            "embedding_model.blocks.1.se_block.conv1.conv.weight": MLXArray.ones([128, 1, 1024]),
            "embedding_model.blocks.1.se_block.conv2.conv.weight": MLXArray.ones([1024, 1, 128]),
        ]
        let sanitized = EcapaTdnn.sanitize(weights: weights)

        #expect(sanitized["embedding_model.block1.se_block.conv1.weight"] != nil)
        #expect(sanitized["embedding_model.block1.se_block.conv2.weight"] != nil)
    }

    @Test func sanitizeFlattensAspBnAndFc() {
        let weights: [String: MLXArray] = [
            "embedding_model.asp_bn.norm.weight": MLXArray.ones([6144]),
            "embedding_model.fc.conv.weight": MLXArray.ones([256, 1, 6144]),
        ]
        let sanitized = EcapaTdnn.sanitize(weights: weights)

        #expect(sanitized["embedding_model.asp_bn.weight"] != nil)
        #expect(sanitized["embedding_model.fc.weight"] != nil)
    }

    @Test func sanitizePreservesRes2netBlocksArray() {
        let weights: [String: MLXArray] = [
            "embedding_model.blocks.1.res2net_block.blocks.0.conv.conv.weight": MLXArray.ones([128, 3, 128]),
            "embedding_model.blocks.1.res2net_block.blocks.1.conv.conv.weight": MLXArray.ones([128, 3, 128]),
        ]
        let sanitized = EcapaTdnn.sanitize(weights: weights)

        #expect(sanitized["embedding_model.block1.res2net_block.blocks.0.conv.weight"] != nil)
        #expect(sanitized["embedding_model.block1.res2net_block.blocks.1.conv.weight"] != nil)
    }
}

// MARK: - ECAPA-TDNN Mel Spectrogram Tests

struct EcapaMelSpectrogramTests {

    @Test func melOutputShape() {
        let audio = MLXRandom.normal([16000])
        let mel = EcapaMelSpectrogram.compute(audio: audio)
        eval(mel)

        #expect(mel.ndim == 3)
        #expect(mel.dim(0) == 1)
        #expect(mel.dim(2) == 60)
        #expect(mel.dim(1) > 0)
    }

    @Test func melEmptyAudio() {
        let audio = MLXArray.zeros([0])
        let mel = EcapaMelSpectrogram.compute(audio: audio)
        eval(mel)

        #expect(mel.dim(0) == 1)
        #expect(mel.dim(2) == 60)
    }

    @Test func melValuesAreFinite() {
        let audio = MLXRandom.normal([32000])
        let mel = EcapaMelSpectrogram.compute(audio: audio)
        eval(mel)

        let hasNan = any(isNaN(mel)).item(Bool.self)
        #expect(!hasNan, "Mel should not contain NaN")
    }
}

// MARK: - ECAPA-TDNN Model Tests

struct EcapaTdnnModelTests {

    static func makeSmallConfig() -> EcapaTdnnConfig {
        EcapaTdnnConfig(
            nMels: 60,
            channels: 64,
            kernelSizes: [5, 3, 3, 3, 1],
            dilations: [1, 2, 3, 4, 1],
            attentionChannels: 16,
            res2netScale: 8,
            seChannels: 16,
            embeddingDim: 32,
            classifierHiddenDim: 64,
            numClasses: 10,
            id2label: [
                "0": "en: English", "1": "fr: French", "2": "de: German",
                "3": "es: Spanish", "4": "it: Italian", "5": "pt: Portuguese",
                "6": "ru: Russian", "7": "zh: Chinese", "8": "ja: Japanese",
                "9": "ko: Korean"
            ]
        )
    }

    @Test func modelCreation() {
        let config = Self.makeSmallConfig()
        let model = EcapaTdnn(config: config)

        #expect(model.id2label.count == 10)
        #expect(model.id2label[0] == "en")
        #expect(model.id2label[1] == "fr")
    }

    @Test func modelLabelParsingExtractsIsoCode() {
        let config = EcapaTdnnConfig(
            numClasses: 2,
            id2label: ["0": "en: English", "1": "ceb: Cebuano"]
        )
        let model = EcapaTdnn(config: config)

        #expect(model.id2label[0] == "en")
        #expect(model.id2label[1] == "ceb")
    }

    @Test func modelForwardPass() {
        let config = Self.makeSmallConfig()
        let model = EcapaTdnn(config: config)

        let mel = MLXRandom.normal([1, 100, 60])
        let logits = model(mel)
        eval(logits)

        #expect(logits.ndim == 2)
        #expect(logits.dim(0) == 1)
        #expect(logits.dim(1) == 10)
    }

    @Test func modelPredictReturnsValidOutput() {
        let config = Self.makeSmallConfig()
        let model = EcapaTdnn(config: config)

        let waveform = MLXRandom.normal([16000])
        let output = model.predict(waveform: waveform, topK: 5)

        #expect(!output.language.isEmpty)
        #expect(output.confidence >= 0 && output.confidence <= 1)
        #expect(output.topLanguages.count == 5)

        var prevConf: Float = 1.0
        for pred in output.topLanguages {
            #expect(pred.confidence <= prevConf)
            prevConf = pred.confidence
        }
    }

    @Test func modelPredictTopKClamped() {
        let config = Self.makeSmallConfig()
        let model = EcapaTdnn(config: config)

        let waveform = MLXRandom.normal([16000])
        let output = model.predict(waveform: waveform, topK: 100)

        #expect(output.topLanguages.count == 10)
    }
}

// MARK: - ECAPA-TDNN Integration Test (requires model download)

struct EcapaTdnnIntegrationTests {

    @Test func loadRealModelAndPredict() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network ECAPA-TDNN test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (_, audioData) = try MLXAudioCore.loadAudioArray(from: audioURL)

        let model = try await EcapaTdnn.fromPretrained("beshkenadze/lang-id-voxlingua107-ecapa-mlx")
        #expect(model.id2label.count == 107)

        let output = model.predict(waveform: audioData, topK: 5)
        #expect(!output.language.isEmpty, "Should detect some language")
        #expect(output.confidence > 0, "Confidence should be positive")
        #expect(output.topLanguages.count == 5, "Should return top-5 languages")
    }
}
