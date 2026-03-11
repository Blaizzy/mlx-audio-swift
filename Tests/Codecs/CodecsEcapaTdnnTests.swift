import Testing
import MLX
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs
@testable import MLXAudioLID

// MARK: - Shared ECAPA-TDNN Tests

struct SharedEcapaTdnnTests {

    @Test func ecapaTdnnConfigSupportsLidDefaults() {
        let config = MLXAudioCodecs.EcapaTdnnConfig(
            inputSize: 60,
            channels: 1024,
            embedDim: 256,
            kernelSizes: [5, 3, 3, 3, 1],
            dilations: [1, 2, 3, 4, 1],
            attentionChannels: 128,
            res2netScale: 8,
            seChannels: 128,
            globalContext: true
        )

        #expect(config.inputSize == 60)
        #expect(config.embedDim == 256)
        #expect(config.globalContext)
    }

    @Test func ecapaTdnnConfigPadsShortKernelAndDilationLists() {
        let config = MLXAudioCodecs.EcapaTdnnConfig(
            inputSize: 60,
            channels: 64,
            embedDim: 32,
            kernelSizes: [7],
            dilations: [2],
            attentionChannels: 16,
            res2netScale: 8,
            seChannels: 16,
            globalContext: true
        )

        #expect(config.kernelSizes.count >= 5)
        #expect(config.dilations.count >= 5)
        #expect(config.kernelSizes[0] == 7)
        #expect(config.dilations[0] == 2)
    }

    @Test func ecapaTdnnConfigDecodingPadsShortKernelAndDilationLists() throws {
        let json = """
        {
            "inputSize": 60,
            "channels": 64,
            "embedDim": 32,
            "kernelSizes": [7],
            "dilations": [2],
            "attentionChannels": 16,
            "res2netScale": 8,
            "seChannels": 16,
            "globalContext": true
        }
        """

        let config = try JSONDecoder().decode(
            MLXAudioCodecs.EcapaTdnnConfig.self,
            from: Data(json.utf8)
        )

        #expect(config.kernelSizes.count >= 5)
        #expect(config.dilations.count >= 5)
        #expect(config.kernelSizes[0] == 7)
        #expect(config.dilations[0] == 2)
    }

    @Test func ecapaTdnnBackboneProducesEmbeddingVectors() {
        Device.withDefaultDevice(.cpu) {
            let config = MLXAudioCodecs.EcapaTdnnConfig(
                inputSize: 60,
                channels: 64,
                embedDim: 32,
                kernelSizes: [5, 3, 3, 3, 1],
                dilations: [1, 2, 3, 4, 1],
                attentionChannels: 16,
                res2netScale: 8,
                seChannels: 16,
                globalContext: true
            )
            let backbone = MLXAudioCodecs.EcapaTdnnBackbone(config: config)

            let features = MLXRandom.normal([1, 100, 60])
            let embeddings = backbone(features)
            eval(embeddings)

            #expect(embeddings.shape == [1, 32])
        }
    }

    @Test func lidEcapaConsumesSharedBackboneContract() {
        Device.withDefaultDevice(.cpu) {
            let lidConfig = MLXAudioLID.EcapaTdnnConfig(
                nMels: 60,
                channels: 64,
                kernelSizes: [5, 3, 3, 3, 1],
                dilations: [1, 2, 3, 4, 1],
                attentionChannels: 16,
                res2netScale: 8,
                seChannels: 16,
                embeddingDim: 32,
                classifierHiddenDim: 64,
                numClasses: 4,
                id2label: ["0": "en: English", "1": "fr: French", "2": "de: German", "3": "es: Spanish"]
            )
            let model = EcapaTdnn(config: lidConfig)
            let mel = MLXRandom.normal([1, 80, 60])
            let logits = model(mel)
            eval(logits)

            #expect(logits.shape == [1, 4])
        }
    }
}
