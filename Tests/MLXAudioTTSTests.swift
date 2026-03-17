//  Run the TTS suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/SopranoTextCleaningTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests/testTextCleaning()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs

private func loadTTSNetworkFixture(sampleRate: Int, maxSamples: Int) throws -> MLXArray {
    let audioURL = Bundle.module.url(
        forResource: "intention",
        withExtension: "wav",
        subdirectory: "media"
    )!
    let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: sampleRate)
    let sampleCount = min(audio.shape[0], maxSamples)
    return audio[0..<sampleCount]
}


// MARK: - Text Cleaning Unit Tests

struct SopranoTextCleaningTests {

    @Test func testTextCleaning() {
        // Test number normalization
        let text1 = "I have $100 and 50 cents."
        let cleaned1 = cleanTextForSoprano(text1)
        #expect(cleaned1.contains("one hundred dollars"), "Should expand dollar amounts")

        // Test abbreviations
        let text2 = "Dr. Smith went to the API conference."
        let cleaned2 = cleanTextForSoprano(text2)
        #expect(cleaned2.contains("doctor"), "Should expand Dr. to doctor")
        #expect(cleaned2.contains("a p i"), "Should expand API")

        // Test ordinals
        let text3 = "This is the 1st and 2nd test."
        let cleaned3 = cleanTextForSoprano(text3)
        #expect(cleaned3.contains("first"), "Should expand 1st to first")
        #expect(cleaned3.contains("second"), "Should expand 2nd to second")

        print("\u{001B}[32mText cleaning tests passed!\u{001B}[0m")
    }
}

struct EchoTTSTests {

    @Test func testTextNormalization() {
        let normalized = echoTtsNormalizeTextPrompt("Hello: world\nnew line")
        #expect(normalized.hasPrefix("[S1] "))
        #expect(normalized.contains(","))
        #expect(!normalized.contains("\n"))
    }

    @Test func testTokenizerEncode() {
        let tokens = echoTtsTokenizerEncode("hello", appendBOS: true, normalize: false)
        #expect(tokens.shape == [6])
        #expect(tokens[0].item(Int32.self) == 0)
    }

    @Test func testTextInputIDsAndMask() {
        let result = echoTtsTextInputIDsAndMask(
            ["hello", "world"],
            maxLength: 10,
            normalize: true,
            padToMax: true
        )
        #expect(result.inputIDs.shape == [2, 10])
        #expect(result.mask.shape == [2, 10])
        #expect(result.normalizedTexts.count == 2)
    }

    @Test func testEchoDiTForwardShapes() {
        let config = EchoDiTConfig(
            latentSize: 8,
            modelSize: 32,
            numLayers: 2,
            numHeads: 4,
            intermediateSize: 64,
            normEps: 1e-5,
            textVocabSize: 256,
            textModelSize: 32,
            textNumLayers: 1,
            textNumHeads: 4,
            textIntermediateSize: 64,
            speakerPatchSize: 2,
            speakerModelSize: 32,
            speakerNumLayers: 1,
            speakerNumHeads: 4,
            speakerIntermediateSize: 64,
            timestepEmbedSize: 16,
            adalnRank: 8
        )
        let model = EchoDiT(
            latentSize: config.latentSize,
            modelSize: config.modelSize,
            numLayers: config.numLayers,
            numHeads: config.numHeads,
            intermediateSize: config.intermediateSize,
            normEps: config.normEps,
            textVocabSize: config.textVocabSize,
            textModelSize: config.textModelSize,
            textNumLayers: config.textNumLayers,
            textNumHeads: config.textNumHeads,
            textIntermediateSize: config.textIntermediateSize,
            speakerPatchSize: config.speakerPatchSize,
            speakerModelSize: config.speakerModelSize,
            speakerNumLayers: config.speakerNumLayers,
            speakerNumHeads: config.speakerNumHeads,
            speakerIntermediateSize: config.speakerIntermediateSize,
            timestepEmbedSize: config.timestepEmbedSize,
            adalnRank: config.adalnRank
        )

        let x = MLXRandom.normal([1, 6, config.latentSize])
        let t = MLXArray([Float(0.7)])
        let textInputIDs = MLXArray([Int32(0), 1, 2, 3, 4]).reshaped([1, 5])
        let textMask = MLXArray([true, true, true, true, true]).reshaped([1, 5])
        let speakerLatent = MLXRandom.normal([1, 8, config.latentSize])
        let speakerMask = MLXArray.ones([1, 8], dtype: .bool)

        let kvText = model.getKVCacheText(textInputIDs, textMask: textMask)
        let kvSpeaker = model.getKVCacheSpeaker(speakerLatent)
        let output = model(
            x: x,
            t: t,
            textMask: textMask,
            speakerMask: speakerMask,
            kvCacheText: kvText,
            kvCacheSpeaker: kvSpeaker
        )

        #expect(output.shape == [1, 6, config.latentSize])
    }

    @Test func testSanitizeAndGenerateSmoke() throws {
        final class FakeFishAE: EchoTTSAudioCodec {
            func encodeZQ(_ audioData: MLXArray) -> MLXArray {
                MLXArray.zeros([audioData.shape[0], 8, max(audioData.shape[2] / 2_048, 1)], dtype: .float32)
            }

            func decodeZQ(_ zQ: MLXArray) -> MLXArray {
                MLXArray.zeros([zQ.shape[0], 1, zQ.shape[2] * 2_048], dtype: .float32)
            }
        }

        let config = EchoTTSConfig(
            dit: EchoDiTConfig(
                latentSize: 8,
                modelSize: 32,
                numLayers: 2,
                numHeads: 4,
                intermediateSize: 64,
                normEps: 1e-5,
                textVocabSize: 256,
                textModelSize: 32,
                textNumLayers: 1,
                textNumHeads: 4,
                textIntermediateSize: 64,
                speakerPatchSize: 2,
                speakerModelSize: 32,
                speakerNumLayers: 1,
                speakerNumHeads: 4,
                speakerIntermediateSize: 64,
                timestepEmbedSize: 16,
                adalnRank: 8
            ),
            sampler: EchoTTSSamplerConfig(
                numSteps: 1,
                cfgScaleText: 1,
                cfgScaleSpeaker: 1,
                sequenceLength: 4
            )
        )
        let model = EchoTTSModel(
            config: config,
            fishAE: FakeFishAE(),
            pcaState: EchoTTSPCAState(
                pcaComponents: MLXArray.eye(8, dtype: .float32),
                pcaMean: MLXArray.zeros([8], dtype: .float32),
                latentScale: 1
            )
        )

        let sanitized = model.sanitize(weights: [
            "cond_module.0.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "pca_components": MLXArray.zeros([1], dtype: .float32),
        ])
        #expect(sanitized["model.condModule.layers.0.weight"] != nil)
        #expect(sanitized["model.pca_components"] == nil)

        let result = try model.generateDetailed(
            text: "hi",
            refAudio: nil,
            rngSeed: 0,
            numSteps: 1,
            sequenceLength: 4
        )
        #expect(model.sampleRate == 44_100)
        #expect(result.audio.shape[0] > 0)
    }

    @Test func testDeleteBlockwiseModules() throws {
        let config = EchoTTSConfig(
            deleteBlockwiseModules: true,
            dit: EchoDiTConfig(
                latentSize: 8,
                modelSize: 32,
                numLayers: 2,
                numHeads: 4,
                intermediateSize: 64,
                normEps: 1e-5,
                textVocabSize: 256,
                textModelSize: 32,
                textNumLayers: 1,
                textNumHeads: 4,
                textIntermediateSize: 64,
                speakerPatchSize: 2,
                speakerModelSize: 32,
                speakerNumLayers: 1,
                speakerNumHeads: 4,
                speakerIntermediateSize: 64,
                timestepEmbedSize: 16,
                adalnRank: 8
            ),
            sampler: EchoTTSSamplerConfig(numSteps: 1, sequenceLength: 4)
        )
        let model = EchoTTSModel(config: config)

        let sanitized = model.sanitize(weights: [
            "latent_encoder.in_proj.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "blocks.0.attention.wk_latent.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "blocks.0.attention.wv_latent.weight": MLXArray.zeros([1, 1], dtype: .float32),
            "out_proj.weight": MLXArray.zeros([8, 32], dtype: .float32),
        ])
        #expect(sanitized["model.outProj.weight"] != nil)
        #expect(!sanitized.keys.contains(where: { $0.contains("latent_encoder") }))
        #expect(!sanitized.keys.contains(where: { $0.contains("wk_latent") }))
        #expect(!sanitized.keys.contains(where: { $0.contains("wv_latent") }))

        #expect(throws: AudioGenerationError.self) {
            try model.model.getKVCacheLatent(MLXArray.zeros([1, 0, 8], dtype: .float32))
        }

        #expect(throws: AudioGenerationError.self) {
            try model.generateLatents(text: "hi", blockSizes: [2], numSteps: 1, sequenceLength: 4)
        }
    }
}

@Suite("Echo TTS Network Tests", .serialized)
struct EchoTTSNetworkTests {

    @Test func echoTTSBaseLoadsConfiguredCodecAndGeneratesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network Echo TTS test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_ECHO_TTS_REPO"] ?? "mlx-community/echo-tts-base"
        let model = try await EchoTTSModel.fromPretrained(repo)
        let refAudio = try loadTTSNetworkFixture(sampleRate: model.sampleRate, maxSamples: model.sampleRate / 4)

        if repo == "mlx-community/echo-tts-base" {
            #expect(model.config.fishCodecRepo == "jordand/fish-s1-dac-min")
        }

        let result = try model.generateDetailed(
            text: "hello",
            refAudio: refAudio,
            rngSeed: 0,
            numSteps: 1,
            sequenceLength: 8
        )

        #expect(result.audio.shape[0] > 0)
        #expect(result.info.generationTokenCount == 8)
        #expect(model.fishAE != nil)
        #expect(model.pcaState != nil)
    }
}
