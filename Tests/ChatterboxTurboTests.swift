//
//  ChatterboxTurboTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 09/01/2026.
//

import Testing
import MLX

@testable import MLXAudioTTS

struct ChatterboxTurboTextUtilsTests {
    @Test func testPuncNormEmpty() async throws {
        try await withCPU {
                    #expect(ChatterboxTurboTextUtils.puncNorm("") == "You need to add some text for me to talk.")
        }
    }

    @Test func testPuncNormCapitalizationAndPeriod() async throws {
        try await withCPU {
                    #expect(ChatterboxTurboTextUtils.puncNorm("hello world") == "Hello world.")
        }
    }

    @Test func testPuncNormWhitespaceCollapse() async throws {
        try await withCPU {
                    #expect(ChatterboxTurboTextUtils.puncNorm("  hello   world  ") == "hello world.")
        }
    }

    @Test func testPuncNormReplacements() async throws {
        try await withCPU {
                    let input = "Hello\u{2026}world\u{2014}ok\u{2013}now \u{201C}quote\u{201D} \u{2018}word\u{2019}"
                    let expected = "Hello, world-ok-now \"quote\" 'word'."
                    #expect(ChatterboxTurboTextUtils.puncNorm(input) == expected)
        }
    }
}

struct ChatterboxTurboConfigTests {
    @Test func testT3TurboDefaults() async throws {
        try await withCPU {
                    let config = T3Config.turbo()

                    #expect(config.textTokensDictSize == 50276)
                    #expect(config.speechTokensDictSize == 6563)
                    #expect(config.startSpeechToken == 6561)
                    #expect(config.stopSpeechToken == 6562)
                    #expect(config.speechCondPromptLen == 375)
                    #expect(config.nChannels == 1024)
        }
    }

    @Test func testGPT2MediumDefaults() async throws {
        try await withCPU {
                    let config = GPT2Config.medium

                    #expect(config.vocabSize == 50276)
                    #expect(config.nPositions == 8196)
                    #expect(config.nEmbeddings == 1024)
                    #expect(config.nLayer == 24)
                    #expect(config.nHead == 16)
        }
    }
}

struct ChatterboxTurboModelTests {
    @Test func testT3CondEncShape() async throws {
        try await withCPU {
                    let hp = T3Config.turbo()
                    let condEnc = T3CondEnc(hp)
                    let speakerEmb = MLXArray.ones([1, hp.speakerEmbedSize])
                    let cond = T3Cond(
                        speakerEmb: speakerEmb,
                        clapEmb: nil,
                        condPromptSpeechTokens: nil,
                        condPromptSpeechEmb: nil,
                        emotionAdv: nil
                    )

                    let output = condEnc(cond)
                    #expect(output.shape == [1, 1, hp.nChannels])
        }
    }

    @Test func testGPT2ModelOutputShape() async throws {
        try await withCPU {
                    let config = GPT2Config(
                        vocabSize: 16,
                        nPositions: 8,
                        nEmbeddings: 8,
                        nLayer: 2,
                        nHead: 2,
                        nInner: nil,
                        activationFunction: "gelu_new",
                        residPdrop: 0.1,
                        embdPdrop: 0.1,
                        attnPdrop: 0.1,
                        layerNormEpsilon: 1e-5
                    )

                    let model = GPT2Model(config)
                    let inputIds = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
                    let (hiddenStates, cache) = model(inputIds: inputIds, inputsEmbeds: nil, cache: nil)

                    #expect(hiddenStates.shape == [1, 3, 8])
                    #expect(cache.count == 2)
        }
    }

    @Test func testVoiceEncoderOutputShape() async throws {
        try await withCPU {
                    let encoder = VoiceEncoder()
                    let hp = VoiceEncConfig()
                    let mels = MLXArray.zeros([1, hp.vePartialFrames, hp.numMels], type: Float.self)
                    let output = encoder(mels)

                    #expect(output.shape == [1, hp.speakerEmbedSize])
        }
    }

    @Test func testUpsampleConformerEncoderShape() async throws {
        try await withCPU {
                    let encoder = UpsampleConformerEncoder(
                        inputSize: 8,
                        outputSize: 8,
                        attentionHeads: 2,
                        linearUnits: 16,
                        numBlocks: 1,
                        dropoutRate: 0.0
                    )
                    let xs = MLXArray.zeros([1, 4, 8], type: Float.self)
                    let lens = MLXArray([Int32(4)])
                    let (out, mask) = encoder(xs, xsLens: lens)

                    #expect(out.shape == [1, 8, 8])
                    #expect(mask.shape == [1, 1, 8])
        }
    }

    @Test func testF0PredictorOutputShape() async throws {
        try await withCPU {
                    let predictor = F0Predictor()
                    let mel = MLXArray.zeros([1, 80, 10], type: Float.self)
                    let f0 = predictor(mel)

                    #expect(f0.shape == [1, 10])
        }
    }
}
