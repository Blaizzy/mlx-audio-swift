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
    @Test func testPuncNormEmpty() {
        #expect(ChatterboxTurboTextUtils.puncNorm("") == "You need to add some text for me to talk.")
    }

    @Test func testPuncNormCapitalizationAndPeriod() {
        #expect(ChatterboxTurboTextUtils.puncNorm("hello world") == "Hello world.")
    }

    @Test func testPuncNormWhitespaceCollapse() {
        #expect(ChatterboxTurboTextUtils.puncNorm("  hello   world  ") == "Hello world.")
    }

    @Test func testPuncNormReplacements() {
        let input = "Hello\u{2026}world\u{2014}ok\u{2013}now \u{201C}quote\u{201D} \u{2018}word\u{2019}"
        let expected = "Hello, world-ok-now \"quote\" 'word'."
        #expect(ChatterboxTurboTextUtils.puncNorm(input) == expected)
    }
}

struct ChatterboxTurboConfigTests {
    @Test func testT3TurboDefaults() {
        let config = T3Config.turbo()

        #expect(config.textTokensDictSize == 50276)
        #expect(config.speechTokensDictSize == 6563)
        #expect(config.startSpeechToken == 6561)
        #expect(config.stopSpeechToken == 6562)
        #expect(config.speechCondPromptLen == 375)
        #expect(config.nChannels == 1024)
    }

    @Test func testGPT2MediumDefaults() {
        let config = GPT2Config.medium

        #expect(config.vocabSize == 50276)
        #expect(config.nPositions == 8196)
        #expect(config.nEmbeddings == 1024)
        #expect(config.nLayer == 24)
        #expect(config.nHead == 16)
    }
}

struct ChatterboxTurboModelTests {
    @Test func testT3CondEncShape() {
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

    @Test func testGPT2ModelOutputShape() {
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
        let inputIds = MLXArray([[Int32(1), 2, 3]])
        let (hiddenStates, cache) = model(inputIds: inputIds, inputsEmbeds: nil, cache: nil)

        #expect(hiddenStates.shape == [1, 3, 8])
        #expect(cache.count == 2)
    }

    @Test func testVoiceEncoderOutputShape() {
        let encoder = VoiceEncoder()
        let hp = VoiceEncConfig()
        let mels = MLXArray.zeros([1, hp.vePartialFrames, hp.numMels], type: Float.self)
        let output = encoder(mels)

        #expect(output.shape == [1, hp.speakerEmbedSize])
    }
}
