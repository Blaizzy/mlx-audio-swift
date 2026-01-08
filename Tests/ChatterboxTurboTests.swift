//
//  ChatterboxTurboTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 09/01/2026.
//

import Testing

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
