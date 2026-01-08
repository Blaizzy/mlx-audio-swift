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
        #expect(ChatterboxTurboTextUtils.puncNorm("") == "You need to add some text for me to talk.")
    }

    @Test func testPuncNormCapitalizationAndPeriod() async throws {
        #expect(ChatterboxTurboTextUtils.puncNorm("hello world") == "Hello world.")
    }

    @Test func testPuncNormKeepsExistingPunctuation() async throws {
        #expect(ChatterboxTurboTextUtils.puncNorm("Hello world!") .hasSuffix("!"))
        #expect(ChatterboxTurboTextUtils.puncNorm("Hello world?") .hasSuffix("?"))
        #expect(ChatterboxTurboTextUtils.puncNorm("Hello world.") .hasSuffix("."))
    }

    @Test func testPuncNormWhitespaceCollapse() async throws {
        #expect(ChatterboxTurboTextUtils.puncNorm("  hello   world  ") == "hello world.")
    }

    @Test func testPuncNormRemovesMultipleSpaces() async throws {
        let result = ChatterboxTurboTextUtils.puncNorm("Hello    world")
        #expect(!result.contains("  "))
    }

    @Test func testPuncNormReplacements() async throws {
        let input = "Hello\u{2026}world\u{2014}ok\u{2013}now \u{201C}quote\u{201D} \u{2018}word\u{2019}"
        let expected = "Hello, world-ok-now \"quote\" 'word'."
        #expect(ChatterboxTurboTextUtils.puncNorm(input) == expected)
    }
}

struct ChatterboxTurboConfigTests {
    @Test func testT3Defaults() async throws {
        let config = T3Config()

        #expect(config.textTokensDictSize == 50276)
        #expect(config.speechTokensDictSize == 6563)
        #expect(config.llamaConfigName == "GPT2_medium")
        #expect(config.speakerEmbedSize == 256)
        #expect(config.speechCondPromptLen == 375)
        #expect(config.emotionAdv == false)
        #expect(config.usePerceiverResampler == false)
    }

    @Test func testT3TurboDefaults() async throws {
        let config = T3Config.turbo()

        #expect(config.textTokensDictSize == 50276)
        #expect(config.speechTokensDictSize == 6563)
        #expect(config.startSpeechToken == 6561)
        #expect(config.stopSpeechToken == 6562)
        #expect(config.speechCondPromptLen == 375)
        #expect(config.nChannels == 1024)
    }

    @Test func testT3IsMultilingual() async throws {
        let turbo = T3Config.turbo()
        #expect(turbo.isMultilingual == false)

        let multilingual = T3Config(textTokensDictSize: 2454)
        #expect(multilingual.isMultilingual == true)
    }

    @Test func testGPT2MediumDefaults() async throws {
        let config = GPT2Config.medium

        #expect(config.vocabSize == 50276)
        #expect(config.nPositions == 8196)
        #expect(config.nEmbeddings == 1024)
        #expect(config.nLayer == 24)
        #expect(config.nHead == 16)
    }
}

struct ChatterboxTurboSanitizeTests {
    @Test func testSanitizeRoutesPrefixedWeights() async throws {
        let model = ChatterboxTurboTTS()

        let weights: [String: MLXArray] = [
            "ve.lstm.weight": MLXArray.zeros([2, 2], type: Float.self),
            "t3.tfmr.weight": MLXArray.zeros([2, 2], type: Float.self),
            "s3gen.flow.weight": MLXArray.zeros([2, 2], type: Float.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized.keys.contains("ve.lstm.weight"))
        #expect(sanitized.keys.contains("t3.tfmr.weight"))
        #expect(sanitized.keys.contains("s3gen.flow.weight"))
    }

    @Test func testSanitizeKeepsOtherWeights() async throws {
        let model = ChatterboxTurboTTS()

        let weights: [String: MLXArray] = [
            "ve.lstm.weight": MLXArray.zeros([2, 2], type: Float.self),
            "unknown.param": MLXArray.zeros([1, 1], type: Float.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized.keys.contains("ve.lstm.weight"))
        #expect(sanitized.keys.contains("unknown.param"))
    }
}

struct ChatterboxTurboConditionalsTests {
    @Test func testConditionalsInit() async throws {
        let t3Cond = T3Cond(
            speakerEmb: MLXArray.zeros([1, 256], type: Float.self),
            clapEmb: nil,
            condPromptSpeechTokens: MLXArray.zeros([1, 375], type: Int32.self),
            condPromptSpeechEmb: nil,
            emotionAdv: nil
        )

        let genRef = S3GenReference(
            promptToken: MLXArray.zeros([1, 1], type: Int32.self),
            promptTokenLen: MLXArray([Int32(1)]),
            promptFeat: MLXArray.zeros([1, 80, 1], type: Float.self),
            promptFeatLen: MLXArray([Int32(1)]),
            embedding: MLXArray.zeros([1, 256], type: Float.self)
        )

        let conds = ChatterboxTurboConditionals(t3: t3Cond, gen: genRef)

        #expect(conds.t3.speakerEmb.shape == [1, 256])
        #expect(conds.gen.embedding.shape == [1, 256])
    }
}

struct ChatterboxTurboModelAliasTests {
    @Test func testModelAlias() async throws {
        #expect(Model.self == ChatterboxTurboTTS.self)
    }
}

struct ChatterboxTurboUtilsTests {
    @Test func testS3ResampleLinearLength() async throws {
        let input: [Float] = [0, 1, 2, 3]
        let output = s3ResampleLinear(input, from: 4, to: 8)
        #expect(output.count == 8)
        #expect(output.first == 0)
        #expect(output.last == 3)
    }

    @Test func testS3MelFiltersShape() async throws {
        let filters = s3MelFilters(
            sampleRate: 16_000,
            nFft: 400,
            nMels: 80,
            fMin: 0,
            fMax: 8_000,
            norm: "slaney",
            melScale: .slaney
        )
        #expect(filters.shape == [80, 201])
    }

    @Test func testS3LogMelSpectrogramShape() async throws {
        let audio = MLXArray.ones([400], type: Float.self)
        let mel = s3LogMelSpectrogram(audio, sampleRate: 16_000, nMels: 128, nFft: 400, hopLength: 160)
        #expect(mel.shape[0] == 128)
    }

    @Test func testS3MakeNonPadMask() async throws {
        let lengths = MLXArray([Int32(2), Int32(4)])
        let mask = s3MakeNonPadMask(lengths: lengths)
        #expect(mask.shape == [2, 4])
        #expect(mask[0, 0].item(Bool.self) == true)
        #expect(mask[0, 2].item(Bool.self) == false)
        #expect(mask[1, 3].item(Bool.self) == true)
    }

    @Test func testS3MaskToBias() async throws {
        let mask = MLXArray([true, false]).reshaped([1, 2])
        let bias = s3MaskToBias(mask)
        #expect(bias[0, 0].item(Float.self) == 0)
        #expect(bias[0, 1].item(Float.self) < -1e9)
    }

    @Test func testS3Padding() async throws {
        let a = MLXArray.ones([3, 2], type: Float.self)
        let b = MLXArray.ones([3, 4], type: Float.self)
        let (padded, lengths) = s3Padding([a, b])
        #expect(padded.shape == [2, 3, 4])
        #expect(lengths.asArray(Int32.self) == [2, 4])
    }

    @Test func testS3MergeTokenizedSegments() async throws {
        let segments = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        let merged = s3MergeTokenizedSegments(segments, overlap: 4, tokenRate: 1)
        #expect(merged == [1, 2, 7, 8])
    }

    @Test func testS3GenMelSpectrogramShape() async throws {
        let audio = MLXArray.ones([1, 960], type: Float.self)
        let mel = s3genMelSpectrogram(audio, nFft: 192, numMels: 80, samplingRate: 24_000, hopSize: 48, winSize: 192)
        #expect(mel.shape[0] == 1)
        #expect(mel.shape[1] == 80)
    }

    @Test func testS3GenSinusoidalPosEmbShape() async throws {
        let t = MLXArray([Float(0), Float(1)])
        let emb = s3genSinusoidalPosEmb(t, dim: 6)
        #expect(emb.shape == [2, 6])
    }
}

struct ChatterboxTurboDecoderTests {
    @Test func testConv1dPTShape() async throws {
        let conv = Conv1dPT(inChannels: 3, outChannels: 5, kernelSize: 3, padding: 1)
        let input = MLXArray.ones([1, 3, 4], type: Float.self)
        let output = conv(input)
        #expect(output.shape == [1, 5, 4])
    }

    @Test func testConvTranspose1dPTShape() async throws {
        let conv = ConvTranspose1dPT(inChannels: 3, outChannels: 2, kernelSize: 4, stride: 2, padding: 1)
        let input = MLXArray.ones([1, 3, 4], type: Float.self)
        let output = conv(input)
        #expect(output.shape == [1, 2, 8])
    }

    @Test func testCausalConv1dShape() async throws {
        let conv = CausalConv1d(inChannels: 3, outChannels: 3, kernelSize: 3)
        let input = MLXArray.ones([1, 3, 4], type: Float.self)
        let output = conv(input)
        #expect(output.shape == [1, 3, 4])
    }

    @Test func testBlock1DMaskedZeros() async throws {
        let block = Block1D(dim: 3, dimOut: 3, groups: 1)
        let input = MLXArray.ones([1, 3, 4], type: Float.self)
        let mask = MLXArray.zeros([1, 1, 4], type: Float.self)
        let output = block(input, mask: mask)
        #expect(output.max().item(Float.self) == 0)
    }

    @Test func testResnetBlock1DShape() async throws {
        let block = ResnetBlock1D(dim: 3, dimOut: 4, timeEmbedDim: 8, causal: false, groups: 1)
        let input = MLXArray.ones([1, 3, 4], type: Float.self)
        let mask = MLXArray.ones([1, 1, 4], type: Float.self)
        let time = MLXArray.ones([1, 8], type: Float.self)
        let output = block(input, mask: mask, timeEmb: time)
        #expect(output.shape == [1, 4, 4])
    }

    @Test func testDownsampleUpsampleShapes() async throws {
        let down = Downsample1D(dim: 3)
        let up = Upsample1D(dim: 3)
        let input = MLXArray.ones([1, 3, 8], type: Float.self)
        let downOut = down(input)
        let upOut = up(downOut)
        #expect(downOut.shape == [1, 3, 4])
        #expect(upOut.shape == [1, 3, 8])
    }

    @Test func testSelfAttention1DShape() async throws {
        let attn = SelfAttention1D(dim: 8, numHeads: 2, headDim: 4)
        let input = MLXArray.ones([1, 4, 8], type: Float.self)
        let output = attn(input)
        #expect(output.shape == [1, 4, 8])
    }

    @Test func testS3GenTransformerBlockShape() async throws {
        let block = S3GenTransformerBlock(dim: 8, numHeads: 2, headDim: 4)
        let input = MLXArray.ones([1, 4, 8], type: Float.self)
        let output = block(input)
        #expect(output.shape == [1, 4, 8])
    }
}

struct ChatterboxTurboTokenizerTests {
    @Test func testS3TokenizerV2QuantizeShape() async throws {
        let tokenizer = S3TokenizerV2(name: "speech_tokenizer_v2_25hz")
        let mel = MLXArray.ones([1, 128, 10], type: Float.self)
        let melLen = MLXArray([Int32(10)])
        let (codes, lengths) = tokenizer.quantize(mel, melLen)
        #expect(codes.ndim == 2)
        #expect(lengths.shape == [1])
    }

    @Test func testTokenizerMatchesHFIds() async throws {
        let model = try await ChatterboxTurboTTS.fromPretrained()
        guard let tokenizer = model.tokenizer else {
            Issue.record("Tokenizer not loaded for Chatterbox Turbo")
            return
        }

        let text = "Quick quality check. Does this sound natural?"
        let ids = tokenizer.encode(text: text)
        let expected = [21063, 3081, 2198, 13, 8314, 428, 2128, 3288, 30]
        #expect(ids == expected)
    }
}

struct ChatterboxTurboVoiceEncoderTests {
    @Test func testVoiceEncoderEmbedsFromWavs() async throws {
        let encoder = VoiceEncoder()
        let wav = [Float](repeating: 0, count: 16_000)
        let embeds = encoder.embedsFromWavs([wav], sampleRate: 16_000)
        #expect(embeds.shape == [1, 256])
    }
}

struct ChatterboxTurboS3GenTests {
    @Test func testS3Token2WavEmbedRefShape() async throws {
        let model = S3Token2Wav(meanflow: true)
        let wav = MLXArray.ones([S3GenSampleRate], type: Float.self)
        let ref = model.embedRef(refWav: wav, refSr: S3GenSampleRate)

        #expect(ref.promptToken.shape[0] == 1)
        #expect(ref.promptFeat.shape[0] == 1)
        #expect(ref.promptFeat.shape[2] == 80)
        #expect(ref.embedding.shape[0] == 1)
    }

    @Test func testS3Token2MelOutputShape() async throws {
        let model = S3Token2Mel(meanflow: true)
        let wav = MLXArray.ones([S3GenSampleRate], type: Float.self)
        let ref = model.embedRef(refWav: wav, refSr: S3GenSampleRate)
        let tokens = MLXArray.zeros([1, 0], type: Int32.self)
        let mel = model(tokens, refDict: ref, nCfmTimesteps: 2, finalize: true)

        #expect(mel.shape[0] == 1)
        #expect(mel.shape[1] == 80)
    }
}

struct ChatterboxTurboModelTests {
    @Test func testT3CondEncShape() async throws {
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

    @Test func testGPT2ModelOutputShape() async throws {
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

    @Test func testVoiceEncoderOutputShape() async throws {
        let encoder = VoiceEncoder()
        let hp = VoiceEncConfig()
        let mels = MLXArray.zeros([1, hp.vePartialFrames, hp.numMels], type: Float.self)
        let output = encoder(mels)

        #expect(output.shape == [1, hp.speakerEmbedSize])
    }

    @Test func testUpsampleConformerEncoderShape() async throws {
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

    @Test func testF0PredictorOutputShape() async throws {
        let predictor = F0Predictor()
        let mel = MLXArray.zeros([1, 80, 10], type: Float.self)
        let f0 = predictor(mel)

        #expect(f0.shape == [1, 10])
    }
}

struct ChatterboxTurboGenerationTests {
    @Test func testPrepareConditionalsRejectsShortAudio() async throws {
        let model = ChatterboxTurboTTS()
        let shortAudio = MLXArray.ones([S3GenSampleRate * 2], type: Float.self)

        var didThrow = false
        do {
            try model.prepareConditionals(refAudio: shortAudio, sampleRate: S3GenSampleRate)
        } catch {
            didThrow = true
        }
        #expect(didThrow)
    }

    @Test func testGenerateSmoke() async throws {
        let model = ChatterboxTurboTTS()
        let refAudio = MLXArray.ones([S3GenSampleRate * 6], type: Float.self)

        let audio = try model.generate(
            text: "Hello from Chatterbox Turbo.",
            refAudio: refAudio,
            sampleRate: S3GenSampleRate,
            splitPattern: nil,
            maxTokens: 6
        )

        let samples = audio.asArray(Float.self)
        #expect(!samples.isEmpty)
        #expect(samples.contains(where: { $0.isFinite }))
        #expect(samples.contains(where: { $0 != 0 }))
        #expect(model.sampleRate == S3GenSampleRate)
    }

    @Test func testGenerateStreamSmoke() async throws {
        let model = ChatterboxTurboTTS()
        let refAudio = MLXArray.ones([S3GenSampleRate * 6], type: Float.self)

        let stream = model.generateStream(
            text: "Streaming test.",
            refAudio: refAudio,
            sampleRate: S3GenSampleRate,
            chunkSize: 4,
            splitPattern: nil,
            maxTokens: 6
        )

        var chunks: [MLXArray] = []
        for try await event in stream {
            if case .audio(let audio) = event {
                chunks.append(audio)
            }
        }

        #expect(!chunks.isEmpty)
        let merged = MLX.concatenated(chunks, axis: 0)
        #expect(merged.shape[0] > 0)
    }

    @Test func testFromPretrainedRejectsInvalidRepo() async throws {
        var didThrow = false
        do {
            _ = try await ChatterboxTurboTTS.fromPretrained("invalid repo id")
        } catch {
            didThrow = true
        }
        #expect(didThrow)
    }

    @Test func testS3Token2WavSanitizeTransposesWeights() async throws {
        let model = S3Token2Wav(meanflow: true)
        let weights: [String: MLXArray] = [
            "conv.weight": MLXArray.zeros([2, 16, 3], type: Float.self),
            "bn.num_batches_tracked": MLXArray.zeros([1], type: Float.self)
        ]

        let sanitized = model.sanitize(weights)

        #expect(!sanitized.keys.contains("bn.num_batches_tracked"))
        #expect(sanitized["conv.weight"]?.shape == [2, 3, 16])
    }
}
