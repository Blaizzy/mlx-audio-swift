import Foundation
import Testing
import MLX
import MLXNN
import MLXAudioCodecs

@testable import MLXAudioCore
@testable import MLXAudioSTS

struct SAMAudioConfigTests {

    @Test func samAudioConfigDefaults() {
        let config = SAMAudioConfig()

        #expect(config.inChannels == 768)
        #expect(config.audioCodec.codebookDim == 128)
        #expect(config.transformer.outChannels == 256)
        #expect(config.transformer.contextDim == config.transformer.dim)
        #expect(config.numAnchors == 3)
    }

    @Test func samAudioConfigInfersInChannelsFromCodec() throws {
        let json = """
        {
            "audio_codec": {
                "codebook_dim": 64
            }
        }
        """
        let config = try JSONDecoder().decode(SAMAudioConfig.self, from: Data(json.utf8))
        #expect(config.inChannels == 384)
    }
}

struct SAMAudioBuildingBlockTests {

    @Test func samConv1dShapeWithStridePadding() {
        let conv = SAMConv1d(inChannels: 2, outChannels: 3, kernelSize: 3, stride: 2)
        let x = MLXArray.ones([1, 2, 5])
        let y = conv(x)

        #expect(y.shape == [1, 3, 3])
    }

    @Test func patcherReshapeContract() {
        let patcher = Patcher(inChannels: 4, outChannels: 8, patchSize: 2)
        let x = MLXArray.ones([2, 4, 16])
        let y = patcher(x)

        #expect(y.shape == [2, 8, 8])
    }

    @Test func anchorGatherSemantics() {
        let anchorIDs = MLXArray([1, 2, 3, 4, 5, 6], [2, 3]).asType(.int32)
        let anchorAlignment = MLXArray([0, 2, 1, 2, 1, 0, 0, 2], [2, 4]).asType(.int32)

        let gathered = MLX.takeAlong(anchorIDs, anchorAlignment, axis: 1).asArray(Int32.self)
        #expect(gathered == [1, 3, 2, 3, 5, 4, 4, 6])
    }

    @Test func embedAnchorsShape() {
        let module = EmbedAnchors(numEmbeddings: 3, embeddingDim: 4, outDim: 6)
        let x = MLXArray.ones([2, 5, 6])
        let anchorIDs = MLXArray([0, 1, 2, 0, 2, 3], [2, 3]).asType(.int32)
        let anchorAlignment = MLXArray([0, 1, 2, 1, 0, 0, 2, 2, 1, 0], [2, 5]).asType(.int32)
        let y = module(x, anchorIDs: anchorIDs, anchorAlignment: anchorAlignment)

        #expect(y.shape == [2, 5, 6])
    }

    @Test func rotaryEmbeddingShapeBHLE() {
        let rope = RotaryEmbedding(theta: 10000, headDim: 8, maxSequenceLength: 32)
        let x = MLXArray.ones([2, 4, 6, 8])  // (B, H, L, E)
        let y = rope(x, bhle: true)

        #expect(y.shape == [2, 4, 6, 8])
    }
}

struct SAMAudioTransformerTests {

    @Test func ditForwardShape() {
        let cfg = TransformerConfig(
            dim: 64,
            nHeads: 8,
            nLayers: 2,
            dropout: 0,
            normEps: 1e-5,
            qkNorm: true,
            fcBias: false,
            ffnExp: 4,
            ffnDimMultiplier: 1,
            multipleOf: 32,
            nonLinearity: "swiglu",
            useRope: true,
            maxPositions: 128,
            frequencyEmbeddingDim: 64,
            timestepNonLinearity: "swiglu",
            tBlockNonLinearity: "silu",
            tBlockBias: true,
            contextDim: 64,
            contextNonLinearity: "swiglu",
            contextEmbedderDropout: 0,
            contextNorm: false,
            outChannels: 32,
            inChannels: nil
        )

        let dit = DiT(config: cfg)
        let x = MLXArray.ones([2, 10, 64])
        let time = MLXArray([Float(0.1), Float(0.8)])
        let memory = MLXArray.ones([2, 6, 64])

        let output = dit(x, time: time, memory: memory)
        #expect(output.shape == [2, 10, 32])
    }
}

struct SAMAudioTextEncoderTests {

    @Test func attentionMaskSemantics() {
        let tokenIDs = [
            [10, 11, 12],
            [20],
        ]

        let (inputIDs, attentionMask) = T5TextEncoder.buildBatchTokenTensors(
            tokenIDs: tokenIDs,
            padTokenID: 0,
            maxLength: nil,
            padMode: "longest"
        )

        #expect(inputIDs.shape == [2, 3])
        #expect(attentionMask.shape == [2, 3])
        #expect(attentionMask.asArray(Bool.self) == [true, true, true, true, false, false])
    }

    @Test func attentionMaskRespectsMaxLength() {
        let tokenIDs = [
            [1, 2, 3, 4],
            [5, 6],
        ]

        let (inputIDs, attentionMask) = T5TextEncoder.buildBatchTokenTensors(
            tokenIDs: tokenIDs,
            padTokenID: 0,
            maxLength: 2,
            padMode: "longest"
        )

        #expect(inputIDs.shape == [2, 2])
        #expect(attentionMask.asArray(Bool.self) == [true, true, true, true])
        #expect(inputIDs.asArray(Int32.self) == [1, 2, 5, 6])
    }
}

struct SAMAudioProcessorTests {

    @Test func processAnchorsSpanIndexing() throws {
        let processor = SAMAudioProcessor(audioHopLength: 2, audioSamplingRate: 10)
        let audio = MLXArray(Array(repeating: Float(0), count: 10))

        let batch = try processor.process(
            descriptions: ["speech"],
            audios: [.array(audio)],
            anchors: [[("+", 0.2, 0.6)]]
        )

        #expect(batch.anchorIDs?.shape == [1, 3])
        #expect(batch.anchorIDs?.asArray(Int32.self) == [0, 3, 1])
        #expect(batch.anchorAlignment?.shape == [1, 5])
        #expect(batch.anchorAlignment?.asArray(Int32.self) == [0, 2, 2, 0, 0])
    }

    @Test func processAnchorsMarksPadding() throws {
        let processor = SAMAudioProcessor(audioHopLength: 2, audioSamplingRate: 10)
        let audioA = MLXArray(Array(repeating: Float(0), count: 10))
        let audioB = MLXArray(Array(repeating: Float(0), count: 4))

        let batch = try processor.process(
            descriptions: ["a", "b"],
            audios: [.array(audioA), .array(audioB)],
            anchors: nil
        )

        #expect(batch.anchorIDs?.shape == [2, 2])
        #expect(batch.anchorIDs?.asArray(Int32.self) == [0, 3, 0, 3])
        #expect(batch.anchorAlignment?.shape == [2, 5])
        #expect(batch.anchorAlignment?.asArray(Int32.self) == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    }

    @Test func processSupportsFileInputs() throws {
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let processor = SAMAudioProcessor(audioHopLength: 512, audioSamplingRate: 48_000)

        let batch = try processor.process(
            descriptions: ["speech"],
            audios: [.file(audioURL.path)],
            anchors: nil
        )

        #expect(batch.audios.shape[0] == 1)
        #expect(batch.audios.shape[1] == 1)
        #expect((batch.sizes?.shape ?? []) == [1])
        #expect((batch.audioPadMask?.shape[0] ?? 0) == 1)
    }
}

struct SAMAudioModelTests {

    private func tinyConfig() -> SAMAudioConfig {
        let audioCodec = DACVAEConfig(
            encoderDim: 8,
            encoderRates: [2],
            latentDim: 16,
            decoderDim: 16,
            decoderRates: [2],
            nCodebooks: 2,
            codebookSize: 32,
            codebookDim: 4,
            quantizerDropout: false,
            sampleRate: 8_000
        )

        let transformer = TransformerConfig(
            dim: 32,
            nHeads: 4,
            nLayers: 1,
            dropout: 0,
            normEps: 1e-5,
            qkNorm: false,
            fcBias: true,
            ffnExp: 2,
            ffnDimMultiplier: 1,
            multipleOf: 8,
            nonLinearity: "silu",
            useRope: false,
            maxPositions: 256,
            frequencyEmbeddingDim: 32,
            timestepNonLinearity: "silu",
            tBlockNonLinearity: "silu",
            tBlockBias: true,
            contextDim: 32,
            contextNonLinearity: "silu",
            contextEmbedderDropout: 0,
            contextNorm: false,
            outChannels: 8,
            inChannels: nil
        )

        return SAMAudioConfig(
            inChannels: 24,
            audioCodec: audioCodec,
            textEncoder: T5EncoderConfig(name: "t5-base", maxLength: 16, padMode: "longest", dim: 12),
            transformer: transformer,
            numAnchors: 3,
            anchorEmbeddingDim: 8
        )
    }

    @Test func alignInputsShape() {
        let model = SAMAudio(config: tinyConfig())
        let noisy = MLXArray.ones([1, 6, 8])
        let features = MLXArray.ones([1, 6, 8])
        let anchorIDs = MLXArray([0, 3, 1], [1, 3]).asType(.int32)
        let anchorAlignment = MLXArray([0, 2, 2, 0, 1, 1], [1, 6]).asType(.int32)

        let aligned = model.alignInputs(
            noisyAudio: noisy,
            audioFeatures: features,
            anchorIDs: anchorIDs,
            anchorAlignment: anchorAlignment
        )

        #expect(aligned.shape == [1, 6, 32])
    }

    @Test func separateWithCachedTextFeatures() async throws {
        let model = SAMAudio(config: tinyConfig())
        let audios = MLXArray(Array(repeating: Float(0.01), count: 64), [1, 1, 64])
        let textFeatures = MLXArray.ones([1, 4, 12])
        let textMask = MLXArray(Array(repeating: Int32(1), count: 4), [1, 4]).asType(.bool)

        let result = try await model.separate(
            audios: audios,
            descriptions: ["speech"],
            ode: SAMAudioODEOptions(method: .euler, stepSize: 0.5),
            _textFeatures: textFeatures,
            _textMask: textMask
        )

        #expect(result.target.count == 1)
        #expect(result.residual.count == 1)
        #expect(result.target[0].ndim == 2)
        #expect(result.target[0].shape[1] == 1)
        #expect(result.target[0].shape == result.residual[0].shape)
    }

    @Test func separateRejectsMissingCachedTextMask() async {
        let model = SAMAudio(config: tinyConfig())
        let audios = MLXArray(Array(repeating: Float(0.01), count: 64), [1, 1, 64])
        let textFeatures = MLXArray.ones([1, 4, 12])

        do {
            _ = try await model.separate(
                audios: audios,
                descriptions: ["speech"],
                ode: SAMAudioODEOptions(method: .euler, stepSize: 0.5),
                _textFeatures: textFeatures
            )
            Issue.record("Expected missingTextMask error")
        } catch SAMAudioError.missingTextMask {
            // Expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func separateLongWithCachedTextFeatures() async throws {
        let model = SAMAudio(config: tinyConfig())
        let audios = MLXArray(Array(repeating: Float(0.02), count: 200), [1, 1, 200])
        let textFeatures = MLXArray.ones([1, 4, 12])
        let textMask = MLXArray(Array(repeating: Int32(1), count: 4), [1, 4]).asType(.bool)

        let result = try await model.separateLong(
            audios: audios,
            descriptions: ["speech"],
            chunkSeconds: 0.01,
            overlapSeconds: 0.0025,
            ode: SAMAudioODEOptions(method: .euler, stepSize: 0.5),
            _textFeatures: textFeatures,
            _textMask: textMask
        )

        #expect(result.target.count == 1)
        #expect(result.residual.count == 1)
        #expect(result.target[0].shape[0] > 0)
        #expect(result.target[0].shape == result.residual[0].shape)
    }

    @Test func separateStreamingYieldsFinalChunk() async throws {
        let model = SAMAudio(config: tinyConfig())
        let audios = MLXArray(Array(repeating: Float(0.02), count: 200), [1, 1, 200])
        let textFeatures = MLXArray.ones([1, 4, 12])
        let textMask = MLXArray(Array(repeating: Int32(1), count: 4), [1, 4]).asType(.bool)

        var chunkCount = 0
        var sawLast = false

        let stream = model.separateStreaming(
            audios: audios,
            descriptions: ["speech"],
            chunkSeconds: 0.01,
            overlapSeconds: 0.0025,
            ode: SAMAudioODEOptions(method: .euler, stepSize: 0.5),
            _textFeatures: textFeatures,
            _textMask: textMask
        )

        for try await chunk in stream {
            #expect(chunk.target.ndim == 2)
            #expect(chunk.target.shape[1] == 1)
            #expect(chunk.target.shape == chunk.residual.shape)
            chunkCount += 1
            if chunk.isLastChunk {
                sawLast = true
            }
        }

        #expect(chunkCount > 0)
        #expect(sawLast)
    }
}

@Suite("SAMAudio Weights Tests", .serialized)
struct SAMAudioWeightsTests {

    private func tinyConfig() -> SAMAudioConfig {
        let audioCodec = DACVAEConfig(
            encoderDim: 8,
            encoderRates: [2],
            latentDim: 16,
            decoderDim: 16,
            decoderRates: [2],
            nCodebooks: 2,
            codebookSize: 32,
            codebookDim: 4,
            quantizerDropout: false,
            sampleRate: 8_000
        )

        let transformer = TransformerConfig(
            dim: 32,
            nHeads: 4,
            nLayers: 1,
            dropout: 0,
            normEps: 1e-5,
            qkNorm: false,
            fcBias: true,
            ffnExp: 2,
            ffnDimMultiplier: 1,
            multipleOf: 8,
            nonLinearity: "silu",
            useRope: false,
            maxPositions: 256,
            frequencyEmbeddingDim: 32,
            timestepNonLinearity: "silu",
            tBlockNonLinearity: "silu",
            tBlockBias: true,
            contextDim: 32,
            contextNonLinearity: "silu",
            contextEmbedderDropout: 0,
            contextNorm: false,
            outChannels: 8,
            inChannels: nil
        )

        return SAMAudioConfig(
            inChannels: 24,
            audioCodec: audioCodec,
            textEncoder: T5EncoderConfig(name: "t5-base", maxLength: 16, padMode: "longest", dim: 12),
            transformer: transformer,
            numAnchors: 3,
            anchorEmbeddingDim: 8
        )
    }

    @Test func convertWeightNameCoversCoreMappings() {
        let encoderResidual = SAMAudio.convertWeightName(
            "audio_codec.encoder.block.1.block.0.block.1.weight_v"
        )
        #expect(encoderResidual == "audio_codec.encoder.blocks.0.res1.conv1.weight_v")

        let wmPre = SAMAudio.convertWeightName(
            "audio_codec.decoder.wm_model.encoder_block.pre.1.bias"
        )
        #expect(wmPre == "audio_codec.decoder.conv_out.bias")

        let lstm = SAMAudio.convertWeightName(
            "audio_codec.decoder.wm_model.encoder_block.post.2.lstm.weight_hh_l1"
        )
        #expect(lstm == "audio_codec.decoder.wm_model.encoder_block.post_2.lstm.layers.1.Wh")
    }

    @Test func sanitizeCombinesLSTMBiasesAndDropsUnsupportedPrefixes() {
        let ones = MLXArray(Array(repeating: Float(1), count: 4))
        let twos = MLXArray(Array(repeating: Float(2), count: 4))
        let raw: [String: MLXArray] = [
            "text_encoder.encoder.weight": MLXArray.ones([2, 2]),
            "audio_codec.decoder.wm_model.encoder_block.post.2.lstm.bias_ih_l0": ones,
            "audio_codec.decoder.wm_model.encoder_block.post.2.lstm.bias_hh_l0": twos,
            "audio_codec.quantizer.in_proj.weight_v": MLXArray.ones([4, 1, 4]),
        ]

        let sanitized = SAMAudio.sanitize(weights: raw)

        #expect(sanitized["text_encoder.encoder.weight"] == nil)
        #expect(sanitized["audio_codec.quantizer_in_proj.weight_v"] != nil)
        let combinedKey = "audio_codec.decoder.wm_model.encoder_block.post_2.lstm.layers.0.bias"
        #expect(sanitized[combinedKey] != nil)
        #expect(sanitized[combinedKey]?.asArray(Float.self) == Array(repeating: Float(3), count: 4))
    }

    @Test func loadConvertedWeightsTransposesLinearWeights() throws {
        let model = SAMAudio(config: tinyConfig())
        let rawWeight = MLXArray(Array(0..<(24 * 32)).map(Float.init), [24, 32])
        let expected = rawWeight.transposed(1, 0).asArray(Float.self)

        try model.loadConvertedWeights(["proj.weight": rawWeight], strict: false)
        #expect(model.proj.weight.shape == [32, 24])
        #expect(model.proj.weight.asArray(Float.self) == expected)
    }

    @Test func fromPretrainedLoadsLocalFixture() async throws {
        let config = tinyConfig()
        let rawWeight = MLXArray(Array(0..<(24 * 32)).map(Float.init), [24, 32])
        let expected = rawWeight.transposed(1, 0).asArray(Float.self)

        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam-audio-fixture-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configURL = fixtureDir.appendingPathComponent("config.json")
        let configData = try JSONEncoder().encode(config)
        try configData.write(to: configURL)

        let weightsURL = fixtureDir.appendingPathComponent("model.safetensors")
        try save(arrays: ["proj.weight": rawWeight], url: weightsURL, stream: .cpu)

        let model = try await SAMAudio.fromPretrained(fixtureDir.path, strict: false)
        #expect(model.config.inChannels == config.inChannels)
        #expect(model.proj.weight.shape == [32, 24])
        #expect(model.proj.weight.asArray(Float.self) == expected)
    }

    @Test func fromPretrainedLoadsRealWeightsNetwork() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network SAMAudio test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_SAMAUDIO_REPO"] ?? SAMAudio.defaultRepo
        let hfToken = env["HF_TOKEN"]

        let model = try await SAMAudio.fromPretrained(repo, hfToken: hfToken, strict: false)

        #expect(model.config.inChannels == 6 * model.config.audioCodec.codebookDim)
        #expect(model.proj.weight.shape == [model.config.transformer.dim, model.config.inChannels])
        #expect(model.sampleRate == model.config.audioCodec.sampleRate)
    }
}
