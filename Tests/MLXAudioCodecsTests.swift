//  Run the codec suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/SharedDSPTests \
//      -only-testing:MLXAudioTests/VocosTests \
//      -only-testing:MLXAudioTests/SharedEcapaTdnnTests \
//      -only-testing:MLXAudioTests/EncodecTests \
//      -only-testing:MLXAudioTests/DACVAETests \
//      -only-testing:MLXAudioTests/BigVGANTests \
//      -only-testing:MLXAudioTests/DescriptDACTests \
//      -only-testing:MLXAudioTests/FishS1DACTests \
//      -only-testing:MLXAudioTests/CodecNetworkTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/SharedDSPTests'
//    -only-testing:'MLXAudioTests/VocosTests'
//    -only-testing:'MLXAudioTests/SharedEcapaTdnnTests'
//    -only-testing:'MLXAudioTests/EncodecTests'
//    -only-testing:'MLXAudioTests/DACVAETests'
//    -only-testing:'MLXAudioTests/BigVGANTests'
//    -only-testing:'MLXAudioTests/DescriptDACTests'
//    -only-testing:'MLXAudioTests/FishS1DACTests'
//    -only-testing:'MLXAudioTests/CodecNetworkTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/EncodecTests/testEncodecConfig()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Testing
import MLX
import MLXNN
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs
@testable import MLXAudioLID

private func loadCodecNetworkFixture(sampleRate: Int, maxSamples: Int) throws -> MLXArray {
    let audioURL = Bundle.module.url(
        forResource: "intention",
        withExtension: "wav",
        subdirectory: "media"
    )!
    let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: sampleRate)
    let sampleCount = min(audio.shape[0], maxSamples)
    return audio[0..<sampleCount]
}

struct MimiConvAdapterTests {

    @Test func convAdaptersKeepFlatParameterKeys() throws {
        let conv = MimiConv1d(inChannels: 3, outChannels: 4, ksize: 3, padding: 1)
        let convKeys = Set(conv.parameters().flattened().map(\.0))
        #expect(convKeys == ["weight", "bias"])

        let y = conv(MLXArray.zeros([2, 3, 8]))
        eval(y)
        #expect(y.shape == [2, 4, 8])

        try conv.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray.zeros(conv.weight.shape),
                "bias": MLXArray.zeros(conv.bias!.shape),
            ]),
            verify: .all
        )
        let zeroed = conv(MLXArray.ones([2, 3, 8]))
        eval(zeroed)
        #expect(MLX.max(MLX.abs(zeroed)).item(Float.self) == 0)

        let convtr = MimiConvTransposed1d(
            inChannels: 3,
            outChannels: 3,
            ksize: 4,
            stride: 2,
            groups: 3
        )
        let convtrKeys = Set(convtr.parameters().flattened().map(\.0))
        #expect(convtrKeys == ["weight", "bias"])

        let upsampled = convtr(MLXArray.zeros([2, 3, 5]))
        eval(upsampled)
        #expect(upsampled.shape == [2, 3, 12])
    }

    @Test func normWrappersKeepExistingCheckpointNesting() throws {
        let conv = NormConv1d(inChannels: 3, outChannels: 4, ksize: 3)
        let convKeys = Set(conv.parameters().flattened().map(\.0))
        #expect(convKeys.contains("conv.weight"))
        #expect(convKeys.contains("conv.bias"))
        #expect(!convKeys.contains("conv.conv.weight"))

        try conv.update(
            parameters: ModuleParameters.unflattened([
                "conv.weight": MLXArray.zeros(conv.conv.weight.shape),
                "conv.bias": MLXArray.zeros(conv.conv.bias!.shape),
            ]),
            verify: .all
        )

        let convtr = NormConvTranspose1d(inChannels: 3, outChannels: 3, ksize: 4, groups: 3)
        let convtrKeys = Set(convtr.parameters().flattened().map(\.0))
        #expect(convtrKeys.contains("convtr.weight"))
        #expect(convtrKeys.contains("convtr.bias"))
        #expect(!convtrKeys.contains("convtr.conv.weight"))
    }
}

struct SharedDSPTests {

    @Test func hammingWindowSupportsPeriodicAndSymmetricVariants() {
        let periodic = hammingWindow(size: 4).asArray(Float.self)
        let symmetric = hammingWindow(size: 4, periodic: false).asArray(Float.self)

        #expect(periodic.count == 4)
        #expect(symmetric.count == 4)

        #expect(abs(periodic[0] - 0.08) < 1e-3)
        #expect(abs(periodic[1] - 0.54) < 1e-3)
        #expect(abs(periodic[3] - 0.54) < 1e-3)

        #expect(abs(symmetric[0] - 0.08) < 1e-3)
        #expect(abs(symmetric[3] - 0.08) < 1e-3)
        #expect(abs(symmetric[1] - symmetric[2]) < 1e-3)
    }

    @Test func powerToDBAppliesTopDBClipping() {
        let spectrogram = MLXArray([Float(1e-10), Float(1e-5), Float(1.0)])
        let clipped = powerToDB(spectrogram, topDB: 80).asArray(Float.self)

        #expect(abs(clipped[0] + 80) < 1e-2)
        #expect(abs(clipped[1] + 50) < 1e-2)
        #expect(abs(clipped[2]) < 1e-3)
    }
}

struct S3TokenizerTests {

    @Test func smallConfigQuantizesMelToSpeechTokens() throws {
        var config = S3TokenizerConfig()
        config.nMels = 4
        config.nAudioState = 16
        config.nAudioHead = 4
        config.nAudioLayer = 1
        config.nCodebookSize = 6_561

        let model = S3TokenizerV2(config: config)
        let mel = MLXRandom.normal([1, config.nMels, 16])
        let melLen = MLXArray([Int32(16)])

        let (tokens, tokenLens) = model.quantize(mel, melLen: melLen)
        eval(tokens, tokenLens)

        #expect(tokens.shape == [1, 4])
        #expect(tokenLens.asArray(Int32.self) == [4])

        let values = tokens.asArray(Int32.self)
        #expect(values.allSatisfy { $0 >= 0 && $0 < Int32(config.nCodebookSize) })
    }
}

struct HiggsAudioTokenizerTests {

    @Test func configDefaultsAndDecodeShape() throws {
        let config = HiggsAudioTokenizerConfig()
        #expect(config.sampleRate == 24_000)
        #expect(config.codebookSize == 1_024)
        #expect(config.codebookDim == 64)
        #expect(config.dacNumCodebooks == 8)
        #expect(config.dacEncoderRatios == [8, 5, 4, 2, 3])

        let model = HiggsAudioTokenizer(config: config)
        let tokens = MLXArray(Array(repeating: Int32(0), count: 8), [1, 8])
        let waveform = model.decode(tokens)
        eval(waveform)

        #expect(waveform.shape == [960])
    }

    @Test func sanitizerMapsBundledCheckpointWeights() {
        let prefix = HiggsAudioTokenizer.defaultCodecPrefix
        let raw: [String: MLXArray] = [
            "\(prefix)quantizer.quantizers.0.codebook.embed": MLXRandom.normal([4, 2]),
            "\(prefix)quantizer.quantizers.0.codebook.embed_avg": MLXRandom.normal([4, 2]),
            "\(prefix)acoustic_decoder.block.0.conv_t1.weight": MLXRandom.normal([3, 4, 5]),
            "\(prefix)acoustic_encoder.conv1.weight": MLXRandom.normal([6, 2, 3]),
            "\(prefix)acoustic_decoder.snake1.alpha": MLXRandom.normal([1, 2, 1]),
            "\(prefix)semantic_model.encoder.pos_conv.weight": MLXRandom.normal([3, 2, 5]),
        ]

        let sanitized = HiggsAudioTokenizer.sanitize(weights: raw, prefix: prefix)

        #expect(sanitized["quantizer.quantizers.0.codebook.weight"]?.shape == [4, 2])
        #expect(sanitized["quantizer.quantizers.0.codebook.embed_avg"] == nil)
        #expect(sanitized["acoustic_decoder.block.0.conv_t1.weight"]?.shape == [4, 5, 3])
        #expect(sanitized["acoustic_encoder.conv1.weight"]?.shape == [6, 3, 2])
        #expect(sanitized["acoustic_decoder.snake1.alpha"]?.shape == [1, 1, 2])
        #expect(sanitized["semantic_model.encoder.pos_conv.weight"] == nil)
    }

    @Test func fromExtractedCheckpointMatchesPythonReferenceDecode() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["MLXAUDIO_HIGGS_AUDIO_CODEC_DIR"], !modelPath.isEmpty else {
            print("Skipping Higgs audio tokenizer parity test. Set MLXAUDIO_HIGGS_AUDIO_CODEC_DIR to an extracted tokenizer directory.")
            return
        }

        let expandedPath = NSString(string: modelPath).expandingTildeInPath
        let model = try HiggsAudioTokenizer.fromModelDirectory(URL(fileURLWithPath: expandedPath))
        let tokens = MLXArray((0..<16).map { Int32($0) }, [2, 8])
        let waveform = model.decode(tokens)
        eval(waveform)

        let expectedFirst16: [Float] = [
            -0.0014731525, -0.0012872171, -0.0006351388, -0.0010763766,
            -0.0013186398, -0.00085378025, 0.00090220413, 0.001662272,
            0.0026287395, 0.0033317907, 0.0018368699, 0.0017412195,
            0.0050528753, 0.0084180925, 0.0085247, 0.006915033,
        ]
        let actualFirst16 = waveform[0..<16].asArray(Float.self)

        #expect(waveform.shape == [1_920])
        for (actual, expected) in zip(actualFirst16, expectedFirst16) {
            #expect(abs(actual - expected) < 1e-4)
        }

        let mean = MLX.mean(waveform).item(Float.self)
        let absMean = MLX.mean(MLX.abs(waveform)).item(Float.self)
        let minValue = MLX.min(waveform).item(Float.self)
        let maxValue = MLX.max(waveform).item(Float.self)
        #expect(abs(mean - -0.00023663044) < 1e-4)
        #expect(abs(absMean - 0.018707583) < 1e-4)
        #expect(abs(minValue - -0.10222458) < 2e-4)
        #expect(abs(maxValue - 0.05652249) < 2e-4)
    }
}

struct StepAudio2Token2WavTests {

    @Test func fromLocalWeightsMatchesPythonReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["MLXAUDIO_STEPAUDIO2_DIR"], !modelPath.isEmpty else {
            print("Skipping StepAudio2 parity test. Set MLXAUDIO_STEPAUDIO2_DIR to a local Step-Audio-2-token2wav directory.")
            return
        }

        let expandedPath = NSString(string: modelPath).expandingTildeInPath
        let model = try StepAudio2Token2Wav.fromModelDirectory(URL(fileURLWithPath: expandedPath))
        let speechTokens = MLXArray([Int32(11), Int32(22)], [1, 2])
        let promptToken = MLXArray([Int32(3), Int32(4)], [1, 2])
        let promptTokenLen = MLXArray([Int32(2)])
        let promptValues = moduloFloatFixtureValues(count: 1 * 4 * 80, modulus: 17, subtracting: 8, divisor: 200.0)
        let promptFeat = MLXArray(promptValues, [1, 4, 80])
        let promptFeatLen = MLXArray([Int32(4)])
        let embeddingValues = moduloFloatFixtureValues(count: 192, modulus: 19, subtracting: 9, divisor: 50.0)
        let embedding = MLXArray(embeddingValues, [1, 192])
        let prompt = StepAudio2Prompt(
            promptToken: promptToken,
            promptTokenLen: promptTokenLen,
            promptFeat: promptFeat,
            promptFeatLen: promptFeatLen,
            embedding: embedding
        )

        let mel = try model.decodeToMel(speechTokens, prompt: prompt, nTimesteps: 1)
        eval(mel)

        let expectedMelFirst16: [Float] = [
            -5.4603024, -5.7961235, -5.953307, -5.4654946,
            -5.4765887, -5.585214, -5.3559914, -4.6866426,
            -1.3002429, -1.2458072, -1.4626238, -1.6580737,
            -0.8557438, -1.4504211, -1.9609354, -2.5217037,
        ]
        let actualMelFirst16 = mel.reshaped([-1])[0..<16].asArray(Float.self)
        #expect(mel.shape == [1, 80, 4])
        for (actual, expected) in zip(actualMelFirst16, expectedMelFirst16) {
            #expect(abs(actual - expected) < 1.2e-2)
        }

        let melMean = MLX.mean(mel).item(Float.self)
        let melAbsMean = MLX.mean(MLX.abs(mel)).item(Float.self)
        let melMin = MLX.min(mel).item(Float.self)
        let melMax = MLX.max(mel).item(Float.self)
        #expect(abs(melMean - -5.3294158) < 8e-3)
        #expect(abs(melAbsMean - 5.3294158) < 8e-3)
        #expect(abs(melMin - -8.932683) < 1e-2)
        #expect(abs(melMax - -0.8557438) < 3e-3)

        MLXRandom.seed(42)
        let waveform = model.vocode(mel)
        eval(waveform)

        let expectedFirst16: [Float] = [
            0.017583441, 0.014671039, 0.010786231, 0.007566597,
            0.0047792904, 0.0016860055, -0.002296098, -0.005033145,
            -0.004722653, -0.0017132716, 0.0027768025, 0.008602597,
            0.014388927, 0.01985283, 0.022311501, 0.019210702,
        ]
        let actualFirst16 = waveform.reshaped([-1])[0..<16].asArray(Float.self)

        #expect(waveform.shape == [1, 1_920])
        for (actual, expected) in zip(actualFirst16, expectedFirst16) {
            #expect(abs(actual - expected) < 6e-4)
        }

        let mean = MLX.mean(waveform).item(Float.self)
        let absMean = MLX.mean(MLX.abs(waveform)).item(Float.self)
        let minValue = MLX.min(waveform).item(Float.self)
        let maxValue = MLX.max(waveform).item(Float.self)
        #expect(abs(mean - -0.00004073924) < 5e-5)
        #expect(abs(absMean - 0.0040236986) < 1e-4)
        #expect(abs(minValue - -0.03442881) < 5e-4)
        #expect(abs(maxValue - 0.03196751) < 5e-4)
    }

    @Test func vocoderMatchesPythonReferenceWithSeededSource() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["MLXAUDIO_STEPAUDIO2_DIR"], !modelPath.isEmpty else {
            print("Skipping StepAudio2 vocoder parity test. Set MLXAUDIO_STEPAUDIO2_DIR to a local Step-Audio-2-token2wav directory.")
            return
        }

        let expandedPath = NSString(string: modelPath).expandingTildeInPath
        let model = try StepAudio2Token2Wav.fromModelDirectory(URL(fileURLWithPath: expandedPath))
        let melValues = moduloFloatFixtureValues(count: 1 * 80 * 4, modulus: 31, subtracting: 15, divisor: 10.0)
        let mel = MLXArray(melValues, [1, 80, 4])

        MLXRandom.seed(42)
        let waveform = model.vocode(mel)
        eval(waveform)

        let expectedFirst16: [Float] = [
            -0.069332026, -0.14232807, -0.33608523, -0.29552692,
            -0.09244516, 0.15974744, 0.25917616, 0.2368055,
            0.23517483, 0.20596091, 0.17615812, 0.23605251,
            0.19910856, 0.18924898, 0.044373818, -0.021684408,
        ]
        let actualFirst16 = waveform.reshaped([-1])[0..<16].asArray(Float.self)

        #expect(waveform.shape == [1, 1_920])
        for (actual, expected) in zip(actualFirst16, expectedFirst16) {
            #expect(abs(actual - expected) < 3e-3)
        }

        let mean = MLX.mean(waveform).item(Float.self)
        let absMean = MLX.mean(MLX.abs(waveform)).item(Float.self)
        let minValue = MLX.min(waveform).item(Float.self)
        let maxValue = MLX.max(waveform).item(Float.self)
        #expect(abs(mean - 0.015998093) < 5e-4)
        #expect(abs(absMean - 0.37149265) < 3e-3)
        #expect(abs(minValue - -0.99) < 1e-5)
        #expect(abs(maxValue - 0.99) < 1e-5)
    }
}


// MARK: - Vocos Tests
// Run Vocos tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -parallel-testing-enabled NO \
// -only-testing:MLXAudioTests/VocosTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct VocosTests {

    @Test func testConvNeXtBlock() throws {
        // Test basic ConvNeXtBlock forward pass
        let dim = 64
        let intermediateDim = 192
        let block = ConvNeXtBlock(
            dim: dim,
            intermediateDim: intermediateDim,
            layerScaleInitValue: 0.125,
            dwKernelSize: 7
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, dim])
        let output = block(input)

        // Output should have same shape as input (residual connection)
        #expect(output.shape == input.shape)
        print("ConvNeXtBlock output shape: \(output.shape)")
    }

    @Test func testConvNeXtBlockWithAdaNorm() throws {
        // Test ConvNeXtBlock with adaptive normalization
        let dim = 64
        let intermediateDim = 192
        let numEmbeddings = 4

        let block = ConvNeXtBlock(
            dim: dim,
            intermediateDim: intermediateDim,
            layerScaleInitValue: 0.125,
            adanormNumEmbeddings: numEmbeddings,
            dwKernelSize: 7
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, dim])
        let condEmbedding = MLXRandom.normal([1, numEmbeddings])

        let output = block(input, condEmbeddingId: condEmbedding)

        // Output should have same shape as input
        #expect(output.shape == input.shape)
        print("ConvNeXtBlock with AdaNorm output shape: \(output.shape)")
    }

    @Test func testVocosBackbone() throws {
        // Test VocosBackbone forward pass
        let inputChannels = 100
        let dim = 512
        let intermediateDim = 1536
        let numLayers = 8

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers
        )

        // Input shape: (batch, length, input_channels)
        let input = MLXRandom.normal([1, 50, inputChannels])
        let output = backbone(input)

        // Output should have shape (batch, length, dim)
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 50)
        #expect(output.shape[2] == dim)
        print("VocosBackbone output shape: \(output.shape)")
    }

    @Test func testVocosBackboneWithAdaNorm() throws {
        // Test VocosBackbone with adaptive normalization
        let inputChannels = 100
        let dim = 256
        let intermediateDim = 768
        let numLayers = 4
        let numEmbeddings = 4

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers,
            adanormNumEmbeddings: numEmbeddings
        )

        // Input shape: (batch, length, input_channels)
        let input = MLXRandom.normal([1, 50, inputChannels])
        let bandwidthId = MLXRandom.normal([1, numEmbeddings])

        let output = backbone(input, bandwidthId: bandwidthId)

        // Output should have shape (batch, length, dim)
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 50)
        #expect(output.shape[2] == dim)
        print("VocosBackbone with AdaNorm output shape: \(output.shape)")
    }

    @Test func testISTFTHead() throws {
        // Test ISTFTHead forward pass
        let dim = 512
        let nFft = 1024
        let hopLength = 256

        let head = MLXAudioCodecs.ISTFTHead(dim: dim, nFft: nFft, hopLength: hopLength)

        // Input shape: (batch, length, dim)
        let numFrames = 100
        let input = MLXRandom.normal([1, numFrames, dim])

        let output = head(input)

        // Output should be audio waveform
        // Expected length: approximately (numFrames - 1) * hopLength after trimming
        #expect(output.ndim == 1 || output.ndim == 2)
        print("ISTFTHead output shape: \(output.shape)")
    }

    @Test func testAdaLayerNorm() throws {
        // Test AdaLayerNorm
        let numEmbeddings = 4
        let embeddingDim = 256

        let adaNorm = AdaLayerNorm(
            numEmbeddings: numEmbeddings,
            embeddingDim: embeddingDim
        )

        // Input shape: (batch, length, dim)
        let input = MLXRandom.normal([2, 50, embeddingDim])
        let condEmbedding = MLXRandom.normal([2, numEmbeddings])

        let output = adaNorm(input, condEmbedding: condEmbedding)

        // Output should have same shape as input
        #expect(output.shape == input.shape)
        print("AdaLayerNorm output shape: \(output.shape)")
    }

    @Test func testVocosModel() throws {
        // Test full Vocos model
        let inputChannels = 100
        let dim = 256
        let intermediateDim = 768
        let numLayers = 4
        let nFft = 1024
        let hopLength = 256

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers
        )

        let head = MLXAudioCodecs.ISTFTHead(dim: dim, nFft: nFft, hopLength: hopLength)

        let vocos = Vocos(backbone: backbone, head: head)

        // Input shape: (batch, length, input_channels)
        let numFrames = 50
        let input = MLXRandom.normal([1, numFrames, inputChannels])

        let output = vocos(input)

        // Output should be audio waveform
        print("Vocos output shape: \(output.shape)")
        #expect(output.shape.count >= 1)
    }

    @Test func testVocosDecodeWithBandwidthId() throws {
        // Test Vocos decode with bandwidth conditioning
        let inputChannels = 128
        let dim = 256
        let intermediateDim = 768
        let numLayers = 4
        let numEmbeddings = 4
        let nFft = 1024
        let hopLength = 256

        let backbone = VocosBackbone(
            inputChannels: inputChannels,
            dim: dim,
            intermediateDim: intermediateDim,
            numLayers: numLayers,
            adanormNumEmbeddings: numEmbeddings
        )

        let head = MLXAudioCodecs.ISTFTHead(dim: dim, nFft: nFft, hopLength: hopLength)

        let vocos = Vocos(backbone: backbone, head: head)

        // Input shape: (batch, length, input_channels)
        let numFrames = 50
        let input = MLXRandom.normal([1, numFrames, inputChannels])
        let bandwidthId = MLXRandom.normal([1, numEmbeddings])

        let output = vocos.decode(input, bandwidthId: bandwidthId)

        // Output should be audio waveform
        print("Vocos decode with bandwidthId output shape: \(output.shape)")
        #expect(output.shape.count >= 1)
    }
}

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


// MARK: - Encodec Tests
// Run Encodec tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -parallel-testing-enabled NO \
// -only-testing:MLXAudioTests/EncodecTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct EncodecTests {

    @Test func testEncodecConfig() throws {
        // Test default config
        let config = EncodecConfig()

        #expect(config.audioChannels == 1)
        #expect(config.numFilters == 32)
        #expect(config.codebookSize == 1024)
        #expect(config.codebookDim == 128)
        #expect(config.hiddenSize == 128)
        #expect(config.numLstmLayers == 2)
        #expect(config.samplingRate == 24000)
        #expect(config.upsamplingRatios == [8, 5, 4, 2])

        print("EncodecConfig default values verified")
    }

    @Test func testEncodecConv1d() throws {
        // Test EncodecConv1d layer
        let config = EncodecConfig()
        let conv = EncodecConv1d(
            config: config,
            inChannels: 32,
            outChannels: 64,
            kernelSize: 7
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, 32])
        let output = conv(input)

        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 64)
        print("EncodecConv1d output shape: \(output.shape)")
    }

    @Test func testEncodecLSTM() throws {
        // Test EncodecLSTM layer
        let lstm = EncodecLSTM(inputSize: 64, hiddenSize: 64)

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 50, 64])
        let output = lstm(input)

        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 50)
        #expect(output.shape[2] == 64)
        print("EncodecLSTM output shape: \(output.shape)")
    }

    @Test func testEncodecResnetBlock() throws {
        // Test EncodecResnetBlock
        let config = EncodecConfig()
        let block = EncodecResnetBlock(
            config: config,
            dim: 64,
            dilations: [1, 1]
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, 64])
        let output = block(input)

        // Output should have same shape (residual connection)
        #expect(output.shape == input.shape)
        print("EncodecResnetBlock output shape: \(output.shape)")
    }

    @Test func testEncodecEuclideanCodebook() throws {
        // Test codebook quantization
        let config = EncodecConfig()
        let codebook = EncodecEuclideanCodebook(config: config)

        // Input shape: (batch, length, dim)
        let input = MLXRandom.normal([1, 50, config.codebookDim])
        let indices = codebook.encode(input)

        #expect(indices.shape[0] == 1)
        #expect(indices.shape[1] == 50)
        print("EncodecEuclideanCodebook indices shape: \(indices.shape)")

        // Decode back
        let decoded = codebook.decode(indices)
        #expect(decoded.shape[0] == 1)
        #expect(decoded.shape[1] == 50)
        #expect(decoded.shape[2] == config.codebookDim)
        print("EncodecEuclideanCodebook decoded shape: \(decoded.shape)")
    }

    @Test func testEncodecRVQ() throws {
        // Test Residual Vector Quantizer
        let config = EncodecConfig()
        let rvq = EncodecResidualVectorQuantizer(config: config)

        // Input shape: (batch, length, dim)
        let input = MLXRandom.normal([1, 50, config.codebookDim])
        let codes = rvq.encode(input, bandwidth: 1.5)

        // Codes shape should be (batch, num_quantizers, length)
        #expect(codes.shape[0] == 1)
        print("EncodecRVQ codes shape: \(codes.shape)")

        // Decode
        let decoded = rvq.decode(codes)
        #expect(decoded.shape[0] == 1)
        #expect(decoded.shape[1] == 50)
        print("EncodecRVQ decoded shape: \(decoded.shape)")
    }

    @Test func testEncodecModel() throws {
        // Test full Encodec model
        let config = EncodecConfig()
        let model = Encodec(config: config)

        // Input shape: (batch, length, channels)
        let audio = MLXRandom.normal([1, 1000, 1])

        // Encode
        let (codes, scales) = model.encode(audio, bandwidth: 1.5)
        print("Encodec codes shape: \(codes.shape)")
        #expect(codes.shape[0] >= 1)

        // Decode
        let decoded = model.decode(codes, scales)
        print("Encodec decoded shape: \(decoded.shape)")
        #expect(decoded.shape[0] == 1)
        #expect(decoded.shape[2] == 1)
    }
}


// MARK: - DACVAE Tests
// Run DACVAE tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -parallel-testing-enabled NO \
// -only-testing:MLXAudioTests/DACVAETests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct DACVAETests {

    @Test func testDACVAEConfig() throws {
        // Test default config
        let config = DACVAEConfig()

        #expect(config.encoderDim == 64)
        #expect(config.encoderRates == [2, 8, 10, 12])
        #expect(config.latentDim == 1024)
        #expect(config.decoderDim == 1536)
        #expect(config.decoderRates == [12, 10, 8, 2])
        #expect(config.codebookDim == 128)
        #expect(config.sampleRate == 48000)
        #expect(config.hopLength == 1920)  // 2 * 8 * 10 * 12

        print("DACVAEConfig default values verified")
    }

    @Test func testDACVAESnake1d() throws {
        // Test Snake activation
        let channels = 64
        let snake = DACVAESnake1d(channels: channels)

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, channels])
        let output = snake(input)

        // Output should have same shape
        #expect(output.shape == input.shape)
        print("DACVAESnake1d output shape: \(output.shape)")
    }

    @Test func testDACVAEWNConv1d() throws {
        // Test weight-normalized Conv1d
        let conv = DACVAEWNConv1d(
            inChannels: 32,
            outChannels: 64,
            kernelSize: 7,
            padding: 3
        )

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, 32])
        let output = conv(input)

        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 64)
        print("DACVAEWNConv1d output shape: \(output.shape)")
    }

    @Test func testDACVAEResidualUnit() throws {
        // Test ResidualUnit
        let dim = 64
        let unit = DACVAEResidualUnit(dim: dim, dilation: 1)

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, dim])
        let output = unit(input)

        // Output should have similar shape (may differ slightly due to padding)
        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == dim)
        print("DACVAEResidualUnit output shape: \(output.shape)")
    }

    @Test func testDACVAEEncoderBlock() throws {
        // Test encoder block
        let dim = 128
        let block = DACVAEEncoderBlock(dim: dim, stride: 2)

        // Input shape: (batch, length, channels)
        let input = MLXRandom.normal([1, 100, dim / 2])
        let output = block(input)

        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == dim)
        print("DACVAEEncoderBlock output shape: \(output.shape)")
    }

    @Test func testDACVAEEncoder() throws {
        // Test full encoder
        let encoder = DACVAEEncoder(
            dModel: 64,
            strides: [2, 4],
            dLatent: 128
        )

        // Input shape: (batch, length, 1)
        let input = MLXRandom.normal([1, 1000, 1])
        let output = encoder(input)

        #expect(output.shape[0] == 1)
        #expect(output.shape[2] == 128)
        print("DACVAEEncoder output shape: \(output.shape)")
    }

    @Test func testDACVAEQuantizerProj() throws {
        // Test quantizer projections
        let inProj = DACVAEQuantizerInProj(inDim: 128, outDim: 64)
        let outProj = DACVAEQuantizerOutProj(inDim: 64, outDim: 128)

        // Input shape: (batch, length, dim)
        let input = MLXRandom.normal([1, 50, 128])
        let projected = inProj(input)

        // Should project to 2*outDim (mean + logvar)
        #expect(projected.shape[0] == 1)
        #expect(projected.shape[2] == 128)  // 64 * 2
        print("DACVAEQuantizerInProj output shape: \(projected.shape)")

        // Take mean (first half)
        let mean = MLXRandom.normal([1, 50, 64])
        let unprojected = outProj(mean)

        #expect(unprojected.shape[0] == 1)
        #expect(unprojected.shape[2] == 128)
        print("DACVAEQuantizerOutProj output shape: \(unprojected.shape)")
    }

    @Test func testDACVAEModel() throws {
        // Test full DACVAE model with smaller config for faster testing
        let config = DACVAEConfig(
            encoderDim: 32,
            encoderRates: [2, 4],
            latentDim: 64,
            decoderDim: 64,
            decoderRates: [4, 2],
            codebookDim: 32
        )
        let model = DACVAE(config: config)

        // Input shape: (batch, 1, length) for callAsFunction
        let audio = MLXRandom.normal([1, 1, 800])

        // Encode to codebook space
        let encoded = model(audio)
        print("DACVAE encoded shape: \(encoded.shape)")
        #expect(encoded.shape[0] == 1)
        #expect(encoded.shape[1] == config.codebookDim)

        // Decode back to audio
        let decoded = model.decode(encoded)
        print("DACVAE decoded shape: \(decoded.shape)")
        #expect(decoded.shape[0] == 1)
        #expect(decoded.shape[2] == 1)
    }

    @Test func testDACVAEHopLength() throws {
        // Test hop length calculation
        let config1 = DACVAEConfig(encoderRates: [2, 4, 8])
        #expect(config1.hopLength == 64)  // 2 * 4 * 8

        let config2 = DACVAEConfig(encoderRates: [2, 8, 10, 12])
        #expect(config2.hopLength == 1920)  // 2 * 8 * 10 * 12

        print("DACVAEConfig hopLength verified")
    }
}


// MARK: - BigVGAN Tests

struct BigVGANTests {

    @Test func testSnakeBetaRegistersDistinctBetaParameter() throws {
        let activation = BigVGANActivation1d(
            channels: 3,
            activation: .snakebeta,
            snakeLogscale: true
        )
        let parameterKeys = Set(activation.parameters().flattened().map(\.0))
        #expect(parameterKeys.contains("act.alpha"))
        #expect(parameterKeys.contains("act.beta"))

        let alpha = MLXArray([-0.5, 0.0, 0.5] as [Float])
        let beta = MLXArray([0.25, 0.5, 0.75] as [Float])
        try activation.act.update(
            parameters: ModuleParameters.unflattened([
                "alpha": alpha,
                "beta": beta,
            ]),
            verify: .all
        )
        eval(activation.act.alpha, activation.act.beta!)

        #expect(activation.act.alpha.asArray(Float.self) == alpha.asArray(Float.self))
        #expect(activation.act.beta!.asArray(Float.self) == beta.asArray(Float.self))
    }

    @Test func testSnakeDoesNotRequireBetaParameter() throws {
        let activation = BigVGANActivation1d(
            channels: 3,
            activation: .snake,
            snakeLogscale: true
        )
        let parameterKeys = Set(activation.parameters().flattened().map(\.0))
        #expect(parameterKeys.contains("act.alpha"))
        #expect(!parameterKeys.contains("act.beta"))
    }

    @Test func testBigVGAN22kHz80BandShape() throws {
        let config = BigVGANConfig(
            numMels: 80,
            upsampleRates: [4, 4, 2, 2, 2, 2],
            upsampleKernelSizes: [8, 8, 4, 4, 4, 4],
            upsampleInitialChannel: 1536,
            resblock: .one,
            resblockKernelSizes: [3, 7, 11],
            resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation: .snakebeta,
            snakeLogscale: true,
            useBiasAtFinal: true,
            useTanhAtFinal: true
        )
        let model = BigVGAN(config: config)

        let input = MLXArray.zeros([1, 80, 800], dtype: .float32)
        let output = model(input)

        #expect(output.shape == [1, 1, 800 * config.upsampleRates.reduce(1, *)])
    }

    @Test func testBigVGAN44kHz128BandShape() throws {
        let config = BigVGANConfig(
            numMels: 128,
            upsampleRates: [8, 4, 2, 2, 2, 2],
            upsampleKernelSizes: [16, 8, 4, 4, 4, 4],
            upsampleInitialChannel: 1536,
            resblock: .one,
            resblockKernelSizes: [3, 7, 11],
            resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            activation: .snakebeta,
            snakeLogscale: true,
            useBiasAtFinal: false,
            useTanhAtFinal: false
        )
        let model = BigVGAN(config: config)

        let input = MLXArray.zeros([1, 128, 800], dtype: .float32)
        let output = model(input)

        #expect(output.shape == [1, 1, 800 * config.upsampleRates.reduce(1, *)])
    }

    @Test func testBigVGANSanitizeTransposesConvWeights() throws {
        let config = BigVGANConfig(
            numMels: 80,
            upsampleRates: [4],
            upsampleKernelSizes: [8],
            upsampleInitialChannel: 16,
            resblock: .one,
            resblockKernelSizes: [3],
            resblockDilationSizes: [[1, 3, 5]],
            activation: .snakebeta,
            snakeLogscale: true
        )
        let model = BigVGAN(config: config)

        let currentWeights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let convWeightEntry = currentWeights.first {
            $0.value.ndim == 3
                && ($0.key.contains("weight_v") || $0.key.contains("weightV"))
                && !$0.key.contains("ups.")
        }
        #expect(convWeightEntry != nil)
        guard let convWeightEntry else {
            return
        }
        let currentShape = convWeightEntry.value.shape
        let fakeTorchLayout = MLXArray.zeros([currentShape[0], currentShape[2], currentShape[1]], dtype: .float32)
        let sanitized = model.sanitize(weights: [convWeightEntry.key: fakeTorchLayout])

        #expect(sanitized[convWeightEntry.key]?.shape == currentShape)
    }
}


// MARK: - Descript DAC Tests

struct DescriptDACTests {

    @Test func testDescript16kHzShapes() throws {
        let model = DescriptDAC(config: DescriptDACConfig(
            encoderDim: 64,
            encoderRates: [2, 4, 5, 8],
            decoderDim: 1536,
            decoderRates: [8, 5, 4, 2],
            nCodebooks: 12,
            codebookSize: 1024,
            codebookDim: 8,
            sampleRate: 16_000
        ))

        let audio = MLXArray.zeros([1, 80_000, 1], dtype: .float32)
        let padded = model.preprocess(audio, sampleRate: 16_000)
        let (z, codes, latents, _, _) = model.encode(padded)
        let decoded = model.decode(z)

        #expect(z.shape == [1, 1024, 250])
        #expect(codes.shape == [1, 12, 250])
        #expect(latents.shape == [1, 96, 250])
        #expect(decoded.shape == [1, 80_043, 1])
    }

    @Test func testDescript24kHzShapes() throws {
        let model = DescriptDAC(config: DescriptDACConfig(
            encoderDim: 64,
            encoderRates: [2, 4, 5, 8],
            decoderDim: 1536,
            decoderRates: [8, 5, 4, 2],
            nCodebooks: 32,
            codebookSize: 1024,
            codebookDim: 8,
            sampleRate: 24_000
        ))

        let audio = MLXArray.zeros([1, 120_000, 1], dtype: .float32)
        let padded = model.preprocess(audio, sampleRate: 24_000)
        let (z, codes, latents, _, _) = model.encode(padded)
        let decoded = model.decode(z)

        #expect(z.shape == [1, 1024, 375])
        #expect(codes.shape == [1, 32, 375])
        #expect(latents.shape == [1, 256, 375])
        #expect(decoded.shape == [1, 120_043, 1])
    }

    @Test func testDescript44kHzShapes() throws {
        let model = DescriptDAC(config: DescriptDACConfig(
            encoderDim: 64,
            encoderRates: [2, 4, 8, 8],
            decoderDim: 1536,
            decoderRates: [8, 8, 4, 2],
            nCodebooks: 9,
            codebookSize: 1024,
            codebookDim: 8,
            sampleRate: 44_100
        ))

        let audio = MLXArray.zeros([1, 220_000, 1], dtype: .float32)
        let padded = model.preprocess(audio, sampleRate: 44_100)
        let (z, codes, latents, _, _) = model.encode(padded)
        let decoded = model.decode(z)

        #expect(z.shape == [1, 1024, 430])
        #expect(codes.shape == [1, 9, 430])
        #expect(latents.shape == [1, 72, 430])
        #expect(decoded.shape == [1, 220_235, 1])
    }

    @Test func testDescriptAudioCodecRoundTripUsesCodes() throws {
        let model = DescriptDAC(config: DescriptDACConfig(
            encoderDim: 8,
            encoderRates: [2, 2],
            latentDim: 32,
            decoderDim: 32,
            decoderRates: [2, 2],
            nCodebooks: 2,
            codebookSize: 16,
            codebookDim: 4,
            sampleRate: 16_000
        ))

        let waveform = MLXArray.zeros([1, 128, 1], dtype: .float32)
        let encoded = model.encodeAudio(waveform)
        let decoded = model.decodeAudio(encoded)

        #expect(encoded.codes.shape[0] == 1)
        #expect(encoded.originalLength == 128)
        #expect(decoded.shape[0] == 1)
        #expect(decoded.shape[2] == 1)
    }

    @Test func testDescriptSanitizeRemovesModuleArrayLayersSegments() throws {
        let model = DescriptDAC(config: DescriptDACConfig())
        let weights: [String: MLXArray] = [
            "encoder.block.layers.0.weight_v": MLXArray.zeros([1, 1, 1], dtype: .float32),
            "decoder.model.layers.1.block.layers.4.weight_g": MLXArray.zeros([1, 1, 1], dtype: .float32),
            "quantizer.quantizers.0.in_proj.weight_v": MLXArray.zeros([1, 1, 1], dtype: .float32),
            "quantizer.quantizers.0.out_proj.weight_g": MLXArray.zeros([1, 1, 1], dtype: .float32)
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["encoder.block.0.weight_v"] != nil)
        #expect(sanitized["decoder.model.1.block.4.weight_g"] != nil)
        #expect(sanitized["quantizer.quantizers.0.inProj.weight_v"] != nil)
        #expect(sanitized["quantizer.quantizers.0.outProj.weight_g"] != nil)
        #expect(sanitized["encoder.block.layers.0.weight_v"] == nil)
    }
}


// MARK: - Fish S1 DAC Tests

struct FishS1DACTests {

    @Test func testTinyEncodeDecode() throws {
        let quantizer = FishS1DownsampleResidualVectorQuantize(
            inputDim: 16,
            nCodebooks: 2,
            codebookDim: 4,
            codebookSize: 16,
            semanticCodebookSize: 32,
            downsampleFactor: [2],
            preModule: FishS1Identity(),
            postModule: FishS1Identity()
        )

        let model = FishS1DAC(
            encoderDim: 4,
            encoderRates: [2, 2],
            latentDim: 16,
            decoderDim: 16,
            decoderRates: [2, 2],
            quantizer: quantizer,
            sampleRate: 44_100,
            causal: true,
            encoderTransformerLayers: [0, 0],
            decoderTransformerLayers: [0, 0],
            transformerConfigFactory: nil
        )

        let audio = MLXArray.zeros([1, 128, 1], dtype: .float32)
        let (indices, featureLengths) = model.encode(audio)

        #expect(indices.shape[0] == 1)
        #expect(indices.shape[1] == 3)
        #expect(featureLengths.shape == [1])
        #expect(featureLengths.item(Int32.self) == 8)

        let (decoded, decodedLengths) = model.decode(indices, featureLengths: featureLengths)
        #expect(decoded.shape == [1, 1, 128])
        #expect(decodedLengths.item(Int32.self) == 128)

        let zQ = model.encodeZQ(audio)
        #expect(zQ.shape == [1, 16, 16])

        let reconstructed = model.decodeZQ(zQ)
        #expect(reconstructed.shape == [1, 1, 128])
    }

    @Test func testAudioCodecModelUsesPublicWaveformLayout() throws {
        let quantizer = FishS1DownsampleResidualVectorQuantize(
            inputDim: 16,
            nCodebooks: 2,
            codebookDim: 4,
            codebookSize: 16,
            semanticCodebookSize: 32,
            downsampleFactor: [2],
            preModule: FishS1Identity(),
            postModule: FishS1Identity()
        )

        let model = FishS1DAC(
            encoderDim: 4,
            encoderRates: [2, 2],
            latentDim: 16,
            decoderDim: 16,
            decoderRates: [2, 2],
            quantizer: quantizer,
            sampleRate: 44_100,
            causal: true,
            encoderTransformerLayers: [0, 0],
            decoderTransformerLayers: [0, 0],
            transformerConfigFactory: nil
        )

        let waveform = MLXArray.zeros([1, 128, 1], dtype: .float32)
        let encoded = model.encodeAudio(waveform)
        let decoded = model.decodeAudio(encoded)

        #expect(encoded.codes.shape[0] == 1)
        #expect(encoded.originalLength == 128)
        #expect(decoded.shape == [1, 128, 1])
    }
}


// MARK: - Codec Network Tests

@Suite("Codec Network Tests", .serialized)
struct CodecNetworkTests {

    @Test func s3TokenizerFromPretrainedMatchesPythonReferenceTokens() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network S3Tokenizer test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_S3_TOKENIZER_REPO"] ?? S3TokenizerV2.defaultRepository
        let model = try await S3TokenizerV2.fromPretrained(repo)
        let audio = MLXArray(moduloFloatFixtureValues(count: 16_000, modulus: 97, subtracting: 48, divisor: 480.0))
        let mel = s3TokenizerLogMelSpectrogram(audio)
        let melLen = MLXArray([Int32(mel.dim(1))])

        let (tokens, tokenLens) = model.quantize(mel.expandedDimensions(axis: 0), melLen: melLen)
        eval(tokens, tokenLens)

        let expectedTokens: [Int32] = [
            365, 4252, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255,
            4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255, 4255,
            4254, 1092,
        ]

        #expect(tokenLens.asArray(Int32.self) == [Int32(expectedTokens.count)])
        #expect(tokens[0, 0..<expectedTokens.count].asArray(Int32.self) == expectedTokens)
    }

    @Test func mossAudioTokenizerFromPretrainedMatchesPythonReferenceCodes() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network Moss audio tokenizer test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_MOSS_AUDIO_TOKENIZER_REPO"] ?? mossDefaultAudioTokenizerRepo
        let model = try await MLXMossAudioTokenizer.fromPretrained(repo)
        let samples = (0..<3_840).flatMap { index -> [Float] in
            [
                Float((index % 97) - 48) / 960.0,
                Float((index % 89) - 44) / 880.0,
            ]
        }
        let audio = MLXArray(samples, [3_840, 2])

        let codes = try model.encodeAudio(audio, numQuantizers: 4)
        eval(codes)

        #expect(codes.shape == [1, 4])
        #expect(codes.asArray(Int32.self) == [364, 162, 461, 742])
    }

    @Test func descriptDACFromPretrainedLoadsRealWeightsAndRoundTripsAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network Descript DAC test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_DESCRIPT_DAC_REPO"] ?? "mlx-community/descript-audio-codec-44khz"
        let model = try await DescriptDAC.fromPretrained(repo)
        let audio = try loadCodecNetworkFixture(sampleRate: model.sampleRate, maxSamples: model.sampleRate / 4)
        let waveform = audio.reshaped([1, audio.shape[0], 1])

        let encoded = model.encodeAudio(waveform)
        let decoded = model.decodeAudio(encoded)
        eval(encoded.codes, decoded)

        #expect(encoded.codes.shape[0] == 1)
        #expect(encoded.codes.shape[1] == model.config.nCodebooks)
        #expect(encoded.originalLength == waveform.shape[1])
        #expect(decoded.shape == waveform.shape)
    }
}
