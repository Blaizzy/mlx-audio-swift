//
//  MLXAudioSTSTests.swift
//  MLXAudioTests
//
//  Created by Claude on 17/02/2026.
//

import Foundation
import Testing
import MLX
import MLXNN
import MLXAudioCodecs

@testable import MLXAudioCore
@testable import MLXAudioSTS

struct MossFormer2SEConfigTests {

    @Test func mossFormer2SEConfigDefaults() {
        let config = MossFormer2SEConfig()

        #expect(config.modelType == "mossformer2_se")
        #expect(config.sampleRate == 48000)
        #expect(config.winLen == 1920)
        #expect(config.winInc == 384)
        #expect(config.fftLen == 1920)
        #expect(config.numMels == 60)
        #expect(config.winType == "hamming")
        #expect(abs(config.preemphasis - 0.97) < 1e-6)
        #expect(config.inChannels == 180)
        #expect(config.outChannels == 512)
        #expect(config.outChannelsFinal == 961)
        #expect(config.numBlocks == 24)
    }

    @Test func mossFormer2SEConfigDecoding() throws {
        let json = """
        {
            "model_type": "mossformer2_se",
            "sample_rate": 16000,
            "win_len": 512,
            "win_inc": 160,
            "fft_len": 512,
            "num_mels": 80,
            "win_type": "hann",
            "preemphasis": 0.95,
            "in_channels": 240,
            "out_channels": 256,
            "out_channels_final": 257,
            "num_blocks": 6
        }
        """

        let config = try JSONDecoder().decode(
            MossFormer2SEConfig.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "mossformer2_se")
        #expect(config.sampleRate == 16000)
        #expect(config.winLen == 512)
        #expect(config.winInc == 160)
        #expect(config.fftLen == 512)
        #expect(config.numMels == 80)
        #expect(config.winType == "hann")
        #expect(abs(config.preemphasis - 0.95) < 1e-6)
        #expect(config.inChannels == 240)
        #expect(config.outChannels == 256)
        #expect(config.outChannelsFinal == 257)
        #expect(config.numBlocks == 6)
    }

    @Test func mossFormer2SEConfigDecodingDefaults() throws {
        let config = try JSONDecoder().decode(
            MossFormer2SEConfig.self,
            from: Data("{}".utf8)
        )

        #expect(config.modelType == "mossformer2_se")
        #expect(config.sampleRate == 48000)
        #expect(config.winLen == 1920)
        #expect(config.winInc == 384)
        #expect(config.fftLen == 1920)
        #expect(config.numMels == 60)
        #expect(config.winType == "hamming")
        #expect(abs(config.preemphasis - 0.97) < 1e-6)
        #expect(config.inChannels == 180)
        #expect(config.outChannels == 512)
        #expect(config.outChannelsFinal == 961)
        #expect(config.numBlocks == 24)
    }

    @Test func quantizationConfigDecoding() throws {
        let json = """
        {
            "bits": 8,
            "group_size": 128
        }
        """

        let quantization = try JSONDecoder().decode(
            QuantizationConfig.self,
            from: Data(json.utf8)
        )

        #expect(quantization.bits == 8)
        #expect(quantization.groupSize == 128)
    }

    @Test func quantizationConfigDecodingDefaults() throws {
        let quantization = try JSONDecoder().decode(
            QuantizationConfig.self,
            from: Data("{}".utf8)
        )

        #expect(quantization.bits == 4)
        #expect(quantization.groupSize == 64)
    }
}

struct MossFormer2SELayerTests {

    @Test func scaleNormShape() {
        let layer = ScaleNorm(dim: 64)
        let x = MLXArray.ones([2, 8, 64])
        let y = layer(x)

        #expect(y.shape == [2, 8, 64])
    }

    @Test func globalLayerNormShape() {
        let layer = GlobalLayerNorm(dim: 32, shape: 3)
        let x = MLXArray.ones([2, 32, 16])
        let y = layer(x)

        #expect(y.shape == [2, 32, 16])
    }

    @Test func cLayerNormShape() {
        let layer = CLayerNorm(normalizedShape: 64)
        let x = MLXArray.ones([2, 8, 64])
        let y = layer(x)

        #expect(y.shape == [2, 8, 64])
    }

    @Test func scaledSinuEmbeddingShape() {
        let layer = ScaledSinuEmbedding(dim: 64)
        let x = MLXArray.ones([1, 8, 64])
        let y = layer(x)

        #expect(y.shape == [8, 64])
    }

    @Test func offsetScaleShape() {
        let layer = OffsetScale(dim: 32, heads: 4)
        let x = MLXArray.ones([2, 8, 32])
        let outputs = layer(x)

        #expect(outputs.count == 4)
        for output in outputs {
            #expect(output.shape == [2, 8, 32])
        }
    }

    @Test func convModuleShape() {
        let layer = ConvModule(inChannels: 64)
        let x = MLXArray.ones([2, 8, 64])
        let y = layer(x)

        #expect(y.shape == [2, 8, 64])
    }

    @Test func ffConvMShape() {
        let layer = FFConvM(dimIn: 64, dimOut: 128, normType: "scalenorm")
        let x = MLXArray.ones([2, 8, 64])
        let y = layer(x)

        #expect(y.shape == [2, 8, 128])
    }

    @Test func preluShape() {
        let layer = PReLU()
        let x = MLXArray.ones([2, 8, 64])
        let y = layer(x)

        #expect(y.shape == [2, 8, 64])
    }

    @Test func gatedFSMNBlockShape() {
        let layer = GatedFSMNBlock(dim: 64, innerChannels: 32)
        let x = MLXArray.ones([2, 8, 64])
        let y = layer(x)

        #expect(y.shape == [2, 8, 64])
    }

    @Test func flashAttentionSimpleKernelShape() {
        let q = MLXArray.ones([1, 2, 4, 16])
        let k = MLXArray.ones([1, 2, 4, 16])
        let v = MLXArray.ones([1, 2, 4, 32])

        let y = FlashAttention.simpleKernel(q, k, v, groupSize: 4)

        #expect(y.shape == [1, 2, 4, 32])
    }
}

struct MossFormer2SEDSPTests {

    @Test func hammingWindowSize() {
        let periodic = MossFormer2DSP.hammingWindow(size: 100)
        let symmetric = MossFormer2DSP.hammingWindow(size: 100, periodic: false)

        #expect(periodic.shape == [100])
        #expect(symmetric.shape == [100])
    }

    @Test func hammingWindowValues() {
        let window = MossFormer2DSP.hammingWindow(size: 100, periodic: false).asArray(Float.self)

        #expect(abs(window[0] - 0.08) < 1e-3)
        #expect(window[50] > 0.99)
    }

    @Test func stftShape() {
        let signal = MLXArray(Array(repeating: Float(0.1), count: 1000))
        let window = MossFormer2DSP.hammingWindow(size: 256)

        let spec = MossFormer2DSP.stft(
            audio: signal,
            fftLen: 256,
            hopLength: 128,
            winLen: 256,
            window: window,
            center: true
        )

        #expect(spec.ndim == 2)
        #expect(spec.shape[1] == 129)
        #expect(spec.shape[0] > 0)
    }

    @Test func istftRoundTrip() {
        let signal = MLXArray(Array(repeating: Float(0.25), count: 1024))
        let window = MossFormer2DSP.hammingWindow(size: 256)

        let spec = MossFormer2DSP.stft(
            audio: signal,
            fftLen: 256,
            hopLength: 128,
            winLen: 256,
            window: window,
            center: true
        )

        let real = spec.realPart().transposed(1, 0).expandedDimensions(axis: 0)
        let imag = spec.imaginaryPart().transposed(1, 0).expandedDimensions(axis: 0)

        let reconstructed = MossFormer2DSP.istft(
            real: real,
            imag: imag,
            fftLen: 256,
            hopLength: 128,
            winLen: 256,
            window: window,
            center: true,
            audioLength: signal.shape[0]
        )

        #expect(reconstructed.ndim == 1)
        #expect(reconstructed.shape[0] > 0)
    }

    @Test func computeFbankKaldiShape() {
        let signal = MLXArray(Array(repeating: Float(0.1), count: 48000))

        let fbank = MossFormer2DSP.computeFbankKaldi(
            audio: signal,
            sampleRate: 48000,
            winLen: 1920,
            winInc: 384,
            numMels: 60,
            winType: "hamming",
            preemphasis: 0.97
        )

        #expect(fbank.ndim == 2)
        #expect(fbank.shape[0] > 0)
        #expect(fbank.shape[1] == 60)
    }

    @Test func computeFbankKaldiDitherAffectsFeatures() {
        let signal = MLXArray(Array(repeating: Float(0.05), count: 9600))

        let clean = MossFormer2DSP.computeFbankKaldi(
            audio: signal,
            sampleRate: 48000,
            winLen: 1920,
            winInc: 384,
            numMels: 60,
            winType: "hamming",
            preemphasis: 0.97,
            dither: 0.0,
            removeDCOffset: true,
            roundToPowerOfTwo: true,
            lowFreq: 20.0
        )

        let dithered = MossFormer2DSP.computeFbankKaldi(
            audio: signal,
            sampleRate: 48000,
            winLen: 1920,
            winInc: 384,
            numMels: 60,
            winType: "hamming",
            preemphasis: 0.97,
            dither: 1.0,
            removeDCOffset: true,
            roundToPowerOfTwo: true,
            lowFreq: 20.0
        )

        #expect(clean.shape == dithered.shape)
        let meanAbsDiff = MLX.mean(MLX.abs(clean - dithered)).item(Float.self)
        #expect(meanAbsDiff > 0)
    }

    @Test func computeFbankKaldiRoundToPowerOfTwoChangesFeatures() {
        let sampleRate = Float(48000)
        let signal = MLXArray(Array(stride(from: Float(0), to: Float(0.3), by: 1.0 / sampleRate).map { sin(2 * .pi * 440 * $0) }))

        let rounded = MossFormer2DSP.computeFbankKaldi(
            audio: signal,
            sampleRate: 48000,
            winLen: 1920,
            winInc: 384,
            numMels: 60,
            winType: "hamming",
            preemphasis: 0.97,
            dither: 0.0,
            removeDCOffset: true,
            roundToPowerOfTwo: true,
            lowFreq: 20.0
        )

        let fixedNfft = MossFormer2DSP.computeFbankKaldi(
            audio: signal,
            sampleRate: 48000,
            winLen: 1920,
            winInc: 384,
            numMels: 60,
            winType: "hamming",
            preemphasis: 0.97,
            dither: 0.0,
            removeDCOffset: true,
            roundToPowerOfTwo: false,
            lowFreq: 20.0
        )

        #expect(rounded.shape == fixedNfft.shape)
        let meanAbsDiff = MLX.mean(MLX.abs(rounded - fixedNfft)).item(Float.self)
        #expect(meanAbsDiff > 0)
    }

    @Test func computeDeltasKaldiShape() {
        let features = MLXArray.ones([10, 60])
        let deltas = MossFormer2DSP.computeDeltasKaldi(features)

        #expect(deltas.shape == [10, 60])
    }

    @Test func computeDeltasKaldiNumerical() {
        // Regression test: verify numerical values match current implementation
        // Input: [1, 2, 3, 4, 5] with winLength=5 (halfWin=2)
        // denom = 2 * (1^2 + 2^2) = 10
        // delta[t] = sum(i * (feat[t+i] - feat[t-i]) for i in 1..2) / 10
        let input = MLXArray([1.0 as Float, 2.0, 3.0, 4.0, 5.0]).reshaped([1, 5])
        let deltas = MossFormer2DSP.computeDeltasKaldi(input)

        // Expected values (from current implementation):
        // t=0: 0.5
        // t=1: 0.8
        // t=2: 1.0
        // t=3: 0.8
        // t=4: 0.5
        let expected: [Float] = [0.5, 0.8, 1.0, 0.8, 0.5]
        let deltasArray = deltas.asArray(Float.self)

        let epsilon: Float = 1e-5
        for (i, exp) in expected.enumerated() {
            #expect(abs(deltasArray[i] - exp) < epsilon, "delta[\(i)]: expected \(exp), got \(deltasArray[i])")
        }
    }

    @Test func melFilterbankShape() {
        let bank = MossFormer2DSP.melFilterbank(sampleRate: 48000, nFft: 256, numMels: 60)

        #expect(bank.shape == [129, 60])
    }

    @Test func istftNumericalAccuracy() {
        // Generate known signal: 440 Hz sine wave
        let sampleRate = Float(4800)
        let duration = Float(1.0)
        let signal = MLXArray(Array(stride(from: Float(0), to: duration, by: 1.0 / sampleRate).map { sin(2 * .pi * 440 * $0) }))
        
        let fftLen = 256
        let hopLength = 128
        let winLen = 256
        let window = MossFormer2DSP.hammingWindow(size: winLen, periodic: false)
        
        // STFT → ISTFT round trip
        let stftResult = MossFormer2DSP.stft(
            audio: signal,
            fftLen: fftLen,
            hopLength: hopLength,
            winLen: winLen,
            window: window,
            center: true
        )
        
        let real = stftResult.realPart().transposed(1, 0).expandedDimensions(axis: 0)
        let imag = stftResult.imaginaryPart().transposed(1, 0).expandedDimensions(axis: 0)
        
        let reconstructed = MossFormer2DSP.istft(
            real: real,
            imag: imag,
            fftLen: fftLen,
            hopLength: hopLength,
            winLen: winLen,
            window: window,
            center: true,
            audioLength: signal.shape[0]
        )
        
        // Check numerical accuracy (exclude boundary regions)
        let margin = winLen
        let interior = reconstructed[margin..<(reconstructed.shape[0] - margin)]
        let reference = signal[margin..<(signal.shape[0] - margin)]
        let maxError = MLX.max(MLX.abs(interior - reference)).item(Float.self)
        
        // Hamming window is not perfect reconstruction, allow tolerance
        #expect(maxError < 0.05)
    }

    @Test func istftRejectsBatchGreaterThanOne() {
        let real = MLXArray.zeros([2, 129, 10])  // batch=2
        let imag = MLXArray.zeros([2, 129, 10])
        let window = MossFormer2DSP.hammingWindow(size: 256, periodic: false)
        let result = MossFormer2DSP.istft(
            real: real,
            imag: imag,
            fftLen: 256,
            hopLength: 128,
            winLen: 256,
            window: window
        )
        
        // Empty array = rejected
        #expect(result.shape[0] == 0)
    }

    @Test func stftCapturesTrailingSamples() {
        // Signal length NOT aligned to hop
        let signal = MLXArray(Array(repeating: Float(0.5), count: 1000))
        let window = MossFormer2DSP.hammingWindow(size: 256, periodic: false)
        let stftOut = MossFormer2DSP.stft(
            audio: signal,
            fftLen: 256,
            hopLength: 128,
            winLen: 256,
            window: window,
            center: false
        )
        
        // ceil((1000 - 256) / 128) + 1 = 7 frames (was 6 with floor)
        #expect(stftOut.shape[0] >= 7)
    }

    @Test func enhanceRejectsInvalidShape() async throws {
        let config = MossFormer2SEConfig()
        let model = MossFormer2SE(config: config)
        let stsModel = MossFormer2SEModel(model: model, config: config)
        let badInput = MLXArray.zeros([2, 100])  // 2D = invalid
        
        do {
            _ = try stsModel.enhance(badInput)
            Issue.record("Should have thrown for invalid shape")
        } catch {
            // Expected
        }
    }
}

struct MossFormer2SEModelTests {

    private func smallConfig() throws -> MossFormer2SEConfig {
        let json = """
        {
            "num_blocks": 1,
            "in_channels": 16,
            "out_channels": 32,
            "out_channels_final": 17
        }
        """
        return try JSONDecoder().decode(MossFormer2SEConfig.self, from: Data(json.utf8))
    }

    @Test func mossFormer2SEForwardShape() throws {
        let config = try smallConfig()
        let model = MossFormer2SE(config: config)
        let input = MLXArray.ones([1, 8, 16])

        let output = model(input)

        #expect(!output.isEmpty)
        #expect(output[0].ndim == 3)
        #expect(output[0].shape[0] == 1)
        #expect(output[0].shape[1] == 8)
        #expect(output[0].shape[2] == 17)
    }

    @Test func mossFormer2SEForwardBatchShape() throws {
        let config = try smallConfig()
        let model = MossFormer2SE(config: config)
        let input = MLXArray.ones([2, 6, 16])

        let output = model(input)

        #expect(!output.isEmpty)
        #expect(output[0].shape[0] == 2)
        #expect(output[0].shape[1] == 6)
        #expect(output[0].shape[2] == 17)
    }
}

struct MossFormer2SESanitizeTests {

    @Test func sanitizeStripModulePrefix() {
        let weights: [String: MLXArray] = [
            "module.mossformer.norm.weight": MLXArray.ones([16])
        ]

        let sanitized = MossFormer2SEModel.sanitize(weights: weights)

        #expect(sanitized["model.mossformer.norm.weight"] != nil)
        #expect(sanitized["module.mossformer.norm.weight"] == nil)
    }

    @Test func sanitizePassthrough() {
        let weights: [String: MLXArray] = [
            "model.mossformer.norm.weight": MLXArray.ones([16])
        ]

        let sanitized = MossFormer2SEModel.sanitize(weights: weights)

        #expect(sanitized["model.mossformer.norm.weight"] != nil)
        #expect(sanitized.count == 1)
    }

    @Test func sanitizeKeepsOtherKeys() {
        let weights: [String: MLXArray] = [
            "some.other.key": MLXArray.ones([8])
        ]

        let sanitized = MossFormer2SEModel.sanitize(weights: weights)

        #expect(sanitized["some.other.key"] != nil)
        #expect(sanitized.count == 1)
    }

    @Test func sanitizeMixedKeyCount() {
        let weights: [String: MLXArray] = [
            "module.mossformer.conv1d_encoder.weight": MLXArray.ones([16, 1, 16]),
            "model.mossformer.norm.weight": MLXArray.ones([16]),
            "some.other.key": MLXArray.ones([8]),
        ]

        let sanitized = MossFormer2SEModel.sanitize(weights: weights)

        #expect(sanitized.count == 3)
        #expect(sanitized["model.mossformer.conv1d_encoder.weight"] != nil)
        #expect(sanitized["model.mossformer.norm.weight"] != nil)
        #expect(sanitized["some.other.key"] != nil)
    }
}

struct MossFormer2SEIntegrationTests {

    @Test func fromLocalRejectsMissingSafetensors() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mossformer2se-empty-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        do {
            _ = try MossFormer2SEModel.fromLocal(tempDir)
            Issue.record("Expected fromLocal to throw when no .safetensors files are present")
        } catch let error as MossFormer2SEError {
            switch error {
            case .missingSafetensors(let directory):
                #expect(directory.path == tempDir.path)
            default:
                Issue.record("Expected missingSafetensors, got \(error)")
            }
        } catch {
            Issue.record("Expected MossFormer2SEError, got \(error)")
        }
    }

    @Test func fromLocalRejectsNormalizedDuplicateWeightKeys() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mossformer2se-dup-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let first = tempDir.appendingPathComponent("a.safetensors")
        let second = tempDir.appendingPathComponent("b.safetensors")

        try MLX.save(
            arrays: ["module.mossformer.norm.weight": MLXArray.ones([1])],
            url: first
        )
        try MLX.save(
            arrays: ["mossformer.norm.weight": MLXArray.ones([1])],
            url: second
        )

        do {
            _ = try MossFormer2SEModel.fromLocal(tempDir)
            Issue.record("Expected fromLocal to throw on normalized duplicate weight keys")
        } catch let error as MossFormer2SEError {
            switch error {
            case .duplicateWeightKey(let key):
                #expect(key == "model.mossformer.norm.weight")
            default:
                Issue.record("Expected duplicateWeightKey, got \(error)")
            }
        } catch {
            Issue.record("Expected MossFormer2SEError, got \(error)")
        }
    }

    @Test func mossFormer2SEEnhance() async throws {
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (_, audioData) = try loadAudioArray(from: audioURL)

        let model = try await MossFormer2SEModel.fromPretrained()
        let enhanced = try model.enhance(audioData)

        #expect(enhanced.ndim == 1)
        #expect(enhanced.shape[0] > 0)
    }
}

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


// MARK: - LFMAudio Config Tests

struct LFMAudioConfigTests {

    // MARK: - PreprocessorConfig

    @Test func preprocessorConfigDefaults() {
        let config = PreprocessorConfig()

        #expect(config.sampleRate == 16000)
        #expect(config.normalize == "per_feature")
        #expect(config.windowSize == 0.025)
        #expect(config.windowStride == 0.01)
        #expect(config.window == "hann")
        #expect(config.features == 128)
        #expect(config.nFft == 512)
        #expect(config.log == true)
        #expect(config.frameSplicing == 1)
        #expect(config.dither == 1e-05)
        #expect(config.padTo == 0)
        #expect(config.padValue == 0.0)
        #expect(config.preemph == 0.97)
    }

    @Test func preprocessorConfigComputedProperties() {
        let config = PreprocessorConfig()

        // hopLength = Int(16000 * 0.01) = 160
        #expect(config.hopLength == 160)
        // winLength = Int(16000 * 0.025) = 400
        #expect(config.winLength == 400)
    }

    @Test func preprocessorConfigDecoding() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(PreprocessorConfig.self, from: data)

        #expect(config.sampleRate == 16000)
        #expect(config.features == 128)
        #expect(config.nFft == 512)
        #expect(config.preemph == 0.97)
    }

    // MARK: - ConformerEncoderConfig

    @Test func conformerEncoderConfigDefaults() {
        let config = ConformerEncoderConfig()

        #expect(config.featIn == 128)
        #expect(config.featOut == -1)
        #expect(config.nLayers == 17)
        #expect(config.dModel == 512)
        #expect(config.subsampling == "dw_striding")
        #expect(config.subsamplingFactor == 8)
        #expect(config.subsamplingConvChannels == 256)
        #expect(config.causalDownsampling == false)
        #expect(config.ffExpansionFactor == 4)
        #expect(config.selfAttentionModel == "rel_pos")
        #expect(config.nHeads == 8)
        #expect(config.attContextSize == [-1, -1])
        #expect(config.xscaling == false)
        #expect(config.untieBiases == true)
        #expect(config.posEmbMaxLen == 5000)
        #expect(config.convKernelSize == 9)
        #expect(config.convNormType == "batch_norm")
        #expect(config.dropout == 0.1)
    }

    @Test func conformerEncoderConfigDecoding() throws {
        let json = """
        {
            "feat_in": 80,
            "n_layers": 12,
            "d_model": 256,
            "n_heads": 4
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ConformerEncoderConfig.self, from: data)

        #expect(config.featIn == 80)
        #expect(config.nLayers == 12)
        #expect(config.dModel == 256)
        #expect(config.nHeads == 4)
        // Defaults for unspecified fields
        #expect(config.subsamplingFactor == 8)
        #expect(config.convKernelSize == 9)
    }

    // MARK: - DepthformerConfig

    @Test func depthformerConfigDefaults() {
        let config = DepthformerConfig()

        #expect(config.layers == 6)
        #expect(config.dim == 1024)
        #expect(config.numHeads == 32)
        #expect(config.numKvHeads == 8)
        #expect(config.tie == true)
    }

    @Test func depthformerConfigDecoding() throws {
        let json = """
        {
            "layers": 4,
            "dim": 512,
            "num_heads": 16,
            "num_kv_heads": 4,
            "tie": false
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(DepthformerConfig.self, from: data)

        #expect(config.layers == 4)
        #expect(config.dim == 512)
        #expect(config.numHeads == 16)
        #expect(config.numKvHeads == 4)
        #expect(config.tie == false)
    }

    // MARK: - DetokenizerConfig

    @Test func detokenizerConfigDefaults() {
        let config = DetokenizerConfig()

        #expect(config.hiddenSize == 512)
        #expect(config.numHiddenLayers == 8)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.slidingWindow == 30)
        #expect(config.intermediateSize == 2304)
        #expect(config.normEps == 1e-5)
        #expect(config.ropeTheta == 1000000.0)
        #expect(config.outputSize == 1282)
        #expect(config.numCodebooks == 8)
        #expect(config.vocabSize == 2048)
        #expect(config.nFft == 1280)
        #expect(config.hopLength == 320)
        #expect(config.upsampleFactor == 6)
    }

    @Test func detokenizerConfigLayerTypes() {
        let config = DetokenizerConfig()

        #expect(config.layerTypes.count == 8)
        // Pattern: conv, conv, sliding_attention, conv, sliding_attention, conv, sliding_attention, conv
        #expect(config.layerTypes[0] == "conv")
        #expect(config.layerTypes[2] == "sliding_attention")
        #expect(config.layerTypes[7] == "conv")
    }

    // MARK: - LFMGenerationConfig

    @Test func generationConfigDefaults() {
        let config = LFMGenerationConfig()

        #expect(config.maxNewTokens == 512)
        #expect(config.temperature == 1.0)
        #expect(config.topK == 50)
        #expect(config.topP == 1.0)
        #expect(config.audioTemperature == 1.0)
        #expect(config.audioTopK == 4)
    }

    @Test func generationConfigCustom() {
        let config = LFMGenerationConfig(
            maxNewTokens: 2048,
            temperature: 0.8,
            topK: 30,
            audioTemperature: 0.7,
            audioTopK: 10
        )

        #expect(config.maxNewTokens == 2048)
        #expect(config.temperature == 0.8)
        #expect(config.topK == 30)
        #expect(config.audioTemperature == 0.7)
        #expect(config.audioTopK == 10)
    }
}


// MARK: - Module Setup Tests

struct LFMAudioModuleSetupTests {

    @Test func modalityConstants() {
        #expect(LFMModality.text.rawValue == 1)
        #expect(LFMModality.audioIn.rawValue == 2)
        #expect(LFMModality.audioOut.rawValue == 3)
    }

    @Test func specialTokenConstants() {
        #expect(lfmAudioStartToken == 128)
        #expect(lfmImEndToken == 7)
        #expect(lfmTextEndToken == 130)
        #expect(lfmAudioEOSToken == 2048)
    }

    @Test func audioEmbeddingShape() {
        let vocabSize = 2049
        let dim = 64
        let numCodebooks = 8
        let emb = AudioEmbedding(vocabSize: vocabSize, dim: dim, numCodebooks: numCodebooks)

        // Input: (B, K) where K = numCodebooks, values in [0, vocabSize)
        let input = MLXArray([0, 1, 2, 3, 4, 5, 6, 7]).expandedDimensions(axis: 0) // (1, 8)
        let output = emb(input)

        // Output should be (1, dim) after summing over codebooks
        #expect(output.shape == [1, dim])
    }

    @Test func audioEmbeddingWithNormShape() {
        let vocabSize = 2049
        let dim = 64
        let emb = AudioEmbeddingWithNorm(vocabSize: vocabSize, dim: dim)

        // embed: (B,) -> (B, dim)
        let input = MLXArray([Int32(42)]).expandedDimensions(axis: 0) // (1, 1)
        let embedded = emb.embed(input.squeezed(axis: 1))
        #expect(embedded.shape == [1, dim])

        // logits: (B, dim) -> (B, vocabSize)
        let hidden = MLXArray.zeros([1, dim])
        let logits = emb.logits(hidden)
        #expect(logits.shape == [1, vocabSize])
    }

    @Test func conformerEncoderConstruction() {
        let config = ConformerEncoderConfig()
        let encoder = ConformerEncoder(config)

        // Verify the encoder was constructed (it has layers)
        #expect(encoder.layers.count == config.nLayers)
    }

    @Test func depthformerConstruction() {
        let config = DepthformerConfig()
        let depthformer = Depthformer(
            layers: config.layers, dim: config.dim,
            numHeads: config.numHeads, numKvHeads: config.numKvHeads
        )

        #expect(depthformer.layersCount == config.layers)
    }
}


