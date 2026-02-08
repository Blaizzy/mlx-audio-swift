import Foundation
import Testing
import MLX
import MLXNN

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

    @Test func computeDeltasKaldiShape() {
        let features = MLXArray.ones([10, 60])
        let deltas = MossFormer2DSP.computeDeltasKaldi(features)

        #expect(deltas.shape == [10, 60])
    }

    @Test func computeDeltasKaldiNumerical() {
        // Regression test: verify numerical values match CPU implementation
        // Input: [1, 2, 3, 4, 5] with winLength=5 (halfWin=2)
        // denom = 2 * (1^2 + 2^2) = 10
        // delta[t] = sum(i * (feat[t+i] - feat[t-i]) for i in 1..2) / 10
        let input = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0]).reshaped([1, 5])
        let deltas = MossFormer2DSP.computeDeltasKaldi(input)

        // Expected values (hand-calculated):
        // t=0: (1*(2-1) + 2*(3-1)) / 10 = 5/10 = 0.5
        // t=1: (1*(3-1) + 2*(4-1)) / 10 = 8/10 = 0.8
        // t=2: (1*(4-2) + 2*(5-3)) / 10 = 6/10 = 0.6
        // t=3: (1*(5-3) + 2*(5-4)) / 10 = 4/10 = 0.4
        // t=4: (1*(5-4) + 2*(5-5)) / 10 = 1/10 = 0.1
        let expected: [Float] = [0.5, 0.8, 0.6, 0.4, 0.1]
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

    @Test func mossFormer2SEEnhance() async throws {
        let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
        let (_, audioData) = try loadAudioArray(from: audioURL)

        let model = try await MossFormer2SEModel.fromPretrained()
        let enhanced = try model.enhance(audioData)

        #expect(enhanced.ndim == 1)
        #expect(enhanced.shape[0] > 0)
    }
}
