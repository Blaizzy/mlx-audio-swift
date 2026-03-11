import Testing
import MLX
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs
@testable import MLXAudioLID

// MARK: - DACVAE Tests
// Run DACVAE tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
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
