import Testing
import MLX
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs
@testable import MLXAudioLID

// MARK: - Encodec Tests
// Run Encodec tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
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
