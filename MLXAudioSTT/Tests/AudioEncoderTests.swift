import Testing
import MLX
import MLXNN
@testable import MLXAudioSTT

struct AudioEncoderTests {
    @Test func outputShapeIsCorrect() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // Input: mel spectrogram [batch, nMels, nFrames]
        let mel = MLXArray.zeros([1, 128, 3000])

        let output = encoder(mel)

        // Output: [batch, nAudioCtx, nAudioState]
        // Conv2 with stride=2 halves frames: 3000 -> 1500
        #expect(output.shape == [1, 1500, 1280])
    }

    @Test func outputShapeWithShorterInput() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // Shorter input: 10 seconds of audio = 1000 frames
        let mel = MLXArray.zeros([1, 128, 1000])

        let output = encoder(mel)

        // After stride=2: 1000 -> 500
        #expect(output.shape == [1, 500, 1280])
    }

    @Test func conv1OutputDimension() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // Conv1: nMels -> nAudioState with kernel=3
        // MLXNN Conv1d weight shape: [outputChannels, kernelSize, inputChannels]
        #expect(encoder.conv1.weight.shape[0] == config.nAudioState)
        #expect(encoder.conv1.weight.shape[2] == config.nMels)
    }

    @Test func conv2OutputDimension() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // Conv2: nAudioState -> nAudioState with kernel=3, stride=2
        // MLXNN Conv1d weight shape: [outputChannels, kernelSize, inputChannels]
        #expect(encoder.conv2.weight.shape[0] == config.nAudioState)
        #expect(encoder.conv2.weight.shape[2] == config.nAudioState)
    }

    @Test func hasCorrectNumberOfBlocks() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        #expect(encoder.blocks.count == config.nAudioLayer)
    }

    @Test func blocksAreEncoderType() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // All encoder blocks should NOT have cross-attention
        for block in encoder.blocks {
            #expect(block.hasCrossAttention == false)
        }
    }

    @Test func positionalEmbeddingShape() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // Positional embedding: [nAudioCtx, nAudioState]
        #expect(encoder.positionalEmbedding.shape == [config.nAudioCtx, config.nAudioState])
    }

    @Test func layerNormDimension() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        #expect(encoder.ln.dimensions == config.nAudioState)
    }

    @Test func batchProcessing() {
        let config = WhisperConfiguration.largeTurbo
        let encoder = AudioEncoder(config: config)

        // Batch of 2
        let mel = MLXArray.zeros([2, 128, 3000])

        let output = encoder(mel)

        #expect(output.shape == [2, 1500, 1280])
    }

    @Test func smallConfigWorks() {
        // Test with a smaller config to verify flexibility
        let config = WhisperConfiguration(
            nMels: 80,
            nAudioCtx: 1500,
            nAudioState: 512,
            nAudioHead: 8,
            nAudioLayer: 6,
            nVocab: 51865,
            nTextCtx: 448,
            nTextState: 512,
            nTextHead: 8,
            nTextLayer: 6,
            alignmentHeads: [(3, 0), (4, 1)]
        )
        let encoder = AudioEncoder(config: config)

        let mel = MLXArray.zeros([1, 80, 3000])
        let output = encoder(mel)

        // After stride=2: 3000 -> 1500
        #expect(output.shape == [1, 1500, 512])
        #expect(encoder.blocks.count == 6)
    }
}
