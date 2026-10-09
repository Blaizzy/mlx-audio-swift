import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

@Suite(.serialized)
struct Qwen3ASREncoderTests {
    private func smallConfig() -> Qwen3ASRConfig {
        Qwen3ASRConfig(
            audioConfig: Qwen3AudioEncoderConfig(
                numMelBins: 16,
                encoderLayers: 1,
                encoderAttentionHeads: 2,
                encoderFfnDim: 32,
                dModel: 16,
                outputDim: 16,
                downsampleHiddenSize: 4
            ),
            textConfig: Qwen3TextConfig(
                vocabSize: 8,
                hiddenSize: 16,
                intermediateSize: 32,
                numHiddenLayers: 1,
                numAttentionHeads: 2,
                numKeyValueHeads: 1,
                headDim: 8
            ),
            audioTokenId: 2,
            audioStartTokenId: 1,
            audioEndTokenId: 3,
            classifyNum: 4
        )
    }

    private func melFrames(_ count: Int) -> MLXArray {
        MLXArray((0..<(count * 16)).map { sin(Float($0) * 0.01) }, [count, 16])
    }

    @Test func outputLengthsIncludeOnlyCompleteChunksAndValidTailRows() {
        let lengths: [Int32] = [0, 1, 7, 8, 9, 25, 50, 75, 99, 100, 101, 125, 150, 199, 200, 300, 750, 800]
        let expected: [Int32] = [0, 1, 1, 1, 2, 4, 7, 10, 13, 13, 14, 17, 20, 26, 26, 39, 98, 104]
        let actual = getFeatExtractOutputLengths(MLXArray(lengths))

        #expect(actual.dtype == .int32)
        #expect(actual.asArray(Int32.self) == expected)
    }

    // Two or more output rows avoid GEMV/GEMM rounding differences.
    @Test(arguments: [9, 25, 50, 75, 99])
    func shortClipEncodingIsIndependentOfBatchPadding(frameCount: Int) throws {
        let encoder = Qwen3ASRAudioEncoder(smallConfig().audioConfig)
        let shortMel = melFrames(frameCount)
        let shortFeatures = shortMel.transposed(1, 0)
        let paddedShort = MLX.padded(
            shortFeatures, widths: [IntOrPair((0, 0)), IntOrPair((0, 100 - frameCount))]
        )
        // A full chunk forces reference-width padding.
        let batchFeatures = MLX.stacked([paddedShort, melFrames(100).transposed(1, 0)])
        let mask = MLXArray(
            Array(repeating: Int32(1), count: frameCount)
                + Array(repeating: Int32(0), count: 100 - frameCount)
                + Array(repeating: Int32(1), count: 100),
            [2, 100]
        )
        let batched = encoder(batchFeatures, featureAttentionMask: mask)
        let standalone = encoder(shortFeatures.expandedDimensions(axis: 0))
        let streamed = encoder.encodeSingleWindow(shortMel)
        eval(batched, standalone, streamed)

        let validRows = (frameCount + 7) / 8
        try #require(batched.dim(0) == validRows + 13)
        try #require(standalone.dim(0) == validRows)
        try #require(streamed.dim(0) == validRows)
        let expected = batched[0..<validRows]
        #expect(MLX.abs(standalone - expected).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(streamed - expected).max().item(Float.self) < 1e-5)
    }

    @Test(arguments: [1, 8, 50, 100, 150, 750, 800])
    func promptCountsMatchEncoderRowsAndReplaceEveryPlaceholder(frameCount: Int) throws {
        let config = smallConfig()
        let model = Qwen3ASRModel(config)
        let aligner = Qwen3ForcedAlignerModel(config)
        let mel = melFrames(frameCount)
        let (features, mask, numAudioTokens) = model.preprocessAudio(mel)
        let (_, _, numAlignerTokens) = aligner.preprocessAudio(mel)
        let encoded = model.getAudioFeatures(features, featureAttentionMask: mask)
        let streamed = model.audioTower.encodeSingleWindow(mel)
        eval(encoded, streamed)

        try #require(numAudioTokens == encoded.dim(0))
        #expect(numAlignerTokens == numAudioTokens)
        try #require(streamed.shape == encoded.shape)
        #expect(MLX.abs(streamed - encoded).max().item(Float.self) < 1e-5)

        let inputIds = MLXArray(
            [Int32(config.audioStartTokenId)]
                + Array(repeating: Int32(config.audioTokenId), count: numAudioTokens)
                + [Int32(config.audioEndTokenId)],
            [1, numAudioTokens + 2]
        )
        let sentinel: Float = 9999
        let embeddings = MLX.ones([1, numAudioTokens + 2, config.textConfig.hiddenSize]) * sentinel
        let merged = model.mergeAudioFeatures(
            inputsEmbeds: embeddings, audioFeatures: encoded, inputIds: inputIds
        )

        #expect(MLX.abs(merged[0, 1..<(numAudioTokens + 1)] - encoded).max().item(Float.self) == 0)
        #expect(MLX.abs(merged[0, 0] - sentinel).max().item(Float.self) == 0)
        #expect(MLX.abs(merged[0, numAudioTokens + 1] - sentinel).max().item(Float.self) == 0)
    }
}
