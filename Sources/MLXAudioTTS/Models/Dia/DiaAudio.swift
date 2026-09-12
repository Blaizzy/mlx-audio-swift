import Foundation
import MLXAudioCodecs
@preconcurrency import MLX

/// Reverts the codebook delay pattern and decodes to a waveform via DAC.
/// `generatedCodes` is [C, T] (leading BOS frame at t=0).
func diaCodebookToAudio(
    _ generatedCodes: MLXArray, dac: DescriptDAC, delayPattern: [Int], maxT: Int, channels: Int
) -> MLXArray {
    var codes = generatedCodes[0..., 1...]
    if codes.shape[1] > maxT { codes = codes[0..., 0 ..< maxT] }
    let seqLen = codes.shape[1]

    let audioBTC = codes.transposed(1, 0).expandedDimensions(axis: 0)

    let base = MLXArray((0 ..< seqLen).map { Int32($0) }).reshaped([seqLen, 1])
    let delay = MLXArray(delayPattern.map { Int32($0) }).reshaped([1, channels])
    let tIdx = clip(base + delay, max: Int32(seqLen - 1)).expandedDimensions(axis: 0)

    var reverted = takeAlong(audioBTC, tIdx, axis: 1)
    if reverted.shape[1] > 30 {
        reverted = reverted[0..., 0 ..< (seqLen - 30), 0...]
    }

    var codebook = reverted.transposed(0, 2, 1)
    let invalid = MLX.logicalOr(MLX.less(codebook, 0), MLX.greater(codebook, 1023))
    codebook = MLX.where(invalid, MLXArray(Int32(0)), codebook)

    return dac.decodeFromCodes(codebook)
}
