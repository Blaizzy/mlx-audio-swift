import Foundation
import MLXAudioCodecs
@preconcurrency import MLX

/// Revert delayed codebooks, keeping only complete frames before channel zero's EOS.
/// `generatedCodes` is [C, T], including the leading BOS frame.
func diaRevertCodebookDelay(
    _ generatedCodes: MLXArray, delayPattern: [Int], maxT: Int, channels: Int,
    eosValue: Int = 1024
) throws -> MLXArray {
    var codes = generatedCodes[0..., 1...]
    if codes.shape[1] > maxT { codes = codes[0..., 0..<maxT] }
    let seqLen = codes.shape[1]
    let maxDelay = delayPattern.max() ?? 0
    let completeFrames = max(0, seqLen - maxDelay)
    let eosIndex = codes[0].asArray(Int32.self).firstIndex(of: Int32(eosValue))
    let frameCount = min(completeFrames, eosIndex ?? completeFrames)
    guard frameCount > 0 else { throw DiaError.noAudio }

    let audioBTC = codes.transposed(1, 0).expandedDimensions(axis: 0)
    let base = MLXArray((0..<frameCount).map { Int32($0) }).reshaped([frameCount, 1])
    let delay = MLXArray(delayPattern.map { Int32($0) }).reshaped([1, channels])
    let tIdx = (base + delay).expandedDimensions(axis: 0)
    let reverted = takeAlong(audioBTC, tIdx, axis: 1)
    let codebook = reverted.transposed(0, 2, 1)
    let invalid = MLX.logicalOr(MLX.less(codebook, 0), MLX.greater(codebook, 1023))
    return MLX.where(invalid, MLXArray(Int32(0)), codebook)
}

func diaCodebookToAudio(
    _ generatedCodes: MLXArray, dac: DescriptDAC, delayPattern: [Int], maxT: Int, channels: Int,
    eosValue: Int = 1024
) throws -> MLXArray {
    let codebook = try diaRevertCodebookDelay(
        generatedCodes, delayPattern: delayPattern, maxT: maxT, channels: channels,
        eosValue: eosValue)
    return dac.decodeFromCodes(codebook)
}
