// Ported from Python mlx-audio chatterbox s3gen/mel.py

import Foundation
import MLX
import MLXAudioCore

/// Reverse an MLXArray along a given axis by gathering with reversed indices.
private func reverseAlongAxis(_ x: MLXArray, axis: Int) -> MLXArray {
    let n = x.dim(axis)
    let indices = MLXArray(Array((0 ..< n).reversed()).map { Int32($0) })
    return x.take(indices, axis: axis)
}

/// Reflect-pad a 2D array (B, T) along axis 1.
func s3genReflectPad2D(_ x: MLXArray, padAmount: Int) -> MLXArray {
    if padAmount == 0 { return x }
    let T = x.dim(1)
    // Reflect at start: x[:, 1:pad+1] reversed
    let prefix = x[0..., 1 ..< min(padAmount + 1, T)]
    let prefixReversed = reverseAlongAxis(prefix, axis: 1)
    // Reflect at end: x[:, -(pad+1):-1] reversed
    let suffix = x[0..., max(0, T - padAmount - 1) ..< (T - 1)]
    let suffixReversed = reverseAlongAxis(suffix, axis: 1)
    return MLX.concatenated([prefixReversed, x, suffixReversed], axis: 1)
}

/// Extract mel-spectrogram from waveform for S3Gen.
///
/// Uses MLXAudioCore DSP functions for STFT and mel filterbank.
///
/// - Parameters:
///   - y: Waveform (B, T) or (T,)
///   - nFft: FFT size (default: 1920)
///   - numMels: Number of mel bins (default: 80)
///   - samplingRate: Sample rate (default: 24000)
///   - hopSize: Hop size (default: 480)
///   - winSize: Window size (default: 1920)
///   - fmin: Minimum frequency (default: 0)
///   - fmax: Maximum frequency (default: 8000)
/// - Returns: Mel-spectrogram (B, numMels, T')
func s3genMelSpectrogram(
    y: MLXArray,
    nFft: Int = 1920, numMels: Int = 80,
    samplingRate: Int = 24000, hopSize: Int = 480,
    winSize: Int = 1920, fmin: Int = 0, fmax: Int = 8000
) -> MLXArray {
    var input = y
    let was1D = input.ndim == 1
    if was1D {
        input = input.expandedDimensions(axis: 0)
    }

    let B = input.dim(0)

    // Reflect pad
    let padAmount = (nFft - hopSize) / 2
    input = s3genReflectPad2D(input, padAmount: padAmount)

    // STFT per batch item — center=false because we already applied reflect padding above
    let window = hanningWindow(size: winSize)
    var specs: [MLXArray] = []
    for i in 0 ..< B {
        let spec = stft(
            audio: input[i], window: window,
            nFft: nFft, hopLength: hopSize,
            center: false)
        specs.append(spec)
    }
    // Stack: each spec is (T', F) -> (B, T', F)
    var spec = MLX.stacked(specs, axis: 0)

    // Magnitude
    let magnitudes = abs(spec) // (B, T', F)

    // Mel filterbank
    let filters = melFilters(
        sampleRate: samplingRate, nFft: nFft, nMels: numMels,
        fMin: Float(fmin), fMax: Float(fmax),
        norm: "slaney", melScale: .slaney)

    // Apply: (B, T', F) @ (F, M) -> (B, T', M)
    // melFilters returns (nFreqs, nMels) = (F, M) — already correct for matmul
    var melSpec = matmul(magnitudes, filters)
    melSpec = melSpec.transposed(0, 2, 1) // (B, M, T')

    // Log compression
    melSpec = MLX.log(MLX.maximum(melSpec, MLXArray(Float(1e-5))))

    return melSpec
}
