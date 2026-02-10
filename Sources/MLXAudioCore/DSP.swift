//
//  MelSpectrogram.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX

// MARK: - Mel Spectrogram Computation


/// Create a Hanning window of given size.
public func hanningWindow(size: Int) -> MLXArray {
    var window = [Float](repeating: 0, count: size)
    let denom = Float(size - 1)
    for n in 0..<size {
        window[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
    }
    return MLXArray(window)
}

/// Create mel filterbank matrix.
public func melFilters(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = "slaney"
) -> MLXArray {
    let fMaxVal = fMax ?? Float(sampleRate) / 2.0

    // Hz to mel conversion (HTK formula)
    func hzToMel(_ freq: Float) -> Float {
        return 2595.0 * log10(1.0 + freq / 700.0)
    }

    // Mel to Hz conversion
    func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    let nFreqs = nFft / 2 + 1

    // Generate frequency points
    var allFreqs = [Float](repeating: 0, count: nFreqs)
    for i in 0..<nFreqs {
        allFreqs[i] = Float(i) * Float(sampleRate) / Float(nFft)
    }

    // Convert to mel scale and back
    let mMin = hzToMel(fMin)
    let mMax = hzToMel(fMaxVal)

    var mPts = [Float](repeating: 0, count: nMels + 2)
    for i in 0..<(nMels + 2) {
        mPts[i] = mMin + Float(i) * (mMax - mMin) / Float(nMels + 1)
    }

    let fPts = mPts.map { melToHz($0) }

    // Compute filterbank
    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nMels), count: nFreqs)

    for i in 0..<nFreqs {
        for j in 0..<nMels {
            let low = fPts[j]
            let center = fPts[j + 1]
            let high = fPts[j + 2]

            if allFreqs[i] >= low && allFreqs[i] < center {
                filterbank[i][j] = (allFreqs[i] - low) / (center - low)
            } else if allFreqs[i] >= center && allFreqs[i] <= high {
                filterbank[i][j] = (high - allFreqs[i]) / (high - center)
            }
        }
    }

    // Apply slaney normalization
    if norm == "slaney" {
        for j in 0..<nMels {
            let enorm = 2.0 / (fPts[j + 2] - fPts[j])
            for i in 0..<nFreqs {
                filterbank[i][j] *= enorm
            }
        }
    }

    // Convert to MLXArray [nFreqs, nMels]
    let flatFilters = filterbank.flatMap { $0 }
    return MLXArray(flatFilters).reshaped([nFreqs, nMels])
}

/// Reverse an array along the first axis using slicing.
private func reverseArray(_ arr: MLXArray) -> MLXArray {
    let len = arr.shape[0]
    var indices = [Int](repeating: 0, count: len)
    for i in 0..<len {
        indices[i] = len - 1 - i
    }
    return arr[MLXArray(indices.map { Int32($0) })]
}

/// Short-time Fourier Transform.
public func stft(
    audio: MLXArray,
    window: MLXArray,
    nFft: Int,
    hopLength: Int
) -> MLXArray {
    // Pad audio for centering
    let padding = nFft / 2
    let audioLen = audio.shape[0]

    // Reflect padding: reverse slices at both ends
    let prefixSlice = audio[1..<(min(padding + 1, audioLen))]
    let prefix = reverseArray(prefixSlice)

    let suffixStart = max(0, audioLen - padding - 1)
    let suffixEnd = max(1, audioLen - 1)
    let suffixSlice = audio[suffixStart..<suffixEnd]
    let suffix = reverseArray(suffixSlice)

    let padded = MLX.concatenated([prefix, audio, suffix])

    // Calculate number of frames
    let paddedLen = padded.shape[0]
    let numFrames = 1 + (paddedLen - nFft) / hopLength

    // Create frames using strided view — zero-copy, no per-frame loop
    let framesStacked = asStrided(padded, [numFrames, nFft], strides: [hopLength, 1], offset: 0)

    // Apply window
    let windowed = framesStacked * window

    // Compute FFT (real FFT)
    let fft = MLXFFT.rfft(windowed, axis: 1)  // [numFrames, nFft/2 + 1]

    return fft
}

/// Compute mel spectrogram from audio waveform.
public func computeMelSpectrogram(
    audio: MLXArray,
    sampleRate: Int,
    nFft: Int,
    hopLength: Int,
    nMels: Int
) -> MLXArray {
    // If audio is 1D, compute proper mel spectrogram
    if audio.ndim == 1 {
        // Create Hanning window
        let window = hanningWindow(size: nFft)

        // Compute STFT
        let freqs = stft(audio: audio, window: window, nFft: nFft, hopLength: hopLength)

        // Compute magnitude squared (power spectrum)
        let magnitudes = MLX.abs(freqs).square()
        eval(magnitudes)  // Allow MLX to free STFT complex output

        // Create mel filterbank [nFreqs, nMels]
        let filters = melFilters(
            sampleRate: sampleRate,
            nFft: nFft,
            nMels: nMels,
            norm: "slaney"
        )

        // Apply mel filterbank: [numFrames, nFreqs] @ [nFreqs, nMels] = [numFrames, nMels]
        var melSpec = MLX.matmul(magnitudes, filters)
        eval(melSpec)  // Allow MLX to free magnitudes and filters

        // Apply log scaling with clamping (Whisper-style normalization)
        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        melSpec = MLX.log10(melSpec)
        let maxVal = melSpec.max()
        melSpec = MLX.maximum(melSpec, maxVal - MLXArray(Float(8.0)))
        melSpec = (melSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

        return melSpec
    }

    // If already 2D, assume it's a spectrogram
    return audio
}
