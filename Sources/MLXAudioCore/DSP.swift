//
//  MelSpectrogram.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import Accelerate

// MARK: - Mel Spectrogram Computation


/// Create a Hanning window of given size (symmetric, period N-1).
///
/// Uses the standard symmetric Hann window formula:
/// `w[n] = 0.5 * (1 - cos(2 * pi * n / (N-1)))` matching numpy.hanning(N).
public func hanningWindow(size: Int) -> MLXArray {
    var window = [Float](repeating: 0, count: size)
    let denom = Float(size - 1)
    for n in 0..<size {
        window[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
    }
    return MLXArray(window)
}

/// Create a Hanning window as a `[Float]` array (symmetric, period N-1).
///
/// Same formula as `hanningWindow(size:)` but returns raw `[Float]` for use
/// in the Accelerate pipeline without MLXArray wrapping overhead.
private func hanningWindowFloat(size: Int) -> [Float] {
    var window = [Float](repeating: 0, count: size)
    let denom = Float(size - 1)
    for n in 0..<size {
        window[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
    }
    return window
}

/// Create mel filterbank matrix as MLXArray `[nFreqs, nMels]`.
public func melFilters(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = "slaney"
) -> MLXArray {
    let flatFilters = melFiltersFlat(
        sampleRate: sampleRate,
        nFft: nFft,
        nMels: nMels,
        fMin: fMin,
        fMax: fMax,
        norm: norm
    )
    let nFreqs = nFft / 2 + 1
    return MLXArray(flatFilters).reshaped([nFreqs, nMels])
}

/// Create mel filterbank as a flat `[Float]` array in row-major order `[nFreqs, nMels]`.
///
/// Shared implementation used by both MLX and Accelerate paths to avoid
/// duplicate filterbank logic.
private func melFiltersFlat(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = "slaney"
) -> [Float] {
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

    return filterbank.flatMap { $0 }
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

/// Short-time Fourier Transform (MLX path).
///
/// Used as fallback when explicitly requested. The default mel spectrogram
/// pipeline uses the Accelerate path instead.
///
/// - Parameters:
///   - audio: 1D audio waveform
///   - window: Hanning window
///   - nFft: FFT size
///   - hopLength: Hop length between frames
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

    // Create frames
    var frames: [MLXArray] = []
    for i in 0..<numFrames {
        let start = i * hopLength
        let frame = padded[start..<(start + nFft)]
        frames.append(frame)
    }

    let framesStacked = MLX.stacked(frames, axis: 0)  // [numFrames, nFft]

    // Apply window
    let windowed = framesStacked * window

    // Compute FFT (real FFT)
    let fft = MLXFFT.rfft(windowed, axis: 1)  // [numFrames, nFft/2 + 1]

    return fft
}

// MARK: - Accelerate-Based DSP (vDSP + BLAS)

/// Perform reflect-padding, framing, windowing, and real FFT entirely in
/// Accelerate/vDSP, returning the power spectrum (magnitude squared) as a
/// flat `[Float]` array in row-major `[numFrames, nFreqs]` layout.
///
/// This avoids all MLXArray overhead (lazy eval, graph build, copy-back)
/// and uses Apple Silicon NEON SIMD via Accelerate.
private func stftPowerSpectrumAccelerate(
    samples: [Float],
    nFft: Int,
    hopLength: Int
) -> (powerSpectrum: [Float], numFrames: Int, nFreqs: Int) {
    let nFreqs = nFft / 2 + 1
    let padding = nFft / 2
    let audioLen = samples.count

    // --- Reflect padding (matches MLX stft padding exactly) ---
    let prefixEnd = min(padding + 1, audioLen)
    let prefixLen = prefixEnd - 1
    let suffixStart = max(0, audioLen - padding - 1)
    let suffixEnd = max(1, audioLen - 1)
    let suffixLen = suffixEnd - suffixStart

    let paddedLen = prefixLen + audioLen + suffixLen
    var padded = [Float](repeating: 0, count: paddedLen)

    // Build prefix: reverse of samples[1..<prefixEnd]
    for i in 0..<prefixLen {
        padded[i] = samples[prefixEnd - 1 - i]
    }

    // Copy original samples
    for i in 0..<audioLen {
        padded[prefixLen + i] = samples[i]
    }

    // Build suffix: reverse of samples[suffixStart..<suffixEnd]
    for i in 0..<suffixLen {
        padded[prefixLen + audioLen + i] = samples[suffixEnd - 1 - i]
    }

    let numFrames = 1 + (paddedLen - nFft) / hopLength

    // --- Hanning window (symmetric, period N-1) ---
    let window = hanningWindowFloat(size: nFft)

    // --- Setup vDSP FFT ---
    let log2n = vDSP_Length(log2(Double(nFft)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        fatalError("Failed to create vDSP FFT setup for log2n=\(log2n)")
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    // Output: power spectrum [numFrames * nFreqs]
    var powerSpectrum = [Float](repeating: 0, count: numFrames * nFreqs)

    // Temp buffers for split complex
    let halfN = nFft / 2
    var realPart = [Float](repeating: 0, count: halfN)
    var imagPart = [Float](repeating: 0, count: halfN)
    var windowedFrame = [Float](repeating: 0, count: nFft)

    for f in 0..<numFrames {
        let start = f * hopLength

        // Extract frame and apply window using vDSP
        vDSP.multiply(padded[start ..< start + nFft], window, result: &windowedFrame)

        // Convert to split complex for vDSP FFT
        windowedFrame.withUnsafeBufferPointer { buf in
            buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
            }
        }

        // Forward real FFT (in-place on split complex)
        var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Scale by 0.5 (vDSP FFT returns values scaled by 2)
        var scale: Float = 0.5
        vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(halfN))
        vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(halfN))

        // Compute power spectrum (magnitude squared)
        let offset = f * nFreqs
        powerSpectrum[offset] = realPart[0] * realPart[0]  // DC bin
        for k in 1..<halfN {
            powerSpectrum[offset + k] = realPart[k] * realPart[k] + imagPart[k] * imagPart[k]
        }
        powerSpectrum[offset + halfN] = imagPart[0] * imagPart[0]  // Nyquist bin
    }

    return (powerSpectrum, numFrames, nFreqs)
}

/// Log scaling mode for mel spectrogram computation.
public enum MelLogScale: Sendable {
    /// Whisper-style: log10 + dynamic range compression + shift/scale.
    case whisper
    /// Standard log-mel: natural log with floor at 1e-5.
    case standard
    /// No log scaling applied.
    case noScaling
}

/// Compute mel spectrogram entirely using Apple Accelerate framework (vDSP + BLAS).
///
/// This path avoids all MLXArray overhead for the core DSP pipeline. The entire
/// computation stays in native `[Float]` arrays and uses NEON SIMD through
/// Accelerate. The result is converted back to MLXArray at the very end.
///
/// - Parameters:
///   - samples: Audio samples as a flat `[Float]` array.
///   - sampleRate: Audio sample rate.
///   - nFft: FFT size.
///   - hopLength: Hop length between STFT frames.
///   - nMels: Number of mel filterbank channels.
///   - fMin: Minimum frequency for mel filterbank (default 0).
///   - fMax: Maximum frequency for mel filterbank (default nil = sampleRate/2).
///   - norm: Filterbank normalization ("slaney" or nil).
///   - logScale: Type of log scaling to apply.
/// - Returns: Mel spectrogram as MLXArray `[numFrames, nMels]`.
public func computeMelSpectrogramAccelerate(
    samples: [Float],
    sampleRate: Int,
    nFft: Int,
    hopLength: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = "slaney",
    logScale: MelLogScale = .whisper
) -> MLXArray {
    let (powerSpectrum, numFrames, nFreqs) = stftPowerSpectrumAccelerate(
        samples: samples, nFft: nFft, hopLength: hopLength
    )

    let flatFilters = melFiltersFlat(
        sampleRate: sampleRate, nFft: nFft, nMels: nMels,
        fMin: fMin, fMax: fMax, norm: norm
    )

    // Matrix multiply via BLAS
    var melSpec = [Float](repeating: 0, count: numFrames * nMels)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(numFrames), Int32(nMels), Int32(nFreqs),
        1.0, powerSpectrum, Int32(nFreqs),
        flatFilters, Int32(nMels),
        0.0, &melSpec, Int32(nMels)
    )

    // Log scaling using vDSP/vForce
    switch logScale {
    case .whisper:
        var clampVal: Float = 1e-10
        vDSP.threshold(melSpec, to: clampVal, with: .clampToThreshold, result: &melSpec)
        var count = Int32(melSpec.count)
        var logResult = [Float](repeating: 0, count: melSpec.count)
        vvlog10f(&logResult, melSpec, &count)
        melSpec = logResult
        let maxVal = vDSP.maximum(melSpec)
        clampVal = maxVal - 8.0
        vDSP.threshold(melSpec, to: clampVal, with: .clampToThreshold, result: &melSpec)
        vDSP.add(4.0, melSpec, result: &melSpec)
        vDSP.multiply(0.25, melSpec, result: &melSpec)

    case .standard:
        var clampVal: Float = 1e-5
        vDSP.threshold(melSpec, to: clampVal, with: .clampToThreshold, result: &melSpec)
        var count = Int32(melSpec.count)
        var logResult = [Float](repeating: 0, count: melSpec.count)
        vvlogf(&logResult, melSpec, &count)
        melSpec = logResult

    case .noScaling:
        break
    }

    return MLXArray(melSpec).reshaped([numFrames, nMels])
}

/// Compute mel spectrogram from audio waveform.
///
/// Uses Apple Accelerate framework (vDSP + BLAS) for the entire DSP pipeline.
public func computeMelSpectrogram(
    audio: MLXArray,
    sampleRate: Int,
    nFft: Int,
    hopLength: Int,
    nMels: Int
) -> MLXArray {
    if audio.ndim == 1 {
        let samples = audio.asArray(Float.self)
        return computeMelSpectrogramAccelerate(
            samples: samples, sampleRate: sampleRate,
            nFft: nFft, hopLength: hopLength, nMels: nMels,
            logScale: .whisper
        )
    }
    return audio
}
