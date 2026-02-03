//
//  DSP.swift
//  MLXAudio
//
//  Digital Signal Processing utilities for audio processing.
//  Ported from mlx_audio/dsp.py
//

import Foundation
import MLX

// MARK: - Window Functions

/// Hanning (Hann) window function.
///
/// - Parameters:
///   - size: Window length
///   - periodic: If true, use periodic window (for spectral analysis)
/// - Returns: Hanning window as MLXArray
public func hanningWindow(size: Int, periodic: Bool = false) -> MLXArray {
    let denom = periodic ? Float(size) : Float(size - 1)
    var values = [Float](repeating: 0, count: size)
    for n in 0..<size {
        values[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
    }
    return MLXArray(values)
}

/// Hamming window function.
///
/// - Parameters:
///   - size: Window length
///   - periodic: If true, use periodic window (for spectral analysis)
/// - Returns: Hamming window as MLXArray
public func hammingWindow(size: Int, periodic: Bool = false) -> MLXArray {
    let denom = periodic ? Float(size) : Float(size - 1)
    var values = [Float](repeating: 0, count: size)
    for n in 0..<size {
        values[n] = 0.54 - 0.46 * cos(2 * Float.pi * Float(n) / denom)
    }
    return MLXArray(values)
}

// MARK: - STFT

/// Short-Time Fourier Transform.
///
/// - Parameters:
///   - x: Input signal [samples]
///   - nFFT: FFT size (default: 800)
///   - hopLength: Hop length between frames (default: nFFT / 4)
///   - winLength: Window length (default: nFFT)
///   - window: Window type or array (default: "hann")
///   - center: If true, pad signal for centering (default: true)
///   - padMode: Padding mode ("reflect" or "constant")
/// - Returns: Complex STFT output [numFrames, nFFT/2 + 1]
public func stft(
    _ x: MLXArray,
    nFFT: Int = 800,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: String = "hann",
    center: Bool = true,
    padMode: String = "reflect"
) -> MLXArray {
    let hop = hopLength ?? (nFFT / 4)
    let win = winLength ?? nFFT

    // Get window
    var w: MLXArray
    switch window.lowercased() {
    case "hann", "hanning":
        w = hanningWindow(size: win)
    case "hamming":
        w = hammingWindow(size: win)
    default:
        w = hanningWindow(size: win)
    }

    // Pad window to nFFT if needed
    if w.shape[0] < nFFT {
        let padSize = nFFT - w.shape[0]
        w = concatenated([w, MLXArray.zeros([padSize])], axis: 0)
    }

    // Pad input signal
    var xPadded = x
    if center {
        let pad = nFFT / 2
        if padMode == "reflect" {
            // Reflect padding - manually construct
            // prefix = x[1:pad+1][::-1]
            // suffix = x[-(pad+1):-1][::-1]
            var prefixSlices: [MLXArray] = []
            for i in (1...pad).reversed() {
                prefixSlices.append(x[i..<(i+1)])
            }
            let prefix = concatenated(prefixSlices, axis: 0)

            var suffixSlices: [MLXArray] = []
            let n = x.shape[0]
            for i in ((n - pad - 1)..<(n - 1)).reversed() {
                suffixSlices.append(x[i..<(i+1)])
            }
            let suffix = concatenated(suffixSlices, axis: 0)

            xPadded = concatenated([prefix, x, suffix], axis: 0)
        } else {
            // Constant padding
            xPadded = padded(x, widths: [IntOrPair(pad)])
        }
    }

    // Calculate number of frames
    let numFrames = 1 + (xPadded.shape[0] - nFFT) / hop
    guard numFrames > 0 else {
        fatalError("Input is too short for the specified nFFT and hopLength")
    }

    // Create frames using strided view
    var frames: [MLXArray] = []
    for i in 0..<numFrames {
        let start = i * hop
        let frame = xPadded[start..<(start + nFFT)]
        frames.append(frame)
    }
    let framedSignal = stacked(frames, axis: 0)  // [numFrames, nFFT]

    // Apply window and compute FFT
    let windowed = framedSignal * w
    let fftResult = MLXFFT.rfft(windowed, axis: -1)

    return fftResult
}

// MARK: - Mel Filterbank

/// Compute mel filterbank matrix.
///
/// - Parameters:
///   - sampleRate: Audio sample rate
///   - nFFT: FFT size
///   - nMels: Number of mel bands
///   - fMin: Minimum frequency
///   - fMax: Maximum frequency (default: sampleRate / 2)
///   - norm: Normalization type ("slaney" or nil)
///   - melScale: Mel scale type ("htk" or "slaney")
/// - Returns: Mel filterbank [nMels, nFFT/2 + 1]
public func melFilters(
    sampleRate: Int,
    nFFT: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = nil,
    melScale: String = "htk"
) -> MLXArray {
    let fMaxActual = fMax ?? Float(sampleRate) / 2

    // Hz to Mel conversion
    func hzToMel(_ freq: Float, scale: String) -> Float {
        if scale == "htk" {
            return 2595.0 * log10(1.0 + freq / 700.0)
        }
        // Slaney scale
        let fMinSlaney: Float = 0.0
        let fSp: Float = 200.0 / 3
        var mels = (freq - fMinSlaney) / fSp
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMinSlaney) / fSp
        let logStep: Float = logf(6.4) / 27.0
        if freq >= minLogHz {
            mels = minLogMel + logf(freq / minLogHz) / logStep
        }
        return mels
    }

    // Mel to Hz conversion
    func melToHz(_ mels: Float, scale: String) -> Float {
        if scale == "htk" {
            return 700.0 * (pow(10.0, mels / 2595.0) - 1.0)
        }
        // Slaney scale
        let fMinSlaney: Float = 0.0
        let fSp: Float = 200.0 / 3
        var freqs = fMinSlaney + fSp * mels
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMinSlaney) / fSp
        let logStep: Float = logf(6.4) / 27.0
        if mels >= minLogMel {
            freqs = minLogHz * expf(logStep * (mels - minLogMel))
        }
        return freqs
    }

    // Generate frequency points
    let nFreqs = nFFT / 2 + 1
    var allFreqs = [Float](repeating: 0, count: nFreqs)
    let halfSampleRate = Float(sampleRate) / 2
    for i in 0..<nFreqs {
        allFreqs[i] = halfSampleRate * Float(i) / Float(nFreqs - 1)
    }
    let allFreqsArray = MLXArray(allFreqs)

    // Convert frequencies to mel and back
    let mMin = hzToMel(fMin, scale: melScale)
    let mMax = hzToMel(fMaxActual, scale: melScale)

    var mPts = [Float](repeating: 0, count: nMels + 2)
    for i in 0..<(nMels + 2) {
        let mel = mMin + (mMax - mMin) * Float(i) / Float(nMels + 1)
        mPts[i] = melToHz(mel, scale: melScale)
    }
    let fPts = MLXArray(mPts)

    // Compute slopes for filterbank
    // fDiff = fPts[1:] - fPts[:-1]
    var fDiffValues = [Float](repeating: 0, count: nMels + 1)
    for i in 0..<(nMels + 1) {
        fDiffValues[i] = mPts[i + 1] - mPts[i]
    }
    let fDiff = MLXArray(fDiffValues)

    // slopes = expand_dims(fPts, 0) - expand_dims(allFreqs, 1)
    // slopes: [nFreqs, nMels + 2]
    let fPtsExpanded = expandedDimensions(fPts, axis: 0)  // [1, nMels + 2]
    let allFreqsExpanded = expandedDimensions(allFreqsArray, axis: 1)  // [nFreqs, 1]
    let slopes = fPtsExpanded - allFreqsExpanded  // [nFreqs, nMels + 2]

    // Calculate overlapping triangular filters
    // downSlopes = (-slopes[:, :-2]) / fDiff[:-1]
    // upSlopes = slopes[:, 2:] / fDiff[1:]
    let slopesNeg = -slopes[0..., 0..<nMels]  // [nFreqs, nMels]
    let slopesPos = slopes[0..., 2...]  // [nFreqs, nMels]
    let fDiffLeft = fDiff[0..<nMels]  // [nMels]
    let fDiffRight = fDiff[1...]  // [nMels]

    let downSlopes = slopesNeg / fDiffLeft
    let upSlopes = slopesPos / fDiffRight

    // filterbank = maximum(0, minimum(downSlopes, upSlopes))
    var filterbank = minimum(downSlopes, upSlopes)
    filterbank = maximum(filterbank, MLXArray(Float(0)))

    // Apply Slaney normalization if requested
    if norm == "slaney" {
        // enorm = 2.0 / (fPts[2:nMels+2] - fPts[:nMels])
        var enormValues = [Float](repeating: 0, count: nMels)
        for i in 0..<nMels {
            enormValues[i] = 2.0 / (mPts[i + 2] - mPts[i])
        }
        let enorm = MLXArray(enormValues)
        filterbank = filterbank * enorm
    }

    // Transpose to [nMels, nFreqs]
    filterbank = filterbank.transposed(1, 0)

    return filterbank
}

// MARK: - Mel Spectrogram

/// Compute mel spectrogram from audio waveform.
///
/// - Parameters:
///   - audio: Audio waveform [samples] or [batch, samples]
///   - nFFT: FFT size (default: 1024)
///   - numMels: Number of mel bands (default: 128)
///   - sampleRate: Audio sample rate (default: 24000)
///   - hopSize: Hop size (default: 256)
///   - winSize: Window size (default: 1024)
///   - fMin: Minimum frequency (default: 0)
///   - fMax: Maximum frequency (default: 12000)
/// - Returns: Mel spectrogram [batch, frames, numMels]
public func melSpectrogram(
    audio: MLXArray,
    nFFT: Int = 1024,
    numMels: Int = 128,
    sampleRate: Int = 24000,
    hopSize: Int = 256,
    winSize: Int = 1024,
    fMin: Float = 0.0,
    fMax: Float = 12000.0
) -> MLXArray {
    // Ensure batch dimension
    var batchedAudio = audio
    if audio.ndim == 1 {
        batchedAudio = expandedDimensions(audio, axis: 0)
    }

    let batchSize = batchedAudio.shape[0]

    // Get mel filterbank
    let melBasis = melFilters(
        sampleRate: sampleRate,
        nFFT: nFFT,
        nMels: numMels,
        fMin: fMin,
        fMax: fMax,
        norm: "slaney",
        melScale: "slaney"
    )

    // Compute STFT for each sample in batch
    // Padding calculation matches PyTorch reference
    let padding = (nFFT - hopSize) / 2
    var mels: [MLXArray] = []
    for i in 0..<batchSize {
        // Manual reflect padding to match PyTorch reference (center=False with manual pad)
        var sample = batchedAudio[i]
        let sampleLen = sample.shape[0]

        // Reflect padding: prefix = sample[1:padding+1][::-1], suffix = sample[-(padding+1):-1][::-1]
        if padding > 0 && sampleLen > padding {
            var prefixSlices: [MLXArray] = []
            for j in (1...min(padding, sampleLen - 1)).reversed() {
                prefixSlices.append(sample[j..<(j+1)])
            }
            let prefix = concatenated(prefixSlices, axis: 0)

            var suffixSlices: [MLXArray] = []
            let startIdx = max(0, sampleLen - padding - 1)
            for j in (startIdx..<(sampleLen - 1)).reversed() {
                suffixSlices.append(sample[j..<(j+1)])
            }
            let suffix = concatenated(suffixSlices, axis: 0)

            sample = concatenated([prefix, sample, suffix], axis: 0)
        }

        let spec = stft(
            sample,
            nFFT: nFFT,
            hopLength: hopSize,
            winLength: winSize,
            window: "hann",
            center: false,
            padMode: "reflect"
        )

        // Get magnitude spectrum (with epsilon for numerical stability)
        let specMag = sqrt(pow(abs(spec), 2) + MLXArray(Float(1e-9)))

        // Apply mel filterbank: specMag is [frames, nFFT/2+1], melBasis is [nMels, nFFT/2+1]
        var mel = matmul(specMag, melBasis.transposed(1, 0))

        // Log scale with clipping
        mel = log(maximum(mel, MLXArray(Float(1e-5))))
        mels.append(mel)
    }

    return stacked(mels, axis: 0)  // [batch, frames, nMels]
}
