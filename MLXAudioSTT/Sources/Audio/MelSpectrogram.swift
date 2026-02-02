import Accelerate
import Foundation
import MLX

/// Computes mel spectrograms for Whisper STT
/// Matches the Python mlx-audio implementation exactly
public enum MelSpectrogram {

    /// Compute mel spectrogram from audio waveform
    /// - Parameters:
    ///   - audio: Audio samples as MLXArray, shape [nSamples]
    ///   - nMels: Number of mel frequency bins (80 for v1/v2, 128 for v3)
    /// - Returns: Mel spectrogram as MLXArray, shape [nMels, nFrames]
    public static func compute(audio: MLXArray, nMels: Int) throws -> MLXArray {
        let rawSamples = audio.asArray(Float.self)

        guard rawSamples.count >= AudioConstants.hopLength else {
            throw MelSpectrogramError.audioTooShort(
                minSamples: AudioConstants.hopLength, actualSamples: rawSamples.count)
        }

        // Center-pad audio using reflect mode (matching Python librosa/Whisper)
        let padAmount = AudioConstants.nFFT / 2
        var samples = reflectPad(rawSamples, padAmount: padAmount)

        // Calculate number of frames (matching Python's frame count)
        // Python drops the last frame with freqs[:-1, :] to get exactly N_SAMPLES/HOP_LENGTH frames
        let computedFrames = 1 + (samples.count - AudioConstants.nFFT) / AudioConstants.hopLength
        let nFrames = computedFrames - 1  // Drop last frame like Python

        // Create Hann window (matching Python's periodic=False hanning)
        let window = hannWindow(size: AudioConstants.nFFT)

        // Setup FFT - find next power of 2 for vDSP
        let fftSize = nextPowerOf2(AudioConstants.nFFT)
        let log2n = vDSP_Length(log2(Double(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw MelSpectrogramError.fftSetupFailed
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let numBins = AudioConstants.nFFT / 2 + 1

        // Compute STFT magnitudes squared (power spectrum)
        var magnitudesSquared = [[Float]](
            repeating: [Float](repeating: 0, count: numBins), count: nFrames)

        for frame in 0..<nFrames {
            let start = frame * AudioConstants.hopLength

            // Extract frame and apply window
            var windowedFrame = [Float](repeating: 0, count: fftSize)
            for i in 0..<AudioConstants.nFFT {
                windowedFrame[i] = samples[start + i] * window[i]
            }

            // Compute FFT magnitude squared (power spectrum)
            magnitudesSquared[frame] = fftMagnitudeSquared(
                frame: windowedFrame, setup: fftSetup, log2n: log2n, numBins: numBins)
        }

        // Apply mel filterbank with Slaney normalization
        let melFilters = createMelFilterbank(nMels: nMels, nFFT: AudioConstants.nFFT)
        var melSpec = [[Float]](
            repeating: [Float](repeating: 0, count: nFrames), count: nMels)

        for mel in 0..<nMels {
            for frame in 0..<nFrames {
                var sum: Float = 0
                for bin in 0..<numBins {
                    sum += melFilters[mel][bin] * magnitudesSquared[frame][bin]
                }
                melSpec[mel][frame] = sum
            }
        }

        // Convert to log10 scale (matches Python Whisper)
        let logMelSpec = melSpec.map { row in
            row.map { log10(max($0, 1e-10)) }
        }

        // Clip to max - 8.0 (Whisper's dynamic range limit)
        let maxLogVal = logMelSpec.flatMap { $0 }.max() ?? 0
        let clippedSpec = logMelSpec.map { row in
            row.map { max($0, maxLogVal - 8.0) }
        }

        // Normalize to roughly [-1, 1] range (Whisper normalization)
        let normalizedSpec = clippedSpec.map { row in
            row.map { ($0 + 4.0) / 4.0 }
        }

        // Convert to MLXArray [nMels, nFrames]
        let flatData = normalizedSpec.flatMap { $0 }
        return MLXArray(flatData, [nMels, nFrames])
    }

    // MARK: - Private Helpers

    private static func nextPowerOf2(_ n: Int) -> Int {
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }

    private static func reflectPad(_ samples: [Float], padAmount: Int) -> [Float] {
        guard padAmount > 0 else { return samples }
        guard samples.count > padAmount else {
            // Fallback to zero padding for very short audio
            var padded = [Float](repeating: 0, count: padAmount)
            padded.append(contentsOf: samples)
            padded.append(contentsOf: [Float](repeating: 0, count: padAmount))
            return padded
        }

        // Reflect prefix: samples[1:padAmount+1] reversed
        var prefix = [Float]()
        for i in stride(from: min(padAmount, samples.count - 1), through: 1, by: -1) {
            prefix.append(samples[i])
        }
        // If we need more padding than available samples, repeat
        while prefix.count < padAmount {
            prefix.append(prefix.last ?? 0)
        }

        // Reflect suffix: samples[-(padAmount+1):-1] reversed
        var suffix = [Float]()
        let startIdx = max(0, samples.count - padAmount - 1)
        let endIdx = samples.count - 1
        for i in stride(from: endIdx - 1, through: startIdx, by: -1) {
            suffix.append(samples[i])
        }
        while suffix.count < padAmount {
            suffix.append(suffix.last ?? 0)
        }

        var result = prefix
        result.append(contentsOf: samples)
        result.append(contentsOf: suffix)
        return result
    }

    private static func hannWindow(size: Int) -> [Float] {
        // Periodic Hann window (matching numpy's hanning)
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(size)))
        }
        return window
    }

    private static func fftMagnitudeSquared(
        frame: [Float], setup: FFTSetup, log2n: vDSP_Length, numBins: Int
    ) -> [Float] {
        let n = frame.count
        let halfN = n / 2

        var realp = [Float](repeating: 0, count: halfN)
        var imagp = [Float](repeating: 0, count: halfN)
        var magnitudes = [Float](repeating: 0, count: numBins)

        realp.withUnsafeMutableBufferPointer { realpPtr in
            imagp.withUnsafeMutableBufferPointer { imagpPtr in
                var splitComplex = DSPSplitComplex(
                    realp: realpPtr.baseAddress!,
                    imagp: imagpPtr.baseAddress!
                )

                frame.withUnsafeBufferPointer { ptr in
                    ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) {
                        complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                    }
                }

                vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                // Compute magnitudes squared (power spectrum)
                // For bins 1 to halfN-1
                for i in 1..<min(halfN, numBins - 1) {
                    magnitudes[i] = realpPtr[i] * realpPtr[i] + imagpPtr[i] * imagpPtr[i]
                }

                // DC component (bin 0)
                magnitudes[0] = realpPtr[0] * realpPtr[0]

                // Nyquist component (bin halfN = numBins-1 for nFFT=400)
                if numBins > halfN {
                    magnitudes[halfN] = imagpPtr[0] * imagpPtr[0]
                }
            }
        }

        // vDSP FFT scaling factor
        let scale = 1.0 / Float(n)
        vDSP_vsmul(magnitudes, 1, [scale * scale], &magnitudes, 1, vDSP_Length(numBins))

        return magnitudes
    }

    private static func createMelFilterbank(nMels: Int, nFFT: Int) -> [[Float]] {
        let sampleRate = AudioConstants.sampleRate
        let numBins = nFFT / 2 + 1

        // HTK mel scale conversion (matching Python)
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let fMin: Float = 0
        let fMax = Float(sampleRate) / 2.0

        // Generate mel points
        let mMin = hzToMel(fMin)
        let mMax = hzToMel(fMax)

        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = mMin + Float(i) * (mMax - mMin) / Float(nMels + 1)
        }

        // Convert mel points to Hz
        let hzPoints = melPoints.map { melToHz($0) }

        // Generate all frequency bins
        var allFreqs = [Float](repeating: 0, count: numBins)
        for i in 0..<numBins {
            allFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        // Compute filterbank using slopes (matching Python librosa)
        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: numBins), count: nMels)

        for m in 0..<nMels {
            let fLeft = hzPoints[m]
            let fCenter = hzPoints[m + 1]
            let fRight = hzPoints[m + 2]

            let fDiffDown = fCenter - fLeft
            let fDiffUp = fRight - fCenter

            for k in 0..<numBins {
                let freq = allFreqs[k]

                if freq >= fLeft && freq < fCenter && fDiffDown > 0 {
                    filterbank[m][k] = (freq - fLeft) / fDiffDown
                } else if freq >= fCenter && freq <= fRight && fDiffUp > 0 {
                    filterbank[m][k] = (fRight - freq) / fDiffUp
                }
            }
        }

        // Apply Slaney normalization: 2.0 / (f_right - f_left)
        for m in 0..<nMels {
            let fLeft = hzPoints[m]
            let fRight = hzPoints[m + 2]
            let bandwidth = fRight - fLeft
            if bandwidth > 0 {
                let enorm = 2.0 / bandwidth
                for k in 0..<numBins {
                    filterbank[m][k] *= enorm
                }
            }
        }

        return filterbank
    }
}

public enum MelSpectrogramError: Error, CustomStringConvertible {
    case fftSetupFailed
    case invalidAudioShape
    case audioTooShort(minSamples: Int, actualSamples: Int)

    public var description: String {
        switch self {
        case .fftSetupFailed:
            return "Failed to create FFT setup"
        case .invalidAudioShape:
            return "Invalid audio shape - expected 1D array"
        case .audioTooShort(let minSamples, let actualSamples):
            return
                "Audio too short: need at least \(minSamples) samples, got \(actualSamples)"
        }
    }
}
