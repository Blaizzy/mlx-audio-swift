import Accelerate
import Foundation
import MLX

/// Computes mel spectrograms for Whisper STT
/// Matches the Python mlx-audio implementation
public enum MelSpectrogram {

    /// Compute mel spectrogram from audio waveform
    /// - Parameter audio: Audio samples as MLXArray, shape [nSamples]
    /// - Returns: Mel spectrogram as MLXArray, shape [nMels, nFrames]
    public static func compute(audio: MLXArray) throws -> MLXArray {
        let samples = audio.asArray(Float.self)
        let nSamples = samples.count

        guard nSamples >= AudioConstants.hopLength else {
            throw MelSpectrogramError.audioTooShort(
                minSamples: AudioConstants.hopLength, actualSamples: nSamples)
        }

        // Calculate number of frames
        let nFrames = nSamples / AudioConstants.hopLength

        // Create Hann window (400 samples) zero-padded to FFT size (512)
        let window = paddedHannWindow(
            windowSize: AudioConstants.whisperWindowSize, fftSize: AudioConstants.nFFT)

        // Setup FFT
        let log2n = vDSP_Length(log2(Double(AudioConstants.nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw MelSpectrogramError.fftSetupFailed
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let numBins = AudioConstants.nFFT / 2 + 1

        // Compute STFT
        var magnitudes = [[Float]](
            repeating: [Float](repeating: 0, count: numBins), count: nFrames)

        for frame in 0..<nFrames {
            let start = frame * AudioConstants.hopLength
            let end = min(start + AudioConstants.whisperWindowSize, nSamples)

            // Extract and window frame (using whisperWindowSize, rest stays zero-padded)
            var windowedFrame = [Float](repeating: 0, count: AudioConstants.nFFT)
            let frameLength = end - start
            for i in 0..<frameLength {
                windowedFrame[i] = samples[start + i] * window[i]
            }

            // Compute FFT magnitude
            magnitudes[frame] = fftMagnitude(frame: windowedFrame, setup: fftSetup, log2n: log2n)
        }

        // Apply mel filterbank
        let melFilters = createMelFilterbank()
        var melSpec = [[Float]](
            repeating: [Float](repeating: 0, count: nFrames), count: AudioConstants.nMels)

        for mel in 0..<AudioConstants.nMels {
            for frame in 0..<nFrames {
                var sum: Float = 0
                for bin in 0..<numBins {
                    sum += melFilters[mel][bin] * magnitudes[frame][bin]
                }
                melSpec[mel][frame] = sum
            }
        }

        // Convert to log scale and normalize
        let logMelSpec = melSpec.map { row in
            row.map { max(log(max($0, 1e-10)), log(1e-10)) }
        }

        // Normalize to Whisper's expected range
        let maxVal = logMelSpec.flatMap { $0 }.max() ?? 0
        let normalizedSpec = logMelSpec.map { row in
            row.map { ($0 - maxVal) / 4.0 + 1.0 }
        }

        // Convert to MLXArray [nMels, nFrames]
        let flatData = normalizedSpec.flatMap { $0 }
        return MLXArray(flatData, [AudioConstants.nMels, nFrames])
    }

    // MARK: - Private Helpers

    private static func paddedHannWindow(windowSize: Int, fftSize: Int) -> [Float] {
        var window = [Float](repeating: 0, count: windowSize)
        vDSP_hann_window(&window, vDSP_Length(windowSize), Int32(vDSP_HANN_NORM))
        if fftSize > windowSize {
            window.append(contentsOf: [Float](repeating: 0, count: fftSize - windowSize))
        }
        return window
    }

    private static func fftMagnitude(frame: [Float], setup: FFTSetup, log2n: vDSP_Length) -> [Float]
    {
        let n = frame.count
        let halfN = n / 2

        var realp = [Float](repeating: 0, count: halfN)
        var imagp = [Float](repeating: 0, count: halfN)
        var magnitudes = [Float](repeating: 0, count: halfN + 1)

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

                // Compute magnitudes
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfN))

                // Handle DC and Nyquist
                magnitudes[0] = realpPtr[0] * realpPtr[0]
                magnitudes[halfN] = imagpPtr[0] * imagpPtr[0]
            }
        }

        // Square root for magnitude
        var result = [Float](repeating: 0, count: halfN + 1)
        vvsqrtf(&result, magnitudes, [Int32(halfN + 1)])

        return result
    }

    private static func createMelFilterbank() -> [[Float]] {
        let nFFT = AudioConstants.nFFT
        let nMels = AudioConstants.nMels
        let sampleRate = AudioConstants.sampleRate
        let numBins = nFFT / 2 + 1

        // Mel scale conversion
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let lowFreq: Float = 0
        let highFreq = Float(sampleRate) / 2.0

        let lowMel = hzToMel(lowFreq)
        let highMel = hzToMel(highFreq)

        // Create mel points
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = lowMel + Float(i) * (highMel - lowMel) / Float(nMels + 1)
        }

        // Convert to Hz and then to FFT bin
        let hzPoints = melPoints.map { melToHz($0) }
        let binPoints = hzPoints.map { Int(($0 * Float(nFFT)) / Float(sampleRate)) }

        // Create filterbank
        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: numBins), count: nMels)

        for m in 0..<nMels {
            let left = binPoints[m]
            let center = binPoints[m + 1]
            let right = binPoints[m + 2]

            for k in left..<center {
                if k < numBins && center > left {
                    filterbank[m][k] = Float(k - left) / Float(center - left)
                }
            }
            for k in center..<right {
                if k < numBins && right > center {
                    filterbank[m][k] = Float(right - k) / Float(right - center)
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
