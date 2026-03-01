import Foundation
import MLX

private let kSampleRate: Int = 16000
private let kNfft: Int = 400
private let kHopLength: Int = 160
private let kWinLength: Int = 400
private let kNMels: Int = 60

/// SpeechBrain-compatible mel spectrogram computed entirely on GPU.
///
/// Uses periodic Hamming window, zero center-padding, HTK mel scale,
/// `10 * log10` normalization, and `top_db = 80` clipping.
enum EcapaMelSpectrogram {

    /// Cached Hamming window (periodic, length 400).
    private nonisolated(unsafe) static let hammingWindow: MLXArray = MLXArray(
        (0..<kWinLength).map { n -> Float in
            0.54 - 0.46 * cos(2.0 * Float.pi * Float(n) / Float(kWinLength))
        }
    )

    /// Cached HTK mel filterbank `[nfft/2+1, nMels]`.
    private nonisolated(unsafe) static let melFilterbank: MLXArray = htkMelFilterbank(
        sampleRate: kSampleRate, nfft: kNfft, nMels: kNMels
    )

    /// Compute mel spectrogram from raw 16 kHz audio.
    /// - Parameter audio: 1-D `MLXArray` of audio samples
    /// - Returns: `[1, numFrames, 60]` log-mel spectrogram
    static func compute(audio: MLXArray) -> MLXArray {

        let padLen = kNfft / 2
        let padded = concatenated([MLXArray.zeros([padLen]), audio, MLXArray.zeros([padLen])])

        let totalLen = padLen + audio.dim(0) + padLen
        let numFrames = max(0, (totalLen - kNfft) / kHopLength + 1)
        if numFrames == 0 { return MLXArray.zeros([1, 0, kNMels]) }

        let frames = asStrided(padded, [numFrames, kNfft], strides: [kHopLength, 1])
        let fftResult = rfft(frames * hammingWindow, n: kNfft, axis: -1)

        let magnitude = abs(fftResult)
        let powerSpec = magnitude * magnitude

        let melSpec = matmul(powerSpec, melFilterbank)

        let logMel = 10.0 * log10(maximum(melSpec, MLXArray(Float(1e-10))))
        let clipped = maximum(logMel, logMel.max() - 80.0)

        return clipped.reshaped(1, numFrames, kNMels)
    }

    /// Build an HTK mel filterbank matrix.
    /// - Returns: `[nfft/2+1, nMels]` MLXArray filterbank
    private static func htkMelFilterbank(sampleRate: Int, nfft: Int, nMels: Int) -> MLXArray {
        func hzToMel(_ f: Float) -> Float { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Float) -> Float { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

        let lowMel = hzToMel(0.0)
        let highMel = hzToMel(Float(sampleRate) / 2.0)
        let melPoints = (0..<(nMels + 2)).map { i in
            melToHz(lowMel + Float(i) * (highMel - lowMel) / Float(nMels + 1))
        }
        let fftBins = (0..<(nfft / 2 + 1)).map { Float(sampleRate) * Float($0) / Float(nfft) }

        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: nfft / 2 + 1),
            count: nMels
        )
        for m in 0..<nMels {
            let fLeft = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fRight = melPoints[m + 2]
            let bandLeft = fCenter - fLeft
            let bandRight = fRight - fCenter
            for k in 0..<fftBins.count {
                let f = fftBins[k]
                if f >= fLeft && f <= fCenter && bandLeft > 0 {
                    filterbank[m][k] = (f - fLeft) / bandLeft
                } else if f > fCenter && f <= fRight && bandRight > 0 {
                    filterbank[m][k] = (fRight - f) / bandRight
                }
            }
        }
        return MLXArray(filterbank.flatMap { $0 })
            .reshaped(nMels, nfft / 2 + 1)
            .transposed()
    }
}
