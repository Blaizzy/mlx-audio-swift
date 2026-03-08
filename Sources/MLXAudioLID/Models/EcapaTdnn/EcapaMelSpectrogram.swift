import Foundation
import MLX
import MLXAudioCore

private let kSampleRate: Int = 16000
private let kNfft: Int = 400
private let kHopLength: Int = 160
private let kWinLength: Int = 400
private let kNMels: Int = 60

/// SpeechBrain-compatible mel spectrogram computed on top of shared DSP helpers.
///
/// Uses periodic Hamming window, zero center-padding, HTK mel scale,
/// `10 * log10` normalization, and `top_db = 80` clipping.
enum EcapaMelSpectrogram {

    /// Cached Hamming window (periodic, length 400).
    private nonisolated(unsafe) static let hammingWindow: MLXArray = MLXAudioCore.hammingWindow(
        size: kWinLength,
        periodic: true
    )

    /// Cached HTK mel filterbank `[nfft/2+1, nMels]`.
    private nonisolated(unsafe) static let melFilterbank: MLXArray = melFilters(
        sampleRate: kSampleRate,
        nFft: kNfft,
        nMels: kNMels,
        norm: nil,
        melScale: .htk
    )

    /// Compute mel spectrogram from raw 16 kHz audio.
    /// - Parameter audio: 1-D `MLXArray` of audio samples
    /// - Returns: `[1, numFrames, 60]` log-mel spectrogram
    static func compute(audio: MLXArray) -> MLXArray {
        let numFrames = max(0, (audio.dim(0) + kNfft - kNfft) / kHopLength + 1)
        if numFrames == 0 { return MLXArray.zeros([1, 0, kNMels]) }

        let fftResult = stft(
            audio: audio,
            window: hammingWindow,
            nFft: kNfft,
            hopLength: kHopLength,
            padMode: .constant
        )

        let powerSpec = MLX.abs(fftResult).square()

        let melSpec = MLX.matmul(powerSpec, melFilterbank)

        let clipped = powerToDB(melSpec, topDB: 80)

        return clipped.reshaped(1, numFrames, kNMels)
    }
}
