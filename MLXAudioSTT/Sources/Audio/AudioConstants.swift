import Foundation

/// Audio processing constants for Whisper STT
/// These values match the Python mlx-audio implementation
public enum AudioConstants {
    /// Sample rate in Hz (Whisper expects 16kHz mono audio)
    public static let sampleRate: Int = 16000

    /// FFT window size for mel spectrogram (matches Whisper's N_FFT=400)
    public static let nFFT: Int = 400

    /// Hop length between frames (whisperWindowSize / 2.5 = 160)
    public static let hopLength: Int = 160

    /// Number of mel frequency bins
    public static let nMels: Int = 80

    /// Number of frames in a full 30-second chunk
    public static let nFrames: Int = 3000

    /// Audio chunk length in seconds
    public static let chunkLength: Int = 30

    /// Number of samples in a full chunk (sampleRate * chunkLength)
    public static let nSamples: Int = 480000
}
