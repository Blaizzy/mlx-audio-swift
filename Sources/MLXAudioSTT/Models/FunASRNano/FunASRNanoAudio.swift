import Foundation
import MLX
import MLXAudioCore

/// The Fun-ASR frontend follows Python's native Kaldi recipe, including its
/// unmodified first sample after preemphasis and mel-domain triangular filters.
enum FunASRNanoAudio {
    static func fbank(_ waveform: MLXArray, config: SenseVoiceFrontendConfig) -> MLXArray {
        let audio = waveform.asType(.float32) * Float(1 << 15)
        let windowSize = config.fs * config.frameLength / 1000
        let hop = config.fs * config.frameShift / 1000
        guard audio.size >= windowSize else { return .zeros([0, config.nMels]) }
        let count = 1 + (audio.size - windowSize) / hop
        var frames = asStrided(audio, [count, windowSize], strides: [hop, 1])
        frames = frames - frames.mean(axis: 1, keepDims: true)
        frames = concatenated([
            frames[0..., 0..<1],
            frames[0..., 1..<windowSize] - 0.97 * frames[0..., 0..<(windowSize - 1)]
        ], axis: 1)
        let phase = arange(windowSize, dtype: .float32) * (2 * Float.pi / Float(windowSize - 1))
        let window: MLXArray
        switch config.window.lowercased() {
        case "hamming": window = 0.54 - 0.46 * cos(phase)
        case "hanning", "hann": window = 0.5 - 0.5 * cos(phase)
        case "povey": window = pow(0.5 - 0.5 * cos(phase), 0.85)
        default: window = .ones([windowSize])
        }
        frames = frames * window
        var nFFT = 1
        while nFFT < windowSize { nFFT *= 2 }
        if nFFT > windowSize { frames = padded(frames, widths: [.init(0), .init((0, nFFT - windowSize))]) }
        let spectrum = abs(MLXFFT.rfft(frames, axis: 1)).square()
        let low = 1127 * log(MLXArray(Float(1 + 20.0 / 700)))
        let high = 1127 * log(MLXArray(Float(1 + Double(config.fs) / 2 / 700)))
        let delta = (high - low) / Float(config.nMels + 1)
        let bins = arange(config.nMels, dtype: .float32).expandedDimensions(axis: 1)
        let left = low + bins * delta
        let center = low + (bins + 1) * delta
        let right = low + (bins + 2) * delta
        let freq = arange(nFFT / 2, dtype: .float32) * (Float(config.fs) / Float(nFFT))
        let mel = (1127 * log(1 + freq / 700)).expandedDimensions(axis: 0)
        var filters = maximum(0, minimum((mel - left) / (center - left), (right - mel) / (right - center)))
        filters = padded(filters, widths: [.init(0), .init((0, 1))])
        return log(maximum(matmul(spectrum, filters.T), 1e-8))
    }

    static func features(_ waveform: MLXArray, config: SenseVoiceFrontendConfig) -> MLXArray {
        SenseVoiceAudio.applyLFR(fbank(waveform, config: config), lfrM: config.lfrM, lfrN: config.lfrN)
    }

    static func fakeTokenLength(_ speechLength: Int) -> Int {
        let first = 1 + (speechLength - 1) / 2
        let second = 1 + (first - 1) / 2
        return max(1, (second - 1) / 2 + 1)
    }
}
