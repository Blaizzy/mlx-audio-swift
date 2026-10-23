import Foundation
import MLX
import MLXAudioCore

func mimoPeriodicHann(_ length: Int) -> MLXArray {
    // Double-precision construction matches the reference's Python math.cos.
    MLXArray((0..<length).map { Float(0.5 * (1 - Foundation.cos(2 * Double.pi * Double($0) / Double(length)))) })
}

func mimoMelFilters(config c: MiMoAudioTokenizerConfig) -> MLXArray {
    // Preserve the reference's Float32 vector arithmetic. Constructing the
    // triangular filters with scalar CPU math can perturb BF16 mel rounding.
    let minMel = 2595.0 * Foundation.log10(1.0 + Double(c.fMin) / 700.0)
    let maxHz = Double(c.fMax ?? Float(c.samplingRate) / 2)
    let maxMel = 2595.0 * Foundation.log10(1.0 + maxHz / 700.0)
    let frequencies = MLX.linspace(Float(0), Float(c.samplingRate / 2), count: c.nFFT / 2 + 1)
    let melPoints = MLX.linspace(minMel, maxMel, count: c.nMels + 2, dtype: .float32)
    let points = 700 * (pow(MLXArray(Float(10)), melPoints / 2595) - 1)
    let differences = points[1...] - points[0..<(c.nMels + 1)]
    let slopes = points.expandedDimensions(axis: 0) - frequencies.expandedDimensions(axis: 1)
    let falling = -slopes[0..., 0..<c.nMels] / differences[0..<c.nMels]
    let rising = slopes[0..., 2...] / differences[1...]
    return maximum(MLXArray(Float(0)), minimum(falling, rising))
}

func mimoLogMelSpectrogram(_ waveform: MLXArray, config c: MiMoAudioTokenizerConfig) -> MLXArray {
    var audio = waveform.asType(.float32)
    if audio.size <= c.nFFT / 2 {
        audio = padded(audio, widths: [.init((0, c.nFFT / 2 + 1 - audio.size))])
    }
    var window = mimoPeriodicHann(c.windowSize)
    if c.windowSize < c.nFFT {
        let extra = c.nFFT - c.windowSize
        window = padded(window, widths: [.init((extra / 2, extra - extra / 2))])
    }
    let spectrum = stft(audio: audio, window: window, nFft: c.nFFT, hopLength: c.hopLength, padMode: .reflect)
    let filters = mimoMelFilters(config: c)
    return log(maximum(matmul(abs(spectrum), filters), MLXArray(Float(1e-7))))
}

/// Batched (B, frames, frequencies) -> (B, frames * hop), with squared-window normalization.
func mimoSameISTFT(_ spectrum: MLXArray, nFFT: Int, hopLength: Int, window: MLXArray) -> MLXArray {
    let batch = spectrum.dim(0), count = spectrum.dim(1)
    let outputLength = (count - 1) * hopLength + nFFT
    let frames = MLXFFT.irfft(spectrum, axis: -1) * window
    var waveform = MLXArray.zeros([batch, outputLength], dtype: .float32)
    var envelope = MLXArray.zeros([outputLength], dtype: .float32)
    // Partition each FFT frame into hop-sized pieces. Within one partition the
    // flattened frames are adjacent, so overlap-add needs only ceil(FFT/hop)
    // vector additions instead of synchronizing once per frame on the CPU.
    for offset in stride(from: 0, to: nFFT, by: hopLength) {
        let width = min(hopLength, nFFT - offset)
        var part = frames[0..., 0..., offset..<(offset + width)]
        var norm = broadcast(window[offset..<(offset + width)].square(), to: [count, width])
        if width < hopLength {
            part = padded(part, widths: [.init(0), .init(0), .init((0, hopLength - width))])
            norm = padded(norm, widths: [.init(0), .init((0, hopLength - width))])
        }
        let length = (count - 1) * hopLength + width
        part = part.reshaped([batch, -1])[0..., 0..<length]
        norm = norm.reshaped([-1])[0..<length]
        let tail = outputLength - offset - length
        waveform = waveform + padded(part, widths: [.init(0), .init((offset, tail))])
        envelope = envelope + padded(norm, widths: [.init((offset, tail))])
    }
    waveform = waveform / MLX.where(envelope .> Float(1e-10), envelope, MLXArray(Float(1)))
    let trim = (nFFT - hopLength) / 2
    return trim > 0 ? waveform[0..., trim..<(outputLength - trim)] : waveform
}
