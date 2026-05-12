import Foundation
import MLX
import MLXAudioCore

enum MiMoAudioPreprocessing {
    static let targetSampleRate = 24_000
    static let nFFT = 960
    static let hopLength = 240
    static let nMels = 128
    static let chunkDurationSeconds = 30

    static func normalizeWaveform(_ waveform: MLXArray) -> MLXArray {
        if waveform.ndim == 1 {
            return waveform.asType(.float32)
        }
        return waveform.mean(axis: -1).reshaped([-1]).asType(.float32)
    }

    static func splitIntoChunks(_ waveform: MLXArray, sampleRate: Int = targetSampleRate) -> [MLXArray] {
        let wav = normalizeWaveform(waveform)
        let totalSamples = wav.shape[0]
        guard totalSamples > 0 else { return [] }

        let chunkSamples = chunkDurationSeconds * sampleRate
        var chunks: [MLXArray] = []
        chunks.reserveCapacity(max(1, totalSamples / max(chunkSamples, 1)))

        var start = 0
        while start < totalSamples {
            var end = min(start + chunkSamples, totalSamples)
            if 0 < totalSamples - end, totalSamples - end < nFFT {
                end = totalSamples
            }

            var chunk = wav[start..<end]
            if chunk.shape[0] < nFFT {
                chunk = MLX.padded(chunk, widths: [IntOrPair((0, nFFT - chunk.shape[0]))])
            }

            chunks.append(chunk)
            start = end
        }

        return chunks
    }

    static func computeLogMel(_ waveform: MLXArray) -> MLXArray {
        let audio = normalizeWaveform(waveform)
        guard audio.shape[0] > 0 else {
            return MLXArray.zeros([0, nMels], type: Float.self)
        }

        let window = hanningWindow(size: nFFT)
        let freqs = stft(audio: audio, window: window, nFft: nFFT, hopLength: hopLength)
        let magnitudes = MLX.abs(freqs)
        let filters = melFilters(
            sampleRate: targetSampleRate,
            nFft: nFFT,
            nMels: nMels,
            fMin: 0,
            fMax: nil,
            norm: nil,
            melScale: .htk
        )

        let mel = MLX.matmul(magnitudes, filters)
        let clipped = MLX.maximum(mel, MLXArray(Float(1e-7)))
        return MLX.log(clipped)
    }
}
