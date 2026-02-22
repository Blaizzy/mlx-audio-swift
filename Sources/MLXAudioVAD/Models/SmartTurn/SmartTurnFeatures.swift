import Foundation
import MLX
import MLXAudioCore

enum SmartTurnError: Error {
    case invalidAudioShape([Int])
    case invalidFeatureShape([Int])
}

func smartTurnPrepareAudioSamples(
    _ audio: MLXArray,
    sampleRate: Int?,
    processor: SmartTurnProcessorConfig
) throws -> [Float] {
    guard audio.ndim == 1 else {
        throw SmartTurnError.invalidAudioShape(audio.shape)
    }

    let sourceRate = sampleRate ?? processor.samplingRate
    let input = audio.asArray(Float.self)
    var prepared = try resampleAudio(
        input,
        from: sourceRate,
        to: processor.samplingRate
    )

    let maxSamples = processor.maxAudioSeconds * processor.samplingRate
    if prepared.count > maxSamples {
        prepared = Array(prepared[(prepared.count - maxSamples)...])
    } else if prepared.count < maxSamples {
        let pad = [Float](repeating: 0, count: maxSamples - prepared.count)
        prepared = pad + prepared
    }

    if processor.normalizeAudio, !prepared.isEmpty {
        let mean = prepared.reduce(0, +) / Float(prepared.count)
        let variance = prepared.reduce(0) { acc, x in
            let d = x - mean
            return acc + d * d
        } / Float(prepared.count)
        let std = max(sqrt(variance), 1e-7)
        prepared = prepared.map { ($0 - mean) / std }
    }

    return prepared
}

func smartTurnLogMelSpectrogram(
    _ audio: [Float],
    sampleRate: Int,
    nFft: Int,
    hopLength: Int,
    nMels: Int
) -> MLXArray {
    let audioArray = MLXArray(audio)
    let window = hanningWindow(size: nFft)
    let freqs = stft(audio: audioArray, window: window, nFft: nFft, hopLength: hopLength)

    let magnitudes: MLXArray
    if freqs.dim(0) > 1 {
        magnitudes = MLX.abs(freqs[0..<(freqs.dim(0) - 1), 0...]).square()
    } else {
        magnitudes = MLX.abs(freqs).square()
    }

    let filters = melFilters(
        sampleRate: sampleRate,
        nFft: nFft,
        nMels: nMels,
        norm: "slaney",
        melScale: .slaney
    )

    var melSpec = MLX.matmul(magnitudes, filters)
    melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
    melSpec = MLX.log10(melSpec)
    let maxVal = melSpec.max()
    melSpec = MLX.maximum(melSpec, maxVal - MLXArray(Float(8.0)))
    melSpec = (melSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))
    return melSpec
}
