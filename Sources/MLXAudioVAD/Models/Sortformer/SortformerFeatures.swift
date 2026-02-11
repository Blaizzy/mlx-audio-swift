import Foundation
import MLX
import MLXAudioCore

private let logGuard: Float = pow(2.0, -24)
private let normConstant: Float = 1e-5

/// Apply preemphasis filter: y[n] = x[n] - coeff * x[n-1].
public func preemphasisFilter(_ waveform: MLXArray, coeff: Float = 0.97) -> MLXArray {
    let first = waveform[.ellipsis, ..<1]
    let rest = waveform[.ellipsis, 1...] - coeff * waveform[.ellipsis, ..<(-1)]
    return MLX.concatenated([first, rest], axis: -1)
}

/// Extract log-mel spectrogram features matching NeMo's FilterbankFeatures.
///
/// - Parameters:
///   - waveform: 1-D `(numSamples,)` or 2-D `(batch, numSamples)` audio
///   - sampleRate: Sample rate in Hz (default 16000)
///   - nFft: FFT size (default 512)
///   - hopLength: STFT hop (default 160)
///   - winLength: Window length in samples (default 400)
///   - nMels: Number of mel bins (default 80)
///   - preemphasisCoeff: Preemphasis coefficient (default 0.97)
///   - normalize: `"per_feature"` for per-mel-bin normalization, `nil` to skip
///   - padTo: Pad output frames to a multiple of this value (0 to disable)
/// - Returns: `(batch, nMels, numFrames)` matching NeMo convention
public func extractMelFeatures(
    _ waveform: MLXArray,
    sampleRate: Int = 16000,
    nFft: Int = 512,
    hopLength: Int = 160,
    winLength: Int = 400,
    nMels: Int = 80,
    preemphasisCoeff: Float = 0.97,
    normalize: String? = "per_feature",
    padTo: Int = 16
) -> MLXArray {
    var wav = waveform
    if wav.ndim == 1 {
        wav = wav.expandedDimensions(axis: 0)
    }

    wav = preemphasisFilter(wav, coeff: preemphasisCoeff)
    let batchSize = wav.dim(0)

    // Mel filterbank using Slaney scale + Slaney norm
    let melFb = melFilters(
        sampleRate: sampleRate,
        nFft: nFft,
        nMels: nMels,
        fMin: 0,
        fMax: nil,
        norm: "slaney",
        melScale: .slaney
    )

    // Build window: center-pad if winLength < nFft (matching torch.stft behavior)
    var window = hanningWindow(size: winLength)
    if winLength < nFft {
        let left = (nFft - winLength) / 2
        let right = nFft - winLength - left
        window = MLX.concatenated([MLXArray.zeros([left]), window, MLXArray.zeros([right])])
    }

    var allFeatures = [MLXArray]()
    for b in 0..<batchSize {
        // STFT with constant (zero) padding — NeMo convention
        let spec = stft(
            audio: wav[b],
            window: window,
            nFft: nFft,
            hopLength: hopLength,
            padMode: .constant
        )
        // Power spectrum
        let power = MLX.abs(spec).square()
        // Apply mel filterbank: [numFrames, nFreqs] @ [nFreqs, nMels] = [numFrames, nMels]
        let melSpec = MLX.matmul(power, melFb)
        // Log with guard (no Whisper-style clamping)
        let logMel = MLX.log(melSpec + logGuard)
        // Transpose to (nMels, numFrames)
        allFeatures.append(logMel.transposed(1, 0))
    }

    var features = MLX.stacked(allFeatures) // (batch, nMels, numFrames)

    // Per-feature normalization with Bessel's correction
    if normalize == "per_feature" {
        let mean = MLX.mean(features, axis: 2, keepDims: true)
        let variance = MLX.sum(
            (features - mean).square(), axis: 2, keepDims: true
        ) / Float(features.dim(2) - 1)
        let std = MLX.sqrt(variance)
        features = (features - mean) / (std + normConstant)
    }

    // Pad frames to multiple of padTo
    if padTo > 0 {
        let numFrames = features.dim(2)
        let remainder = numFrames % padTo
        if remainder > 0 {
            let padSize = padTo - remainder
            features = MLX.padded(
                features,
                widths: [.init((0, 0)), .init((0, 0)), .init((0, padSize))]
            )
        }
    }

    return features
}

/// Trim leading and trailing silence from audio using frame energy.
///
/// - Returns: `(trimmedWaveform, trimOffsetSamples)`
public func trimSilence(
    _ waveform: MLXArray,
    sampleRate: Int,
    frameMs: Int = 30,
    energyRatio: Float = 0.01,
    minSpeechSec: Float = 0.5
) -> (MLXArray, Int) {
    let frameLen = Int(Float(sampleRate) * Float(frameMs) / 1000.0)
    let minSpeechFrames = max(3, Int(minSpeechSec * 1000.0 / Float(frameMs)))
    let numFrames = waveform.dim(0) / frameLen

    if numFrames < minSpeechFrames * 2 {
        return (waveform, 0)
    }

    let frames = waveform[..<(numFrames * frameLen)].reshaped(numFrames, frameLen)
    let energy = MLX.sqrt(MLX.mean(frames.square(), axis: 1))
    let thresholdVal = energy.max().item(Float.self) * energyRatio
    let speech = energy .> thresholdVal
    eval(speech)
    let speechList: [Bool] = (0..<numFrames).map { speech[$0].item(Bool.self) }

    var startFrame = 0
    for i in 0..<(numFrames - minSpeechFrames + 1) {
        if speechList[i..<(i + minSpeechFrames)].allSatisfy({ $0 }) {
            startFrame = i
            break
        }
    }

    var endFrame = numFrames
    for i in stride(from: numFrames - 1, through: minSpeechFrames - 1, by: -1) {
        if speechList[(i - minSpeechFrames + 1)...(i)].allSatisfy({ $0 }) {
            endFrame = i + 1
            break
        }
    }

    let startSample = startFrame * frameLen
    let endSample = min(endFrame * frameLen, waveform.dim(0))

    if startSample == 0 && endSample == waveform.dim(0) {
        return (waveform, 0)
    }

    return (waveform[startSample..<endSample], startSample)
}
