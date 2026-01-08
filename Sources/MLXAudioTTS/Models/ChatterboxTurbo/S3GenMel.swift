//
//  S3GenMel.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXAudioCore

private func s3genReverseArray(_ arr: MLXArray) -> MLXArray {
    let len = arr.shape[0]
    var indices = [Int32]()
    indices.reserveCapacity(len)
    for i in 0..<len {
        indices.append(Int32(len - 1 - i))
    }
    return arr[MLXArray(indices)]
}

private func s3genReflectPad(_ signal: MLXArray, padLeft: Int, padRight: Int) -> MLXArray {
    guard padLeft > 0 || padRight > 0 else { return signal }
    let length = signal.shape[0]
    var prefix = MLXArray([])
    var suffix = MLXArray([])

    if padLeft > 0 {
        let end = min(padLeft + 1, length)
        let slice = signal[1..<end]
        prefix = s3genReverseArray(slice)
    }

    if padRight > 0 {
        let start = max(0, length - padRight - 1)
        let end = max(1, length - 1)
        let slice = signal[start..<end]
        suffix = s3genReverseArray(slice)
    }

    if padLeft > 0 && padRight > 0 {
        return MLX.concatenated([prefix, signal, suffix])
    }
    if padLeft > 0 {
        return MLX.concatenated([prefix, signal])
    }
    return MLX.concatenated([signal, suffix])
}

private func s3genStft(
    _ audio: MLXArray,
    window: MLXArray,
    nFft: Int,
    hopLength: Int
) -> MLXArray {
    let audioLen = audio.shape[0]
    let numFrames = 1 + (audioLen - nFft) / hopLength
    var frames: [MLXArray] = []
    frames.reserveCapacity(numFrames)

    for i in 0..<numFrames {
        let start = i * hopLength
        let frame = audio[start..<(start + nFft)]
        frames.append(frame)
    }

    let stacked = MLX.stacked(frames, axis: 0)
    let windowed = stacked * window
    return MLXFFT.rfft(windowed, axis: 1)
}

private func dynamicRangeCompression(_ x: MLXArray, clipVal: Float = 1e-5) -> MLXArray {
    let clipped = MLX.maximum(x, MLXArray(clipVal))
    return MLX.log(clipped)
}

private func spectralNormalize(_ magnitudes: MLXArray) -> MLXArray {
    dynamicRangeCompression(magnitudes)
}

private func s3genMelFilters(
    nFft: Int,
    numMels: Int,
    samplingRate: Int,
    fMin: Int,
    fMax: Int
) -> MLXArray {
    let fMaxVal = fMax > 0 ? Float(fMax) : Float(samplingRate) / 2.0
    return s3MelFilters(
        sampleRate: samplingRate,
        nFft: nFft,
        nMels: numMels,
        fMin: Float(fMin),
        fMax: fMaxVal,
        norm: "slaney",
        melScale: .slaney
    )
}

func s3genMelSpectrogram(
    _ audio: MLXArray,
    nFft: Int = 1920,
    numMels: Int = 80,
    samplingRate: Int = 24_000,
    hopSize: Int = 480,
    winSize: Int = 1920,
    fMin: Int = 0,
    fMax: Int = 8_000,
    center: Bool = false
) -> MLXArray {
    var waveform = audio
    if waveform.ndim == 1 {
        waveform = waveform.expandedDimensions(axis: 0)
    }

    let padAmount = (nFft - hopSize) / 2

    let batch = waveform.shape[0]
    let window = hanningWindow(size: winSize)

    var specs: [MLXArray] = []
    specs.reserveCapacity(batch)

    for b in 0..<batch {
        var signal = waveform[b, 0...]
        if padAmount > 0 {
            signal = s3genReflectPad(signal, padLeft: padAmount, padRight: padAmount)
        }
        let stftResult = s3genStft(signal, window: window, nFft: nFft, hopLength: hopSize)
        let magnitudes = MLX.abs(stftResult).transposed()
        specs.append(magnitudes)
    }

    let stacked = MLX.stacked(specs, axis: 0)
    let melBasis = s3genMelFilters(
        nFft: nFft,
        numMels: numMels,
        samplingRate: samplingRate,
        fMin: fMin,
        fMax: fMax
    )

    let melSpec = MLX.matmul(melBasis, stacked)
    let normalized = spectralNormalize(melSpec)
    return normalized
}
