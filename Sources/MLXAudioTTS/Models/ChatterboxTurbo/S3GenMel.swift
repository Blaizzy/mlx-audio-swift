//
//  S3GenMel.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXAudioCore

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
        norm: nil,
        melScale: .htk
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
    if padAmount > 0 {
        waveform = MLX.padded(waveform, widths: [.init(0), .init((padAmount, padAmount))])
    }

    let batch = waveform.shape[0]
    let window = hanningWindow(size: winSize)

    var specs: [MLXArray] = []
    specs.reserveCapacity(batch)

    for b in 0..<batch {
        let signal = waveform[b, 0...]
        let stftResult = stft(audio: signal, window: window, nFft: nFft, hopLength: hopSize)
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
