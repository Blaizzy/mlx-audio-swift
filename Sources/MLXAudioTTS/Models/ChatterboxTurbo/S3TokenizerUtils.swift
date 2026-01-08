//
//  S3TokenizerUtils.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXAudioCore

let S3SampleRate = 16_000
let S3Hop = 160
let S3TokenHop = 640
let S3TokenRate = 25
let S3SpeechVocabSize = 6_561

enum S3MelScale {
    case htk
    case slaney
}

private func s3HzToMel(_ freq: Float, melScale: S3MelScale) -> Float {
    switch melScale {
    case .htk:
        return 2595.0 * log10f(1.0 + freq / 700.0)
    case .slaney:
        let fMin: Float = 0.0
        let fSp: Float = 200.0 / 3.0
        var mels = (freq - fMin) / fSp
        let minLogHz: Float = 1000.0
        let minLogMel: Float = (minLogHz - fMin) / fSp
        let logStep: Float = logf(6.4) / 27.0
        if freq >= minLogHz {
            mels = minLogMel + logf(freq / minLogHz) / logStep
        }
        return mels
    }
}

private func s3MelToHz(_ mel: Float, melScale: S3MelScale) -> Float {
    switch melScale {
    case .htk:
        return 700.0 * (powf(10.0, mel / 2595.0) - 1.0)
    case .slaney:
        let fMin: Float = 0.0
        let fSp: Float = 200.0 / 3.0
        var freq = fMin + fSp * mel
        let minLogHz: Float = 1000.0
        let minLogMel: Float = (minLogHz - fMin) / fSp
        let logStep: Float = logf(6.4) / 27.0
        if mel >= minLogMel {
            freq = minLogHz * expf(logStep * (mel - minLogMel))
        }
        return freq
    }
}

func s3MelFilters(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float,
    fMax: Float,
    norm: String? = nil,
    melScale: S3MelScale
) -> MLXArray {
    let nFreqs = nFft / 2 + 1
    let fMaxVal = fMax > 0 ? fMax : Float(sampleRate) / 2.0

    let allFreqs = (0..<nFreqs).map { Float($0) * Float(sampleRate) / Float(nFft) }

    let mMin = s3HzToMel(fMin, melScale: melScale)
    let mMax = s3HzToMel(fMaxVal, melScale: melScale)

    let mPtsCount = nMels + 2
    let step = (mMax - mMin) / Float(mPtsCount - 1)
    let mPts = (0..<mPtsCount).map { mMin + Float($0) * step }
    let fPts = mPts.map { s3MelToHz($0, melScale: melScale) }

    var filterbank = Array(repeating: Array(repeating: Float(0), count: nFreqs), count: nMels)

    for m in 0..<nMels {
        let low = fPts[m]
        let center = fPts[m + 1]
        let high = fPts[m + 2]
        let denomDown = center - low
        let denomUp = high - center

        for (i, freq) in allFreqs.enumerated() {
            if freq >= low && freq < center, denomDown > 0 {
                filterbank[m][i] = (freq - low) / denomDown
            } else if freq >= center && freq <= high, denomUp > 0 {
                filterbank[m][i] = (high - freq) / denomUp
            }
        }
    }

    if norm == "slaney" {
        for m in 0..<nMels {
            let denom = fPts[m + 2] - fPts[m]
            if denom > 0 {
                let enorm = 2.0 / denom
                for i in 0..<nFreqs {
                    filterbank[m][i] *= enorm
                }
            }
        }
    }

    let flat = filterbank.flatMap { $0 }
    return MLXArray(flat).reshaped([nMels, nFreqs])
}

func s3LogMelSpectrogram(
    _ audio: MLXArray,
    sampleRate: Int = 16_000,
    nMels: Int = 128,
    nFft: Int = 400,
    hopLength: Int = 160,
    padding: Int = 0
) -> MLXArray {
    var signal = audio
    if padding > 0 {
        signal = MLX.padded(signal, widths: [.init((0, padding))])
    }

    let window = hanningWindow(size: nFft)
    var spec = stft(audio: signal, window: window, nFft: nFft, hopLength: hopLength)
    if spec.shape[0] > 1 {
        spec = spec[0..<(spec.shape[0] - 1), 0...]
    }
    let magnitudes = MLX.abs(spec).square()

    let filters = s3MelFilters(
        sampleRate: sampleRate,
        nFft: nFft,
        nMels: nMels,
        fMin: 0,
        fMax: Float(sampleRate) / 2.0,
        norm: "slaney",
        melScale: .slaney
    )

    let melSpec = MLX.matmul(filters, magnitudes.transposed())
    var logSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
    logSpec = MLX.log10(logSpec)
    logSpec = MLX.maximum(logSpec, logSpec.max() - MLXArray(Float(8.0)))
    logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))
    return logSpec
}

func s3MakeNonPadMask(lengths: MLXArray, maxLen: Int = 0) -> MLXArray {
    let batch = lengths.shape[0]
    let maxLenVal: Int
    if maxLen > 0 {
        maxLenVal = maxLen
    } else {
        maxLenVal = Int(lengths.max().item(Int32.self))
    }

    let seqRange = MLXArray.arange(0, maxLenVal, dtype: .int32)
    let seqRangeExpand = seqRange.expandedDimensions(axis: 0).broadcasted(to: [batch, maxLenVal])
    let seqLengthExpand = lengths.expandedDimensions(axis: -1)
    let mask = seqRangeExpand .>= seqLengthExpand
    let inverted = MLXArray(Int32(1)) - mask.asType(DType.int32)
    return inverted.asType(DType.bool)
}

func s3MaskToBias(_ mask: MLXArray, dtype: DType = .float32) -> MLXArray {
    precondition(mask.dtype == .bool)
    let casted = mask.asType(dtype)
    return (MLXArray(Float(1.0)) - casted) * MLXArray(Float(-1.0e10))
}

func s3Padding(_ data: [MLXArray]) -> (MLXArray, MLXArray) {
    precondition(!data.isEmpty)
    let lengths = data.map { Int32($0.shape[1]) }
    let maxLen = data.map { $0.shape[1] }.max() ?? 0
    let batch = data.count
    let nMels = data[0].shape[0]

    var padded = MLXArray.zeros([batch, nMels, maxLen], type: Float.self)

    for (i, feat) in data.enumerated() {
        let seqLen = feat.shape[1]
        padded[i, 0..<nMels, 0..<seqLen] = feat
    }

    return (padded, MLXArray(lengths))
}

func s3MergeTokenizedSegments(
    _ tokenizedSegments: [[Int]],
    overlap: Int,
    tokenRate: Int
) -> [Int] {
    var merged: [Int] = []
    let overlapTokens = (overlap / 2) * tokenRate

    for (index, tokens) in tokenizedSegments.enumerated() {
        let left = index == 0 ? 0 : overlapTokens
        let right = index == tokenizedSegments.count - 1 ? tokens.count : max(tokens.count - overlapTokens, left)
        if left < right {
            merged.append(contentsOf: tokens[left..<right])
        }
    }

    return merged
}
