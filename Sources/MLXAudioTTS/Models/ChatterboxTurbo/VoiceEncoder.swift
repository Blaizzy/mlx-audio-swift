//
//  VoiceEncoder.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore

struct VoiceEncConfig: Hashable, Sendable {
    var numMels: Int = 40
    var sampleRate: Int = 16_000
    var speakerEmbedSize: Int = 256
    var veHiddenSize: Int = 256
    var flattenLstmParams: Bool = false
    var nFft: Int = 400
    var hopSize: Int = 160
    var winSize: Int = 400
    var fMax: Float = 8_000
    var fMin: Float = 0
    var preemphasis: Float = 0
    var melPower: Float = 2.0
    var melType: String = "amp"
    var normalizedMels: Bool = false
    var vePartialFrames: Int = 160
    var veFinalRelu: Bool = true
    var stftMagnitudeMin: Float = 1e-4
}

private enum MelScale {
    case htk
    case slaney
}

private func hzToMel(_ freq: Float, melScale: MelScale) -> Float {
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

private func melToHz(_ mel: Float, melScale: MelScale) -> Float {
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

private func melFilterBank(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float,
    fMax: Float,
    norm: String? = nil,
    melScale: MelScale
) -> MLXArray {
    let nFreqs = nFft / 2 + 1
    let fMaxVal = fMax > 0 ? fMax : Float(sampleRate) / 2.0

    let allFreqs = (0..<nFreqs).map { Float($0) * Float(sampleRate) / Float(nFft) }

    let mMin = hzToMel(fMin, melScale: melScale)
    let mMax = hzToMel(fMaxVal, melScale: melScale)

    let mPtsCount = nMels + 2
    let step = (mMax - mMin) / Float(mPtsCount - 1)
    let mPts = (0..<mPtsCount).map { mMin + Float($0) * step }
    let fPts = mPts.map { melToHz($0, melScale: melScale) }

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

private func preemphasis(_ wav: [Float], coeff: Float) -> [Float] {
    guard coeff > 0, !wav.isEmpty else { return wav }
    var output = wav
    var prev: Float = wav[0]
    for i in 1..<wav.count {
        let current = wav[i]
        output[i] = current - coeff * prev
        prev = current
    }
    return output.map { min(max($0, -1.0), 1.0) }
}

private func melspectrogram(_ wav: [Float], hp: VoiceEncConfig, pad: Bool) -> MLXArray {
    var signal = wav
    if hp.preemphasis > 0 {
        signal = preemphasis(signal, coeff: hp.preemphasis)
    }

    let audio = MLXArray(signal)
    let window = hanningWindow(size: hp.winSize)
    let spec = stft(audio: audio, window: window, nFft: hp.nFft, hopLength: hp.hopSize)
    var magnitudes = MLX.abs(spec)

    if hp.melPower != 1.0 {
        magnitudes = MLX.pow(magnitudes, hp.melPower)
    }

    let melBasis = melFilterBank(
        sampleRate: hp.sampleRate,
        nFft: hp.nFft,
        nMels: hp.numMels,
        fMin: hp.fMin,
        fMax: hp.fMax,
        norm: "slaney",
        melScale: .slaney
    )

    let magnitudesT = magnitudes.transposed()
    var mel = MLX.matmul(melBasis, magnitudesT)

    if hp.melType == "db" {
        mel = 20 * MLX.log10(MLX.maximum(MLXArray(hp.stftMagnitudeMin), mel))
    }

    if hp.normalizedMels {
        let minLevelDb = 20 * log10f(hp.stftMagnitudeMin)
        mel = (mel - MLXArray(minLevelDb)) / MLXArray(-minLevelDb + 15.0)
    }

    if pad {
        let expected = 1 + signal.count / hp.hopSize
        if mel.shape[1] != expected {
            // Keep best-effort shape; STFT padding differences are acceptable.
        }
    }

    return mel
}

private func getNumWins(
    nFrames: Int,
    step: Int,
    minCoverage: Float,
    hp: VoiceEncConfig
) -> (Int, Int) {
    precondition(nFrames > 0)
    let winSize = hp.vePartialFrames
    let remainderBase = max(nFrames - winSize + step, 0)
    let nWins = remainderBase / step
    let remainder = remainderBase % step
    var count = nWins
    if count == 0 || Float(remainder + (winSize - step)) / Float(winSize) >= minCoverage {
        count += 1
    }
    let target = winSize + step * (count - 1)
    return (count, target)
}

private func getFrameStep(
    overlap: Float,
    rate: Float?,
    hp: VoiceEncConfig
) -> Int {
    precondition(overlap >= 0 && overlap < 1)
    let frameStep: Int
    if let rate {
        frameStep = Int(round((Float(hp.sampleRate) / rate) / Float(hp.vePartialFrames)))
    } else {
        frameStep = Int(round(Float(hp.vePartialFrames) * (1 - overlap)))
    }
    precondition(frameStep > 0 && frameStep <= hp.vePartialFrames)
    return frameStep
}

final class VoiceEncoder: Module {
    let hp: VoiceEncConfig

    @ModuleInfo(key: "lstm1") private var lstm1: LSTM
    @ModuleInfo(key: "lstm2") private var lstm2: LSTM
    @ModuleInfo(key: "lstm3") private var lstm3: LSTM
    @ModuleInfo(key: "proj") private var proj: Linear

    private let similarityWeight = MLXArray([Float(10.0)])
    private let similarityBias = MLXArray([Float(-5.0)])

    init(_ hp: VoiceEncConfig = VoiceEncConfig()) {
        self.hp = hp
        self._lstm1.wrappedValue = LSTM(inputSize: hp.numMels, hiddenSize: hp.veHiddenSize)
        self._lstm2.wrappedValue = LSTM(inputSize: hp.veHiddenSize, hiddenSize: hp.veHiddenSize)
        self._lstm3.wrappedValue = LSTM(inputSize: hp.veHiddenSize, hiddenSize: hp.veHiddenSize)
        self._proj.wrappedValue = Linear(hp.veHiddenSize, hp.speakerEmbedSize)
    }

    func callAsFunction(_ mels: MLXArray) -> MLXArray {
        if hp.normalizedMels {
            let minVal = mels.min().item(Float.self)
            let maxVal = mels.max().item(Float.self)
            if minVal < 0 || maxVal > 1 {
                fatalError("Mels outside [0, 1]. Min=\(minVal), Max=\(maxVal)")
            }
        }

        let (x1, _) = lstm1(mels)
        let (x2, _) = lstm2(x1)
        let (x3, _) = lstm3(x2)

        let last = x3[0..., (x3.shape[1] - 1), 0...]
        var rawEmbeds = proj(last)

        if hp.veFinalRelu {
            rawEmbeds = MLX.maximum(rawEmbeds, MLXArray(Float(0)))
        }

        let norm = MLX.sqrt(rawEmbeds.square().sum(axis: 1, keepDims: true)) + 1e-8
        return rawEmbeds / norm
    }

    func inference(
        _ mels: MLXArray,
        melLens: [Int],
        overlap: Float = 0.5,
        rate: Float? = nil,
        minCoverage: Float = 0.8,
        batchSize: Int? = nil
    ) -> MLXArray {
        let frameStep = getFrameStep(overlap: overlap, rate: rate, hp: hp)

        var nPartialsList: [Int] = []
        var targetLens: [Int] = []
        for length in melLens {
            let (count, target) = getNumWins(nFrames: length, step: frameStep, minCoverage: minCoverage, hp: hp)
            nPartialsList.append(count)
            targetLens.append(target)
        }

        let maxTarget = targetLens.max() ?? 0
        let batch = mels.dim(0)
        let melBins = mels.dim(2)

        let melsArray = mels.asArray(Float.self)
        var padded = [Float](repeating: 0, count: batch * maxTarget * melBins)
        for b in 0..<batch {
            let length = melLens[b]
            for t in 0..<length {
                let srcStart = (b * mels.dim(1) + t) * melBins
                let dstStart = (b * maxTarget + t) * melBins
                padded[dstStart..<(dstStart + melBins)] = melsArray[srcStart..<(srcStart + melBins)]
            }
        }

        let paddedMels = MLXArray(padded).reshaped([batch, maxTarget, melBins])

        var partials: [MLXArray] = []
        partials.reserveCapacity(nPartialsList.reduce(0, +))

        for b in 0..<batch {
            for i in 0..<nPartialsList[b] {
                let start = i * frameStep
                let end = start + hp.vePartialFrames
                let slice = paddedMels[b, start..<end, 0..<melBins]
                partials.append(slice)
            }
        }

        let batchSize = batchSize ?? partials.count
        var partialEmbeds: [MLXArray] = []
        var index = 0
        while index < partials.count {
            let end = min(index + batchSize, partials.count)
            let batchPartials = Array(partials[index..<end])
            let stacked = MLX.stacked(batchPartials, axis: 0)
            let embeds = self(stacked)
            partialEmbeds.append(embeds)
            index = end
        }

        let allEmbeds = MLX.concatenated(partialEmbeds, axis: 0)

        var utteranceEmbeds: [MLXArray] = []
        utteranceEmbeds.reserveCapacity(batch)

        var offset = 0
        for count in nPartialsList {
            let slice = allEmbeds[offset..<(offset + count), 0...]
            let meanEmbed = MLX.mean(slice, axis: 0)
            utteranceEmbeds.append(meanEmbed)
            offset += count
        }

        let stacked = MLX.stacked(utteranceEmbeds, axis: 0)
        let norm = MLX.sqrt(stacked.square().sum(axis: 1, keepDims: true)) + 1e-8
        return stacked / norm
    }

    func embedsFromMels(
        _ mels: [MLXArray],
        melLens: [Int]? = nil,
        asSpeaker: Bool = false,
        batchSize: Int = 32,
        overlap: Float = 0.5,
        rate: Float? = nil,
        minCoverage: Float = 0.8
    ) -> MLXArray {
        let lengths = melLens ?? mels.map { $0.dim(0) }
        let packed = packMels(mels)
        let uttEmbeds = inference(
            packed,
            melLens: lengths,
            overlap: overlap,
            rate: rate,
            minCoverage: minCoverage,
            batchSize: batchSize
        )

        if asSpeaker {
            let meanEmbed = MLX.mean(uttEmbeds, axis: 0)
            let norm = MLX.sqrt(MLX.sum(meanEmbed.square())) + 1e-8
            return meanEmbed / norm
        }

        return uttEmbeds
    }

    func embedsFromWavs(
        _ wavs: [[Float]],
        sampleRate: Int,
        asSpeaker: Bool = false,
        batchSize: Int = 32,
        trimTopDb: Float? = 20,
        overlap: Float = 0.5,
        rate: Float? = 1.3,
        minCoverage: Float = 0.8
    ) -> MLXArray {
        var processed: [[Float]] = wavs

        if sampleRate != hp.sampleRate {
            processed = processed.map { resampleLinear($0, from: sampleRate, to: hp.sampleRate) }
        }

        if let trimTopDb {
            processed = processed.map { trimSilence($0, topDb: trimTopDb) }
        }

        let mels = processed.map { melspectrogram($0, hp: hp, pad: true).transposed() }

        return embedsFromMels(
            mels,
            asSpeaker: asSpeaker,
            batchSize: batchSize,
            overlap: overlap,
            rate: rate,
            minCoverage: minCoverage
        )
    }

    private func packMels(_ mels: [MLXArray]) -> MLXArray {
        let maxLen = mels.map { $0.dim(0) }.max() ?? 0
        guard let first = mels.first else { return MLXArray([]) }
        let melBins = first.dim(1)

        var packed = [Float](repeating: 0, count: mels.count * maxLen * melBins)

        for (index, mel) in mels.enumerated() {
            let melData = mel.asArray(Float.self)
            let len = mel.dim(0)
            for t in 0..<len {
                let srcStart = t * melBins
                let dstStart = (index * maxLen + t) * melBins
                packed[dstStart..<(dstStart + melBins)] = melData[srcStart..<(srcStart + melBins)]
            }
        }

        return MLXArray(packed).reshaped([mels.count, maxLen, melBins])
    }
}

private func resampleLinear(_ wav: [Float], from: Int, to: Int) -> [Float] {
    guard from != to, wav.count > 1 else { return wav }
    let ratio = Float(to) / Float(from)
    let newLength = Int(round(Float(wav.count) * ratio))
    guard newLength > 1 else { return wav }

    var output = [Float](repeating: 0, count: newLength)
    for i in 0..<newLength {
        let pos = Float(i) / ratio
        let idx = Int(floor(pos))
        let frac = pos - Float(idx)
        let idxNext = min(idx + 1, wav.count - 1)
        output[i] = wav[idx] * (1 - frac) + wav[idxNext] * frac
    }
    return output
}

private func trimSilence(_ wav: [Float], topDb: Float) -> [Float] {
    guard !wav.isEmpty else { return wav }
    let maxAmp = wav.map { abs($0) }.max() ?? 0
    guard maxAmp > 0 else { return wav }

    let threshold = maxAmp * powf(10.0, -topDb / 20.0)
    var start = 0
    var end = wav.count - 1

    while start < wav.count, abs(wav[start]) < threshold {
        start += 1
    }

    while end > start, abs(wav[end]) < threshold {
        end -= 1
    }

    if start >= end {
        return wav
    }

    return Array(wav[start...end])
}
