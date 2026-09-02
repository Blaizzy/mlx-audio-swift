import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore

struct DramaBoxReferenceAudio {
    var waveform: MLXArray
    var sampleRate: Int
}

struct DramaBoxAudioProcessor {
    var targetSampleRate: Int
    var nFft: Int
    var winLength: Int
    var hopLength: Int
    var nMels: Int
    var fMin: Float
    var fMax: Float
    let window: MLXArray
    let melFilterbank: MLXArray

    init(
        targetSampleRate: Int = 16_000,
        nFft: Int = 1024,
        winLength: Int = 1024,
        hopLength: Int = 160,
        nMels: Int = 64,
        fMin: Float = 0,
        fMax: Float? = nil
    ) {
        self.targetSampleRate = targetSampleRate
        self.nFft = nFft
        self.winLength = winLength
        self.hopLength = hopLength
        self.nMels = nMels
        self.fMin = fMin
        self.fMax = fMax ?? Float(targetSampleRate) / 2
        self.window = dramaBoxPeriodicHann(winLength)
        self.melFilterbank = melFilters(
            sampleRate: targetSampleRate,
            nFft: nFft,
            nMels: nMels,
            fMin: fMin,
            fMax: self.fMax,
            norm: "slaney",
            melScale: .slaney
        )
    }

    func waveformToMel(_ waveform: MLXArray, sampleRate: Int) throws -> MLXArray {
        guard waveform.ndim == 3 else {
            throw DramaBoxError.invalidAudioShape(waveform.shape)
        }
        var wav = waveform.asType(.float32)
        if sampleRate != targetSampleRate {
            wav = try dramaBoxResampleLastAxis(wav, from: sampleRate, to: targetSampleRate)
        }
        let B = wav.dim(0)
        let C = wav.dim(1)
        var mels: [MLXArray] = []
        mels.reserveCapacity(B * C)
        for b in 0..<B {
            for c in 0..<C {
                let samples = wav[b, c]
                let freqs = stft(
                    audio: samples,
                    window: window,
                    nFft: nFft,
                    hopLength: hopLength,
                    padMode: .reflect
                )
                let magnitude = MLX.abs(freqs)
                let mel = MLX.matmul(magnitude, melFilterbank)
                let logMel = MLX.log(MLX.maximum(mel, MLXArray(Float(1e-5))))
                mels.append(logMel)
            }
        }
        let stackedMel = stacked(mels, axis: 0)
        return stackedMel.reshaped(B, C, stackedMel.dim(1), stackedMel.dim(-1))
    }
}

func dramaBoxPeriodicHann(_ size: Int) -> MLXArray {
    var window = [Float](repeating: 0, count: size)
    for n in 0..<size {
        window[n] = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(size))
    }
    return MLXArray(window)
}

func dramaBoxResampleLastAxis(_ waveform: MLXArray, from source: Int, to target: Int) throws -> MLXArray {
    if source == target { return waveform.asType(.float32) }
    let samples = waveform.asArray(Float.self)
    let leading = waveform.shape.dropLast().reduce(1, *)
    let length = waveform.dim(-1)
    var outRows: [[Float]] = []
    outRows.reserveCapacity(leading)
    let duration = Double(length) / Double(source)
    let targetSamples = max(1, Int((duration * Double(target)).rounded()))
    let srcPos = (0..<length).map { Double($0) * duration / Double(length) }
    let dstPos = (0..<targetSamples).map { Double($0) * duration / Double(targetSamples) }
    for row in 0..<leading {
        let start = row * length
        let rowSamples = Array(samples[start ..< (start + length)])
        var dest = [Float](repeating: 0, count: targetSamples)
        var j = 0
        for i in 0..<targetSamples {
            let t = dstPos[i]
            while j + 1 < srcPos.count && srcPos[j + 1] <= t { j += 1 }
            if j + 1 >= srcPos.count {
                dest[i] = rowSamples[length - 1]
            } else {
                let x0 = srcPos[j]
                let x1 = srcPos[j + 1]
                let w = x1 > x0 ? (t - x0) / (x1 - x0) : 0
                dest[i] = Float((1 - w) * Double(rowSamples[j]) + w * Double(rowSamples[j + 1]))
            }
        }
        outRows.append(dest)
    }
    let flat = outRows.flatMap { $0 }
    var shape = waveform.shape
    shape[shape.count - 1] = targetSamples
    return MLXArray(flat).reshaped(shape)
}

func dramaBoxForceStereo(_ waveform: MLXArray) -> MLXArray {
    var x = waveform.asType(.float32)
    if x.ndim == 1 {
        x = stacked([x, x], axis: 0).expandedDimensions(axis: 0)
        return x
    }
    if x.ndim == 2 {
        if x.dim(0) <= 8 && x.dim(0) < x.dim(1) {
            if x.dim(0) == 1 {
                let mono = x[0]
                return stacked([mono, mono], axis: 0).expandedDimensions(axis: 0)
            }
            if x.dim(0) == 2 {
                return x.expandedDimensions(axis: 0)
            }
            let mono = MLX.mean(x, axis: 0)
            return stacked([mono, mono], axis: 0).expandedDimensions(axis: 0)
        }
        let frames = x.dim(0)
        let ch = x.dim(1)
        if ch == 1 {
            let mono = x[0..., 0]
            return stacked([mono, mono], axis: 0).expandedDimensions(axis: 0)
        }
        if ch >= 2 {
            return x[0..., 0..<2].transposed(1, 0).expandedDimensions(axis: 0)
        }
        _ = frames
    }
    if x.ndim == 3 {
        if x.dim(1) == 2 { return x }
        if x.dim(1) == 1 {
            let mono = x[0..., 0, 0...]
            return stacked([mono, mono], axis: 1)
        }
        let mono = MLX.mean(x, axis: 1)
        return stacked([mono, mono], axis: 1)
    }
    return x
}

func dramaBoxCropOrLoop(_ waveform: MLXArray, targetSamples: Int) -> MLXArray {
    let T = waveform.dim(-1)
    if T == targetSamples { return waveform }
    if T > targetSamples {
        return waveform[.ellipsis, 0..<targetSamples]
    }
    let repeats = Int(ceil(Double(targetSamples) / Double(max(T, 1))))
    let tiled = concatenated(Array(repeating: waveform, count: repeats), axis: -1)
    return tiled[.ellipsis, 0..<targetSamples]
}

func dramaBoxPeakNormalize(_ waveform: MLXArray, targetDBFS: Float = -4) throws -> MLXArray {
    let peak = MLX.abs(waveform).max().item(Float.self)
    guard peak > 0 else { throw DramaBoxError.silentReferenceAudio }
    let targetPeak = pow(10.0 as Float, targetDBFS / 20.0)
    return waveform * (targetPeak / peak)
}

func prepareDramaBoxReferenceAudio(
    _ audio: MLXArray,
    sampleRate: Int,
    refDurationS: Float = 10,
    targetSampleRate: Int = 16_000,
    targetPeakDBFS: Float = -4
) throws -> DramaBoxReferenceAudio {
    var waveform = dramaBoxForceStereo(audio)
    waveform = try dramaBoxResampleLastAxis(waveform, from: sampleRate, to: targetSampleRate)
    let targetSamples = Int((refDurationS * Float(targetSampleRate)).rounded())
    waveform = dramaBoxCropOrLoop(waveform, targetSamples: targetSamples)
    waveform = try dramaBoxPeakNormalize(waveform, targetDBFS: targetPeakDBFS)
    return DramaBoxReferenceAudio(waveform: waveform.asType(.float32), sampleRate: targetSampleRate)
}

func prepareDramaBoxReferenceAudio(
    url: URL,
    refDurationS: Float = 10,
    targetSampleRate: Int = 16_000
) throws -> DramaBoxReferenceAudio {
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    let frameCount = AVAudioFrameCount(file.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw DramaBoxError.invalidAudioShape([])
    }
    try file.read(into: buffer)
    guard let data = buffer.floatChannelData else {
        throw DramaBoxError.invalidAudioShape([])
    }
    let n = Int(buffer.frameLength)
    let ch = Int(format.channelCount)
    var channels: [MLXArray] = []
    for c in 0..<ch {
        channels.append(MLXArray(Array(UnsafeBufferPointer(start: data[c], count: n))))
    }
    let stacked: MLXArray
    if channels.count == 1 {
        stacked = channels[0]
    } else {
        stacked = MLX.stacked(channels, axis: 0)
    }
    return try prepareDramaBoxReferenceAudio(
        stacked,
        sampleRate: Int(format.sampleRate),
        refDurationS: refDurationS,
        targetSampleRate: targetSampleRate
    )
}
