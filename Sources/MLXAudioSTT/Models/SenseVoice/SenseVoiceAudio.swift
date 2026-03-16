import Foundation
import MLX
import MLXAudioCore

enum SenseVoiceAudio {
    static func computeFbank(
        _ waveform: MLXArray,
        sampleRate: Int = 16000,
        nMels: Int = 80,
        frameLengthMS: Int = 25,
        frameShiftMS: Int = 10,
        window: String = "hamming"
    ) -> MLXArray {
        let winLength = sampleRate * frameLengthMS / 1000
        let hopLength = sampleRate * frameShiftMS / 1000

        var audio = waveform
        if audio.ndim > 1 {
            audio = audio.mean(axis: -1)
        }
        audio = audio.reshaped([-1]).asType(.float32)

        if audio.shape[0] == 0 {
            return MLXArray.zeros([0, nMels], type: Float.self)
        }

        if abs(audio).max().item(Float.self) <= 1.0 {
            audio = audio * MLXArray(Float(1 << 15))
        }

        return computeKaldiFbank(
            audio,
            sampleRate: sampleRate,
            nMels: nMels,
            winLength: winLength,
            hopLength: hopLength,
            window: window
        )
    }

    static func applyLFR(_ feats: MLXArray, lfrM: Int = 7, lfrN: Int = 6) -> MLXArray {
        let time = feats.shape[0]
        let lfrFrames = Int(ceil(Double(time) / Double(lfrN)))
        let leftPad = max(0, (lfrM - 1) / 2)

        var padded = feats
        if leftPad > 0, time > 0 {
            let pad = MLX.repeated(feats[0..<1], count: leftPad, axis: 0)
            padded = MLX.concatenated([pad, feats], axis: 0)
        }

        let paddedTime = padded.shape[0]
        var frames: [MLXArray] = []
        frames.reserveCapacity(lfrFrames)

        for index in 0..<lfrFrames {
            let start = index * lfrN
            let end = start + lfrM
            let stacked: MLXArray
            if end <= paddedTime {
                stacked = padded[start..<end].reshaped([-1])
            } else {
                let available = padded[start..<paddedTime]
                let padCount = end - paddedTime
                let pad = MLX.repeated(padded[(paddedTime - 1)..<paddedTime], count: padCount, axis: 0)
                stacked = MLX.concatenated([available, pad], axis: 0).reshaped([-1])
            }
            frames.append(stacked)
        }

        return MLX.stacked(frames, axis: 0)
    }

    static func applyCMVN(_ feats: MLXArray, means: MLXArray, istd: MLXArray) -> MLXArray {
        (feats + means) * istd
    }

    static func parseAMMVN(_ url: URL) throws -> (means: [Float], istd: [Float]) {
        let text = try String(contentsOf: url, encoding: .utf8)
        let shiftPattern = #"<AddShift>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]"#
        let rescalePattern = #"<Rescale>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]"#

        let means = try extractBracketedFloatList(text: text, pattern: shiftPattern)
        let istd = try extractBracketedFloatList(text: text, pattern: rescalePattern)
        return (means, istd)
    }

    private static func extractBracketedFloatList(text: String, pattern: String) throws -> [Float] {
        let regex = try NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, options: [], range: range),
              let captureRange = Range(match.range(at: 1), in: text) else {
            throw NSError(
                domain: "SenseVoiceAudio",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to parse CMVN stats with pattern: \(pattern)"]
            )
        }

        return text[captureRange]
            .split(whereSeparator: \.isWhitespace)
            .compactMap { Float($0) }
    }

    private static func computeKaldiFbank(
        _ waveform: MLXArray,
        sampleRate: Int,
        nMels: Int,
        winLength: Int,
        hopLength: Int,
        window: String
    ) -> MLXArray {
        let signalLength = waveform.shape[0]
        guard signalLength >= winLength, winLength > 0, hopLength > 0 else {
            return MLXArray.zeros([0, nMels], type: Float.self)
        }

        let numFrames = 1 + (signalLength - winLength) / hopLength
        var frames = asStrided(waveform, [numFrames, winLength], strides: [hopLength, 1])
        frames = frames - MLX.mean(frames, axis: 1, keepDims: true)

        let preemph = MLXArray(Float(0.97))
        let first = frames[0..., 0..<1] - preemph * frames[0..., 0..<1]
        let rest = frames[0..., 1..<winLength] - preemph * frames[0..., 0..<(winLength - 1)]
        frames = MLX.concatenated([first, rest], axis: 1)

        let windowArray: MLXArray
        if window.lowercased().contains("hann") {
            windowArray = hanningWindow(size: winLength)
        } else {
            windowArray = hammingWindow(size: winLength, periodic: false)
        }
        frames = frames * windowArray

        let fftLength = nextPowerOfTwo(winLength)
        if fftLength > winLength {
            let rightPad = MLXArray.zeros([numFrames, fftLength - winLength], type: Float.self)
            frames = MLX.concatenated([frames, rightPad], axis: 1)
        }

        let powerSpectrum = MLX.abs(MLXFFT.rfft(frames, axis: 1)).square()
        let melBank = melFilters(
            sampleRate: sampleRate,
            nFft: fftLength,
            nMels: nMels,
            fMin: 20.0,
            norm: nil,
            melScale: .htk
        )
        let fbanks = MLX.matmul(powerSpectrum, melBank)
        return MLX.log(MLX.maximum(fbanks, MLXArray(Float(1e-10))))
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        guard value > 1 else { return max(value, 1) }
        var n = 1
        while n < value {
            n <<= 1
        }
        return n
    }
}
