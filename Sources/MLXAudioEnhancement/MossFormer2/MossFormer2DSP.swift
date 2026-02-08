import Foundation
import MLX
import MLXAudioCore

public enum MossFormer2DSP {
    public static func hammingWindow(size: Int, periodic: Bool = true) -> MLXArray {
        guard size > 0 else { return MLXArray.zeros([0], type: Float.self) }
        if size == 1 { return MLXArray([Float(1.0)]) }

        let effectiveSize = periodic ? size + 1 : size
        let denom = Float(effectiveSize - 1)

        var values = [Float](repeating: 0, count: effectiveSize)
        for n in 0..<effectiveSize {
            let phase = 2.0 * Float.pi * Float(n) / denom
            values[n] = 0.54 - 0.46 * cos(phase)
        }

        if periodic {
            return MLXArray(Array(values.prefix(size)))
        }
        return MLXArray(values)
    }

    public static func stft(
        audio: MLXArray,
        fftLen: Int,
        hopLength: Int,
        winLen: Int,
        window: MLXArray,
        center: Bool = true
    ) -> MLXArray {
        guard fftLen > 0, hopLength > 0, winLen > 0 else {
            return MLXArray.zeros([0, 0], dtype: .complex64)
        }

        var signal = audio
        if center {
            let padAmount = fftLen / 2
            if padAmount > 0 {
                let left = MLXArray.zeros([padAmount], type: Float.self)
                let right = MLXArray.zeros([padAmount], type: Float.self)
                signal = MLX.concatenated([left, signal, right], axis: 0)
            }
        }

        let signalLen = signal.shape[0]
        guard signalLen >= winLen else {
            return MLXArray.zeros([0, fftLen / 2 + 1], dtype: .complex64)
        }

        let numFrames = 1 + (signalLen - winLen) / hopLength
        guard numFrames > 0 else {
            return MLXArray.zeros([0, fftLen / 2 + 1], dtype: .complex64)
        }

        var frames: [MLXArray] = []
        frames.reserveCapacity(numFrames)
        for i in 0..<numFrames {
            let start = i * hopLength
            let end = start + winLen
            frames.append(signal[start..<end])
        }

        var stackedFrames = MLX.stacked(frames, axis: 0)
        let win = adjustedWindow(window, targetLength: winLen)
        stackedFrames = stackedFrames * win

        if winLen < fftLen {
            let rightPad = MLXArray.zeros([numFrames, fftLen - winLen], type: Float.self)
            stackedFrames = MLX.concatenated([stackedFrames, rightPad], axis: 1)
        } else if winLen > fftLen {
            stackedFrames = stackedFrames[0..<numFrames, 0..<fftLen]
        }

        return MLXFFT.rfft(stackedFrames, axis: 1)
    }

    public static func istft(
        real: MLXArray,
        imag: MLXArray,
        fftLen: Int,
        hopLength: Int,
        winLen: Int,
        window: MLXArray,
        center: Bool = false,
        audioLength: Int? = nil
    ) -> MLXArray {
        guard fftLen > 0, hopLength > 0, winLen > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }
        guard real.ndim == 3, imag.ndim == 3, real.shape == imag.shape else {
            return MLXArray.zeros([0], type: Float.self)
        }
        guard real.shape[0] > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        let realT = real[0].transposed(1, 0)
        let imagT = imag[0].transposed(1, 0)
        let complexSpec = realT + MLXArray(real: Float(0), imaginary: Float(1)) * imagT

        var frames = MLXFFT.irfft(complexSpec, axis: 1)
        let numFrames = frames.shape[0]
        guard numFrames > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        let frameWidth = Swift.min(winLen, frames.shape[1])
        if frameWidth <= 0 {
            return MLXArray.zeros([0], type: Float.self)
        }

        frames = frames[0..<numFrames, 0..<frameWidth]
        let synthesisWindow = adjustedWindow(window, targetLength: frameWidth)
        let windowedFrames = frames * synthesisWindow

        let fullLength = (numFrames - 1) * hopLength + frameWidth
        guard fullLength > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        let frameValues = windowedFrames.asArray(Float.self)
        let windowValues = synthesisWindow.asArray(Float.self)
        var output = [Float](repeating: 0, count: fullLength)
        var windowSum = [Float](repeating: 0, count: fullLength)

        // TODO: Vectorize overlap-add using scatter-add (mlx_scatter_add).
        // Approach: precompute flat index buffer mapping frame positions to output indices,
        // then use `array.at(indices).add(values)` for fully on-device accumulation.
        // Reference: Python mlx-audio ISTFTCache uses `output.at[indices].add(updates)`.
        for frameIndex in 0..<numFrames {
            let start = frameIndex * hopLength
            if start >= fullLength { break }
            let maxLen = Swift.min(frameWidth, fullLength - start)
            let base = frameIndex * frameWidth

            for j in 0..<maxLen {
                output[start + j] += frameValues[base + j]
                let w = windowValues[j]
                windowSum[start + j] += w * w
            }
        }

        let eps: Float = 1e-8
        for i in 0..<fullLength {
            let denom = max(windowSum[i], eps)
            output[i] /= denom
        }

        var result = output
        if center {
            let trim = fftLen / 2
            if result.count > 2 * trim {
                result = Array(result[trim..<(result.count - trim)])
            }
        }
        if let audioLength, result.count > audioLength {
            result = Array(result.prefix(audioLength))
        }
        return MLXArray(result)
    }

    public static func computeFbankKaldi(
        audio: MLXArray,
        sampleRate: Int,
        winLen: Int,
        winInc: Int,
        numMels: Int,
        winType: String,
        preemphasis: Float
    ) -> MLXArray {
        guard sampleRate > 0, winLen > 0, winInc > 0, numMels > 0 else {
            return MLXArray.zeros([0, 0], type: Float.self)
        }

        var signal = audio
        let audioLen = signal.shape[0]
        guard audioLen > 0 else {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }

        if preemphasis > 0, audioLen > 1 {
            let first = signal[0..<1]
            let rest = signal[1..<audioLen] - MLXArray(preemphasis) * signal[0..<(audioLen - 1)]
            signal = MLX.concatenated([first, rest], axis: 0)
        }

        let signalLen = signal.shape[0]
        guard signalLen >= winLen else {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }

        let numFrames = 1 + (signalLen - winLen) / winInc
        guard numFrames > 0 else {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }

        var frames: [MLXArray] = []
        frames.reserveCapacity(numFrames)
        for i in 0..<numFrames {
            let start = i * winInc
            frames.append(signal[start..<(start + winLen)])
        }

        var frameTensor = MLX.stacked(frames, axis: 0)
        let lowerType = winType.lowercased()
        let analysisWindow: MLXArray
        if lowerType.contains("hann") {
            analysisWindow = hannWindow(size: winLen, periodic: false)
        } else {
            analysisWindow = hammingWindow(size: winLen, periodic: false)
        }

        frameTensor = frameTensor * analysisWindow
        let powerSpectrum = MLX.abs(MLXFFT.rfft(frameTensor, axis: 1)).square()
        let melBank = melFilterbank(sampleRate: sampleRate, nFft: winLen, numMels: numMels)
        let fbanks = MLX.matmul(powerSpectrum, melBank)
        return MLX.log(MLX.maximum(fbanks, MLXArray(Float(1e-10))))
    }

    public static func computeDeltasKaldi(_ features: MLXArray, winLength: Int = 5) -> MLXArray {
        if features.ndim == 1 {
            let expanded = features.expandedDimensions(axis: 0)
            return computeDeltasKaldi2D(expanded, winLength: winLength).squeezed(axis: 0)
        }
        guard features.ndim == 2 else {
            return features
        }
        return computeDeltasKaldi2D(features, winLength: winLength)
    }

    public static func melFilterbank(sampleRate: Int, nFft: Int, numMels: Int) -> MLXArray {
        guard sampleRate > 0, nFft > 0, numMels > 0 else {
            return MLXArray.zeros([0, 0], type: Float.self)
        }
        return melFilters(sampleRate: sampleRate, nFft: nFft, nMels: numMels)
    }

    private static func adjustedWindow(_ window: MLXArray, targetLength: Int) -> MLXArray {
        guard targetLength > 0 else { return MLXArray.zeros([0], type: Float.self) }
        if window.shape[0] == targetLength { return window }
        if window.shape[0] > targetLength {
            return window[0..<targetLength]
        }
        let rightPad = MLXArray.zeros([targetLength - window.shape[0]], type: Float.self)
        return MLX.concatenated([window, rightPad], axis: 0)
    }

    static func hannWindow(size: Int, periodic: Bool = true) -> MLXArray {
        guard size > 0 else { return MLXArray.zeros([0], type: Float.self) }
        if size == 1 { return MLXArray([Float(1.0)]) }

        let effectiveSize = periodic ? size + 1 : size
        let denom = Float(effectiveSize - 1)
        var values = [Float](repeating: 0, count: effectiveSize)
        for n in 0..<effectiveSize {
            let phase = 2.0 * Float.pi * Float(n) / denom
            values[n] = 0.5 - 0.5 * cos(phase)
        }
        if periodic {
            return MLXArray(Array(values.prefix(size)))
        }
        return MLXArray(values)
    }

    private static func computeDeltasKaldi2D(_ features: MLXArray, winLength: Int) -> MLXArray {
        let channels = features.shape[0]
        let time = features.shape[1]
        if channels <= 0 || time <= 0 {
            return MLXArray.zeros([max(channels, 0), max(time, 0)], type: Float.self)
        }

        let halfWin = max(winLength / 2, 1)
        var denom: Float = 0
        for i in 1...halfWin {
            denom += Float(i * i)
        }
        denom *= 2.0
        if denom <= 0 {
            return MLXArray.zeros([channels, time], type: Float.self)
        }

        // Kernel: [-halfWin..halfWin] / denom for Kaldi-style finite-difference deltas
        let kernelSize = 2 * halfWin + 1
        var kernelValues = [Float](repeating: 0, count: kernelSize)
        for i in (-halfWin)...halfWin {
            kernelValues[i + halfWin] = Float(i) / denom
        }
        let singleKernel = MLXArray(kernelValues).reshaped([1, kernelSize, 1])
        let weight = MLX.repeated(singleKernel, count: channels, axis: 0)

        // [channels, time] → NLC [1, time, channels]
        let nlc = features.transposed(1, 0).expandedDimensions(axis: 0)

        let padded = MLX.padded(
            nlc,
            widths: [.init(0), .init((halfWin, halfWin)), .init(0)],
            mode: .edge
        )

        let convOut = MLX.conv1d(padded, weight, stride: 1, padding: 0, groups: channels)

        return convOut.squeezed(axis: 0).transposed(1, 0)
    }
}
