//
//  IncrementalMelSpectrogram.swift
//  MLXAudioSTT
//
//  Created by Prince Canuma on 07/02/2026.
//

import Foundation
import MLX
import MLXAudioCore

/// Computes mel spectrograms incrementally using an overlap-save approach.
///
/// Maintains a rolling buffer of `nFft - hopLength` samples between calls
/// so that STFT frames spanning chunk boundaries are computed correctly.
/// The first chunk uses reflect padding at the start; subsequent chunks
/// overlap with the tail of the previous chunk.
public class IncrementalMelSpectrogram {
    private let nFft: Int
    private let hopLength: Int
    private let nMels: Int
    private let sampleRate: Int

    /// Overlap samples kept between chunks
    private let overlapSize: Int

    /// Pre-computed resources
    private let window: MLXArray
    private let filters: MLXArray

    /// Rolling buffer of leftover samples from previous chunk
    private var overlapBuffer: [Float] = []

    /// Whether this is the first chunk (needs reflect padding)
    private var isFirstChunk: Bool = true

    /// Running max for log normalization (grows monotonically)
    private var runningLogMax: Float = -Float.infinity

    /// Total mel frames produced so far
    private(set) var totalFrames: Int = 0

    public init(
        sampleRate: Int = 16000,
        nFft: Int = 400,
        hopLength: Int = 160,
        nMels: Int = 128
    ) {
        self.sampleRate = sampleRate
        self.nFft = nFft
        self.hopLength = hopLength
        self.nMels = nMels
        self.overlapSize = nFft - hopLength

        self.window = hanningWindow(size: nFft)
        self.filters = melFilters(
            sampleRate: sampleRate,
            nFft: nFft,
            nMels: nMels,
            norm: "slaney"
        )
    }

    /// Process new audio samples and return new mel frames.
    ///
    /// - Parameter samples: Raw audio samples as Float array
    /// - Returns: Mel spectrogram frames `[newFrames, nMels]`, or nil if not enough samples
    public func process(samples: [Float]) -> MLXArray? {
        guard !samples.isEmpty else { return nil }

        let signal: [Float]

        if isFirstChunk {
            // Reflect padding at the start (nFft/2 samples)
            let padSize = nFft / 2
            var prefix: [Float] = []
            if samples.count > 1 {
                let reflectLen = min(padSize, samples.count - 1)
                if reflectLen > 0 {
                    prefix = Array(samples[1...reflectLen].reversed())
                }
            }

            if prefix.isEmpty {
                let fill = samples.first ?? 0
                prefix = [Float](repeating: fill, count: padSize)
            } else if prefix.count < padSize {
                // If samples are shorter than padSize, repeat the reflected prefix
                while prefix.count < padSize {
                    let needed = padSize - prefix.count
                    prefix.append(contentsOf: prefix.prefix(needed))
                }
            }
            signal = prefix + samples
            isFirstChunk = false
        } else {
            // Prepend overlap from previous chunk
            signal = overlapBuffer + samples
        }

        // Calculate how many complete frames we can compute
        let numFrames = max(0, (signal.count - nFft) / hopLength + 1)
        guard numFrames > 0 else {
            // Not enough samples yet - save everything as overlap
            overlapBuffer = signal
            return nil
        }

        // Save leftover samples for next chunk
        let consumedSamples = (numFrames - 1) * hopLength + nFft
        if consumedSamples < signal.count {
            overlapBuffer = Array(signal[(consumedSamples - overlapSize)...])
        } else {
            overlapBuffer = Array(signal.suffix(overlapSize))
        }

        // Compute STFT on the signal
        let signalArray = MLXArray(signal)
        let framesStacked = asStrided(
            signalArray,
            [numFrames, nFft],
            strides: [hopLength, 1],
            offset: 0
        )

        let windowed = framesStacked * window
        let fft = MLXFFT.rfft(windowed, axis: 1)
        let magnitudes = MLX.abs(fft).square()
        eval(magnitudes)

        // Apply mel filterbank
        var melSpec = MLX.matmul(magnitudes, filters)
        eval(melSpec)

        // Log scaling with running max normalization
        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        melSpec = MLX.log10(melSpec)

        let chunkMax = melSpec.max().item(Float.self)
        runningLogMax = max(runningLogMax, chunkMax)

        melSpec = MLX.maximum(melSpec, MLXArray(runningLogMax - 8.0))
        melSpec = (melSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

        totalFrames += numFrames
        return melSpec  // [numFrames, nMels]
    }

    /// Process remaining samples at session end.
    /// Pads with zeros to fill the last frame if needed.
    public func flush() -> MLXArray? {
        guard !overlapBuffer.isEmpty else { return nil }

        // Pad with zeros to make at least one frame
        let needed = nFft
        var signal = overlapBuffer
        if signal.count < needed {
            signal += [Float](repeating: 0, count: needed - signal.count)
        }

        // Add reflect padding at the end
        let padSize = nFft / 2
        let signalLen = signal.count
        let reflectLen = min(padSize, signalLen - 1)
        let suffix = Array(signal[(signalLen - 1 - reflectLen)..<(signalLen - 1)].reversed())
        signal += suffix

        overlapBuffer = []

        let numFrames = max(0, (signal.count - nFft) / hopLength + 1)
        guard numFrames > 0 else { return nil }

        let signalArray = MLXArray(signal)
        let framesStacked = asStrided(
            signalArray,
            [numFrames, nFft],
            strides: [hopLength, 1],
            offset: 0
        )

        let windowed = framesStacked * window
        let fft = MLXFFT.rfft(windowed, axis: 1)
        let magnitudes = MLX.abs(fft).square()
        eval(magnitudes)

        var melSpec = MLX.matmul(magnitudes, filters)
        eval(melSpec)

        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        melSpec = MLX.log10(melSpec)

        let chunkMax = melSpec.max().item(Float.self)
        runningLogMax = max(runningLogMax, chunkMax)

        melSpec = MLX.maximum(melSpec, MLXArray(runningLogMax - 8.0))
        melSpec = (melSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

        totalFrames += numFrames
        return melSpec
    }

    /// Reset state for a new session.
    public func reset() {
        overlapBuffer = []
        isFirstChunk = true
        runningLogMax = -Float.infinity
        totalFrames = 0
    }
}
