//
//  SopranoDecoder.swift
//  MLXAudio
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXAudioCodecs
import MLXNN

// MARK: - Interpolate 1D

/// 1D linear interpolation for upscaling hidden states.
///
/// - Parameters:
///   - input: Input tensor of shape (B, C, L)
///   - size: Target output size
///   - alignCorners: If true, align corners of input and output tensors
/// - Returns: Interpolated tensor of shape (B, C, size)
func interpolate1d(_ input: MLXArray, size: Int, alignCorners: Bool = true) -> MLXArray {
    let batchSize = input.shape[0]
    let channels = input.shape[1]
    let inWidth = input.shape[2]

    // Handle edge cases
    if size < 1 || inWidth < 1 {
        return input
    }

    if size == inWidth {
        return input
    }

    // Handle single element input
    if inWidth == 1 {
        return MLX.broadcast(input, to: [batchSize, channels, size])
    }

    // Compute sampling positions
    let x: MLXArray
    if alignCorners && size > 1 {
        // align_corners mode: endpoints match exactly
        x = MLXArray(Array(stride(from: Float(0), to: Float(size), by: 1)))
            * (Float(inWidth - 1) / Float(size - 1))
    } else {
        // Default mode
        if size == 1 {
            x = MLXArray([Float(0)])
        } else {
            let scale = Float(inWidth) / Float(size)
            x = MLXArray(Array(stride(from: Float(0), to: Float(size), by: 1)))
                * scale + 0.5 * scale - 0.5
        }
    }

    // Compute interpolation indices and weights
    let xLow = floor(x).asType(.int32)
    let xHigh = minimum(xLow + 1, MLXArray(Int32(inWidth - 1)))
    let xFrac = x - xLow.asType(.float32)

    // Gather values at low and high indices
    // input shape: (B, C, L), xLow/xHigh shape: (size,)
    // We need to gather along axis 2

    // Reshape for broadcasting: (1, 1, size)
    let xLowExpanded = xLow.reshaped([1, 1, size])
    let xHighExpanded = xHigh.reshaped([1, 1, size])
    let xFracExpanded = xFrac.reshaped([1, 1, size])

    // Use take along axis for gathering
    let yLow = take(input, xLowExpanded.squeezed(axes: [0, 1]), axis: 2)
    let yHigh = take(input, xHighExpanded.squeezed(axes: [0, 1]), axis: 2)

    // Linear interpolation
    let output = yLow * (1 - xFracExpanded) + yHigh * xFracExpanded

    return output
}

// MARK: - ISTFT Head

/// ISTFT Head for converting decoder output to audio waveforms.
///
/// Predicts magnitude and phase, then uses ISTFT to reconstruct audio.
public class ISTFTHead: Module {
    let nFft: Int
    let hopLength: Int
    let out: Linear

    public init(dim: Int, nFft: Int, hopLength: Int) {
        self.nFft = nFft
        self.hopLength = hopLength
        // Output n_fft + 2 for magnitude and phase (n_fft/2 + 1 each)
        self.out = Linear(dim, nFft + 2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Project to STFT coefficients: (B, L, C) -> (B, L, n_fft+2)
        var h = out(x)

        // Transpose: (B, L, n_fft+2) -> (B, n_fft+2, L)
        h = h.swappedAxes(1, 2)

        // Split into magnitude and phase
        let halfSize = (nFft + 2) / 2
        let mag = exp(h[0..., 0..<halfSize, 0...])
        let clippedMag = clip(mag, max: MLXArray(Float(1e2)))
        let phase = h[0..., halfSize..., 0...]

        // Construct complex STFT: S = mag * e^(i * phase)
        let cosPhase = cos(phase)
        let sinPhase = sin(phase)

        // For ISTFT, we need complex representation
        // Create complex array: real + i*imag
        let stftReal = clippedMag * cosPhase
        let stftImag = clippedMag * sinPhase

        // Perform ISTFT
        let audio = performISTFT(real: stftReal, imag: stftImag)

        return audio
    }

    /// Perform inverse STFT using overlap-add synthesis.
    /// Matches Python mlx_audio/dsp.py istft implementation.
    private func performISTFT(real: MLXArray, imag: MLXArray) -> MLXArray {
        // real/imag shape: (B, n_fft/2+1, L)
        let batchSize = real.shape[0]
        let numFrames = real.shape[2]

        // Create window (matches Python hanning function)
        let window = hanningWindow(length: nFft)

        // Output length: t = (num_frames - 1) * hop_length + win_length
        let outputLength = (numFrames - 1) * hopLength + nFft

        var outputs: [MLXArray] = []

        for b in 0..<batchSize {
            // Get single batch: (n_fft/2+1, L)
            let realB = real[b]
            let imagB = imag[b]

            // Create complex STFT: S = real + i*imag
            // Shape: (n_fft/2+1, num_frames)
            let complexSpec = realB + MLXArray(real: Float(0), imaginary: Float(1)) * imagB

            // Perform IRFFT along axis 0 (frequency axis): (n_fft/2+1, L) -> (n_fft, L)
            let framesFreq = MLXFFT.irfft(complexSpec, axis: 0)

            // Transpose to get (num_frames, n_fft) - matches Python's transpose(1, 0)
            let framesTime = framesFreq.transposed(1, 0)

            // Apply window to frames: (num_frames, n_fft) * (n_fft,) = (num_frames, n_fft)
            let windowedFrames = framesTime * window

            // Overlap-add synthesis
            var audioSamples = [Float](repeating: 0, count: outputLength)
            var windowSum = [Float](repeating: 0, count: outputLength)

            let windowArray = window.asArray(Float.self)

            // Process each frame - overlap-add
            for i in 0..<numFrames {
                let start = i * hopLength
                let frameData = windowedFrames[i].asArray(Float.self)

                for j in 0..<min(nFft, frameData.count) {
                    if start + j < outputLength {
                        audioSamples[start + j] += frameData[j]
                        // Use window values (not squared) for normalization - matches Python
                        windowSum[start + j] += windowArray[j]
                    }
                }
            }

            // Normalize by window sum (avoid division by zero)
            // Matches: reconstructed = mx.where(window_sum != 0, reconstructed / window_sum, reconstructed)
            for i in 0..<outputLength {
                if windowSum[i] != 0 {
                    audioSamples[i] /= windowSum[i]
                }
            }

            // Trim to remove center padding: reconstructed[win_length // 2 : -win_length // 2]
            let trimStart = nFft / 2
            let trimEnd = outputLength - nFft / 2
            let trimmedAudio: [Float]
            if trimEnd > trimStart {
                trimmedAudio = Array(audioSamples[trimStart..<trimEnd])
            } else {
                trimmedAudio = audioSamples
            }

            outputs.append(MLXArray(trimmedAudio))
        }

        // Stack outputs
        if outputs.count == 1 {
            return outputs[0].expandedDimensions(axis: 0)
        }
        return MLX.stacked(outputs, axis: 0)
    }

    /// Generate Hanning window
    private func hanningWindow(length: Int) -> MLXArray {
        if length == 1 {
            return MLXArray([Float(1.0)])
        }
        let n = Array(stride(from: 0, to: length, by: 1)).map { Float($0) }
        let factor = Float.pi / Float(length - 1)
        let window = n.map { 0.5 - 0.5 * cos(2.0 * factor * $0) }
        return MLXArray(window)
    }
}

// MARK: - Soprano Decoder

/// Soprano Decoder that converts LLM hidden states to audio waveforms.
///
/// Uses a Vocos-like architecture with ConvNeXt blocks and ISTFT head.
public class SopranoDecoder: Module {
    let decoderInitialChannels: Int
    let upscale: Int
    let decoder: VocosBackbone
    let head: ISTFTHead

    public init(
        numInputChannels: Int = 512,
        decoderNumLayers: Int = 8,
        decoderDim: Int = 768,
        decoderIntermediateDim: Int? = nil,
        hopLength: Int = 512,
        nFft: Int = 2048,
        upscale: Int = 4,
        inputKernel: Int = 1,
        dwKernel: Int = 3
    ) {
        self.decoderInitialChannels = numInputChannels
        self.upscale = upscale

        let intermediateDim = decoderIntermediateDim ?? (decoderDim * 3)

        self.decoder = VocosBackbone(
            inputChannels: numInputChannels,
            dim: decoderDim,
            intermediateDim: intermediateDim,
            numLayers: decoderNumLayers,
            inputKernelSize: inputKernel,
            dwKernelSize: dwKernel
        )

        self.head = ISTFTHead(
            dim: decoderDim,
            nFft: nFft,
            hopLength: hopLength
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x is (B, L, C) in MLX convention
        // For interpolate, we need (B, C, L)
        var h = x.transposed(0, 2, 1)  // (B, C, L)

        let L = h.shape[2]

        // Upscale hidden states
        let targetSize = upscale * (L - 1) + 1
        h = interpolate1d(h, size: targetSize, alignCorners: true)

        // Convert back to MLX convention (B, L, C)
        h = h.transposed(0, 2, 1)

        // Decode through backbone
        h = decoder(h)

        // Convert to audio via ISTFT
        let audio = head(h)

        return audio
    }
}
