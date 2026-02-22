//
//  VoiceEncoder.swift
//  MLXAudio
//
//  LSTM-based voice encoder for speaker embeddings.
//  Ported from mlx-audio Python: chatterbox/voice_encoder/voice_encoder.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Stacked LSTM

/// Multi-layer LSTM matching PyTorch's nn.LSTM(num_layers=N).
class StackedLSTM: Module {
    let inputSize: Int
    let hiddenSize: Int
    let numLayers: Int
    let layers: [LSTM]

    init(inputSize: Int, hiddenSize: Int, numLayers: Int = 1) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        self.layers = (0 ..< numLayers).map { i in
            LSTM(inputSize: i == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize)
        }
    }

    /// Forward pass through stacked LSTM layers.
    ///
    /// - Parameters:
    ///   - x: Input tensor (B, T, inputSize).
    ///   - hidden: Optional tuple (h_0, c_0), each (numLayers, B, hiddenSize).
    /// - Returns: (output, (h_n, c_n)) where output is (B, T, hiddenSize).
    func callAsFunction(_ x: MLXArray, hidden: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray)) {
        var output = x
        var newH = [MLXArray]()
        var newC = [MLXArray]()

        for (i, layer) in layers.enumerated() {
            let h: MLXArray? = hidden.map { $0.0[i] }
            let c: MLXArray? = hidden.map { $0.1[i] }

            let (allH, allC) = layer(output, hidden: h, cell: c)
            output = allH

            // Extract final timestep
            let lastH = allH.ndim == 3 ? allH[0..., -1, 0...] : allH
            let lastC = allC.ndim == 3 ? allC[0..., -1, 0...] : allC
            newH.append(lastH)
            newC.append(lastC)
        }

        let hN = MLX.stacked(newH, axis: 0)
        let cN = MLX.stacked(newC, axis: 0)

        return (output, (hN, cN))
    }
}

// MARK: - Voice Encoder

/// LSTM-based voice encoder for speaker embeddings.
///
/// 3-layer LSTM (40→256) + linear projection (256→256) + L2 normalization.
/// Processes mel spectrogram windows via sliding window inference.
public class VoiceEncoder: Module {
    let hp: VoiceEncoderConfiguration

    @ModuleInfo var lstm: StackedLSTM
    @ModuleInfo var proj: Linear

    // Cosine similarity parameters (not used in inference, but loaded from weights)
    @ParameterInfo(key: "similarity_weight") var similarityWeight: MLXArray
    @ParameterInfo(key: "similarity_bias") var similarityBias: MLXArray

    public init(_ hp: VoiceEncoderConfiguration = .default) {
        self.hp = hp
        self._lstm.wrappedValue = StackedLSTM(
            inputSize: hp.numMels, hiddenSize: hp.veHiddenSize, numLayers: 3
        )
        self._proj.wrappedValue = Linear(hp.veHiddenSize, hp.speakerEmbedSize)
        self._similarityWeight.wrappedValue = MLXArray(Float(10.0))
        self._similarityBias.wrappedValue = MLXArray(Float(-5.0))
    }

    // MARK: - Weight Sanitization

    /// Sanitize PyTorch LSTM weights for MLX.
    ///
    /// Handles:
    /// - `lstm.weight_ih_l0` → `lstm.layers.0.Wx`
    /// - `lstm.weight_hh_l0` → `lstm.layers.0.Wh`
    /// - `lstm.bias_ih_l0` + `lstm.bias_hh_l0` → `lstm.layers.0.bias` (combined)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()
        var biasIH = [Int: MLXArray]()
        var biasHH = [Int: MLXArray]()

        for (key, value) in weights {
            if key.contains("lstm.") {
                // Parse LSTM weight keys like "lstm.weight_ih_l0"
                if let match = key.range(of: #"lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)"#, options: .regularExpression) {
                    let component = String(key[match])
                    let parts = component.split(separator: "_")

                    // Extract layer index (last char of last part)
                    let layerStr = String(parts.last!).replacingOccurrences(of: "l", with: "")
                    guard let layerIdx = Int(layerStr) else {
                        newWeights[key] = value
                        continue
                    }

                    if component.contains("weight_ih") {
                        newWeights["lstm.layers.\(layerIdx).Wx"] = value
                    } else if component.contains("weight_hh") {
                        newWeights["lstm.layers.\(layerIdx).Wh"] = value
                    } else if component.contains("bias_ih") {
                        biasIH[layerIdx] = value
                    } else if component.contains("bias_hh") {
                        biasHH[layerIdx] = value
                    }
                } else {
                    newWeights[key] = value
                }
            } else {
                newWeights[key] = value
            }
        }

        // Combine ih and hh biases (MLX LSTM uses a single combined bias)
        for (layerIdx, ih) in biasIH {
            if let hh = biasHH[layerIdx] {
                newWeights["lstm.layers.\(layerIdx).bias"] = ih + hh
            }
        }

        return newWeights
    }

    // MARK: - Forward Pass

    /// Compute embeddings from a batch of mel spectrogram windows.
    ///
    /// - Parameter mels: Batch of mel spectrograms (B, T, M) where T = vePartialFrames.
    /// - Returns: L2-normalized embeddings (B, speakerEmbedSize).
    public func callAsFunction(_ mels: MLXArray) -> MLXArray {
        // Pass through LSTM
        let (_, (hN, _)) = lstm(mels)

        // Get final hidden state from last layer
        let finalHidden = hN[-1] // (B, H)

        // Project
        var rawEmbeds = proj(finalHidden)

        // ReLU if configured
        if hp.veFinalRelu {
            rawEmbeds = relu(rawEmbeds)
        }

        // L2 normalize
        let norm = MLX.sqrt(MLX.sum(rawEmbeds * rawEmbeds, axis: 1, keepDims: true))
        let embeds = rawEmbeds / (norm + MLXArray(Float(1e-10)))

        return embeds
    }

    // MARK: - Inference

    /// Compute speaker embeddings from full utterance mels using sliding window.
    ///
    /// - Parameters:
    ///   - mels: Mel spectrograms (B, T, M).
    ///   - melLens: Valid mel lengths for each batch item.
    ///   - overlap: Overlap between windows (0–1).
    ///   - minCoverage: Minimum coverage for partial windows.
    /// - Returns: L2-normalized speaker embeddings (B, speakerEmbedSize).
    public func inference(
        mels: MLXArray,
        melLens: [Int],
        overlap: Float = 0.5,
        minCoverage: Float = 0.8
    ) -> MLXArray {
        let frameStep = Int(round(Float(hp.vePartialFrames) * (1 - overlap)))

        var nPartialsList = [Int]()
        var targetLens = [Int]()

        for l in melLens {
            let (nWins, targetN) = getNumWins(nFrames: l, step: frameStep, minCoverage: minCoverage)
            nPartialsList.append(nWins)
            targetLens.append(targetN)
        }

        // Pad mels if needed
        var paddedMels = mels
        let lenDiff = (targetLens.max() ?? 0) - paddedMels.dim(1)
        if lenDiff > 0 {
            let pad = MLX.zeros([paddedMels.dim(0), lenDiff, hp.numMels])
            paddedMels = MLX.concatenated([paddedMels, pad], axis: 1)
        }

        // Extract all partial windows
        var partialList = [MLXArray]()
        for (bIdx, nPartial) in nPartialsList.enumerated() {
            if nPartial > 0 {
                let mel = paddedMels[bIdx] // (T, M)
                for p in 0 ..< nPartial {
                    let start = p * frameStep
                    let end = start + hp.vePartialFrames
                    partialList.append(mel[start ..< end].expandedDimensions(axis: 0))
                }
            }
        }

        let partials = MLX.concatenated(partialList, axis: 0) // (totalPartials, T, M)

        // Forward all partials
        let partialEmbeds = self.callAsFunction(partials)

        // Reduce partial embeds into full embeds (mean per utterance)
        var slices = [0]
        for n in nPartialsList {
            slices.append(slices.last! + n)
        }

        var rawEmbeds = [MLXArray]()
        for i in 0 ..< nPartialsList.count {
            let start = slices[i]
            let end = slices[i + 1]
            rawEmbeds.append(MLX.mean(partialEmbeds[start ..< end], axis: 0))
        }
        let stacked = MLX.stacked(rawEmbeds)

        // L2 normalize
        let norm = MLX.sqrt(MLX.sum(stacked * stacked, axis: 1, keepDims: true))
        return stacked / (norm + MLXArray(Float(1e-10)))
    }

    // MARK: - Helpers

    private func getNumWins(nFrames: Int, step: Int, minCoverage: Float) -> (Int, Int) {
        precondition(nFrames > 0)
        let winSize = hp.vePartialFrames
        let (nWins, remainder) = max(nFrames - winSize + step, 0).quotientAndRemainder(dividingBy: step)
        var finalNWins = nWins
        if nWins == 0 || Float(remainder + (winSize - step)) / Float(winSize) >= minCoverage {
            finalNWins += 1
        }
        let targetN = winSize + step * (finalNWins - 1)
        return (finalNWins, targetN)
    }
}
