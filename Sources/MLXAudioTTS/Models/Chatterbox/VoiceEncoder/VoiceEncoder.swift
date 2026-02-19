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

// MARK: - Voice Encoder

/// LSTM-based voice encoder for speaker embeddings.
///
/// 3-layer LSTM (40→256) + linear projection (256→256) + L2 normalization.
/// Processes mel spectrogram windows via sliding window inference.
///
/// Weight keys use 1-based naming (`lstm1`, `lstm2`, `lstm3`) matching Python MLX
/// serialization of list-stored modules.
public class VoiceEncoder: Module {
    let hp: VoiceEncoderConfiguration

    @ModuleInfo(key: "lstm1") var lstm1: LSTM
    @ModuleInfo(key: "lstm2") var lstm2: LSTM
    @ModuleInfo(key: "lstm3") var lstm3: LSTM
    @ModuleInfo var proj: Linear

    // Cosine similarity parameters (not used in inference, but loaded from weights)
    @ParameterInfo(key: "similarity_weight") var similarityWeight: MLXArray
    @ParameterInfo(key: "similarity_bias") var similarityBias: MLXArray

    public init(_ hp: VoiceEncoderConfiguration = .default) {
        self.hp = hp
        self._lstm1.wrappedValue = LSTM(inputSize: hp.numMels, hiddenSize: hp.veHiddenSize)
        self._lstm2.wrappedValue = LSTM(inputSize: hp.veHiddenSize, hiddenSize: hp.veHiddenSize)
        self._lstm3.wrappedValue = LSTM(inputSize: hp.veHiddenSize, hiddenSize: hp.veHiddenSize)
        self._proj.wrappedValue = Linear(hp.veHiddenSize, hp.speakerEmbedSize)
        self._similarityWeight.wrappedValue = MLXArray(Float(10.0))
        self._similarityBias.wrappedValue = MLXArray(Float(-5.0))
    }

    // MARK: - Weight Sanitization

    /// Sanitize PyTorch LSTM weights for MLX format.
    ///
    /// Handles two weight formats:
    /// 1. **MLX format** (from converted models): `lstm1.Wx`, `lstm2.Wh`, etc. — passed through as-is.
    /// 2. **PyTorch format** (raw checkpoints): `lstm.weight_ih_l0` → `lstm1.Wx`, `lstm.bias_ih_l0` + `lstm.bias_hh_l0` → `lstm1.bias`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Check if weights are already in MLX format (lstm1.Wx, lstm2.Wx, etc.)
        let hasMLXFormat = weights.keys.contains { $0.hasPrefix("lstm1.") || $0.hasPrefix("lstm2.") || $0.hasPrefix("lstm3.") }
        if hasMLXFormat {
            // Already in correct format — pass through
            return weights
        }

        var newWeights = [String: MLXArray]()
        var biasIH = [Int: MLXArray]()
        var biasHH = [Int: MLXArray]()

        for (key, value) in weights {
            // Regular Chatterbox MLX format: "lstm.layers.0.Wx" → "lstm1.Wx"
            if key.hasPrefix("lstm.layers.") {
                let remapped = key
                    .replacingOccurrences(of: "lstm.layers.0", with: "lstm1")
                    .replacingOccurrences(of: "lstm.layers.1", with: "lstm2")
                    .replacingOccurrences(of: "lstm.layers.2", with: "lstm3")
                newWeights[remapped] = value
            }
            // PyTorch format: "lstm.weight_ih_l0" → "lstm1.Wx"
            else if key.contains("lstm.") {
                if let _ = key.range(of: #"lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)"#, options: .regularExpression) {
                    let component = String(key.split(separator: ".").last ?? "")
                    let parts = component.split(separator: "_")
                    let layerStr = String(parts.last!).replacingOccurrences(of: "l", with: "")
                    guard let layerIdx = Int(layerStr) else {
                        newWeights[key] = value
                        continue
                    }

                    let lstmKey = "lstm\(layerIdx + 1)"

                    if component.contains("weight_ih") {
                        newWeights["\(lstmKey).Wx"] = value
                    } else if component.contains("weight_hh") {
                        newWeights["\(lstmKey).Wh"] = value
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
                newWeights["lstm\(layerIdx + 1).bias"] = ih + hh
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
        // Run through 3 LSTM layers sequentially
        let lstmLayers: [LSTM] = [lstm1, lstm2, lstm3]
        var output = mels
        var finalHiddenStates = [MLXArray]()

        for layer in lstmLayers {
            let (allH, allC) = layer(output)
            output = allH

            // Extract final timestep hidden state
            let lastH = allH.ndim == 3 ? allH[0..., -1, 0...] : allH
            finalHiddenStates.append(lastH)
        }

        // Get final hidden state from last layer
        let finalHidden = finalHiddenStates.last! // (B, H)

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
