//
//  DACVAEDecoder.swift
//  MLXAudioCodecs
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN

// MARK: - DACVAE LSTM

/// LSTM layer
public class DACVAEStackedLSTM: Module {
    let inputSize: Int
    let hiddenSize: Int
    let numLayers: Int
    @ModuleInfo(key: "layers") var layers: [LSTM]

    public init(inputSize: Int, hiddenSize: Int, numLayers: Int = 1) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        var lstmLayers: [LSTM] = []
        for i in 0..<numLayers {
            let inSize = i == 0 ? inputSize : hiddenSize
            lstmLayers.append(LSTM(inputSize: inSize, hiddenSize: hiddenSize))
        }
        self._layers.wrappedValue = lstmLayers
    }

    public func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil) -> (MLXArray, (MLXArray?, MLXArray?)) {
        var output = x
        var newH: [MLXArray] = []
        var newC: [MLXArray] = []

        for layer in layers {
            let (allH, allC) = layer(output)
            output = allH
            // Keep final timestep for hidden state
            if allH.ndim == 3 {
                newH.append(allH[0..., allH.shape[1] - 1, 0...])
            } else {
                newH.append(allH)
            }
            if allC.ndim == 3 {
                newC.append(allC[0..., allC.shape[1] - 1, 0...])
            } else {
                newC.append(allC)
            }
        }

        let hN = newH.isEmpty ? nil : MLX.stacked(newH, axis: 0)
        let cN = newC.isEmpty ? nil : MLX.stacked(newC, axis: 0)

        return (output, (hN, cN))
    }
}

// MARK: - LSTM Block

/// LSTM block with optional skip connection.
public class DACVAELSTMBlock: Module {
    let skip: Bool
    @ModuleInfo(key: "lstm") var lstm: DACVAEStackedLSTM

    public init(inputSize: Int, hiddenSize: Int, numLayers: Int, skip: Bool = true) {
        self.skip = skip
        self._lstm.wrappedValue = DACVAEStackedLSTM(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (y, _) = lstm(x)
        if skip {
            return y + x
        }
        return y
    }
}

// MARK: - Decoder Block

/// Decoder block with upsampling, residual units, and watermarking paths.
public class DACVAEDecoderBlock: Module {
    let stride: Int
    let strideWM: Int
    let downsamplingFactor: Int

    // Block 0: Snake activation
    @ModuleInfo(key: "block_0") var block0: DACVAESnake1d

    // Block 1: Main upsample ConvTranspose (Snake path)
    @ModuleInfo(key: "block_1") var block1: DACVAEWNConvTranspose1d

    // Block 2: ELU activation (for watermark path)
    @ModuleInfo(key: "block_2") var block2: DACVAEElu

    // Block 3: Watermark upsample ConvTranspose (ELU path)
    @ModuleInfo(key: "block_3") var block3: DACVAEWNConvTranspose1d

    // Block 4: ResidualUnit (Snake, dilation=1)
    @ModuleInfo(key: "block_4") var block4: DACVAEResidualUnit

    // Block 5: ResidualUnit (Snake, dilation=3)
    @ModuleInfo(key: "block_5") var block5: DACVAEResidualUnit

    // Block 6: ResidualUnit (ELU, causal, kernel=3)
    @ModuleInfo(key: "block_6") var block6: DACVAEResidualUnit

    // Block 7: ResidualUnit (ELU, causal, kernel=3)
    @ModuleInfo(key: "block_7") var block7: DACVAEResidualUnit

    // Block 8: ResidualUnit (Snake, dilation=9)
    @ModuleInfo(key: "block_8") var block8: DACVAEResidualUnit

    // Block 10: ELU activation
    @ModuleInfo(key: "block_10") var block10: DACVAEElu

    // Block 11: Downsample Conv for watermark path
    @ModuleInfo(key: "block_11") var block11: DACVAEWNConv1d

    public init(
        inputDim: Int = 16,
        outputDim: Int = 8,
        stride: Int = 1,
        strideWM: Int = 1,
        downsamplingFactor: Int = 3
    ) {
        self.stride = stride
        self.strideWM = strideWM
        self.downsamplingFactor = downsamplingFactor

        // Block 0: Snake activation
        self._block0.wrappedValue = DACVAESnake1d(channels: inputDim)

        // Block 1: Main upsample ConvTranspose (Snake path)
        self._block1.wrappedValue = DACVAEWNConvTranspose1d(
            inChannels: inputDim,
            outChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            causal: false,
            padMode: "none",
            norm: "weight_norm"
        )

        // Block 2: ELU activation (for watermark path)
        self._block2.wrappedValue = DACVAEElu()

        // Block 3: Watermark upsample ConvTranspose (ELU path)
        let wmIn = inputDim / downsamplingFactor
        let wmOut = outputDim / downsamplingFactor
        self._block3.wrappedValue = DACVAEWNConvTranspose1d(
            inChannels: wmIn,
            outChannels: wmOut,
            kernelSize: 2 * strideWM,
            stride: strideWM,
            causal: true,
            padMode: "auto",
            norm: "none"
        )

        // Block 4: ResidualUnit (Snake, dilation=1)
        self._block4.wrappedValue = DACVAEResidualUnit(
            dim: outputDim,
            dilation: 1,
            act: "Snake",
            compress: 1,
            causal: false,
            padMode: "none",
            norm: "weight_norm",
            trueSkip: false
        )

        // Block 5: ResidualUnit (Snake, dilation=3)
        self._block5.wrappedValue = DACVAEResidualUnit(
            dim: outputDim,
            dilation: 3,
            act: "Snake",
            compress: 1,
            causal: false,
            padMode: "none",
            norm: "weight_norm",
            trueSkip: false
        )

        // Block 6: ResidualUnit (ELU, causal, kernel=3)
        self._block6.wrappedValue = DACVAEResidualUnit(
            dim: outputDim / downsamplingFactor,
            kernel: 3,
            dilation: 1,
            act: "ELU",
            compress: 2,
            causal: true,
            padMode: "auto",
            norm: "none",
            trueSkip: true
        )

        // Block 7: ResidualUnit (ELU, causal, kernel=3)
        self._block7.wrappedValue = DACVAEResidualUnit(
            dim: outputDim / downsamplingFactor,
            kernel: 3,
            dilation: 1,
            act: "ELU",
            compress: 2,
            causal: true,
            padMode: "auto",
            norm: "none",
            trueSkip: true
        )

        // Block 8: ResidualUnit (Snake, dilation=9)
        self._block8.wrappedValue = DACVAEResidualUnit(
            dim: outputDim,
            dilation: 9,
            act: "Snake",
            compress: 1,
            causal: false,
            padMode: "none",
            norm: "weight_norm",
            trueSkip: false
        )

        // Block 10: ELU activation
        self._block10.wrappedValue = DACVAEElu()

        // Block 11: Downsample Conv for watermark path
        self._block11.wrappedValue = DACVAEWNConv1d(
            inChannels: wmOut,
            outChannels: wmIn,
            kernelSize: 2 * strideWM,
            stride: strideWM,
            causal: true,
            padMode: "auto",
            norm: "none"
        )
    }

    /// Main forward pass (only uses main path blocks).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Main decoder path: blocks 0, 1, 4, 5, 8
        var h = block0(x)
        h = block1(h)
        h = block4(h)
        h = block5(h)
        h = block8(h)
        return h
    }

    /// Watermark upsample path: blocks 2, 3, 6, 7.
    public func upsampleGroup(_ x: MLXArray) -> MLXArray {
        var h = block2(x)
        h = block3(h)
        h = block6(h)
        h = block7(h)
        return h
    }

    /// Watermark downsample path: blocks 10, 11.
    public func downsampleGroup(_ x: MLXArray) -> MLXArray {
        var h = block10(x)
        h = block11(h)
        return h
    }
}
