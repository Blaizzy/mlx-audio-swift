//
//  SparkWaveGenerator.swift
//  MLXAudio
//
//  HiFiGAN/DAC-style wave generator for Spark-TTS BiCodec (ports
//  spark/modules/encoder_decoder/wave_generator.py). Reuses the Descript DAC
//  primitives, which are the same classes the Python imports from descript.dac,
//  so weight keys line up. Spark uses explicit transpose-conv kernel sizes
//  (not 2*stride) and transposes to NLC at the boundaries.
//

import Foundation
import MLXAudioCodecs
@preconcurrency import MLX
import MLXNN

private final class SparkDecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") var block: [Module]

    init(inputDim: Int, outputDim: Int, kernelSize: Int, stride: Int) {
        self._block = ModuleInfo(wrappedValue: [
            DescriptSnake1d(channels: inputDim),
            DescriptWNConvTranspose1d(
                inChannels: inputDim, outChannels: outputDim,
                kernelSize: kernelSize, stride: stride,
                padding: (kernelSize - stride) / 2, outputPadding: 0),
            DescriptResidualUnit(dim: outputDim, dilation: 1),
            DescriptResidualUnit(dim: outputDim, dilation: 3),
            DescriptResidualUnit(dim: outputDim, dilation: 9),
        ])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block { out = (layer as! UnaryLayer).callAsFunction(out) }
        return out
    }
}

/// Feature latents [B, input_channel, T] -> waveform [B, 1, T*prod(rates)].
public final class SparkWaveGenerator: Module {
    @ModuleInfo(key: "model") var model: [Module]

    public init(inputChannel: Int, channels: Int, rates: [Int], kernelSizes: [Int], dOut: Int = 1) {
        var layers: [Module] = [
            DescriptWNConv1d(inChannels: inputChannel, outChannels: channels, kernelSize: 7, padding: 3)
        ]
        var outputDim = channels
        for (i, (kernelSize, stride)) in zip(kernelSizes, rates).enumerated() {
            let inputDim = channels / (1 << i)
            outputDim = channels / (1 << (i + 1))
            layers.append(
                SparkDecoderBlock(
                    inputDim: inputDim, outputDim: outputDim,
                    kernelSize: kernelSize, stride: stride))
        }
        layers.append(DescriptSnake1d(channels: outputDim))
        layers.append(DescriptWNConv1d(inChannels: outputDim, outChannels: dOut, kernelSize: 7, padding: 3))
        layers.append(Tanh())
        self._model = ModuleInfo(wrappedValue: layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)   // [B, T, input_channel]
        for layer in model { h = (layer as! UnaryLayer).callAsFunction(h) }
        return h.transposed(0, 2, 1)    // [B, 1, samples]
    }
}
