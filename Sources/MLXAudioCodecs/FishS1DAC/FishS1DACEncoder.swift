import Foundation
import MLX
import MLXNN

// MARK: - Residual Unit

/// Residual unit with dilated causal convolutions and Snake activation.
public class FishEncoderResidualUnit: Module {
    @ModuleInfo(key: "block") var block: [Module]
    // We use a sequential-like pattern manually

    let snake1: FishSnake1d
    let conv1: FishCausalWNConv1d
    let snake2: FishSnake1d
    let conv2: FishCausalWNConv1d

    public init(dim: Int, dilation: Int = 1) {
        self.snake1 = FishSnake1d(channels: dim)
        self.conv1 = FishCausalWNConv1d(
            inChannels: dim, outChannels: dim, kernelSize: 7, dilation: dilation
        )
        self.snake2 = FishSnake1d(channels: dim)
        self.conv2 = FishCausalWNConv1d(
            inChannels: dim, outChannels: dim, kernelSize: 1
        )
        self._block.wrappedValue = [snake1, conv1, snake2, conv2]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        h = snake1(h)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return x + h
    }
}

// MARK: - Encoder Block

/// Encoder block with residual units, downsampling conv, and optional transformer.
public class FishEncoderBlock: Module {
    @ModuleInfo(key: "res_units") var resUnits: [FishEncoderResidualUnit]
    @ModuleInfo(key: "snake") var snake: FishSnake1d
    @ModuleInfo(key: "conv") var conv: FishCausalWNConv1d
    @ModuleInfo(key: "transformer") var transformer: FishWindowLimitedTransformer?

    public init(
        inputDim: Int, outputDim: Int, stride: Int,
        nTransformerLayers: Int = 0, config: FishS1DACConfig? = nil
    ) {
        self._resUnits.wrappedValue = [
            FishEncoderResidualUnit(dim: inputDim, dilation: 1),
            FishEncoderResidualUnit(dim: inputDim, dilation: 3),
            FishEncoderResidualUnit(dim: inputDim, dilation: 9),
        ]
        self._snake.wrappedValue = FishSnake1d(channels: inputDim)
        self._conv.wrappedValue = FishCausalWNConv1d(
            inChannels: inputDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride,
            padding: Int(ceil(Float(stride) / 2.0))
        )

        if nTransformerLayers > 0, let config = config {
            var tConfig = config
            tConfig.nLayer = nTransformerLayers
            tConfig.dim = outputDim
            self._transformer.wrappedValue = FishWindowLimitedTransformer(
                config: tConfig,
                inputDim: outputDim,
                outputDim: outputDim,
                channelsFirst: true,
                windowSize: config.blockSize
            )
        } else {
            self._transformer.wrappedValue = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for unit in resUnits {
            h = unit(h)
        }
        h = snake(h)
        h = conv(h)
        if let t = transformer {
            h = t(h)
        }
        return h
    }
}

// MARK: - Encoder

/// Fish S1 DAC Encoder: CausalWNConv1d -> EncoderBlocks -> Snake1d -> CausalWNConv1d.
public class FishS1DACEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: FishCausalWNConv1d
    @ModuleInfo(key: "blocks") var blocks: [FishEncoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: FishSnake1d
    @ModuleInfo(key: "conv_out") var convOut: FishCausalWNConv1d

    public init(config: FishS1DACConfig) {
        // Initial conv: 1 -> encoderDim (64)
        self._convIn.wrappedValue = FishCausalWNConv1d(
            inChannels: 1, outChannels: config.encoderDim,
            kernelSize: 7, padding: 3
        )

        // Encoder blocks with increasing channel dimensions
        var encoderBlocks: [FishEncoderBlock] = []
        var currentDim = config.encoderDim
        for (i, stride) in config.encoderRates.enumerated() {
            let outputDim = currentDim * 2
            let nTLayers = config.encoderTransformerLayers[i]
            encoderBlocks.append(FishEncoderBlock(
                inputDim: currentDim, outputDim: outputDim, stride: stride,
                nTransformerLayers: nTLayers, config: config
            ))
            currentDim = outputDim
        }
        self._blocks.wrappedValue = encoderBlocks

        // Final layers
        self._snakeOut.wrappedValue = FishSnake1d(channels: currentDim)
        self._convOut.wrappedValue = FishCausalWNConv1d(
            inChannels: currentDim, outChannels: config.latentDim,
            kernelSize: 3, padding: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in blocks {
            h = block(h)
        }
        h = snakeOut(h)
        h = convOut(h)
        return h
    }
}
