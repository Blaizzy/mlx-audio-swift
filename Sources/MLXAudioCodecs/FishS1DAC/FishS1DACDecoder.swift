import Foundation
import MLX
import MLXNN

// MARK: - Decoder Residual Unit

public class FishDecoderResidualUnit: Module {
    let snake1: FishSnake1d
    let conv1: FishCausalWNConv1d
    let snake2: FishSnake1d
    let conv2: FishCausalWNConv1d

    @ModuleInfo(key: "block") var block: [Module]

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

// MARK: - Decoder Block

/// Decoder block: Snake -> CausalWNConvTranspose1d -> ResidualUnits.
public class FishDecoderBlock: Module {
    @ModuleInfo(key: "snake") var snake: FishSnake1d
    @ModuleInfo(key: "conv") var conv: FishCausalWNConvTranspose1d
    @ModuleInfo(key: "res_units") var resUnits: [FishDecoderResidualUnit]

    public init(inputDim: Int, outputDim: Int, stride: Int) {
        self._snake.wrappedValue = FishSnake1d(channels: inputDim)
        self._conv.wrappedValue = FishCausalWNConvTranspose1d(
            inChannels: inputDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride
        )
        self._resUnits.wrappedValue = [
            FishDecoderResidualUnit(dim: outputDim, dilation: 1),
            FishDecoderResidualUnit(dim: outputDim, dilation: 3),
            FishDecoderResidualUnit(dim: outputDim, dilation: 9),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = conv(h)
        for unit in resUnits {
            h = unit(h)
        }
        return h
    }
}

// MARK: - Decoder

/// Fish S1 DAC Decoder: CausalWNConv1d -> DecoderBlocks -> Snake1d -> CausalWNConv1d -> Tanh.
public class FishS1DACDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: FishCausalWNConv1d
    @ModuleInfo(key: "blocks") var blocks: [FishDecoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: FishSnake1d
    @ModuleInfo(key: "conv_out") var convOut: FishCausalWNConv1d

    public init(config: FishS1DACConfig) {
        // Initial conv: latentDim (1024) -> decoderDim (1536)
        self._convIn.wrappedValue = FishCausalWNConv1d(
            inChannels: config.latentDim, outChannels: config.decoderDim,
            kernelSize: 7, padding: 3
        )

        // Decoder blocks with decreasing channel dimensions
        var decoderBlocks: [FishDecoderBlock] = []
        var currentDim = config.decoderDim
        for stride in config.decoderRates {
            let outputDim = currentDim / 2
            decoderBlocks.append(FishDecoderBlock(
                inputDim: currentDim, outputDim: outputDim, stride: stride
            ))
            currentDim = outputDim
        }
        self._blocks.wrappedValue = decoderBlocks

        // Final layers
        self._snakeOut.wrappedValue = FishSnake1d(channels: currentDim)
        self._convOut.wrappedValue = FishCausalWNConv1d(
            inChannels: currentDim, outChannels: 1,
            kernelSize: 7, padding: 3
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in blocks {
            h = block(h)
        }
        h = snakeOut(h)
        h = convOut(h)
        h = MLX.tanh(h)
        return h
    }
}
