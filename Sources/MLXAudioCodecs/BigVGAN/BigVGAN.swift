import Foundation
import MLX
import MLXNN

public final class BigVGANAMPBlock1: Module, UnaryLayer {
    @ModuleInfo(key: "convs1") public var convs1: [BigVGANWNConv1d]
    @ModuleInfo(key: "convs2") public var convs2: [BigVGANWNConv1d]
    @ModuleInfo(key: "activations") public var activations: [BigVGANActivation1d]

    public init(
        channels: Int,
        snakeLogscale: Bool,
        activation: BigVGANActivationType,
        kernelSize: Int = 3,
        dilation: [Int] = [1, 3, 5]
    ) {
        self._convs1 = ModuleInfo(wrappedValue: dilation.map { currentDilation in
            BigVGANWNConv1d(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: ((kernelSize - 1) * currentDilation) / 2,
                dilation: currentDilation
            )
        })
        self._convs2 = ModuleInfo(wrappedValue: dilation.map { _ in
            BigVGANWNConv1d(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2
            )
        })

        var activationLayers: [BigVGANActivation1d] = []
        activationLayers.reserveCapacity(dilation.count * 2)
        for _ in dilation {
            activationLayers.append(BigVGANActivation1d(channels: channels, activation: activation, snakeLogscale: snakeLogscale))
            activationLayers.append(BigVGANActivation1d(channels: channels, activation: activation, snakeLogscale: snakeLogscale))
        }
        self._activations = ModuleInfo(wrappedValue: activationLayers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for index in 0..<convs1.count {
            let activation1 = activations[index * 2]
            let activation2 = activations[index * 2 + 1]
            out = out + convs2[index](activation2(convs1[index](activation1(out))))
        }
        return out
    }
}

public final class BigVGANAMPBlock2: Module, UnaryLayer {
    @ModuleInfo(key: "convs") public var convs: [BigVGANWNConv1d]
    @ModuleInfo(key: "activations") public var activations: [BigVGANActivation1d]

    public init(
        channels: Int,
        snakeLogscale: Bool,
        activation: BigVGANActivationType,
        kernelSize: Int = 3,
        dilation: [Int] = [1, 3, 5]
    ) {
        self._convs = ModuleInfo(wrappedValue: dilation.map { currentDilation in
            BigVGANWNConv1d(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: ((kernelSize - 1) * currentDilation) / 2,
                dilation: currentDilation
            )
        })
        self._activations = ModuleInfo(wrappedValue: dilation.map { _ in
            BigVGANActivation1d(channels: channels, activation: activation, snakeLogscale: snakeLogscale)
        })
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for index in 0..<convs.count {
            out = out + convs[index](activations[index](out))
        }
        return out
    }
}

public final class BigVGAN: Module {
    public let config: BigVGANConfig
    public let numKernels: Int
    public let numUpsamples: Int

    @ModuleInfo(key: "conv_pre") public var convPre: BigVGANWNConv1d
    @ModuleInfo(key: "ups") public var ups: [BigVGANUpsampleStage]
    @ModuleInfo(key: "resblocks") public var resblocks: [Module]
    @ModuleInfo(key: "activation_post") public var activationPost: BigVGANActivation1d
    @ModuleInfo(key: "conv_post") public var convPost: BigVGANWNConv1d

    public init(config: BigVGANConfig) {
        self.config = config
        self.numKernels = config.resblockKernelSizes.count
        self.numUpsamples = config.upsampleRates.count

        self._convPre = ModuleInfo(wrappedValue: BigVGANWNConv1d(
            inChannels: config.numMels,
            outChannels: config.upsampleInitialChannel,
            kernelSize: 7,
            stride: 1,
            padding: 3
        ))

        self._ups = ModuleInfo(wrappedValue: zip(config.upsampleRates, config.upsampleKernelSizes).enumerated().map { index, pair in
            let (stride, kernelSize) = pair
            return BigVGANUpsampleStage(conv: BigVGANWNConvTranspose1d(
                inChannels: config.upsampleInitialChannel / (1 << index),
                outChannels: config.upsampleInitialChannel / (1 << (index + 1)),
                kernelSize: kernelSize,
                stride: stride,
                padding: (kernelSize - stride) / 2
            ))
        })

        var blockModules: [Module] = []
        for upsampleIndex in 0..<config.upsampleRates.count {
            let channels = config.upsampleInitialChannel / (1 << (upsampleIndex + 1))
            for (kernelSize, dilation) in zip(config.resblockKernelSizes, config.resblockDilationSizes) {
                switch config.resblock {
                case .one:
                    blockModules.append(BigVGANAMPBlock1(
                        channels: channels,
                        snakeLogscale: config.snakeLogscale,
                        activation: config.activation,
                        kernelSize: kernelSize,
                        dilation: dilation
                    ))
                case .two:
                    blockModules.append(BigVGANAMPBlock2(
                        channels: channels,
                        snakeLogscale: config.snakeLogscale,
                        activation: config.activation,
                        kernelSize: kernelSize,
                        dilation: dilation
                    ))
                }
            }
        }
        self._resblocks = ModuleInfo(wrappedValue: blockModules)

        let finalChannels = config.upsampleInitialChannel / (1 << config.upsampleRates.count)
        self._activationPost = ModuleInfo(wrappedValue: BigVGANActivation1d(
            channels: finalChannels,
            activation: config.activation,
            snakeLogscale: config.snakeLogscale
        ))
        self._convPost = ModuleInfo(wrappedValue: BigVGANWNConv1d(
            inChannels: finalChannels,
            outChannels: 1,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            bias: config.useBiasAtFinal
        ))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x.transposed(0, 2, 1)
        hidden = convPre(hidden)

        for step in 0..<numUpsamples {
            hidden = ups[step](hidden)

            var stageSum = (resblocks[step * numKernels] as! UnaryLayer).callAsFunction(hidden)
            for index in 1..<numKernels {
                let block = resblocks[step * numKernels + index] as! UnaryLayer
                stageSum = stageSum + block.callAsFunction(hidden)
            }
            hidden = stageSum / Float(numKernels)
        }

        hidden = activationPost(hidden)
        hidden = convPost(hidden)
        hidden = config.useTanhAtFinal ? MLX.tanh(hidden) : MLX.clip(hidden, min: -1.0, max: 1.0)
        return hidden.transposed(0, 2, 1)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let currentWeights = Dictionary(uniqueKeysWithValues: parameters().flattened())
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, originalValue) in weights {
            if key.contains("num_batches_tracked") {
                continue
            }

            var value = originalValue
            if let current = currentWeights[key] {
                if (key.contains("conv") || key.contains("ups.")) && value.ndim == 3 && value.shape != current.shape {
                    if key.contains("ups.") {
                        value = value.transposed(1, 2, 0)
                    } else {
                        value = value.transposed(0, 2, 1)
                    }
                } else if value.ndim == 4 && value.shape != current.shape {
                    value = value.transposed(0, 2, 3, 1)
                }
            }

            sanitized[key] = value
        }

        return sanitized
    }
}

extension BigVGAN: AudioDecoderModel {
    public typealias DecoderInput = MLXArray

    public var codecSampleRate: Double? { nil }

    public func decodeAudio(_ input: MLXArray) -> MLXArray {
        callAsFunction(input)
    }
}
