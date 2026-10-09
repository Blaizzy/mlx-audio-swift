import Foundation
import MLX
import MLXAudioCodecs
import MLXNN

public final class IndexTTSBigVGANConditioning: Module {
    public let config: IndexTTSBigVGANConditioningConfig
    public let numKernels: Int
    public let numUpsamples: Int
    public let conditionInEachUpsamplingLayer: Bool

    @ModuleInfo(key: "conv_pre") public var convPre: BigVGANWNConv1d
    @ModuleInfo(key: "ups") public var ups: [BigVGANUpsampleStage]
    @ModuleInfo(key: "resblocks") public var resblocks: [Module]
    @ModuleInfo(key: "activation_post") public var activationPost: BigVGANActivation1d
    @ModuleInfo(key: "conv_post") public var convPost: BigVGANWNConv1d
    @ModuleInfo(key: "cond_layer") public var condLayer: MLXNN.Conv1d
    @ModuleInfo(key: "conds") public var conds: [MLXNN.Conv1d]
    @ModuleInfo(key: "speaker_encoder") public var speakerEncoder: EcapaTdnnBackbone

    public init(
        config: IndexTTSBigVGANConditioningConfig,
        speakerEncoderConfig: EcapaTdnnConfig? = nil
    ) {
        self.config = config
        self.numKernels = config.resblockKernelSizes.count
        self.numUpsamples = config.upsampleRates.count
        self.conditionInEachUpsamplingLayer = config.condDVectorInEachUpsamplingLayer

        let resblock = BigVGANResBlockType(rawValue: config.resblock) ?? .one
        let activation = BigVGANActivationType(rawValue: config.activation) ?? .snakebeta

        _convPre.wrappedValue = BigVGANWNConv1d(
            inChannels: config.gptDim,
            outChannels: config.upsampleInitialChannel,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )

        _condLayer.wrappedValue = MLXNN.Conv1d(
            inputChannels: config.speakerEmbeddingDim,
            outputChannels: config.upsampleInitialChannel,
            kernelSize: 1
        )

        _speakerEncoder.wrappedValue = EcapaTdnnBackbone(
            config: speakerEncoderConfig ?? Self.defaultSpeakerEncoderConfig(for: config)
        )

        _ups.wrappedValue = zip(config.upsampleRates, config.upsampleKernelSizes).enumerated().map { index, pair in
            let (stride, kernelSize) = pair
            return BigVGANUpsampleStage(conv: BigVGANWNConvTranspose1d(
                inChannels: config.upsampleInitialChannel / (1 << index),
                outChannels: config.upsampleInitialChannel / (1 << (index + 1)),
                kernelSize: kernelSize,
                stride: stride,
                padding: (kernelSize - stride) / 2
            ))
        }

        var blockModules: [Module] = []
        for upsampleIndex in 0..<config.upsampleRates.count {
            let channels = config.upsampleInitialChannel / (1 << (upsampleIndex + 1))
            for (kernelSize, dilation) in zip(config.resblockKernelSizes, config.resblockDilationSizes) {
                switch resblock {
                case .one:
                    blockModules.append(BigVGANAMPBlock1(
                        channels: channels,
                        snakeLogscale: config.snakeLogscale,
                        activation: activation,
                        kernelSize: kernelSize,
                        dilation: dilation
                    ))
                case .two:
                    blockModules.append(BigVGANAMPBlock2(
                        channels: channels,
                        snakeLogscale: config.snakeLogscale,
                        activation: activation,
                        kernelSize: kernelSize,
                        dilation: dilation
                    ))
                }
            }
        }
        _resblocks.wrappedValue = blockModules

        let finalChannels = config.upsampleInitialChannel / (1 << config.upsampleRates.count)
        _activationPost.wrappedValue = BigVGANActivation1d(
            channels: finalChannels,
            activation: activation,
            snakeLogscale: config.snakeLogscale
        )
        _convPost.wrappedValue = BigVGANWNConv1d(
            inChannels: finalChannels,
            outChannels: 1,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            bias: config.useBiasAtFinal
        )

        _conds.wrappedValue = conditionInEachUpsamplingLayer
            ? (0..<config.upsampleRates.count).map { index in
                MLXNN.Conv1d(
                    inputChannels: config.speakerEmbeddingDim,
                    outputChannels: config.upsampleInitialChannel / (1 << (index + 1)),
                    kernelSize: 1
                )
            }
            : []
    }

    public static func defaultSpeakerEncoderConfig(
        for config: IndexTTSBigVGANConditioningConfig
    ) -> EcapaTdnnConfig {
        EcapaTdnnConfig(
            inputSize: config.numMels,
            channels: 512,
            embedDim: config.speakerEmbeddingDim,
            kernelSizes: [5, 3, 3, 3, 1],
            dilations: [1, 2, 3, 4, 1],
            attentionChannels: 128,
            res2netScale: 8,
            seChannels: 128,
            globalContext: true,
            reflectPadding: true
        )
    }

    public func callAsFunction(
        latentStates: MLXArray,
        speakerEmbedding: MLXArray
    ) throws -> MLXArray {
        guard latentStates.ndim == 3, latentStates.dim(2) == config.gptDim else {
            throw IndexTTSError.invalidInput(
                "latentStates must have shape [batch, time, \(config.gptDim)]; got \(latentStates.shape)."
            )
        }
        let speaker = try normalizedSpeakerEmbedding(speakerEmbedding, batchSize: latentStates.dim(0))

        var hidden = convPre(latentStates)
        hidden = hidden + condLayer(speaker)

        for step in 0..<numUpsamples {
            hidden = ups[step](hidden)
            if conditionInEachUpsamplingLayer {
                hidden = hidden + conds[step](speaker)
            }

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

    public func callAsFunction(
        latentStates: MLXArray,
        referenceFeatures: MLXArray
    ) throws -> MLXArray {
        let embedding = try speakerEmbedding(referenceFeatures: referenceFeatures)
        return try self(latentStates: latentStates, speakerEmbedding: embedding)
    }

    public func speakerEmbedding(referenceFeatures: MLXArray) throws -> MLXArray {
        var features = referenceFeatures.asType(.float32)
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }
        guard features.ndim == 3 else {
            throw IndexTTSError.invalidInput(
                "referenceFeatures must have shape [batch, frames, \(config.numMels)] or [batch, \(config.numMels), frames]; got \(referenceFeatures.shape)."
            )
        }
        if features.dim(2) == config.numMels {
            return speakerEncoder(features)
        }
        if features.dim(1) == config.numMels {
            return speakerEncoder(features.transposed(0, 2, 1))
        }
        throw IndexTTSError.invalidInput(
            "referenceFeatures must include \(config.numMels) mel bins; got \(referenceFeatures.shape)."
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let currentWeights = Dictionary(uniqueKeysWithValues: parameters().flattened())
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (originalKey, originalValue) in weights {
            if originalKey.contains("num_batches_tracked") {
                continue
            }
            var key = originalKey
            if key.hasPrefix("bigvgan.") {
                key.removeFirst("bigvgan.".count)
            }
            for index in 0..<numUpsamples {
                key = key.replacingOccurrences(of: "ups.\(index).0.", with: "ups.\(index).conv.")
            }
            key = key
                .replacingOccurrences(of: "speaker_encoder.blocks.0.", with: "speaker_encoder.block0.")
                .replacingOccurrences(of: "speaker_encoder.blocks.1.", with: "speaker_encoder.block1.")
                .replacingOccurrences(of: "speaker_encoder.blocks.2.", with: "speaker_encoder.block2.")
                .replacingOccurrences(of: "speaker_encoder.blocks.3.", with: "speaker_encoder.block3.")
                .replacingOccurrences(of: "norm.norm", with: "norm")
                .replacingOccurrences(of: "conv.conv", with: "conv")
                .replacingOccurrences(of: "conv1.conv", with: "conv1")
                .replacingOccurrences(of: "conv2.conv", with: "conv2")
                .replacingOccurrences(of: "fc.conv", with: "fc")
                .replacingOccurrences(of: "asp_bn.norm", with: "asp_bn")

            guard currentWeights[key] != nil else {
                continue
            }

            var value = originalValue
            if let current = currentWeights[key], value.shape != current.shape {
                if key.contains("ups."), value.ndim == 3 {
                    value = value.transposed(1, 2, 0)
                } else if value.ndim == 3 {
                    value = value.transposed(0, 2, 1)
                } else if value.ndim == 4 {
                    value = value.transposed(0, 2, 3, 1)
                }
            }
            sanitized[key] = value
        }
        return sanitized
    }

    private func normalizedSpeakerEmbedding(_ embedding: MLXArray, batchSize: Int) throws -> MLXArray {
        var speaker = embedding.asType(.float32)
        if speaker.ndim == 2 {
            speaker = speaker.expandedDimensions(axis: 1)
        }
        guard speaker.ndim == 3,
              speaker.dim(0) == batchSize,
              speaker.dim(1) == 1,
              speaker.dim(2) == config.speakerEmbeddingDim
        else {
            throw IndexTTSError.invalidInput(
                "speakerEmbedding must have shape [batch, \(config.speakerEmbeddingDim)] or [batch, 1, \(config.speakerEmbeddingDim)]; got \(embedding.shape)."
            )
        }
        return speaker
    }
}
