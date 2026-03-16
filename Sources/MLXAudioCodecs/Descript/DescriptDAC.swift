import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

public final class DescriptResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]

    public init(dim: Int = 16, dilation: Int = 1) {
        let pad = ((7 - 1) * dilation) / 2
        self._block = ModuleInfo(wrappedValue: [
            DescriptSnake1d(channels: dim),
            DescriptWNConv1d(dim: dim, dilation: dilation, padding: pad),
            DescriptSnake1d(channels: dim),
            DescriptWNConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in block {
            y = (layer as! UnaryLayer).callAsFunction(y)
        }
        let pad = (x.shape[1] - y.shape[1]) / 2
        let residual = pad > 0 ? x[0..., pad..<(x.shape[1] - pad), 0...] : x
        return residual + y
    }
}

extension DescriptWNConv1d {
    convenience init(dim: Int, dilation: Int, padding: Int) {
        self.init(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 7,
            stride: 1,
            padding: padding,
            dilation: dilation
        )
    }
}

public final class DescriptEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]

    public init(dim: Int = 16, stride: Int = 1) {
        self._block = ModuleInfo(wrappedValue: [
            DescriptResidualUnit(dim: dim / 2, dilation: 1),
            DescriptResidualUnit(dim: dim / 2, dilation: 3),
            DescriptResidualUnit(dim: dim / 2, dilation: 9),
            DescriptSnake1d(channels: dim / 2),
            DescriptWNConv1d(
                inChannels: dim / 2,
                outChannels: dim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = (layer as! UnaryLayer).callAsFunction(out)
        }
        return out
    }
}

public final class DescriptEncoder: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]
    public let encDim: Int

    public init(dModel: Int = 64, strides: [Int] = [2, 4, 8, 8], dLatent: Int = 64) {
        var layers: [Module] = [
            DescriptWNConv1d(inChannels: 1, outChannels: dModel, kernelSize: 7, padding: 3)
        ]

        var currentDim = dModel
        for stride in strides {
            currentDim *= 2
            layers.append(DescriptEncoderBlock(dim: currentDim, stride: stride))
        }

        layers.append(DescriptSnake1d(channels: currentDim))
        layers.append(DescriptWNConv1d(inChannels: currentDim, outChannels: dLatent, kernelSize: 3, padding: 1))

        self._block = ModuleInfo(wrappedValue: layers)
        self.encDim = currentDim
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = (layer as! UnaryLayer).callAsFunction(out)
        }
        return out.transposed(0, 2, 1)
    }
}

public final class DescriptDecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]

    public init(inputDim: Int = 16, outputDim: Int = 8, stride: Int = 1) {
        self._block = ModuleInfo(wrappedValue: [
            DescriptSnake1d(channels: inputDim),
            DescriptWNConvTranspose1d(
                inChannels: inputDim,
                outChannels: outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0)),
                outputPadding: 1
            ),
            DescriptResidualUnit(dim: outputDim, dilation: 1),
            DescriptResidualUnit(dim: outputDim, dilation: 3),
            DescriptResidualUnit(dim: outputDim, dilation: 9)
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = (layer as! UnaryLayer).callAsFunction(out)
        }
        return out
    }
}

public final class DescriptDecoder: Module, UnaryLayer {
    @ModuleInfo(key: "model") public var model: [Module]

    public init(inputChannel: Int, channels: Int, rates: [Int], dOut: Int = 1) {
        var layers: [Module] = [
            DescriptWNConv1d(inChannels: inputChannel, outChannels: channels, kernelSize: 7, padding: 3)
        ]

        var outputDim = channels
        for (index, stride) in rates.enumerated() {
            let inputDim = channels / Int(pow(2.0, Double(index)))
            outputDim = channels / Int(pow(2.0, Double(index + 1)))
            layers.append(DescriptDecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: stride))
        }

        layers.append(DescriptSnake1d(channels: outputDim))
        layers.append(DescriptWNConv1d(inChannels: outputDim, outChannels: dOut, kernelSize: 7, padding: 3))
        layers.append(Tanh())

        self._model = ModuleInfo(wrappedValue: layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in model {
            out = (layer as! UnaryLayer).callAsFunction(out)
        }
        return out
    }
}

public struct DescriptEncodedAudio {
    public let codes: MLXArray
    public let originalLength: Int

    public init(codes: MLXArray, originalLength: Int) {
        self.codes = codes
        self.originalLength = originalLength
    }
}

public final class DescriptDAC: Module {
    public let config: DescriptDACConfig
    public let encoderDim: Int
    public let encoderRates: [Int]
    public let latentDim: Int
    public let decoderDim: Int
    public let decoderRates: [Int]
    public let sampleRate: Int
    public let hopLength: Int

    @ModuleInfo(key: "encoder") public var encoder: DescriptEncoder
    @ModuleInfo(key: "quantizer") public var quantizer: DescriptResidualVectorQuantize
    @ModuleInfo(key: "decoder") public var decoder: DescriptDecoder

    public init(config: DescriptDACConfig) {
        self.config = config
        self.encoderDim = config.encoderDim
        self.encoderRates = config.encoderRates
        self.decoderDim = config.decoderDim
        self.decoderRates = config.decoderRates
        self.sampleRate = config.sampleRate

        let resolvedLatent = config.latentDim ?? (config.encoderDim * Int(pow(2.0, Double(config.encoderRates.count))))
        self.latentDim = resolvedLatent
        self.hopLength = config.encoderRates.reduce(1, *)

        self._encoder = ModuleInfo(wrappedValue: DescriptEncoder(
            dModel: config.encoderDim,
            strides: config.encoderRates,
            dLatent: resolvedLatent
        ))
        self._quantizer = ModuleInfo(wrappedValue: DescriptResidualVectorQuantize(
            inputDim: resolvedLatent,
            nCodebooks: config.nCodebooks,
            codebookSize: config.codebookSize,
            codebookDim: config.codebookDim
        ))
        self._decoder = ModuleInfo(wrappedValue: DescriptDecoder(
            inputChannel: resolvedLatent,
            channels: config.decoderDim,
            rates: config.decoderRates
        ))
    }

    public func preprocess(_ audioData: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        let resolvedSampleRate = sampleRate ?? self.sampleRate
        precondition(resolvedSampleRate == self.sampleRate, "Sample rate mismatch: \(resolvedSampleRate) != \(self.sampleRate)")

        let length = audioData.shape[1]
        let paddedLength = Int(ceil(Double(length) / Double(hopLength))) * hopLength
        let rightPad = paddedLength - length
        guard rightPad > 0 else {
            return audioData
        }
        return MLX.padded(audioData, widths: [IntOrPair(0), IntOrPair((0, rightPad)), IntOrPair(0)])
    }

    public func encode(_ audioData: MLXArray, nQuantizers: Int? = nil) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let z = encoder(audioData)
        return quantizer(z, nQuantizers: nQuantizers)
    }

    public func decode(_ z: MLXArray) -> MLXArray {
        decoder(z.transposed(0, 2, 1))
    }

    public func decodeFromCodes(_ codes: MLXArray) -> MLXArray {
        let zQ = quantizer.fromCodes(codes).0
        return decode(zQ)
    }

    public func callAsFunction(
        _ audioData: MLXArray,
        sampleRate: Int? = nil,
        nQuantizers: Int? = nil,
        useRVQ: Bool = true
    ) -> (audio: MLXArray, z: MLXArray, codes: MLXArray, latents: MLXArray, commitmentLoss: MLXArray, codebookLoss: MLXArray) {
        let length = audioData.shape[1]
        let padded = preprocess(audioData, sampleRate: sampleRate)

        let z: MLXArray
        let codes: MLXArray
        let latents: MLXArray
        let commitmentLoss: MLXArray
        let codebookLoss: MLXArray

        if useRVQ {
            let encoded = encode(padded, nQuantizers: nQuantizers)
            z = encoded.0
            codes = encoded.1
            latents = encoded.2
            commitmentLoss = encoded.3
            codebookLoss = encoded.4
        } else {
            z = encoder(padded)
            codes = MLXArray.zeros([padded.shape[0], 0, z.shape[2]], dtype: .int32)
            latents = MLXArray.zeros([padded.shape[0], 0, z.shape[2]], dtype: z.dtype)
            commitmentLoss = MLXArray(0.0)
            codebookLoss = MLXArray(0.0)
        }

        let decoded = decode(z)
        return (
            audio: decoded[0..., 0..<length, 0...],
            z: z,
            codes: codes,
            latents: latents,
            commitmentLoss: commitmentLoss,
            codebookLoss: codebookLoss
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            let sanitizedKey = key
                .replacingOccurrences(of: ".layers.", with: ".")
                .replacingOccurrences(of: ".in_proj.", with: ".inProj.")
                .replacingOccurrences(of: ".out_proj.", with: ".outProj.")
            sanitized[sanitizedKey] = value
        }

        return sanitized
    }

    public static func fromPretrained(
        _ repoId: String,
        cache: HubCache = .default
    ) async throws -> DescriptDAC {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "DescriptDAC",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }

        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(DescriptDACConfig.self, from: configData)

        let model = DescriptDAC(config: config)
        let weightsURL = modelURL.appendingPathComponent("model.safetensors")
        let weights = model.sanitize(weights: try loadArrays(url: weightsURL))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }
}

extension DescriptDAC: AudioCodecModel {
    public typealias EncodedAudio = DescriptEncodedAudio

    public var codecSampleRate: Double? { Double(sampleRate) }

    public func encodeAudio(_ waveform: MLXArray) -> DescriptEncodedAudio {
        let originalLength = waveform.shape[1]
        let processed = preprocess(waveform, sampleRate: sampleRate)
        let (_, codes, _, _, _) = encode(processed)
        return DescriptEncodedAudio(codes: codes, originalLength: originalLength)
    }

    public func decodeAudio(_ input: DescriptEncodedAudio) -> MLXArray {
        let decoded = decodeFromCodes(input.codes)
        return decoded[0..., 0..<input.originalLength, 0...]
    }
}
