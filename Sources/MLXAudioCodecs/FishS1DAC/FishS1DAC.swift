import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

public typealias FishS1TransformerConfigFactory = (_ nLayer: Int, _ nHead: Int, _ dim: Int, _ intermediateSize: Int) -> FishS1DACModelArgs

public final class FishS1ResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]

    public let causal: Bool

    public init(dim: Int = 16, dilation: Int = 1, causal: Bool = false) {
        self.causal = causal
        let pad = ((7 - 1) * dilation) / 2

        let conv1: Module = causal
            ? FishS1CausalWNConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, padding: pad, dilation: dilation)
            : FishS1WNConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, padding: pad, dilation: dilation)
        let conv2: Module = causal
            ? FishS1CausalWNConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
            : FishS1WNConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)

        self._block = ModuleInfo(wrappedValue: [
            FishS1Snake1d(channels: dim),
            conv1,
            FishS1Snake1d(channels: dim),
            conv2
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in block {
            y = fishS1CallUnary(layer, y)
        }

        let pad = x.shape[2] - y.shape[2]
        let residual: MLXArray
        if pad > 0 {
            if causal {
                residual = x[0..., 0..., ..<(x.shape[2] - pad)]
            } else {
                let left = pad / 2
                residual = x[0..., 0..., left..<(x.shape[2] - left)]
            }
        } else {
            residual = x
        }
        return residual + y
    }
}

public final class FishS1EncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]

    public init(
        dim: Int = 16,
        stride: Int = 1,
        causal: Bool = false,
        nTransformerLayers: Int = 0,
        transformerConfigFactory: FishS1TransformerConfigFactory? = nil
    ) {
        let conv: Module = causal
            ? FishS1CausalWNConv1d(
                inChannels: dim / 2,
                outChannels: dim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )
            : FishS1WNConv1d(
                inChannels: dim / 2,
                outChannels: dim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )

        let transformer: Module
        if nTransformerLayers == 0 {
            transformer = FishS1Identity()
        } else {
            precondition(transformerConfigFactory != nil, "Transformer config factory required when encoder transformer layers are enabled")
            transformer = FishS1WindowLimitedTransformer(
                config: transformerConfigFactory!(
                    nTransformerLayers,
                    max(dim / 64, 1),
                    dim,
                    dim * 3
                ),
                inputDim: dim,
                windowSize: 512,
                causal: causal
            )
        }

        self._block = ModuleInfo(wrappedValue: [
            FishS1ResidualUnit(dim: dim / 2, dilation: 1, causal: causal),
            FishS1ResidualUnit(dim: dim / 2, dilation: 3, causal: causal),
            FishS1ResidualUnit(dim: dim / 2, dilation: 9, causal: causal),
            FishS1Snake1d(channels: dim / 2),
            conv,
            transformer
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = fishS1CallUnary(layer, out)
        }
        return out
    }
}

public final class FishS1Encoder: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]
    public let encDim: Int

    public init(
        dModel: Int = 64,
        strides: [Int] = [2, 4, 8, 8],
        dLatent: Int = 64,
        nTransformerLayers: [Int] = [0, 0, 4, 4],
        transformerConfigFactory: FishS1TransformerConfigFactory? = nil,
        causal: Bool = false
    ) {
        let inputConv: Module = causal
            ? FishS1CausalWNConv1d(inChannels: 1, outChannels: dModel, kernelSize: 7, padding: 3)
            : FishS1WNConv1d(inChannels: 1, outChannels: dModel, kernelSize: 7, padding: 3)

        var layers: [Module] = [inputConv]
        var currentDim = dModel

        for (stride, layerCount) in zip(strides, nTransformerLayers) {
            currentDim *= 2
            layers.append(FishS1EncoderBlock(
                dim: currentDim,
                stride: stride,
                causal: causal,
                nTransformerLayers: layerCount,
                transformerConfigFactory: transformerConfigFactory
            ))
        }

        let outputConv: Module = causal
            ? FishS1CausalWNConv1d(inChannels: currentDim, outChannels: dLatent, kernelSize: 3, padding: 1)
            : FishS1WNConv1d(inChannels: currentDim, outChannels: dLatent, kernelSize: 3, padding: 1)

        layers.append(FishS1Snake1d(channels: currentDim))
        layers.append(outputConv)

        self._block = ModuleInfo(wrappedValue: layers)
        self.encDim = currentDim
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = fishS1CallUnary(layer, out)
        }
        return out
    }
}

public final class FishS1DecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var block: [Module]

    public init(
        inputDim: Int = 16,
        outputDim: Int = 8,
        stride: Int = 1,
        causal: Bool = false
    ) {
        let convTranspose: Module = causal
            ? FishS1CausalWNConvTranspose1d(
                inChannels: inputDim,
                outChannels: outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )
            : FishS1WNConvTranspose1d(
                inChannels: inputDim,
                outChannels: outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )

        self._block = ModuleInfo(wrappedValue: [
            FishS1Snake1d(channels: inputDim),
            convTranspose,
            FishS1ResidualUnit(dim: outputDim, dilation: 1, causal: causal),
            FishS1ResidualUnit(dim: outputDim, dilation: 3, causal: causal),
            FishS1ResidualUnit(dim: outputDim, dilation: 9, causal: causal)
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = fishS1CallUnary(layer, out)
        }
        return out
    }
}

public final class FishS1Decoder: Module, UnaryLayer {
    @ModuleInfo(key: "model") public var model: [Module]

    public init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        dOut: Int = 1,
        causal: Bool = false
    ) {
        let inputConv: Module = causal
            ? FishS1CausalWNConv1d(inChannels: inputChannel, outChannels: channels, kernelSize: 7, padding: 3)
            : FishS1WNConv1d(inChannels: inputChannel, outChannels: channels, kernelSize: 7, padding: 3)

        var layers: [Module] = [inputConv]
        var outputDim = channels

        for (index, stride) in rates.enumerated() {
            let inputDim = channels / Int(pow(2.0, Double(index)))
            outputDim = channels / Int(pow(2.0, Double(index + 1)))
            layers.append(FishS1DecoderBlock(
                inputDim: inputDim,
                outputDim: outputDim,
                stride: stride,
                causal: causal
            ))
        }

        let outputConv: Module = causal
            ? FishS1CausalWNConv1d(inChannels: outputDim, outChannels: dOut, kernelSize: 7, padding: 3)
            : FishS1WNConv1d(inChannels: outputDim, outChannels: dOut, kernelSize: 7, padding: 3)

        layers.append(FishS1Snake1d(channels: outputDim))
        layers.append(outputConv)
        layers.append(Tanh())

        self._model = ModuleInfo(wrappedValue: layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in model {
            out = fishS1CallUnary(layer, out)
        }
        return out
    }
}

public struct FishS1EncodedAudio {
    public let codes: MLXArray
    public let featureLengths: MLXArray
    public let originalLength: Int

    public init(codes: MLXArray, featureLengths: MLXArray, originalLength: Int) {
        self.codes = codes
        self.featureLengths = featureLengths
        self.originalLength = originalLength
    }
}

public final class FishS1DAC: Module {
    public let encoderDim: Int
    public let encoderRates: [Int]
    public let latentDim: Int
    public let decoderDim: Int
    public let decoderRates: [Int]
    public let sampleRate: Int
    public let hopLength: Int
    public let frameLength: Int
    public let causal: Bool

    @ModuleInfo(key: "encoder") public var encoder: FishS1Encoder
    @ModuleInfo(key: "quantizer") public var quantizer: FishS1DownsampleResidualVectorQuantize
    @ModuleInfo(key: "decoder") public var decoder: FishS1Decoder

    public init(
        encoderDim: Int = 64,
        encoderRates: [Int] = [2, 4, 8, 8],
        latentDim: Int? = nil,
        decoderDim: Int = 1536,
        decoderRates: [Int] = [8, 8, 4, 2],
        quantizer: FishS1DownsampleResidualVectorQuantize,
        sampleRate: Int = 44_100,
        causal: Bool = true,
        encoderTransformerLayers: [Int] = [0, 0, 0, 0],
        decoderTransformerLayers: [Int] = [0, 0, 0, 0],
        transformerConfigFactory: FishS1TransformerConfigFactory? = nil
    ) {
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.sampleRate = sampleRate
        self.causal = causal

        let resolvedLatent = latentDim ?? (encoderDim * Int(pow(2.0, Double(encoderRates.count))))
        self.latentDim = resolvedLatent
        self.hopLength = encoderRates.reduce(1, *)
        self.frameLength = self.hopLength * 4

        self._encoder = ModuleInfo(wrappedValue: FishS1Encoder(
            dModel: encoderDim,
            strides: encoderRates,
            dLatent: resolvedLatent,
            nTransformerLayers: encoderTransformerLayers,
            transformerConfigFactory: transformerConfigFactory,
            causal: causal
        ))
        self._quantizer = ModuleInfo(wrappedValue: quantizer)
        self._decoder = ModuleInfo(wrappedValue: FishS1Decoder(
            inputChannel: resolvedLatent,
            channels: decoderDim,
            rates: decoderRates,
            causal: causal
        ))

        _ = decoderTransformerLayers
    }

    func toNCL(_ audio: MLXArray) -> MLXArray {
        if audio.ndim == 2 {
            return audio.expandedDimensions(axis: 1)
        }
        if audio.ndim == 3 {
            if audio.shape[1] == 1 {
                return audio
            }
            if audio.shape[2] == 1 {
                return audio.transposed(0, 2, 1)
            }
        }
        preconditionFailure("Expected waveform shape [B, T], [B, 1, T], or [B, T, 1], got \(audio.shape)")
    }

    func toBTC(_ audio: MLXArray) -> MLXArray {
        audio.transposed(0, 2, 1)
    }

    func waveformLength(_ audio: MLXArray) -> Int {
        if audio.ndim == 2 {
            return audio.shape[1]
        }
        if audio.ndim == 3, audio.shape[1] == 1 {
            return audio.shape[2]
        }
        return audio.shape[1]
    }

    public func preprocess(_ audioData: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        let resolvedSampleRate = sampleRate ?? self.sampleRate
        precondition(resolvedSampleRate == self.sampleRate, "Sample rate mismatch: \(resolvedSampleRate) != \(self.sampleRate)")

        let ncl = toNCL(audioData)
        let length = ncl.shape[2]
        let rightPad = Int(ceil(Double(length) / Double(hopLength))) * hopLength - length
        guard rightPad > 0 else {
            return ncl
        }
        return MLX.padded(
            ncl,
            widths: [
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair((0, rightPad))
            ]
        )
    }

    public func encode(
        _ audioData: MLXArray,
        audioLengths: MLXArray? = nil,
        nQuantizers: Int? = nil
    ) -> (MLXArray, MLXArray) {
        var ncl = toNCL(audioData)
        let length = ncl.shape[2]
        let rightPad = Int(ceil(Double(length) / Double(frameLength))) * frameLength - length
        if rightPad > 0 {
            ncl = MLX.padded(
                ncl,
                widths: [
                    IntOrPair(0),
                    IntOrPair(0),
                    IntOrPair((0, rightPad))
                ]
            )
        }

        let resolvedLengths = audioLengths ?? MLXArray([Int32(length + rightPad)])
        let z = encoder(ncl)
        let vqResult = quantizer(z, nQuantizers: nQuantizers)
        let featureLengths = MLX.ceil(resolvedLengths.asType(.float32) / Float(frameLength)).asType(.int32)
        return (vqResult.codes, featureLengths)
    }

    public func decode(_ indices: MLXArray, featureLengths: MLXArray) -> (MLXArray, MLXArray) {
        let batchedIndices = indices.ndim == 2 ? indices.expandedDimensions(axis: 0) : indices
        let z = quantizer.decode(batchedIndices)
        let audioLengths = featureLengths.asType(.int32) * Int32(frameLength)
        let decoded = decoder(z)
        let maxLength = Int(audioLengths.asArray(Int32.self).max() ?? 0)
        let trimmed = maxLength > 0 && decoded.shape[2] > maxLength
            ? decoded[0..., 0..., 0..<maxLength]
            : decoded
        return (trimmed, audioLengths)
    }

    public func encodeZQ(_ audioData: MLXArray) -> MLXArray {
        let (indices, _) = encode(audioData)
        let semanticIndices = MLX.clip(
            indices[0..., 0..<1, 0...],
            min: 0,
            max: quantizer.semanticQuantizer.codebookSize - 1
        )
        let semantic = quantizer.semanticQuantizer.fromCodes(semanticIndices).0

        let residual: MLXArray
        if indices.shape[1] > 1 {
            let residualIndices = MLX.clip(
                indices[0..., 1..., 0...],
                min: 0,
                max: quantizer.quantizer.codebookSize - 1
            )
            residual = quantizer.quantizer.fromCodes(residualIndices).0
        } else {
            residual = MLXArray.zeros(semantic.shape, dtype: semantic.dtype)
        }
        return semantic + residual
    }

    public func decodeZQ(_ zQ: MLXArray) -> MLXArray {
        var hidden = fishS1CallUnary(quantizer.postModule, zQ)
        for stage in quantizer.upsample {
            hidden = stage(hidden)
        }
        let decoded = decoder(hidden)
        let expectedLength = zQ.shape[2] * hopLength * quantizer.downsampleFactor.reduce(1, *)
        if decoded.shape[2] > expectedLength {
            return decoded[0..., 0..., 0..<expectedLength]
        }
        return decoded
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func normalizedComponent(_ component: String) -> String {
            guard component.contains("_") else { return component }
            if component == "weight_g" || component == "weight_v" {
                return component
            }
            let parts = component.split(separator: "_")
            guard let head = parts.first else { return component }
            return String(head) + parts.dropFirst().map {
                $0.prefix(1).uppercased() + $0.dropFirst()
            }.joined()
        }

        let wnPrefixes = Set(weights.keys.compactMap { key -> String? in
            let marker = ".conv.parametrizations.weight.original0"
            guard let range = key.range(of: marker) else { return nil }
            return String(key[..<range.lowerBound])
        })

        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var sanitizedKey = key
            if sanitizedKey.hasSuffix(".causal_mask") || sanitizedKey.hasSuffix(".causalMask") {
                continue
            }
            if sanitizedKey.contains(".conv.parametrizations.weight.original0") {
                sanitizedKey = sanitizedKey.replacingOccurrences(
                    of: ".conv.parametrizations.weight.original0",
                    with: ".weight_g"
                )
            } else if sanitizedKey.contains(".conv.parametrizations.weight.original1") {
                sanitizedKey = sanitizedKey.replacingOccurrences(
                    of: ".conv.parametrizations.weight.original1",
                    with: ".weight_v"
                )
            } else if sanitizedKey.hasSuffix(".conv.bias") {
                let prefix = String(sanitizedKey.dropLast(".conv.bias".count))
                if wnPrefixes.contains(prefix) {
                    sanitizedKey = prefix + ".bias"
                }
            } else if sanitizedKey.contains(".parametrizations.weight.original0") {
                sanitizedKey = sanitizedKey.replacingOccurrences(
                    of: ".parametrizations.weight.original0",
                    with: ".weight_g"
                )
            } else if sanitizedKey.contains(".parametrizations.weight.original1") {
                sanitizedKey = sanitizedKey.replacingOccurrences(
                    of: ".parametrizations.weight.original1",
                    with: ".weight_v"
                )
            }
            sanitizedKey = sanitizedKey
                .split(separator: ".")
                .map { part -> String in
                    let component = String(part)
                    return Int(component) == nil ? normalizedComponent(component) : component
                }
                .joined(separator: ".")

            var pathParts = sanitizedKey.split(separator: ".").map(String.init)
            if pathParts.count > 4,
               pathParts[0] == "quantizer",
               (pathParts[1] == "downsample" || pathParts[1] == "upsample"),
               Int(pathParts[2]) != nil
            {
                if pathParts[3] == "0" {
                    pathParts[3] = "conv"
                } else if pathParts[3] == "1" {
                    pathParts[3] = "block"
                }
                sanitizedKey = pathParts.joined(separator: ".")
            }
            out[sanitizedKey] = value
        }
        return out
    }

    private static func buildConfig(from modelURL: URL) -> FishS1DACBuildConfig {
        let configURL = modelURL.appendingPathComponent("config.json")
        let buildConfig: FishS1DACBuildConfig
        if FileManager.default.fileExists(atPath: configURL.path),
           let data = try? Data(contentsOf: configURL) {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            buildConfig = (try? decoder.decode(FishS1DACBuildConfig.self, from: data)) ?? FishS1DACBuildConfig()
        } else {
            buildConfig = FishS1DACBuildConfig()
        }
        return buildConfig
    }

    private static func weightsURL(from modelURL: URL) throws -> URL {
        let codecWeightsURL = modelURL.appendingPathComponent("codec.safetensors")
        let modelWeightsURL = modelURL.appendingPathComponent("model.safetensors")
        let pytorchWeightsURL = modelURL.appendingPathComponent("pytorch_model.safetensors")
        if FileManager.default.fileExists(atPath: codecWeightsURL.path) {
            return codecWeightsURL
        } else if FileManager.default.fileExists(atPath: modelWeightsURL.path) {
            return modelWeightsURL
        } else if FileManager.default.fileExists(atPath: pytorchWeightsURL.path) {
            return pytorchWeightsURL
        } else {
            throw NSError(
                domain: "FishS1DAC",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No MLX-converted weights found in \(modelURL.path)"]
            )
        }
    }

    public static func fromModelDirectory(_ modelURL: URL) throws -> FishS1DAC {
        let buildConfig = buildConfig(from: modelURL)
        let model = buildFishS1DAC(buildConfig)
        let weightsURL = try weightsURL(from: modelURL)

        let weights = model.sanitize(weights: try loadArrays(url: weightsURL))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }

    public static func fromPretrained(
        _ repoId: String,
        cache: HubCache = .default
    ) async throws -> FishS1DAC {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "FishS1DAC",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }

        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        return try fromModelDirectory(modelURL)
    }
}

public func buildFishS1DAC(_ config: FishS1DACBuildConfig = FishS1DACBuildConfig()) -> FishS1DAC {
    let quantizerTransformerArgs = FishS1DACModelArgs(
        blockSize: config.quantizerTransformerBlockSize,
        nLayer: config.quantizerTransformerLayers,
        nHead: config.quantizerTransformerHeads,
        dim: config.quantizerTransformerDim,
        intermediateSize: config.quantizerTransformerIntermediateSize,
        headDim: config.quantizerTransformerHeadDim,
        ropeBase: config.transformerRopeBase,
        normEps: config.transformerNormEps,
        channelsFirst: true
    )

    let makeTransformer = {
        FishS1WindowLimitedTransformer(
            config: quantizerTransformerArgs,
            inputDim: config.latentDim,
            windowSize: config.quantizerWindowSize,
            causal: true
        )
    }

    let quantizer = FishS1DownsampleResidualVectorQuantize(
        inputDim: config.latentDim,
        nCodebooks: config.nCodebooks,
        codebookDim: config.codebookDim,
        codebookSize: config.codebookSize,
        semanticCodebookSize: config.semanticCodebookSize,
        downsampleFactor: config.downsampleFactor,
        downsampleDims: config.downsampleDims,
        preModule: makeTransformer(),
        postModule: makeTransformer()
    )

    let transformerFactory: FishS1TransformerConfigFactory = { nLayer, nHead, dim, intermediateSize in
        FishS1DACModelArgs(
            blockSize: config.transformerBlockSize,
            nLayer: nLayer,
            nHead: nHead,
            dim: dim,
            intermediateSize: fishS1FindMultiple(intermediateSize, 256),
            headDim: config.transformerHeadDim,
            ropeBase: config.transformerRopeBase,
            normEps: config.transformerNormEps,
            channelsFirst: true
        )
    }

    return FishS1DAC(
        encoderDim: config.encoderDim,
        encoderRates: config.encoderRates,
        latentDim: config.latentDim,
        decoderDim: config.decoderDim,
        decoderRates: config.decoderRates,
        quantizer: quantizer,
        sampleRate: config.sampleRate,
        causal: config.causal,
        encoderTransformerLayers: config.encoderTransformerLayers,
        decoderTransformerLayers: config.decoderTransformerLayers,
        transformerConfigFactory: transformerFactory
    )
}

extension FishS1DAC: AudioCodecModel {
    public typealias EncodedAudio = FishS1EncodedAudio

    public var codecSampleRate: Double? { Double(sampleRate) }

    public func encodeAudio(_ waveform: MLXArray) -> FishS1EncodedAudio {
        let originalLength = waveformLength(waveform)
        let (codes, featureLengths) = encode(waveform)
        return FishS1EncodedAudio(
            codes: codes,
            featureLengths: featureLengths,
            originalLength: originalLength
        )
    }

    public func decodeAudio(_ input: FishS1EncodedAudio) -> MLXArray {
        let (decoded, _) = decode(input.codes, featureLengths: input.featureLengths)
        return toBTC(decoded)[0..., 0..<input.originalLength, 0...]
    }
}
