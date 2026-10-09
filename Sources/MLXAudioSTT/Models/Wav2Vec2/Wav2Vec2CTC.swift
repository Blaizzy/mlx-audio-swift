import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

private func wav2Vec2Activation(_ x: MLXArray, name: String) -> MLXArray {
    switch name.lowercased() {
    case "relu":
        return relu(x)
    case "silu", "swish":
        return silu(x)
    default:
        return gelu(x)
    }
}

private final class Wav2Vec2STTConvNorm: Module {
    enum Kind {
        case layer
        case group
    }

    let kind: Kind
    let groupCount: Int
    let dimensions: Int
    let eps: Float

    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray

    init(kind: Kind, groupCount: Int, dimensions: Int, eps: Float) {
        self.kind = kind
        self.groupCount = groupCount
        self.dimensions = dimensions
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        _bias.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized: MLXArray
        switch kind {
        case .layer:
            normalized = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
        case .group:
            normalized = pytorchGroupNorm(x)
        }
        return weight.asType(x.dtype) * normalized + bias.asType(x.dtype)
    }

    private func pytorchGroupNorm(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let dims = x.dim(-1)
        let rest = x.shape.dropFirst().dropLast()
        let groupSize = dims / groupCount

        var out = x.reshaped(batch, -1, groupCount, groupSize)
        out = out.transposed(0, 2, 1, 3).reshaped(batch, groupCount, -1)
        out = MLXFast.layerNorm(out, weight: nil, bias: nil, eps: eps)
        out = out.reshaped(batch, groupCount, -1, groupSize)
        return out.transposed(0, 2, 1, 3).reshaped([batch] + rest + [dims])
    }
}

private final class Wav2Vec2STTConvLayer: Module {
    enum Normalization {
        case none
        case layer
        case group
    }

    let activation: String

    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var norm: Wav2Vec2STTConvNorm?

    init(config: Wav2Vec2STTConfig, layerId: Int, normalization: Normalization) {
        let inputChannels = layerId > 0 ? config.convDim[layerId - 1] : 1
        let outputChannels = config.convDim[layerId]
        self.activation = config.featExtractActivation
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: config.convKernel[layerId],
            stride: config.convStride[layerId],
            bias: config.convBias
        )

        switch normalization {
        case .none:
            _norm.wrappedValue = nil
        case .layer:
            _norm.wrappedValue = Wav2Vec2STTConvNorm(
                kind: .layer,
                groupCount: outputChannels,
                dimensions: outputChannels,
                eps: config.layerNormEps
            )
        case .group:
            _norm.wrappedValue = Wav2Vec2STTConvNorm(
                kind: .group,
                groupCount: outputChannels,
                dimensions: outputChannels,
                eps: config.layerNormEps
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x.transposed(0, 2, 1))
        if let norm {
            out = norm(out)
        }
        out = out.transposed(0, 2, 1)
        return wav2Vec2Activation(out, name: activation)
    }
}

private final class Wav2Vec2STTFeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [Wav2Vec2STTConvLayer]

    init(config: Wav2Vec2STTConfig) {
        let count = min(config.numFeatExtractLayers, config.convDim.count)
        switch config.featExtractNorm {
        case "layer":
            _convLayers.wrappedValue = (0..<count).map {
                Wav2Vec2STTConvLayer(config: config, layerId: $0, normalization: .layer)
            }
        default:
            _convLayers.wrappedValue = (0..<count).map {
                Wav2Vec2STTConvLayer(
                    config: config,
                    layerId: $0,
                    normalization: $0 == 0 ? .group : .none
                )
            }
        }
    }

    func callAsFunction(_ inputValues: MLXArray) -> MLXArray {
        var x = inputValues
        if x.ndim == 1 {
            x = x.expandedDimensions(axis: 0)
        }
        x = x.expandedDimensions(axis: 1)
        for layer in convLayers {
            x = layer(x)
        }
        return x
    }
}

private final class Wav2Vec2STTFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var projection: Linear
    @ModuleInfo var dropout: Dropout

    init(config: Wav2Vec2STTConfig) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.convDim.last ?? 512, eps: config.layerNormEps)
        _projection.wrappedValue = Linear(config.convDim.last ?? 512, config.hiddenSize)
        _dropout.wrappedValue = Dropout(p: config.featProjDropout)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (MLXArray, MLXArray) {
        let normHiddenStates = layerNorm(hiddenStates)
        let projected = dropout(projection(normHiddenStates))
        return (projected, normHiddenStates)
    }
}

private final class Wav2Vec2STTPositionalConvEmbedding: Module {
    let removeLast: Bool

    @ModuleInfo var conv: Conv1d

    init(config: Wav2Vec2STTConfig) {
        self.removeLast = config.numConvPosEmbeddings % 2 == 0
        _conv.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.numConvPosEmbeddings,
            padding: config.numConvPosEmbeddings / 2,
            groups: config.numConvPosEmbeddingGroups,
            bias: true
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var out = conv(hiddenStates)
        if removeLast && out.dim(1) > 0 {
            out = out[0..., ..<(out.dim(1) - 1), 0...]
        }
        return gelu(out)
    }
}

private final class Wav2Vec2STTAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(config: Wav2Vec2STTConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let time = hiddenStates.dim(1)

        let q = qProj(hiddenStates).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(hiddenStates).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(hiddenStates).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: attentionMask
        )
        let out = attended.transposed(0, 2, 1, 3).reshaped(batch, time, -1)
        return outProj(out)
    }
}

private final class Wav2Vec2STTFeedForward: Module {
    let hiddenAct: String

    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "intermediate_dropout") var intermediateDropout: Dropout
    @ModuleInfo(key: "output_dense") var outputDense: Linear
    @ModuleInfo(key: "output_dropout") var outputDropout: Dropout

    init(config: Wav2Vec2STTConfig) {
        self.hiddenAct = config.hiddenAct
        _intermediateDense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _intermediateDropout.wrappedValue = Dropout(p: config.activationDropout)
        _outputDense.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        _outputDropout.wrappedValue = Dropout(p: config.hiddenDropout)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var out = intermediateDense(hiddenStates)
        out = wav2Vec2Activation(out, name: hiddenAct)
        out = intermediateDropout(out)
        out = outputDense(out)
        return outputDropout(out)
    }
}

private final class Wav2Vec2STTAttnAdapterLayer: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(config: Wav2Vec2STTConfig, adapterDim: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _linear1.wrappedValue = Linear(config.hiddenSize, adapterDim)
        _linear2.wrappedValue = Linear(adapterDim, config.hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        linear2(relu(linear1(norm(hiddenStates))))
    }
}

private final class Wav2Vec2STTEncoderLayer: Module {
    let stableLayerNorm: Bool

    @ModuleInfo var attention: Wav2Vec2STTAttention
    @ModuleInfo var dropout: Dropout
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Wav2Vec2STTFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    @ModuleInfo(key: "adapter_layer") var adapterLayer: Wav2Vec2STTAttnAdapterLayer?

    init(config: Wav2Vec2STTConfig) {
        self.stableLayerNorm = config.doStableLayerNorm
        _attention.wrappedValue = Wav2Vec2STTAttention(config: config)
        _dropout.wrappedValue = Dropout(p: config.hiddenDropout)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _feedForward.wrappedValue = Wav2Vec2STTFeedForward(config: config)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _adapterLayer.wrappedValue = config.doStableLayerNorm ? config.adapterAttnDim.map {
            Wav2Vec2STTAttnAdapterLayer(config: config, adapterDim: $0)
        } : nil
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        if stableLayerNorm {
            return stableCall(hiddenStates, attentionMask: attentionMask)
        }

        let residual = hiddenStates
        var out = attention(hiddenStates, attentionMask: attentionMask)
        out = dropout(out)
        out = residual + out
        out = layerNorm(out)
        out = out + feedForward(out)
        return finalLayerNorm(out)
    }

    private func stableCall(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let residual = hiddenStates
        var out = layerNorm(hiddenStates)
        out = attention(out, attentionMask: attentionMask)
        out = dropout(out)
        out = residual + out
        out = out + feedForward(finalLayerNorm(out))
        if let adapterLayer {
            out = out + adapterLayer(out)
        }
        return out
    }
}

private final class Wav2Vec2STTStableEncoderLayer: Module {
    @ModuleInfo var attention: Wav2Vec2STTAttention
    @ModuleInfo var dropout: Dropout
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Wav2Vec2STTFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    @ModuleInfo(key: "adapter_layer") var adapterLayer: Wav2Vec2STTAttnAdapterLayer?

    init(config: Wav2Vec2STTConfig) {
        _attention.wrappedValue = Wav2Vec2STTAttention(config: config)
        _dropout.wrappedValue = Dropout(p: config.hiddenDropout)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _feedForward.wrappedValue = Wav2Vec2STTFeedForward(config: config)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _adapterLayer.wrappedValue = config.adapterAttnDim.map {
            Wav2Vec2STTAttnAdapterLayer(config: config, adapterDim: $0)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let residual = hiddenStates
        var out = layerNorm(hiddenStates)
        out = attention(out, attentionMask: attentionMask)
        out = dropout(out)
        out = residual + out
        out = out + feedForward(finalLayerNorm(out))
        if let adapterLayer {
            out = out + adapterLayer(out)
        }
        return out
    }
}

private final class Wav2Vec2STTEncoder: Module {
    let stableLayerNorm: Bool

    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: Wav2Vec2STTPositionalConvEmbedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var dropout: Dropout
    @ModuleInfo var layers: [Wav2Vec2STTEncoderLayer]

    init(config: Wav2Vec2STTConfig) {
        self.stableLayerNorm = config.doStableLayerNorm
        _posConvEmbed.wrappedValue = Wav2Vec2STTPositionalConvEmbedding(config: config)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _dropout.wrappedValue = Dropout(p: config.hiddenDropout)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            Wav2Vec2STTEncoderLayer(config: config)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var out = hiddenStates + posConvEmbed(hiddenStates)
        if stableLayerNorm {
            out = dropout(out)
            for layer in layers {
                out = layer(out, attentionMask: attentionMask)
            }
            return layerNorm(out)
        }

        out = layerNorm(out)
        out = dropout(out)
        for layer in layers {
            out = layer(out, attentionMask: attentionMask)
        }
        return out
    }
}

private struct Wav2Vec2STTBaseOutput {
    let lastHiddenState: MLXArray
    let extractFeatures: MLXArray
}

public final class Wav2Vec2STTModel: Module {
    public let config: Wav2Vec2STTConfig

    @ModuleInfo(key: "feature_extractor") private var featureExtractor: Wav2Vec2STTFeatureExtractor
    @ModuleInfo(key: "feature_projection") private var featureProjection: Wav2Vec2STTFeatureProjection
    @ModuleInfo private var encoder: Wav2Vec2STTEncoder

    public init(config: Wav2Vec2STTConfig) {
        self.config = config
        _featureExtractor.wrappedValue = Wav2Vec2STTFeatureExtractor(config: config)
        _featureProjection.wrappedValue = Wav2Vec2STTFeatureProjection(config: config)
        _encoder.wrappedValue = Wav2Vec2STTEncoder(config: config)
        super.init()
    }

    fileprivate func callAsFunction(_ inputValues: MLXArray) -> Wav2Vec2STTBaseOutput {
        var extractFeatures = featureExtractor(inputValues)
        extractFeatures = extractFeatures.transposed(0, 2, 1)
        let projected = featureProjection(extractFeatures)
        let hiddenStates = encoder(projected.0)
        return Wav2Vec2STTBaseOutput(lastHiddenState: hiddenStates, extractFeatures: projected.1)
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitizeWav2Vec2CTCWeights(weights, includeLMHead: false)
    }
}

public final class Wav2Vec2CTCModel: Module, STTGenerationModel {
    public let config: Wav2Vec2STTConfig
    public var vocabularies: [String: [Int: String]]
    public var defaultVocabulary: [Int: String]

    @ModuleInfo public var wav2vec2: Wav2Vec2STTModel
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(maxTokens: 0, temperature: 0.0, topP: 1.0, topK: 0, chunkDuration: 1200.0)
    }

    public init(
        config: Wav2Vec2STTConfig,
        vocabulary: [Int: String] = [:],
        vocabularies: [String: [Int: String]] = [:]
    ) {
        self.config = config
        self.defaultVocabulary = vocabulary
        self.vocabularies = vocabularies
        _wav2vec2.wrappedValue = Wav2Vec2STTModel(config: config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize)
        super.init()
    }

    public func callAsFunction(_ inputValues: MLXArray) -> MLXArray {
        let outputs = wav2vec2(inputValues)
        return lmHead(outputs.lastHiddenState)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        let start = CFAbsoluteTimeGetCurrent()
        var input = Self.normalizeToBatch(audio).asType(.float32)
        let mean = MLX.mean(input, axis: -1, keepDims: true)
        let centered = input - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        input = (input - mean) / (sqrt(variance) + 1e-7)

        let logits = self(input)
        eval(logits)
        let tokenIds = Self.greedyCTCTokens(logits: logits, blankTokenId: config.padTokenId)
        let text = decode(tokens: tokenIds.first ?? [], language: generationParameters.language)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let totalTime = CFAbsoluteTimeGetCurrent() - start

        return STTOutput(
            text: text,
            segments: [["text": text, "start": 0.0, "end": 0.0]],
            language: generationParameters.language,
            generationTokens: tokenIds.first?.count ?? 0,
            totalTokens: tokenIds.first?.count ?? 0,
            generationTps: Double(tokenIds.first?.count ?? 0) / max(totalTime, 0.001),
            totalTime: totalTime
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let output = self.generate(audio: audio, generationParameters: generationParameters)
            if !output.text.isEmpty {
                continuation.yield(.token(output.text))
            }
            continuation.yield(.result(output))
            continuation.finish()
        }
    }

    public func decode(tokens: [Int], language: String? = nil) -> String {
        let vocab = vocabulary(for: language)
        guard !vocab.isEmpty else {
            return tokens.map(String.init).joined(separator: " ")
        }
        return tokens.map { vocab[$0] ?? "" }.joined().replacingOccurrences(of: "|", with: " ")
    }

    public func vocabulary(for language: String?) -> [Int: String] {
        guard let language, !language.isEmpty else {
            return defaultVocabulary
        }
        let key = language.lowercased()
        return vocabularies[key]
            ?? vocabularies[Self.iso3LanguageAlias(key)]
            ?? defaultVocabulary
    }

    public static func greedyCTCTokens(logits: MLXArray, blankTokenId: Int = 0) -> [[Int]] {
        let predictions = logits.argMax(axis: -1).asType(.int32)
        eval(predictions)
        let values = predictions.asArray(Int32.self).map(Int.init)
        let batch = predictions.dim(0)
        let time = predictions.dim(1)
        var decoded: [[Int]] = []
        decoded.reserveCapacity(batch)

        for row in 0..<batch {
            var rowTokens: [Int] = []
            var previous = -1
            for t in 0..<time {
                let token = values[row * time + t]
                if token != previous && token != blankTokenId {
                    rowTokens.append(token)
                }
                previous = token
            }
            decoded.append(rowTokens)
        }
        return decoded
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitizeWav2Vec2CTCWeights(weights, includeLMHead: true)
    }

    public static func fromPretrained(
        _ modelName: String,
        language: String? = nil
    ) async throws -> Wav2Vec2CTCModel {
        let expanded = (modelName as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            return try fromModelDirectory(URL(fileURLWithPath: expanded), language: language)
        }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelName) else {
            throw STTError.invalidInput("Invalid Wav2Vec2/MMS repository ID: \(modelName)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.json"],
            hfToken: hfToken
        )
        return try fromModelDirectory(modelDir, language: language)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        language: String? = nil
    ) throws -> Wav2Vec2CTCModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(Wav2Vec2STTConfig.self, from: Data(contentsOf: configURL))
        let vocabStore = try loadVocabularies(from: modelDir)
        let model = Wav2Vec2CTCModel(
            config: config,
            vocabulary: selectDefaultVocabulary(from: vocabStore, language: language),
            vocabularies: vocabStore
        )

        let weights = try loadSafetensorWeights(from: modelDir, includeAdapters: false)
        let sanitized = sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: Module.VerifyUpdate.noUnusedKeys)

        if let adapterURL = selectAdapter(in: modelDir, language: language) {
            let adapterWeights = try MLX.loadArrays(url: adapterURL)
            let sanitizedAdapter = sanitize(weights: adapterWeights)
            try model.update(parameters: ModuleParameters.unflattened(sanitizedAdapter), verify: Module.VerifyUpdate.noUnusedKeys)
        }

        model.train(false)
        eval(model)
        return model
    }

    private static func normalizeToBatch(_ audio: MLXArray) -> MLXArray {
        var input = audio
        if input.ndim == 1 {
            input = input.expandedDimensions(axis: 0)
        } else if input.ndim > 2 {
            input = input.reshaped(input.dim(0), -1)
        }
        return input
    }

    fileprivate static func iso3LanguageAlias(_ language: String) -> String {
        switch language {
        case "en", "english":
            return "eng"
        case "fr", "french":
            return "fra"
        case "de", "german":
            return "deu"
        case "es", "spanish":
            return "spa"
        default:
            return language
        }
    }
}

private func sanitizeWav2Vec2CTCWeights(_ weights: [String: MLXArray], includeLMHead: Bool) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]
    var posConvWeightG: MLXArray?
    var posConvWeightV: MLXArray?

    for (rawKey, rawValue) in weights {
        var key = rawKey
        var value = rawValue

        if key.hasPrefix("wav2vec2.") {
            key = String(key.dropFirst("wav2vec2.".count))
            if includeLMHead {
                key = "wav2vec2." + key
            }
        }

        if key.hasPrefix("quantizer.") || key.hasPrefix("wav2vec2.quantizer.")
            || key.hasPrefix("project_") || key.hasPrefix("wav2vec2.project_")
            || key == "masked_spec_embed" || key == "wav2vec2.masked_spec_embed" {
            continue
        }
        if key.contains("lm_head.") && !includeLMHead {
            continue
        }

        if key.hasSuffix(".parametrizations.weight.original0") {
            key = key.replacingOccurrences(of: ".parametrizations.weight.original0", with: ".weight_g")
            if value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
        } else if key.hasSuffix(".parametrizations.weight.original1") {
            key = key.replacingOccurrences(of: ".parametrizations.weight.original1", with: ".weight_v")
            if value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
        } else if key.hasSuffix(".conv.weight") && value.ndim == 3 {
            value = value.transposed(0, 2, 1)
        } else if (key.hasSuffix(".conv.weight_g") || key.hasSuffix(".conv.weight_v")) && value.ndim == 3 {
            value = value.transposed(0, 2, 1)
        }

        if key == "wav2vec2.encoder.pos_conv_embed.conv.weight_g" || key == "encoder.pos_conv_embed.conv.weight_g" {
            posConvWeightG = value
            continue
        }
        if key == "wav2vec2.encoder.pos_conv_embed.conv.weight_v" || key == "encoder.pos_conv_embed.conv.weight_v" {
            posConvWeightV = value
            continue
        }

        sanitized[key] = value
    }

    if let g = posConvWeightG, let v = posConvWeightV {
        let norm = sqrt(sum(v * v, axes: [0, 2], keepDims: true) + 1e-12)
        let weight = g * v / norm
        sanitized[includeLMHead ? "wav2vec2.encoder.pos_conv_embed.conv.weight" : "encoder.pos_conv_embed.conv.weight"] = weight
    }

    return sanitized
}

private func loadSafetensorWeights(from modelDir: URL, includeAdapters: Bool) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .filter { includeAdapters || !$0.lastPathComponent.hasPrefix("adapter.") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !files.isEmpty else {
        throw STTError.modelNotInitialized("No safetensors files found in \(modelDir.path)")
    }

    var weights: [String: MLXArray] = [:]
    for file in files {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

private func selectAdapter(in modelDir: URL, language: String?) -> URL? {
    let files = (try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)) ?? []
    let adapters = files
        .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("adapter.") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !adapters.isEmpty else { return nil }

    let languageKeys: [String]
    if let language, !language.isEmpty {
        let lower = language.lowercased()
        languageKeys = [lower, Wav2Vec2CTCModel.iso3LanguageAlias(lower)]
    } else {
        languageKeys = ["eng", "en"]
    }
    for key in languageKeys {
        if let match = adapters.first(where: { $0.lastPathComponent == "adapter.\(key).safetensors" }) {
            return match
        }
    }
    return adapters.first
}

private func loadVocabularies(from modelDir: URL) throws -> [String: [Int: String]] {
    let vocabURL = modelDir.appendingPathComponent("vocab.json")
    guard FileManager.default.fileExists(atPath: vocabURL.path) else {
        return [:]
    }

    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL))
    if let nested = object as? [String: [String: Int]] {
        var out: [String: [Int: String]] = [:]
        for (language, vocab) in nested {
            out[language.lowercased()] = invertVocabulary(vocab)
        }
        return out
    }
    if let flat = object as? [String: Int] {
        return ["default": invertVocabulary(flat)]
    }
    return [:]
}

private func selectDefaultVocabulary(from store: [String: [Int: String]], language: String?) -> [Int: String] {
    if let language, !language.isEmpty {
        let lower = language.lowercased()
        if let vocab = store[lower] ?? store[Wav2Vec2CTCModel.iso3LanguageAlias(lower)] {
            return vocab
        }
    }
    return store["eng"] ?? store["en"] ?? store["default"] ?? store.values.first ?? [:]
}

private func invertVocabulary(_ vocab: [String: Int]) -> [Int: String] {
    var out: [Int: String] = [:]
    for (token, id) in vocab {
        out[id] = token
    }
    return out
}
