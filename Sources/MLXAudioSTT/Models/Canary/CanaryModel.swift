import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

public final class CanaryTokenizer {
    private let tokenizer: SentencePieceTokenizer?
    private let tokenToId: [String: Int]
    private let idToToken: [Int: String]

    private init(
        tokenizer: SentencePieceTokenizer?,
        tokenToId: [String: Int],
        idToToken: [Int: String]
    ) {
        self.tokenizer = tokenizer
        self.tokenToId = tokenToId
        self.idToToken = idToToken
    }

    public static func fromModelDirectory(_ modelDir: URL, config: CanaryConfig) throws -> CanaryTokenizer? {
        let tokensURL = modelDir.appendingPathComponent("tokens.txt")
        let tokenMaps = try loadTokenMaps(from: tokensURL)

        let sentencePiece: SentencePieceTokenizer?
        let sentencePieceURL = modelDir.appendingPathComponent("tokenizer.model")
        let tokenizerJSONURL = modelDir.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: sentencePieceURL.path) {
            sentencePiece = try SentencePieceTokenizer.from(sentencePieceModelURL: sentencePieceURL)
        } else if FileManager.default.fileExists(atPath: tokenizerJSONURL.path) {
            sentencePiece = try SentencePieceTokenizer.from(tokenizerJSONURL: tokenizerJSONURL)
        } else if let base64 = config.tokenizerModelBase64, let data = Data(base64Encoded: base64) {
            sentencePiece = try SentencePieceTokenizer(sentencePieceModelData: data)
        } else {
            sentencePiece = nil
        }

        if sentencePiece == nil && tokenMaps.tokenToId.isEmpty {
            return nil
        }

        var tokenToId = tokenMaps.tokenToId
        var idToToken = tokenMaps.idToToken
        if let sentencePiece {
            let candidates = specialTokenCandidates(languages: config.supportedLanguages)
            for token in candidates {
                if tokenToId[token] == nil, let id = sentencePiece.tokenID(for: token) {
                    tokenToId[token] = id
                    idToToken[id] = token
                }
            }
        }

        return CanaryTokenizer(tokenizer: sentencePiece, tokenToId: tokenToId, idToToken: idToToken)
    }

    public static func defaultPromptTokens(config: CanaryConfig) -> [Int] {
        [config.startOfContextId, config.startOfTranscriptId, config.emotionUndefinedId]
    }

    public func buildPromptTokens(
        config: CanaryConfig,
        sourceLanguage: String,
        targetLanguage: String,
        usePunctuationAndCapitalization: Bool = true
    ) -> [Int] {
        var tokens: [Int] = []
        tokens.append(tokenId("<|startofcontext|>", fallback: config.startOfContextId))
        tokens.append(tokenId("<|startoftranscript|>", fallback: config.startOfTranscriptId))
        tokens.append(tokenId("<|emo:undefined|>", fallback: config.emotionUndefinedId))
        appendIfPresent("<|\(sourceLanguage)|>", to: &tokens)
        appendIfPresent("<|\(targetLanguage)|>", to: &tokens)
        appendIfPresent(usePunctuationAndCapitalization ? "<|pnc|>" : "<|nopnc|>", to: &tokens)
        appendIfPresent("<|noitn|>", to: &tokens)
        appendIfPresent("<|notimestamp|>", to: &tokens)
        appendIfPresent("<|nodiarize|>", to: &tokens)
        return tokens
    }

    public func eosTokenId(config: CanaryConfig) -> Int {
        tokenId("<|endoftext|>", fallback: config.endOfTextId)
    }

    public func decode(_ ids: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(ids)
        }
        let pieces = ids.compactMap { id -> String? in
            guard let token = idToToken[id] else { return nil }
            if token.hasPrefix("<|"), token.hasSuffix("|>") { return nil }
            return token
        }
        return pieces.joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func tokenId(_ token: String, fallback: Int) -> Int {
        if let id = tokenToId[token] {
            return id
        }
        if let id = tokenizer?.tokenID(for: token) {
            return id
        }
        return fallback
    }

    private func appendIfPresent(_ token: String, to tokens: inout [Int]) {
        if let id = tokenToId[token] ?? tokenizer?.tokenID(for: token) {
            tokens.append(id)
        }
    }

    private static func specialTokenCandidates(languages: [String]) -> [String] {
        var tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|endoftext|>",
            "<|pnc|>",
            "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        tokens.append(contentsOf: languages.map { "<|\($0)|>" })
        return tokens
    }

    private static func loadTokenMaps(from url: URL) throws -> (tokenToId: [String: Int], idToToken: [Int: String]) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ([:], [:])
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        var tokenToId: [String: Int] = [:]
        var idToToken: [Int: String] = [:]

        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = String(rawLine)
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            let fields = trimmed.split(whereSeparator: \.isWhitespace)

            let token: String
            let id: Int
            if fields.count == 2, let parsed = Int(fields[1]) {
                token = line.first == " " ? " " + String(fields[0]) : String(fields[0])
                id = parsed
            } else if fields.count == 1, let parsed = Int(fields[0]) {
                token = " "
                id = parsed
            } else {
                continue
            }

            tokenToId[token] = id
            idToToken[id] = token
        }

        return (tokenToId, idToToken)
    }
}

private final class CanaryFixedPositionalEncoding {
    let table: MLXArray

    init(dModel: Int, maxLength: Int = 1024) {
        var values = Array(repeating: Float(0), count: maxLength * dModel)
        for position in 0..<maxLength {
            for channel in stride(from: 0, to: dModel, by: 2) {
                let div = exp(-log(Float(10_000)) * Float(channel) / Float(dModel))
                let angle = Float(position) * div
                values[position * dModel + channel] = sin(angle)
                if channel + 1 < dModel {
                    values[position * dModel + channel + 1] = cos(angle)
                }
            }
        }
        table = MLXArray(values, [maxLength, dModel]) / MLXArray(sqrt(Float(dModel)))
    }

    func callAsFunction(batch: Int, length: Int, startPosition: Int, dtype: DType) -> MLXArray {
        let end = min(startPosition + length, table.dim(0))
        let start = min(startPosition, end)
        var positions = table[start..<end]
        if positions.dim(0) < length {
            let pad = MLX.repeated(positions[(positions.dim(0) - 1)..<positions.dim(0)], count: length - positions.dim(0), axis: 0)
            positions = MLX.concatenated([positions, pad], axis: 0)
        }
        positions = positions.expandedDimensions(axis: 0)
        if batch > 1 {
            positions = MLX.repeated(positions, count: batch, axis: 0)
        }
        return positions.asType(dtype)
    }
}

private final class CanaryAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(self.headDim), -0.5)
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        keyValue: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let targetTime = x.dim(1)
        let sourceTime = keyValue.dim(1)

        let q = qProj(x).reshaped(batch, targetTime, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(keyValue).reshaped(batch, sourceTime, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(keyValue).reshaped(batch, sourceTime, numHeads, headDim).transposed(0, 2, 1, 3)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, targetTime, -1))
    }
}

private final class CanaryDecoderBlock: Module {
    @ModuleInfo(key: "self_attn_norm") var selfAttentionNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: CanaryAttention
    @ModuleInfo(key: "cross_attn_norm") var crossAttentionNorm: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: CanaryAttention
    @ModuleInfo(key: "ff_norm") var feedForwardNorm: LayerNorm
    @ModuleInfo(key: "ff1") var feedForward1: Linear
    @ModuleInfo(key: "ff2") var feedForward2: Linear

    init(config: CanaryDecoderConfig) {
        _selfAttentionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _selfAttention.wrappedValue = CanaryAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        _crossAttentionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _crossAttention.wrappedValue = CanaryAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        _feedForwardNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _feedForward1.wrappedValue = Linear(config.hiddenSize, config.innerSize, bias: true)
        _feedForward2.wrappedValue = Linear(config.innerSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        encoderMask: MLXArray? = nil,
        selfAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        let selfNorm = selfAttentionNorm(x)
        var out = x + selfAttention(selfNorm, keyValue: selfNorm, mask: selfAttentionMask)

        let crossMask: MLXArray?
        if let encoderMask {
            let expanded = encoderMask.expandedDimensions(axes: [1, 2])
            crossMask = MLX.where(expanded .== 0, MLXArray(Float(-1e9)), MLXArray(Float(0))).asType(out.dtype)
        } else {
            crossMask = nil
        }

        out = out + crossAttention(crossAttentionNorm(out), keyValue: encoderOutput, mask: crossMask)
        out = out + feedForward2(relu(feedForward1(feedForwardNorm(out))))
        return out
    }
}

public final class CanaryDecoder: Module {
    @ModuleInfo var embedding: Embedding
    private let positionEmbedding: CanaryFixedPositionalEncoding
    @ModuleInfo(key: "embedding_layer_norm") var embeddingLayerNorm: LayerNorm
    @ModuleInfo fileprivate var blocks: [CanaryDecoderBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "output_proj") var outputProjection: Linear

    public init(config: CanaryDecoderConfig, vocabSize: Int, hiddenSize: Int) {
        _embedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        positionEmbedding = CanaryFixedPositionalEncoding(dModel: hiddenSize)
        _embeddingLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _blocks.wrappedValue = (0..<config.numLayers).map { _ in CanaryDecoderBlock(config: config) }
        _finalNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _outputProjection.wrappedValue = Linear(hiddenSize, vocabSize, bias: true)
        super.init()
    }

    public func callAsFunction(
        _ tokens: MLXArray,
        encoderOutput: MLXArray,
        encoderMask: MLXArray? = nil,
        startPosition: Int = 0
    ) -> MLXArray {
        let batch = tokens.dim(0)
        let time = tokens.dim(1)
        var x = embedding(tokens)
        x = x + positionEmbedding(batch: batch, length: time, startPosition: startPosition, dtype: x.dtype)
        x = embeddingLayerNorm(x)

        let selfMask = time > 1 ? MultiHeadAttention.createAdditiveCausalMask(time).asType(x.dtype) : nil
        for block in blocks {
            x = block(
                x,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                selfAttentionMask: selfMask
            )
        }
        return outputProjection(finalNorm(x))
    }
}

public final class CanaryEncoder: Module {
    @ModuleInfo var conformer: ParakeetConformer
    @ModuleInfo var projection: Linear?

    public init(config: CanaryConfig) {
        _conformer.wrappedValue = ParakeetConformer(args: config.encoder.parakeetConfig)
        if config.encoder.dModel == config.encoderOutputDim {
            _projection.wrappedValue = nil
        } else {
            _projection.wrappedValue = Linear(config.encoder.dModel, config.encoderOutputDim)
        }
        super.init()
    }

    public func callAsFunction(_ mel: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var encoded = conformer(mel, lengths: lengths)
        if let projection {
            encoded.0 = projection(encoded.0)
        }
        return encoded
    }
}

public final class CanaryModel: Module, STTGenerationModel {
    public let config: CanaryConfig
    public var tokenizer: CanaryTokenizer?

    @ModuleInfo public var encoder: CanaryEncoder
    @ModuleInfo public var decoder: CanaryDecoder

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 200,
            temperature: 0,
            topP: 1,
            topK: 0,
            language: "en",
            chunkDuration: 1200,
            minChunkDuration: 1
        )
    }

    public init(config: CanaryConfig, tokenizer: CanaryTokenizer? = nil) {
        self.config = config
        self.tokenizer = tokenizer
        _encoder.wrappedValue = CanaryEncoder(config: config)
        _decoder.wrappedValue = CanaryDecoder(
            config: config.decoder,
            vocabSize: config.vocabSize,
            hiddenSize: config.encoderOutputDim
        )
        super.init()
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        let start = CFAbsoluteTimeGetCurrent()
        let language = generationParameters.language ?? "en"
        let mel = preprocessAudio(audio).asType(.float32)
        let encoded = encode(mel: mel)

        let promptTokens = tokenizer?.buildPromptTokens(
            config: config,
            sourceLanguage: language,
            targetLanguage: language
        ) ?? CanaryTokenizer.defaultPromptTokens(config: config)
        let eosTokenId = tokenizer?.eosTokenId(config: config) ?? config.endOfTextId

        var tokens = promptTokens
        var generated: [Int] = []

        for _ in 0..<generationParameters.maxTokens {
            let tokenIds = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count).asType(.int32)
            let logits = decoder(tokenIds, encoderOutput: encoded.hidden, encoderMask: encoded.mask)
            let nextLogits = logits[0..., (logits.dim(1) - 1)..., 0...].squeezed(axis: 1)
            eval(nextLogits)
            let nextToken: Int
            if generationParameters.temperature > 0 {
                nextToken = MLXRandom.categorical(nextLogits / generationParameters.temperature).item(Int.self)
            } else {
                nextToken = nextLogits.argMax(axis: -1).item(Int.self)
            }
            if nextToken == eosTokenId {
                break
            }
            tokens.append(nextToken)
            generated.append(nextToken)
        }

        let text = decode(tokens: generated).trimmingCharacters(in: .whitespacesAndNewlines)
        let totalTime = CFAbsoluteTimeGetCurrent() - start
        return STTOutput(
            text: text,
            segments: [["text": text, "start": 0.0, "end": 0.0]],
            language: language,
            promptTokens: promptTokens.count,
            generationTokens: generated.count,
            totalTokens: tokens.count,
            promptTps: Double(promptTokens.count) / max(totalTime, 0.001),
            generationTps: Double(generated.count) / max(totalTime, 0.001),
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

    public func preprocessAudio(_ audio: MLXArray) -> MLXArray {
        if audio.ndim == 3 {
            return audio
        }
        if audio.ndim == 2, audio.dim(1) == config.preprocessor.features {
            return audio.expandedDimensions(axis: 0)
        }
        let waveform = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        return ParakeetAudio.logMelSpectrogram(waveform, config: config.preprocessor.parakeetConfig)
    }

    public func encode(mel: MLXArray) -> (hidden: MLXArray, lengths: MLXArray, mask: MLXArray) {
        let batch = mel.dim(0)
        let time = mel.dim(1)
        let lengths = MLXArray(Array(repeating: Int32(time), count: batch)).asType(.int32)
        let encoded = encoder(mel, lengths: lengths)
        eval(encoded.0, encoded.1)

        let positions = MLX.arange(encoded.0.dim(1), dtype: .int32).expandedDimensions(axis: 0)
        let mask = (positions .< encoded.1.expandedDimensions(axis: 1)).asType(.float32)
        return (encoded.0, encoded.1, mask)
    }

    public func decode(tokens: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(tokens)
        }
        return tokens.map { "<\($0)>" }.joined()
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if weights.keys.contains(where: { $0.hasPrefix("decoder.blocks.") }) {
            return weights
        }

        let mlxNative = weights["head.classifier.weight"] != nil
            || weights.keys.contains(where: { $0.hasPrefix("transf_decoder.layers.") })
        return mlxNative ? sanitizeMLXNative(weights: weights) : sanitizeNemo(weights: weights)
    }

    public static func fromPretrained(_ modelName: String) async throws -> CanaryModel {
        let expanded = (modelName as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            return try await fromModelDirectory(URL(fileURLWithPath: expanded))
        }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelName) else {
            throw STTError.invalidInput("Invalid Canary repository ID: \(modelName)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.json", "*.model", "tokens.txt"],
            hfToken: hfToken
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> CanaryModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(CanaryConfig.self, from: Data(contentsOf: configURL))
        let tokenizer = try CanaryTokenizer.fromModelDirectory(modelDir, config: config)
        let model = CanaryModel(config: config, tokenizer: tokenizer)
        let weights = try loadCanarySafetensorWeights(from: modelDir)
        let sanitized = sanitize(weights: weights)
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitized["\(path).scales"] != nil else {
                    return nil
                }
                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path) {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: Module.VerifyUpdate.noUnusedKeys)
        model.train(false)
        eval(model)
        return model
    }

    private static func sanitizeMLXNative(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            let newKey: String?
            if key.hasPrefix("encoder."), !key.hasPrefix("encoder_decoder") {
                newKey = "encoder.conformer." + key.dropPrefix("encoder.")
            } else if key.hasPrefix("transf_decoder.token_embedding.") {
                newKey = "decoder.embedding." + key.dropPrefix("transf_decoder.token_embedding.")
            } else if key.hasPrefix("transf_decoder.embedding_layer_norm.") {
                newKey = "decoder.embedding_layer_norm." + key.dropPrefix("transf_decoder.embedding_layer_norm.")
            } else if key.hasPrefix("transf_decoder.final_layer_norm.") {
                newKey = "decoder.final_norm." + key.dropPrefix("transf_decoder.final_layer_norm.")
            } else if key.hasPrefix("transf_decoder.layers.") {
                newKey = remapMLXNativeDecoderLayer(key.dropPrefix("transf_decoder.layers."))
            } else if key.hasPrefix("head.classifier.") {
                newKey = "decoder.output_proj." + key.dropPrefix("head.classifier.")
            } else {
                newKey = nil
            }

            if let newKey {
                if let converted = remapCanaryPreEncodeConvListKey(newKey) {
                    sanitized[normalizeCanaryParameterKey(converted)] = value
                } else if !shouldSkipCanaryPreEncodeConvListKey(newKey) {
                    sanitized[normalizeCanaryParameterKey(newKey)] = value
                }
            }
        }
        return sanitized
    }

    private static func sanitizeNemo(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, var value) in weights {
            if key.contains("attn_dropout") || key.contains("layer_dropout") || key.contains("num_batches_tracked") {
                continue
            }
            if key == "log_softmax.mlp.log_softmax" || key.hasPrefix("encoder_decoder_proj.") {
                continue
            }

            var newKey: String?
            if key.hasPrefix("encoder."), !key.hasPrefix("encoder_decoder") {
                newKey = "encoder.conformer." + key.dropPrefix("encoder.")
            } else if key.hasPrefix("transf_decoder._embedding.token_embedding.") {
                newKey = "decoder.embedding." + key.dropPrefix("transf_decoder._embedding.token_embedding.")
            } else if key.hasPrefix("transf_decoder._embedding.position_embedding.") {
                newKey = nil
            } else if key.hasPrefix("transf_decoder._embedding.layer_norm.") {
                newKey = "decoder.embedding_layer_norm." + key.dropPrefix("transf_decoder._embedding.layer_norm.")
            } else if key.hasPrefix("transf_decoder._decoder.layers.") {
                newKey = remapNemoDecoderLayer(key.dropPrefix("transf_decoder._decoder.layers."))
            } else if key.hasPrefix("transf_decoder._decoder.final_layer_norm.") {
                newKey = "decoder.final_norm." + key.dropPrefix("transf_decoder._decoder.final_layer_norm.")
            } else if key.hasPrefix("log_softmax.mlp.layer0.") {
                newKey = "decoder.output_proj." + key.dropPrefix("log_softmax.mlp.layer0.")
            } else {
                newKey = key
            }

            guard var mappedKey = newKey else { continue }
            if let converted = remapCanaryPreEncodeConvListKey(mappedKey) {
                mappedKey = converted
            } else if shouldSkipCanaryPreEncodeConvListKey(mappedKey) {
                continue
            }
            let normalizedKey = normalizeCanaryParameterKey(mappedKey)
            if mappedKey.contains("conv"), mappedKey.hasSuffix(".weight"), value.ndim >= 3 {
                if value.ndim == 3 {
                    value = value.transposed(0, 2, 1)
                } else if value.ndim == 4 {
                    value = value.transposed(0, 2, 3, 1)
                }
            }
            sanitized[normalizedKey] = value
        }
        return sanitized
    }

    private static func remapMLXNativeDecoderLayer(_ rest: String) -> String? {
        let parts = rest.split(separator: ".", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }
        let layer = parts[0]
        let sub = parts[1]
        return "decoder.blocks.\(layer)." + remapMLXNativeDecoderSubLayer(sub)
    }

    private static func remapMLXNativeDecoderSubLayer(_ sub: String) -> String {
        if sub.hasPrefix("first_sub_layer.") {
            return "self_attn." + remapMLXNativeAttention(sub.dropPrefix("first_sub_layer."))
        }
        if sub.hasPrefix("second_sub_layer.") {
            return "cross_attn." + remapMLXNativeAttention(sub.dropPrefix("second_sub_layer."))
        }
        if sub.hasPrefix("third_sub_layer.linear1.") {
            return "ff1." + sub.dropPrefix("third_sub_layer.linear1.")
        }
        if sub.hasPrefix("third_sub_layer.linear2.") {
            return "ff2." + sub.dropPrefix("third_sub_layer.linear2.")
        }
        if sub.hasPrefix("layer_norm_1.") {
            return "self_attn_norm." + sub.dropPrefix("layer_norm_1.")
        }
        if sub.hasPrefix("layer_norm_2.") {
            return "cross_attn_norm." + sub.dropPrefix("layer_norm_2.")
        }
        if sub.hasPrefix("layer_norm_3.") {
            return "ff_norm." + sub.dropPrefix("layer_norm_3.")
        }
        return sub
    }

    private static func remapMLXNativeAttention(_ sub: String) -> String {
        if sub.hasPrefix("linear_q.") { return "q_proj." + sub.dropPrefix("linear_q.") }
        if sub.hasPrefix("linear_k.") { return "k_proj." + sub.dropPrefix("linear_k.") }
        if sub.hasPrefix("linear_v.") { return "v_proj." + sub.dropPrefix("linear_v.") }
        if sub.hasPrefix("linear_out.") { return "out_proj." + sub.dropPrefix("linear_out.") }
        return sub
    }

    private static func remapNemoDecoderLayer(_ rest: String) -> String? {
        let parts = rest.split(separator: ".", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }
        let layer = parts[0]
        var sub = parts[1]

        if sub.hasPrefix("first_sub_layer.") {
            sub = sub.dropPrefix("first_sub_layer.")
                .replacingOccurrences(of: "query_net.", with: "self_attn.q_proj.")
                .replacingOccurrences(of: "key_net.", with: "self_attn.k_proj.")
                .replacingOccurrences(of: "value_net.", with: "self_attn.v_proj.")
                .replacingOccurrences(of: "out_projection.", with: "self_attn.out_proj.")
        } else if sub.hasPrefix("second_sub_layer.") {
            sub = sub.dropPrefix("second_sub_layer.")
                .replacingOccurrences(of: "query_net.", with: "cross_attn.q_proj.")
                .replacingOccurrences(of: "key_net.", with: "cross_attn.k_proj.")
                .replacingOccurrences(of: "value_net.", with: "cross_attn.v_proj.")
                .replacingOccurrences(of: "out_projection.", with: "cross_attn.out_proj.")
        } else if sub.hasPrefix("third_sub_layer.") {
            sub = sub.dropPrefix("third_sub_layer.")
                .replacingOccurrences(of: "dense_in.", with: "ff1.")
                .replacingOccurrences(of: "dense_out.", with: "ff2.")
        } else if sub.hasPrefix("layer_norm_1.") {
            sub = "self_attn_norm." + sub.dropPrefix("layer_norm_1.")
        } else if sub.hasPrefix("layer_norm_2.") {
            sub = "cross_attn_norm." + sub.dropPrefix("layer_norm_2.")
        } else if sub.hasPrefix("layer_norm_3.") {
            sub = "ff_norm." + sub.dropPrefix("layer_norm_3.")
        }

        return "decoder.blocks.\(layer).\(sub)"
    }
}

private func loadCanarySafetensorWeights(from modelDir: URL) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
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

private func normalizeCanaryParameterKey(_ key: String) -> String {
    key
        .replacingOccurrences(of: ".pos_bias_u", with: ".posBiasU")
        .replacingOccurrences(of: ".pos_bias_v", with: ".posBiasV")
}

private func remapCanaryPreEncodeConvListKey(_ key: String) -> String? {
    let pieces = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
    guard pieces.count >= 6 else { return nil }
    guard pieces[0] == "encoder",
          pieces[1] == "conformer",
          pieces[2] == "pre_encode",
          pieces[3] == "conv",
          let rawIndex = Int(pieces[4])
    else {
        return nil
    }

    let suffix = pieces.dropFirst(5).joined(separator: ".")
    if rawIndex == 0 {
        return "encoder.conformer.pre_encode.conv0.\(suffix)"
    }
    if rawIndex < 2 {
        return nil
    }

    let shifted = rawIndex - 2
    let block = shifted / 3
    switch shifted % 3 {
    case 0:
        return "encoder.conformer.pre_encode.depthwise_layers.\(block).\(suffix)"
    case 1:
        return "encoder.conformer.pre_encode.pointwise_layers.\(block).\(suffix)"
    default:
        return nil
    }
}

private func shouldSkipCanaryPreEncodeConvListKey(_ key: String) -> Bool {
    let pieces = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
    guard pieces.count >= 6 else { return false }
    guard pieces[0] == "encoder",
          pieces[1] == "conformer",
          pieces[2] == "pre_encode",
          pieces[3] == "conv",
          let rawIndex = Int(pieces[4]),
          rawIndex >= 2
    else {
        return false
    }

    return (rawIndex - 2) % 3 == 2
}

private extension String {
    func dropPrefix(_ prefix: String) -> String {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : self
    }
}
