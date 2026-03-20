import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

private let fishSpeechRASWindowSize = 10
private let fishSpeechRASHighTemperature: Float = 1.0
private let fishSpeechRASHighTopP: Float = 0.9

struct FishSpeechForwardResult {
    let logits: MLXArray
    let hiddenStates: MLXArray
}

struct FishSpeechGeneratedSegment {
    let audio: MLXArray
    let promptTokenCount: Int
    let generationTokenCount: Int
    let elapsed: TimeInterval
    let peakMemoryUsage: Double
}

@inline(__always)
private func fishSpeechCallUnary(_ module: Module, _ x: MLXArray) -> MLXArray {
    (module as! UnaryLayer).callAsFunction(x)
}

final class FishSpeechIdentity: Module, UnaryLayer {
    override init() {}

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }
}

final class FishSpeechRotaryEmbedding: @unchecked Sendable {
    private let cosCache: MLXArray
    private let sinCache: MLXArray

    init(headDim: Int, ropeBase: Float, maxPositionEmbeddings: Int) {
        let freqs = 1.0 / pow(
            MLXArray(ropeBase),
            MLX.arange(0, headDim, step: 2, dtype: .float32) / MLXArray(Float(headDim))
        )
        let positions = MLX.arange(maxPositionEmbeddings, dtype: .float32)
        let angles = MLX.outer(positions, freqs)
        self.cosCache = cos(angles).asType(.bfloat16)
        self.sinCache = sin(angles).asType(.bfloat16)
    }

    func apply(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let sequenceLength = x.dim(2)
        let cosValues = cosCache[offset..<(offset + sequenceLength), 0...]
            .reshaped([1, 1, sequenceLength, cosCache.dim(1)])
        let sinValues = sinCache[offset..<(offset + sequenceLength), 0...]
            .reshaped([1, 1, sequenceLength, sinCache.dim(1)])

        let reshaped = x.asType(.float32).reshaped(x.shape.dropLast() + [x.shape.last! / 2, 2])
        let xEven = reshaped[0..., 0..., 0..., 0..., 0]
        let xOdd = reshaped[0..., 0..., 0..., 0..., 1]
        let rotated = MLX.stacked([
            xEven * cosValues - xOdd * sinValues,
            xOdd * cosValues + xEven * sinValues,
        ], axis: -1)
        return rotated.reshaped(x.shape).asType(x.dtype)
    }
}

final class FishSpeechFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class FishSpeechAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Module
    @ModuleInfo(key: "k_norm") var kNorm: Module

    let rope: FishSpeechRotaryEmbedding

    init(
        dim: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        ropeBase: Float,
        maxPositionEmbeddings: Int,
        attentionQKVBias: Bool,
        attentionOBias: Bool,
        attentionQKNorm: Bool,
        normEps: Float
    ) {
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        let totalHeadDim = (nHeads + 2 * nKVHeads) * headDim
        self._wqkv.wrappedValue = Linear(dim, totalHeadDim, bias: attentionQKVBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: attentionOBias)
        self._qNorm.wrappedValue = attentionQKNorm
            ? RMSNorm(dimensions: headDim, eps: normEps)
            : FishSpeechIdentity()
        self._kNorm.wrappedValue = attentionQKNorm
            ? RMSNorm(dimensions: headDim, eps: normEps)
            : FishSpeechIdentity()
        self.rope = FishSpeechRotaryEmbedding(
            headDim: headDim,
            ropeBase: ropeBase,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        let qSize = nHeads * headDim
        let kvSize = nKVHeads * headDim

        let qkv = wqkv(x)
        let q = qkv[0..., 0..., 0..<qSize]
        let k = qkv[0..., 0..., qSize..<(qSize + kvSize)]
        let v = qkv[0..., 0..., (qSize + kvSize)...]

        var queries = q.reshaped(batch, length, nHeads, headDim).transposed(0, 2, 1, 3)
        var keys = k.reshaped(batch, length, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = v.reshaped(batch, length, nKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = fishSpeechCallUnary(qNorm, queries)
        keys = fishSpeechCallUnary(kNorm, keys)

        if let cache {
            queries = rope.apply(queries, offset: cache.offset)
            keys = rope.apply(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope.apply(queries)
            keys = rope.apply(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)

        return wo(output)
    }
}

final class FishSpeechTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: FishSpeechAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FishSpeechFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(
        dim: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        intermediateSize: Int,
        ropeBase: Float,
        maxPositionEmbeddings: Int,
        attentionQKVBias: Bool,
        attentionOBias: Bool,
        attentionQKNorm: Bool,
        normEps: Float
    ) {
        self._attention.wrappedValue = FishSpeechAttention(
            dim: dim,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            ropeBase: ropeBase,
            maxPositionEmbeddings: maxPositionEmbeddings,
            attentionQKVBias: attentionQKVBias,
            attentionOBias: attentionOBias,
            attentionQKNorm: attentionQKNorm,
            normEps: normEps
        )
        self._feedForward.wrappedValue = FishSpeechFeedForward(dim: dim, hiddenDim: intermediateSize)
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(attentionNorm(x), mask: mask, cache: cache)
        return h + feedForward(ffnNorm(h))
    }
}

final class FishSpeechDualARTransformer: Module {
    let config: FishSpeechConfig

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "codebook_embeddings") var codebookEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [FishSpeechTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    @ModuleInfo(key: "fast_project_in") var fastProjectIn: Module
    @ModuleInfo(key: "fast_embeddings") var fastEmbeddings: Embedding
    @ModuleInfo(key: "fast_layers") var fastLayers: [FishSpeechTransformerBlock]
    @ModuleInfo(key: "fast_norm") var fastNorm: RMSNorm
    @ModuleInfo(key: "fast_output") var fastOutput: Linear

    var numCodebooks: Int {
        config.audioDecoderConfig.numCodebooks
    }

    init(_ config: FishSpeechConfig) {
        self.config = config

        let textConfig = config.textConfig
        let audioConfig = config.audioDecoderConfig

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: textConfig.vocabSize,
            dimensions: textConfig.dim
        )
        self._codebookEmbeddings.wrappedValue = Embedding(
            embeddingCount: audioConfig.vocabSize * audioConfig.numCodebooks,
            dimensions: textConfig.dim
        )
        self._layers.wrappedValue = (0 ..< textConfig.nLayer).map { _ in
            FishSpeechTransformerBlock(
                dim: textConfig.dim,
                nHeads: textConfig.nHead,
                nKVHeads: textConfig.resolvedLocalHeads,
                headDim: textConfig.headDim,
                intermediateSize: textConfig.intermediateSize,
                ropeBase: textConfig.ropeBase,
                maxPositionEmbeddings: textConfig.maxSeqLen,
                attentionQKVBias: textConfig.attentionQKVBias,
                attentionOBias: textConfig.attentionOBias,
                attentionQKNorm: textConfig.attentionQKNorm,
                normEps: textConfig.normEps
            )
        }
        self._norm.wrappedValue = RMSNorm(dimensions: textConfig.dim, eps: textConfig.normEps)

        self._fastProjectIn.wrappedValue = textConfig.dim == audioConfig.dim
            ? FishSpeechIdentity()
            : Linear(textConfig.dim, audioConfig.dim, bias: false)
        self._fastEmbeddings.wrappedValue = Embedding(
            embeddingCount: audioConfig.vocabSize,
            dimensions: audioConfig.dim
        )
        self._fastLayers.wrappedValue = (0 ..< audioConfig.nLayer).map { _ in
            FishSpeechTransformerBlock(
                dim: audioConfig.dim,
                nHeads: audioConfig.nHead,
                nKVHeads: audioConfig.resolvedLocalHeads,
                headDim: audioConfig.headDim,
                intermediateSize: audioConfig.intermediateSize,
                ropeBase: audioConfig.ropeBase,
                maxPositionEmbeddings: audioConfig.numCodebooks,
                attentionQKVBias: audioConfig.attentionQKVBias,
                attentionOBias: audioConfig.attentionOBias,
                attentionQKNorm: audioConfig.attentionQKNorm,
                normEps: audioConfig.normEps
            )
        }
        self._fastNorm.wrappedValue = RMSNorm(dimensions: audioConfig.dim, eps: audioConfig.normEps)
        self._fastOutput.wrappedValue = Linear(audioConfig.dim, audioConfig.vocabSize, bias: false)
    }

    func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    func makeFastCache() -> [KVCache] {
        fastLayers.map { _ in KVCacheSimple() }
    }

    private func embed(_ input: MLXArray) -> MLXArray {
        let semanticIDs = input[0..., 0, 0...]
        let codebookRows = input[0..., 1..., 0...]

        var vqEmbeddings: [MLXArray] = []
        vqEmbeddings.reserveCapacity(numCodebooks)

        for index in 0 ..< numCodebooks {
            let row = codebookRows[0..., index, 0...]
            let rowOffset = row + Int32(index * config.audioDecoderConfig.vocabSize)
            vqEmbeddings.append(codebookEmbeddings(rowOffset))
        }

        let vqSum = vqEmbeddings.reduce(MLXArray.zeros([semanticIDs.dim(0), semanticIDs.dim(1), config.textConfig.dim], dtype: .float32)) {
            $0 + $1
        }

        let semanticMask =
            (semanticIDs .>= MLXArray(Int32(config.semanticStartTokenID)))
            .&& (semanticIDs .<= MLXArray(Int32(config.semanticEndTokenID)))
        let semanticMaskExpanded = semanticMask.expandedDimensions(axis: -1)
        let semanticEmbeddings = embeddings(semanticIDs)
        let semanticEmbeddingDType = semanticEmbeddings.dtype
        let combined = semanticEmbeddings + MLX.where(
            semanticMaskExpanded,
            vqSum.asType(semanticEmbeddingDType),
            MLXArray.zeros(vqSum.shape, dtype: semanticEmbeddingDType)
        )
        let scale = MLXArray(Float(sqrt(Double(numCodebooks + 1)))).asType(semanticEmbeddingDType)
        let scaled = combined / scale
        return MLX.where(semanticMaskExpanded, scaled, combined)
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: [KVCache]? = nil
    ) -> FishSpeechForwardResult {
        var hidden = embed(input)
        let caches: [KVCache?] = cache?.map { Optional($0) }
            ?? Array(repeating: nil, count: layers.count)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (layer, layerCache) in zip(layers, caches) {
            hidden = layer(hidden, mask: mask, cache: layerCache)
        }

        let slowOut = norm(hidden)
        return FishSpeechForwardResult(
            logits: embeddings.asLinear(slowOut),
            hiddenStates: fishSpeechCallUnary(fastProjectIn, slowOut)
        )
    }

    func fastForwardCached(_ x: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden: MLXArray
        if x.ndim == 2 {
            hidden = x.expandedDimensions(axis: 1)
        } else if x.ndim == 3 {
            let last = x.dim(1) - 1
            hidden = x[0..., last..<(last + 1), 0...]
        } else {
            hidden = x
        }

        let caches: [KVCache?] = cache?.map { Optional($0) }
            ?? Array(repeating: nil, count: fastLayers.count)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (layer, layerCache) in zip(fastLayers, caches) {
            hidden = layer(hidden, mask: mask, cache: layerCache)
        }
        return fastOutput(fastNorm(hidden))[0..., -1, 0...]
    }
}

private func fishSpeechFormatDuration(_ seconds: Double) -> String {
    let hours = Int(seconds / 3600)
    let minutes = Int(seconds.truncatingRemainder(dividingBy: 3600) / 60)
    let secs = seconds.truncatingRemainder(dividingBy: 60)
    return String(format: "%02d:%02d:%06.3f", hours, minutes, secs)
}

private func fishSpeechAdjustSpeed(_ audio: MLXArray, speed: Float) -> MLXArray {
    guard abs(speed - 1.0) > 1e-6 else { return audio }

    let oldLength = audio.dim(0)
    let newLength = max(1, Int(Float(oldLength) / speed))
    let positions = MLX.linspace(Float(0), Float(oldLength - 1), count: newLength)
    let left = floor(positions).asType(.int32)
    let right = minimum(left + Int32(1), Int32(oldLength - 1))
    let rightWeight = positions - left.asType(MLX.DType.float32)
    let leftWeight = MLXArray(1.0, dtype: .float32) - rightWeight

    return leftWeight * audio[left] + rightWeight * audio[right]
}

private func fishSpeechSampleToken(
    logits: MLXArray,
    temperature: Float,
    topP: Float,
    topK: Int
) -> MLXArray {
    if temperature <= 0 {
        let maxValues = MLX.max(logits, axis: -1, keepDims: true)
        let vocabSize = logits.dim(logits.ndim - 1)
        var indices = MLXArray(0 ..< vocabSize).reshaped([1, -1]).asType(.int32)
        if logits.ndim > 1 {
            indices = MLX.broadcast(indices, to: logits.shape)
        }
        let firstMaxIndices = MLX.where(
            logits .== maxValues,
            indices,
            MLXArray(Int32.max)
        )
        return MLX.min(firstMaxIndices, axis: -1).asType(.int32)
    }

    let vocabSize = logits.dim(logits.ndim - 1)
    let effectiveTopK: Int
    if topK <= 0 || topK > vocabSize {
        effectiveTopK = vocabSize
    } else {
        effectiveTopK = topK
    }

    let sortedIndices = argSort(-logits, axis: -1)
    let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
    let cumulativeProbabilities = cumsum(softmax(sortedLogits, axis: -1), axis: -1)

    var rankIndices = MLXArray(0 ..< vocabSize).reshaped([1, -1]).asType(.int32)
    if sortedLogits.ndim > 1 {
        rankIndices = MLX.broadcast(rankIndices, to: sortedLogits.shape)
    }

    let removeTopP = cumulativeProbabilities .> MLXArray(topP)
    let removeTopK = rankIndices .>= MLXArray(Int32(effectiveTopK))
    let removeRaw = removeTopP .|| removeTopK
    let removeSorted = MLX.where(
        rankIndices .== MLXArray(Int32(0)),
        MLXArray(false),
        removeRaw
    )

    let indices = MLXArray(0 ..< vocabSize).reshaped([1, -1]).asType(.int32)
    let inverseIndices = putAlong(
        MLXArray.zeros(sortedIndices.shape, type: Int32.self),
        sortedIndices.asType(.int32),
        values: indices,
        axis: -1
    )
    let removeOriginalOrder = takeAlong(removeSorted, inverseIndices, axis: -1)
    let filteredLogits = MLX.where(removeOriginalOrder, MLXArray(-Float.infinity), logits)
        .asType(.float32)
    let probabilities = softmax(
        filteredLogits * (1.0 / max(temperature, 1e-5)),
        axis: -1
    )
    let noise = -log(MLXRandom.uniform(low: 1e-6, high: 1.0, probabilities.shape))
    return argMax(probabilities / noise, axis: -1).asType(.int32)
}

private func fishSpeechLoadWeights(from directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    var weights: [String: MLXArray] = [:]
    for file in files {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

public final class FishSpeechModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public static let defaultRepositoryID = "mlx-community/fish-audio-s2-pro-8bit"

    public let config: FishSpeechConfig
    public private(set) var modelDirectory: URL?

    @ModuleInfo(key: "model") var model: FishSpeechDualARTransformer

    var tokenizer: FishSpeechTokenizer?
    var codec: FishS1DAC?
    var semanticLogitBias: MLXArray?

    public var sampleRate: Int {
        config.sampleRate
    }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 1_024,
            temperature: 0.7,
            topP: 0.7
        )
    }

    public init(config: FishSpeechConfig) {
        self.config = config
        self._model.wrappedValue = FishSpeechDualARTransformer(config)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var remapped: [String: MLXArray] = [:]
        remapped.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.hasPrefix("model.") {
                remapped[key] = value
            } else if key.hasPrefix("text_model.model.") {
                let suffix = String(key.dropFirst("text_model.model.".count))
                remapped["model.\(suffix)"] = value
            } else if key.hasPrefix("audio_decoder.") {
                let suffix = String(key.dropFirst("audio_decoder.".count))
                if suffix.hasPrefix("codebook_embeddings.") {
                    remapped["model.\(suffix)"] = value
                } else {
                    remapped["model.fast_\(suffix)"] = value
                }
            }
        }

        return remapped
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = voice
        _ = language

        let segments = try generateSegments(
            text: text,
            refAudio: refAudio,
            refText: refText,
            maxTokens: generationParameters.maxTokens ?? 1_024,
            temperature: generationParameters.temperature,
            topP: generationParameters.topP,
            topK: 30,
            speed: 1.0,
            chunkLength: 300
        )
        return concatenateAudioSegments(segments)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = voice
        _ = language
        _ = streamingInterval

        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish()
                return
            }

            do {
                let segments = try self.generateSegments(
                    text: text,
                    refAudio: refAudio,
                    refText: refText,
                    maxTokens: generationParameters.maxTokens ?? 1_024,
                    temperature: generationParameters.temperature,
                    topP: generationParameters.topP,
                    topK: 30,
                    speed: 1.0,
                    chunkLength: 300
                )
                let audio = self.concatenateAudioSegments(segments)
                let totalPromptTokens = segments.reduce(into: 0) { $0 += $1.promptTokenCount }
                let totalGenerationTokens = segments.reduce(into: 0) { $0 += $1.generationTokenCount }
                let totalTime = segments.reduce(0.0) { $0 + $1.elapsed }
                let peakMemory = segments.map(\.peakMemoryUsage).max() ?? 0

                continuation.yield(.info(AudioGenerationInfo(
                    promptTokenCount: totalPromptTokens,
                    generationTokenCount: totalGenerationTokens,
                    prefillTime: 0,
                    generateTime: totalTime,
                    tokensPerSecond: totalTime > 0 ? Double(totalGenerationTokens) / totalTime : 0,
                    peakMemoryUsage: peakMemory
                )))
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    private func postLoadHook(modelDir: URL) async throws {
        self.modelDirectory = modelDir
        self.tokenizer = try await FishSpeechTokenizer.fromModelDirectory(
            modelDir,
            vocabSizeHint: config.textConfig.vocabSize
        )
        self.codec = try FishS1DAC.fromModelDirectory(modelDir)
        self.semanticLogitBias = try buildSemanticLogitBias()
    }

    private func buildSemanticLogitBias() throws -> MLXArray {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let imEndID = tokenizer.tokenID(for: fishSpeechIMEndToken) ?? 0
        let vocabSize = max(
            config.textConfig.vocabSize,
            tokenizer.vocabSize,
            config.semanticEndTokenID + 1,
            imEndID + 1
        )
        var bias = Array(repeating: Float(-1e9), count: vocabSize)
        let semanticUpperBound = min(config.semanticEndTokenID, vocabSize - 1)
        if config.semanticStartTokenID <= semanticUpperBound {
            for index in config.semanticStartTokenID ... semanticUpperBound {
                bias[index] = 0
            }
        }
        if imEndID < bias.count {
            bias[imEndID] = 0
        }
        return MLXArray(bias).reshaped([1, vocabSize])
    }

    private func buildConversation(
        promptTexts: [String],
        promptTokens: [MLXArray]
    ) -> FishSpeechConversation {
        var conversation = FishSpeechConversation()

        let systemParts: [FishSpeechPart]
        if !promptTexts.isEmpty, !promptTokens.isEmpty {
            let taggedPromptTexts = promptTexts.enumerated().map { index, text in
                text.contains("<|speaker:") ? text : "<|speaker:\(index)|>\(text)"
            }
            let allPromptTokens = concatenated(promptTokens, axis: 1)
            systemParts = [
                .text(FishSpeechTextPart(text: "convert the provided text to speech reference to the following:\n\nText:\n")),
                .text(FishSpeechTextPart(text: taggedPromptTexts.joined(separator: "\n"))),
                .text(FishSpeechTextPart(text: "\n\nSpeech:\n")),
                .vq(FishSpeechVQPart(allPromptTokens)),
            ]
        } else {
            systemParts = [.text(FishSpeechTextPart(text: "convert the provided text to speech"))]
        }

        conversation.append(FishSpeechMessage(
            role: .system,
            parts: systemParts,
            addIMStart: true,
            addIMEnd: true,
            modality: nil
        ))
        return conversation
    }

    private func prepareReferencePromptAudio(_ refAudio: MLXArray) -> MLXArray {
        switch refAudio.ndim {
        case 1:
            return refAudio.expandedDimensions(axis: 0).expandedDimensions(axis: 0)

        case 2:
            if refAudio.shape[0] == 1 {
                return refAudio.expandedDimensions(axis: 0)
            }
            if refAudio.shape[1] == 1 {
                return refAudio.transposed(1, 0).expandedDimensions(axis: 0)
            }
            if refAudio.shape[0] <= 8 {
                return mean(refAudio, axis: 0, keepDims: true).expandedDimensions(axis: 0)
            }
            if refAudio.shape[1] <= 8 {
                return mean(refAudio, axis: 1, keepDims: true).transposed(1, 0).expandedDimensions(axis: 0)
            }
            return mean(refAudio, axis: 0, keepDims: true).expandedDimensions(axis: 0)

        case 3:
            if refAudio.shape[0] != 1 {
                return prepareReferencePromptAudio(refAudio[0, 0..., 0...])
            }
            if refAudio.shape[1] == 1 {
                return refAudio
            }
            if refAudio.shape[2] == 1 {
                return refAudio.transposed(0, 2, 1)
            }
            return mean(refAudio, axis: 1, keepDims: true)

        default:
            return refAudio
        }
    }

    private func sampleSemantic(
        logits: MLXArray,
        previousSemanticTokens: [Int],
        topP: Float,
        topK: Int,
        temperature: Float
    ) throws -> MLXArray {
        guard let semanticLogitBias else {
            throw AudioGenerationError.modelNotInitialized("Semantic logit bias not initialized")
        }

        let biasedLogits = logits + semanticLogitBias.asType(logits.dtype)
        let normal = fishSpeechSampleToken(
            logits: biasedLogits,
            temperature: temperature,
            topP: topP,
            topK: topK
        )
        let highTemperature = fishSpeechSampleToken(
            logits: biasedLogits,
            temperature: fishSpeechRASHighTemperature,
            topP: fishSpeechRASHighTopP,
            topK: topK
        )
        eval(normal, highTemperature)

        let tokenValue = Int(normal.item(Int32.self))
        let shouldUseHighTemperature =
            previousSemanticTokens.contains(tokenValue)
            && tokenValue >= config.semanticStartTokenID
            && tokenValue <= config.semanticEndTokenID

        return shouldUseHighTemperature ? highTemperature : normal
    }

    private func generateCodesForBatch(
        conversation: FishSpeechConversation,
        batchText: String,
        maxNewTokens: Int,
        topP: Float,
        topK: Int,
        temperature: Float
    ) throws -> MLXArray {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        var promptConversation = FishSpeechConversation(messages: conversation.messages)
        promptConversation.append(FishSpeechMessage(
            role: .assistant,
            parts: [],
            addIMStart: true,
            addIMEnd: false,
            modality: .voice
        ))

        let prompt = promptConversation.encodeForInference(
            tokenizer: tokenizer,
            numCodebooks: model.numCodebooks
        ).expandedDimensions(axis: 0)

        let cache = model.makeCache()
        var result = model(prompt, cache: cache)
        var logits = result.logits[0..., (result.logits.dim(1) - 1)..<result.logits.dim(1), 0...]
            .squeezed(axis: 1)
        var hiddenState = result.hiddenStates[0..., (result.hiddenStates.dim(1) - 1)..<result.hiddenStates.dim(1), 0...]
            .squeezed(axis: 1)

        let imEndID = tokenizer.tokenID(for: fishSpeechIMEndToken) ?? config.eosTokenID
        let textTokenCount = tokenizer.encode(batchText, addSpecialTokens: false).count
        let semanticBudget = min(maxNewTokens, max(32, textTokenCount * 12))

        var previousSemanticTokens: [Int] = []
        var generatedSteps: [[Int32]] = []
        generatedSteps.reserveCapacity(semanticBudget)

        for _ in 0 ..< semanticBudget {
            let semanticToken = try sampleSemantic(
                logits: logits,
                previousSemanticTokens: previousSemanticTokens,
                topP: topP,
                topK: topK,
                temperature: temperature
            )

            let semanticTokenID = Int(semanticToken.item(Int32.self))
            if semanticTokenID == imEndID {
                break
            }

            previousSemanticTokens.append(semanticTokenID)
            if previousSemanticTokens.count > fishSpeechRASWindowSize {
                previousSemanticTokens.removeFirst(previousSemanticTokens.count - fishSpeechRASWindowSize)
            }

            let semanticCode = clip(
                semanticToken - Int32(config.semanticStartTokenID),
                min: 0,
                max: Int32(config.audioDecoderConfig.vocabSize - 1)
            ).asType(.int32)

            var previousCodebooks = semanticCode.reshaped([1, 1])
            let fastCache = model.makeFastCache()
            let fastPrefill = model.fastForwardCached(hiddenState, cache: fastCache)
            eval(fastPrefill)
            var fastHidden = model.fastEmbeddings(semanticCode)

            for _ in 0 ..< (model.numCodebooks - 1) {
                let residualLogits = model.fastForwardCached(fastHidden, cache: fastCache)
                let residualToken = fishSpeechSampleToken(
                    logits: residualLogits,
                    temperature: temperature,
                    topP: topP,
                    topK: topK
                ).asType(.int32)
                eval(residualToken)
                previousCodebooks = concatenated(
                    [previousCodebooks, residualToken.reshaped([1, 1])],
                    axis: 1
                )
                fastHidden = model.fastEmbeddings(residualToken)
            }

            generatedSteps.append(previousCodebooks.asArray(Int32.self))

            let nextInput = concatenated(
                [semanticToken.asType(.int32).reshaped([1, 1]), previousCodebooks],
                axis: 1
            ).expandedDimensions(axis: 2)
            result = model(nextInput, cache: cache)
            logits = result.logits[0..., (result.logits.dim(1) - 1)..<result.logits.dim(1), 0...]
                .squeezed(axis: 1)
            hiddenState = result.hiddenStates[0..., (result.hiddenStates.dim(1) - 1)..<result.hiddenStates.dim(1), 0...]
                .squeezed(axis: 1)
        }

        guard !generatedSteps.isEmpty else {
            throw AudioGenerationError.generationFailed(
                "No audio tokens were generated for batch text: \(batchText)"
            )
        }

        var rows = Array(
            repeating: Array(repeating: Int32(0), count: generatedSteps.count),
            count: model.numCodebooks
        )
        for (stepIndex, step) in generatedSteps.enumerated() {
            for codebookIndex in 0 ..< min(model.numCodebooks, step.count) {
                rows[codebookIndex][stepIndex] = step[codebookIndex]
            }
        }

        return MLXArray(rows.flatMap { $0 }).reshaped([model.numCodebooks, generatedSteps.count])
    }

    private func decodeCodes(_ codes: MLXArray) throws -> MLXArray {
        guard let codec else {
            throw AudioGenerationError.modelNotInitialized("Codec not loaded")
        }

        let featureLengths = MLXArray([Int32(codes.dim(1))])
        let (audio, audioLengths) = codec.decode(codes.expandedDimensions(axis: 0), featureLengths: featureLengths)
        let length = Int(audioLengths.item(Int32.self))
        return audio[0, 0, 0..<length]
    }

    private func generateSegments(
        text: String,
        refAudio: MLXArray?,
        refText: String?,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        speed: Float,
        chunkLength: Int
    ) throws -> [FishSpeechGeneratedSegment] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("Text prompt cannot be empty")
        }
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        guard let codec else {
            throw AudioGenerationError.modelNotInitialized("Codec not loaded")
        }

        var promptTokens: [MLXArray] = []
        var promptTexts: [String] = []
        if let refAudio {
            let audio = prepareReferencePromptAudio(refAudio)
            let (indices, featureLengths) = codec.encode(audio)
            let promptLength = Int(featureLengths.item(Int32.self))
            promptTokens.append(indices[0, 0..., 0..<promptLength])
            promptTexts.append(refText ?? "")
        }

        let baseConversation = buildConversation(promptTexts: promptTexts, promptTokens: promptTokens)
        let turns = fishSpeechSplitTextBySpeaker(text)
        let batches = turns.isEmpty
            ? [text]
            : fishSpeechGroupTurnsIntoBatches(turns, maxSpeakers: 5, maxBytes: chunkLength)

        var conversation = baseConversation
        var segments: [FishSpeechGeneratedSegment] = []
        segments.reserveCapacity(batches.count)

        for batchText in batches {
            conversation.append(FishSpeechMessage(
                role: .user,
                parts: [.text(FishSpeechTextPart(text: batchText))],
                addIMStart: true,
                addIMEnd: true,
                modality: nil
            ))

            let startTime = CFAbsoluteTimeGetCurrent()
            let codes = try generateCodesForBatch(
                conversation: conversation,
                batchText: batchText,
                maxNewTokens: maxTokens,
                topP: topP,
                topK: topK,
                temperature: temperature
            )
            var audio = try decodeCodes(codes)
            if abs(speed - 1.0) > 1e-6 {
                audio = fishSpeechAdjustSpeed(audio, speed: speed)
            }
            eval(audio)

            conversation.append(FishSpeechMessage(
                role: .assistant,
                parts: [.vq(FishSpeechVQPart(codes))],
                addIMStart: true,
                addIMEnd: true,
                modality: .voice
            ))

            let elapsed = max(CFAbsoluteTimeGetCurrent() - startTime, 1e-6)
            segments.append(FishSpeechGeneratedSegment(
                audio: audio,
                promptTokenCount: tokenizer.encode(batchText, addSpecialTokens: false).count,
                generationTokenCount: codes.dim(1),
                elapsed: elapsed,
                peakMemoryUsage: Double(Memory.peakMemory) / 1e9
            ))
        }

        return segments
    }

    private func concatenateAudioSegments(_ segments: [FishSpeechGeneratedSegment]) -> MLXArray {
        guard let first = segments.first else { return MLXArray.zeros([0], dtype: .float32) }
        guard segments.count > 1 else { return first.audio }
        return concatenated(segments.map(\.audio), axis: 0)
    }

    public static func fromPretrained(
        _ modelRepo: String = FishSpeechModel.defaultRepositoryID,
        cache: HubCache = .default
    ) async throws -> FishSpeechModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(FishSpeechConfig.self, from: configData)
        let model = FishSpeechModel(config: config)

        let weights = try fishSpeechLoadWeights(from: modelDir)
        let sanitizedWeights = model.sanitize(weights: weights)

        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitizedWeights["\(path).scales"] != nil else { return nil }
                if let perLayerQuantization = config.perLayerQuantization,
                   let layerQuant = perLayerQuantization.quantization(layer: path)
                {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .all)
        eval(model)
        try await model.postLoadHook(modelDir: modelDir)
        return model
    }
}
