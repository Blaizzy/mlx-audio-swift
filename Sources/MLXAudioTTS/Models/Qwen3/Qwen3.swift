//
//  Qwen3.swift
//  MLXAudio
//
//  Based on mlx-swift-lm Qwen3 model with TTS extensions.
//

import Foundation
@preconcurrency import MLX
import HuggingFace
import Tokenizers
@preconcurrency import MLXLMCommon
import MLXNN
import MLXAudioCodecs
import MLXAudioCore
import Combine

// MARK: - VyvoTTS special token IDs (Qwen3-based tokenizer)
let tokenizerLength = 151669
let startOfText = 151643
let endOfText = 151645
let startOfSpeech = tokenizerLength + 1  // 151670
let endOfSpeech = tokenizerLength + 2  // 151671
let startOfHuman = tokenizerLength + 3  // 151672
let endOfHuman = tokenizerLength + 4  // 151673
let startOfAI = tokenizerLength + 5  // 151674
let endOfAI = tokenizerLength + 6  // 151675
let padTokenId = tokenizerLength + 7  // 151676
let audioTokensStart = tokenizerLength + 10  // 151679

// MARK: - Type Aliases
public typealias Qwen3Error = AudioGenerationError
public typealias Qwen3GenerationInfo = AudioGenerationInfo
public typealias Qwen3Generation = AudioGeneration

// MARK: - Decode
func decodeAudioFromCodes(codeList: [Int], snacModel: SNAC, chunkSize: Int = 50) -> MLXArray {
    let numGroups = (codeList.count + 1) / 7
    if numGroups <= chunkSize {
        let result = decodeAudioChunk(codeList: codeList, snacModel: snacModel)
        eval(result)
        return result
    }
    let estimatedSamples = numGroups * snacModel.hopLength
    var audioSamples = ContiguousArray<Float>()
    audioSamples.reserveCapacity(estimatedSamples)
    var groupStart = 0
    while groupStart < numGroups {
        let groupEnd = min(groupStart + chunkSize, numGroups)
        let codeStart = groupStart * 7
        let codeEnd = groupEnd * 7
        let chunkCodes = Array(codeList[codeStart..<min(codeEnd, codeList.count)])
        let audioChunk = decodeAudioChunk(codeList: chunkCodes, snacModel: snacModel)
        eval(audioChunk)
        audioSamples.append(contentsOf: audioChunk.asArray(Float.self))
        Memory.clearCache()
        groupStart = groupEnd
    }
    return MLXArray(Array(audioSamples))
}

private func decodeAudioChunk(codeList: [Int], snacModel: SNAC) -> MLXArray {
    var layer1: [Int] = []
    var layer2: [Int] = []
    var layer3: [Int] = []
    let numGroups = (codeList.count + 1) / 7
    for i in 0..<numGroups {
        let baseIdx = 7 * i
        layer1.append(codeList[baseIdx])
        layer2.append(codeList[baseIdx + 1] - 4096)
        layer3.append(codeList[baseIdx + 2] - (2 * 4096))
        layer3.append(codeList[baseIdx + 3] - (3 * 4096))
        layer2.append(codeList[baseIdx + 4] - (4 * 4096))
        layer3.append(codeList[baseIdx + 5] - (5 * 4096))
        layer3.append(codeList[baseIdx + 6] - (6 * 4096))
    }
    let codes = [
        MLXArray(layer1).expandedDimensions(axis: 0),
        MLXArray(layer2).expandedDimensions(axis: 0),
        MLXArray(layer3).expandedDimensions(axis: 0)
    ]
    let audioHat = snacModel.decode(codes).squeezed()
    return audioHat
}

func encodeAudioToCodes(audio: MLXArray, snacModel: SNAC) -> MLXArray {
    let audioExpanded = audio.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let codes = snacModel.encode(audioExpanded)
    let layer1 = codes[0].squeezed(axis: 0).asArray(Int.self)
    let layer2 = codes[1].squeezed(axis: 0).asArray(Int.self)
    let layer3 = codes[2].squeezed(axis: 0).asArray(Int.self)
    var codeList: [Int] = []
    let numGroups = layer1.count
    for i in 0..<numGroups {
        codeList.append(layer1[i])
        codeList.append(layer2[2 * i] + 4096)
        codeList.append(layer3[4 * i] + 2 * 4096)
        codeList.append(layer3[4 * i + 1] + 3 * 4096)
        codeList.append(layer2[2 * i + 1] + 4 * 4096)
        codeList.append(layer3[4 * i + 2] + 5 * 4096)
        codeList.append(layer3[4 * i + 3] + 6 * 4096)
    }
    return MLXArray(codeList).expandedDimensions(axis: 0)
}

// MARK: - Attention
public class Qwen3Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

public class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

public class Qwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    let mlp: Qwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(args)
        self.mlp = Qwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Qwen3Model: Module, KVCacheDimensionProvider, SpeechGenerationModel, @unchecked Sendable {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var tokenizer: Tokenizer?
    public var _snacModel: SNAC?

    public let model: Qwen3ModelInner
    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var numLayers: Int {
        return self.configuration.hiddenLayers
    }

    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func parseOutputRow(_ tokens: [Int32]) -> [Int] {
        let tokensI = tokens.map(Int.init)
        var startIdx: Int? = tokensI.lastIndex(of: startOfSpeech)
        if startIdx == nil, let soaIdx = tokensI.lastIndex(of: startOfAI) {
            if let firstAudio = tokensI[(soaIdx + 1)...].firstIndex(where: { $0 >= audioTokensStart }) {
                startIdx = firstAudio - 1
            }
        }
        let slice = (startIdx != nil) ? Array(tokensI[(startIdx! + 1)...]) : tokensI
        let filtered = slice.filter { $0 != endOfSpeech }
        let newLen = (filtered.count / 7) * 7
        guard newLen > 0 else { return [] }
        return filtered.prefix(newLen).map { $0 - audioTokensStart }
    }

    public func parseOutputBatch(_ tokensPerBatch: [[Int32]]) -> [[Int]] {
        return tokensPerBatch.map { parseOutputRow($0) }
    }

    public func parseOutput(_ inputIds: MLXArray) -> [[Int]] {
        var codeLists: [[Int]] = []
        for i in 0..<inputIds.shape[0] {
            let row = inputIds[i].asArray(Int32.self)
            codeLists.append(parseOutputRow(row))
        }
        return codeLists
    }

    public func prepareInputIds(
        prompts: [String],
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil
    ) -> (MLXArray, MLXArray) {
        var refAudioCodes: [Int32]? = nil
        var refTranscriptIds: [Int32]? = nil

        if let refAudio = refAudio, let refText = refText {
            print("\u{001B}[93mWARNING: Audio cloning doesn't work reliably on this model.\u{001B}[0m")
            guard let snacModel = self._snacModel else {
                fatalError("SNAC model not loaded. Call post_load_hook first.")
            }
            let codes = encodeAudioToCodes(audio: refAudio, snacModel: snacModel)
            let codesArray = (codes + audioTokensStart).asArray(Int32.self)
            refAudioCodes = codesArray
            refTranscriptIds = tokenizer!.encode(text: refText).map { Int32($0) }
        }

        let modifiedPrompts: [String]
        if let voice = voice {
            modifiedPrompts = prompts.map { "\(voice): \($0)" }
        } else {
            modifiedPrompts = prompts
        }

        let encodedPrompts: [[Int32]] = modifiedPrompts.map { prompt in
            tokenizer!.encode(text: prompt).map { Int32($0) }
        }

        let maxPromptLen = encodedPrompts.map { $0.count }.max() ?? 0
        var allBatchIds: [[Int32]] = []

        for encodedPrompt in encodedPrompts {
            var sequence: [Int32] = []
            let paddingLen = maxPromptLen - encodedPrompt.count
            if paddingLen > 0 {
                sequence.append(contentsOf: [Int32](repeating: Int32(padTokenId), count: paddingLen))
            }
            if let refCodes = refAudioCodes, let refTranscript = refTranscriptIds {
                sequence.append(Int32(startOfHuman))
                sequence.append(contentsOf: refTranscript)
                sequence.append(Int32(endOfText))
                sequence.append(Int32(endOfHuman))
                sequence.append(Int32(startOfAI))
                sequence.append(Int32(startOfSpeech))
                sequence.append(contentsOf: refCodes)
                sequence.append(Int32(endOfSpeech))
                sequence.append(Int32(endOfAI))
            }
            sequence.append(Int32(startOfHuman))
            sequence.append(contentsOf: encodedPrompt)
            sequence.append(Int32(endOfText))
            sequence.append(Int32(endOfHuman))
            allBatchIds.append(sequence)
        }

        let maxLen = allBatchIds.map { $0.count }.max() ?? 0
        var flattenedIds: [Int32] = []
        flattenedIds.reserveCapacity(allBatchIds.count * maxLen)

        for var sequence in allBatchIds {
            let padCount = maxLen - sequence.count
            if padCount > 0 {
                sequence.insert(contentsOf: [Int32](repeating: Int32(padTokenId), count: padCount), at: 0)
            }
            flattenedIds.append(contentsOf: sequence)
        }

        let batchSize = allBatchIds.count
        let finalBatchInputIds = MLXArray(flattenedIds).reshaped([batchSize, maxLen])
        let batchMask = finalBatchInputIds .!= Int32(padTokenId)
        return (finalBatchInputIds, batchMask)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func forwardWithEmbeddings(
        inputsEmbeds: MLXArray,
        cache: [KVCache]? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        var h = inputsEmbeds
        let resolvedMask: MLXFast.ScaledDotProductAttentionMaskMode = mask ?? .none
        for (i, layer) in model.layers.enumerated() {
            h = layer(h, mask: resolvedMask, cache: cache?[i])
        }
        h = model.norm(h)
        return h
    }

    public func getEmbeddings(for inputIds: MLXArray) -> MLXArray {
        return model.embedTokens(inputIds)
    }

    public var sampleRate: Int {
        return self.configuration.sampleRate
    }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 1200,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights
    }

    public func post_load_hook(model: Qwen3Model, modelDir: URL, cache: HubCache = .default) async throws {
        if model.tokenizer == nil {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        }
        if model._snacModel == nil {
            model._snacModel = try await SNAC.fromPretrained("mlx-community/snac_24khz", cache: cache)
        }
    }

    public func makeCache() -> [KVCache] {
        return (0..<self.configuration.hiddenLayers).map { _ in
            KVCacheSimple()
        }
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language _: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try await generate(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            cache: nil,
            parameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language _: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            cache: nil,
            parameters: generationParameters
        )
    }

    public func generate(
        text: String,
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 1200,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )
    ) async throws -> MLXArray {
        guard let snacModel = _snacModel else {
            throw Qwen3Error.modelNotInitialized("SNAC model not loaded")
        }
        guard tokenizer != nil else {
            throw Qwen3Error.modelNotInitialized("Tokenizer not loaded")
        }

        let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        let (inputIds, _) = prepareInputIds(
            prompts: [prompt],
            voice: voice,
            refAudio: refAudio,
            refText: refText
        )

        let sampler = parameters.sampler()
        var processor = parameters.processor()

        let promptTokens = inputIds.squeezed(axis: 0)
        processor?.prompt(promptTokens)

        var cache = cache
        if cache == nil {
            cache = self.makeCache()
        }

        let maxTokens = parameters.maxTokens ?? 1200
        let promptTokensList = inputIds.squeezed(axis: 0).asArray(Int32.self)
        var generatedOnly = ContiguousArray<Int32>()
        generatedOnly.reserveCapacity(maxTokens)

        var logits = self(inputIds, cache: cache)
        logits = logits[0..., -1, 0...]
        eval(logits)

        for _ in 0..<maxTokens {
            let tokenValue: Int = autoreleasepool {
                var lastLogits = logits
                lastLogits = processor?.process(logits: lastLogits) ?? lastLogits
                let nextToken = sampler.sample(logits: lastLogits)
                processor?.didSample(token: nextToken)
                let value = nextToken.item(Int.self)
                if value != endOfSpeech {
                    let nextTokenExpanded = nextToken.reshaped([1, 1])
                    logits = self(nextTokenExpanded, cache: cache)
                    logits = logits[0..., -1, 0...]
                    eval(logits)
                }
                return value
            }
            if tokenValue == endOfSpeech {
                break
            }
            generatedOnly.append(Int32(tokenValue))
        }

        Memory.clearCache()
        var fullTokens = ContiguousArray<Int32>()
        fullTokens.reserveCapacity(promptTokensList.count + generatedOnly.count)
        fullTokens.append(contentsOf: promptTokensList)
        fullTokens.append(contentsOf: generatedOnly)
        let codeList = parseOutputRow(Array(fullTokens))
        guard !codeList.isEmpty else {
            throw Qwen3Error.generationFailed("No audio codes generated")
        }
        let audio = decodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
        audio.eval()
        Memory.clearCache()
        return audio
    }

    public func generateStream(
        text: String,
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 1200,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )
    ) -> AsyncThrowingStream<Qwen3Generation, Error> {
        let (stream, continuation) = AsyncThrowingStream<Qwen3Generation, Error>.makeStream()
        Task { @Sendable [weak self, continuation] in
            guard let self else { return }
            do {
                guard let snacModel = self._snacModel else {
                    throw Qwen3Error.modelNotInitialized("SNAC model not loaded")
                }
                guard self.tokenizer != nil else {
                    throw Qwen3Error.modelNotInitialized("Tokenizer not loaded")
                }
                let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
                    .replacingOccurrences(of: "\\t", with: "\t")
                let (inputIds, _) = self.prepareInputIds(
                    prompts: [prompt],
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText
                )
                let sampler = parameters.sampler()
                var processor = parameters.processor()
                let promptTokens = inputIds.squeezed(axis: 0)
                processor?.prompt(promptTokens)
                var cache = cache
                if cache == nil {
                    cache = self.makeCache()
                }
                let maxTokens = parameters.maxTokens ?? 1200
                let promptTokensList = inputIds.squeezed(axis: 0).asArray(Int32.self)
                var generatedTokens = ContiguousArray<Int32>()
                generatedTokens.reserveCapacity(maxTokens)
                let startTime = Date()
                var tokenCount: Int = 0
                var logits = self(inputIds, cache: cache)
                logits = logits[0..., -1, 0...]
                eval(logits)
                let prefillTime = Date().timeIntervalSince(startTime)
                let generateStartTime = Date()
                for _ in 0..<maxTokens {
                    if Task.isCancelled { break }
                    let tokenValue: Int = autoreleasepool {
                        var lastLogits = logits
                        lastLogits = processor?.process(logits: lastLogits) ?? lastLogits
                        let nextToken = sampler.sample(logits: lastLogits)
                        processor?.didSample(token: nextToken)
                        let value = nextToken.item(Int.self)
                        if value != endOfSpeech {
                            let nextTokenExpanded = nextToken.reshaped([1, 1])
                            logits = self(nextTokenExpanded, cache: cache)
                            logits = logits[0..., -1, 0...]
                            eval(logits)
                        }
                        return value
                    }
                    tokenCount += 1
                    continuation.yield(.token(tokenValue))
                    if tokenValue == endOfSpeech {
                        break
                    }
                    generatedTokens.append(Int32(tokenValue))
                }
                Memory.clearCache()
                let generateTime = Date().timeIntervalSince(generateStartTime)
                var fullTokens = ContiguousArray<Int32>()
                fullTokens.reserveCapacity(promptTokensList.count + generatedTokens.count)
                fullTokens.append(contentsOf: promptTokensList)
                fullTokens.append(contentsOf: generatedTokens)
                let codeList = self.parseOutputRow(Array(fullTokens))
                guard !codeList.isEmpty else {
                    throw Qwen3Error.generationFailed("No audio codes generated")
                }
                let audio = decodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
                audio.eval()
                Memory.clearCache()
                let info = Qwen3GenerationInfo(
                    promptTokenCount: inputIds.shape[1],
                    generationTokenCount: tokenCount,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: Double(tokenCount) / generateTime,
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9
                )
                continuation.yield(.info(info))
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> Qwen3Model {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "Qwen3Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        cache: HubCache = .default
    ) async throws -> Qwen3Model {
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: configData)
        let perLayerQuantization = config.perLayerQuantization
        let model = Qwen3Model(config)
        let weights = try loadWeights(from: modelDir)
        let sanitizedWeights = model.sanitize(weights: weights)
        if perLayerQuantization != nil {
            print("Applying quantizaiton from config...")
            if let perLayerQuant = perLayerQuantization {
                print(" Per-layer: \(perLayerQuant)")
            }
            quantize(model: model) { path, module in
                if weights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: path)?.asTuple
                } else {
                    return nil
                }
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .all)
        eval(model)
        try await model.post_load_hook(model: model, modelDir: modelDir, cache: cache)
        return model
    }
}

func loadWeights(from directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}
