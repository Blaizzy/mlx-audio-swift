//
//  GLMASR.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace
import Tokenizers

// MARK: - Audio Processing Constants

enum AudioConstants {
    static let sampleRate = 16000
    static let nFft = 400
    static let hopLength = 160
}

// MARK: - Prompt Templates

private enum PromptTemplate {
    static let userPrefix = "<|user|>\n<|begin_of_audio|>"
    static let userSuffix = "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
}

extension GLMASRModel: STTGenerationModel {
    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(maxTokens: 128, temperature: 0.0, topP: 0.95, topK: 0, verbose: false)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        generate(
            audio: audio,
            maxTokens: generationParameters.maxTokens,
            temperature: generationParameters.temperature,
            topP: generationParameters.topP,
            topK: generationParameters.topK,
            verbose: generationParameters.verbose
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(
            audio: audio,
            maxTokens: generationParameters.maxTokens,
            temperature: generationParameters.temperature,
            topP: generationParameters.topP
        )
    }
}

// MARK: - LLaMA Components for STT

class GLMASRRoPE: Module {
    let dims: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    init(dims: Int, traditional: Bool = false, base: Float = 10000.0, scale: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset
        )
    }
}

class GLMASRAttention: Module {
    let config: LlamaConfig
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: GLMASRRoPE

    init(_ config: LlamaConfig) {
        self.config = config

        let dim = config.hiddenSize
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads

        let headDim = config.headDim ?? (dim / heads)
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: config.attentionBias)

        self.rope = GLMASRRoPE(
            dims: headDim,
            traditional: config.ropeTraditional,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, config.numAttentionHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, config.numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, config.numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(output)
    }
}

class GLMASRMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: LlamaConfig) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

class GLMASRTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: GLMASRAttention
    @ModuleInfo(key: "mlp") var mlp: GLMASRMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: LlamaConfig) {
        self._attention.wrappedValue = GLMASRAttention(config)
        self._mlp.wrappedValue = GLMASRMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

class GLMASRModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLMASRTransformerBlock]
    let norm: RMSNorm

    init(_ config: LlamaConfig) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        self.layers = (0..<config.numHiddenLayers).map { _ in GLMASRTransformerBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings = inputEmbeddings {
            h = inputEmbeddings
        } else if let inputs = inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Either inputs or inputEmbeddings must be provided")
        }

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Language Model

/// Language model wrapper for GLM-ASR that supports input embeddings.
public class GLMASRLanguageModel: Module, KVCacheDimensionProvider {
    let config: LlamaConfig

    @ModuleInfo(key: "model") var model: GLMASRModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var kvHeads: [Int] {
        return (0..<config.numHiddenLayers).map { _ in config.numKeyValueHeads }
    }

    public init(config: LlamaConfig) {
        self.config = config
        self._model.wrappedValue = GLMASRModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(
        inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)

        if let lmHead = lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public var embedTokens: Embedding {
        return model.embedTokens
    }
}

// MARK: - Generation Context

/// Internal context for managing generation state.
private struct GenerationContext {
    let tokenizer: Tokenizer
    let cache: [KVCache]
    let eosTokenIds: [Int]
    var logits: MLXArray

    /// Sample next token from logits.
    func sampleNextToken(temperature: Float) -> Int {
        var lastLogits = logits[0..., -1, 0...]
        if temperature > 0 {
            lastLogits = lastLogits / temperature
        }
        return lastLogits.argMax(axis: -1).item(Int.self)
    }

    /// Check if token is an end-of-sequence token.
    func isEOS(_ token: Int) -> Bool {
        eosTokenIds.contains(token)
    }

    /// Decode token to text.
    func decode(_ token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    /// Decode tokens to text.
    func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }
}

// MARK: - GLM-ASR Model

/// GLM-ASR model combining Whisper encoder with LLaMA decoder.
///
/// Weight structure matches HuggingFace format:
/// - audio_encoder.* : Audio encoder with Whisper + MLP adapter
/// - model.* / language_model.model.* : LLaMA decoder
/// - lm_head.* / language_model.lm_head.* : Language modeling head
public class GLMASRModel: Module {
    public let config: GLMASRModelConfig
    public let vocabSize: Int

    @ModuleInfo(key: "audio_encoder") var audioEncoder: AudioEncoder
    @ModuleInfo(key: "language_model") var languageModel: GLMASRLanguageModel

    public var tokenizer: Tokenizer?

    public init(config: GLMASRModelConfig) {
        self.config = config
        self.vocabSize = config.lmConfig.vocabSize

        self._audioEncoder.wrappedValue = AudioEncoder(config: config)
        self._languageModel.wrappedValue = GLMASRLanguageModel(config: config.lmConfig)
    }

    // MARK: - Public API

    /// Get the input embeddings from the language model.
    public func getInputEmbeddings() -> Embedding {
        return languageModel.embedTokens
    }

    /// Forward pass.
    public func callAsFunction(
        inputIds: MLXArray,
        audios: MLXArray? = nil,
        audioEmbeds: MLXArray? = nil,
        audioOffsets: [[Int]]? = nil,
        audioLength: [[Int]]? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var computedAudioEmbeds = audioEmbeds

        // Compute audio embeddings if raw audio provided and no pre-computed embeds
        if let audios = audios, audioEmbeds == nil {
            let (embeds, _) = audioEncoder(audios)
            computedAudioEmbeds = embeds
        }

        let inputEmbeds = mergeAudioTextEmbeddings(
            inputIds: inputIds,
            audioEmbeds: computedAudioEmbeds,
            audioOffsets: audioOffsets,
            audioLength: audioLength,
            cache: cache
        )

        return languageModel(cache: cache, inputEmbeddings: inputEmbeds)
    }

    /// Preprocess audio to mel spectrogram.
    public func preprocessAudio(_ audio: MLXArray) -> MLXArray {
        let nMels = config.whisperConfig.numMelBins

        // If already 3D (batch, seq, mels), assume it's mel spectrogram
        if audio.ndim == 3 {
            return audio
        }

        // Compute mel spectrogram
        let melSpec = MLXAudioCore.computeMelSpectrogram(
            audio: audio,
            sampleRate: AudioConstants.sampleRate,
            nFft: AudioConstants.nFft,
            hopLength: AudioConstants.hopLength,
            nMels: nMels
        )

        // Add batch dimension: (seq_len, n_mels) -> (1, seq_len, n_mels)
        return melSpec.expandedDimensions(axis: 0)
    }

    /// Generate transcription from audio.
    public func generate(
        audio: MLXArray,
        maxTokens: Int = 128,
        temperature: Float = 0.0,
        topP: Float = 0.95,
        topK: Int = 0,
        verbose: Bool = false
    ) -> STTOutput {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded")
        }

        let startTime = Date()

        // Prepare for generation
        let (context, promptTokenCount) = prepareGeneration(audio: audio, tokenizer: tokenizer)
        var ctx = context

        // Generate tokens
        var generatedTokens: [Int] = []

        for _ in 0..<maxTokens {
            let nextToken = ctx.sampleNextToken(temperature: temperature)

            if ctx.isEOS(nextToken) {
                break
            }

            generatedTokens.append(nextToken)

            if verbose {
                print(ctx.decode(nextToken), terminator: "")
            }

            // Step to next token
            ctx = stepGeneration(context: ctx, nextToken: nextToken)
        }

        let endTime = Date()

        if verbose {
            print()
        }

        Memory.clearCache()

        let text = ctx.decode(generatedTokens)
        let totalTime = endTime.timeIntervalSince(startTime)

        return STTOutput(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            promptTokens: promptTokenCount,
            generationTokens: generatedTokens.count,
            totalTokens: promptTokenCount + generatedTokens.count,
            promptTps: Double(promptTokenCount) / totalTime,
            generationTps: Double(generatedTokens.count) / totalTime,
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    /// Generate transcription from audio with streaming.
    public func generateStream(
        audio: MLXArray,
        maxTokens: Int = 128,
        temperature: Float = 0.0,
        topP: Float = 0.95
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            do {
                guard let tokenizer = self.tokenizer else {
                    throw STTError.modelNotInitialized("Tokenizer not loaded")
                }
                
                let startTime = Date()
                
                // Prepare for generation
                let (context, promptTokenCount) = self.prepareGeneration(audio: audio, tokenizer: tokenizer)
                var ctx = context
                
                let prefillEndTime = Date()
                let prefillTime = prefillEndTime.timeIntervalSince(startTime)
                
                let generateStartTime = Date()
                var generatedTokens: [Int] = []
                
                // Generate tokens
                for _ in 0..<maxTokens {
                    let nextToken = ctx.sampleNextToken(temperature: temperature)
                    
                    if ctx.isEOS(nextToken) {
                        break
                    }
                    
                    generatedTokens.append(nextToken)
                    
                    // Emit token
                    let tokenText = ctx.decode(nextToken)
                    continuation.yield(.token(tokenText))
                    
                    // Step to next token
                    ctx = self.stepGeneration(context: ctx, nextToken: nextToken)
                }
                
                let endTime = Date()
                let generateTime = endTime.timeIntervalSince(generateStartTime)
                let totalTime = endTime.timeIntervalSince(startTime)
                
                Memory.clearCache()
                
                // Emit generation info
                let tokensPerSecond = generateTime > 0 ? Double(generatedTokens.count) / generateTime : 0
                let peakMemory = Double(Memory.peakMemory) / 1e9
                let info = STTGenerationInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generatedTokens.count,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: tokensPerSecond,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.info(info))
                
                // Emit final result
                let text = ctx.decode(generatedTokens)
                let output = STTOutput(
                    text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                    promptTokens: promptTokenCount,
                    generationTokens: generatedTokens.count,
                    totalTokens: promptTokenCount + generatedTokens.count,
                    promptTps: Double(promptTokenCount) / prefillTime,
                    generationTps: tokensPerSecond,
                    totalTime: totalTime,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.result(output))
                
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    /// Create KV cache for generation.
    public func makeCache() -> [KVCache] {
        return (0..<config.lmConfig.numHiddenLayers).map { _ in
            KVCacheSimple()
        }
    }

    /// Sanitize weights for loading.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (k, v) in weights {
            var newKey = k

            // Remap adapting layer names: 0 -> fc1, 2 -> fc2
            if newKey.contains("audio_encoder.adapting.0.") {
                newKey = newKey.replacingOccurrences(
                    of: "audio_encoder.adapting.0.",
                    with: "audio_encoder.adapting.fc1."
                )
            } else if newKey.contains("audio_encoder.adapting.2.") {
                newKey = newKey.replacingOccurrences(
                    of: "audio_encoder.adapting.2.",
                    with: "audio_encoder.adapting.fc2."
                )
            }

            // Remap model.* -> language_model.model.* for LanguageModel wrapper
            if newKey.hasPrefix("model.") {
                newKey = "language_model." + newKey
            }

            // Remap lm_head.* -> language_model.lm_head.*
            if newKey.hasPrefix("lm_head.") {
                newKey = "language_model." + newKey
            }

            // Handle conv weight transposition
            if newKey.contains("conv") && newKey.contains("weight") {
                if v.ndim == 3 && v.shape[2] < v.shape[1] {
                    sanitized[newKey] = v.transposed(0, 2, 1)
                } else {
                    sanitized[newKey] = v
                }
            } else {
                sanitized[newKey] = v
            }
        }

        return sanitized
    }

    /// Load model from pretrained weights.
    public static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> GLMASRModel {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "GLMASRModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        // Load config
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(GLMASRModelConfig.self, from: configData)

        // Get per-layer quantization
        let perLayerQuantization = config.perLayerQuantization

        // Create model
        let model = GLMASRModel(config: config)

        // Load tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        // Load weights
        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        // Sanitize and load weights
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed
        if perLayerQuantization != nil {
            print("Applying quantization from config...")

            if let perLayerQuant = perLayerQuantization {
                print(" Per-layer: \(perLayerQuant)")
            }

            quantize(model: model) { path, module in
                // Convert model path back to original weight path for scales check
                var origPath = path
                if origPath.hasPrefix("language_model.model.") {
                    origPath = String(origPath.dropFirst("language_model.".count))
                } else if origPath.hasPrefix("language_model.lm_head") {
                    origPath = String(origPath.dropFirst("language_model.".count))
                }

                // Check if scales exist for this layer in sanitized weights
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: origPath)?.asTuple
                } else {
                    return nil
                }
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])

        eval(model)

        return model
    }

    // MARK: - Private Helpers

    /// Merge pre-computed audio embeddings into text embeddings.
    private func mergeAudioTextEmbeddings(
        inputIds: MLXArray,
        audioEmbeds: MLXArray?,
        audioOffsets: [[Int]]?,
        audioLength: [[Int]]?,
        cache: [KVCache]?
    ) -> MLXArray {
        let textEmbeds = getInputEmbeddings()(inputIds)

        // Skip if no audio or cache already populated
        guard let audioEmbeds = audioEmbeds else { return textEmbeds }
        if let cache = cache, let firstCache = cache.first as? KVCacheSimple, firstCache.offset > 0 {
            return textEmbeds
        }

        let batchSize = textEmbeds.shape[0]

        for b in 0..<batchSize {
            guard let offsets = audioOffsets, offsets.count > b else { continue }
            let offsetList = offsets[b]
            let lengths = audioLength?[b] ?? [audioEmbeds.shape[1]]

            var audioIdx = 0
            for (offset, length) in zip(offsetList, lengths) {
                if audioIdx < audioEmbeds.shape[0] {
                    let audioChunk = audioEmbeds[audioIdx, 0..<length]
                    let endPos = min(offset + length, textEmbeds.shape[1])
                    let actualLength = endPos - offset

                    for i in 0..<actualLength {
                        textEmbeds[b, offset + i] = audioChunk[i]
                    }
                    audioIdx += 1
                }
            }
        }

        return textEmbeds
    }

    /// Prepare generation context with audio encoding and prompt setup.
    private func prepareGeneration(audio: MLXArray, tokenizer: Tokenizer) -> (GenerationContext, Int) {
        // Preprocess audio to mel spectrogram
        let mel = preprocessAudio(audio)

        // Encode audio once
        let (audioEmbeds, audioLen) = audioEncoder(mel)
        eval(audioEmbeds)

        // Build prompt tokens
        var tokens = tokenizer.encode(text: PromptTemplate.userPrefix)
        tokens.append(contentsOf: Array(repeating: 0, count: audioLen))
        tokens.append(contentsOf: tokenizer.encode(text: PromptTemplate.userSuffix))

        let inputIds = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptTokenCount = inputIds.shape[1]

        let audioStart = tokenizer.encode(text: PromptTemplate.userPrefix).count
        let audioOffsets = [[audioStart]]
        let audioLength = [[audioLen]]

        // Create cache and run initial forward pass
        let cache = makeCache()
        let logits = self(
            inputIds: inputIds,
            audioEmbeds: audioEmbeds,
            audioOffsets: audioOffsets,
            audioLength: audioLength,
            cache: cache
        )

        let context = GenerationContext(
            tokenizer: tokenizer,
            cache: cache,
            eosTokenIds: config.lmConfig.eosTokenId,
            logits: logits
        )

        return (context, promptTokenCount)
    }

    /// Step generation forward with a new token.
    private func stepGeneration(context: GenerationContext, nextToken: Int) -> GenerationContext {
        let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
        let logits = languageModel(inputs: nextTokenArray, cache: context.cache)
        eval(logits)

        return GenerationContext(
            tokenizer: context.tokenizer,
            cache: context.cache,
            eosTokenIds: context.eosTokenIds,
            logits: logits
        )
    }

}
