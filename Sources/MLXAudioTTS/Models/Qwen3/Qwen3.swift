//
//  Qwen3.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/2025.
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

// MARK: - Type Aliases (using shared types from MLXAudioCore)

public typealias Qwen3Error = AudioGenerationError
public typealias Qwen3GenerationInfo = AudioGenerationInfo
public typealias Qwen3Generation = AudioGeneration

// MARK: - Decode

/// Decode audio codes in chunks to reduce memory spikes.
/// Uses Float array accumulation instead of MLXArray concatenation for efficiency.
///
/// - Parameters:
///   - codeList: Flat list of audio codes (7 codes per group)
///   - snacModel: The SNAC decoder model
///   - chunkSize: Number of code groups to decode at once (default 128)
/// - Returns: Decoded audio as MLXArray
func decodeAudioFromCodes(codeList: [Int], snacModel: SNAC, chunkSize: Int = 50) -> MLXArray {
    let numGroups = (codeList.count + 1) / 7

    // For small inputs, decode all at once
    if numGroups <= chunkSize {
        let result = decodeAudioChunk(codeList: codeList, snacModel: snacModel)
        eval(result)
        return result
    }

    // Pre-allocate output buffer (441 samples per group at 24kHz)
    let estimatedSamples = numGroups * snacModel.hopLength
    var audioSamples = ContiguousArray<Float>()
    audioSamples.reserveCapacity(estimatedSamples)

    // Decode in large chunk
    var groupStart = 0
    while groupStart < numGroups {
        let groupEnd = min(groupStart + chunkSize, numGroups)
        let codeStart = groupStart * 7
        let codeEnd = groupEnd * 7

        let chunkCodes = Array(codeList[codeStart..<min(codeEnd, codeList.count)])

        let audioChunk = decodeAudioChunk(codeList: chunkCodes, snacModel: snacModel)
        eval(audioChunk)

        audioSamples.append(contentsOf: audioChunk.asArray(Float.self))

        // Clear GPU memory after each chunk
        Memory.clearCache()

        groupStart = groupEnd
    }


    return MLXArray(Array(audioSamples))
}

/// Decode a single chunk of audio codes (internal helper)
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

    // SNAC decode returns [batch, channels, samples] - squeeze batch and channel dims
    let audioHat = snacModel.decode(codes).squeezed()
    return audioHat
}

func encodeAudioToCodes(audio: MLXArray, snacModel: SNAC) -> MLXArray {
    // Add batch and channel dimensions: [samples] -> [1, 1, samples]
    let audioExpanded = audio
        .expandedDimensions(axis: 0)
        .expandedDimensions(axis: 0)

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

public class Attention: Module {
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

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
        let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a Float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta, scale: ropeScale
        )


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
            // Update cache and get full key/value history
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

        return wo(
            output
        )
    }

}


// MARK: - MLP

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}


private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }


}


private class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers)
            .map { _ in TransformerBlock(args) }

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

    private let model: Qwen3ModelInner

    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // KVCacheDimensionProvider conformance
    public var numLayers: Int {
        return self.configuration.hiddenLayers
    }

    public init(_ args: Qwen3Configuration){
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0..<args.hiddenLayers).map {_ in args.kvHeads}
        self.model = Qwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    /// Parse a single row of tokens on CPU
    public func parseOutputRow(_ tokens: [Int32]) -> [Int] {
        let tokensI = tokens.map(Int.init)

        // Find start index: first try START_OF_SPEECH (use lastIndex for latest occurrence)
        var startIdx: Int? = tokensI.lastIndex(of: startOfSpeech)

        // If not found, look for START_OF_AI and find first audio token after it
        if startIdx == nil, let soaIdx = tokensI.lastIndex(of: startOfAI) {
            if let firstAudio = tokensI[(soaIdx + 1)...].firstIndex(where: { $0 >= audioTokensStart }) {
                startIdx = firstAudio - 1
            }
        }

        // Slice from start index
        let slice = (startIdx != nil) ? Array(tokensI[(startIdx! + 1)...]) : tokensI

        // Filter out END_OF_SPEECH tokens
        let filtered = slice.filter { $0 != endOfSpeech }

        // Trim to multiple of 7
        let newLen = (filtered.count / 7) * 7
        guard newLen > 0 else { return [] }

        // Subtract audioTokensStart
        return filtered.prefix(newLen).map { $0 - audioTokensStart }
    }

    /// Parse output tokens for multiple batch rows using CPU-based parsing.
    public func parseOutputBatch(_ tokensPerBatch: [[Int32]]) -> [[Int]] {
        return tokensPerBatch.map { parseOutputRow($0) }
    }

    /// Legacy parseOutput for MLXArray input (kept for compatibility).
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

        // Apply voice prefix if provided
        let modifiedPrompts: [String]
        if let voice = voice {
            modifiedPrompts = prompts.map { "\(voice): \($0)" }
        } else {
            modifiedPrompts = prompts
        }

        // Encode all prompts to CPU arrays first (avoid multiple MLXArray creations)
        let encodedPrompts: [[Int32]] = modifiedPrompts.map { prompt in
            tokenizer!.encode(text: prompt).map { Int32($0) }
        }

        // Find max length for padding
        let maxPromptLen = encodedPrompts.map { $0.count }.max() ?? 0

        // Build all input IDs on CPU, then create single MLXArray
        var allBatchIds: [[Int32]] = []

        for encodedPrompt in encodedPrompts {
            var sequence: [Int32] = []

            // Add padding if needed (at the start)
            let paddingLen = maxPromptLen - encodedPrompt.count
            if paddingLen > 0 {
                sequence.append(contentsOf: [Int32](repeating: Int32(padTokenId), count: paddingLen))
            }

            // Add reference audio and transcript if provided
            if let refCodes = refAudioCodes, let refTranscript = refTranscriptIds {
                // [START_OF_HUMAN] + transcript + [END_OF_TEXT, END_OF_HUMAN]
                sequence.append(Int32(startOfHuman))
                sequence.append(contentsOf: refTranscript)
                sequence.append(Int32(endOfText))
                sequence.append(Int32(endOfHuman))
                // [START_OF_AI, START_OF_SPEECH] + audio codes + [END_OF_SPEECH, END_OF_AI]
                sequence.append(Int32(startOfAI))
                sequence.append(Int32(startOfSpeech))
                sequence.append(contentsOf: refCodes)
                sequence.append(Int32(endOfSpeech))
                sequence.append(Int32(endOfAI))
            }

            // Add prompt: [START_OF_HUMAN] + prompt + [END_OF_TEXT, END_OF_HUMAN]
            sequence.append(Int32(startOfHuman))
            sequence.append(contentsOf: encodedPrompt)
            sequence.append(Int32(endOfText))
            sequence.append(Int32(endOfHuman))

            allBatchIds.append(sequence)
        }

        // Pad all sequences to same length and create single MLXArray
        let maxLen = allBatchIds.map { $0.count }.max() ?? 0
        var flattenedIds: [Int32] = []
        flattenedIds.reserveCapacity(allBatchIds.count * maxLen)

        for var sequence in allBatchIds {
            // Pad to maxLen if needed
            let padCount = maxLen - sequence.count
            if padCount > 0 {
                sequence.insert(contentsOf: [Int32](repeating: Int32(padTokenId), count: padCount), at: 0)
            }
            flattenedIds.append(contentsOf: sequence)
        }

        // Single MLXArray creation from CPU data
        let batchSize = allBatchIds.count
        let finalBatchInputIds = MLXArray(flattenedIds).reshaped([batchSize, maxLen])

        // Create attention mask (1 for real tokens, 0 for pad tokens)
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

    public func post_load_hook(model: Qwen3Model, modelDir: URL) async throws {
        if model.tokenizer == nil {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        }
        if model._snacModel == nil {
            model._snacModel = try await SNAC.fromPretrained("mlx-community/snac_24khz")
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

    // MARK: - Generation using MLXLMCommon evaluate pattern

    /// Generate audio from text using MLXLMCommon's evaluate-style token generation.
    ///
    /// This follows the pattern from MLXLMCommon's `generate` function with:
    /// - Configurable sampling (temperature, top-p)
    /// - Repetition penalty via LogitProcessor
    /// - KV cache for efficient generation
    ///
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - voice: Optional voice identifier (e.g., "en-us-1")
    ///   - refAudio: Optional reference audio for voice cloning
    ///   - refText: Optional transcription of the reference audio
    ///   - parameters: Generation parameters (temperature, topP, maxTokens, etc.)
    /// - Returns: Generated audio as MLXArray
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

        // Prepare input
        let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        let (inputIds, _) = prepareInputIds(
            prompts: [prompt],
            voice: voice,
            refAudio: refAudio,
            refText: refText
        )

        // Create sampler and processor from parameters
        let sampler = parameters.sampler()
        var processor = parameters.processor()

        // Initialize prompt tokens for processor
        let promptTokens = inputIds.squeezed(axis: 0)
        processor?.prompt(promptTokens)

        // Create KV cache
        var cache = cache
        if cache == nil {
            cache = self.makeCache()
        }

        let maxTokens = parameters.maxTokens ?? 1200

        let promptTokensList = inputIds.squeezed(axis: 0).asArray(Int32.self)

        var generatedOnly = ContiguousArray<Int32>()
        generatedOnly.reserveCapacity(maxTokens)

        // Prefill: process the prompt, slice immediately to [1, V]
        var logits = self(inputIds, cache: cache)
        logits = logits[0..., -1, 0...]  // [1, V] - avoid keeping [1, L, V]
        eval(logits)

        // Generate tokens
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
                    logits = logits[0..., -1, 0...]  // [1, V]
                    eval(logits)
                }

                return value
            }

            if tokenValue == endOfSpeech {
                break
            }

            // Only store generated tokens (not prompt)
            generatedOnly.append(Int32(tokenValue))
        }

        Memory.clearCache()

        // Reconstruct full tokens only once at the end for parsing
        var fullTokens = ContiguousArray<Int32>()
        fullTokens.reserveCapacity(promptTokensList.count + generatedOnly.count)
        fullTokens.append(contentsOf: promptTokensList)
        fullTokens.append(contentsOf: generatedOnly)

        // Parse output to audio codes using CPU-based parsing
        let codeList = parseOutputRow(Array(fullTokens))

        guard !codeList.isEmpty else {
            throw Qwen3Error.generationFailed("No audio codes generated")
        }

        // Decode audio using SNAC
        let audio = decodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
        audio.eval()

        // Clear SNAC decoder intermediates
        Memory.clearCache()

        return audio
    }

    /// Generate audio with streaming token output.
    ///
    /// Returns an AsyncThrowingStream that yields generation events including tokens and final audio.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - voice: Optional voice name/style identifier
    ///   - refAudio: Optional reference audio for voice cloning
    ///   - refText: Optional transcription of the reference audio
    ///   - parameters: Generation parameters
    /// - Returns: AsyncThrowingStream of Qwen3Generation events
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
                
                // DEDUP: Pull prompt tokens ONCE to CPU - this is your anchor
                let promptTokensList = inputIds.squeezed(axis: 0).asArray(Int32.self)
                
                // Store only generated tokens (not prompt tokens) - dedup approach
                var generatedTokens = ContiguousArray<Int32>()
                generatedTokens.reserveCapacity(maxTokens)
                
                let startTime = Date()
                
                // Prefill: process the prompt, slice immediately to [1, V]
                var tokenCount: Int = 0
                var logits = self(inputIds, cache: cache)
                logits = logits[0..., -1, 0...]  // [1, V] - avoid keeping [1, L, V]
                eval(logits)
                let prefillTime = Date().timeIntervalSince(startTime)
                
                let generateStartTime = Date()
                
                // Generate tokens
                for _ in 0..<maxTokens {
                    if Task.isCancelled { break }
                    
                    // Extract token value and advance - minimize intermediate tensor lifetime
                    let tokenValue: Int = autoreleasepool {
                        var lastLogits = logits
                        lastLogits = processor?.process(logits: lastLogits) ?? lastLogits
                        
                        let nextToken = sampler.sample(logits: lastLogits)
                        processor?.didSample(token: nextToken)
                        
                        let value = nextToken.item(Int.self)
                        
                        // Forward pass with cache
                        if value != endOfSpeech {
                            let nextTokenExpanded = nextToken.reshaped([1, 1])
                            logits = self(nextTokenExpanded, cache: cache)
                            logits = logits[0..., -1, 0...]  // [1, V]
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
                
                // Reconstruct full tokens only once at the end for parsing
                var fullTokens = ContiguousArray<Int32>()
                fullTokens.reserveCapacity(promptTokensList.count + generatedTokens.count)
                fullTokens.append(contentsOf: promptTokensList)
                fullTokens.append(contentsOf: generatedTokens)
                
                // Parse output to audio codes using CPU-based parsing
                let codeList = self.parseOutputRow(Array(fullTokens))
                
                guard !codeList.isEmpty else {
                    throw Qwen3Error.generationFailed("No audio codes generated")
                }
                
                let audio = decodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
                audio.eval()
                
                Memory.clearCache()
                
                // Yield completion info
                let info = Qwen3GenerationInfo(
                    promptTokenCount: inputIds.shape[1],
                    generationTokenCount: tokenCount,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: Double(tokenCount) / generateTime,
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9
                )
                continuation.yield(.info(info))
                
                // Yield final audio
                continuation.yield(.audio(audio))
                
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3Model {
        // Check for HF token in environment (macOS) or Info.plist (iOS)
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            client = HubClient.default
        }
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "Qwen3Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        // Check if model is already fully cached (has weight files)
        let modelDir = try await resolveOrDownloadModel(
            client: client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors"
        )


        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: configData)

        let perLayerQuantization = config.perLayerQuantization

        let model = Qwen3Model(config)


        // Load weights from safetensors
        let weights = try loadWeights(from: modelDir)

        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed

        if perLayerQuantization != nil {
            print("Applying quantizaiton from config...")

            if let perLayerQuant = perLayerQuantization {
                print(" Per-layer: \(perLayerQuant)")
            }

            quantize(model: model) { path, module in
                // Only quantize if scales exist for this layer
                if weights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: path)?.asTuple
                } else {
                    return nil
                }
            }
        }



        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])
        eval(model)

        try await model.post_load_hook(model: model, modelDir: modelDir)

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

/// Resolves a model from cache or downloads it if not cached.
/// - Parameters:
///   - client: The HuggingFace Hub client
///   - cache: The HuggingFace cache
///   - repoID: The repository ID
///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
/// - Returns: The model directory URL
func resolveOrDownloadModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID,
    requiredExtension: String
) async throws -> URL {
    // Use a persistent cache directory based on repo ID
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    // Check if model already exists with required files
    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasRequiredFiles = files?.contains { $0.pathExtension == requiredExtension } ?? false

        if hasRequiredFiles {
            // Validate that config.json is valid JSON
            let configPath = modelDir.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configPath.path) {
                if let configData = try? Data(contentsOf: configPath),
                   let _ = try? JSONSerialization.jsonObject(with: configData) {
                    print("Using cached model at: \(modelDir.path)")
                    return modelDir
                } else {
                    print("Cached config.json is invalid, clearing cache...")
                    try? FileManager.default.removeItem(at: modelDir)
                }
            }
        }
    }

    // Create directory if needed
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    print("Downloading model \(repoID)...")
    _ = try await client.downloadSnapshot(
        of: repoID,
        kind: .model,
        to: modelDir,
        revision: "main",
        progressHandler: { progress in
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        }
    )

    print("Model downloaded to: \(modelDir.path)")
    return modelDir
}
