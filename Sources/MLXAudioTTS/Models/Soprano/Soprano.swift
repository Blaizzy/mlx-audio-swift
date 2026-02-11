//
//  Soprano.swift
//  MLXAudio
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation
@preconcurrency import MLX
import HuggingFace
import Tokenizers
import MLXLMCommon
import MLXNN
import MLXAudioCore

// MARK: - Type Aliases

public typealias SopranoError = AudioGenerationError
public typealias SopranoGenerationInfo = AudioGenerationInfo
public typealias SopranoGeneration = AudioGeneration

// MARK: - Soprano Attention

private class SopranoAttention: Module {
    let args: SopranoConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: SopranoConfiguration) {
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

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: args.ropeTheta,
            scale: 1.0
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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

// MARK: - MLP

private class SopranoMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

private class SopranoTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: SopranoAttention
    let mlp: SopranoMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: SopranoConfiguration) {
        self._attention.wrappedValue = SopranoAttention(args)
        self.mlp = SopranoMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Inner Model

private class SopranoModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [SopranoTransformerBlock]
    let norm: RMSNorm

    init(_ args: SopranoConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers).map { _ in
            SopranoTransformerBlock(args)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Soprano Model

public class SopranoModel: Module, KVCacheDimensionProvider, SpeechGenerationModel, @unchecked Sendable {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var tokenizer: Tokenizer?

    private let model: SopranoModelInner
    let configuration: SopranoConfiguration
    let decoder: SopranoDecoder

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // Token IDs
    private var stopTokenId: Int?

    // KVCacheDimensionProvider conformance
    public var numLayers: Int {
        return configuration.hiddenLayers
    }

    public init(_ config: SopranoConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0..<config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SopranoModelInner(config)

        // Initialize decoder
        self.decoder = SopranoDecoder(
            numInputChannels: config.hiddenSize,
            decoderNumLayers: config.decoderNumLayers,
            decoderDim: config.decoderDim,
            decoderIntermediateDim: config.decoderIntermediateDim,
            hopLength: config.hopLength,
            nFft: config.nFft,
            upscale: config.upscale,
            inputKernel: config.inputKernel,
            dwKernel: config.dwKernel
        )

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public var sampleRate: Int {
        return configuration.sampleRate
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)

        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    /// Forward pass that returns both logits and hidden states.
    func forwardWithHiddenStates(_ inputs: MLXArray, cache: [KVCache]? = nil) -> (logits: MLXArray, hiddenStates: MLXArray) {
        var h = model.embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in model.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        // Get hidden states before lm_head
        let hiddenStates = model.norm(h)

        // Compute logits
        let logits: MLXArray
        if let lmHead = lmHead {
            logits = lmHead(hiddenStates)
        } else {
            logits = model.embedTokens.asLinear(hiddenStates)
        }

        return (logits, hiddenStates)
    }

    public func makeCache() -> [KVCache] {
        return (0..<configuration.hiddenLayers).map { _ in
            KVCacheSimple()
        }
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio _: MLXArray?,
        refText _: String?,
        language _: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try await generate(
            text: text,
            voice: voice,
            splitPattern: "\n",
            parameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio _: MLXArray?,
        refText _: String?,
        language _: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            parameters: generationParameters
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Remove "model." prefix if present (e.g. "model.language_model.*" → "language_model.*")
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }

            // Decoder weights should be float32
            var newValue = value
            if newKey.hasPrefix("decoder.") {
                newValue = value.asType(.float32)
            } else if newKey.hasPrefix("language_model.lm_head") {
                // lm_head is directly on SopranoModel, not inside model
                newKey = newKey.replacingOccurrences(of: "language_model.", with: "")
            } else if newKey.hasPrefix("language_model.") {
                // Other language_model weights go to model.* (SopranoModelInner)
                newKey = newKey.replacingOccurrences(of: "language_model.", with: "model.")
            } else if !newKey.hasPrefix("lm_head") {
                // Bare inner model keys (e.g. "embed_tokens.*", "layers.*", "norm.*")
                // need "model." prefix to map to SopranoModelInner
                newKey = "model." + newKey
            }

            sanitized[newKey] = newValue
        }

        // Remove lm_head if tying embeddings
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized
    }

    // MARK: - Text Preprocessing

    private func preprocessText(_ texts: [String], minLength: Int = 30) -> [(prompt: String, textIdx: Int, sentenceIdx: Int)] {
        var results: [(String, Int, Int)] = []

        for (textIdx, text) in texts.enumerated() {
            let trimmedText = text.trimmingCharacters(in: .whitespaces)
            let cleanedText = cleanTextForSoprano(trimmedText)

            // Split at sentence boundaries (matching Python: re.split(r"(?<=[.!?])\s+", text))
            // This regex splits after .!? followed by whitespace
            let sentences = splitIntoSentences(cleanedText)

            // Process sentences, merging short ones
            var processed: [(text: String, textIdx: Int)] = sentences.map { (text: $0, textIdx: textIdx) }

            if minLength > 0 && processed.count > 1 {
                var merged: [(text: String, textIdx: Int)] = []
                var i = 0
                while i < processed.count {
                    let cur = processed[i]
                    if cur.text.count < minLength {
                        if !merged.isEmpty {
                            // Merge with previous
                            let prev = merged.removeLast()
                            merged.append((text: (prev.text + " " + cur.text).trimmingCharacters(in: .whitespaces), textIdx: prev.textIdx))
                        } else if i + 1 < processed.count {
                            // Merge with next
                            processed[i + 1] = (text: (cur.text + " " + processed[i + 1].text).trimmingCharacters(in: .whitespaces), textIdx: processed[i + 1].textIdx)
                        } else {
                            merged.append(cur)
                        }
                    } else {
                        merged.append(cur)
                    }
                    i += 1
                }
                processed = merged
            }

            // Create prompts for each sentence
            for (sentenceIdx, item) in processed.enumerated() {
                let prompt = "[STOP][TEXT]\(item.text)[START]"
                results.append((prompt, item.textIdx, sentenceIdx))
            }
        }

        return results
    }

    /// Split text into sentences at .!? followed by whitespace
    private func splitIntoSentences(_ text: String) -> [String] {
        // Match Python: re.split(r"(?<=[.!?])\s+", text)
        // Split after .!? when followed by whitespace
        guard let regex = try? NSRegularExpression(pattern: "(?<=[.!?])\\s+", options: []) else {
            return [text]
        }

        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        let matches = regex.matches(in: text, options: [], range: range)

        if matches.isEmpty {
            return [text]
        }

        var sentences: [String] = []
        var lastEnd = 0

        for match in matches {
            let sentenceRange = NSRange(location: lastEnd, length: match.range.location - lastEnd)
            let sentence = nsText.substring(with: sentenceRange)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            lastEnd = match.range.location + match.range.length
        }

        // Add remaining text after last match
        if lastEnd < nsText.length {
            let remaining = nsText.substring(from: lastEnd)
            if !remaining.isEmpty {
                sentences.append(remaining)
            }
        }

        return sentences
    }
    /// Space token ID in Soprano vocabulary
    private let spaceTokenId = 8004

    /// Special tokens that should be preserved as-is
    private let specialTokenPattern = #"\[(STOP|TEXT|START)\]"#

    /// Pre-tokenize pattern matching Soprano's tokenizer config
    /// Pattern: `\s+|\w+|[^\w\s]+` with behavior "Isolated"
    private let preTokenizePattern = #"\s+|\w+|[^\w\s]+"#

    private func tokenize(_ text: String) -> MLXArray {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not initialized")
        }

        // Split text into special tokens and regular segments
        // Special tokens like [STOP], [TEXT], [START] need to be encoded as-is
        let segments = splitBySpecialTokens(text)

        var allTokens: [Int] = []

        for segment in segments {
            if isSpecialToken(segment) {
                // Special tokens are handled directly by the tokenizer
                let tokens = tokenizer.encode(text: segment, addSpecialTokens: false)
                allTokens.append(contentsOf: tokens)
            } else {
                // For regular text, pre-tokenize to handle spaces correctly
                // swift-transformers has a bug where BPETokenizer.tokenize() drops space tokens
                // because " ".split(separator: " ") returns empty array
                let chunks = preTokenizeText(segment)

                for chunk in chunks {
                    if chunk.allSatisfy({ $0.isWhitespace }) {
                        // For each space character, add a space token
                        for _ in chunk {
                            allTokens.append(spaceTokenId)
                        }
                    } else {
                        // Use tokenizer for non-whitespace chunks
                        let chunkTokens = tokenizer.encode(text: chunk, addSpecialTokens: false)
                        allTokens.append(contentsOf: chunkTokens)
                    }
                }
            }
        }


        return MLXArray(allTokens.map { Int32($0) })
    }

    /// Check if a string is a special token
    private func isSpecialToken(_ text: String) -> Bool {
        return text == "[STOP]" || text == "[TEXT]" || text == "[START]"
    }

    /// Split text into special tokens and regular segments
    private func splitBySpecialTokens(_ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: specialTokenPattern, options: []) else {
            return [text]
        }

        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        let matches = regex.matches(in: text, options: [], range: range)

        if matches.isEmpty {
            return [text]
        }

        var segments: [String] = []
        var lastEnd = 0

        for match in matches {
            // Add text before this special token
            if match.range.location > lastEnd {
                let beforeRange = NSRange(location: lastEnd, length: match.range.location - lastEnd)
                let before = nsText.substring(with: beforeRange)
                if !before.isEmpty {
                    segments.append(before)
                }
            }

            // Add the special token itself
            let specialToken = nsText.substring(with: match.range)
            segments.append(specialToken)

            lastEnd = match.range.location + match.range.length
        }

        // Add remaining text after the last special token
        if lastEnd < nsText.length {
            let remaining = nsText.substring(from: lastEnd)
            if !remaining.isEmpty {
                segments.append(remaining)
            }
        }

        return segments
    }

    /// Pre-tokenize text using Soprano's tokenizer pattern
    /// Splits into chunks: words, whitespace runs, and punctuation
    private func preTokenizeText(_ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: preTokenizePattern, options: []) else {
            return [text]
        }

        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        let matches = regex.matches(in: text, options: [], range: range)

        return matches.map { match in
            nsText.substring(with: match.range)
        }
    }

    // MARK: - Generation

    /// Generate audio from text.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - voice: Voice name (unused in base Soprano)
    ///   - parameters: Generation parameters
    /// - Returns: Generated audio as MLXArray
    public func generate(
        text: String,
        voice: String? = nil,
        splitPattern: String = "\n",  // Add split pattern parameter
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 1200,
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.5,
            repetitionContextSize: 30
        )
    ) async throws -> MLXArray {
        guard self.tokenizer != nil else {
            throw SopranoError.modelNotInitialized("Tokenizer not loaded")
        }

        // Process escape sequences and split by pattern
        let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        // Split text by pattern, then further split long chunks at sentence boundaries
        let prompts = prompt.components(separatedBy: splitPattern)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .flatMap { chunk -> [String] in
                guard chunk.count > 500 else { return [chunk] }

                // Split long chunks at sentence boundaries
                let boundaries = CharacterSet(charactersIn: ".?!:;")
                var result: [String] = []
                var current = ""

                for char in chunk {
                    current.append(char)
                    let atBoundary = char.unicodeScalars.first.map { boundaries.contains($0) } ?? false

                    if (atBoundary && current.count >= 100) || current.count >= 500 {
                        result.append(current.trimmingCharacters(in: .whitespaces))
                        current = ""
                    }
                }

                if !current.isEmpty {
                    result.append(current.trimmingCharacters(in: .whitespaces))
                }

                return result.filter { !$0.isEmpty }
            }

        var audioParts: [MLXArray] = []
        var totalTokens = 0
        let maxTokens = parameters.maxTokens ?? 512

        // Process each chunk separately
        for promptChunk in prompts {
            let sentenceData = self.preprocessText([promptChunk])


            for (promptText, _, _) in sentenceData {
                let inputIds = self.tokenize(promptText)
                var allHiddenStates: [MLXArray] = []

                for await (token, hiddenState) in self.streamGenerate(
                    inputIds: inputIds,
                    maxTokens: maxTokens,
                    temperature: parameters.temperature,
                    topP: parameters.topP,
                    repetitionPenalty: parameters.repetitionPenalty ?? 1.0,  // Match Python (no penalty)
                    repetitionContextSize: parameters.repetitionContextSize
                ) {
                    allHiddenStates.append(hiddenState)

                    if token != nil {
                        totalTokens += 1
                    }
                }

                let tokenCount = allHiddenStates.count

                // Stack hidden states
                let hiddenStates = MLX.concatenated(allHiddenStates, axis: 1)

                // Decode to audio
                var audio = self.decoder(hiddenStates)

                let tokenSize = self.configuration.tokenSize
                let audioLength = tokenCount * tokenSize - tokenSize

                if audioLength > 0 {
                    audio = audio[0, (-audioLength)...]
                } else {
                    audio = audio.squeezed(axis: 0)
                }

                audioParts.append(audio)
            }
        }

        // Concatenate all audio parts
        let finalAudio: MLXArray
        if audioParts.count > 1 {
            finalAudio = MLX.concatenated(audioParts, axis: 0)
        } else if audioParts.count == 1 {
            finalAudio = audioParts[0]
        } else {
            throw SopranoError.generationFailed("No audio generated")
        }

        return finalAudio
    }

    /// Generate audio with streaming events.
    public func generateStream(
        text: String,
        voice: String? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 512,
            temperature: 0.3,
            topP: 0.95,
            repetitionPenalty: 1.5,
            repetitionContextSize: 30
        )
    ) -> AsyncThrowingStream<SopranoGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<SopranoGeneration, Error>.makeStream()
        Task { @Sendable [weak self, continuation] in
            guard let self else { return }
            
            do {
                guard self.tokenizer != nil else {
                    throw SopranoError.modelNotInitialized("Tokenizer not loaded")
                }
                
                let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
                    .replacingOccurrences(of: "\\t", with: "\t")
                
                let startTime = Date()
                let sentenceData = self.preprocessText([prompt])
                
                var audioParts: [MLXArray] = []
                var totalTokens = 0
                let maxTokens = parameters.maxTokens ?? 512
                
                for (promptText, _, _) in sentenceData {
                    let inputIds = self.tokenize(promptText)
                    var allHiddenStates: [MLXArray] = []
                    
                    for await (token, hiddenState) in self.streamGenerate(
                        inputIds: inputIds,
                        maxTokens: maxTokens,
                        temperature: parameters.temperature,
                        topP: parameters.topP,
                        repetitionPenalty: parameters.repetitionPenalty ?? 1.0,  // Match Python (no penalty)
                        repetitionContextSize: parameters.repetitionContextSize
                    ) {
                        allHiddenStates.append(hiddenState)
                        
                        if let tokenVal = token {
                            continuation.yield(.token(tokenVal))
                        }
                    }
                    
                    let tokenCount = allHiddenStates.count
                    totalTokens += tokenCount
                    
                    // Stack hidden states
                    let hiddenStates = MLX.concatenated(allHiddenStates, axis: 1)
                    
                    // Decode to audio
                    var audio = self.decoder(hiddenStates)
                    
                    let tokenSize = self.configuration.tokenSize
                    let audioLength = tokenCount * tokenSize - tokenSize
                    
                    if audioLength > 0 {
                        audio = audio[0, (-audioLength)...]
                    } else {
                        audio = audio[0]
                    }
                    
                    audioParts.append(audio)
                }
                
                // Concatenate audio
                let finalAudio: MLXArray
                if audioParts.count > 1 {
                    finalAudio = MLX.concatenated(audioParts, axis: 0)
                } else {
                    finalAudio = audioParts[0]
                }
                
                let elapsed = Date().timeIntervalSince(startTime)
                
                // Yield info
                let info = SopranoGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: totalTokens,
                    prefillTime: 0,
                    generateTime: elapsed,
                    tokensPerSecond: Double(totalTokens) / elapsed,
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9
                )
                continuation.yield(.info(info))
                
                // Yield audio
                continuation.yield(.audio(finalAudio))
                
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    /// Stream generate tokens and hidden states.
    private func streamGenerate(
        inputIds: MLXArray,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float = 1.5,
        repetitionContextSize: Int = 30
    ) -> AsyncStream<(Int?, MLXArray)> {
        AsyncStream { continuation in
            Task {
                var ids = inputIds
                if ids.ndim == 1 {
                    ids = ids.expandedDimensions(axis: 0)
                }

                // Create KV cache
                let cache = self.makeCache()

                // Prefill
                let (logits, hiddenStates) = self.forwardWithHiddenStates(ids, cache: cache)
                eval(logits, hiddenStates)

                // Yield last hidden state from prefill (last position along sequence dim)
                let lastHiddenState = hiddenStates[0..., (hiddenStates.shape[1] - 1)..<hiddenStates.shape[1], 0...]
                continuation.yield((nil, lastHiddenState))

                // Create sampler
                let sampler = TopPSampler(temperature: temperature, topP: topP)

                // Track generated tokens for repetition penalty
                var generatedTokens: [Int] = []

                // Generate tokens
                var currentLogits = logits

                for _ in 0..<maxTokens {
                    // Get last logits
                    var lastLogits = currentLogits[0..., -1, 0...]
                    eval(lastLogits)

                    // Apply repetition penalty
                    if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
                        let contextTokens = Array(generatedTokens.suffix(repetitionContextSize))
                        lastLogits = applyRepetitionPenalty(
                            logits: lastLogits,
                            tokens: contextTokens,
                            penalty: repetitionPenalty
                        )
                    }

                    // Sample next token
                    let nextToken: MLXArray
                    if temperature == 0 {
                        nextToken = argMax(lastLogits, axis: -1, keepDims: true)
                    } else {
                        nextToken = sampler.sample(logits: lastLogits)
                    }

                    let tokenId = nextToken.item(Int.self)

                    // Check for stop token ([STOP] = token ID 3)
                    if tokenId == self.stopTokenId {
                        break
                    }

                    // Track token for repetition penalty
                    generatedTokens.append(tokenId)

                    // Forward pass with new token
                    let nextTokenExpanded = nextToken.reshaped([1, 1])
                    let (newLogits, newHiddenStates) = self.forwardWithHiddenStates(nextTokenExpanded, cache: cache)

                    let newLastHiddenState = newHiddenStates[0..., (newHiddenStates.shape[1] - 1)..<newHiddenStates.shape[1], 0...]
                    eval(newLastHiddenState)
                    continuation.yield((tokenId, newLastHiddenState))

                    currentLogits = newLogits
                }

                continuation.finish()
            }
        }
    }

    /// Apply repetition penalty to logits
    private func applyRepetitionPenalty(logits: MLXArray, tokens: [Int], penalty: Float) -> MLXArray {
        // Convert logits to array, apply penalty, convert back
        var logitsArray = logits.asArray(Float.self)
        for token in tokens {
            if token < logitsArray.count {
                if logitsArray[token] > 0 {
                    logitsArray[token] /= penalty
                } else {
                    logitsArray[token] *= penalty
                }
            }
        }
        return MLXArray(logitsArray)
    }

    // MARK: - Loading

    public static func fromPretrained(_ modelRepo: String) async throws -> SopranoModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            client = HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            client = HubClient.default
        }
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "SopranoModel", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"
            ])
        }

        let modelDir = try await resolveOrDownloadSopranoModel(
            client: client,
            cache: cache,
            repoID: repoID
        )

        // Load config
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        var config = try JSONDecoder().decode(SopranoConfiguration.self, from: configData)

        // Adjust decoder config based on model version (matching Python's __post_init__)
        // Soprano-1.1 uses the defaults (decoder_dim=768, input_kernel=1)
        // Older models use decoder_dim=512, input_kernel=3
        if !modelRepo.lowercased().contains("soprano-1.1") {
            config.decoderDim = 512
            config.decoderIntermediateDim = 1536
            config.inputKernel = 3
        }

        let model = SopranoModel(config)

        // Load weights
        let weights = try loadSopranoWeights(from: modelDir)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Apply quantization if needed
        if let perLayerQuant = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                if weights["\(path).scales"] != nil {
                    return perLayerQuant.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])

        eval(model)

        // Load tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        if model.tokenizer != nil {
            model.stopTokenId = model.tokenizer?.eosTokenId ?? 3
        }

        return model
    }
}

// MARK: - Helper Functions

private func loadSopranoWeights(from directory: URL) throws -> [String: MLXArray] {
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

private func resolveOrDownloadSopranoModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID
) async throws -> URL {
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasWeights = files?.contains { $0.pathExtension == "safetensors" } ?? false

        if hasWeights {
            let configPath = modelDir.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configPath.path),
               let configData = try? Data(contentsOf: configPath),
               (try? JSONSerialization.jsonObject(with: configData)) != nil {
                return modelDir
            }
        }
    }

    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    _ = try await client.downloadSnapshot(
        of: repoID,
        kind: .model,
        to: modelDir,
        revision: "main",
        progressHandler: { progress in
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        }
    )

    return modelDir
}

// MARK: - TopP Sampler (matching Python's mlx_lm implementation)

private struct TopPSampler {
    let temperature: Float
    let topP: Float

    /// Apply top-p filtering to logits (returns logits with -inf for excluded tokens)
    /// Matches Python's apply_top_p from mlx_lm/sample_utils.py
    private func applyTopP(logprobs: MLXArray) -> MLXArray {
        // Convert to probabilities for cumsum calculation
        let probs = exp(logprobs)

        // Sort in ascending order (smallest probs first)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedProbs = take(probs, sortedIndices, axis: -1)

        // Compute cumulative probabilities
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Create inverse indices to map back to original order
        // This replicates Python's put_along_axis approach
        let vocabSize = sortedIndices.shape[0]

        // Create inverse mapping: for each position in original order,
        // find its cumulative probability
        var inverseIndices = [Int32](repeating: 0, count: vocabSize)
        let sortedIndicesArray = sortedIndices.asArray(Int32.self)
        for (sortedPos, originalPos) in sortedIndicesArray.enumerated() {
            inverseIndices[Int(originalPos)] = Int32(sortedPos)
        }
        let inverseIndicesArray = MLXArray(inverseIndices)

        // Rearrange cumulative probs back to original vocabulary order
        let cumulativeProbsOriginalOrder = take(cumulativeProbs, inverseIndicesArray, axis: -1)

        // Keep tokens where cumulative prob > (1 - top_p)
        // These are the TOP tokens that make up the nucleus
        let threshold = MLXArray(1.0 - topP)
        let mask = cumulativeProbsOriginalOrder .> threshold

        // Set excluded tokens to -inf in log space
        let negInf = MLXArray(-Float.infinity)
        return MLX.where(mask, logprobs, negInf)
    }

    func sample(logits: MLXArray) -> MLXArray {
        // Ensure logits is 1D for processing
        let logits1D: MLXArray
        if logits.ndim == 2 {
            logits1D = logits.squeezed(axis: 0)  // [1, vocab] -> [vocab]
        } else {
            logits1D = logits
        }

        // Apply top-p filtering in log space (sets excluded tokens to -inf)
        let filteredLogits = applyTopP(logprobs: logits1D)

        // Sample using categorical with temperature scaling
        // categorical expects log probabilities (logits), not probabilities!
        // This matches Python's: mx.random.categorical(logits * (1 / temp))
        let scaledLogits = filteredLogits / temperature
        let scaledLogits2D = scaledLogits.expandedDimensions(axis: 0)  // [1, vocab]
        let sampledToken = categorical(scaledLogits2D)  // [1]

        return sampledToken.reshaped([1])
    }
}
