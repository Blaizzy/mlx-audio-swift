//
//  LlamaTTS.swift
//  MLXAudio
//
//  Created by Prince Canuma on 02/01/2026.
//

import Foundation
@preconcurrency import MLX
import HuggingFace
import Tokenizers
@preconcurrency import MLXLMCommon
import MLXNN
import MLXAudioCodecs
import MLXAudioCore

// MARK: - Orpheus TTS Special Token IDs

/// Orpheus-based tokenizer special tokens (for Llama-based TTS models)
public enum OrpheusTokens {
    public static let startOfHuman = 128259
    public static let endOfHuman = 128260
    public static let endOfText = 128009
    public static let startOfSpeech = 128257
    public static let endOfSpeech = 128258
    public static let padToken = 128263
    public static let audioStart = 128261
    public static let audioEnd = 128262
    public static let audioTokenOffset = 128266
}

// MARK: - Type Aliases (using shared types from MLXAudioCore)

public typealias LlamaTTSError = AudioGenerationError
public typealias LlamaTTSGenerationInfo = AudioGenerationInfo
public typealias LlamaTTSGeneration = AudioGeneration

// MARK: - SNAC Audio Codec Functions

/// Decode SNAC audio codes to waveform.
func llamaDecodeAudioFromCodes(codeList: [Int], snacModel: SNAC) -> MLXArray {
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

/// Encode audio waveform to SNAC codes.
func llamaEncodeAudioToCodes(audio: MLXArray, snacModel: SNAC) -> MLXArray {
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

// MARK: - Llama3 Scaled RoPE

/// Llama3-style scaled RoPE using MLXFast.RoPE with custom frequencies.
/// This matches the Python mlx-lm Llama3RoPE implementation exactly.
private class Llama3ScaledRoPE: Module {
    let dims: Int
    let traditional: Bool

    /// Pre-computed scaled frequencies for Llama3-style RoPE
    private let _freqs: MLXArray

    init(
        dims: Int,
        traditional: Bool = false,
        base: Float = 500000.0,
        scaleFactor: Float = 32.0,
        lowFreqFactor: Float = 1.0,
        highFreqFactor: Float = 4.0,
        oldContextLen: Float = 8192.0
    ) {
        precondition(dims % 2 == 0, "RoPE dims must be even")
        self.dims = dims
        self.traditional = traditional

        // Compute Llama3-scaled frequencies exactly like Python mlx-lm
        // Python: freqs = base ** (mx.arange(0, dims, 2) / dims)
        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        let exponents = indices / MLXArray(Float(dims))
        var freqs = MLX.pow(MLXArray(base), exponents)  // base^(i/d), NOT negative!

        // Compute wavelengths: wavelens = 2 * pi * freqs
        let wavelens = MLXArray(2.0 * Float.pi) * freqs

        // Threshold wavelengths
        let lowFreqWavelen = oldContextLen / lowFreqFactor   // e.g., 8192 / 1 = 8192
        let highFreqWavelen = oldContextLen / highFreqFactor // e.g., 8192 / 4 = 2048

        // Scale LOW frequencies (long wavelengths > lowFreqWavelen)
        // Python: freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        freqs = MLX.where(wavelens .> MLXArray(lowFreqWavelen), freqs * scaleFactor, freqs)

        // Identify MEDIUM frequencies (wavelens > highFreqWavelen AND wavelens < lowFreqWavelen)
        let isMediumFreq = logicalAnd(wavelens .> MLXArray(highFreqWavelen), wavelens .< MLXArray(lowFreqWavelen))

        // Compute smooth interpolation factors
        // Python: smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        let smoothFactors = (MLXArray(oldContextLen) / wavelens - MLXArray(lowFreqFactor))
            / MLXArray(highFreqFactor - lowFreqFactor)

        // Compute smooth frequencies
        // Python: smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        let denominator = (MLXArray(1.0) - smoothFactors) / MLXArray(scaleFactor) + smoothFactors
        let smoothFreqs = freqs / denominator

        // Final frequencies: use smooth for medium, keep (scaled) freqs for low/high
        // Python: self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)
        self._freqs = MLX.where(isMediumFreq, smoothFreqs, freqs)

        super.init()
    }

    convenience init(dims: Int, config: LlamaTTSConfiguration) {
        let base = config.ropeTheta
        let rs = config.ropeScaling

        func num(_ k: String, _ d: Float) -> Float {
            guard let v = rs?[k] else { return d }
            switch v {
            case .float(let x): return Float(x)
            case .int(let x): return Float(x)
            case .string(let s): return Float(s) ?? d
            default:
                assertionFailure("unexpected value for \(k): \(v)")
                return d
            }
        }

        self.init(
            dims: dims,
            traditional: config.ropeTraditional,
            base: base,
            scaleFactor: num("factor", 32.0),
            lowFreqFactor: num("low_freq_factor", 1.0),
            highFreqFactor: num("high_freq_factor", 4.0),
            oldContextLen: num("original_max_position_embeddings", 8192.0)
        )
    }

    func callAsFunction(_ x: MLXArray, offset: Int? = nil) -> MLXArray {
        // Use MLXFast.RoPE with custom frequencies, exactly like Python:
        // return mx.fast.rope(x, self.dims, traditional=self.traditional,
        //                     base=None, scale=1.0, offset=offset, freqs=self._freqs)
        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,  // Use custom freqs instead
            scale: 1.0,
            offset: offset ?? 0,
            freqs: _freqs
        )
    }
}

// MARK: - Attention

private class LlamaTTSAttention: Module {
    let args: LlamaTTSConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: Llama3ScaledRoPE

    init(_ args: LlamaTTSConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = Llama3ScaledRoPE(dims: headDim, config: args)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
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

private class LlamaTTSMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: LlamaTTSConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

private class LlamaTTSTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: LlamaTTSAttention
    @ModuleInfo(key: "mlp") var mlp: LlamaTTSMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: LlamaTTSConfiguration) {
        self._attention.wrappedValue = LlamaTTSAttention(args)
        self._mlp.wrappedValue = LlamaTTSMLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

// MARK: - Inner Model

private class LlamaTTSModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [LlamaTTSTransformerBlock]
    let norm: RMSNorm

    init(_ args: LlamaTTSConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers)
            .map { _ in LlamaTTSTransformerBlock(args) }

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

// MARK: - Main TTS Model

/// Llama-based TTS model for Orpheus and similar architectures.
///
/// This model generates audio from text using SNAC audio codec tokens.
/// It supports voice cloning and streaming generation.
public class LlamaTTSModel: Module, KVCacheDimensionProvider, SpeechGenerationModel, @unchecked Sendable {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var tokenizer: Tokenizer?
    public var _snacModel: SNAC?

    private let model: LlamaTTSModelInner
    let configuration: LlamaTTSConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var numLayers: Int {
        return configuration.hiddenLayers
    }

    public init(_ args: LlamaTTSConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0..<args.hiddenLayers).map { _ in args.kvHeads }
        self.model = LlamaTTSModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    // MARK: - Parse Output

    /// Parse generated tokens to extract audio codes.
    public func parseOutput(_ inputIds: MLXArray) -> [[Int]] {
        let tokenToFind = OrpheusTokens.startOfSpeech
        let tokenToRemove = OrpheusTokens.endOfSpeech

        // Find last occurrence of START_OF_SPEECH token
        let mask = inputIds .== tokenToFind
        var lastOccurrenceIdx: Int? = nil

        for i in 0..<mask.shape[0] {
            for j in 0..<mask.shape[1] {
                if mask[i, j].item(Int.self) != 0 {
                    lastOccurrenceIdx = j
                }
            }
        }

        var croppedTensor: MLXArray

        if let idx = lastOccurrenceIdx {
            croppedTensor = inputIds[0..., (idx + 1)...]
        } else {
            croppedTensor = inputIds
        }

        // Process each row
        var processedRows: [MLXArray] = []

        for i in 0..<croppedTensor.shape[0] {
            let row = croppedTensor[i]
            let rowList = row.asArray(Int.self)

            // Filter out END_OF_SPEECH tokens
            let maskedRow = rowList.filter { $0 != tokenToRemove }
            processedRows.append(MLXArray(maskedRow))
        }

        // Create code lists
        var codeLists: [[Int]] = []

        for row in processedRows {
            let rowLength = row.shape[0]
            let newLength = (rowLength / 7) * 7
            let trimmedRow = row[0..<newLength]

            // Subtract audio token offset from each token
            let trimmedList = trimmedRow.asArray(Int.self)
            let codeList = trimmedList.map { $0 - OrpheusTokens.audioTokenOffset }
            codeLists.append(codeList)
        }

        return codeLists
    }

    // MARK: - Prepare Input IDs

    /// Prepare input token IDs for generation.
    ///
    /// - Parameters:
    ///   - prompts: Array of text prompts
    ///   - voice: Optional voice identifier (e.g., "tara")
    ///   - refAudio: Optional reference audio for voice cloning
    ///   - refText: Optional transcript of reference audio
    /// - Returns: Tuple of (input IDs, attention mask)
    public func prepareInputIds(
        prompts: [String],
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil
    ) -> (MLXArray, MLXArray) {

        var audioInputIds: MLXArray?
        var audioTranscriptIds: MLXArray?

        // Handle reference audio and text for voice cloning
        if let refAudio = refAudio, let refText = refText {
            print("\u{001B}[93mWARNING: Audio cloning doesn't work reliably on Orpheus.\u{001B}[0m")
            print("A known issue affecting Torch and MLX versions.")

            guard let snacModel = self._snacModel else {
                fatalError("SNAC model not loaded. Call post_load_hook first.")
            }

            let codes = llamaEncodeAudioToCodes(audio: refAudio, snacModel: snacModel)
            audioInputIds = codes + OrpheusTokens.audioTokenOffset
            let encodedIds = tokenizer!.encode(text: refText)
            audioTranscriptIds = MLXArray(encodedIds.map { Int32($0) }).expandedDimensions(axis: 0)
        }

        // Apply voice prefix if provided
        var modifiedPrompts = prompts
        if let voice = voice {
            modifiedPrompts = prompts.map { "\(voice): \($0)" }
        }

        // Define special tokens
        let startToken = MLXArray([Int32(OrpheusTokens.startOfHuman)]).expandedDimensions(axis: 0)
        let endTokens = MLXArray([
            Int32(OrpheusTokens.endOfText),
            Int32(OrpheusTokens.endOfHuman)
        ]).expandedDimensions(axis: 0)

        // Encode all prompts
        var promptInputIds: [MLXArray] = []
        for prompt in modifiedPrompts {
            let encodedIds = tokenizer!.encode(text: prompt)
            let encoded = MLXArray(encodedIds.map { Int32($0) }).expandedDimensions(axis: 0)
            promptInputIds.append(encoded)
        }

        // Prepare batch with padding
        var batchInputIds: [MLXArray] = []
        let padToken = MLXArray([Int32(OrpheusTokens.padToken)])
        let maxLen = promptInputIds.map { $0.shape[1] }.max() ?? 0

        for inputIds in promptInputIds {
            var modifiedInputIds: [MLXArray] = []

            // Add padding if needed
            let paddingLen = maxLen - inputIds.shape[1]
            if paddingLen > 0 {
                let padding = repeated(padToken, count: paddingLen, axis: 0)
                    .expandedDimensions(axis: 0)
                modifiedInputIds.append(padding)
            }

            // Add reference audio and transcript if provided
            if let audioInputIds = audioInputIds, let audioTranscriptIds = audioTranscriptIds {
                let audioStartTokens = MLXArray([
                    Int32(OrpheusTokens.audioStart),
                    Int32(OrpheusTokens.startOfSpeech)
                ]).expandedDimensions(axis: 0)

                let audioEndTokens = MLXArray([
                    Int32(OrpheusTokens.endOfSpeech),
                    Int32(OrpheusTokens.audioEnd)
                ]).expandedDimensions(axis: 0)

                let refInputIds = concatenated([
                    startToken,
                    audioTranscriptIds,
                    endTokens,
                    audioStartTokens,
                    audioInputIds,
                    audioEndTokens
                ], axis: 1)

                modifiedInputIds.append(refInputIds)
            }

            // Add prompt with start/end tokens: [SOH] prompt [EOT] [EOH]
            let onePromptInputIds = concatenated([
                startToken,
                inputIds,
                endTokens
            ], axis: 1)

            modifiedInputIds.append(onePromptInputIds)

            // Concatenate all parts for this prompt
            let fullInputIds = concatenated(modifiedInputIds, axis: 1)
            batchInputIds.append(fullInputIds)
        }

        // Concatenate all prompts in batch
        let finalBatchInputIds = concatenated(batchInputIds, axis: 0)

        // Create attention mask (False for pad tokens, True otherwise)
        let batchMask = finalBatchInputIds .!= padToken

        return (finalBatchInputIds, batchMask)
    }

    // MARK: - Forward Pass

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
        return configuration.sampleRate
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
        var weights = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    public func post_load_hook(model: LlamaTTSModel, modelDir: URL) async throws {
        if model.tokenizer == nil {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        }
        if model._snacModel == nil {
            model._snacModel = try await SNAC.fromPretrained("mlx-community/snac_24khz")
        }
    }

    public func makeCache() -> [KVCache] {
        return (0..<configuration.hiddenLayers).map { _ in
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

    // MARK: - Generation

    /// Generate audio from text.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - voice: Optional voice identifier (e.g., "tara")
    ///   - refAudio: Optional reference audio for voice cloning
    ///   - refText: Optional transcript of reference audio
    ///   - cache: Optional pre-existing KV cache
    ///   - parameters: Generation parameters
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
            throw LlamaTTSError.modelNotInitialized("SNAC model not loaded")
        }
        guard tokenizer != nil else {
            throw LlamaTTSError.modelNotInitialized("Tokenizer not loaded")
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

        var generatedTokens: [Int32] = []
        let promptTokensList = inputIds.squeezed(axis: 0).asArray(Int32.self)
        generatedTokens.append(contentsOf: promptTokensList)

        let maxTokens = parameters.maxTokens ?? 1200

        // Prefill: process the prompt
        var logits = self(inputIds, cache: cache)

        // Generate tokens
        for i in 0..<maxTokens {
            let tokenValue: Int = autoreleasepool {
                var lastLogits = logits[0..., -1, 0...]
                lastLogits = processor?.process(logits: lastLogits) ?? lastLogits

                let nextToken = sampler.sample(logits: lastLogits)
                processor?.didSample(token: nextToken)

                let value = nextToken.item(Int.self)

                if value != OrpheusTokens.endOfSpeech {
                    let nextTokenExpanded = nextToken.reshaped([1, 1])
                    logits = self(nextTokenExpanded, cache: cache)
                    eval(logits)
                }

                return value
            }

            if tokenValue == OrpheusTokens.endOfSpeech {
                break
            }

            generatedTokens.append(Int32(tokenValue))

            // Periodically clear GPU cache
            if i % 50 == 0 {
                Memory.clearCache()
            }
        }

        Memory.clearCache()

        let allTokens = MLXArray(generatedTokens).expandedDimensions(axis: 0)

        // Parse output to audio codes
        let codeLists = parseOutput(allTokens)

        guard let codeList = codeLists.first, !codeList.isEmpty else {
            throw LlamaTTSError.generationFailed("No audio codes generated")
        }

        // Decode audio using SNAC
        let audio = llamaDecodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
        audio.eval()

        Memory.clearCache()

        return audio
    }

    /// Generate audio with streaming token output.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - voice: Optional voice identifier
    ///   - refAudio: Optional reference audio for voice cloning
    ///   - refText: Optional transcript of reference audio
    ///   - cache: Optional pre-existing KV cache
    ///   - parameters: Generation parameters
    /// - Returns: AsyncThrowingStream of generation events
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
    ) -> AsyncThrowingStream<LlamaTTSGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<LlamaTTSGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else { return }
            
            do {
                guard let snacModel = self._snacModel else {
                    throw LlamaTTSError.modelNotInitialized("SNAC model not loaded")
                }
                guard self.tokenizer != nil else {
                    throw LlamaTTSError.modelNotInitialized("Tokenizer not loaded")
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
                
                var generatedTokens: [Int32] = []
                let promptTokensList = inputIds.squeezed(axis: 0).asArray(Int32.self)
                generatedTokens.append(contentsOf: promptTokensList)
                
                let maxTokens = parameters.maxTokens ?? 1200
                
                let startTime = Date()
                
                // Prefill
                var tokenCount: Int = 0
                var logits = self(inputIds, cache: cache)
                let prefillTime = Date().timeIntervalSince(startTime)
                
                let generateStartTime = Date()
                
                // Generate tokens
                for i in 0..<maxTokens {
                    if Task.isCancelled { break }
                    
                    let tokenValue: Int = autoreleasepool {
                        var lastLogits = logits[0..., -1, 0...]
                        lastLogits = processor?.process(logits: lastLogits) ?? lastLogits
                        
                        let nextToken = sampler.sample(logits: lastLogits)
                        processor?.didSample(token: nextToken)
                        
                        let value = nextToken.item(Int.self)
                        
                        if value != OrpheusTokens.endOfSpeech {
                            let nextTokenExpanded = nextToken.reshaped([1, 1])
                            logits = self(nextTokenExpanded, cache: cache)
                            eval(logits)
                        }
                        
                        return value
                    }
                    
                    tokenCount += 1
                    continuation.yield(.token(tokenValue))
                    
                    if tokenValue == OrpheusTokens.endOfSpeech {
                        break
                    }
                    
                    generatedTokens.append(Int32(tokenValue))
                    
                    if i % 50 == 0 {
                        Memory.clearCache()
                    }
                }
                
                let generateTime = Date().timeIntervalSince(generateStartTime)
                
                let allTokens = MLXArray(generatedTokens).expandedDimensions(axis: 0)
                
                // Parse and decode audio
                let codeLists = self.parseOutput(allTokens)
                
                guard let codeList = codeLists.first, !codeList.isEmpty else {
                    throw LlamaTTSError.generationFailed("No audio codes generated")
                }
                
                let audio = llamaDecodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
                audio.eval()
                
                Memory.clearCache()
                
                // Yield completion info
                let info = LlamaTTSGenerationInfo(
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

    // MARK: - Loading

    /// Load a pre-trained Llama TTS model from Hugging Face Hub.
    ///
    /// - Parameter modelRepo: The model repository ID (e.g., "mlx-community/orpheus-3b-0.1-ft-bf16")
    /// - Returns: The loaded model
    public static func fromPretrained(_ modelRepo: String) async throws -> LlamaTTSModel {
        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(
                domain: "LlamaTTSModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"]
            )
        }

        let modelDir = try await llamaTTSResolveOrDownloadModel(
            client: client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors"
        )

        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(LlamaTTSConfiguration.self, from: configData)

        let perLayerQuantization = config.perLayerQuantization

        let model = LlamaTTSModel(config)

        // Load weights from safetensors
        let weights = try llamaTTSLoadWeights(from: modelDir)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed
        if perLayerQuantization != nil {
            print("Applying quantization from config...")

            quantize(model: model) { path, module in
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

// MARK: - Helper Functions

private func llamaTTSLoadWeights(from directory: URL) throws -> [String: MLXArray] {
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

private func llamaTTSResolveOrDownloadModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID,
    requiredExtension: String
) async throws -> URL {
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    // Check if model already exists with required files (config.json + safetensors)
    let configPath = modelDir.appendingPathComponent("config.json")
    if FileManager.default.fileExists(atPath: configPath.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasRequiredFiles = files?.contains { $0.pathExtension == requiredExtension } ?? false

        if hasRequiredFiles {
            print("Using cached model at: \(modelDir.path)")
            return modelDir
        }
    }

    // Create directory if needed
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    // Remove any partial downloads to avoid "file exists" errors
    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for file in files ?? [] {
            try? FileManager.default.removeItem(at: file)
        }
    }

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
