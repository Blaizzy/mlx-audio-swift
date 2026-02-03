//
//  Qwen3TTSTalker.swift
//  Qwen3TTS
//
//  Main Talker model and ForConditionalGeneration wrapper for Qwen3-TTS.
//  Ported from mlx_audio/tts/models/qwen3_tts/talker.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Qwen3TTSTalkerModel

/// Main talker transformer model.
///
/// Contains embeddings, transformer layers, and rotary embeddings.
public class Qwen3TTSTalkerModel: Module {
    public let config: Qwen3TTSTalkerConfig
    public let hiddenSize: Int

    // Embeddings
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding

    // Transformer layers
    @ModuleInfo(key: "layers") var layers: [TalkerDecoderLayer]

    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Rotary embeddings (non-trainable, computed)
    public let rotaryEmb: TalkerRotaryEmbedding

    public init(config: Qwen3TTSTalkerConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize

        // Initialize embeddings
        self._codecEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._textEmbedding.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize,
            dimensions: config.textHiddenSize
        )

        // Initialize transformer layers
        var decoderLayers: [TalkerDecoderLayer] = []
        for i in 0..<config.numHiddenLayers {
            decoderLayers.append(TalkerDecoderLayer(config: config, layerIdx: i))
        }
        self._layers.wrappedValue = decoderLayers

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Initialize rotary embeddings with MRoPE section from config
        self.rotaryEmb = TalkerRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta,
            mropeSection: config.mropeSection
        )
    }

    /// Forward pass through the transformer.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings [batch, seq_len, hidden_size]
    ///   - positionIds: Optional 3D position IDs [3, batch, seq_len] for MRoPE
    ///   - mask: Optional attention mask
    ///   - cache: Optional KV cache for all layers
    /// - Returns: Hidden states [batch, seq_len, hidden_size]
    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [TalkerKVCache]? = nil
    ) -> MLXArray {
        let batch = inputsEmbeds.shape[0]
        let seqLen = inputsEmbeds.shape[1]

        // Get offset from cache for position calculation
        let offset: Int
        if let cache = cache, !cache.isEmpty, let firstCache = cache[0] as? TalkerSimpleKVCache {
            offset = firstCache.sequenceLength
        } else {
            offset = 0
        }

        // Generate 3D position ids if not provided (for MRoPE)
        let posIds: MLXArray
        if let providedPosIds = positionIds {
            posIds = providedPosIds
        } else {
            // Create [3, batch, seq_len] position ids - all three dimensions same for text
            let positions = MLXArray(Array(offset..<(offset + seqLen)).map { Int32($0) })
            let posBatch = broadcast(positions.reshaped([1, seqLen]), to: [batch, seqLen])
            posIds = stacked([posBatch, posBatch, posBatch], axis: 0)
        }

        var x = inputsEmbeds

        // Compute position embeddings (MRoPE)
        let positionEmbeddings = rotaryEmb(x, positionIds: posIds)

        // Create causal mask if not provided
        let attentionMask: MLXArray?
        if mask == nil && seqLen > 1 {
            attentionMask = createCausalMask(seqLen: seqLen, dtype: x.dtype)
        } else {
            attentionMask = mask
        }

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            x = layer(x, positionEmbeddings: positionEmbeddings, mask: attentionMask, cache: layerCache)
        }

        x = norm(x)
        return x
    }

    /// Create KV cache for all layers.
    public func makeCache() -> [TalkerKVCache] {
        return layers.map { _ in TalkerSimpleKVCache() }
    }
}

// MARK: - Qwen3TTSTalkerForConditionalGeneration

/// Full talker model for conditional generation.
///
/// Wraps the main model with text projection, codec head, and code predictor.
public class Qwen3TTSTalkerForConditionalGeneration: Module {
    public let config: Qwen3TTSTalkerConfig

    @ModuleInfo(key: "model") var model: Qwen3TTSTalkerModel

    // Text projection: text_hidden_size -> hidden_size
    @ModuleInfo(key: "text_projection") var textProjection: ResizeMLP

    // Codec head for first token prediction
    @ModuleInfo(key: "codec_head") var codecHead: Linear

    // Code predictor for remaining tokens (groups 1 to N-1)
    @ModuleInfo(key: "code_predictor") var codePredictor: Qwen3TTSTalkerCodePredictor

    public init(config: Qwen3TTSTalkerConfig) {
        self.config = config

        self._model.wrappedValue = Qwen3TTSTalkerModel(config: config)

        // Text projection MLP
        self._textProjection.wrappedValue = ResizeMLP(
            inputSize: config.textHiddenSize,
            intermediateSize: config.textHiddenSize,
            outputSize: config.hiddenSize,
            hiddenAct: config.hiddenAct,
            bias: true
        )

        // Codec head for first token (group 0) prediction
        self._codecHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        // Code predictor for remaining code groups (1 to numCodeGroups-1)
        let codePredictorConfig = config.codePredictorConfig ?? Qwen3TTSTalkerCodePredictorConfig()
        self._codePredictor.wrappedValue = Qwen3TTSTalkerCodePredictor(
            config: codePredictorConfig,
            talkerHiddenSize: config.hiddenSize
        )
    }

    /// Get the codec embedding layer.
    public var codecEmbedding: Embedding {
        return model.codecEmbedding
    }

    /// Get the text embedding layer.
    public var textEmbedding: Embedding {
        return model.textEmbedding
    }

    /// Forward pass for the talker model.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings [batch, seq_len, hidden_size]
    ///   - positionIds: Optional 3D position IDs for MRoPE
    ///   - mask: Optional attention mask
    ///   - cache: Optional KV cache
    /// - Returns: Tuple of (logits for next token, hidden states)
    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [TalkerKVCache]? = nil
    ) -> (logits: MLXArray, hiddenStates: MLXArray) {
        let hiddenStates = model(inputsEmbeds, positionIds: positionIds, mask: mask, cache: cache)
        let logits = codecHead(hiddenStates)
        return (logits, hiddenStates)
    }

    /// Create KV cache for all layers.
    public func makeCache() -> [TalkerKVCache] {
        return model.makeCache()
    }

    // MARK: - Weight Sanitization

    /// Sanitize weights from PyTorch/safetensors format to MLX format.
    ///
    /// - Parameter weights: Raw weights dictionary
    /// - Returns: Sanitized weights for this model (without "talker." prefix)
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Only process talker weights
            guard key.hasPrefix("talker.") else { continue }

            // Remove "talker." prefix
            let newKey = String(key.dropFirst("talker.".count))

            // No transpositions needed for linear layers in MLX Swift
            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Weight Loading

    /// Load weights from a model directory with quantization support.
    ///
    /// - Parameters:
    ///   - directory: URL to the model directory containing safetensor files
    ///   - quantization: Optional quantization config for quantized models
    /// - Throws: Error if files cannot be loaded
    public func loadWeights(from directory: URL, quantization: Qwen3TTSQuantization? = nil) throws {
        // Load all weights from safetensor files
        var allWeights = [String: MLXArray]()
        let fileManager = FileManager.default

        let weightsURL = directory.appendingPathComponent("model.safetensors")
        if fileManager.fileExists(atPath: weightsURL.path) {
            let weights = try loadArrays(url: weightsURL)
            for (key, value) in weights {
                allWeights[key] = value
            }
        } else {
            // Try multiple safetensors files
            let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

            for file in safetensorFiles {
                let weights = try loadArrays(url: file)
                for (key, value) in weights {
                    allWeights[key] = value
                }
            }
        }

        // Sanitize weights (remove "talker." prefix)
        let sanitizedWeights = Self.sanitize(allWeights)

        // Handle quantization if present
        if let quant = quantization {
            // Quantize layers that have .scales weights
            MLXNN.quantize(
                model: self,
                filter: { (path: String, module: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                    if sanitizedWeights["\(path).scales"] != nil {
                        return (quant.groupSize, quant.bits, .affine)
                    }
                    return nil
                }
            )
        }

        // Apply weights using ModuleParameters
        let parameters = ModuleParameters.unflattened(sanitizedWeights)
        try self.update(parameters: parameters, verify: [])

        eval(self)
    }

    /// Load weights from a safetensors file URL (legacy method for non-quantized models).
    ///
    /// - Parameters:
    ///   - url: URL to the safetensors file
    ///   - strict: Whether to require all weights to be present
    /// - Throws: Error if file cannot be loaded
    public func loadWeights(from url: URL, strict: Bool = false) throws {
        let rawWeights = try loadArrays(url: url)
        let sanitizedWeights = Self.sanitize(rawWeights)

        // Apply weights using ModuleParameters
        let parameters = ModuleParameters.unflattened(sanitizedWeights)
        try self.update(parameters: parameters, verify: [])

        eval(self)
    }

    // MARK: - Generation Methods

    /// Sample a token from logits.
    ///
    /// - Parameters:
    ///   - logits: Logits tensor [batch, seq_len, vocab_size]
    ///   - temperature: Temperature for sampling (higher = more random)
    ///   - topK: Top-k filtering (0 = disabled)
    ///   - topP: Top-p (nucleus) filtering (1.0 = disabled)
    ///   - repetitionPenalty: Penalty for repeated tokens (1.0 = disabled)
    ///   - generatedTokens: Previously generated tokens for repetition penalty
    ///   - eosTokenId: EOS token ID to preserve through filtering (prevents premature termination)
    /// - Returns: Sampled token [batch, 1]
    public func sampleToken(
        logits: MLXArray,
        temperature: Float = 1.0,
        topK: Int = 0,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int] = [],
        eosTokenId: Int? = nil
    ) -> MLXArray {
        // Take last position logits [batch, vocab_size]
        let seqLen = logits.shape[1]
        let lastLogits = logits[0..., (seqLen - 1)..<seqLen, 0...].squeezed(axis: 1)

        var processedLogits = lastLogits
        let vocabSize = processedLogits.shape[processedLogits.ndim - 1]

        // Apply repetition penalty
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            processedLogits = applyRepetitionPenalty(
                processedLogits,
                generatedTokens: generatedTokens,
                penalty: repetitionPenalty
            )
        }

        // Greedy decoding: return argmax if temperature is 0 or less
        if temperature <= 0 {
            let maxIdx = argMax(processedLogits, axis: -1)
            return maxIdx.reshaped([1, 1])
        }

        // Save EOS logit before filtering (to preserve it as a valid candidate)
        // This prevents top-k/top-p from removing the EOS token, which can cause
        // the model to fail to terminate properly (word skipping / runaway generation)
        var eosLogit: MLXArray? = nil
        if let eosId = eosTokenId, eosId < vocabSize {
            eosLogit = processedLogits[0..., eosId..<(eosId + 1)]
        }

        // Apply top-k filtering (on raw logits, before temperature)
        if topK > 0 {
            processedLogits = topKFilter(processedLogits, k: topK)
        }

        // Apply top-p filtering (on raw logits, before temperature)
        if topP < 1.0 {
            processedLogits = topPFilter(processedLogits, p: topP)
        }

        // Restore EOS logit after filtering so it's always a valid candidate
        if let eosId = eosTokenId, let savedEosLogit = eosLogit, eosId < vocabSize {
            // Use putAlong to restore the EOS logit at its original position
            let eosIdx = MLXArray([Int32(eosId)]).reshaped([1, 1])
            processedLogits = putAlong(processedLogits, eosIdx, values: savedEosLogit, axis: -1)
        }

        // Apply temperature scaling right before sampling (matches Python categorical_sampling)
        // categorical_sampling(logits, temp) = categorical(logits / temp)
        if temperature != 1.0 {
            processedLogits = processedLogits / temperature
        }

        // Sample from distribution (categorical expects unnormalized logits)
        let sampled = categorical(processedLogits)

        // Return as [batch, 1]
        return sampled.reshaped([1, 1])
    }

    /// Prepare generation inputs from tokenized text.
    ///
    /// - Parameters:
    ///   - inputIds: Tokenized input IDs [batch, seq_len]
    ///   - speaker: Optional speaker name for voice selection (CustomVoice models)
    ///   - speakerEmbedding: Optional speaker embedding from reference audio [1, enc_dim]
    ///   - instructIds: Optional tokenized instruct IDs for voice design/style (VoiceDesign/CustomVoice models)
    /// - Returns: Tuple of (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    public func prepareGenerationInputs(
        inputIds: MLXArray,
        speaker: String? = nil,
        speakerEmbedding: MLXArray? = nil,
        instructIds: MLXArray? = nil
    ) -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        // Get text embeddings and project to talker hidden size
        // Note: textEmbed is computed but currently unused as we build embeddings piece by piece
        _ = textProjection(model.textEmbedding(inputIds))

        // TTS special token embeddings from config
        let ttsBosTokenId = config.ttsBosTokenId
        let ttsEosTokenId = config.ttsEosTokenId
        let ttsPadTokenId = config.ttsPadTokenId

        let ttsTokens = MLXArray([Int32(ttsBosTokenId), Int32(ttsEosTokenId), Int32(ttsPadTokenId)]).reshaped([1, 3])
        let ttsEmbeds = textProjection(model.textEmbedding(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]  // [1, 1, hidden]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]  // [1, 1, hidden]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]  // [1, 1, hidden]

        // Codec prefix tokens for language/thinking
        let codecNothinkId = config.codecNothinkId
        let codecThinkBosId = config.codecThinkBosId
        let codecThinkEosId = config.codecThinkEosId
        let codecPadId = config.codecPadId
        let codecBosId = config.codecBosId

        // Create codec prefix embeddings
        let codecPrefill = MLXArray([Int32(codecNothinkId), Int32(codecThinkBosId), Int32(codecThinkEosId)]).reshaped([1, 3])
        var codecEmbed = model.codecEmbedding(codecPrefill)

        // Add speaker embedding - prioritize ref_audio embedding over speaker name lookup
        if let refSpeakerEmbed = speakerEmbedding {
            // Use speaker embedding from reference audio
            // Reshape to [1, 1, hidden] if needed
            let spkEmbed = refSpeakerEmbed.reshaped([1, 1, -1])
            codecEmbed = concatenated([codecEmbed, spkEmbed], axis: 1)
        } else if let speakerName = speaker?.lowercased(),
                  let spkId = config.spkId?[speakerName] {
            // Fall back to speaker ID lookup
            let spkIdArray = MLXArray([Int32(spkId)]).reshaped([1, 1])
            let speakerEmbed = model.codecEmbedding(spkIdArray)  // [1, 1, hidden]
            codecEmbed = concatenated([codecEmbed, speakerEmbed], axis: 1)
        }

        // Add pad and bos suffix
        let codecSuffix = MLXArray([Int32(codecPadId), Int32(codecBosId)]).reshaped([1, 2])
        let codecSuffixEmbed = model.codecEmbedding(codecSuffix)
        codecEmbed = concatenated([codecEmbed, codecSuffixEmbed], axis: 1)

        // Instruct embedding (for VoiceDesign/CustomVoice models)
        var instructEmbed: MLXArray? = nil
        if let ids = instructIds {
            instructEmbed = textProjection(model.textEmbedding(ids))
        }

        // Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = textProjection(model.textEmbedding(inputIds[0..., 0..<3]))

        // Build combined input: tts_pad * (codec_len - 2) + tts_bos + codec_embed[:, :-1]
        let padCount = codecEmbed.shape[1] - 2  // 3 padding tokens
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, config.hiddenSize])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)  // [1, 4, hidden]

        // Add codec embeddings (all but last)
        combinedEmbed = combinedEmbed + codecEmbed[0..., 0..<(codecEmbed.shape[1] - 1), 0...]

        // Combine role and prefix embeddings
        // If instruct is provided, prepend it
        var inputEmbeds: MLXArray
        if let instruct = instructEmbed {
            inputEmbeds = concatenated([instruct, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // Add first text token (after role tokens)
        if inputIds.shape[1] > 3 {
            let firstTextEmbed = textProjection(model.textEmbedding(inputIds[0..., 3..<4]))
            let lastCodecEmbed = codecEmbed[0..., (codecEmbed.shape[1] - 1)..., 0...]
            inputEmbeds = concatenated([inputEmbeds, firstTextEmbed + lastCodecEmbed], axis: 1)
        }

        // Trailing text hidden (rest of text + tts_eos)
        var trailingTextHidden: MLXArray
        if inputIds.shape[1] > 4 {
            // Skip first 4 tokens (role + first text) and last few tokens
            let trailingEnd = max(4, inputIds.shape[1] - 5)
            if trailingEnd > 4 {
                let trailingTextEmbed = textProjection(model.textEmbedding(inputIds[0..., 4..<trailingEnd]))
                trailingTextHidden = concatenated([trailingTextEmbed, ttsEosEmbed], axis: 1)
            } else {
                trailingTextHidden = ttsEosEmbed
            }
        } else {
            trailingTextHidden = ttsEosEmbed
        }

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    /// Generate audio codes autoregressively.
    ///
    /// - Parameters:
    ///   - inputIds: Tokenized input IDs [batch, seq_len]
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - temperature: Temperature for sampling
    ///   - topK: Top-k filtering
    ///   - topP: Top-p filtering
    ///   - repetitionPenalty: Penalty for repeated tokens
    ///   - speaker: Optional speaker name for voice selection (CustomVoice models)
    ///   - speakerEmbedding: Optional speaker embedding from reference audio [1, enc_dim]
    ///   - instructIds: Optional tokenized instruct IDs for voice design/style
    /// - Returns: Generated codes [batch, seq_len, num_code_groups]
    public func generate(
        inputIds: MLXArray,
        maxTokens: Int = 1000,
        temperature: Float = 1.0,
        topK: Int = 50,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.0,
        speaker: String? = nil,
        speakerEmbedding: MLXArray? = nil,
        instructIds: MLXArray? = nil
    ) -> MLXArray {
        // Prepare initial embeddings
        let (initialEmbeds, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(inputIds: inputIds, speaker: speaker, speakerEmbedding: speakerEmbedding, instructIds: instructIds)
        eval(initialEmbeds)
        eval(trailingTextHidden)
        eval(ttsPadEmbed)

        // Initialize caches
        let talkerCache = makeCache()
        var inputEmbeds = initialEmbeds
        var trailingIdx = 0
        var generatedCodes: [MLXArray] = []
        generatedCodes.reserveCapacity(maxTokens)
        var generatedTokens: [Int] = []
        generatedTokens.reserveCapacity(maxTokens)

        let eosTokenId = config.codecEosTokenId

        // Autoregressive generation loop
        for step in 0..<maxTokens {
            // Forward pass through talker
            let (logits, hiddenStates) = self(inputEmbeds, cache: talkerCache)
            eval(logits)
            eval(hiddenStates)

            // Sample first codebook token (code 0)
            let nextToken = sampleToken(
                logits: logits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokens,
                eosTokenId: eosTokenId
            )
            eval(nextToken)

            let tokenValue = nextToken[0, 0].item(Int32.self)

            // Check for EOS
            if tokenValue == Int32(eosTokenId) {
                break
            }

            generatedTokens.append(Int(tokenValue))

            // Generate remaining codebook tokens (codes 1-15) with code predictor
            var codeTokens: [MLXArray] = [nextToken]
            let hiddenSeqLen = hiddenStates.shape[1]
            let codeHidden = hiddenStates[0..., (hiddenSeqLen - 1)..<hiddenSeqLen, 0...]  // [1, 1, hidden]
            let codePredictorCache = codePredictor.makeCache()


            for codeIdx in 0..<(config.numCodeGroups - 1) {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    // Prefill: concatenate hidden state with code 0 embedding
                    let code0Embed = model.codecEmbedding(nextToken.asType(.int32))
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    // Generation: embedding of previous code token
                    let prevCode = codeTokens[codeTokens.count - 1]
                    codeInput = codePredictor.codecEmbedding[codeIdx - 1](prevCode.asType(.int32))
                }

                let (codeLogits, _, _) = codePredictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                eval(codeLogits)

                // Sample from last position
                let nextCode = sampleToken(
                    logits: codeLogits,
                    temperature: temperature,
                    topK: topK,
                    topP: topP
                )
                eval(nextCode)

                codeTokens.append(nextCode)
            }

            // Stack all codebook tokens: [1, 16]
            let allCodes = concatenated(codeTokens, axis: 1)
            eval(allCodes)
            generatedCodes.append(allCodes)

            // Prepare next input embedding
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.shape[1] {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Codec embedding for next step (sum of all codebook embeddings)
            var codecEmbed = model.codecEmbedding(nextToken.asType(.int32))
            for (i, code) in codeTokens.dropFirst().enumerated() {
                let embedding = codePredictor.codecEmbedding[i](code.asType(.int32))
                codecEmbed = codecEmbed + embedding
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds)

            // Clear GPU cache periodically to prevent memory buildup on iOS
            // MLX lazy evaluation builds computation graphs that grow until iOS kills the app
            if (step + 1) % 25 == 0 {
                Memory.clearCache()
            }
        }

        // Final cache clear after generation
        Memory.clearCache()

        // Return empty array if no codes generated
        if generatedCodes.isEmpty {
            return MLXArray.zeros([1, 0, config.numCodeGroups]).asType(.int32)
        }

        // Stack all codes: [1, seq_len, 16]
        let codes = stacked(generatedCodes, axis: 1)
        return codes.asType(.int32)
    }
}
