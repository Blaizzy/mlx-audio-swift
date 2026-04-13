import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - OmniVoice Model

/// OmniVoice: A massively multilingual zero-shot TTS model supporting over 600 languages.
///
/// Built on a novel diffusion language model architecture with a Qwen3 LLM backbone,
/// OmniVoice supports:
/// - **Voice Cloning**: Clone any voice from a reference audio + transcript
/// - **Voice Design**: Create custom voices via text instructions
/// - **Auto Voice**: Default voice when no voice specification is provided
public final class OmniVoiceModel: Module, SpeechGenerationModel, @unchecked Sendable {
    // MARK: - Properties

    let config: OmniVoiceConfig

    /// Qwen3 LLM backbone
    @ModuleInfo(key: "llm") private var llm: Qwen3Model

    /// Audio embeddings: array of embeddings, one per codebook
    /// Each maps audio token IDs to hidden states: [audioVocabSize, hiddenSize]
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: [Embedding]

    /// Audio heads: array of Linear layers, one per codebook
    /// Each projects hidden states to codebook logits: [hiddenSize, audioVocabSize]
    @ModuleInfo(key: "audio_heads") var audioHeads: [Linear]

    /// Normalized codebook weights for loss computation
    private var normalizedCodebookWeights: [Float]

    public var tokenizer: Tokenizer?
    var audioTokenizer: OmniVoiceAudioTokenizer?

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 4096,
            temperature: 1.0,
            topP: 0.95,
            repetitionPenalty: 1.05
        )
    }

    // MARK: - OmniVoice-specific defaults (unused; parameters are passed via generate())

    // MARK: - Initialization

    init(config: OmniVoiceConfig) throws {
        self.config = config
        let llmConfig = config.llmConfig
        print("DEBUG init: llmConfig.hiddenSize=\(llmConfig.hiddenSize)")
        print("DEBUG init: llmConfig.vocabSize=\(llmConfig.vocabSize)")
        print("DEBUG init: config.audioVocabSize=\(config.audioVocabSize)")
        print("DEBUG init: config.numAudioCodebook=\(config.numAudioCodebook)")

        // Create the Qwen3 LLM from config
        let llmConfigWrapper = Qwen3Configuration(
            hiddenSize: llmConfig.hiddenSize,
            hiddenLayers: llmConfig.numHiddenLayers,
            intermediateSize: llmConfig.intermediateSize,
            attentionHeads: llmConfig.numAttentionHeads,
            kvHeads: llmConfig.numKeyValueHeads,
            headDim: llmConfig.headDim,
            vocabularySize: llmConfig.vocabSize,
            rmsNormEps: llmConfig.rmsNormEps,
            ropeTheta: llmConfig.ropeTheta,
            ropeScaling: nil,
            tieWordEmbeddings: llmConfig.tieWordEmbeddings,
            sampleRate: 24000
        )
        self._llm.wrappedValue = Qwen3Model(llmConfigWrapper)

        // Audio embeddings: array of [numAudioCodebook] embeddings, each [audioVocabSize, hiddenSize]
        self._audioEmbeddings.wrappedValue = (0..<config.numAudioCodebook).map { _ in
            Embedding(embeddingCount: config.audioVocabSize, dimensions: llmConfig.hiddenSize)
        }

        // Audio heads: array of [numAudioCodebook] Linear layers, each [hiddenSize, audioVocabSize]
        self._audioHeads.wrappedValue = (0..<config.numAudioCodebook).map { _ in
            Linear(inputDimensions: llmConfig.hiddenSize, outputDimensions: config.audioVocabSize, bias: false)
        }

        // Normalized codebook weights
        let totalWeight = config.audioCodebookWeights.reduce(0, +)
        self.normalizedCodebookWeights = config.audioCodebookWeights.map { Float($0) / Float(totalWeight) }
    }

    // MARK: - Forward Pass

    /// Prepare embeddings from input_ids with audio/text masking.
    private func prepareEmbedInputs(
        inputIds: MLXArray,
        audioMask: MLXArray
    ) -> MLXArray {
        print("DEBUG prepareEmbedInputs: inputIds.shape=\(inputIds.shape), audioMask.shape=\(audioMask.shape)")
        // Text embeddings from LLM
        let textIds = inputIds[0..., 0, 0...]  // [B, S]
        print("DEBUG prepareEmbedInputs: textIds.shape=\(textIds.shape)")
        let textEmbeds = llm.getEmbeddings(for: textIds)
        print("DEBUG prepareEmbedInputs: textEmbeds.shape=\(textEmbeds.shape)")

        // Apply audio mask to inputIds
        let maskedIds = inputIds * audioMask.reshaped([inputIds.shape[0], 1, inputIds.shape[2]])

        // Embed each codebook separately and sum
        var audioEmbeds: MLXArray?
        print("DEBUG prepareEmbedInputs: audioEmbeddings.count=\(audioEmbeddings.count)")
        for (i, embedding) in audioEmbeddings.enumerated() {
            let codebookIds = maskedIds[0..., i, 0...]  // [B, S]
            let codebookEmbeds = embedding(codebookIds)  // [B, S, D]
            print("DEBUG prepareEmbedInputs: codebook \(i) codebookEmbeds.shape=\(codebookEmbeds.shape)")
            if audioEmbeds == nil {
                audioEmbeds = codebookEmbeds
            } else {
                audioEmbeds = audioEmbeds! + codebookEmbeds
            }
        }
        print("DEBUG prepareEmbedInputs: final audioEmbeds.shape=\(audioEmbeds!.shape)")

        // Where audio: use audio_embeds, else use text_embeds
        let result = MLX.where(
            audioMask.reshaped([audioMask.shape[0], audioMask.shape[1], 1]),
            audioEmbeds!,
            textEmbeds
        )
        print("DEBUG prepareEmbedInputs: result.shape=\(result.shape)")
        return result
    }

    /// Forward pass through the model.
    ///
    /// - Parameters:
    ///   - inputIds: [batch, num_codebooks, seq_len]
    ///   - audioMask: [batch, seq_len]
    ///   - attentionMask: unused (mask handling done via token unmasking in diffusion loop)
    ///   - cache: optional KV cache
    /// - Returns: Audio logits [batch, num_codebooks, seq_len, vocab_size]
    func forward(
        inputIds: MLXArray,
        audioMask: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let inputsEmbeds = prepareEmbedInputs(inputIds: inputIds, audioMask: audioMask)
        print("DEBUG forward: inputsEmbeds.shape=\(inputsEmbeds.shape)")

        // Run through LLM (causal masking is handled internally by Qwen3Model)
        let hiddenStates = llm.forwardWithEmbeddings(
            inputsEmbeds: inputsEmbeds,
            cache: cache
        )
        print("DEBUG forward: hiddenStates.shape=\(hiddenStates.shape)")
        print("DEBUG forward: audioHeads.count=\(audioHeads.count)")
        print("DEBUG forward: llmConfig.hiddenSize=\(config.llmConfig.hiddenSize)")
        print("DEBUG forward: audioVocabSize=\(config.audioVocabSize)")

        // Project to audio codebook logits via per-codebook heads
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        var logitsPerCodebook: [MLXArray] = []
        for (i, head) in audioHeads.enumerated() {
            print("DEBUG forward: processing head \(i)")
            let logits = head(hiddenStates)  // [B, S, V]
            print("DEBUG forward: head \(i) logits.shape=\(logits.shape), size=\(logits.size)")
            let reshaped = logits.reshaped([batchSize, seqLen, 1, config.audioVocabSize])
            print("DEBUG forward: head \(i) reshaped.shape=\(reshaped.shape)")
            logitsPerCodebook.append(reshaped)
        }
        let audioLogits = MLX.concatenated(logitsPerCodebook, axis: 2)  // [B, S, C, V]
        print("DEBUG forward returning audioLogits.shape=\(audioLogits.shape)")

        return audioLogits
    }

    // MARK: - SpeechGenerationModel Protocol

    /// Generate audio using the standard protocol (uses sensible defaults).
    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        // Use OmniVoice-specific defaults internally
        let ovParams = OmniVoiceGenerateParameters()
        return try await generateAudio(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            ovParameters: ovParams
        )
    }

    /// Generate audio with custom OmniVoice diffusion parameters.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice design instruction (e.g., "male, British accent") or nil for auto voice
    ///   - refAudio: Reference audio for voice cloning
    ///   - refText: Transcript of reference audio
    ///   - language: Language code
    ///   - ovParameters: OmniVoice-specific diffusion and generation parameters
    /// - Returns: Generated audio waveform at 24kHz
    public func generate(
        text: String,
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        language: String? = nil,
        ovParameters: OmniVoiceGenerateParameters
    ) async throws -> MLXArray {
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        return try await generateAudio(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            ovParameters: ovParameters
        )
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
        let ovParams = OmniVoiceGenerateParameters()
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                guard tokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
                }
                let audio = try await generateAudio(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    ovParameters: ovParams
                )
                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: 0,
                    prefillTime: 0,
                    generateTime: 0,
                    tokensPerSecond: 0,
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

    // MARK: - Generation

    private func generateAudio(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        ovParameters: OmniVoiceGenerateParameters
    ) async throws -> MLXArray {
        guard let audioTok = audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Audio tokenizer not loaded")
        }

        // 1. Encode reference audio to tokens if provided
        var refAudioTokens: MLXArray?
        if let refAudio {
            refAudioTokens = try audioTok.encode(refAudio)
            print("DEBUG refAudioTokens.shape=\(refAudioTokens!.shape)")
        }

        // 2. Estimate target token count
        let numRefTokCount = refAudioTokens?.shape.last ?? 0
        let numTargetTokens = estimateTargetTokens(
            text: text,
            refText: refText,
            numRefAudioTokens: numRefTokCount,
            speed: ovParameters.speed
        )

        // 3. Prepare inference inputs
        print("DEBUG preparing inference inputs with numTargetTokens=\(numTargetTokens)")
        let prepared = try prepareInferenceInputs(
            text: text,
            numTargetTokens: numTargetTokens,
            refText: refText,
            refAudioTokens: refAudioTokens,
            language: language,
            instruct: voice,
            denoise: ovParameters.denoise
        )
        print("DEBUG prepared inputIds.shape=\(prepared.inputIds.shape), audioMask.shape=\(prepared.audioMask.shape)")

        let inputIds = prepared.inputIds
        let audioMask = prepared.audioMask
        let condLength = inputIds.shape[2]

        // 4. Build batched inputs for CFG (cond + uncond)
        let B = 1
        let numCodebooks = config.numAudioCodebook
        let targetLen = numTargetTokens

        print("DEBUG inputIds.shape=\(inputIds.shape), condLength=\(condLength), targetLen=\(targetLen)")
        
        // Unconditional input: pad target with leading masks to match cond length
        // cond: [style, text, ref_audio, target] (length = condLength)
        // uncond: [mask...mask, target] (same length, but prefix is masked)
        let prefixLen = condLength - targetLen
        let prefixMask = MLXArray.full(
            [1, numCodebooks, prefixLen],
            values: MLXArray(Int32(config.audioMaskId))
        )
        let targetOnly = inputIds[0..., 0..., prefixLen...]
        let uncondInputIds = MLX.concatenated([prefixMask, targetOnly], axis: 2)
        let uncondAudioMaskPrefix = MLXArray.zeros([1, prefixLen], type: Bool.self)
        let uncondAudioMaskTarget = audioMask[0..., prefixLen...]
        let uncondAudioMask = MLX.concatenated([uncondAudioMaskPrefix, uncondAudioMaskTarget], axis: 1)
        print("DEBUG uncondInputIds.shape=\(uncondInputIds.shape)")

        // Concatenate cond + uncond along batch axis
        print("DEBUG concatenating inputIds and uncondInputIds along axis 0")
        let batchInputIds = MLX.concatenated([inputIds, uncondInputIds], axis: 0)
        let batchAudioMask = MLX.concatenated(
            [audioMask, uncondAudioMask],
            axis: 0
        )

        // 5. Initialize target tokens to all MASK
        print("DEBUG initializing tokens with shape [\(B), \(numCodebooks), \(targetLen)]")
        var tokens = MLXArray.full(
            [B, numCodebooks, targetLen],
            values: MLXArray(Int32(config.audioMaskId))
        )
        print("DEBUG tokens.shape=\(tokens.shape)")

        // 6. Compute timesteps and unmasking schedule
        let timesteps = getTimeSteps(tStart: 0.0, tEnd: 1.0, numStep: ovParameters.numStep + 1, tShift: ovParameters.tShift)

        let totalMask = targetLen * numCodebooks
        var rem = totalMask
        var schedule: [Int] = []
        for step in 0..<ovParameters.numStep {
            let k: Int
            if step == ovParameters.numStep - 1 {
                k = rem
            } else {
                let ceilVal = Int(ceil(Float(totalMask) * (timesteps[step + 1] - timesteps[step])))
                k = min(ceilVal, rem)
            }
            schedule.append(k)
            rem -= k
        }

        let layerIds = MLXArray((0..<numCodebooks).map { Int32($0) }).reshaped([1, numCodebooks, 1])

        // 7. Iterative diffusion generation
        for step in 0..<ovParameters.numStep {
            let k = schedule[step]
            if k <= 0 { continue }

            // Forward pass
            let logits = forward(
                inputIds: batchInputIds,
                audioMask: batchAudioMask
            ).asType(.float32)

            print("DEBUG logits.shape=\(logits.shape), condLength=\(condLength), targetLen=\(targetLen)")
            // Extract conditional and unconditional logits for the target region
            // logits shape: [B, S, C, V] = [2, 817, 9, 1025]
            let cLogits = logits[0, (condLength - targetLen)..<condLength, 0..., 0...]  // [T, C, V]
            let uLogits = logits[1, 0..<targetLen, 0..., 0...]  // [T, C, V]
            print("DEBUG cLogits.shape=\(cLogits.shape), uLogits.shape=\(uLogits.shape)")

            // Add batch dimension back for scoring
            // cLogits is [T, C, V], need [1, C, T, V]
            print("DEBUG cLogits transposing from \(cLogits.shape)")
            let cLogitsT = cLogits.transposed(1, 0, 2)  // [C, T, V]
            print("DEBUG cLogitsT.shape=\(cLogitsT.shape)")
            let cLogitsBatch = cLogitsT.reshaped([1, numCodebooks, targetLen, config.audioVocabSize])
            print("DEBUG cLogitsBatch.shape=\(cLogitsBatch.shape)")
            let uLogitsT = uLogits.transposed(1, 0, 2)  // [C, T, V]
            let uLogitsBatch = uLogitsT.reshaped([1, numCodebooks, targetLen, config.audioVocabSize])

            // Token prediction with CFG
            print("DEBUG calling predictTokensWithScoring")
            let (predTokens, scores) = predictTokensWithScoring(
                cLogits: cLogitsBatch,
                uLogits: uLogitsBatch,
                guidanceScale: ovParameters.guidanceScale,
                classTemperature: ovParameters.classTemperature
            )
            print("DEBUG predTokens.shape=\(predTokens.shape), scores.shape=\(scores.shape)")

            // Apply layer penalty
            print("DEBUG layerIds.shape=\(layerIds.shape)")
            let adjustedScores = scores - (layerIds.asType(.float32) * ovParameters.layerPenaltyFactor)

            // Gumbel sampling for position selection
            var finalScores = adjustedScores
            if ovParameters.positionTemperature > 0.0 {
                finalScores = gumbelSample(logits: adjustedScores, temperature: ovParameters.positionTemperature)
            }

            // Mask out already-filled positions
            print("DEBUG tokens.shape=\(tokens.shape), tokens[0].shape=\(tokens[0].shape)")
            let mask = tokens[0] .!= Int32(config.audioMaskId)
            print("DEBUG mask.shape=\(mask.shape)")
            let maskInf = MLX.where(mask, MLXArray(Float(-Float.infinity)), finalScores).asType(.float32)
            print("DEBUG maskInf.shape=\(maskInf.shape)")

            // Flatten for top-k selection
            let flatScores = maskInf.reshaped([-1])
            let flatTokens = tokens[0].reshaped([-1])
            let flatPreds = predTokens[0].reshaped([-1])
            print("DEBUG flatScores.shape=\(flatScores.shape), flatTokens.shape=\(flatTokens.shape), flatPreds.shape=\(flatPreds.shape)")

            // Select top-k positions to unmask
            let partitionInput = MLXArray(-1.0) * flatScores.asType(.float32)
            print("DEBUG partitionInput.shape=\(partitionInput.shape), k=\(k)")
            // Use argsort instead of argPartition for safety
            let sortedIndices = MLX.argSort(partitionInput, axis: 0)
            print("DEBUG sortedIndices.shape=\(sortedIndices.shape)")
            let topkIndices = sortedIndices[0..., ..<k]
            print("DEBUG topkIndices.shape=\(topkIndices.shape)")

            // Build update mask: positions in topkIndices get updated
            let linearTopkIndices = topkIndices.reshaped([-1])
            print("DEBUG linearTopkIndices.shape=\(linearTopkIndices.shape)")
            let updateMask = MLXArray.zeros([flatTokens.shape[0]], type: Bool.self)
            var updatedTokens = flatTokens

            // Mark positions to update
            let indicesArray = linearTopkIndices.asArray(Int.self)
            print("DEBUG indicesArray.count=\(indicesArray.count)")
            for idx in indicesArray {
                if idx >= 0 && idx < flatTokens.shape[0] {
                    updatedTokens[idx] = flatPreds[idx]
                }
            }

            print("DEBUG reshaping updatedTokens to [\(numCodebooks), \(targetLen)]")
            let reshapedTokens = updatedTokens.reshaped([numCodebooks, targetLen])
            print("DEBUG reshapedTokens.shape=\(reshapedTokens.shape)")
            tokens[0] = reshapedTokens

            // Update batch inputs for next step
            // Update cond: replace target region
            let condHead = batchInputIds[0, 0..., 0..., 0..<(condLength - targetLen)]
            let condUpdatedFull = MLX.concatenated(
                [condHead, tokens[0]],
                axis: 1
            )
            batchInputIds[0] = condUpdatedFull

            // Update uncond
            batchInputIds[1] = tokens[0]

            eval(tokens)
        }

        // 8. Decode tokens to waveform
        print("DEBUG tokens.shape=\(tokens.shape), slicing [0, 0..., 0..., 0..<\(targetLen)]")
        let outputTokens = tokens[0, 0..., 0..., 0..<targetLen]
        print("DEBUG outputTokens.shape=\(outputTokens.shape)")
        let audio = try audioTok.decode(outputTokens)

        // 9. Post-process
        return postProcessAudio(audio, refRms: nil, postprocessOutput: ovParameters.postprocessOutput)
    }

    // MARK: - Token Prediction with CFG

    private func predictTokensWithScoring(
        cLogits: MLXArray,
        uLogits: MLXArray,
        guidanceScale: Float,
        classTemperature: Float
    ) -> (MLXArray, MLXArray) {
        let predTokens: MLXArray
        let scores: MLXArray

        if guidanceScale != 0 {
            let cLogProbs = logSoftmax(cLogits, axis: -1)
            let uLogProbs = logSoftmax(uLogits, axis: -1)
            let combinedLogProbs = cLogProbs + guidanceScale * (cLogProbs - uLogProbs)
            var logProbs = logSoftmax(combinedLogProbs, axis: -1)

            // Mask out the audio_mask_id
            let maskIdOnehot = MLXArray((0..<config.audioVocabSize).map { $0 == config.audioMaskId ? Float(-Float.infinity) : Float(0) })
            let maskArr = MLXArray.ones(logProbs.shape) * maskIdOnehot.reshaped([1, 1, 1, -1])
            logProbs = logProbs + maskArr

            if classTemperature > 0.0 {
                let filteredLogProbs = filterTopK(logits: logProbs, ratio: 0.1)
                let sampled = gumbelSample(logits: filteredLogProbs, temperature: classTemperature)
                predTokens = MLX.argMax(sampled, axis: -1)
            } else {
                predTokens = MLX.argMax(logProbs, axis: -1)
            }
            scores = logProbs.max(axis: -1)
        } else {
            var logProbs = logSoftmax(cLogits, axis: -1)
            let maskIdOnehot = MLXArray((0..<config.audioVocabSize).map { $0 == config.audioMaskId ? Float(-Float.infinity) : Float(0) })
            let maskArr = MLXArray.ones(logProbs.shape) * maskIdOnehot.reshaped([1, 1, 1, -1])
            logProbs = logProbs + maskArr

            if classTemperature > 0.0 {
                let filteredLogProbs = filterTopK(logits: logProbs, ratio: 0.1)
                let sampled = gumbelSample(logits: filteredLogProbs, temperature: classTemperature)
                predTokens = MLX.argMax(sampled, axis: -1)
            } else {
                predTokens = MLX.argMax(logProbs, axis: -1)
            }
            scores = logProbs.max(axis: -1)
        }

        return (predTokens, scores)
    }

    // MARK: - Utility Functions

    private func filterTopK(logits: MLXArray, ratio: Float) -> MLXArray {
        let k = max(1, Int(ceil(ratio * Float(logits.shape[-1]))))
        // Use argSort to get top-k indices
        let sortedIndices = MLX.argSort(-logits, axis: -1)
        let topIndices = sortedIndices[0..., 0..., 0..., 0..<k]
        let topVals = MLX.takeAlong(logits, topIndices, axis: -1)

        var filtered = MLXArray.full(logits.shape, values: MLXArray(Float(-Float.infinity)))
        // Fill in top-k values by iterating (simple approach)
        let batchSize = logits.shape[0]
        let seqLen = logits.shape[1]
        let numCodebooks = logits.shape[2]

        for b in 0..<batchSize {
            for c in 0..<numCodebooks {
                for s in 0..<seqLen {
                    for ki in 0..<k {
                        let idx = topIndices[b, c, s, ki].item(Int.self)
                        let val = topVals[b, c, s, ki]
                        filtered[b, c, s, idx] = val
                    }
                }
            }
        }
        return filtered
    }

    private func gumbelSample(logits: MLXArray, temperature: Float) -> MLXArray {
        let scaledLogits = logits / temperature
        let u = MLXRandom.uniform(low: Float(1e-10), high: 1.0, scaledLogits.shape)
        let gumbelNoise = -MLX.log(-MLX.log(u + 1e-10) + 1e-10)
        return scaledLogits + gumbelNoise
    }

    private func getTimeSteps(tStart: Float, tEnd: Float, numStep: Int, tShift: Float) -> [Float] {
        var steps: [Float] = []
        for i in 0...numStep {
            let t = tStart + (tEnd - tStart) * Float(i) / Float(numStep)
            let shifted = tShift * t / (1.0 + (tShift - 1.0) * t)
            steps.append(shifted)
        }
        return steps
    }

    private func makeCausalMask(length: Int) -> MLXArray {
        // Lower triangular matrix
        let rowIdx = MLXArray((0..<length).map { Int32($0) }).reshaped([length, 1])
        let colIdx = MLXArray((0..<length).map { Int32($0) }).reshaped([1, length])
        return (rowIdx .>= colIdx)
    }

    private func estimateTargetTokens(
        text: String,
        refText: String?,
        numRefAudioTokens: Int,
        speed: Float
    ) -> Int {
        // Simple heuristic: ~4 tokens per character
        let chars = text.count
        let baseEstimate = max(25, chars * 4)
        let adjusted = speed > 0 && speed != 1.0 ? Int(Float(baseEstimate) / speed) : baseEstimate
        return max(1, adjusted)
    }

    private func prepareInferenceInputs(
        text: String,
        numTargetTokens: Int,
        refText: String?,
        refAudioTokens: MLXArray?,
        language: String?,
        instruct: String?,
        denoise: Bool
    ) throws -> (inputIds: MLXArray, audioMask: MLXArray) {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let numCodebooks = config.numAudioCodebook
        print("DEBUG prepareInferenceInputs: numCodebooks=\(numCodebooks)")

        // Build style tokens
        var styleText = ""
        if denoise && refAudioTokens != nil {
            styleText += "<|denoise|>"
        }
        let langStr = language ?? "None"
        styleText += "<|lang_start|>\(langStr)<|lang_end|>"
        let instructStr = instruct ?? "None"
        styleText += "<|instruct_start|>\(instructStr)<|instruct_end|>"

        let styleTokenIds = try tokenizeText(styleText)
        print("DEBUG styleTokenIds.count=\(styleTokenIds.count)")
        var styleIds = MLXArray(styleTokenIds.map { Int32($0) })
        styleIds = styleIds.reshaped([1, -1])
        print("DEBUG styleIds before broadcast.shape=\(styleIds.shape)")
        print("DEBUG broadcasting styleIds to [1, \(numCodebooks), \(styleIds.shape[1])]")
        styleIds = MLX.broadcast(styleIds.reshaped([1, 1, -1]), to: [1, numCodebooks, styleIds.shape[1]])

        // Build text tokens
        let fullText = combineText(refText: refText, text: text)
        let wrappedText = "<|text_start|>\(fullText)<|text_end|>"
        let textTokenIds = try tokenizeText(wrappedText)
        print("DEBUG textTokenIds.count=\(textTokenIds.count)")
        var textIds = MLXArray(textTokenIds.map { Int32($0) })
        textIds = textIds.reshaped([1, -1])
        print("DEBUG textIds before broadcast.shape=\(textIds.shape)")
        print("DEBUG broadcasting textIds to [1, \(numCodebooks), \(textIds.shape[1])]")
        textIds = MLX.broadcast(textIds.reshaped([1, 1, -1]), to: [1, numCodebooks, textIds.shape[1]])

        // Target: all MASK
        print("DEBUG creating targetIds with shape [1, \(numCodebooks), \(numTargetTokens)]")
        let targetIds = MLXArray.full(
            [1, numCodebooks, numTargetTokens],
            values: MLXArray(Int32(config.audioMaskId))
        )
        print("DEBUG targetIds.shape=\(targetIds.shape)")

        // Concatenate: [style, text, ref_audio (optional), target]
        var parts: [MLXArray] = [styleIds, textIds]
        print("DEBUG styleIds.shape=\(styleIds.shape), textIds.shape=\(textIds.shape)")
        if let refTok = refAudioTokens {
            print("DEBUG refTok.shape=\(refTok.shape)")
            let reshaped = refTok.reshaped([1, refTok.shape[0], refTok.shape[1]])
            print("DEBUG reshaped refTok.shape=\(reshaped.shape)")
            parts.append(reshaped)
        }
        print("DEBUG targetIds.shape=\(targetIds.shape)")
        parts.append(targetIds)

        print("DEBUG concatenating \(parts.count) parts along axis 2")
        let condInputIds = MLX.concatenated(parts, axis: 2)
        let totalLength = condInputIds.shape[2]

        // Build audio mask: true for ref_audio + target regions
        let audioStartIdx: Int
        if refAudioTokens != nil {
            let refTokLen = refAudioTokens!.shape[1]
            audioStartIdx = totalLength - numTargetTokens - refTokLen
        } else {
            audioStartIdx = totalLength - numTargetTokens
        }

        print("DEBUG condInputIds.shape=\(condInputIds.shape)")
        
        var condAudioMask = MLXArray.zeros([1, totalLength], type: Bool.self)
        print("DEBUG condAudioMask.shape before assignment=\(condAudioMask.shape)")
        print("DEBUG audioStartIdx=\(audioStartIdx), totalLength=\(totalLength)")
        let onesArray = MLXArray.ones([totalLength - audioStartIdx], type: Bool.self)
        print("DEBUG onesArray.shape=\(onesArray.shape)")
        condAudioMask[0..., audioStartIdx...] = onesArray
        print("DEBUG condAudioMask.shape after assignment=\(condAudioMask.shape)")

        return (condInputIds, condAudioMask)
    }

    private func tokenizeText(_ text: String) throws -> [Int] {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        return try tokenizer.encode(text: text, addSpecialTokens: false)
    }

    private func combineText(refText: String?, text: String) -> String {
        var fullText = ""
        if let refText, !refText.isEmpty {
            fullText = refText.trimmingCharacters(in: .whitespacesAndNewlines) + " "
        }
        fullText += text.trimmingCharacters(in: .whitespacesAndNewlines)
        fullText = fullText.components(separatedBy: .newlines).joined(separator: " ")
        fullText = fullText.replacingOccurrences(of: "  ", with: " ", options: .regularExpression)
        return fullText
    }

    private func postProcessAudio(_ audio: MLXArray, refRms: Float?, postprocessOutput: Bool) -> MLXArray {
        var result = audio

        if let refRms, refRms < 0.1 {
            result = result * MLXArray(refRms / 0.1)
        } else if refRms == nil {
            let peak = MLX.abs(result).max().item(Float.self)
            if peak > 1e-6 {
                result = result * MLXArray(0.5 / peak)
            }
        }

        if postprocessOutput {
            let len = result.shape[0]
            let fadeLen = min(480, len / 2)
            if fadeLen > 0 {
                let fadeIn = MLXArray((0..<fadeLen).map { Float($0) / Float(fadeLen) })
                let fadeOut = MLXArray((0..<fadeLen).reversed().map { Float($0) / Float(fadeLen) })
                result[0..<fadeLen] = result[0..<fadeLen] * fadeIn
                result[(len - fadeLen)...] = result[(len - fadeLen)...] * fadeOut
            }
        }

        eval(result)
        return result
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        _ repoID: String,
        cache: HubCache = .default
    ) async throws -> OmniVoiceModel {
        guard let repo = Repo.ID(rawValue: repoID) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(repoID)")
        }

        // Download and parse config
        let configURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: "json",
            additionalMatchingPatterns: ["config.json"]
        ).appendingPathComponent("config.json")

        let configData = try Data(contentsOf: configURL)
        
        // Load tokenizer config first to get correct n_codebooks
        let tokenizerConfigURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: "json",
            additionalMatchingPatterns: ["audio_tokenizer/config.json"]
        ).appendingPathComponent("audio_tokenizer/config.json")
        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfig = try JSONDecoder().decode(OmniVoiceAudioTokenizerConfig.self, from: tokenizerConfigData)
        let correctNumCodebooks = tokenizerConfig.nCodebooks
        
        // Parse and modify main config to use correct num_codebooks
        var configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        if let currentNum = configDict["num_audio_codebook"] as? Int, currentNum != correctNumCodebooks {
            print("DEBUG: overriding num_audio_codebook from \(currentNum) to \(correctNumCodebooks) to match tokenizer")
            configDict["num_audio_codebook"] = correctNumCodebooks
            // Also update audio_codebook_weights if needed
            if let weights = configDict["audio_codebook_weights"] as? [Int], weights.count != correctNumCodebooks {
                var newWeights = weights
                while newWeights.count < correctNumCodebooks {
                    newWeights.append(newWeights.last ?? 2)
                }
                configDict["audio_codebook_weights"] = newWeights
            }
        }
        let modifiedConfigData = try JSONSerialization.data(withJSONObject: configDict)
        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: modifiedConfigData)

        let model = try OmniVoiceModel(config: config)

        // Load weights
        let weightsURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["model.safetensors"]
        ).appendingPathComponent("model.safetensors")

        let weights = try MLX.loadArrays(url: weightsURL)
        let sanitizedWeights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .noUnusedKeys)
        eval(model)

        // Load text tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: {
            let dir = try await ModelUtils.resolveOrDownloadModel(
                repoID: repo,
                requiredExtension: "json",
                additionalMatchingPatterns: ["tokenizer.json"]
            )
            return dir
        }())

        // Load audio tokenizer
        model.audioTokenizer = try await OmniVoiceAudioTokenizer.fromPretrained(
            repoID: repoID,
            cache: cache
        )

        return model
    }

    // MARK: - Weight Sanitization

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("audio_embeddings.") || key.hasPrefix("audio_heads.") {
                sanitized[key] = value
            } else if key == "lm_head.weight" {
                // lm_head lives on Qwen3Model, not Qwen3ModelInner
                sanitized["llm.lm_head.weight"] = value
            } else if key.hasPrefix("model.") {
                // model.X -> llm.model.X
                let stripped = String(key.dropFirst("model.".count))
                sanitized["llm.model.\(stripped)"] = value
            } else if key.hasPrefix("backbone.") {
                // backbone.X -> llm.model.X
                let stripped = String(key.dropFirst("backbone.".count))
                sanitized["llm.model.\(stripped)"] = value
            } else if key.hasPrefix("llm.") {
                // llm.X -> llm.model.X
                let stripped = String(key.dropFirst(4))
                sanitized["llm.model.\(stripped)"] = value
            } else {
                // Bare key -> llm.model.X
                sanitized["llm.model.\(key)"] = value
            }
        }

        return sanitized
    }
}

// MARK: - Quantizer modules matching Higgs Audio V2 checkpoint

/// Codebook embedding: stores the quantization codebook.
final class OmniVoiceQuantizerCodebook: Module {
    @ModuleInfo(key: "embed") var embed: MLXArray  // [codebook_size, codebook_dim]

    init(codebookSize: Int, codebookDim: Int) {
        self._embed.wrappedValue = MLXRandom.uniform(
            low: -1.0, high: 1.0, [codebookSize, codebookDim]
        )
    }
}

/// Single quantizer block: projects input → codebook dim, quantizes, projects back.
final class OmniVoiceSingleQuantizer: Module {
    @ModuleInfo(key: "codebook") var codebook: OmniVoiceQuantizerCodebook
    @ModuleInfo(key: "project_in") var projectIn: MLXNN.Linear
    @ModuleInfo(key: "project_out") var projectOut: MLXNN.Linear

    init(inputDim: Int, outputDim: Int, codebookSize: Int, codebookDim: Int) {
        self._codebook.wrappedValue = OmniVoiceQuantizerCodebook(
            codebookSize: codebookSize, codebookDim: codebookDim
        )
        self._projectIn.wrappedValue = MLXNN.Linear(
            inputDimensions: inputDim, outputDimensions: codebookDim
        )
        self._projectOut.wrappedValue = MLXNN.Linear(
            inputDimensions: codebookDim, outputDimensions: outputDim
        )
    }
}

// MARK: - OmniVoice ConvTranspose1d (PyTorch weight convention)

/// ConvTranspose1d using MLX weight layout [in_channels, kernel_size, out_channels].
final class OmniVoiceConvTranspose1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let strideVal: Int
    let paddingVal: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.strideVal = stride
        self.paddingVal = padding

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        // MLX format: [in_channels, kernel_size, out_channels]
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [inChannels, kernelSize, outChannels]
        )
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Data flows in NCL [B, C, L]; transpose to NLC for MLX convTransposed1d
        let xNLC = x.transposed(0, 2, 1)
        var h = MLX.convTransposed1d(xNLC, weight, stride: strideVal, padding: paddingVal)
        if let b = bias {
            let n = b.size
            h = h + b.reshaped([n])
        }
        // Convert back to NCL [B, C, L]
        let out = h.transposed(0, 2, 1)
        print("DEBUG convTranspose1d: in=\(x.shape) w=\(weight.shape) pad=\(paddingVal) out=\(out.shape)")
        return out
    }
}

/// Conv1d using MLX weight layout [out_channels, kernel_size, in_channels].
final class OmniVoiceConv1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let strideVal: Int
    let paddingVal: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.strideVal = stride
        self.paddingVal = padding

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        // MLX format: [out_channels, kernel_size, in_channels]
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outChannels, kernelSize, inChannels]
        )
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Weight stored as [out, in, kernel] (PyTorch) → transpose to [out, kernel, in] (MLX)
        let w = weight.transposed(0, 2, 1)
        // Data flows in NCL [B, C, L] format; transpose to NLC for MLX conv1d, then back
        let xNLC = x.transposed(0, 2, 1)
        var h = MLX.conv1d(xNLC, w, stride: strideVal, padding: paddingVal)
        if let b = bias {
            let n = b.size
            h = h + b.reshaped([n])
        }
        // Convert back to NCL [B, C, L]
        let out = h.transposed(0, 2, 1)
        return out
    }
}

// MARK: - DAC-style Audio Codec

/// Snake activation: x + (1/a) * sin(a*x)^2
func snakeActivation(_ x: MLXArray) -> MLXArray {
    let alpha: Float = 1.0
    return x + (1.0 / alpha) * MLX.square(MLX.sin(alpha * x))
}

/// DAC-style residual unit with Snake activations.
public final class OmniVoiceDACResidualUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d
    @ModuleInfo(key: "conv2") var conv2: OmniVoiceConv1d
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "snake2") var snake2: snakeAlpha

    init(channels: Int, kernelSize: Int, dilation: Int) {
        // Standard "same" padding for stride=1
        // output_length = input_length - kernel + 2*padding + 1
        // For same: padding = (kernel - 1) * dilation / 2
        let padding = (kernelSize - 1) * dilation / 2

        self._conv1.wrappedValue = OmniVoiceConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding
        )
        self._conv2.wrappedValue = OmniVoiceConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding
        )
        self._snake1.wrappedValue = snakeAlpha(channels: channels)
        self._snake2.wrappedValue = snakeAlpha(channels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        print("DEBUG residual unit input x.shape=\(x.shape)")
        let c1 = conv1(x)
        print("DEBUG residual unit after conv1 c1.shape=\(c1.shape)")
        let s1 = snake1.callAsFunction(c1)
        print("DEBUG residual unit after snake1 s1.shape=\(s1.shape)")
        let c2 = conv2(s1)
        print("DEBUG residual unit after conv2 c2.shape=\(c2.shape)")
        let h = snake2.callAsFunction(c2)
        print("DEBUG residual unit after snake2 h.shape=\(h.shape)")
        
        // Handle potential length mismatch for residual connection
        let xLen = x.shape[2]
        let hLen = h.shape[2]
        let minLen = min(xLen, hLen)
        var xTrimmed = x
        var hTrimmed = h
        if xLen != hLen {
            let xPad = (xLen - minLen) / 2
            let hPad = (hLen - minLen) / 2
            xTrimmed = x[0..., 0..., xPad..<(xLen - xPad)]
            hTrimmed = h[0..., 0..., hPad..<(hLen - hPad)]
        }
        return xTrimmed + hTrimmed
    }
}

/// Learnable Snake activation parameter.
public final class snakeAlpha: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, channels, 1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = MLX.square(MLX.log(1.0 + MLX.exp(alpha)))
        return x + (1.0 / a) * MLX.square(MLX.sin(a * x))
    }
}

/// DAC downsampling block (Higgs Audio V2 EncoderBlock):
/// 3 ResidualUnits(dilation 1,3,9) + Snake1d + WNConv1d.
public final class OmniVoiceDACDownBlock: Module {
    @ModuleInfo var res_unit1: OmniVoiceDACResidualUnit
    @ModuleInfo var res_unit2: OmniVoiceDACResidualUnit
    @ModuleInfo var res_unit3: OmniVoiceDACResidualUnit
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d

    init(inputChannels: Int, outputChannels: Int, stride: Int, kernelSize: Int) {
        self._res_unit1.wrappedValue = OmniVoiceDACResidualUnit(
            channels: inputChannels, kernelSize: kernelSize, dilation: 1
        )
        self._res_unit2.wrappedValue = OmniVoiceDACResidualUnit(
            channels: inputChannels, kernelSize: kernelSize, dilation: 3
        )
        self._res_unit3.wrappedValue = OmniVoiceDACResidualUnit(
            channels: inputChannels, kernelSize: kernelSize, dilation: 9
        )
        self._snake1.wrappedValue = snakeAlpha(channels: inputChannels)
        self._conv1.wrappedValue = OmniVoiceConv1d(
            inChannels: inputChannels,
            outChannels: outputChannels,
            kernelSize: stride * 2,
            stride: stride,
            padding: stride / 2 + stride % 2
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = res_unit1(x)
        h = res_unit2(h)
        h = res_unit3(h)
        h = snake1.callAsFunction(h)
        h = conv1(h)
        return h
    }
}

/// DAC upsampling block (Higgs Audio V2 DecoderBlock):
/// Snake1d + ConvTranspose1d + 3 ResidualUnits(dilation 1,3,9).
public final class OmniVoiceDACUpBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "conv_t1") var convT1: OmniVoiceConvTranspose1d
    @ModuleInfo var res_unit1: OmniVoiceDACResidualUnit
    @ModuleInfo var res_unit2: OmniVoiceDACResidualUnit
    @ModuleInfo var res_unit3: OmniVoiceDACResidualUnit

    init(inputChannels: Int, outputChannels: Int, stride: Int, kernelSize: Int) {
        self._snake1.wrappedValue = snakeAlpha(channels: outputChannels)
        self._convT1.wrappedValue = OmniVoiceConvTranspose1d(
            inChannels: inputChannels,
            outChannels: outputChannels,
            kernelSize: stride * 2,
            stride: stride,
            padding: stride / 2 + stride % 2
        )
        self._res_unit1.wrappedValue = OmniVoiceDACResidualUnit(
            channels: outputChannels, kernelSize: kernelSize, dilation: 1
        )
        self._res_unit2.wrappedValue = OmniVoiceDACResidualUnit(
            channels: outputChannels, kernelSize: kernelSize, dilation: 3
        )
        self._res_unit3.wrappedValue = OmniVoiceDACResidualUnit(
            channels: outputChannels, kernelSize: kernelSize, dilation: 9
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1.callAsFunction(convT1(x))
        h = res_unit1(h)
        h = res_unit2(h)
        h = res_unit3(h)
        return h
    }
}

/// Higgs Audio V2 acoustic encoder: conv1 → down blocks → snake1 → conv2.
public final class OmniVoiceDACAcousticEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d
    @ModuleInfo var block: [OmniVoiceDACDownBlock]
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "conv2") var conv2: OmniVoiceConv1d

    init(config: OmniVoiceAudioTokenizerConfig) {
        let hiddenSize = config.encoderHiddenSize
        let downsamplingRatios = config.downsamplingRatios

        self._conv1.wrappedValue = OmniVoiceConv1d(
            inChannels: 1,
            outChannels: hiddenSize,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )

        var blocks: [OmniVoiceDACDownBlock] = []
        var currentChannels = hiddenSize
        for stride in downsamplingRatios {
            let outChannels = currentChannels * 2
            blocks.append(OmniVoiceDACDownBlock(
                inputChannels: currentChannels,
                outputChannels: outChannels,
                stride: stride,
                kernelSize: config.kernelSize
            ))
            currentChannels = outChannels
        }
        self._block.wrappedValue = blocks

        self._snake1.wrappedValue = snakeAlpha(channels: currentChannels)
        print("DEBUG encoder init: currentChannels=\(currentChannels), conv2 will output \(currentChannels / 32)")
        self._conv2.wrappedValue = OmniVoiceConv1d(
            inChannels: currentChannels,
            outChannels: currentChannels / 32,  // 2048 → 64 to match checkpoint
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block {
            h = b(h)
        }
        h = snake1.callAsFunction(h)
        print("DEBUG encoder: before conv2 h.shape=\(h.shape)")
        h = conv2(h)
        print("DEBUG encoder: after conv2 h.shape=\(h.shape)")
        return h
    }
}

/// Higgs Audio V2 acoustic decoder: conv1 → up blocks → snake1 → conv2.
public final class OmniVoiceDACAcousticDecoder: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d
    @ModuleInfo var block: [OmniVoiceDACUpBlock]
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "conv2") var conv2: OmniVoiceConv1d

    init(config: OmniVoiceAudioTokenizerConfig) {
        let hiddenSize = config.decoderHiddenSize
        let upsamplingRatios = config.upsamplingRatios

        // Initial projection to decoder hidden size
        self._conv1.wrappedValue = OmniVoiceConv1d(
            inChannels: config.encoderHiddenSize * 4,  // 256
            outChannels: hiddenSize,                     // 1024
            kernelSize: 7,
            stride: 1,
            padding: 3
        )

        var blocks: [OmniVoiceDACUpBlock] = []
        var currentChannels = hiddenSize
        for stride in upsamplingRatios {
            let outChannels = currentChannels / 2
            blocks.append(OmniVoiceDACUpBlock(
                inputChannels: currentChannels,
                outputChannels: outChannels,
                stride: stride,
                kernelSize: config.kernelSize
            ))
            currentChannels = outChannels
        }
        self._block.wrappedValue = blocks

        self._snake1.wrappedValue = snakeAlpha(channels: currentChannels)
        self._conv2.wrappedValue = OmniVoiceConv1d(
            inChannels: currentChannels,
            outChannels: 1,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block {
            h = b(h)
        }
        h = snake1.callAsFunction(h)
        h = MLX.tanh(conv2(h))
        return h
    }
}

/// Residual Vector Quantization with projection layers (Higgs Audio V2 style).
public final class OmniVoiceRVQQuantizer: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [OmniVoiceSingleQuantizer]
    @ModuleInfo(key: "pre_project") var preProject: MLXNN.Linear
    let outputDim: Int

    init(config: OmniVoiceAudioTokenizerConfig) {
        let nQuantizers = config.nCodebooks
        let codebookSize = config.codebookSize
        let codebookDim = config.codebookDim
        // From checkpoint:
        // - encoder.conv2 outputs 256 channels
        // - quantizer.project_in expects 1024 input (weight shape [64, 1024])
        let encoderOutputDim = 256  // Actual encoder output
        let inputDim = 1024  // What project_in expects
        self.outputDim = config.decoderHiddenSize     // 1024

        // Pre-projection from encoder output to quantizer input
        self._preProject.wrappedValue = MLXNN.Linear(
            inputDimensions: encoderOutputDim,
            outputDimensions: inputDim
        )

        var qs: [OmniVoiceSingleQuantizer] = []
        for _ in 0..<nQuantizers {
            qs.append(OmniVoiceSingleQuantizer(
                inputDim: inputDim,
                outputDim: outputDim,
                codebookSize: codebookSize,
                codebookDim: codebookDim
            ))
        }
        self._quantizers.wrappedValue = qs
    }

    /// Quantize: [B, D, T] -> (codes [B, n_quantizers, T], quantized [B, outputDim, T])
    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        print("DEBUG RVQQuantizer input z.shape=\(z.shape)")
        
        // Pre-project from encoder output (256) to quantizer input (1024)
        // z is [B, C, L] (NCL), need [B, L, C] for Linear
        let zNLC = z.transposed(0, 2, 1)  // [B, L, 256]
        let zProjectedNLC = preProject(zNLC)  // [B, L, 1024]
        let zProjected = zProjectedNLC.transposed(0, 2, 1)  // [B, 1024, L]
        print("DEBUG RVQQuantizer after preProject: zProjected.shape=\(zProjected.shape)")
        
        let batchSize = zProjected.shape[0]
        let seqLen = zProjected.shape[2]
        let nQuantizers = quantizers.count

        var allCodes: [MLXArray] = []
        var quantized = MLXArray.zeros([batchSize, outputDim, seqLen])

        for qIdx in 0..<nQuantizers {
            print("DEBUG quantizer qIdx=\(qIdx)")
            let q = quantizers[qIdx]
            let codebook = q.codebook.embed
            let codebookSize = codebook.shape[0]
            let codebookDim = codebook.shape[1]
            print("DEBUG codebook.shape=\(codebook.shape)")

            // Project input to codebook dimension
            // zProjected is [B, C, L] (NCL) = [1, 1024, 332], Linear expects [B, L, C] (NLC)
            let zNLC = zProjected.transposed(0, 2, 1)  // [B, L, C] = [1, 332, 1024]
            let zProjNLC = q.projectIn(zNLC)  // [B, L, codebookDim] = [1, 332, 64]
            let zProj = zProjNLC.transposed(0, 2, 1)  // [B, codebookDim, L] = [1, 64, 332]
            print("DEBUG zProj.shape=\(zProj.shape)")

            // [B, codebookDim, T] -> [B, T, codebookDim] for distance computation
            let zPermute = zProj.transposed(0, 2, 1)
            let zFlat = zPermute.reshaped([batchSize * seqLen, codebookDim])
            print("DEBUG zFlat.shape=\(zFlat.shape), codebookSize=\(codebookSize), codebookDim=\(codebookDim)")

            // Compute distances: [B*T, K]
            let diff = zFlat.reshaped([zFlat.shape[0], 1, codebookDim])
                - codebook.reshaped([1, codebookSize, codebookDim])
            let dist = MLX.sum(diff * diff, axis: -1)

            // Nearest codebook index
            let codes = MLX.argMin(dist, axis: -1)  // [B*T]
            let codes2d = codes.reshaped([batchSize, seqLen])
            allCodes.append(codes2d)

            // Gather quantized vectors and project to output dimension
            let qVecs = MLX.take(codebook, codes, axis: 0)  // [B*T, codebookDim]
            print("DEBUG qIdx=\(qIdx): qVecs.shape=\(qVecs.shape)")
            let qOut = q.projectOut(qVecs)  // [B*T, outputDim]
            print("DEBUG qIdx=\(qIdx): qOut.shape=\(qOut.shape), outputDim=\(outputDim)")
            let q3d = qOut.reshaped([batchSize, seqLen, -1]).transposed(0, 2, 1)
            print("DEBUG qIdx=\(qIdx): q3d.shape=\(q3d.shape), quantized.shape=\(quantized.shape)")

            quantized = quantized + q3d
        }

        print("DEBUG allCodes.count=\(allCodes.count)")
        for (i, c) in allCodes.enumerated() {
            print("DEBUG allCodes[\(i)].shape=\(c.shape)")
        }
        let codes = MLX.stacked(allCodes, axis: 1)  // [B, n_quantizers, T]
        print("DEBUG stacked codes.shape=\(codes.shape)")
        return (codes, quantized)
    }

    /// Decode: [B, n_quantizers, T] -> [B, outputDim, T]
    func decode(_ codes: MLXArray) -> MLXArray {
        let batchSize = codes.shape[0]
        let nQuantizers = codes.shape[1]
        let seqLen = codes.shape[2]

        var quantized = MLXArray.zeros([batchSize, outputDim, seqLen])

        for qIdx in 0..<nQuantizers {
            let q = quantizers[qIdx]
            let codebook = q.codebook.embed
            let cbCodes = codes[0..., qIdx, 0...]  // [B, T]
            let flatCodes = cbCodes.reshaped([-1])  // [B*T]

            let qVecs = MLX.take(codebook, flatCodes, axis: 0)
            let qOut = q.projectOut(qVecs)
            let q3d = qOut.reshaped([batchSize, seqLen, -1]).transposed(0, 2, 1)

            quantized = quantized + q3d
        }

        return quantized
    }
}

// MARK: - OmniVoice Higgs Audio Tokenizer

/// Audio tokenizer for OmniVoice: DAC encoder/decoder with RVQ quantization.
public final class OmniVoiceAudioTokenizer: Module {
    let config: OmniVoiceAudioTokenizerConfig

    @ModuleInfo(key: "acoustic_encoder") var acousticEncoder: OmniVoiceDACAcousticEncoder
    @ModuleInfo(key: "acoustic_decoder") var acousticDecoder: OmniVoiceDACAcousticDecoder
    @ModuleInfo(key: "quantizer") var quantizer: OmniVoiceRVQQuantizer
    @ModuleInfo(key: "fc2") var fc2: MLXNN.Linear

    init(config: OmniVoiceAudioTokenizerConfig) {
        self.config = config

        self._acousticEncoder.wrappedValue = OmniVoiceDACAcousticEncoder(config: config)
        self._acousticDecoder.wrappedValue = OmniVoiceDACAcousticDecoder(config: config)
        self._quantizer.wrappedValue = OmniVoiceRVQQuantizer(config: config)

        // fc2 projects quantized features to decoder input
        self._fc2.wrappedValue = MLXNN.Linear(
            inputDimensions: config.codebookDim,
            outputDimensions: config.decoderHiddenSize
        )
    }

    /// Encode audio waveform to discrete tokens.
    /// - Parameter audio: [samples] or [1, samples]
    /// - Returns: [num_codebooks, seq_len]
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        var wav = audio
        if wav.ndim == 1 {
            // [T] -> [1, 1, T]  (batch, channels, length) NCL
            wav = wav.reshaped([1, 1, -1])
        } else if wav.ndim == 2 {
            // [B, T] -> [B, 1, T]
            wav = wav.reshaped([wav.shape[0], 1, wav.shape[1]])
        } else if wav.ndim == 3 && wav.shape[1] > wav.shape[2] {
            // NLC [B, L, C] -> NCL [B, C, L]
            wav = wav.transposed(0, 2, 1)
        }
        print("DEBUG encode: wav.shape=\(wav.shape)")

        // Encoder: [B, 1, T] -> [B, D, T']
        let z = acousticEncoder(wav)
        print("DEBUG encode: acousticEncoder output z.shape=\(z.shape)")

        // RVQ: [B, D, T'] -> (codes [B, n_codebooks, T'], quantized [B, D, T'])
        let (codes, _) = quantizer(z)

        // Return [n_codebooks, T'] (squeeze batch dim)
        print("DEBUG encode returning codes[0].shape=\(codes[0].shape)")
        return codes[0]
    }

    /// Decode discrete tokens back to audio waveform.
    /// - Parameter tokens: [num_codebooks, seq_len]
    /// - Returns: [samples]
    public func decode(_ tokens: MLXArray) throws -> MLXArray {
        print("DEBUG decode input tokens.shape=\(tokens.shape)")
        // Add batch dim: [n_codebooks, T] -> [1, n_codebooks, T]
        let batchedTokens = tokens.reshaped([1, tokens.shape[0], tokens.shape[1]])
        print("DEBUG decode batchedTokens.shape=\(batchedTokens.shape)")

        // RVQ decode: [1, n_codebooks, T] -> [1, D, T]
        let z = quantizer.decode(batchedTokens)
        print("DEBUG decode z.shape=\(z.shape)")

        // fc2 project: [1, D, T] -> [1, D', T]
        print("DEBUG decode fc2 input shape=\(z.transposed(0, 2, 1).shape)")
        let h = fc2(z.transposed(0, 2, 1)).transposed(0, 2, 1)
        print("DEBUG decode h.shape=\(h.shape)")

        // Decoder: [1, D', T] -> [1, 1, T']
        print("DEBUG decode calling acousticDecoder with h.shape=\(h.shape)")
        let audio = acousticDecoder(h)
        print("DEBUG decode audio.shape=\(audio.shape)")

        return audio.reshaped([-1])
    }

    public static func fromPretrained(
        repoID: String,
        cache: HubCache = .default
    ) async throws -> OmniVoiceAudioTokenizer {
        guard let repo = Repo.ID(rawValue: repoID) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(repoID)")
        }

        let configURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: "json",
            additionalMatchingPatterns: ["audio_tokenizer/config.json"]
        ).appendingPathComponent("audio_tokenizer/config.json")

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(OmniVoiceAudioTokenizerConfig.self, from: configData)

        let tokenizer = OmniVoiceAudioTokenizer(config: config)

        let weightsURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["audio_tokenizer/model.safetensors"]
        ).appendingPathComponent("audio_tokenizer/model.safetensors")

        let weights = try MLX.loadArrays(url: weightsURL)
        // Debug: print shapes of key weights
        for (key, value) in weights {
            if key.contains("project_in") || key.contains("conv2") || key.contains("acoustic_encoder") || key.contains("fc2") {
                print("DEBUG checkpoint: \(key) shape=\(value.shape)")
            }
        }
        try tokenizer.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        // Check fc2 weight after loading
        print("DEBUG after loading: fc2 weight shape=\(tokenizer.fc2.weight.shape)")
        eval(tokenizer)

        return tokenizer
    }
}

