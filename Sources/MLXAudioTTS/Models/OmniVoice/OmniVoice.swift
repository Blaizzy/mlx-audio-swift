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
        // Text embeddings from LLM
        let textIds = inputIds[0..., 0, 0...]  // [B, S]
        let textEmbeds = llm.getEmbeddings(for: textIds)

        // Apply audio mask to inputIds
        let maskedIds = inputIds * audioMask.reshaped([inputIds.shape[0], 1, inputIds.shape[2]])

        // Embed each codebook separately and sum
        var audioEmbeds: MLXArray?
        for (i, embedding) in audioEmbeddings.enumerated() {
            let codebookIds = maskedIds[0..., i, 0...]  // [B, S]
            let codebookEmbeds = embedding(codebookIds)  // [B, S, D]
            if audioEmbeds == nil {
                audioEmbeds = codebookEmbeds
            } else {
                audioEmbeds = audioEmbeds! + codebookEmbeds
            }
        }

        // Where audio: use audio_embeds, else use text_embeds
        let result = MLX.where(
            audioMask.reshaped([audioMask.shape[0], audioMask.shape[1], 1]),
            audioEmbeds!,
            textEmbeds
        )
        return result
    }

    /// Forward pass through the model.
    ///
    /// - Parameters:
    ///   - inputIds: [batch, num_codebooks, seq_len]
    ///   - audioMask: [batch, seq_len]
    ///   - attentionMask: optional custom attention mask (defaults to causal)
    ///   - cache: optional KV cache
    /// - Returns: Audio logits [batch, num_codebooks, seq_len, vocab_size]
    func forward(
        inputIds: MLXArray,
        audioMask: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let inputsEmbeds = prepareEmbedInputs(inputIds: inputIds, audioMask: audioMask)

        // Run through LLM (use custom mask if provided, else causal)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let attentionMask {
            mask = attentionMask
        } else {
            mask = createAttentionMask(h: inputsEmbeds, cache: cache) ?? .none
        }
        let hiddenStates = llm.forwardWithEmbeddings(
            inputsEmbeds: inputsEmbeds,
            cache: cache,
            mask: mask
        )

        // Project to audio codebook logits via per-codebook heads
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        var logitsPerCodebook: [MLXArray] = []
        for (i, head) in audioHeads.enumerated() {
            let logits = head(hiddenStates)  // [B, S, V]
            let reshaped = logits.reshaped([batchSize, seqLen, 1, config.audioVocabSize])
            logitsPerCodebook.append(reshaped)
        }
        let audioLogits = MLX.concatenated(logitsPerCodebook, axis: 2)  // [B, S, C, V]

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
        let prepared = try prepareInferenceInputs(
            text: text,
            numTargetTokens: numTargetTokens,
            refText: refText,
            refAudioTokens: refAudioTokens,
            language: language,
            instruct: voice,
            denoise: ovParameters.denoise
        )

        var inputIds = prepared.inputIds
        let audioMask = prepared.audioMask
        let condLength = inputIds.shape[2]

        // 4. Build batched inputs for CFG (cond + uncond)
        let B = 1
        let numCodebooks = config.numAudioCodebook
        let targetLen = numTargetTokens

        
        // Unconditional input: pad target with leading masks to match cond length
        // cond: [style, text, ref_audio, target] (length = condLength)
        // uncond: [mask...mask, target] (same length, but prefix is masked)
        let prefixLen = condLength - targetLen
        let prefixMask = MLXArray.full(
            [1, numCodebooks, prefixLen],
            values: MLXArray(Int32(config.audioMaskId))
        )
        let targetOnly = inputIds[0..., 0..., prefixLen...]
        var uncondInputIds = MLX.concatenated([prefixMask, targetOnly], axis: 2)
        let uncondAudioMaskPrefix = MLXArray.zeros([1, prefixLen], type: Bool.self)
        let uncondAudioMaskTarget = audioMask[0..., prefixLen...]
        let uncondAudioMask = MLX.concatenated([uncondAudioMaskPrefix, uncondAudioMaskTarget], axis: 1)

        // 5. Initialize target tokens to all MASK
        var tokens = MLXArray.full(
            [B, numCodebooks, targetLen],
            values: MLXArray(Int32(config.audioMaskId))
        )

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

            // Separate forward passes for cond and uncond to avoid custom batch mask issues
            let condLogits = forward(
                inputIds: inputIds,
                audioMask: audioMask,
                attentionMask: .none
            ).asType(.float32)
            let uLogitsFull = forward(
                inputIds: uncondInputIds,
                audioMask: uncondAudioMask,
                attentionMask: .none
            ).asType(.float32)

            // Diagnostic: print logits statistics on step 0
            if step == 0 {
                let cMin = condLogits.min().item(Float.self)
                let cMax = condLogits.max().item(Float.self)
                let cMean = condLogits.mean().item(Float.self)
                let cAnyNaN = isNaN(condLogits).any().item(Bool.self)
                let cAnyInf = isInf(condLogits).any().item(Bool.self)
                let uMin = uLogitsFull.min().item(Float.self)
                let uMax = uLogitsFull.max().item(Float.self)
                let uMean = uLogitsFull.mean().item(Float.self)
                let uAnyNaN = isNaN(uLogitsFull).any().item(Bool.self)
                let uAnyInf = isInf(uLogitsFull).any().item(Bool.self)
                print("[OmniVoice] Step 0 cond logits: min=\(cMin), max=\(cMax), mean=\(cMean), hasNaN=\(cAnyNaN), hasInf=\(cAnyInf)")
                print("[OmniVoice] Step 0 uncond logits: min=\(uMin), max=\(uMax), mean=\(uMean), hasNaN=\(uAnyNaN), hasInf=\(uAnyInf)")
            }

            // Extract target region logits
            let cLogits = condLogits[0, (condLength - targetLen)..<condLength, 0..., 0...]
            let uLogits = uLogitsFull[0, (condLength - targetLen)..<condLength, 0..., 0...]

            // Reshape for scoring: [T, C, V] -> [1, C, T, V]
            let cLogitsBatch = cLogits.transposed(1, 0, 2).reshaped([1, numCodebooks, targetLen, config.audioVocabSize])
            let uLogitsBatch = uLogits.transposed(1, 0, 2).reshaped([1, numCodebooks, targetLen, config.audioVocabSize])

            // Token prediction with CFG
            let (predTokens, scores) = predictTokensWithScoring(
                cLogits: cLogitsBatch,
                uLogits: uLogitsBatch,
                guidanceScale: ovParameters.guidanceScale,
                classTemperature: ovParameters.classTemperature
            )

            // Apply layer penalty
            let adjustedScores = scores - (layerIds.asType(.float32) * ovParameters.layerPenaltyFactor)

            // Gumbel sampling for position selection
            var finalScores = adjustedScores
            if ovParameters.positionTemperature > 0.0 {
                finalScores = gumbelSample(logits: adjustedScores, temperature: ovParameters.positionTemperature)
            }

            // Mask out already-filled positions
            let mask = tokens[0] .!= Int32(config.audioMaskId)
            let maskInf = MLX.where(mask, MLXArray(Float(-Float.infinity)), finalScores).asType(.float32)

            // Flatten for top-k selection
            let flatScores = maskInf.reshaped([-1])
            let flatTokens = tokens[0].reshaped([-1])
            let flatPreds = predTokens[0].reshaped([-1])

            // Select top-k positions to unmask
            let negScores = MLXArray(-1.0) * flatScores.asType(.float32)
            let sortedIndices = MLX.argSort(negScores, axis: 0)
            let rangeIndices = MLXArray((0..<k).map { Int32($0) })
            let topkIndices = MLX.take(sortedIndices, rangeIndices, axis: 0)

            // Vectorized update using putAlong
            let linearTopkIndices = topkIndices.reshaped([-1])
            let updateValues = MLX.take(flatPreds, linearTopkIndices, axis: 0)
            let updatedTokens = putAlong(flatTokens, linearTopkIndices, values: updateValues, axis: 0)

            let reshapedTokens = updatedTokens.reshaped([numCodebooks, targetLen])
            tokens = reshapedTokens.reshaped([1, numCodebooks, targetLen])

            // Update cond and uncond inputs for next step
            let prefixLen = condLength - targetLen
            let condHead = inputIds[0, 0..., 0..<prefixLen]
            let condUpdatedFull = MLX.concatenated([condHead, tokens[0]], axis: 1)
                .reshaped([1, numCodebooks, condLength])
            let uncondHead = uncondInputIds[0, 0..., 0..<prefixLen]
            let uncondUpdatedFull = MLX.concatenated([uncondHead, tokens[0]], axis: 1)
                .reshaped([1, numCodebooks, condLength])

            inputIds = condUpdatedFull
            uncondInputIds = uncondUpdatedFull

            eval(inputIds, uncondInputIds, tokens)
        }

        // Safeguard: fill any remaining mask tokens with a final deterministic prediction
        let finalMask = tokens .== Int32(config.audioMaskId)
        if finalMask.any().item(Bool.self) {
            let finalCondLogits = forward(
                inputIds: inputIds,
                audioMask: audioMask,
                attentionMask: .none
            ).asType(.float32)
            let finalULogitsFull = forward(
                inputIds: uncondInputIds,
                audioMask: uncondAudioMask,
                attentionMask: .none
            ).asType(.float32)
            let finalC = finalCondLogits[0, (condLength - targetLen)..<condLength, 0..., 0...]
                .transposed(1, 0, 2).reshaped([1, numCodebooks, targetLen, config.audioVocabSize])
            let finalU = finalULogitsFull[0, (condLength - targetLen)..<condLength, 0..., 0...]
                .transposed(1, 0, 2).reshaped([1, numCodebooks, targetLen, config.audioVocabSize])
            let (finalPredTokens, _) = predictTokensWithScoring(
                cLogits: finalC,
                uLogits: finalU,
                guidanceScale: ovParameters.guidanceScale,
                classTemperature: 0.0
            )
            tokens = MLX.where(finalMask, finalPredTokens, tokens)
        }

        // 8. Decode tokens to waveform
        var outputTokens = tokens[0, 0..., 0..<targetLen]
        
        // Replace any remaining mask tokens with 0 (matching Python reference)
        let remainingMask = outputTokens .== Int32(config.audioMaskId)
        if remainingMask.any().item(Bool.self) {
            outputTokens = MLX.where(remainingMask, MLXArray.zeros(outputTokens.shape, type: Int32.self), outputTokens)
        }
        
        // Diagnostic: print token statistics to debug noise issues
        let tokenVals = outputTokens.asArray(Int32.self)
        let uniqueVals = Set(tokenVals)
        let maskCount = tokenVals.filter { $0 == Int32(config.audioMaskId) }.count
        print("[OmniVoice] Token stats: min=\(uniqueVals.min() ?? -1), max=\(uniqueVals.max() ?? -1), unique=\(uniqueVals.count), maskRemaining=\(maskCount)")

        // Diagnostic: bypass diffusion and decode refAudioTokens directly if available
        if let refTok = refAudioTokens {
            do {
                let refDecoded = try audioTok.decode(refTok)
                let tempDir = FileManager.default.temporaryDirectory
                let refURL = tempDir.appendingPathComponent("omnivoice_diagnostic_ref_direct.wav")
                try AudioUtils.writeWavFile(samples: refDecoded.asArray(Float.self), sampleRate: Double(config.sampleRate), fileURL: refURL)
                print("[OmniVoice] DIAGNOSTIC: saved direct ref decode to \(refURL.path)")
            } catch {
                print("[OmniVoice] DIAGNOSTIC: ref direct decode failed: \(error)")
            }
        }

        let audio = try audioTok.decode(outputTokens)
        
        // Diagnostic: print vocoder output statistics
        let audioMin = audio.min().item(Float.self)
        let audioMax = audio.max().item(Float.self)
        let audioMean = audio.mean().item(Float.self)
        let audioNaN = isNaN(audio).any().item(Bool.self)
        let audioInf = isInf(audio).any().item(Bool.self)
        print("[OmniVoice] Vocoder decode stats: min=\(audioMin), max=\(audioMax), mean=\(audioMean), hasNaN=\(audioNaN), hasInf=\(audioInf)")

        // Temporary diagnostics: save intermediate audio to temp directory
        do {
            let tempDir = FileManager.default.temporaryDirectory
            let rawURL = tempDir.appendingPathComponent("omnivoice_diagnostic_raw.wav")
            let finalURL = tempDir.appendingPathComponent("omnivoice_diagnostic_final.wav")
            let rawSamples = audio.asArray(Float.self)
            try AudioUtils.writeWavFile(samples: rawSamples, sampleRate: Double(config.sampleRate), fileURL: rawURL)
            print("[OmniVoice] DIAGNOSTIC: saved raw vocoder output to \(rawURL.path)")
            let processed = postProcessAudio(audio, refRms: nil, postprocessOutput: ovParameters.postprocessOutput)
            try AudioUtils.writeWavFile(samples: processed.asArray(Float.self), sampleRate: Double(config.sampleRate), fileURL: finalURL)
            print("[OmniVoice] DIAGNOSTIC: saved post-processed audio to \(finalURL.path)")
            return processed
        } catch {
            print("[OmniVoice] DIAGNOSTIC: failed to save wav: \(error)")
        }

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
        // Vectorized scatter along the last axis to avoid indexed-assignment crashes
        filtered = putAlong(filtered, topIndices, values: topVals, axis: -1)
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
        var styleIds = MLXArray(styleTokenIds.map { Int32($0) })
        styleIds = styleIds.reshaped([1, -1])
        styleIds = MLX.broadcast(styleIds.reshaped([1, 1, -1]), to: [1, numCodebooks, styleIds.shape[1]])

        // Build text tokens
        let fullText = combineText(refText: refText, text: text)
        let wrappedText = "<|text_start|>\(fullText)<|text_end|>"
        let textTokenIds = try tokenizeText(wrappedText)
        var textIds = MLXArray(textTokenIds.map { Int32($0) })
        textIds = textIds.reshaped([1, -1])
        textIds = MLX.broadcast(textIds.reshaped([1, 1, -1]), to: [1, numCodebooks, textIds.shape[1]])

        // Target: all MASK
        let targetIds = MLXArray.full(
            [1, numCodebooks, numTargetTokens],
            values: MLXArray(Int32(config.audioMaskId))
        )

        // Concatenate: [style, text, ref_audio (optional), target]
        var parts: [MLXArray] = [styleIds, textIds]
        if let refTok = refAudioTokens {
            var alignedRefTok = refTok
            if refTok.ndim == 2 && refTok.shape[0] != numCodebooks {
                if refTok.shape[0] < numCodebooks {
                    // Pad with mask tokens to match numCodebooks
                    let padShape = [numCodebooks - refTok.shape[0], refTok.shape[1]]
                    let pad = MLXArray.full(padShape, values: MLXArray(Int32(config.audioMaskId)))
                    alignedRefTok = MLX.concatenated([refTok, pad], axis: 0)
                } else {
                    // Truncate to numCodebooks
                    alignedRefTok = refTok[0..<numCodebooks, 0...]
                }
            }
            let reshaped = alignedRefTok.reshaped([1, alignedRefTok.shape[0], alignedRefTok.shape[1]])
            parts.append(reshaped)
        }
        parts.append(targetIds)

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

        
        let zerosPrefix = MLXArray.zeros([audioStartIdx], type: Bool.self)
        let onesSuffix = MLXArray.ones([totalLength - audioStartIdx], type: Bool.self)
        let condAudioMask = MLX.concatenated([zerosPrefix, onesSuffix], axis: 0)
            .reshaped([1, totalLength])

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
                let head = result[0..<fadeLen] * fadeIn
                let mid = result[fadeLen..<(len - fadeLen)]
                let tail = result[(len - fadeLen)...] * fadeOut
                result = MLX.concatenated([head, mid, tail], axis: 0)
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

        // Load model weights first to infer actual num_audio_codebooks from checkpoint
        let weightsURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["model.safetensors"]
        ).appendingPathComponent("model.safetensors")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let inferredNumCodebooks = Self.inferNumCodebooks(from: rawWeights) ?? 9

        // Parse and modify main config to match checkpoint
        var configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        if let currentNum = configDict["num_audio_codebook"] as? Int, currentNum != inferredNumCodebooks {
            print("[OmniVoiceModel] INFO: overriding num_audio_codebook from \(currentNum) to \(inferredNumCodebooks) to match checkpoint")
            configDict["num_audio_codebook"] = inferredNumCodebooks
            if let weights = configDict["audio_codebook_weights"] as? [Int], weights.count != inferredNumCodebooks {
                var newWeights = weights
                while newWeights.count < inferredNumCodebooks {
                    newWeights.append(newWeights.last ?? 2)
                }
                if newWeights.count > inferredNumCodebooks {
                    newWeights = Array(newWeights.prefix(inferredNumCodebooks))
                }
                configDict["audio_codebook_weights"] = newWeights
            }
        }
        let modifiedConfigData = try JSONSerialization.data(withJSONObject: configDict)
        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: modifiedConfigData)

        let model = try OmniVoiceModel(config: config)
        let sanitizedWeights = model.sanitize(weights: rawWeights)
        let moduleParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let weightKeys = Set(sanitizedWeights.keys)
        let paramKeys = Set(moduleParams.keys)
        let missing = paramKeys.subtracting(weightKeys).sorted()
        let extra = weightKeys.subtracting(paramKeys).sorted()
        if !missing.isEmpty {
            print("[OmniVoiceModel] WARNING: \(missing.count) parameters missing from checkpoint: \(missing.prefix(10))")
        }
        if !extra.isEmpty {
            print("[OmniVoiceModel] WARNING: \(extra.count) extra keys after sanitize: \(extra.prefix(10))")
        }
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

    // MARK: - Weight Inspection

    private static func inferNumCodebooks(from weights: [String: MLXArray]) -> Int? {
        var maxIdx = -1
        for key in weights.keys {
            if key.hasPrefix("audio_embeddings."), key.hasSuffix(".weight") {
                let suffix = key.dropFirst("audio_embeddings.".count)
                if let dotIdx = suffix.firstIndex(of: ".") {
                    let numStr = suffix.prefix(upTo: dotIdx)
                    if let idx = Int(numStr), idx > maxIdx {
                        maxIdx = idx
                    }
                }
            }
        }
        return maxIdx >= 0 ? maxIdx + 1 : nil
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
    let outputPaddingVal: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int, outputPadding: Int = 0) {
        self.strideVal = stride
        self.paddingVal = padding
        self.outputPaddingVal = outputPadding

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        // PyTorch format: [in_channels, out_channels, kernel_size]
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [inChannels, outChannels, kernelSize]
        )
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Weight stored as [in, out, kernel] (PyTorch) → transpose to [out, kernel, in] (MLX)
        let w = weight.transposed(1, 2, 0).asType(.float32)
        // Data flows in NCL [B, C, L]; transpose to NLC for MLX convTransposed1d
        let xNLC = x.transposed(0, 2, 1).asType(.float32)
        var h = MLX.convTransposed1d(xNLC, w, stride: strideVal, padding: paddingVal, outputPadding: outputPaddingVal)
        if let b = bias {
            let n = b.size
            h = h + b.asType(.float32).reshaped([n])
        }
        // Convert back to NCL [B, C, L]
        let out = h.transposed(0, 2, 1).asType(x.dtype)
        return out
    }
}

/// Conv1d using MLX weight layout [out_channels, kernel_size, in_channels].
final class OmniVoiceConv1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let strideVal: Int
    let paddingVal: Int
    let dilationVal: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int, dilation: Int = 1) {
        self.strideVal = stride
        self.paddingVal = padding
        self.dilationVal = dilation

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        // MLX format: [out_channels, kernel_size, in_channels]
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outChannels, kernelSize, inChannels]
        )
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Weight stored as [out, in, kernel] (PyTorch format) → transpose to [out, kernel, in] (MLX)
        let w = weight.transposed(0, 2, 1).asType(.float32)
        // Data flows in NCL [B, C, L]; transpose to NLC for MLX conv1d, then back
        let xNLC = x.transposed(0, 2, 1).asType(.float32)
        var h = MLX.conv1d(xNLC, w, stride: strideVal, padding: paddingVal, dilation: dilationVal)
        if let b = bias {
            let n = b.size
            h = h + b.asType(.float32).reshaped([n])
        }
        // Convert back to NCL [B, C, L]
        let out = h.transposed(0, 2, 1).asType(x.dtype)
        return out
    }
}

// MARK: - DAC-style Audio Codec

/// Snake activation: x + (1/a) * sin(a*x)^2
func snakeActivation(_ x: MLXArray) -> MLXArray {
    let alpha: Float = 1.0
    let x32 = x.asType(.float32)
    let recip = 1.0 / (alpha + 1e-9)
    return (x32 + recip * MLX.square(MLX.sin(alpha * x32))).asType(x.dtype)
}

/// DAC-style residual unit with Snake activations.
public final class OmniVoiceDACResidualUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d
    @ModuleInfo(key: "conv2") var conv2: OmniVoiceConv1d
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "snake2") var snake2: snakeAlpha

    init(channels: Int, kernelSize: Int, dilation: Int) {
        // Match PyTorch DAC: kernel_size=7 for conv1 with dilation-dependent same-padding,
        // and kernel_size=1 for conv2 (pointwise).
        let conv1KernelSize = 7
        let conv1Padding = ((conv1KernelSize - 1) * dilation) / 2
        let conv2KernelSize = 1
        let conv2Padding = 0

        self._conv1.wrappedValue = OmniVoiceConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: conv1KernelSize,
            stride: 1,
            padding: conv1Padding,
            dilation: dilation
        )
        self._conv2.wrappedValue = OmniVoiceConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: conv2KernelSize,
            stride: 1,
            padding: conv2Padding
        )
        self._snake1.wrappedValue = snakeAlpha(channels: channels)
        self._snake2.wrappedValue = snakeAlpha(channels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // DAC residual unit order: Snake → Conv → Snake → Conv + residual
        let s1 = snake1.callAsFunction(x)
        let c1 = conv1(s1)
        let s2 = snake2.callAsFunction(c1)
        let h = conv2(s2)
        
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
        let x32 = x.asType(.float32)
        let a32 = alpha.asType(.float32)
        let channels = a32.size
        // Swift DAC conv layers use NCL [B,C,L]; reshape alpha to [1,C,1] for broadcasting
        let aExpanded = a32.reshaped([1, channels, 1])
        let recip = 1.0 / (aExpanded + 1e-9)
        return (x32 + recip * MLX.square(MLX.sin(aExpanded * x32))).asType(x.dtype)
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
        self._snake1.wrappedValue = snakeAlpha(channels: inputChannels)
        self._convT1.wrappedValue = OmniVoiceConvTranspose1d(
            inChannels: inputChannels,
            outChannels: outputChannels,
            kernelSize: stride * 2,
            stride: stride,
            padding: stride / 2 + stride % 2,
            outputPadding: stride % 2
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
        var h = convT1(snake1.callAsFunction(x))
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
        self._conv2.wrappedValue = OmniVoiceConv1d(
            inChannels: currentChannels,
            outChannels: currentChannels / 8,   // 2048 → 256 to match checkpoint
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        print("[OmniVoice Decoder] after conv1: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self)), mean=\(h.mean().item(Float.self))")
        for (i, b) in block.enumerated() {
            h = b(h)
            print("[OmniVoice Decoder] after upBlock \(i): shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self)), mean=\(h.mean().item(Float.self))")
        }
        h = snake1.callAsFunction(h)
        print("[OmniVoice Decoder] after snake1: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self)), mean=\(h.mean().item(Float.self))")
        h = conv2(h)
        print("[OmniVoice Decoder] after conv2: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self)), mean=\(h.mean().item(Float.self))")
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
        print("[OmniVoice Dec] conv1 out: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self))")
        for (i, b) in block.enumerated() {
            h = b(h)
            print("[OmniVoice Dec] upBlock \(i) out: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self))")
        }
        h = snake1.callAsFunction(h)
        print("[OmniVoice Dec] snake1 out: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self))")
        h = conv2(h)
        print("[OmniVoice Dec] conv2 out: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self))")
        return h
    }
}

/// Residual Vector Quantization with projection layers (Higgs Audio V2 style).
public final class OmniVoiceRVQQuantizer: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [OmniVoiceSingleQuantizer]
    let outputDim: Int

    init(config: OmniVoiceAudioTokenizerConfig) {
        let nQuantizers = config.nCodebooks
        let codebookSize = config.codebookSize
        let codebookDim = config.codebookDim
        let inputDim = config.decoderHiddenSize  // 1024, matching project_in weight shape [64, 1024]
        self.outputDim = config.decoderHiddenSize  // 1024

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
    /// Note: D must be 1024 (decoderHiddenSize). The caller must pre-project if needed.
    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        // z is [B, C, L] (NCL) where C = 1024
        let batchSize = z.shape[0]
        let seqLen = z.shape[2]
        let nQuantizers = quantizers.count

        var allCodes: [MLXArray] = []
        var quantized = MLXArray.zeros([batchSize, outputDim, seqLen])

        for qIdx in 0..<nQuantizers {
            let q = quantizers[qIdx]
            let codebook = q.codebook.embed
            let codebookSize = codebook.shape[0]
            let codebookDim = codebook.shape[1]

            // Project input to codebook dimension
            // z is [B, C, L] (NCL), Linear expects [B, L, C] (NLC)
            let zNLC = z.transposed(0, 2, 1)  // [B, L, C]
            let zProjNLC = q.projectIn(zNLC)  // [B, L, codebookDim]
            let zProj = zProjNLC.transposed(0, 2, 1)  // [B, codebookDim, L]

            // [B, codebookDim, T] -> [B, T, codebookDim] for distance computation
            let zPermute = zProj.transposed(0, 2, 1)
            let zFlat = zPermute.reshaped([batchSize * seqLen, codebookDim])

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
            let qOut = q.projectOut(qVecs)  // [B*T, outputDim]
            let q3d = qOut.reshaped([batchSize, seqLen, -1]).transposed(0, 2, 1)

            quantized = quantized + q3d
        }

        let codes = MLX.stacked(allCodes, axis: 1)  // [B, n_quantizers, T]
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

        // fc2 projects quantized features (decoderHiddenSize) to decoder input (encoderHiddenSize * 4)
        self._fc2.wrappedValue = MLXNN.Linear(
            inputDimensions: config.decoderHiddenSize,
            outputDimensions: config.encoderHiddenSize * 4
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

        // Encoder: [B, 1, T] -> [B, D, T'] where D=256
        let z = acousticEncoder(wav)

        // Project encoder output (256) to quantizer input dimension (1024)
        // fc2 is Linear(1024→256), weight shape [256, 1024]. To reverse: x @ W = [B,T',256] @ [256,1024] = [B,T',1024]
        let zNLC = z.transposed(0, 2, 1)  // [B, T', 256]
        let zProjectedNLC = MLX.matmul(zNLC, fc2.weight)  // [B, T', 1024]
        let zProjected = zProjectedNLC.transposed(0, 2, 1)  // [B, 1024, T']

        // RVQ: [B, D, T'] -> (codes [B, n_codebooks, T'], quantized [B, D, T'])
        let (codes, _) = quantizer(zProjected)

        // Temporary diagnostic: encode->decode roundtrip to isolate vocoder issues
        do {
            let decoded = try self.decode(codes[0])
            let tempDir = FileManager.default.temporaryDirectory
            let originalURL = tempDir.appendingPathComponent("omnivoice_tokenizer_roundtrip_original.wav")
            let decodedURL = tempDir.appendingPathComponent("omnivoice_tokenizer_roundtrip_decoded.wav")
            try AudioUtils.writeWavFile(samples: wav[0].asArray(Float.self), sampleRate: Double(config.sampleRate), fileURL: originalURL)
            try AudioUtils.writeWavFile(samples: decoded.asArray(Float.self), sampleRate: Double(config.sampleRate), fileURL: decodedURL)
            print("[OmniVoiceAudioTokenizer] DIAGNOSTIC: saved original to \(originalURL.path)")
            print("[OmniVoiceAudioTokenizer] DIAGNOSTIC: saved roundtrip to \(decodedURL.path)")
        } catch {
            print("[OmniVoiceAudioTokenizer] DIAGNOSTIC: roundtrip save failed: \(error)")
        }

        // Return [n_codebooks, T'] (squeeze batch dim)
        return codes[0]
    }

    /// Decode discrete tokens back to audio waveform.
    /// - Parameter tokens: [num_codebooks, seq_len]
    /// - Returns: [samples]
    public func decode(_ tokens: MLXArray) throws -> MLXArray {
        // Add batch dim: [n_codebooks, T] -> [1, n_codebooks, T]
        let batchedTokens = tokens.reshaped([1, tokens.shape[0], tokens.shape[1]])

        // RVQ decode: [1, n_codebooks, T] -> [1, D, T]
        let z = quantizer.decode(batchedTokens)
        print("[OmniVoice Decode] after quantizer.decode: shape=\(z.shape), min=\(z.min().item(Float.self)), max=\(z.max().item(Float.self)), mean=\(z.mean().item(Float.self))")

        // fc2 project: [1, D, T] -> [1, D', T]
        let h = fc2(z.transposed(0, 2, 1)).transposed(0, 2, 1)
        print("[OmniVoice Decode] after fc2: shape=\(h.shape), min=\(h.min().item(Float.self)), max=\(h.max().item(Float.self)), mean=\(h.mean().item(Float.self))")

        // Verify decoder conv1 weight shape at runtime
        let paramDict = Dictionary(uniqueKeysWithValues: acousticDecoder.parameters().flattened())
        if let w = paramDict["conv1.weight"] {
            print("[OmniVoice Decode] RUNTIME CHECK acousticDecoder.conv1.weight: shape=\(w.shape), min=\(w.min().item(Float.self)), max=\(w.max().item(Float.self))")
        } else {
            print("[OmniVoice Decode] RUNTIME CHECK acousticDecoder.conv1.weight: MISSING")
        }
        if let w = paramDict["conv2.weight"] {
            print("[OmniVoice Decode] RUNTIME CHECK acousticDecoder.conv2.weight: shape=\(w.shape), min=\(w.min().item(Float.self)), max=\(w.max().item(Float.self))")
        } else {
            print("[OmniVoice Decode] RUNTIME CHECK acousticDecoder.conv2.weight: MISSING")
        }

        // Decoder: [1, D', T] -> [1, 1, T']
        let audio = acousticDecoder(h)
        print("[OmniVoice Decode] after acousticDecoder: shape=\(audio.shape), min=\(audio.min().item(Float.self)), max=\(audio.max().item(Float.self)), mean=\(audio.mean().item(Float.self))")

        return audio.reshaped([-1])
    }

    /// Sanitize and remap checkpoint weights for the audio tokenizer.
    /// Mirrors Python HiggsAudioTokenizer.sanitize logic, but skips conv
    /// weight transposes because Swift's OmniVoiceConv1d/ConvTranspose1d
    /// already handle PyTorch-to-MLX layout conversion internally.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        let keepPrefixes = [
            "acoustic_encoder.",
            "acoustic_decoder.",
            "quantizer.",
            "fc2.",
            "semantic_model.",
            "encoder_semantic.",
        ]
        let keepExact: Set<String> = ["fc.weight", "fc.bias"]
        let dropPrefixes = ["decoder_semantic.", "fc1."]
        let dropSuffixes = [".embed_avg", ".cluster_size", ".inited"]

        for (var k, var v) in weights {
            // Explicit drops
            if dropPrefixes.contains(where: { k.hasPrefix($0) }) { continue }
            if !keepPrefixes.contains(where: { k.hasPrefix($0) }) && !keepExact.contains(k) { continue }
            if dropSuffixes.contains(where: { k.hasSuffix($0) }) { continue }

            // === Acoustic path weight transforms ===
            if k.hasPrefix("acoustic_encoder.") || k.hasPrefix("acoustic_decoder.") || k.hasPrefix("quantizer.") || k.hasPrefix("fc2.") {
                // Python uses nn.Embedding with key "weight"; Swift uses MLXArray with key "embed"
                if k.hasSuffix(".codebook.weight") {
                    k = String(k.dropLast("weight".count)) + "embed"
                }
                // NOTE: checkpoint alpha is [1,1,C] for NLC; Swift DAC uses NCL,
                // so we reshape at runtime in snakeAlpha.callAsFunction instead.
                // NOTE: we do NOT transpose 3D conv weights here because
                // OmniVoiceConv1d and OmniVoiceConvTranspose1d already
                // transpose from PyTorch [out,in,k] / [in,out,k] to MLX
                // [out,k,in] at runtime.
            }

            result[k] = v
        }
        return result
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

        // Load tokenizer weights first to infer actual n_codebooks
        let weightsURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["audio_tokenizer/model.safetensors"]
        ).appendingPathComponent("audio_tokenizer/model.safetensors")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let inferredNCodebooks = Self.inferNCodebooks(from: rawWeights) ?? 9

        var configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        
        // Pull in nested acoustic_model_config values if present
        if let acoustic = configDict["acoustic_model_config"] as? [String: Any] {
            for key in ["codebook_size", "codebook_dim", "n_codebooks", "hop_length", "sampling_rate",
                        "downsampling_ratios", "upsampling_ratios", "encoder_hidden_size",
                        "decoder_hidden_size", "kernel_size"] {
                if configDict[key] == nil, let val = acoustic[key] {
                    configDict[key] = val
                }
            }
        }
        
        if let currentNum = configDict["n_codebooks"] as? Int, currentNum != inferredNCodebooks {
            print("[OmniVoiceAudioTokenizer] INFO: overriding n_codebooks from \(currentNum) to \(inferredNCodebooks) to match checkpoint")
            configDict["n_codebooks"] = inferredNCodebooks
        } else if configDict["n_codebooks"] == nil {
            print("[OmniVoiceAudioTokenizer] INFO: setting n_codebooks to \(inferredNCodebooks) from checkpoint")
            configDict["n_codebooks"] = inferredNCodebooks
        }
        let patchedConfigData = try JSONSerialization.data(withJSONObject: configDict)
        let config = try JSONDecoder().decode(OmniVoiceAudioTokenizerConfig.self, from: patchedConfigData)

        let tokenizer = OmniVoiceAudioTokenizer(config: config)
        let weights = tokenizer.sanitize(weights: rawWeights)

        // Verify weight coverage before loading
        let moduleParams = Dictionary(uniqueKeysWithValues: tokenizer.parameters().flattened())
        let weightKeys = Set(weights.keys)
        let paramKeys = Set(moduleParams.keys)
        let missing = paramKeys.subtracting(weightKeys).sorted()
        let extra = weightKeys.subtracting(paramKeys).sorted()
        if !missing.isEmpty {
            print("[OmniVoiceAudioTokenizer] WARNING: \(missing.count) parameters missing from checkpoint: \(missing.prefix(10))")
        }
        if !extra.isEmpty {
            print("[OmniVoiceAudioTokenizer] WARNING: \(extra.count) extra keys in checkpoint: \(extra.prefix(10))")
        }
        
        // Test: load all tokenizer weights in float32 to rule out bfloat16 precision issues
        let float32Weights = weights.mapValues { $0.asType(.float32) }
        try tokenizer.update(parameters: ModuleParameters.unflattened(float32Weights), verify: .noUnusedKeys)
        eval(tokenizer)
        print("[OmniVoiceAudioTokenizer] INFO: loaded all weights as float32 for precision test")

        return tokenizer
    }

    private static func inferNCodebooks(from weights: [String: MLXArray]) -> Int? {
        var maxIdx = -1
        for key in weights.keys {
            if key.hasPrefix("quantizer.quantizers."), key.contains(".codebook.") {
                let suffix = key.dropFirst("quantizer.quantizers.".count)
                if let dotIdx = suffix.firstIndex(of: ".") {
                    let numStr = suffix.prefix(upTo: dotIdx)
                    if let idx = Int(numStr), idx > maxIdx {
                        maxIdx = idx
                    }
                }
            }
        }
        return maxIdx >= 0 ? maxIdx + 1 : nil
    }
}

