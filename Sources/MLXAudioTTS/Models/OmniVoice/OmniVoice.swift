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

    public var tokenizer: Tokenizers.Tokenizer?
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
    public func forward(
        inputIds: MLXArray,
        audioMask: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let inputsEmbeds = prepareEmbedInputs(inputIds: inputIds, audioMask: audioMask)

        // Run through LLM. OmniVoice is a bidirectional NAR diffusion model:
        // the default must be NO attention mask (matching the Python reference,
        // which never applies a causal mask). Call sites pass `nil` (Optional.none),
        // which this `?? .none` resolves to the no-mask enum case. Passing the
        // enum `.none` literally would be ambiguous with Optional.none and once
        // fell through to a CAUSAL mask here, garbling generation.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = attentionMask ?? .none
        let hiddenStates = llm.forwardWithEmbeddings(
            inputsEmbeds: inputsEmbeds,
            cache: cache,
            mask: mask
        )

        // Project to audio codebook logits via per-codebook heads
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        var logitsPerCodebook: [MLXArray] = []
        for head in audioHeads {
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
            speed: ovParameters.speed,
            duration: ovParameters.duration
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


        // Unconditional input: only the target region (matching Python reference)
        let prefixLen = condLength - targetLen
        var uncondInputIds = inputIds[0..., 0..., prefixLen...]  // [1, C, T]
        let uncondAudioMask = audioMask[0..., prefixLen...]  // [1, T]

        // 5. Initialize target tokens to all MASK
        var tokens = MLXArray.full(
            [B, numCodebooks, targetLen],
            values: MLXArray(Int32(config.audioMaskId))
        )

        // 6. Compute timesteps and unmasking schedule (parity with the Python
        // reference: num_step+1 points spaced 1/num_step, k = max(1, ceil(total*dt)),
        // capped by the remaining masked count so putAlong never overwrites
        // already-revealed positions)
        // Clamp to >= 1: numStep <= 0 would trap in getTimeSteps' 0...numStep range
        // (a remote crash for server embedders) or skip diffusion entirely.
        let numSteps = max(1, ovParameters.numStep)
        let timesteps = getTimeSteps(tStart: 0.0, tEnd: 1.0, numStep: numSteps, tShift: ovParameters.tShift)

        let totalMask = targetLen * numCodebooks
        var rem = totalMask
        var schedule: [Int] = []
        for step in 0..<numSteps {
            var k: Int
            if step == numSteps - 1 {
                k = rem
            } else {
                let ceilVal = max(1, Int(ceil(Float(totalMask) * (timesteps[step + 1] - timesteps[step]))))
                k = min(ceilVal, rem)
            }
            if k < 0 { k = 0 }
            schedule.append(k)
            rem -= k
        }

        let layerIds = MLXArray((0..<numCodebooks).map { Int32($0) }).reshaped([1, numCodebooks, 1])

        // 7. Iterative diffusion generation
        for step in 0..<numSteps {
            let k = schedule[step]
            if k <= 0 { continue }

            // Separate forward passes for cond and uncond (bidirectional attention)
            let condLogits = forward(
                inputIds: inputIds,
                audioMask: audioMask
            ).asType(.float32)
            let uLogitsFull = forward(
                inputIds: uncondInputIds,
                audioMask: uncondAudioMask
            ).asType(.float32)

            // Extract target region logits
            let cLogits = condLogits[0, (condLength - targetLen)..<condLength, 0..., 0...]
            let uLogits = uLogitsFull[0, 0..<targetLen, 0..., 0...]

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
            let condHead = inputIds[0, 0..., 0..<prefixLen]
            inputIds = MLX.concatenated([condHead, tokens[0]], axis: 1)
                .reshaped([1, numCodebooks, condLength])
            uncondInputIds = tokens  // uncond is just the target region

            eval(inputIds, uncondInputIds, tokens)
        }

        // Safeguard: fill any remaining mask tokens with a final deterministic prediction
        let finalMask = tokens .== Int32(config.audioMaskId)
        if finalMask.any().item(Bool.self) {
            let finalCondLogits = forward(
                inputIds: inputIds,
                audioMask: audioMask,
                attentionMask: nil
            ).asType(.float32)
            let finalULogitsFull = forward(
                inputIds: uncondInputIds,
                audioMask: uncondAudioMask,
                attentionMask: nil
            ).asType(.float32)
            let finalC = finalCondLogits[0, (condLength - targetLen)..<condLength, 0..., 0...]
                .transposed(1, 0, 2).reshaped([1, numCodebooks, targetLen, config.audioVocabSize])
            let finalU = finalULogitsFull[0, 0..<targetLen, 0..., 0...]
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

    private func estimateTargetTokens(
        text: String,
        refText: String?,
        numRefAudioTokens: Int,
        speed: Float,
        duration: Float?
    ) -> Int {
        let tokensPerSecond = Float(config.sampleRate) / 960.0
        if let duration {
            return max(1, Int(ceil(duration * tokensPerSecond)))
        }

        // Parity with mlx-audio Python:
        // RuleDurationEstimator().estimate_duration(text, "Nice to meet you.", 25)
        // followed by a 1.15 safety margin. The previous `characters * 4`
        // heuristic over-allocated English by roughly 2x, causing the diffusion
        // model to place speech near the end of a long target window.
        let rawTokens = estimateRuleDurationTokens(
            targetText: text,
            refText: "Nice to meet you.",
            refDuration: 25.0
        )
        let baseEstimate = max(10, Int(rawTokens * 1.15))
        let adjusted = speed > 0 && speed != 1.0 ? Int(Float(baseEstimate) / speed) : baseEstimate
        return max(1, adjusted)
    }

    private func estimateRuleDurationTokens(
        targetText: String,
        refText: String,
        refDuration: Float,
        lowThreshold: Float = 50.0,
        boostStrength: Float = 3.0
    ) -> Float {
        guard refDuration > 0, !refText.isEmpty else { return 0 }
        let refWeight = phoneticWeight(refText)
        guard refWeight > 0 else { return 0 }

        let speedFactor = refWeight / refDuration
        let estimated = phoneticWeight(targetText) / speedFactor
        if estimated < lowThreshold {
            let alpha = 1.0 / boostStrength
            return lowThreshold * pow(estimated / lowThreshold, alpha)
        }
        return estimated
    }

    private func phoneticWeight(_ text: String) -> Float {
        text.unicodeScalars.reduce(Float(0)) { total, scalar in
            total + phoneticWeight(scalar)
        }
    }

    private func phoneticWeight(_ scalar: Unicode.Scalar) -> Float {
        let code = scalar.value
        if (65...90).contains(code) || (97...122).contains(code) {
            return 1.0
        }
        if code == 32 {
            return 0.2
        }
        if code == 0x0640 {
            return 0.0
        }

        switch scalar.properties.generalCategory {
        case .nonspacingMark, .spacingMark, .enclosingMark:
            return 0.0
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation,
             .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol:
            return 0.5
        case .spaceSeparator, .lineSeparator, .paragraphSeparator:
            return 0.2
        case .decimalNumber, .letterNumber, .otherNumber:
            return 3.5
        default:
            break
        }

        return scriptWeight(for: code)
    }

    private func scriptWeight(for code: UInt32) -> Float {
        if code <= 0x02AF { return 1.0 }
        if code <= 0x03FF { return 1.0 }
        if code <= 0x052F { return 1.0 }
        if code <= 0x058F { return 1.0 }
        if code <= 0x05FF { return 1.5 }
        if code <= 0x08FF { return 1.5 }
        if code <= 0x0DFF { return 1.8 }
        if code <= 0x0EFF { return 1.5 }
        if code <= 0x0FFF { return 1.8 }
        if code <= 0x109F { return 1.8 }
        if code <= 0x10FF { return 1.0 }
        if code <= 0x11FF { return 2.5 }
        if code <= 0x139F { return 3.0 }
        if code <= 0x17FF { return 1.8 }
        if code <= 0x1C7F { return 1.8 }
        if code <= 0x1C8F { return 1.0 }
        if code <= 0x1CBF { return 1.0 }
        if code <= 0x1CFF { return 1.8 }
        if code <= 0x1EFF { return 1.0 }
        if code <= 0x309F { return 2.2 }
        if code <= 0x30FF { return 2.2 }
        if code <= 0x312F { return 3.0 }
        if code <= 0x318F { return 2.5 }
        if code <= 0x9FFF { return 3.0 }
        if code <= 0xA4CF { return 3.0 }
        if code <= 0xA69F { return 1.0 }
        if code <= 0xA7FF { return 1.0 }
        if code <= 0xA8FF { return 1.8 }
        if code <= 0xA97F { return 2.5 }
        if code <= 0xAADF { return 1.8 }
        if code <= 0xAB2F { return 3.0 }
        if code <= 0xAB6F { return 1.0 }
        if code <= 0xABFF { return 1.8 }
        if code <= 0xD7AF { return 2.5 }
        if code <= 0xFAFF { return 3.0 }
        if code <= 0xFEFF { return 1.5 }
        if code <= 0xFFEF { return 1.0 }
        if code > 0x20000 { return 3.0 }
        return 1.0
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
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let numCodebooks = config.numAudioCodebook

        // Build style tokens
        var styleText = ""
        if denoise && refAudioTokens != nil {
            // Python reference passes denoise=has_ref at the generate call sites
            // (omnivoice.py:387,585): <|denoise|> conditions cloning only.
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
        return tokenizer.encode(text: text, addSpecialTokens: false)
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

        // Parse and modify main config to match checkpoint
        var configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        // Prefer the codebook count inferred from split checkpoint keys; fused
        // checkpoints (audio_embeddings.weight [C*V, H]) carry no per-codebook
        // keys, so fall back to the config value before the legacy default.
        let inferredNumCodebooks = Self.inferNumCodebooks(from: rawWeights)
            ?? (configDict["num_audio_codebook"] as? Int ?? 9)
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
        // Weights run as float32: the diffusion loop is sensitive to logit
        // precision and the reference checkpoint ships fp32 (no-op there).
        let float32Weights = sanitizedWeights.mapValues { $0.asType(.float32) }
        try model.update(parameters: ModuleParameters.unflattened(float32Weights), verify: .noUnusedKeys)
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
            if key.hasSuffix("codebook_layer_offsets") {
                // Precomputed offsets from the PyTorch implementation; embeddings
                // are indexed per-codebook here so the offsets are not needed
                // (the Python port drops this key too).
                continue
            }
            if key == "audio_embeddings.weight" || key == "audio_heads.weight" {
                // Fused checkpoint layout [C*V, H]: split into C per-codebook slices
                // (parity with the Python port's sanitize).
                let prefix = key == "audio_embeddings.weight" ? "audio_embeddings" : "audio_heads"
                let count = audioHeads.count
                let vocabSize = value.dim(0) / count
                for i in 0..<count {
                    sanitized["\(prefix).\(i).weight"] = value[(i * vocabSize)..<((i + 1) * vocabSize)]
                }
            } else if key.hasPrefix("audio_embeddings.") || key.hasPrefix("audio_heads.") {
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
    @ModuleInfo(key: "snake1") var snake1: SnakeAlpha
    @ModuleInfo(key: "snake2") var snake2: SnakeAlpha

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
        self._snake1.wrappedValue = SnakeAlpha(channels: channels)
        self._snake2.wrappedValue = SnakeAlpha(channels: channels)
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
public final class SnakeAlpha: Module {
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
    @ModuleInfo(key: "res_unit1") var resUnit1: OmniVoiceDACResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: OmniVoiceDACResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: OmniVoiceDACResidualUnit
    @ModuleInfo(key: "snake1") var snake1: SnakeAlpha
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d

    init(inputChannels: Int, outputChannels: Int, stride: Int, kernelSize: Int) {
        self._resUnit1.wrappedValue = OmniVoiceDACResidualUnit(
            channels: inputChannels, kernelSize: kernelSize, dilation: 1
        )
        self._resUnit2.wrappedValue = OmniVoiceDACResidualUnit(
            channels: inputChannels, kernelSize: kernelSize, dilation: 3
        )
        self._resUnit3.wrappedValue = OmniVoiceDACResidualUnit(
            channels: inputChannels, kernelSize: kernelSize, dilation: 9
        )
        self._snake1.wrappedValue = SnakeAlpha(channels: inputChannels)
        self._conv1.wrappedValue = OmniVoiceConv1d(
            inChannels: inputChannels,
            outChannels: outputChannels,
            kernelSize: stride * 2,
            stride: stride,
            padding: stride / 2 + stride % 2
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resUnit1(x)
        h = resUnit2(h)
        h = resUnit3(h)
        h = snake1.callAsFunction(h)
        h = conv1(h)
        return h
    }
}

/// DAC upsampling block (Higgs Audio V2 DecoderBlock):
/// Snake1d + ConvTranspose1d + 3 ResidualUnits(dilation 1,3,9).
public final class OmniVoiceDACUpBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: SnakeAlpha
    @ModuleInfo(key: "conv_t1") var convT1: OmniVoiceConvTranspose1d
    @ModuleInfo(key: "res_unit1") var resUnit1: OmniVoiceDACResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: OmniVoiceDACResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: OmniVoiceDACResidualUnit

    init(inputChannels: Int, outputChannels: Int, stride: Int, kernelSize: Int) {
        self._snake1.wrappedValue = SnakeAlpha(channels: inputChannels)
        self._convT1.wrappedValue = OmniVoiceConvTranspose1d(
            inChannels: inputChannels,
            outChannels: outputChannels,
            kernelSize: stride * 2,
            stride: stride,
            padding: stride / 2 + stride % 2,
            outputPadding: stride % 2
        )
        self._resUnit1.wrappedValue = OmniVoiceDACResidualUnit(
            channels: outputChannels, kernelSize: kernelSize, dilation: 1
        )
        self._resUnit2.wrappedValue = OmniVoiceDACResidualUnit(
            channels: outputChannels, kernelSize: kernelSize, dilation: 3
        )
        self._resUnit3.wrappedValue = OmniVoiceDACResidualUnit(
            channels: outputChannels, kernelSize: kernelSize, dilation: 9
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convT1(snake1.callAsFunction(x))
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}

/// Higgs Audio V2 acoustic encoder: conv1 → down blocks → snake1 → conv2.
public final class OmniVoiceDACAcousticEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d
    @ModuleInfo var block: [OmniVoiceDACDownBlock]
    @ModuleInfo(key: "snake1") var snake1: SnakeAlpha
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

        self._snake1.wrappedValue = SnakeAlpha(channels: currentChannels)
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
        for b in block {
            h = b(h)
        }
        h = snake1.callAsFunction(h)
        return conv2(h)
    }
}

/// Higgs Audio V2 acoustic decoder: conv1 → up blocks → snake1 → conv2.
public final class OmniVoiceDACAcousticDecoder: Module {
    @ModuleInfo(key: "conv1") var conv1: OmniVoiceConv1d
    @ModuleInfo var block: [OmniVoiceDACUpBlock]
    @ModuleInfo(key: "snake1") var snake1: SnakeAlpha
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

        self._snake1.wrappedValue = SnakeAlpha(channels: currentChannels)
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
        // Python _adjust_dac_decoder removes the final Tanh; keep output unbounded
        return conv2(h)
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

    /// Encode: [B, T, D] (NLC) -> [B, T, n_quantizers] int32 via greedy
    /// residual quantization (parity with the Python
    /// ResidualVectorQuantizer.encode: each stage quantizes the residual left
    /// by the previous stages, not the original input).
    func encode(_ z: MLXArray) -> MLXArray {
        var residual = z
        var tokens: [MLXArray] = []

        for q in quantizers {
            let codebook = q.codebook.embed  // [K, codebookDim]
            let zq = q.projectIn(residual)  // [B, T, codebookDim]

            // Squared distances to each codebook entry: [B, T, K]
            let dists =
                MLX.sum(zq * zq, axis: -1, keepDims: true)
                + MLX.sum(codebook * codebook, axis: -1)
                - 2 * MLX.matmul(zq, codebook.transposed(1, 0))
            let idx = MLX.argMin(dists, axis: -1).asType(.int32)  // [B, T]
            tokens.append(idx)

            let qVecs = MLX.take(codebook, idx.reshaped([-1]), axis: 0)  // [B*T, codebookDim]
            let recon = q.projectOut(qVecs).reshaped(residual.shape)  // [B, T, D]
            residual = residual - recon
        }

        return MLX.stacked(tokens, axis: -1)  // [B, T, n_quantizers]
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

    // Encode path (voice cloning): HuBERT semantic features fused with the
    // acoustic features. Only present when the checkpoint ships
    // semantic_model.* weights (the full mlx-community/OmniVoice does;
    // the bf16 variant is stripped).
    @ModuleInfo(key: "semantic_model") var semanticModel: OmniVoiceHubertModel?
    @ModuleInfo(key: "encoder_semantic") var encoderSemantic: OmniVoiceSemanticEncoder?
    @ModuleInfo(key: "fc") var fc: MLXNN.Linear?

    init(config: OmniVoiceAudioTokenizerConfig, includeSemantic: Bool = false) {
        self.config = config

        self._acousticEncoder.wrappedValue = OmniVoiceDACAcousticEncoder(config: config)
        self._acousticDecoder.wrappedValue = OmniVoiceDACAcousticDecoder(config: config)
        self._quantizer.wrappedValue = OmniVoiceRVQQuantizer(config: config)

        // fc2 projects quantized features (decoderHiddenSize) to decoder input (encoderHiddenSize * 4)
        self._fc2.wrappedValue = MLXNN.Linear(
            inputDimensions: config.decoderHiddenSize,
            outputDimensions: config.encoderHiddenSize * 4
        )

        if includeSemantic {
            self._semanticModel.wrappedValue = OmniVoiceHubertModel(config: config)
            self._encoderSemantic.wrappedValue = OmniVoiceSemanticEncoder(config: config)
            // fc fuses [acoustic 256 | semantic hidden] -> quantizer input
            let fusionDim = config.hiddenSize + config.encoderHiddenSize * 4
            self._fc.wrappedValue = MLXNN.Linear(
                inputDimensions: fusionDim, outputDimensions: fusionDim
            )
        } else {
            self._semanticModel.wrappedValue = nil
            self._encoderSemantic.wrappedValue = nil
            self._fc.wrappedValue = nil
        }
    }

    /// Stride factor mapping HuBERT frame rate (16 kHz / 320 = 50 fps) onto the
    /// acoustic frame rate (24 kHz / 960 = 25 fps).
    private var semanticDownsampleFactor: Int {
        let hubertFPS = Double(config.semanticSampleRate) / Double(config.downsampleFactor)
        let acousticFPS = Double(config.sampleRate) / Double(config.hopLength)
        return max(1, Int((hubertFPS / acousticFPS).rounded()))
    }

    /// Encode audio waveform to discrete tokens (parity with the Python
    /// HiggsAudioTokenizer.encode):
    ///   1. acoustic_encoder on the 24 kHz waveform -> [B, Ta, 256]
    ///   2. sinc-resample to 16 kHz, pad downsample_factor/2, HuBERT
    ///      (mean of all hidden states), stride-slice to 25 fps,
    ///      encoder_semantic CNN -> [B, Ts, hidden]
    ///   3. concat [acoustic | semantic] -> fc -> residual RVQ encode
    /// - Parameter audio: [samples] or [1, samples] at 24 kHz
    /// - Returns: [num_codebooks, seq_len]
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        guard let semanticModel, let encoderSemantic, let fc else {
            throw AudioGenerationError.modelNotInitialized(
                "audio tokenizer checkpoint lacks the semantic encode path (semantic_model.*) "
                    + "required for voice cloning; use the full mlx-community/OmniVoice checkpoint"
            )
        }

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
        let wav32 = wav.asType(.float32)
        let batchSize = wav32.shape[0]

        // 1. Acoustic features: [B, 1, T] -> [B, 256, Ta] (NCL) -> [B, Ta, 256]
        let acoustic = acousticEncoder(wav32).transposed(0, 2, 1)

        // 2. Semantic features. Sinc resampling matches torchaudio
        //    (AVAudioConverter does not); HuBERT input is padded by
        //    downsample_factor/2 on both sides.
        var resampled: [[Float]] = []
        for b in 0..<batchSize {
            let samples = wav32[b, 0].asArray(Float.self)
            resampled.append(
                omniVoiceSincResample(
                    samples, from: config.sampleRate, to: config.semanticSampleRate))
        }
        let targetLen = resampled.map(\.count).min() ?? 0
        let hubertPad = config.downsampleFactor / 2
        var flat = [Float]()
        flat.reserveCapacity(batchSize * (targetLen + 2 * hubertPad))
        for r in resampled {
            flat.append(contentsOf: [Float](repeating: 0, count: hubertPad))
            flat.append(contentsOf: r[0..<targetLen])
            flat.append(contentsOf: [Float](repeating: 0, count: hubertPad))
        }
        let audio16k = MLXArray(flat).reshaped([batchSize, targetLen + 2 * hubertPad])

        var semantic = semanticModel.meanHiddenStates(audio16k)  // [B, Th, hidden]
        let dsf = semanticDownsampleFactor
        if dsf > 1 {
            let indices = MLXArray(
                Swift.stride(from: 0, to: semantic.shape[1], by: dsf).map(Int32.init))
            semantic = MLX.take(semantic, indices, axis: 1)
        }
        semantic = encoderSemantic(semantic)  // [B, Ts, hidden]

        // 3. Fuse, project, quantize.
        let timeSteps = min(acoustic.shape[1], semantic.shape[1])
        let fused = MLX.concatenated(
            [acoustic[0..., 0..<timeSteps, 0...], semantic[0..., 0..<timeSteps, 0...]],
            axis: -1
        )
        let codes = quantizer.encode(fc(fused))  // [B, T, n_codebooks]

        // Return [n_codebooks, T] (squeeze batch dim)
        return codes[0].transposed(1, 0)
    }

    /// Decode discrete tokens back to audio waveform.
    /// - Parameter tokens: [num_codebooks, seq_len]
    /// - Returns: [samples]
    public func decode(_ tokens: MLXArray) throws -> MLXArray {
        // Add batch dim: [n_codebooks, T] -> [1, n_codebooks, T]
        let batchedTokens = tokens.reshaped([1, tokens.shape[0], tokens.shape[1]])

        // RVQ decode: [1, n_codebooks, T] -> [1, D, T] where D=1024
        let z = quantizer.decode(batchedTokens)

        // fc2 project: [1, D, T] -> [1, D', T] where D'=256
        // fc2.weight shape is [256, 1024] (outputDim x inputDim)
        // Decode uses: fc2(zNLC) = zNLC @ W.T + b where zNLC=[B,T,1024], W.T=[1024,256], b=[256]
        let zNLC = z.transposed(0, 2, 1)  // [B, T, 1024]
        let hNLC = MLX.matmul(zNLC, fc2.weight.transposed(1, 0))  // [B, T, 256]
        let h = (hNLC + (fc2.bias ?? MLXArray.zeros([1]))).transposed(0, 2, 1)  // [B, 256, T]

        // Decoder: [1, D', T] -> [1, 1, T']
        let audio = acousticDecoder(h)

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
        let dropExact: Set<String> = ["semantic_model.masked_spec_embed"]
        let dropSuffixes = [".embed_avg", ".cluster_size", ".inited"]

        for (key, v) in weights {
            var k = key
            // Explicit drops
            if dropExact.contains(k) { continue }
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
                // so we reshape at runtime in SnakeAlpha.callAsFunction instead.
                // NOTE: we do NOT transpose 3D conv weights here because
                // OmniVoiceConv1d and OmniVoiceConvTranspose1d already
                // transpose from PyTorch [out,in,k] / [in,out,k] to MLX
                // [out,k,in] at runtime.
            }

            // === Semantic path (HuBERT) ===
            // Remap parametrized weight-norm keys; conv weights keep the
            // PyTorch [out, in/groups, K] layout (transposed at runtime by
            // OmniVoiceSemanticConv1d / OmniVoiceWeightNormConv1d).
            if k.hasPrefix("semantic_model.") {
                if k.contains(".parametrizations.weight.original0") {
                    k = k.replacingOccurrences(
                        of: ".parametrizations.weight.original0", with: ".weight_g")
                } else if k.contains(".parametrizations.weight.original1") {
                    k = k.replacingOccurrences(
                        of: ".parametrizations.weight.original1", with: ".weight_v")
                }
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

        // Pull in nested semantic_model_config (HuBERT) values if present
        if let semantic = configDict["semantic_model_config"] as? [String: Any] {
            for key in ["hidden_size", "num_hidden_layers", "num_attention_heads",
                        "intermediate_size", "conv_dim", "conv_kernel", "conv_stride"] {
                if configDict[key] == nil, let val = semantic[key] {
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

        // The semantic encode path (voice cloning) is only built when the
        // checkpoint actually ships its weights (the bf16 release is stripped).
        let includeSemantic = rawWeights.keys.contains { $0.hasPrefix("semantic_model.") }
        let tokenizer = OmniVoiceAudioTokenizer(config: config, includeSemantic: includeSemantic)
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

        // Codec weights run as float32 (reference checkpoint ships fp32).
        let float32Weights = weights.mapValues { $0.asType(.float32) }
        try tokenizer.update(parameters: ModuleParameters.unflattened(float32Weights), verify: .noUnusedKeys)
        eval(tokenizer)

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
