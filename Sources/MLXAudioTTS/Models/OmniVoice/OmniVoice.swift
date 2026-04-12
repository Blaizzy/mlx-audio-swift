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
        return MLX.where(
            audioMask.reshaped([audioMask.shape[0], audioMask.shape[1], 1]),
            audioEmbeds!,
            textEmbeds
        )
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

        // Run through LLM (causal masking is handled internally by Qwen3Model)
        let hiddenStates = llm.forwardWithEmbeddings(
            inputsEmbeds: inputsEmbeds,
            cache: cache
        )

        // Project to audio codebook logits via per-codebook heads
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        var logitsPerCodebook: [MLXArray] = []
        for head in audioHeads {
            let logits = head(hiddenStates)  // [B, S, V]
            logitsPerCodebook.append(logits.reshaped([batchSize, seqLen, 1, config.audioVocabSize]))
        }
        let audioLogits = MLX.concatenated(logitsPerCodebook, axis: 2)  // [B, C, S, V]

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

        let inputIds = prepared.inputIds
        let audioMask = prepared.audioMask
        let condLength = inputIds.shape[2]

        // 4. Build batched inputs for CFG (cond + uncond)
        let B = 1
        let numCodebooks = config.numAudioCodebook
        let targetLen = numTargetTokens

        // Unconditional input: only the target portion
        let uncondInputIds = inputIds[0..., 0..., (condLength - targetLen)...]
        let uncondAudioMask = audioMask[0..., (condLength - targetLen)...]

        // Concatenate cond + uncond along batch axis
        let batchInputIds = MLX.concatenated([inputIds, uncondInputIds], axis: 0)
        let batchAudioMask = MLX.concatenated(
            [audioMask, uncondAudioMask],
            axis: 0
        )

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

            // Forward pass
            let logits = forward(
                inputIds: batchInputIds,
                audioMask: batchAudioMask
            ).asType(.float32)

            // Extract conditional and unconditional logits for the target region
            let cLogits = logits[0, 0..., (condLength - targetLen)..<condLength, 0...]  // [C, T, V]
            let uLogits = logits[1, 0..., 0..<targetLen, 0...]  // [C, T, V]

            // Add batch dimension back for scoring
            let cLogitsBatch = cLogits.reshaped([1, numCodebooks, targetLen, config.audioVocabSize])
            let uLogitsBatch = uLogits.reshaped([1, numCodebooks, targetLen, config.audioVocabSize])

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
            let topkIndices = MLX.argPartition(MLXArray(-1.0) * flatScores.asType(.float32), kth: k - 1, axis: 0)[0..., ..<k]

            // Build update mask: positions in topkIndices get updated
            let linearTopkIndices = topkIndices.reshaped([-1])
            let updateMask = MLXArray.zeros([flatTokens.shape[0]], type: Bool.self)
            var updatedTokens = flatTokens

            // Mark positions to update
            for idx in linearTopkIndices.asArray(Int.self) {
                if idx >= 0 && idx < flatTokens.shape[0] {
                    updatedTokens[idx] = flatPreds[idx]
                }
            }

            tokens[0] = updatedTokens.reshaped([numCodebooks, targetLen])

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
        let outputTokens = tokens[0, 0..., 0..., 0..<targetLen]
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
        styleIds = MLX.broadcast(styleIds.reshaped([1, 1, -1]), to: [1, numCodebooks, styleIds.shape[0]])

        // Build text tokens
        let fullText = combineText(refText: refText, text: text)
        let wrappedText = "<|text_start|>\(fullText)<|text_end|>"
        let textTokenIds = try tokenizeText(wrappedText)
        var textIds = MLXArray(textTokenIds.map { Int32($0) })
        textIds = textIds.reshaped([1, -1])
        textIds = MLX.broadcast(textIds.reshaped([1, 1, -1]), to: [1, numCodebooks, textIds.shape[0]])

        // Target: all MASK
        let targetIds = MLXArray.full(
            [1, numCodebooks, numTargetTokens],
            values: MLXArray(Int32(config.audioMaskId))
        )

        // Concatenate: [style, text, ref_audio (optional), target]
        var parts: [MLXArray] = [styleIds, textIds]
        if let refTok = refAudioTokens {
            parts.append(refTok.reshaped([1, refTok.shape[0], refTok.shape[1]]))
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

        var condAudioMask = MLXArray.zeros([1, totalLength], type: Bool.self)
        condAudioMask[0..., audioStartIdx...] = MLXArray.ones([totalLength - audioStartIdx], type: Bool.self)

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
        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: configData)

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

// MARK: - OmniVoice ConvTranspose1d (PyTorch weight convention)

/// Simple ConvTranspose1d matching PyTorch checkpoint weight layout [in, out, kernel].
/// Used by OmniVoice decoder which does NOT use weight normalization.
final class OmniVoiceConvTranspose1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let strideVal: Int
    let paddingVal: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.strideVal = stride
        self.paddingVal = padding

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [inChannels, outChannels, kernelSize]
        )
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, C_in, L]  (channel-first, matching Conv1d layout)
        var h = MLX.convTransposed1d(x, weight, stride: strideVal, padding: paddingVal)
        if let b = bias {
            h = h + b.reshaped(1, -1, 1)
        }
        return h
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
    @ModuleInfo(key: "conv1") var conv1: MLXNN.Conv1d
    @ModuleInfo(key: "conv2") var conv2: MLXNN.Conv1d
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "snake2") var snake2: snakeAlpha

    init(channels: Int, kernelSize: Int, dilation: Int) {
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: ((kernelSize - 1) * dilation + 1) / 2
        )
        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: ((kernelSize - 1) * dilation + 1) / 2
        )
        self._snake1.wrappedValue = snakeAlpha(channels: channels)
        self._snake2.wrappedValue = snakeAlpha(channels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1.callAsFunction(conv1(x))
        h = snake2.callAsFunction(conv2(h))
        return x + h
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
    @ModuleInfo(key: "conv1") var conv1: MLXNN.Conv1d

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
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
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

/// DAC-style acoustic encoder: Conv1d + downsampling blocks + final conv.
public final class OmniVoiceDACAcousticEncoder: Module {
    @ModuleInfo(key: "conv_pre") var convPre: MLXNN.Conv1d
    @ModuleInfo var block: [OmniVoiceDACDownBlock]
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "conv1") var conv1: MLXNN.Conv1d
    @ModuleInfo(key: "conv2") var conv2: MLXNN.Conv1d
    @ModuleInfo(key: "conv_post") var convPost: MLXNN.Conv1d

    init(config: OmniVoiceAudioTokenizerConfig) {
        let hiddenSize = config.encoderHiddenSize
        let downsamplingRatios = config.downsamplingRatios

        self._convPre.wrappedValue = MLXNN.Conv1d(
            inputChannels: 1,
            outputChannels: hiddenSize,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )

        var blocks: [OmniVoiceDACDownBlock] = []
        var currentChannels = hiddenSize
        for (i, stride) in downsamplingRatios.enumerated() {
            let outChannels = (i == downsamplingRatios.count - 1) ? config.codebookDim : currentChannels * 2
            blocks.append(OmniVoiceDACDownBlock(
                inputChannels: currentChannels,
                outputChannels: outChannels,
                stride: stride,
                kernelSize: config.kernelSize
            ))
            currentChannels = outChannels
        }
        self._block.wrappedValue = blocks

        // Final layers after down blocks
        self._snake1.wrappedValue = snakeAlpha(channels: currentChannels)
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: currentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: currentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self._convPost.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: currentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, 1, T] -> [B, D, T']
        var h = convPre(x)
        for b in block {
            h = b(h)
        }
        h = snake1.callAsFunction(h)
        h = conv1(h)
        h = conv2(h)
        h = convPost(h)
        return h
    }
}

/// DAC-style acoustic decoder: ConvTranspose1d + upsampling blocks.
public final class OmniVoiceDACAcousticDecoder: Module {
    @ModuleInfo var block: [OmniVoiceDACUpBlock]
    @ModuleInfo(key: "snake1") var snake1: snakeAlpha
    @ModuleInfo(key: "conv1") var conv1: MLXNN.Conv1d
    @ModuleInfo(key: "conv2") var conv1_: MLXNN.Conv1d
    @ModuleInfo(key: "conv_post") var convPost: MLXNN.Conv1d

    init(config: OmniVoiceAudioTokenizerConfig) {
        let hiddenSize = config.decoderHiddenSize
        let upsamplingRatios = config.upsamplingRatios

        var blocks: [OmniVoiceDACUpBlock] = []
        var currentChannels = hiddenSize
        for (i, stride) in upsamplingRatios.enumerated() {
            let outChannels = (i == upsamplingRatios.count - 1) ? 1 : currentChannels / 2
            blocks.append(OmniVoiceDACUpBlock(
                inputChannels: currentChannels,
                outputChannels: outChannels,
                stride: stride,
                kernelSize: config.kernelSize
            ))
            currentChannels = outChannels
        }
        self._block.wrappedValue = blocks

        // Final layers after up blocks (matching checkpoint)
        self._snake1.wrappedValue = snakeAlpha(channels: currentChannels)
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: currentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self._conv1_.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: currentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self._convPost.wrappedValue = MLXNN.Conv1d(
            inputChannels: currentChannels,
            outputChannels: 1,
            kernelSize: 7,
            stride: 1,
            padding: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, D, T] -> [B, 1, T']
        var h = x
        for b in block {
            h = snakeActivation(b(h))
        }
        h = snake1.callAsFunction(h)
        h = conv1(h)
        h = conv1_(h)
        h = MLX.tanh(convPost(h))
        return h
    }
}

/// Residual Vector Quantization.
public final class OmniVoiceRVQQuantizer: Module {
    @ModuleInfo var codebook: MLXArray  // [n_codebooks, codebook_size, codebook_dim]

    init(config: OmniVoiceAudioTokenizerConfig) {
        let nCodebooks = config.nCodebooks
        let codebookSize = config.codebookSize
        let codebookDim = config.codebookDim

        // Initialize codebooks with small random values
        var cbs: [MLXArray] = []
        for _ in 0..<nCodebooks {
            let scale = sqrt(1.0 / Float(codebookDim))
            let cb = MLXRandom.uniform(low: -scale, high: scale, [codebookSize, codebookDim])
            cbs.append(cb)
        }
        self._codebook.wrappedValue = MLX.stacked(cbs)  // [n_codebooks, codebook_size, codebook_dim]
    }

    /// Quantize: [B, D, T] -> (codes [B, n_codebooks, T], quantized [B, D, T])
    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        let batchSize = z.shape[0]
        let codebookDim = z.shape[1]
        let seqLen = z.shape[2]
        let nCodebooks = codebook.shape[0]
        let codebookSize = codebook.shape[1]

        var residual = z  // [B, D, T]
        var allCodes: [MLXArray] = []
        var quantized = MLXArray.zeros([batchSize, codebookDim, seqLen])

        for cbIdx in 0..<nCodebooks {
            let cb = codebook[cbIdx]  // [codebook_size, D]

            // [B, D, T] -> [B, T, D] for distance computation
            let zPermute = z.transposed(0, 2, 1)  // [B, T, D]
            let zFlat = zPermute.reshaped([batchSize * seqLen, codebookDim])

            // Compute distances: [B*T, K]
            // [B*T, 1, D] - [1, K, D] -> [B*T, K]
            let diff = zFlat.reshaped([zFlat.shape[0], 1, zFlat.shape[1]])
                - cb.reshaped([1, codebookSize, codebookDim])
            let dist = MLX.sum(diff * diff, axis: -1)

            // Nearest codebook index
            let codes = MLX.argMin(dist, axis: -1)  // [B*T]
            let codes2d = codes.reshaped([batchSize, seqLen])
            allCodes.append(codes2d)

            // Gather quantized vectors
            let q = MLX.take(cb, codes, axis: 0)  // [B*T, D]
            let q3d = q.reshaped([batchSize, seqLen, codebookDim]).transposed(0, 2, 1)

            quantized = quantized + q3d
            residual = residual - q3d
        }

        let codes = MLX.stacked(allCodes, axis: 1)  // [B, n_codebooks, T]
        return (codes, quantized)
    }

    /// Decode: [B, n_codebooks, T] -> [B, D, T]
    func decode(_ codes: MLXArray) -> MLXArray {
        let batchSize = codes.shape[0]
        let nCodebooks = codes.shape[1]
        let seqLen = codes.shape[2]
        let codebookDim = codebook.shape[2]

        var quantized = MLXArray.zeros([batchSize, codebookDim, seqLen])

        for cbIdx in 0..<nCodebooks {
            let cb = codebook[cbIdx]  // [codebook_size, D]
            let cbCodes = codes[0..., cbIdx, 0...]  // [B, T]
            let flatCodes = cbCodes.reshaped([-1])  // [B*T]

            let q = MLX.take(cb, flatCodes, axis: 0)  // [B*T, D]
            let q3d = q.reshaped([batchSize, seqLen, codebookDim]).transposed(0, 2, 1)

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
    @ModuleInfo(key: "quantizers") var quantizer: OmniVoiceRVQQuantizer
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
            wav = wav.reshaped([1, 1, -1])
        } else if wav.ndim == 2 {
            wav = wav.reshaped([1, 1, wav.shape[1]])
        }

        // Encoder: [B, 1, T] -> [B, D, T']
        let z = acousticEncoder(wav)

        // RVQ: [B, D, T'] -> (codes [B, n_codebooks, T'], quantized [B, D, T'])
        let (codes, _) = quantizer(z)

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

        // fc2 project: [1, D, T] -> [1, D', T]
        let h = fc2(z.transposed(0, 2, 1)).transposed(0, 2, 1)

        // Decoder: [1, D', T] -> [1, 1, T']
        let audio = acousticDecoder(h)

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
        try tokenizer.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        eval(tokenizer)

        return tokenizer
    }
}

