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
    private var llm: Qwen3Model

    /// Audio embeddings: maps shifted audio token IDs to hidden states
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding

    /// Audio heads: projects hidden states to codebook logits
    @ModuleInfo(key: "audio_heads") var audioHeads: Linear

    /// Codebook layer offsets for shifting audio token IDs
    private var codebookLayerOffsets: MLXArray

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

    // MARK: - OmniVoice-specific parameters

    private var numStep: Int = 32
    private var cfgGuidanceScale: Float = 2.0
    private var cfgSpeed: Float = 1.0
    private var cfgDuration: Float?
    private var cfgTShift: Float = 0.1
    private var cfgDenoise: Bool = true
    private var cfgPostprocessOutput: Bool = true
    private var cfgLayerPenaltyFactor: Float = 5.0
    private var cfgPositionTemperature: Float = 5.0
    private var cfgClassTemperature: Float = 0.0

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
        self.llm = Qwen3Model(llmConfigWrapper)

        // Audio embeddings: [num_codebooks * vocab_size, hidden_size]
        let embedDim = config.numAudioCodebook * config.audioVocabSize
        self._audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: embedDim,
            dimensions: llmConfig.hiddenSize
        )

        // Audio heads: Linear(hidden_size, num_codebooks * vocab_size, bias=false)
        self._audioHeads.wrappedValue = Linear(
            inputDimensions: llmConfig.hiddenSize,
            outputDimensions: config.numAudioCodebook * config.audioVocabSize,
            bias: false
        )

        // Codebook layer offsets
        self.codebookLayerOffsets = MLXArray(
            (0..<config.numAudioCodebook).map { Int32($0 * config.audioVocabSize) }
        ).reshaped([1, config.numAudioCodebook, 1])

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

        // Shift audio IDs by codebook layer offsets
        let shiftedIds = (inputIds * audioMask.reshaped([inputIds.shape[0], 1, inputIds.shape[2]]))
            + codebookLayerOffsets

        // Embed and sum across codebooks
        let audioEmbeds = audioEmbeddings(shiftedIds).sum(axis: 1)

        // Where audio: use audio_embeds, else use text_embeds
        return MLX.where(
            audioMask.reshaped([audioMask.shape[0], audioMask.shape[1], 1]),
            audioEmbeds,
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

        // Project to audio codebook logits
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]
        let logitsFlat = audioHeads(hiddenStates)
        let audioLogits = logitsFlat
            .reshaped([batchSize, seqLen, config.numAudioCodebook, config.audioVocabSize])
            .transposed(0, 2, 1, 3)

        return audioLogits
    }

    // MARK: - SpeechGenerationModel Protocol

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
        return try await generateAudio(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
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
                    generationParameters: generationParameters
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

    // MARK: - Generation Config

    public func setGenerationConfig(
        numStep: Int = 32,
        guidanceScale: Float = 2.0,
        speed: Float = 1.0,
        duration: Float? = nil,
        tShift: Float = 0.1,
        denoise: Bool = true,
        postprocessOutput: Bool = true,
        layerPenaltyFactor: Float = 5.0,
        positionTemperature: Float = 5.0,
        classTemperature: Float = 0.0
    ) {
        self.numStep = numStep
        self.cfgGuidanceScale = guidanceScale
        self.cfgSpeed = speed
        self.cfgDuration = duration
        self.cfgTShift = tShift
        self.cfgDenoise = denoise
        self.cfgPostprocessOutput = postprocessOutput
        self.cfgLayerPenaltyFactor = layerPenaltyFactor
        self.cfgPositionTemperature = positionTemperature
        self.cfgClassTemperature = classTemperature
    }

    // MARK: - Generation

    private func generateAudio(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // 1. Encode reference audio to tokens if provided
        var refAudioTokens: MLXArray?
        if let refAudio {
            guard let audioTok = audioTokenizer else {
                throw AudioGenerationError.modelNotInitialized("Audio tokenizer not loaded for refAudio")
            }
            refAudioTokens = try audioTok.encode(refAudio)
        }

        // 2. Estimate target token count
        let numRefTokCount = refAudioTokens?.shape.last ?? 0
        let numTargetTokens = estimateTargetTokens(
            text: text,
            refText: refText,
            numRefAudioTokens: numRefTokCount,
            speed: self.cfgSpeed
        )

        // 3. Prepare inference inputs
        let prepared = try prepareInferenceInputs(
            text: text,
            numTargetTokens: numTargetTokens,
            refText: refText,
            refAudioTokens: refAudioTokens,
            language: language,
            instruct: voice
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
        let timesteps = getTimeSteps(tStart: 0.0, tEnd: 1.0, numStep: self.numStep + 1, tShift: self.cfgTShift)

        let totalMask = targetLen * numCodebooks
        var rem = totalMask
        var schedule: [Int] = []
        for step in 0..<numStep {
            let k: Int
            if step == numStep - 1 {
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
        for step in 0..<numStep {
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
                guidanceScale: self.cfgGuidanceScale,
                classTemperature: self.cfgClassTemperature
            )

            // Apply layer penalty
            let adjustedScores = scores - (layerIds.asType(.float32) * self.cfgLayerPenaltyFactor)

            // Gumbel sampling for position selection
            var finalScores = adjustedScores
            if self.cfgPositionTemperature > 0.0 {
                finalScores = gumbelSample(logits: adjustedScores, temperature: self.cfgPositionTemperature)
            }

            // Mask out already-filled positions
            let mask = tokens[0] .!= Int32(config.audioMaskId)
            let maskInf = MLX.where(mask, MLXArray(Float(-Float.infinity)), finalScores).asType(.float32)

            // Flatten for top-k selection
            let flatScores = maskInf.reshaped([-1])
            let flatTokens = tokens[0].reshaped([-1])
            let flatPreds = predTokens[0].reshaped([-1])

            // Select top-k positions to unmask
            let topkIndices = MLX.argPartition(MLXArray(-1.0, dtype: .float32) * flatScores.asType(.float32), kth: k - 1, axis: 0)[0..., ..<k]

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
        guard let audioTok = audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Audio tokenizer not loaded for decoding")
        }
        let outputTokens = tokens[0, 0..., 0..., 0..<targetLen]
        let audio = try audioTok.decode(outputTokens)

        // 9. Post-process
        return postProcessAudio(audio, refRms: nil)
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
        instruct: String?
    ) throws -> (inputIds: MLXArray, audioMask: MLXArray) {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let numCodebooks = config.numAudioCodebook

        // Build style tokens
        var styleText = ""
        if self.cfgDenoise && refAudioTokens != nil {
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

    private func postProcessAudio(_ audio: MLXArray, refRms: Float?) -> MLXArray {
        var result = audio

        if let refRms, refRms < 0.1 {
            result = result * MLXArray(refRms / 0.1)
        } else if refRms == nil {
            let peak = MLX.abs(result).max().item(Float.self)
            if peak > 1e-6 {
                result = result * MLXArray(0.5 / peak)
            }
        }

        if self.cfgPostprocessOutput {
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
            var newKey = key
            if newKey.hasPrefix("llm.") {
                newKey = String(newKey.dropFirst(4))
            }
            sanitized[newKey] = value
        }
        return sanitized
    }
}

// MARK: - OmniVoice Audio Tokenizer

/// Audio tokenizer for OmniVoice - handles encoding/decoding of audio tokens.
/// Wraps a DAC + RVQ codec for discrete audio tokenization.
public final class OmniVoiceAudioTokenizer: Module {
    let config: OmniVoiceAudioTokenizerConfig
    var codec: OmniVoiceHiggsCodec?

    init(config: OmniVoiceAudioTokenizerConfig) {
        self.config = config
    }

    /// Encode audio waveform to discrete tokens.
    /// - Parameter audio: [samples] or [1, samples]
    /// - Returns: [num_codebooks, seq_len]
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        guard let codec else {
            throw AudioGenerationError.modelNotInitialized("Audio codec not loaded")
        }
        var wav = audio
        if wav.ndim == 1 {
            wav = wav.reshaped([1, 1, -1])
        } else if wav.ndim == 2 {
            wav = wav.reshaped([1, 1, wav.shape[1]])
        }
        return codec.encode(wav)[0]
    }

    /// Decode discrete tokens back to audio waveform.
    /// - Parameter tokens: [num_codebooks, seq_len]
    /// - Returns: [samples]
    public func decode(_ tokens: MLXArray) throws -> MLXArray {
        guard let codec else {
            throw AudioGenerationError.modelNotInitialized("Audio codec not loaded")
        }
        let batchedTokens = tokens.reshaped([1, tokens.shape[0], tokens.shape[1]])
        return codec.decode(batchedTokens).reshaped([-1])
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
        let sanitized = tokenizer.sanitizeCodecWeights(weights)
        try tokenizer.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)
        eval(tokenizer)

        return tokenizer
    }

    func sanitizeCodecWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var newKey = key
            if newKey.hasPrefix("audio_tokenizer.") {
                newKey = String(newKey.dropFirst("audio_tokenizer.".count))
            }
            sanitized[newKey] = value
        }
        return sanitized
    }
}

// MARK: - OmniVoice Higgs Codec (DAC + RVQ)

/// Audio codec for OmniVoice: DAC-style encoder/decoder with RVQ quantization.
public final class OmniVoiceHiggsCodec: Module {
    let config: OmniVoiceAudioTokenizerConfig

    @ModuleInfo(key: "encoder") var encoder: OmniVoiceDACLikeEncoder
    @ModuleInfo(key: "quantizer") var quantizer: OmniVoiceRVQQuantizer
    @ModuleInfo(key: "decoder") var decoder: OmniVoiceDACLikeDecoder

    init(config: OmniVoiceAudioTokenizerConfig) {
        self.config = config

        self._encoder.wrappedValue = OmniVoiceDACLikeEncoder(
            inputChannels: 1,
            channels: 64,
            strides: config.downsamplingRatios,
            latentDim: config.codebookDim
        )

        self._quantizer.wrappedValue = OmniVoiceRVQQuantizer(
            codebookSize: config.codebookSize,
            codebookDim: config.codebookDim,
            nCodebooks: config.nCodebooks
        )

        self._decoder.wrappedValue = OmniVoiceDACLikeDecoder(
            inputChannels: config.codebookDim,
            channels: 512,
            strides: config.upsamplingRatios,
            outputChannels: 1
        )
    }

    func encode(_ waveform: MLXArray) -> MLXArray {
        let z = encoder(waveform)
        let (codes, _) = quantizer(z)
        return codes
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let z = quantizer.decode(codes)
        return decoder(z)
    }
}

// MARK: - DAC-like Encoder

public final class OmniVoiceDACLikeEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: MLXNN.Conv1d
    @ModuleInfo(key: "down_blocks") var downBlocks: [OmniVoiceDownBlock]
    @ModuleInfo(key: "conv_out") var convOut: MLXNN.Conv1d

    init(inputChannels: Int, channels: Int, strides: [Int], latentDim: Int) {
        self._convIn.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputChannels,
            outputChannels: channels,
            kernelSize: 7,
            padding: 3
        )

        var blocks: [OmniVoiceDownBlock] = []
        var currentChannels = channels
        for (i, stride) in strides.enumerated() {
            let outChannels = (i == strides.count - 1) ? latentDim : currentChannels * 2
            blocks.append(OmniVoiceDownBlock(
                inChannels: currentChannels,
                outChannels: outChannels,
                stride: stride
            ))
            currentChannels = outChannels
        }
        self._downBlocks.wrappedValue = blocks

        self._convOut.wrappedValue = MLXNN.Conv1d(
            inputChannels: latentDim,
            outputChannels: latentDim,
            kernelSize: 3,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in downBlocks {
            h = block(h)
        }
        return snakeActivation(convOut(h))
    }
}

// MARK: - DAC-like Decoder

public final class OmniVoiceDACLikeDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: MLXNN.Conv1d
    @ModuleInfo(key: "up_blocks") var upBlocks: [OmniVoiceUpBlock]
    @ModuleInfo(key: "conv_out") var convOut: MLXNN.Conv1d

    init(inputChannels: Int, channels: Int, strides: [Int], outputChannels: Int) {
        self._convIn.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputChannels,
            outputChannels: channels,
            kernelSize: 7,
            padding: 3
        )

        var blocks: [OmniVoiceUpBlock] = []
        var currentChannels = channels
        for (i, stride) in strides.enumerated() {
            let outChannels = (i == strides.count - 1) ? outputChannels : currentChannels / 2
            blocks.append(OmniVoiceUpBlock(
                inChannels: currentChannels,
                outChannels: outChannels,
                stride: stride
            ))
            currentChannels = outChannels
        }
        self._upBlocks.wrappedValue = blocks

        self._convOut.wrappedValue = MLXNN.Conv1d(
            inputChannels: outputChannels,
            outputChannels: outputChannels,
            kernelSize: 7,
            padding: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in upBlocks {
            h = block(h)
        }
        return MLX.tanh(convOut(h))
    }
}

// MARK: - Down Block (Encoder)

public final class OmniVoiceDownBlock: Module {
    @ModuleInfo(key: "conv_1") var conv1: MLXNN.Conv1d
    @ModuleInfo(key: "conv_2") var conv2: MLXNN.Conv1d
    @ModuleInfo(key: "conv_res") var convRes: MLXNN.Conv1d?
    @ModuleInfo(key: "norm_1") var norm1: GroupNorm
    @ModuleInfo(key: "norm_2") var norm2: GroupNorm

    init(inChannels: Int, outChannels: Int, stride: Int) {
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: stride,
            stride: stride
        )
        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            padding: 1
        )
        self._norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels)
        self._norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels)

        if inChannels != outChannels {
            self._convRes.wrappedValue = MLXNN.Conv1d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: 1,
                stride: stride
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        h = norm1(h)
        h = snakeActivation(h)
        h = conv2(h)
        h = norm2(h)

        let residual = convRes != nil ? convRes!(x) : x
        return snakeActivation(h + residual)
    }
}

// MARK: - Up Block (Decoder)

/// Upsampling block for the decoder using transposed convolution.
public final class OmniVoiceUpBlock: Module {
    @ModuleInfo(key: "conv_1") var conv1: DACVAEWNConvTranspose1d
    @ModuleInfo(key: "conv_2") var conv2: MLXNN.Conv1d
    @ModuleInfo(key: "conv_res") var convRes: DACVAEWNConvTranspose1d?
    @ModuleInfo(key: "norm_1") var norm1: GroupNorm
    @ModuleInfo(key: "norm_2") var norm2: GroupNorm

    init(inChannels: Int, outChannels: Int, stride: Int) {
        self._conv1.wrappedValue = DACVAEWNConvTranspose1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: stride * 2,
            stride: stride,
            padding: stride / 2
        )
        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            padding: 1
        )
        self._norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels)
        self._norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels)

        if inChannels != outChannels {
            self._convRes.wrappedValue = DACVAEWNConvTranspose1d(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: stride,
                stride: stride
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        h = norm1(h)
        h = snakeActivation(h)
        h = conv2(h)
        h = norm2(h)

        let residual: MLXArray
        if let convRes {
            residual = convRes(x)
        } else {
            residual = x
        }

        return snakeActivation(h + residual)
    }
}

// MARK: - RVQ Quantizer

/// Residual Vector Quantization for audio tokenization.
public final class OmniVoiceRVQQuantizer: Module {
    let codebookSize: Int
    let codebookDim: Int
    let nCodebooks: Int

    @ModuleInfo(key: "codebooks") var codebooks: [MLXArray]

    init(codebookSize: Int, codebookDim: Int, nCodebooks: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.nCodebooks = nCodebooks

        var cbs: [MLXArray] = []
        for _ in 0..<nCodebooks {
            let scale = sqrt(1.0 / Float(codebookDim))
            let cb = MLXRandom.uniform(low: -scale, high: scale, [codebookSize, codebookDim])
            cbs.append(cb)
        }
        self._codebooks.wrappedValue = cbs
    }

    /// Quantize: [batch, dim, seq] -> (codes [batch, n_codebooks, seq], quantized [batch, dim, seq])
    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        let batchSize = z.shape[0]
        let seqLen = z.shape[2]

        var residual = z
        var allCodes: [MLXArray] = []
        var quantized = MLXArray.zeros(z.shape)

        for cbIdx in 0..<nCodebooks {
            let cb = codebooks[cbIdx]
            let zFlat = residual.transposed(0, 2, 1).reshaped([-1, codebookDim])

            // Compute distances: [B*T, K]
            let diff = zFlat.reshaped([zFlat.shape[0], 1, zFlat.shape[1]])
                - cb.reshaped([1, codebookSize, codebookDim])
            let dist = MLX.sum(diff * diff, axis: -1)

            let codes = MLX.argMin(dist, axis: -1)
            let codes2d = codes.reshaped([batchSize, seqLen])
            allCodes.append(codes2d.reshaped([batchSize, 1, seqLen]))

            let q = MLX.take(cb, codes, axis: 0)
            let q2d = q.reshaped([batchSize, seqLen, codebookDim]).transposed(0, 2, 1)

            quantized = quantized + q2d
            residual = residual - q2d
        }

        return (MLX.concatenated(allCodes, axis: 1), quantized)
    }

    /// Decode: [batch, n_codebooks, seq] -> [batch, dim, seq]
    func decode(_ codes: MLXArray) -> MLXArray {
        let batchSize = codes.shape[0]
        let seqLen = codes.shape[2]
        var quantized = MLXArray.zeros([batchSize, codebookDim, seqLen])

        for cbIdx in 0..<nCodebooks {
            let cb = codebooks[cbIdx]
            let cbCodes = codes[0..., cbIdx, 0...]
            let flatCodes = cbCodes.reshaped([-1])
            let q = MLX.take(cb, flatCodes, axis: 0)
            let q2d = q.reshaped([batchSize, seqLen, codebookDim]).transposed(0, 2, 1)
            quantized = quantized + q2d
        }

        return quantized
    }
}

// MARK: - Snake Activation

/// Snake activation: x + (1/a) * sin(a*x)^2
func snakeActivation(_ x: MLXArray) -> MLXArray {
    let alpha: Float = 1.0
    return x + (1.0 / alpha) * MLX.square(MLX.sin(alpha * x))
}
