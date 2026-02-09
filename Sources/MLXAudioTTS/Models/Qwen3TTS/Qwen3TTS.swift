// Port of mlx_audio/tts/models/qwen3_tts/qwen3_tts.py
// Main Qwen3-TTS conditional generation model (VoiceDesign path)

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXAudioCore
import HuggingFace
import Tokenizers
import Foundation

// MARK: - Qwen3TTS Model

public final class Qwen3TTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: Qwen3TTSModelConfig
    let talker: Qwen3TTSTalkerForConditionalGeneration
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var tokenizer: Tokenizer?

    public var sampleRate: Int { config.sampleRate }

    init(config: Qwen3TTSModelConfig) {
        let talkerConfig = config.talkerConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        }()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)
    }

    // MARK: - Generation path routing

    /// The generation path that will be used based on model type and inputs.
    enum GenerationPath: Equatable, Sendable {
        case voiceDesign
        case customVoice
        case base
        case icl
    }

    /// Determines which generation path to use based on config, inputs, and encoder availability.
    ///
    /// - Parameters:
    ///   - refAudio: Reference audio for voice cloning (optional).
    ///   - refText: Reference text transcript for voice cloning (optional).
    /// - Returns: The generation path to use.
    /// - Throws: `AudioGenerationError.invalidInput` if `ttsModelType` is unknown.
    func resolveGenerationPath(refAudio: MLXArray?, refText: String?) throws -> GenerationPath {
        switch config.ttsModelType {
        case "voice_design":
            return .voiceDesign
        case "custom_voice":
            return .customVoice
        case "base":
            if refAudio != nil, refText != nil, speechTokenizer?.hasEncoder == true {
                return .icl
            } else {
                return .base
            }
        default:
            throw AudioGenerationError.invalidInput(
                "Unknown tts_model_type: '\(config.ttsModelType)'. Expected 'voice_design', 'custom_voice', or 'base'."
            )
        }
    }

    // MARK: - SpeechGenerationModel protocol

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }

        let lang = language ?? "auto"
        let temp = generationParameters.temperature
        let topP = generationParameters.topP
        let repPenalty = generationParameters.repetitionPenalty ?? 1.05
        let maxTokens = generationParameters.maxTokens ?? 4096

        let path = try resolveGenerationPath(refAudio: refAudio, refText: refText)

        switch path {
        case .voiceDesign:
            // VoiceDesign: voice parameter is the instruct (voice description)
            return generateVoiceDesign(
                text: text,
                instruct: voice,
                language: lang,
                temperature: temp,
                topP: topP,
                repetitionPenalty: repPenalty,
                maxTokens: maxTokens
            )

        case .customVoice:
            return try generateCustomVoice(
                text: text,
                speaker: voice,
                language: lang,
                temperature: temp,
                topP: topP,
                repetitionPenalty: repPenalty,
                maxTokens: maxTokens
            )

        case .base:
            return try generateBase(
                text: text,
                voice: voice,
                language: lang,
                temperature: temp,
                topP: topP,
                repetitionPenalty: repPenalty,
                maxTokens: maxTokens
            )

        case .icl:
            return try generateICL(
                text: text,
                refAudio: refAudio!,
                refText: refText!,
                language: lang,
                temperature: temp,
                topP: topP,
                repetitionPenalty: repPenalty,
                maxTokens: maxTokens
            )
        }
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                guard speechTokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
                }
                guard tokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
                }

                let lang = language ?? "auto"
                let temp = generationParameters.temperature
                let topP = generationParameters.topP
                let repPenalty = generationParameters.repetitionPenalty ?? 1.05
                let maxTokens = generationParameters.maxTokens ?? 4096

                let path = try resolveGenerationPath(refAudio: refAudio, refText: refText)
                let audio: MLXArray

                switch path {
                case .voiceDesign:
                    audio = generateVoiceDesign(
                        text: text,
                        instruct: voice,
                        language: lang,
                        temperature: temp,
                        topP: topP,
                        repetitionPenalty: repPenalty,
                        maxTokens: maxTokens,
                        onToken: { tokenId in
                            continuation.yield(.token(tokenId))
                        },
                        onInfo: { info in
                            continuation.yield(.info(info))
                        }
                    )

                case .customVoice:
                    // Streaming not yet implemented for CustomVoice; fall back to non-streaming
                    audio = try generateCustomVoice(
                        text: text,
                        speaker: voice,
                        language: lang,
                        temperature: temp,
                        topP: topP,
                        repetitionPenalty: repPenalty,
                        maxTokens: maxTokens
                    )

                case .icl:
                    // Streaming not yet implemented for ICL; fall back to non-streaming
                    audio = try generateICL(
                        text: text,
                        refAudio: refAudio!,
                        refText: refText!,
                        language: lang,
                        temperature: temp,
                        topP: topP,
                        repetitionPenalty: repPenalty,
                        maxTokens: maxTokens
                    )

                case .base:
                    // Streaming not yet implemented for Base; fall back to non-streaming
                    audio = try generateBase(
                        text: text,
                        voice: voice,
                        language: lang,
                        temperature: temp,
                        topP: topP,
                        repetitionPenalty: repPenalty,
                        maxTokens: maxTokens
                    )
                }

                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - Language resolution

    /// ISO 639-1 to Qwen3-TTS internal language name mapping.
    private static let isoToLanguageName: [String: String] = [
        "en": "english",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "ru": "russian",
        "pt": "portuguese",
        "es": "spanish",
        "it": "italian",
    ]

    /// Resolves a language code to the internal language string used by Qwen3-TTS.
    ///
    /// Accepts ISO 639-1 codes (e.g. "en", "zh"), full language names (e.g. "english",
    /// "chinese"), or the special value "auto". The resolved language is validated against
    /// the model's `codecLanguageId` dictionary when a config is provided.
    ///
    /// - Parameters:
    ///   - code: An ISO 639-1 language code, a full language name, or "auto".
    ///   - config: Optional talker config used to validate the resolved language against
    ///     supported languages. When nil, validation is skipped and the static mapping
    ///     is used directly.
    /// - Returns: The resolved language string, or nil if the code is unsupported.
    public static func resolveLanguage(_ code: String, config: Qwen3TTSTalkerConfig? = nil) -> String? {
        let lowered = code.lowercased()

        // "auto" is always a valid pass-through
        if lowered == "auto" {
            return "auto"
        }

        // Try ISO 639-1 mapping first
        if let mapped = isoToLanguageName[lowered] {
            // If config is provided, validate against supported languages
            if let langMap = config?.codecLanguageId {
                return langMap[mapped] != nil ? mapped : nil
            }
            return mapped
        }

        // Try as a full language name (pass-through if valid)
        if let langMap = config?.codecLanguageId {
            // Validate against config's supported languages
            return langMap[lowered] != nil ? lowered : nil
        }

        // Without config, check if it's a known full language name from our mapping values
        if isoToLanguageName.values.contains(lowered) {
            return lowered
        }

        return nil
    }

    // MARK: - Shared autoregressive generation loop

    /// Run the autoregressive generation loop shared across all generation modes
    /// (VoiceDesign, Base, CustomVoice, ICL). Takes prepared input embeddings and
    /// produces a sequence of codec code tensors.
    ///
    /// The loop performs: Talker forward pass -> sample first codebook token ->
    /// CodePredictor loop for codes 2-16 -> prepare next input embedding from
    /// trailing text hidden states and summed code embeddings.
    ///
    /// - Parameters:
    ///   - inputEmbeds: Initial input embeddings from the mode-specific preparation step.
    ///   - trailingTextHidden: Text hidden states to feed one-per-step after the first step.
    ///   - ttsPadEmbed: Pad embedding used when trailing text is exhausted.
    ///   - temperature: Sampling temperature.
    ///   - topP: Nucleus sampling threshold.
    ///   - repetitionPenalty: Penalty for repeated tokens.
    ///   - maxTokens: Maximum number of autoregressive steps.
    ///   - onToken: Optional callback invoked with each generated token ID.
    /// - Returns: Array of generated code tensors, each of shape `[1, num_code_groups]`.
    func generateFromEmbeddings(
        inputEmbeds inputEmbedsInit: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        onToken: ((Int) -> Void)? = nil
    ) -> [MLXArray] {
        let talkerConfig = config.talkerConfig!
        let cache = talker.makeCache()
        var generatedCodes = [MLXArray]()
        let eosTokenId = talkerConfig.codecEosTokenId

        // Suppress special tokens
        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var trailingIdx = 0
        var inputEmbeds = inputEmbedsInit

        for step in 0 ..< maxTokens {
            // Forward pass through talker
            let (logits, hidden) = talker(inputEmbeds, cache: cache)

            // Sample first codebook token
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedCodes.map { Int($0[0, 0].item(Int32.self)) },
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId
            )

            // Check EOS
            let tokenId = Int(nextToken[0, 0].item(Int32.self))
            onToken?(tokenId)
            if tokenId == eosTokenId { break }

            // Generate remaining codebook tokens with code predictor
            var codeTokens = [nextToken]
            let codeHidden = hidden[0..., (-1)..., 0...]
            var codeCache: [any KVCache]? = talker.codePredictor.makeCache()

            for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = talker.getInputEmbeddings()(nextToken)
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
                }

                let (codeLogits, newCache, _) = talker.codePredictor(
                    codeInput, cache: codeCache, generationStep: codeIdx
                )
                codeCache = newCache

                let nextCode = sampleToken(codeLogits, temperature: temperature, topP: topP)
                codeTokens.append(nextCode)
            }

            let allCodes = concatenated(codeTokens, axis: 1)  // [1, num_code_groups]
            generatedCodes.append(allCodes)

            codeCache = nil
            Memory.clearCache()

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx ..< (trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Sum all code embeddings for next step
            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            for (i, code) in codeTokens.dropFirst().enumerated() {
                codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code)
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds)

            if step > 0 && step % 50 == 0 {
                Memory.clearCache()
            }
        }

        return generatedCodes
    }

    // MARK: - VoiceDesign generation

    func generateVoiceDesign(
        text: String,
        instruct: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        onToken: ((Int) -> Void)? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil
    ) -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            return MLXArray.zeros([1])
        }

        // Prepare inputs
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
            text: text, language: language, instruct: instruct
        )

        // Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Run the shared autoregressive generation loop
        let startTime = Date()
        let generatedCodes = generateFromEmbeddings(
            inputEmbeds: inputEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: effectiveMaxTokens,
            onToken: onToken
        )

        guard !generatedCodes.isEmpty else {
            return MLXArray.zeros([1])
        }

        // Emit generation info
        let generateTime = Date().timeIntervalSince(startTime)
        let tokenCount = generatedCodes.count
        let info = AudioGenerationInfo(
            promptTokenCount: 0,  // Not tracked for VoiceDesign
            generationTokenCount: tokenCount,
            prefillTime: 0,  // Included in generateTime
            generateTime: generateTime,
            tokensPerSecond: Double(tokenCount) / generateTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        onInfo?(info)

        // Stack and decode
        let codes = stacked(generatedCodes, axis: 1)  // [1, seq_len, num_code_groups]

        // Streaming decode for memory efficiency
        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: 100) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]  // Remove batch dim

        // Trim to valid length
        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self)) * speechTokenizer.decodeUpsampleRate
        if validLen > 0 && validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        return audio
    }

    // MARK: - Base generation (stub)

    /// Generate audio using the Base model path (no reference audio / no ICL).
    /// - Note: Not yet implemented. Will be completed in Task 17.
    func generateBase(
        text: String,
        voice: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int
    ) throws -> MLXArray {
        throw AudioGenerationError.generationFailed(
            "Base generation is not yet implemented. This will be added in a future update."
        )
    }

    // MARK: - CustomVoice generation (stub)

    /// Generate audio using a predefined speaker from the CustomVoice model.
    /// - Note: Not yet implemented. Will be completed in Task 18.
    func generateCustomVoice(
        text: String,
        speaker: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int
    ) throws -> MLXArray {
        throw AudioGenerationError.generationFailed(
            "CustomVoice generation is not yet implemented. This will be added in a future update."
        )
    }

    // MARK: - ICL voice cloning generation (stub)

    /// Generate audio using in-context learning (voice cloning) with reference audio.
    /// - Note: Not yet implemented. Will be completed in Task 14.
    func generateICL(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int
    ) throws -> MLXArray {
        throw AudioGenerationError.generationFailed(
            "ICL voice cloning generation is not yet implemented. This will be added in a future update."
        )
    }

    // MARK: - Prepare Base/CustomVoice inputs

    /// Prepares input embeddings for Base and CustomVoice generation paths.
    ///
    /// This is the Swift port of `_prepare_generation_inputs()` from the Python
    /// reference (`qwen3_tts.py:249-404`). It handles:
    /// - Speaker ID lookup from `config.talkerConfig.spkId`
    /// - Dialect override from `config.talkerConfig.spkIsDialect`
    /// - Optional instruct embedding
    /// - Codec prefix construction: `[think/nothink, thinkBos, langId?, thinkEos, speaker?, pad, bos]`
    /// - Text embedding and trailing text hidden states
    ///
    /// Key differences from VoiceDesign's `prepareGenerationInputs()`:
    /// - VoiceDesign does NOT handle speaker IDs or dialect override
    /// - The codec prefix for Base/CustomVoice includes speaker token(s) after thinkEos
    ///
    /// - Parameters:
    ///   - text: The text to synthesise.
    ///   - language: Resolved language name (e.g. "english", "chinese", "auto").
    ///   - speaker: Optional speaker name to look up in `spkId`.
    ///   - speakerEmbedding: Optional pre-computed speaker embedding from speaker encoder
    ///     (x-vector). Takes priority over speaker name lookup.
    ///   - instruct: Optional delivery instruction (e.g. "speak in a whisper").
    /// - Returns: Tuple of (inputEmbeds, trailingTextHidden, ttsPadEmbed).
    /// - Throws: `AudioGenerationError.invalidInput` if the specified speaker is not found
    ///   in the model's `spkId` configuration.
    func prepareBaseInputs(
        text: String,
        language: String,
        speaker: String? = nil,
        speakerEmbedding: MLXArray? = nil,
        instruct: String? = nil
    ) throws -> (MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer/config not loaded")
        }

        // --- Speaker embedding ---
        // Priority: pre-computed speaker embedding > speaker name lookup > none
        var speakerEmbed: MLXArray? = nil
        var effectiveLanguage = language

        if let speakerEmbedding {
            // Pre-computed embedding from speaker encoder (used by ICL and Base with ref audio)
            speakerEmbed = speakerEmbedding
        } else if let speaker, let spkIdMap = talkerConfig.spkId,
                  let tokenIds = spkIdMap[speaker.lowercased()] {
            // Look up speaker token IDs from config.
            // Each speaker maps to an array of codec token IDs (typically one).
            // Embed each token via the codec embedding layer and sum.
            let spkIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, -1)
            let embeds = talker.getInputEmbeddings()(spkIds)  // [1, N, hidden]
            speakerEmbed = embeds.sum(axis: 1, keepDims: true)  // [1, 1, hidden]
        } else if let speaker, speaker.lowercased() != "",
                  let spkIdMap = talkerConfig.spkId, !spkIdMap.isEmpty {
            // Speaker was specified but not found in spkId
            let available = spkIdMap.keys.sorted().joined(separator: ", ")
            throw AudioGenerationError.invalidInput(
                "Speaker '\(speaker)' not found in model configuration. Available speakers: \(available)"
            )
        }

        // --- Dialect override ---
        // If the speaker has a dialect entry and the language is compatible
        // (Chinese or auto), override the language with the dialect.
        if let speaker,
           let dialectMap = talkerConfig.spkIsDialect,
           let dialect = dialectMap[speaker.lowercased()],
           (effectiveLanguage.lowercased() == "chinese" || effectiveLanguage.lowercased() == "auto"),
           let langMap = talkerConfig.codecLanguageId,
           langMap[dialect] != nil {
            effectiveLanguage = dialect
        }

        // --- Tokenize text with ChatML template ---
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).reshaped(1, -1)
        let textEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds))

        // --- TTS special token embeddings ---
        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        // --- Language ID ---
        var languageId: Int? = nil
        if effectiveLanguage.lowercased() != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[effectiveLanguage.lowercased()]
        }

        // --- Build codec prefix ---
        // Sequence: [think/nothink, thinkBos, langId?, thinkEos]
        var codecPrefill: [Int32]
        if let langId = languageId {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))

        // Suffix: [pad, bos]
        let codecEmbedSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )

        // Insert speaker embed between prefix and suffix if present
        if let speakerEmbed {
            codecEmbed = concatenated(
                [codecEmbed, speakerEmbed.reshaped(1, 1, -1), codecEmbedSuffix],
                axis: 1
            )
        } else {
            codecEmbed = concatenated([codecEmbed, codecEmbedSuffix], axis: 1)
        }

        // --- Instruct embedding ---
        var instructEmbed: MLXArray? = nil
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        // --- Role embedding (first 3 tokens: <|im_start|>assistant\n) ---
        let roleEmbed = textEmbed[0..., ..<3, 0...]

        // --- Build combined embed from pad + bos + codec prefix ---
        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        // --- Assemble full input embedding ---
        var inputEmbeds: MLXArray
        if let instructEmbed {
            inputEmbeds = concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // Add first text token (index 3) + last codec embed
        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        // --- Trailing text hidden states ---
        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    // MARK: - Prepare VoiceDesign inputs

    func prepareGenerationInputs(
        text: String,
        language: String,
        instruct: String?
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            fatalError("Tokenizer/config not loaded")
        }

        // Tokenize text with ChatML template
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).reshaped(1, -1)

        // Get text embeddings
        let textEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds))

        // TTS special tokens
        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        // Language ID
        var languageId: Int? = nil
        if language.lowercased() != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[language.lowercased()]
        }

        // Build codec prefix
        var codecPrefill: [Int32]
        if let langId = languageId {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        let codecEmbedSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )
        codecEmbed = concatenated([codecEmbed, codecEmbedSuffix], axis: 1)

        // Instruct embedding
        var instructEmbed: MLXArray? = nil
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        // Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = textEmbed[0..., ..<3, 0...]

        // Build pad/bos prefix
        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        // Full input embedding
        var inputEmbeds: MLXArray
        if let instructEmbed {
            inputEmbeds = concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // Add first text token (index 3) + last codec embed
        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        // Trailing text (tokens 4 to -5, plus EOS)
        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    // MARK: - Token sampling

    func sampleToken(
        _ logits: MLXArray,
        temperature: Float = 0.9,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int]? = nil,
        suppressTokens: [Int]? = nil,
        eosTokenId: Int? = nil
    ) -> MLXArray {
        var logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1)  // [batch, vocab_size]

        // Suppress tokens by setting to -inf
        if let suppress = suppressTokens, !suppress.isEmpty {
            let suppressArr = MLXArray(suppress.map { Int32($0) }).reshaped(1, -1)
            let negInf = MLXArray.full([1, suppress.count], values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            logitsSlice = putAlong(logitsSlice, suppressArr, values: negInf, axis: -1)
        }

        // Repetition penalty
        if let tokens = generatedTokens, !tokens.isEmpty, repetitionPenalty != 1.0 {
            let unique = Array(Set(tokens)).filter { $0 < logitsSlice.dim(-1) }
            if !unique.isEmpty {
                let tokenIds = MLXArray(unique.map { Int32($0) }).reshaped(1, -1)
                let selected = takeAlong(logitsSlice, tokenIds, axis: -1)
                let penalized = which(
                    selected .< 0,
                    selected * repetitionPenalty,
                    selected / repetitionPenalty
                )
                logitsSlice = putAlong(logitsSlice, tokenIds, values: penalized, axis: -1)
            }
        }

        // Greedy if temperature 0
        if temperature <= 0 {
            return argMax(logitsSlice, axis: -1, keepDims: true)
        }

        // Apply top-p (nucleus) sampling
        // Implementation matches mlx_lm.sample_utils.apply_top_p
        var filteredLogits = logitsSlice
        if topP > 0 && topP < 1.0 {
            // Convert to probabilities
            let probs = softmax(logitsSlice, axis: -1)

            // Sort in ASCENDING order (like Python)
            let sortedIndices = argSort(logitsSlice, axis: -1)
            let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

            // Cumulative probabilities
            let cumProbs = cumsum(sortedProbs, axis: -1)

            // Rearrange cumulative probs back to original order
            // Create inverse index mapping using putAlong
            let vocabSize = sortedIndices.dim(-1)
            let arangeIndices = MLXArray(0..<vocabSize).reshaped(1, -1).asType(Int32.self)
            let zeros = MLXArray.zeros(sortedIndices.shape, type: Int32.self)
            let inverseIndices = putAlong(zeros, sortedIndices, values: arangeIndices, axis: -1)
            let cumProbsOrigOrder = takeAlong(cumProbs, inverseIndices, axis: -1)

            // Mask tokens where cumulative prob > (1 - top_p)
            // Keep tokens that are in the top_p nucleus
            let threshold = 1.0 - topP
            let mask = cumProbsOrigOrder .> threshold
            let negInf = MLXArray.full(logitsSlice.shape, values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            filteredLogits = which(mask, logitsSlice, negInf)
        }

        // Sample with temperature
        let token = categorical(filteredLogits / temperature)
        return token.reshaped(1, 1)
    }

    // MARK: - fromPretrained

    public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3TTSModel {
        let repoID = Repo.ID(rawValue: modelRepo)!
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: "safetensors"
        )

        // Load main config
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)

        let model = Qwen3TTSModel(config: config)

        // Load talker weights
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        let modelFiles = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for file in modelFiles where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        // Sanitize and load talker weights
        let talkerWeights = Qwen3TTSTalkerForConditionalGeneration.sanitize(weights: allWeights)
        let talkerPairs = talkerWeights.map { ($0.key, $0.value) }
        try model.talker.update(parameters: ModuleParameters.unflattened(talkerPairs), verify: .noUnusedKeys)
        eval(model.talker.parameters())

        // Generate tokenizer.json if missing (Qwen3-TTS ships without it)
        let tokenizerJsonPath = modelDir.appendingPathComponent("tokenizer.json")
        if !fm.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            let hasVocab = fm.fileExists(atPath: vocabPath.path)
            let hasMerges = fm.fileExists(atPath: mergesPath.path)
            if hasVocab && hasMerges {
                do {
                    try generateTokenizerJson(
                        vocabPath: vocabPath,
                        mergesPath: mergesPath,
                        tokenizerConfigPath: modelDir.appendingPathComponent("tokenizer_config.json"),
                        outputPath: tokenizerJsonPath
                    )
                    print("Generated tokenizer.json from vocab.json + merges.txt")
                } catch {
                    print("Warning: Failed to generate tokenizer.json: \(error)")
                }
            } else {
                print("Warning: Cannot generate tokenizer.json — vocab.json: \(hasVocab), merges.txt: \(hasMerges)")
            }
        }

        // Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            print("Warning: Could not load tokenizer: \(error)")
        }

        // Load speech tokenizer — check that it's a directory, not a stale file
        let speechTokenizerPath = modelDir.appendingPathComponent("speech_tokenizer")
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: speechTokenizerPath.path, isDirectory: &isDir), isDir.boolValue {
            try loadSpeechTokenizer(model: model, path: speechTokenizerPath)
        } else if fm.fileExists(atPath: speechTokenizerPath.path) {
            // speech_tokenizer exists but is not a directory — stale cache
            // Remove the entire model cache so it re-downloads cleanly next time
            print("speech_tokenizer is not a directory (stale cache), clearing model cache...")
            try? fm.removeItem(at: modelDir)
            throw AudioGenerationError.modelNotInitialized(
                "Model cache was corrupted (speech_tokenizer). It has been cleared. Please try loading again."
            )
        } else {
            print("Warning: speech_tokenizer directory not found, speech decoding unavailable")
        }

        print("Loaded Qwen3-TTS model (\(config.ttsModelType))")
        return model
    }

    private static func loadSpeechTokenizer(model: Qwen3TTSModel, path: URL) throws {
        // Load config — fall back to defaults if config.json is missing
        let tokenizerConfig: Qwen3TTSTokenizerConfig
        let configPath = path.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configPath) {
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: configData)
        } else {
            print("Warning: speech_tokenizer/config.json not found, using defaults")
            let defaultJson = "{}".data(using: .utf8)!
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: defaultJson)
        }

        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: tokenizerConfig)

        // Load weights
        var tokenizerWeights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            tokenizerWeights.merge(weights) { _, new in new }
        }

        if !tokenizerWeights.isEmpty {
            let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: tokenizerWeights)
            let pairs = sanitized.map { ($0.key, $0.value) }
            try speechTokenizer.update(parameters: ModuleParameters.unflattened(pairs), verify: .noUnusedKeys)
            eval(speechTokenizer.parameters())
        }

        model.speechTokenizer = speechTokenizer
        print("Loaded speech tokenizer decoder")
    }

    // MARK: - Generate tokenizer.json from vocab.json + merges.txt

    /// Qwen3-TTS repos ship with a slow tokenizer (vocab.json + merges.txt) but
    /// swift-transformers requires tokenizer.json (fast tokenizer format). This
    /// generates the fast tokenizer JSON from the available files.
    private static func generateTokenizerJson(
        vocabPath: URL,
        mergesPath: URL,
        tokenizerConfigPath: URL,
        outputPath: URL
    ) throws {
        // Read vocab
        let vocabData = try Data(contentsOf: vocabPath)
        let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

        // Read merges (skip header line "#version: ...")
        let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
        let mergeLines = mergesText.components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        // Read added_tokens from tokenizer_config.json
        var addedTokens = [[String: Any]]()
        if let configData = try? Data(contentsOf: tokenizerConfigPath),
           let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]] {
            for (idStr, tokenInfo) in addedTokensDecoder {
                guard let tokenId = Int(idStr),
                      let content = tokenInfo["content"] as? String else { continue }
                let entry: [String: Any] = [
                    "id": tokenId,
                    "content": content,
                    "single_word": tokenInfo["single_word"] as? Bool ?? false,
                    "lstrip": tokenInfo["lstrip"] as? Bool ?? false,
                    "rstrip": tokenInfo["rstrip"] as? Bool ?? false,
                    "normalized": tokenInfo["normalized"] as? Bool ?? false,
                    "special": tokenInfo["special"] as? Bool ?? true
                ]
                addedTokens.append(entry)
            }
            addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        }

        // Build tokenizer.json
        // Qwen2 uses ByteLevel BPE with a GPT-2-style regex pre-tokenizer
        let tokenizerJson: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": addedTokens,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": [
                            "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        ],
                        "behavior": "Isolated",
                        "invert": false
                    ] as [String: Any],
                    [
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    ] as [String: Any]
                ] as [[String: Any]]
            ] as [String: Any],
            "post_processor": NSNull(),
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            ] as [String: Any],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocabDict,
                "merges": mergeLines
            ] as [String: Any]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: tokenizerJson, options: [.sortedKeys])
        try jsonData.write(to: outputPath)
    }
}
