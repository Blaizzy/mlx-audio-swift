import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Qwen3TTS Model

public final class Qwen3TTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: Qwen3TTSModelConfig
    let talker: Qwen3TTSTalkerForConditionalGeneration
    var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var tokenizer: Tokenizers.Tokenizer?

    private struct ReferenceAudioContext {
        let audioID: ObjectIdentifier
        let speakerEmbedding: MLXArray?
        let refCodes: MLXArray
        let codecEmbedIcl: MLXArray
    }

    private let inputPreparationCacheLock = NSLock()
    private var cachedReferenceAudioContext: ReferenceAudioContext?

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 4096,
            temperature: 0.9,
            topP: 1.0,
            repetitionPenalty: 1.05
        )
    }

    init(config: Qwen3TTSModelConfig) {
        let talkerConfig = config.talkerConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        }()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)
        self.speakerEncoder = config.ttsModelType == "base"
            ? Qwen3TTSSpeakerEncoder(config: config.speakerEncoderConfig)
            : nil
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
        try requireGenerationComponents()
        let settings = resolveVoiceDesignGenerationSettings(
            language: language,
            generationParameters: generationParameters
        )

        return try generateVoiceDesign(
            text: text,
            instruct: voice,
            language: settings.language,
            refAudio: refAudio,
            refText: refText,
            temperature: settings.temperature,
            topK: settings.topK,
            topP: settings.topP,
            repetitionPenalty: settings.repetitionPenalty,
            minP: settings.minP,
            maxTokens: settings.maxTokens
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
        let settings = resolveVoiceDesignGenerationSettings(
            language: language,
            generationParameters: generationParameters
        )
        return makeGenerationStream { model, onToken, onInfo, onAudioChunk in
            _ = try model.generateVoiceDesign(
                text: text,
                instruct: voice,
                language: settings.language,
                refAudio: refAudio,
                refText: refText,
                temperature: settings.temperature,
                topK: settings.topK,
                topP: settings.topP,
                repetitionPenalty: settings.repetitionPenalty,
                minP: settings.minP,
                maxTokens: settings.maxTokens,
                streamingInterval: streamingInterval,
                onToken: onToken,
                onInfo: onInfo,
                onAudioChunk: onAudioChunk
            )
        }
    }

    public func generate(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try requireGenerationComponents()
        let settings = resolveVoiceDesignGenerationSettings(
            language: conditioning.resolvedLanguage,
            generationParameters: generationParameters
        )

        return try generateVoiceDesign(
            text: text,
            instruct: nil,
            language: settings.language,
            conditioning: conditioning,
            refAudio: nil,
            refText: nil,
            temperature: settings.temperature,
            topK: settings.topK,
            topP: settings.topP,
            repetitionPenalty: settings.repetitionPenalty,
            minP: settings.minP,
            maxTokens: settings.maxTokens
        )
    }

    public func generateStream(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            conditioning: conditioning,
            generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    public func generateStream(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let settings = resolveVoiceDesignGenerationSettings(
            language: conditioning.resolvedLanguage,
            generationParameters: generationParameters
        )
        return makeGenerationStream { model, onToken, onInfo, onAudioChunk in
            _ = try model.generateVoiceDesign(
                text: text,
                instruct: nil,
                language: settings.language,
                conditioning: conditioning,
                refAudio: nil,
                refText: nil,
                temperature: settings.temperature,
                topK: settings.topK,
                topP: settings.topP,
                repetitionPenalty: settings.repetitionPenalty,
                minP: settings.minP,
                maxTokens: settings.maxTokens,
                streamingInterval: streamingInterval,
                onToken: onToken,
                onInfo: onInfo,
                onAudioChunk: onAudioChunk
            )
        }
    }

    // MARK: - Decode chunk helper

    /// Decode a chunk of codec codes to audio waveform.
    /// - Parameters:
    ///   - codes: Codec codes [1, time, numCodeGroups]
    ///   - chunkTokens: Tokens per decode chunk (controls decode granularity)
    /// - Returns: Decoded audio waveform (1D)
    private func decodeChunk(_ codes: MLXArray, chunkTokens: Int = 300) -> MLXArray {
        guard let speechTokenizer else { return MLXArray.zeros([1]) }

        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: chunkTokens) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]

        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self))
            * speechTokenizer.decodeUpsampleRate
        if validLen > 0, validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        return audio
    }

    private func withInputPreparationCacheLock<T>(_ body: () -> T) -> T {
        inputPreparationCacheLock.lock()
        defer { inputPreparationCacheLock.unlock() }
        return body()
    }

    private func referenceAudioForEncoder(_ refAudio: MLXArray) -> MLXArray {
        if refAudio.ndim == 1 {
            return refAudio.reshaped(1, 1, refAudio.dim(0))
        } else if refAudio.ndim == 2 {
            return refAudio.reshaped(1, refAudio.dim(0), refAudio.dim(1))
        } else {
            return refAudio
        }
    }

    private func codecEmbedIcl(from refCodes: MLXArray, talkerConfig: Qwen3TTSTalkerConfig) -> MLXArray {
        let firstCbCodes = refCodes[0..., 0, 0...]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCbCodes)
        if talkerConfig.numCodeGroups > 1 {
            for i in 0 ..< (talkerConfig.numCodeGroups - 1) {
                let codeIdx = i + 1
                if codeIdx >= refCodes.dim(1) { break }
                let cbCodes = refCodes[0..., codeIdx, 0...]
                refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[i](cbCodes)
            }
        }

        let codecBosEmbed = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecBosId)]).reshaped(1, 1)
        )
        return concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
    }

    private func referenceAudioContext(for refAudio: MLXArray) -> ReferenceAudioContext {
        guard let speechTokenizer, let talkerConfig = config.talkerConfig else {
            fatalError("Speech tokenizer/config not loaded")
        }

        let audioID = ObjectIdentifier(refAudio)
        print("Got audio ID: '\(audioID)'")
        if let cached = withInputPreparationCacheLock({ cachedReferenceAudioContext }),
           cached.audioID == audioID {
            print("Returning cached context")
            return cached
        }

        let speakerEmbedding = extractSpeakerEmbedding(refAudio)
        let refCodes = speechTokenizer.encode(referenceAudioForEncoder(refAudio))
        let codecEmbedIcl = codecEmbedIcl(from: refCodes, talkerConfig: talkerConfig)
        if let speakerEmbedding {
            eval(speakerEmbedding, refCodes, codecEmbedIcl)
        } else {
            eval(refCodes, codecEmbedIcl)
        }
        let context = ReferenceAudioContext(
            audioID: audioID,
            speakerEmbedding: speakerEmbedding,
            refCodes: refCodes,
            codecEmbedIcl: codecEmbedIcl
        )

        return withInputPreparationCacheLock {
            if let cached = cachedReferenceAudioContext, cached.audioID == audioID {
                return cached
            }
            cachedReferenceAudioContext = context
            return context
        }
    }

    // MARK: - VoiceDesign generation

    func generateVoiceDesign(
        text: String,
        instruct: String?,
        language: String,
        conditioning: Qwen3TTSReferenceConditioning? = nil,
        refAudio: MLXArray?,
        refText: String?,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        minP: Float,
        maxTokens: Int,
        streamingInterval: Double = 2.0,
        onToken: ((Int) -> Void)? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil,
        onAudioChunk: ((MLXArray) -> Void)? = nil
    ) throws -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            throw AudioGenerationError.modelNotInitialized(
                "Qwen3TTS generateVoiceDesign requires both the speech tokenizer and the text tokenizer to be loaded."
            )
        }

        let talkerConfig = config.talkerConfig!

        // Prepare inputs
        let inputEmbedsInit: MLXArray
        let trailingTextHidden: MLXArray
        let ttsPadEmbed: MLXArray
        let refCodes: MLXArray?

        if let conditioning {
            let prepared = try prepareICLGenerationInputs(
                text: text,
                conditioning: conditioning
            )
            inputEmbedsInit = prepared.0
            trailingTextHidden = prepared.1
            ttsPadEmbed = prepared.2
            refCodes = prepared.3
        } else if let refAudio,
           let refText,
           speechTokenizer.hasEncoder {
            let prepared = try prepareICLGenerationInputs(
                text: text,
                refAudio: refAudio,
                refText: refText,
                language: language
            )
            inputEmbedsInit = prepared.0
            trailingTextHidden = prepared.1
            ttsPadEmbed = prepared.2
            refCodes = prepared.3
        } else {
            // For CustomVoice models: voice is a speaker name, not free-text instruct.
            let isCVModel = config.ttsModelType == "custom_voice"
            let speaker: String? = isCVModel ? instruct : nil
            let effectiveInstruct: String? = isCVModel ? nil : instruct
            let prepared = prepareGenerationInputs(
                text: text,
                language: language,
                instruct: effectiveInstruct,
                speaker: speaker
            )
            inputEmbedsInit = prepared.0
            trailingTextHidden = prepared.1
            ttsPadEmbed = prepared.2
            refCodes = nil
        }

        // Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Initialize cache and timing
        let startTime = Date()
        let cache = talker.makeCache()
        var generatedCodes = [MLXArray]()
        var generatedCodebookTokens = [Int]()
        let eosTokenId = talkerConfig.codecEosTokenId

        // Suppress special tokens
        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        // Streaming decode state
        let codecTokenRateHz = 12.5
        let streamingChunkSize = max(1, Int(streamingInterval * codecTokenRateHz))
        var decodedTokens = 0

        var trailingIdx = 0
        var inputEmbeds = inputEmbedsInit
        let eosTokenArray = MLXArray([Int32(eosTokenId)]).reshaped(1, 1)
        let codeCache = talker.codePredictor.makeCache()

        if onAudioChunk != nil {
            speechTokenizer.decoder.resetStreamingState()
        }
        defer {
            if onAudioChunk != nil {
                speechTokenizer.decoder.resetStreamingState()
            }
        }

        for step in 0 ..< effectiveMaxTokens {
            // Forward pass through talker
            let (logits, hidden) = talker(inputEmbeds, cache: cache)

            // Sample first codebook token
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topP: topP,
                topK: topK,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedCodebookTokens,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId,
                minP: minP
            )

            // Defer sync to the eval boundary with inputEmbeds.
            let isEOS = nextToken .== eosTokenArray

            // Generate remaining codebook tokens with code predictor
            var codeTokens = [nextToken]
            let codeHidden = hidden[0..., (-1)..., 0...]
            for layerCache in codeCache {
                _ = layerCache.trim(layerCache.offset)
            }

            for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = talker.getInputEmbeddings()(nextToken)
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
                }

                let (codeLogits, _, _) = talker.codePredictor(
                    codeInput, cache: codeCache, generationStep: codeIdx
                )

                let nextCode = sampleToken(
                    codeLogits,
                    temperature: temperature,
                    topP: topP,
                    topK: topK,
                    minP: minP
                )
                codeTokens.append(nextCode)
            }

            let allCodes = concatenated(codeTokens, axis: 1) // [1, num_code_groups]

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
            eval(inputEmbeds, isEOS)

            let tokenId = Int(nextToken[0, 0].item(Int32.self))
            onToken?(tokenId)
            if isEOS.item(Bool.self) {
                break
            }
            generatedCodebookTokens.append(tokenId)
            generatedCodes.append(allCodes)

            // Streaming: decode and yield audio chunks during generation
            if let onAudioChunk {
                let newTokens = generatedCodes.count - decodedTokens
                if newTokens >= streamingChunkSize {
                    let codesChunk = stacked(Array(generatedCodes[decodedTokens...]), axis: 1)
                    let codesForDecoder = codesChunk.transposed(0, 2, 1)
                    eval(codesForDecoder)
                    let decoded = speechTokenizer.decoder.streamingStep(codesForDecoder).squeezed(axis: 1)
                    let audioChunk = decoded[0]
                    eval(audioChunk)

                    decodedTokens = generatedCodes.count
                    onAudioChunk(audioChunk)
                }
            }

            if step > 0, step % 50 == 0 {
                Memory.clearCache()
            }
        }

        guard !generatedCodes.isEmpty else {
            return MLXArray.zeros([1])
        }

        // Emit generation info
        let generateTime = Date().timeIntervalSince(startTime)
        let tokenCount = generatedCodes.count
        let info = AudioGenerationInfo(
            promptTokenCount: 0, // Not tracked for VoiceDesign
            generationTokenCount: tokenCount,
            prefillTime: 0, // Included in generateTime
            generateTime: generateTime,
            tokensPerSecond: Double(tokenCount) / generateTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        onInfo?(info)

        // Streaming path: yield remaining tokens and return early
        if let onAudioChunk {
            if generatedCodes.count > decodedTokens {
                let codesChunk = stacked(Array(generatedCodes[decodedTokens...]), axis: 1)
                let codesForDecoder = codesChunk.transposed(0, 2, 1)
                eval(codesForDecoder)
                let decoded = speechTokenizer.decoder.streamingStep(codesForDecoder).squeezed(axis: 1)
                let audioChunk = decoded[0]
                eval(audioChunk)
                onAudioChunk(audioChunk)
            }
            // Streaming chunks already yielded; return empty (caller uses chunks)
            return MLXArray.zeros([1])
        }

        // Non-streaming path: full decode (existing behavior)
        let codes = stacked(generatedCodes, axis: 1) // [1, seq_len, num_code_groups]

        var decodeCodes = codes
        if let refCodes {
            let refCodesT = refCodes.transposed(0, 2, 1)
            decodeCodes = concatenated([refCodesT, codes], axis: 1)
        }

        var audio = decodeChunk(decodeCodes)

        if let refCodes {
            let refLen = refCodes.dim(2)
            let totalLen = decodeCodes.dim(1)
            let cut = Int(Double(refLen) / Double(max(totalLen, 1)) * Double(audio.dim(0)))
            if cut > 0, cut < audio.dim(0) {
                audio = audio[cut...]
            }
        }

        eval(audio)
        return audio
    }

    // MARK: - Reference conditioning

    public func prepareReferenceConditioning(
        refAudio: MLXArray,
        refText: String,
        language: String?
    ) throws -> Qwen3TTSReferenceConditioning {
        try prepareReferenceConditioning(
            refAudio: refAudio,
            refText: refText,
            speakerEmbedding: nil,
            language: language ?? "auto"
        )
    }

    private struct VoiceDesignGenerationSettings {
        let language: String
        let temperature: Float
        let topK: Int
        let topP: Float
        let repetitionPenalty: Float
        let minP: Float
        let maxTokens: Int
    }

    public struct Qwen3TTSReferenceConditioning: @unchecked Sendable {
        public let speakerEmbedding: MLXArray?
        public let referenceSpeechCodes: MLXArray
        public let referenceTextTokenIDs: MLXArray
        public let resolvedLanguage: String
        public let codecLanguageID: Int?

        public init(
            speakerEmbedding: MLXArray?,
            referenceSpeechCodes: MLXArray,
            referenceTextTokenIDs: MLXArray,
            resolvedLanguage: String,
            codecLanguageID: Int?
        ) {
            self.speakerEmbedding = speakerEmbedding
            self.referenceSpeechCodes = referenceSpeechCodes
            self.referenceTextTokenIDs = referenceTextTokenIDs
            self.resolvedLanguage = resolvedLanguage
            self.codecLanguageID = codecLanguageID
        }
    }

    private func requireGenerationComponents() throws {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }
    }

    private func resolveVoiceDesignGenerationSettings(
        language: String?,
        generationParameters: GenerateParameters
    ) -> VoiceDesignGenerationSettings {
        VoiceDesignGenerationSettings(
            language: language ?? "auto",
            temperature: generationParameters.temperature,
            topK: generationParameters.topK,
            topP: generationParameters.topP,
            repetitionPenalty: generationParameters.repetitionPenalty ?? 1.05,
            minP: generationParameters.minP,
            maxTokens: generationParameters.maxTokens ?? 4096
        )
    }

    private func makeGenerationStream(
        _ body: @escaping @Sendable (
            Qwen3TTSModel,
            @escaping (Int) -> Void,
            @escaping (AudioGenerationInfo) -> Void,
            @escaping (MLXArray) -> Void
        ) throws -> Void
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                try requireGenerationComponents()
                try body(
                    self,
                    { tokenId in continuation.yield(.token(tokenId)) },
                    { info in continuation.yield(.info(info)) },
                    { chunk in continuation.yield(.audio(chunk)) }
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    func prepareICLGenerationInputs(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let conditioning = try prepareReferenceConditioning(
            refAudio: refAudio,
            refText: refText,
            speakerEmbedding: nil,
            language: language
        )
        return try prepareICLGenerationInputs(text: text, conditioning: conditioning)
    }

    func prepareReferenceConditioning(
        refAudio: MLXArray,
        refText: String,
        speakerEmbedding: MLXArray?,
        language: String
    ) throws -> Qwen3TTSReferenceConditioning {
        guard let tokenizer, let talkerConfig = config.talkerConfig, let speechTokenizer else {
            throw AudioGenerationError.modelNotInitialized(
                "Qwen3TTS reference conditioning requires the text tokenizer, talker config, and speech tokenizer to be loaded."
            )
        }
        guard speechTokenizer.hasEncoder else {
            throw AudioGenerationError.invalidInput(
                "Qwen3TTS reference conditioning requires a speech tokenizer encoder, but this checkpoint does not provide one."
            )
        }

        let refContext = referenceAudioContext(for: refAudio)

        let resolvedLanguage = language.lowercased() == "auto" ? "auto" : language.lowercased()

        // Reference text tokenization
        let refChatText = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(text: refChatText).map { Int32($0) }).reshaped(1, -1)
        let refCount = refIds.dim(1)
        let refStart = min(3, refCount)
        let refEnd = max(refStart, refCount - 2)
        let refTextIds = refIds[0..., refStart ..< refEnd]

        // Language ID
        var languageId: Int?
        if resolvedLanguage != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[resolvedLanguage]
        }

        return Qwen3TTSReferenceConditioning(
            speakerEmbedding: speakerEmbedding ?? refContext.speakerEmbedding,
            referenceSpeechCodes: refContext.refCodes,
            referenceTextTokenIDs: refTextIds,
            resolvedLanguage: resolvedLanguage,
            codecLanguageID: languageId
        )
    }

    func prepareICLGenerationInputs(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            throw AudioGenerationError.modelNotInitialized(
                "Qwen3TTS request assembly requires the text tokenizer and talker config to be loaded."
            )
        }

        let targetChatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChatText).map { Int32($0) }).reshaped(1, -1)
        let targetCount = targetIds.dim(1)
        let targetStart = min(3, targetCount)
        let targetEnd = max(targetStart, targetCount - 5)
        let targetTextIds = targetIds[0..., targetStart ..< targetEnd]

        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        let combinedTextIds = concatenated([conditioning.referenceTextTokenIDs, targetTextIds], axis: 1)
        var textEmbed = talker.textProjection(talker.getTextEmbeddings()(combinedTextIds))
        textEmbed = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLen = textEmbed.dim(1)

        let refCodes = conditioning.referenceSpeechCodes
        let codecEmbedIcl = codecEmbedIcl(from: refCodes, talkerConfig: talkerConfig)

        let codecPadEmbed = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId)]).reshaped(1, 1)
        )
        let textWithCodecPad = textEmbed + broadcast(
            codecPadEmbed,
            to: [1, textLen, codecPadEmbed.dim(-1)]
        )
        let codecWithTextPad = codecEmbedIcl + broadcast(
            ttsPadEmbed,
            to: [1, codecEmbedIcl.dim(1), ttsPadEmbed.dim(-1)]
        )

        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed

        let codecPrefill: [Int32] = if let langId = conditioning.codecLanguageID {
            [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        var codecPrefixEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        let codecPrefixSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )
        if let speakerEmbedding = conditioning.speakerEmbedding {
            let speakerEmbed = speakerEmbedding.reshaped(1, 1, -1)
            codecPrefixEmbed = concatenated([codecPrefixEmbed, speakerEmbed, codecPrefixSuffix], axis: 1)
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., 0 ..< 3]))

        let padCount = codecPrefixEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0 ..< (codecPrefixEmbed.dim(1) - 1), 0...]

        let inputEmbeds = concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes)
    }

    func extractSpeakerEmbedding(_ refAudio: MLXArray) -> MLXArray? {
        guard let speakerEncoder else { return nil }

        let rawAudio: MLXArray
        if refAudio.ndim == 1 {
            rawAudio = refAudio.reshaped(1, refAudio.dim(0))
        } else if refAudio.ndim == 2 {
            if refAudio.dim(0) == 1 {
                rawAudio = refAudio
            } else {
                rawAudio = refAudio[0 ..< 1]
            }
        } else if refAudio.ndim == 3, refAudio.dim(1) == 1 {
            let squeezed = refAudio[0..., 0...]
            if squeezed.dim(0) == 1 {
                rawAudio = squeezed
            } else {
                rawAudio = squeezed[0 ..< 1]
            }
        } else {
            return nil
        }

        let batchSize = rawAudio.dim(0)
        var mels = [MLXArray]()
        mels.reserveCapacity(batchSize)

        for batch in 0 ..< batchSize {
            let waveform = rawAudio[batch]
            let mel = computeMelSpectrogram(
                audio: waveform,
                sampleRate: speakerEncoder.config.sampleRate,
                nFft: 1024,
                hopLength: 256,
                nMels: 128
            )
            mels.append(mel)
        }

        let stackedMels = stacked(mels, axis: 0)
        let embedding = speakerEncoder(stackedMels)
        return embedding
    }

    func prepareGenerationInputs(
        text: String,
        language: String,
        instruct: String?,
        speaker: String? = nil
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

        // Language ID — may be overridden by speaker dialect below
        var languageId: Int?
        if language.lowercased() != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[language.lowercased()]
        }

        // Speaker embedding (CustomVoice models only)
        var speakerEmbed: MLXArray? = nil
        if let speaker = speaker {
            if let spkIdMap = talkerConfig.spkId,
               let spkIdValue = spkIdMap[speaker.lowercased()] {
                let spkIdArray = MLXArray([Int32(spkIdValue.intValue)]).reshaped(1, 1)
                speakerEmbed = talker.getInputEmbeddings()(spkIdArray)
                print("[Qwen3TTS] CustomVoice: speaker '\(speaker)' → spkId=\(spkIdValue.intValue)")
            } else {
                print("[Qwen3TTS] CustomVoice: WARNING - speaker '\(speaker)' not found in spkId map")
            }
            if let dialectMap = talkerConfig.spkIsDialect,
               let dialectVal = dialectMap[speaker.lowercased()],
               dialectVal.isDialect,
               let dialectName = dialectVal.dialectName,
               let langMap = talkerConfig.codecLanguageId,
               let dialectLangId = langMap[dialectName] {
                print("[Qwen3TTS] CustomVoice: dialect override '\(dialectName)' langId=\(dialectLangId)")
                languageId = dialectLangId
            }
        }

        // Build codec prefix
        let codecPrefill: [Int32] = if let langId = languageId {
            [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        let codecEmbedSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )
        if let spkEmbed = speakerEmbed {
            codecEmbed = concatenated([codecEmbed, spkEmbed.reshaped([1, 1, -1]), codecEmbedSuffix], axis: 1)
            print("[Qwen3TTS] CustomVoice: codec prefix length = \(codecEmbed.dim(1))")
        } else {
            codecEmbed = concatenated([codecEmbed, codecEmbedSuffix], axis: 1)
        }

        // Instruct embedding (VoiceDesign only — not CustomVoice)
        var instructEmbed: MLXArray?
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
        var inputEmbeds: MLXArray = if let instructEmbed {
            concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            concatenated([roleEmbed, combinedEmbed], axis: 1)
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
        topK: Int = 50,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int]? = nil,
        suppressTokens: [Int]? = nil,
        eosTokenId: Int? = nil,
        minP: Float = 0.0
    ) -> MLXArray {
        var logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1) // [batch, vocab_size]

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

        // Preserve EOS logit so top-k/top-p/min-p do not permanently suppress it.
        let eosLogit: MLXArray? = if let eosTokenId, eosTokenId >= 0, eosTokenId < logitsSlice.dim(-1) {
            logitsSlice[0..., eosTokenId ..< (eosTokenId + 1)]
        } else {
            nil
        }

        // Apply top-k filtering (match mlx_lm.apply_top_k ordering and masking semantics)
        var filteredLogits = logitsSlice
        let vocabSize = logitsSlice.dim(-1)
        if topK > 0, topK < vocabSize {
            let kth = min(topK - 1, max(vocabSize - 1, 0))
            if kth >= 0 {
                let maskIdx = argPartition(-logitsSlice, kth: kth, axis: -1)[0..., topK...]
                let negInf = MLXArray.full(maskIdx.shape, values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
                filteredLogits = putAlong(filteredLogits, maskIdx, values: negInf, axis: -1)
            }
        }

        // Apply top-p (nucleus) sampling
        if topP > 0, topP < 1.0 {
            let probs = softmax(filteredLogits, axis: -1)

            // Sort in ASCENDING order (like Python)
            let sortedIndices = argSort(filteredLogits, axis: -1)
            let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

            // Cumulative probabilities
            let cumProbs = cumsum(sortedProbs, axis: -1)

            // Rearrange cumulative probs back to original order
            // Create inverse index mapping using putAlong
            let vocabSize = sortedIndices.dim(-1)
            let arangeIndices = MLXArray(0 ..< vocabSize).reshaped(1, -1).asType(Int32.self)
            let zeros = MLXArray.zeros(sortedIndices.shape, type: Int32.self)
            let inverseIndices = putAlong(zeros, sortedIndices, values: arangeIndices, axis: -1)
            let cumProbsOrigOrder = takeAlong(cumProbs, inverseIndices, axis: -1)

            // Mask tokens where cumulative prob > (1 - top_p)
            // Keep tokens that are in the top_p nucleus
            let threshold = 1.0 - topP
            let mask = cumProbsOrigOrder .> threshold
            let negInf = MLXArray.full(filteredLogits.shape, values: MLXArray(-Float.infinity), dtype: filteredLogits.dtype)
            filteredLogits = which(mask, filteredLogits, negInf)
        }

        // Apply min-p sampling behavior (default kept at 0.0 for now)
        if minP > 0.0 {
            let scaledMinP = Float(log(Double(minP)))
            // Indices sorted in descending order (like Python `argsort(-logits)`)
            let sortedIndices = argSort(-filteredLogits, axis: -1)
            let sortedLogits = takeAlong(filteredLogits, sortedIndices, axis: -1)
            let topLogits = sortedLogits[0..., 0 ..< 1]
            let scaledMinPArray = MLXArray.full(
                topLogits.shape,
                values: MLXArray(scaledMinP),
                dtype: sortedLogits.dtype
            ) + topLogits
            let removeMask = sortedLogits .< scaledMinPArray
            let negInf = MLXArray.full(sortedLogits.shape, values: MLXArray(-Float.infinity), dtype: sortedLogits.dtype)
            let filteredSortedLogits = which(removeMask, negInf, sortedLogits)

            let invArange = MLXArray(0 ..< vocabSize).reshaped(1, -1).asType(Int32.self)
            let inverseIndices = putAlong(MLXArray.zeros(sortedIndices.shape, type: Int32.self), sortedIndices, values: invArange, axis: -1)
            filteredLogits = takeAlong(filteredSortedLogits, inverseIndices, axis: -1)
        }

        if let eosLogit, let eosTokenId {
            let eosIdx = MLXArray([Int32(eosTokenId)]).reshaped(1, 1)
            filteredLogits = putAlong(filteredLogits, eosIdx, values: eosLogit, axis: -1)
        }

        // Sample with temperature
        let token = categorical(filteredLogits / temperature)
        return token.reshaped(1, 1)
    }

    // MARK: - fromPretrained

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> Qwen3TTSModel {
        let repoID = Repo.ID(rawValue: modelRepo)!
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            cache: cache
        )

        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> Qwen3TTSModel {
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

        // Quantized checkpoints store packed weights and companion .scales tensors.
        // Convert talker Linear layers before loading those tensors.
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model.talker) { path, _ in
                guard talkerWeights["\(path).scales"] != nil else {
                    return nil
                }

                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path) {
                    return layerQuant.asTuple
                }

                return config.quantization?.asTuple
            }
        }

        try model.talker.update(parameters: ModuleParameters.unflattened(talkerPairs), verify: .all)
        eval(model.talker.parameters())

        // Generate tokenizer.json if missing (Qwen3-TTS ships without it)
        let tokenizerJsonPath = modelDir.appendingPathComponent("tokenizer.json")
        if !fm.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            let hasVocab = fm.fileExists(atPath: vocabPath.path)
            let hasMerges = fm.fileExists(atPath: mergesPath.path)
            if hasVocab, hasMerges {
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

        // Load speaker encoder for base models when available
        if config.ttsModelType == "base" {
            let speakerWeights = Qwen3TTSSpeakerEncoder.sanitize(weights: allWeights)
            if !speakerWeights.isEmpty {
                if let speakerEncoder = model.speakerEncoder {
                    let speakerPairs = speakerWeights.map { ($0.key, $0.value) }
                    try speakerEncoder.update(parameters: ModuleParameters.unflattened(speakerPairs), verify: .all)
                    eval(speakerEncoder.parameters())
                }
            }
            if model.speakerEncoder != nil {
                print("Loaded speaker encoder")
            } else {
                print("Warning: speaker encoder config missing, skipping speaker encoder load")
            }
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
            try speechTokenizer.update(parameters: ModuleParameters.unflattened(pairs), verify: .all)
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
