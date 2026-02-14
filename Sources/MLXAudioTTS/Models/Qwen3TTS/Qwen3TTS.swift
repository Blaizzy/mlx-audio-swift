// Port of mlx_audio/tts/models/qwen3_tts/qwen3_tts.py
// Main Qwen3-TTS conditional generation model supporting VoiceDesign, Base, CustomVoice, and ICL

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXAudioCore
import HuggingFace
import Tokenizers
import Foundation

// MARK: - Model Definition

/// Qwen3-TTS conditional generation model supporting multiple generation paths:
/// VoiceDesign, Base, CustomVoice, and ICL (in-context learning voice cloning).
///
/// Wraps a `Qwen3TTSTalkerForConditionalGeneration` transformer, an optional
/// speech tokenizer (encoder + decoder), and an optional ECAPA-TDNN speaker
/// encoder.  Use ``fromPretrained(_:)`` to load a model from a HuggingFace
/// repository.
public final class Qwen3TTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: Qwen3TTSModelConfig
    let talker: Qwen3TTSTalkerForConditionalGeneration
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var tokenizer: Tokenizer?

    /// The output audio sample rate in Hz (typically 24000).
    public var sampleRate: Int { config.sampleRate }


    // MARK: - Initialization
    init(config: Qwen3TTSModelConfig) {
        let talkerConfig = config.talkerConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        }()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)
    }

    // MARK: - Speaker Encoder

    /// Extract x-vector speaker embedding from raw audio using the ECAPA-TDNN speaker encoder.
    ///
    /// Computes a mel spectrogram from the audio waveform using the speaker encoder's
    /// expected parameters (n_fft=1024, num_mels=128, sr=24000, hop_size=256, win_size=1024,
    /// fmin=0, fmax=12000), then passes it through the speaker encoder to produce an
    /// x-vector embedding suitable for use as a speaker conditioning signal.
    ///
    /// - Parameter audio: Raw audio waveform as MLXArray, shape `[samples]` or `[1, samples]`.
    ///   Expected sample rate: 24000 Hz (matching `speakerEncoderConfig.sampleRate`).
    /// - Returns: Speaker embedding of shape `[1, enc_dim]`.
    /// - Throws: `AudioGenerationError.modelNotInitialized` if the speaker encoder is not loaded
    ///   (only Base models ship with a speaker encoder).
    func extractSpeakerEmbedding(audio: MLXArray) throws -> MLXArray {
        guard let speakerEncoder else {
            throw AudioGenerationError.modelNotInitialized(
                "Speaker encoder not loaded. Only Base models have a speaker encoder."
            )
        }

        let speakerConfig = speakerEncoder.config

        // Flatten to 1D if batched [1, samples] -> [samples]
        var wav = audio
        if wav.ndim == 2 {
            wav = wav.squeezed(axis: 0)
        }

        // Compute mel spectrogram with speaker-encoder-specific parameters
        // Python reference: n_fft=1024, num_mels=128, sr=24000, hop_size=256,
        //                   win_size=1024, fmin=0, fmax=12000
        let mel = computeSpeakerEncoderMel(
            audio: wav,
            sampleRate: speakerConfig.sampleRate,
            nFft: 1024,
            hopLength: 256,
            winSize: 1024,
            nMels: speakerConfig.melDim,
            fMin: 0,
            fMax: 12000
        )

        // Add batch dimension: [time, mel_dim] -> [1, time, mel_dim]
        let melBatched = mel.expandedDimensions(axis: 0)

        // Run through speaker encoder -> [1, enc_dim]
        return speakerEncoder(melBatched)
    }

    /// Compute mel spectrogram for the speaker encoder.
    ///
    /// This uses a standard mel spectrogram computation (power spectrum + mel filterbank + log)
    /// without Whisper-style normalization. The parameters match the Python reference's
    /// `extract_speaker_embedding()` function.
    ///
    /// - Parameters:
    ///   - audio: 1D audio waveform `[samples]`
    ///   - sampleRate: Audio sample rate (typically 24000)
    ///   - nFft: FFT size (1024)
    ///   - hopLength: Hop length between STFT frames (256)
    ///   - winSize: Window size (1024, same as nFft for speaker encoder)
    ///   - nMels: Number of mel filterbank channels (128)
    ///   - fMin: Minimum frequency for mel filterbank (0)
    ///   - fMax: Maximum frequency for mel filterbank (12000)
    /// - Returns: Mel spectrogram of shape `[time, nMels]`
    private func computeSpeakerEncoderMel(
        audio: MLXArray,
        sampleRate: Int,
        nFft: Int,
        hopLength: Int,
        winSize: Int,
        nMels: Int,
        fMin: Float,
        fMax: Float
    ) -> MLXArray {
        // Use Accelerate path (vDSP + BLAS) -- avoids MLXArray overhead
        // and uses Apple Silicon NEON SIMD for the entire DSP pipeline.
        let samples = audio.asArray(Float.self)
        return computeMelSpectrogramAccelerate(
            samples: samples,
            sampleRate: sampleRate,
            nFft: nFft,
            hopLength: hopLength,
            nMels: nMels,
            fMin: fMin,
            fMax: fMax,
            norm: "slaney",
            logScale: .standard
        )
    }


    // MARK: - ICL Input Preparation

    /// Prepares input embeddings for in-context learning (voice cloning) generation.
    ///
    /// This is the Swift port of `_prepare_icl_generation_inputs()` from the Python
    /// reference (`qwen3_tts.py`). It encodes reference audio into codec tokens,
    /// builds combined text+codec embeddings with non-streaming overlay, and
    /// constructs the full input sequence for the autoregressive generation loop.
    ///
    /// - Parameters:
    ///   - text: The target text to synthesise.
    ///   - refAudio: Reference audio waveform for voice cloning.
    ///   - refText: Transcript of the reference audio.
    ///   - language: Resolved language name (e.g. "english", "chinese", "auto").
    /// - Returns: Tuple of (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes).
    /// - Throws: If speech tokenizer encoder is unavailable or tokenizer is not loaded.
    func prepareICLInputs(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String,
        instruct: String? = nil
    ) throws -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray, refCodes: MLXArray) {
        guard let speechTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }

        // --- Step 1: Encode reference audio → refCodes [1, 16, refTime] ---
        let audioForSpk = refAudio

        var refAudioReshaped = refAudio
        if refAudioReshaped.ndim == 1 {
            refAudioReshaped = refAudioReshaped.reshaped(1, 1, -1)
        } else if refAudioReshaped.ndim == 2 {
            refAudioReshaped = refAudioReshaped.expandedDimensions(axis: 1)
        }

        let refCodes = try speechTokenizer.encode(refAudioReshaped)  // [1, 16, refTime]
        eval(refCodes)

        // --- Step 2: Extract speaker embedding ---
        var speakerEmbed: MLXArray? = nil
        if speakerEncoder != nil {
            speakerEmbed = try extractSpeakerEmbedding(audio: audioForSpk)
        }

        // --- Step 3: Delegate to the pre-computed overload ---
        let result = try prepareICLInputs(
            text: text,
            refCodes: refCodes,
            speakerEmbedding: speakerEmbed,
            refText: refText,
            language: language,
            instruct: instruct
        )

        return (inputEmbeds: result.inputEmbeds, trailingTextHidden: result.trailingTextHidden, ttsPadEmbed: result.ttsPadEmbed, refCodes: refCodes)
    }

    /// Prepare ICL input embeddings from pre-computed reference codes and speaker embedding.
    ///
    /// This overload skips the expensive audio encoding and speaker embedding extraction
    /// steps, reusing cached values instead. Both `prepareICLInputs(text:refAudio:...)` and
    /// `generateWithClonePrompt()` delegate to this method.
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize
    ///   - refCodes: Pre-encoded reference audio codes, shape `[1, 16, refTime]`
    ///   - speakerEmbedding: Pre-extracted speaker embedding, shape `[1, encDim]` (or nil)
    ///   - refText: Transcript of the reference audio
    ///   - language: Resolved language string
    /// - Returns: Tuple of (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    func prepareICLInputs(
        text: String,
        refCodes: MLXArray,
        speakerEmbedding: MLXArray?,
        refText: String,
        language: String,
        instruct: String? = nil
    ) throws -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer/config not loaded")
        }

        // --- Step 1: Tokenize reference text ---
        // Template: "<|im_start|>assistant\n{refText}<|im_end|>\n"
        // Skip first 3 tokens (<|im_start|>assistant\n) and last 2 (<|im_end|>\n)
        let refChat = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refChatIds = MLXArray(tokenizer.encode(text: refChat).map { Int32($0) }).reshaped(1, -1)
        let refTextIds = refChatIds[0..., 3 ..< (refChatIds.dim(1) - 2)]

        // --- Step 2: Tokenize target text ---
        // Template: "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        // Skip first 3 and last 5 tokens
        let targetChat = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChat).map { Int32($0) }).reshaped(1, -1)
        let textIds = targetIds[0..., 3 ..< (targetIds.dim(1) - 5)]

        // --- Step 3: TTS special token embeddings ---
        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        // --- Step 4: Build text_embed ---
        // Concatenate ref text + target text tokens, project, and append tts_eos
        let combinedTextIds = concatenated([refTextIds, textIds], axis: 1)
        var textEmbed = talker.textProjection(talker.getTextEmbeddings()(combinedTextIds))
        textEmbed = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLens = textEmbed.dim(1)

        // --- Step 5: Build codec_embed ---
        // codec_bos + sum of all 16 codebook embeddings for reference codes
        let numCodeGroups = talkerConfig.numCodeGroups

        // First codebook embedding
        var refCodecEmbed = talker.getInputEmbeddings()(refCodes[0..., 0, 0...])  // [1, refTime]

        // Sum remaining codebook embeddings (codebooks 1-15)
        for i in 0 ..< (numCodeGroups - 1) {
            refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[i](refCodes[0..., i + 1, 0...])
        }

        // Prepend codec_bos
        let codecBosEmbed = talker.getInputEmbeddings()(MLXArray([Int32(talkerConfig.codecBosId)]).reshaped(1, -1))
        let codecEmbedICL = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)  // [1, refTime+1, hidden]
        let codecLens = codecEmbedICL.dim(1)

        // --- Step 6: Non-streaming overlay ---
        let hiddenDim = ttsPadEmbed.dim(-1)
        let codecPadEmbed = talker.getInputEmbeddings()(MLXArray([Int32(talkerConfig.codecPadId)]).reshaped(1, -1))
        let textWithCodecPad = textEmbed + broadcast(codecPadEmbed, to: [1, textLens, hiddenDim])
        let codecWithTextPad = codecEmbedICL + broadcast(ttsPadEmbed, to: [1, codecLens, hiddenDim])
        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed  // Single pad embed for non-streaming mode

        // --- Step 7: Language ID ---
        var languageId: Int? = nil
        if language.lowercased() != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[language.lowercased()]
        }

        // --- Step 8: Build codec prefix ---
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

        var codecPrefixEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))

        // Suffix: [pad, bos]
        let codecEmbedSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )

        // Insert speaker embed between prefix and suffix if present
        if let speakerEmbedding {
            codecPrefixEmbed = concatenated(
                [codecPrefixEmbed, speakerEmbedding.reshaped(1, 1, -1), codecEmbedSuffix],
                axis: 1
            )
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecEmbedSuffix], axis: 1)
        }

        // --- Step 9: Instruct embedding (optional) ---
        var instructEmbed: MLXArray? = nil
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        // --- Step 10: Role embedding (first 3 tokens of target chat) ---
        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., ..<3]))

        // --- Step 11: Build pad/bos prefix ---
        let padCount = codecPrefixEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenDim])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., ..<(-1), 0...]

        // --- Step 12: Assemble full input_embeds ---
        var inputEmbeds: MLXArray
        if let instructEmbed {
            inputEmbeds = concatenated([instructEmbed, roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)
        } else {
            inputEmbeds = concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)
        }

        eval(inputEmbeds)
        return (inputEmbeds: inputEmbeds, trailingTextHidden: trailingTextHidden, ttsPadEmbed: ttsPadEmbed)
    }

    // MARK: - Generation Path Routing

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

    // MARK: - SpeechGenerationModel Protocol

    /// Generate audio from text using the appropriate generation path.
    ///
    /// The generation path is automatically selected based on the model type
    /// and the presence of reference audio:
    /// - VoiceDesign: `voice` is interpreted as a voice description; `instruct` is ignored
    ///   (VoiceDesign uses `voice` as the instruct).
    /// - CustomVoice: `voice` is a predefined speaker name; `instruct` provides delivery
    ///   hints (e.g., "speak in a whisper").
    /// - Base: `voice` is an optional speaker name; `instruct` provides delivery hints;
    ///   if `refAudio` and `refText` are provided and the speech encoder is available,
    ///   ICL is used instead.
    /// - ICL: In-context learning voice cloning from reference audio.
    ///
    /// - Parameters:
    ///   - text: The text to synthesise.
    ///   - voice: Voice description (VoiceDesign) or speaker name (Base/CustomVoice).
    ///   - refAudio: Reference audio waveform for ICL voice cloning (optional).
    ///   - refText: Transcript of the reference audio (optional).
    ///   - language: Language code (e.g. "en", "chinese", "auto"). Defaults to "auto".
    ///   - instruct: Delivery instruction (e.g., "speak slowly", "speak in a whisper").
    ///     For VoiceDesign models, this parameter is ignored; use `voice` instead.
    ///   - generationParameters: Sampling parameters (temperature, topP, etc.).
    /// - Returns: Generated audio waveform as a 1-D MLXArray.
    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        language: String? = nil,
        instruct: String? = nil,
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
            // Ignore the separate instruct parameter for VoiceDesign models
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
                instruct: instruct,
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
                instruct: instruct,
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

    /// Generate audio from text as an asynchronous stream of ``AudioGeneration`` events.
    ///
    /// Yields `.token(id)` for each generated codec token, `.info(...)` with
    /// generation statistics, and `.audio(waveform)` as the final event.
    /// See ``generate(text:voice:refAudio:refText:language:instruct:generationParameters:)``
    /// for details on path selection.
    ///
    /// - Parameters:
    ///   - text: The text to synthesise.
    ///   - voice: Voice description or speaker name (depends on model type).
    ///   - refAudio: Reference audio waveform for ICL voice cloning (optional).
    ///   - refText: Transcript of the reference audio (optional).
    ///   - language: Language code (e.g. "en", "chinese", "auto"). Defaults to "auto".
    ///   - instruct: Delivery instruction (e.g., "speak slowly", "speak in a whisper").
    ///   - generationParameters: Sampling parameters (temperature, topP, etc.).
    /// - Returns: An ``AsyncThrowingStream`` of ``AudioGeneration`` events.
    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        instruct: String?,
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
                    audio = try generateCustomVoice(
                        text: text,
                        speaker: voice,
                        instruct: instruct,
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
                    audio = try generateBase(
                        text: text,
                        voice: voice,
                        instruct: instruct,
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
                }

                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - Language Resolution

    /// ISO 639-1, ISO 639-2, and full language name mapping to Qwen3-TTS internal language names.
    /// Includes 30+ language codes as required by Task 4 (H2) of the execution plan.
    private static let isoToLanguageName: [String: String] = [
        // ISO 639-1 two-letter codes
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
        "ar": "arabic",
        "hi": "hindi",
        "tr": "turkish",
        "pl": "polish",
        "nl": "dutch",
        "sv": "swedish",
        "fi": "finnish",
        "cs": "czech",
        "ro": "romanian",
        "hu": "hungarian",
        "el": "greek",
        "th": "thai",
        "vi": "vietnamese",
        "id": "indonesian",
        "ms": "malay",
        "uk": "ukrainian",
        "da": "danish",
        "no": "norwegian",
        "he": "hebrew",
        "fa": "persian",

        // ISO 639-2/T three-letter codes
        "eng": "english",
        "zho": "chinese",
        "jpn": "japanese",
        "kor": "korean",
        "deu": "german",
        "fra": "french",
        "rus": "russian",
        "por": "portuguese",
        "spa": "spanish",
        "ita": "italian",
        "ara": "arabic",
        "hin": "hindi",
        "tur": "turkish",
        "pol": "polish",
        "nld": "dutch",
        "swe": "swedish",
        "fin": "finnish",
        "ces": "czech",
        "ron": "romanian",
        "hun": "hungarian",
        "ell": "greek",
        "tha": "thai",
        "vie": "vietnamese",
        "ind": "indonesian",
        "msa": "malay",
        "ukr": "ukrainian",
        "dan": "danish",
        "nor": "norwegian",
        "heb": "hebrew",
        "fas": "persian",

        // Full language names (pass-through)
        "english": "english",
        "chinese": "chinese",
        "japanese": "japanese",
        "korean": "korean",
        "german": "german",
        "french": "french",
        "russian": "russian",
        "portuguese": "portuguese",
        "spanish": "spanish",
        "italian": "italian",
        "arabic": "arabic",
        "hindi": "hindi",
        "turkish": "turkish",
        "polish": "polish",
        "dutch": "dutch",
        "swedish": "swedish",
        "finnish": "finnish",
        "czech": "czech",
        "romanian": "romanian",
        "hungarian": "hungarian",
        "greek": "greek",
        "thai": "thai",
        "vietnamese": "vietnamese",
        "indonesian": "indonesian",
        "malay": "malay",
        "ukrainian": "ukrainian",
        "danish": "danish",
        "norwegian": "norwegian",
        "hebrew": "hebrew",
        "persian": "persian",
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

    // MARK: - Shared Autoregressive Generation Loop

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

            // Schedule GPU evaluation asynchronously so CPU can proceed
            // with sampling setup while GPU computes the forward pass result.
            asyncEval(logits, hidden)

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

    // MARK: - VoiceDesign Generation

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

    // MARK: - Base Generation

    /// Generate audio using the Base model path (no reference audio / no ICL).
    /// Uses `prepareBaseInputs()` which handles speaker lookups, dialect override,
    /// and instruct embedding.
    func generateBase(
        text: String,
        voice: String?,
        instruct: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        onToken: ((Int) -> Void)? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil
    ) throws -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer or tokenizer not loaded")
        }

        // Prepare inputs using Base path (handles speaker, dialect, instruct)
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = try prepareBaseInputs(
            text: text,
            language: language,
            speaker: voice,  // 'voice' maps to speaker name for Base model
            instruct: instruct
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
            promptTokenCount: 0,
            generationTokenCount: tokenCount,
            prefillTime: 0,
            generateTime: generateTime,
            tokensPerSecond: Double(tokenCount) / generateTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        onInfo?(info)

        // Stack and decode
        let codes = stacked(generatedCodes, axis: 1)

        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: 100) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]

        // Trim to valid length
        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self)) * speechTokenizer.decodeUpsampleRate
        if validLen > 0 && validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        return audio
    }

    // MARK: - CustomVoice Generation

    /// Generate audio using a predefined speaker from the CustomVoice model.
    /// Validates that the speaker exists in the model's `spkId` configuration,
    /// then delegates to `prepareBaseInputs()` with the speaker name and optional instruct.
    func generateCustomVoice(
        text: String,
        speaker: String?,
        instruct: String? = nil,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        onToken: ((Int) -> Void)? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil
    ) throws -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer or tokenizer not loaded")
        }

        // Validate speaker exists in config
        guard let speaker, !speaker.isEmpty else {
            throw AudioGenerationError.invalidInput(
                "CustomVoice generation requires a speaker name."
            )
        }

        guard let spkIdMap = config.talkerConfig?.spkId, spkIdMap[speaker.lowercased()] != nil else {
            let available = config.talkerConfig?.spkId?.keys.sorted().joined(separator: ", ") ?? "none"
            throw AudioGenerationError.invalidInput(
                "Speaker '\(speaker)' not found. Available speakers: \(available)"
            )
        }

        // CustomVoice delegates to prepareBaseInputs with speaker name and optional instruct
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = try prepareBaseInputs(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct
        )

        // Cap max tokens
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

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

        let generateTime = Date().timeIntervalSince(startTime)
        let tokenCount = generatedCodes.count
        let info = AudioGenerationInfo(
            promptTokenCount: 0,
            generationTokenCount: tokenCount,
            prefillTime: 0,
            generateTime: generateTime,
            tokensPerSecond: Double(tokenCount) / generateTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        onInfo?(info)

        let codes = stacked(generatedCodes, axis: 1)

        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: 100) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]

        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self)) * speechTokenizer.decodeUpsampleRate
        if validLen > 0 && validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        return audio
    }

    // MARK: - ICL Voice Cloning Generation

    /// Generate audio using in-context learning (voice cloning) with reference audio.
    ///
    /// This is the Swift port of `_generate_icl()` from the Python reference. It prepares
    /// combined text+codec embeddings from the reference audio and text, runs the shared
    /// autoregressive generation loop, then prepends reference codes and proportionally
    /// trims to produce cloned voice output.
    ///
    /// - Parameters:
    ///   - text: The target text to synthesise in the cloned voice.
    ///   - refAudio: Reference audio waveform for voice cloning.
    ///   - refText: Transcript of the reference audio.
    ///   - language: Resolved language name (e.g. "english", "chinese", "auto").
    ///   - instruct: Optional voice delivery hints (e.g. "speak in a whisper").
    ///   - temperature: Sampling temperature.
    ///   - topP: Nucleus sampling threshold.
    ///   - repetitionPenalty: Penalty for repeated tokens (minimum 1.5 for ICL).
    ///   - maxTokens: Maximum number of autoregressive steps.
    /// - Returns: Generated audio waveform as MLXArray.
    func generateICL(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String,
        instruct: String? = nil,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int
    ) throws -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer or text tokenizer not loaded")
        }

        // Step 1: Prepare ICL inputs
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes) = try prepareICLInputs(
            text: text,
            refAudio: refAudio,
            refText: refText,
            language: language,
            instruct: instruct
        )

        // Step 2: Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Step 3: Apply minimum repetition penalty of 1.5 for ICL
        // Prevents code degeneration with long reference prefills
        let effectiveRepPenalty = max(repetitionPenalty, 1.5)

        // Step 4: Run shared autoregressive generation loop
        let generatedCodes = generateFromEmbeddings(
            inputEmbeds: inputEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: effectiveRepPenalty,
            maxTokens: effectiveMaxTokens
        )

        // Step 5: Check for empty generation
        guard !generatedCodes.isEmpty else {
            return MLXArray.zeros([1])
        }

        // Step 6: Prepend reference codes to generated codes before decoding
        let genCodes = stacked(generatedCodes, axis: 1)  // [1, genLen, numCodeGroups]
        let refCodesT = refCodes.transposed(0, 2, 1)  // [1, refTime, 16]
        let fullCodes = concatenated([refCodesT, genCodes], axis: 1)  // [1, refTime+genLen, 16]

        // Step 7: Decode full codes
        let (audio, audioLengths) = speechTokenizer.decode(fullCodes)
        var audioOut = audio[0]  // Remove batch dim

        // Step 8: Trim to valid length
        let validLen = Int(audioLengths[0].item(Int32.self))
        if validLen > 0 && validLen < audioOut.dim(0) {
            audioOut = audioOut[..<validLen]
        }

        // Step 9: Proportional trimming — remove the reference audio portion
        let refLen = refCodes.dim(2)  // refTime
        let totalLen = fullCodes.dim(1)  // refTime + genLen
        let cut = Int(Float(refLen) / Float(max(totalLen, 1)) * Float(audioOut.dim(0)))
        if cut > 0 && cut < audioOut.dim(0) {
            audioOut = audioOut[cut...]
        }

        // Step 10: Evaluate and return
        eval(audioOut)
        return audioOut
    }

    // MARK: - Base/CustomVoice Input Preparation

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

    // MARK: - VoiceDesign Input Preparation

    /// Prepare input embeddings for VoiceDesign generation mode.
    ///
    /// This method tokenizes the input text with ChatML templates, builds codec prefix
    /// embeddings with language ID and optional instruct hints, and returns the assembled
    /// input embeddings along with trailing text hidden states for the autoregressive loop.
    ///
    /// - Parameters:
    ///   - text: Text to synthesise
    ///   - language: Resolved language string (e.g. "english", "auto")
    ///   - instruct: Optional voice description/delivery instruction
    /// - Returns: Tuple of (inputEmbeds, trailingTextHidden, ttsPadEmbed)
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

    // MARK: - Token Sampling

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

    // MARK: - Model Loading

    /// Load a Qwen3-TTS model from a HuggingFace repository.
    ///
    /// Downloads (or resolves from cache) the model weights, tokenizer, speech
    /// tokenizer, and optional speaker encoder.  If the repository ships only a
    /// slow tokenizer (`vocab.json` + `merges.txt`), a fast `tokenizer.json`
    /// is generated automatically.
    ///
    /// - Parameter modelRepo: HuggingFace repository ID (e.g. "Qwen/Qwen3-TTS").
    /// - Returns: A fully initialised ``Qwen3TTSModel`` ready for generation.
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

        // Load speaker encoder if config has speaker_encoder_config (Base model only)
        if let speakerEncoderConfig = config.speakerEncoderConfig {
            let speakerEncoder = Qwen3TTSSpeakerEncoder(config: speakerEncoderConfig)
            let sanitizedWeights = Qwen3TTSSpeakerEncoder.sanitize(weights: allWeights)
            if !sanitizedWeights.isEmpty {
                let pairs = sanitizedWeights.map { ($0.key, $0.value) }
                try speakerEncoder.update(parameters: ModuleParameters.unflattened(pairs), verify: .noUnusedKeys)
                eval(speakerEncoder.parameters())
                model.speakerEncoder = speakerEncoder
                print("Loaded speaker encoder (\(speakerEncoderConfig.encDim)-dim)")
            } else {
                print("Warning: speaker_encoder_config present but no speaker_encoder weights found")
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
            try speechTokenizer.update(parameters: ModuleParameters.unflattened(pairs), verify: .noUnusedKeys)
            eval(speechTokenizer.parameters())
        }

        model.speechTokenizer = speechTokenizer
        print("Loaded speech tokenizer decoder")
    }

    // MARK: - Tokenizer Generation

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
