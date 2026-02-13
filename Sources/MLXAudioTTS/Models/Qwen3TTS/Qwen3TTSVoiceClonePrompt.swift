// VoiceClonePrompt — pre-computed ICL prompt data for reuse

@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import Foundation

/// Pre-computed ICL (In-Context Learning) prompt data for voice cloning.
///
/// Captures the encoded reference audio codes and speaker embedding so that
/// multiple generation calls can reuse the same voice identity without
/// re-encoding the reference audio each time.
///
/// Create with `Qwen3TTSModel.createVoiceClonePrompt(refAudio:refText:language:)`.
public struct VoiceClonePrompt: Sendable {
    /// Encoded reference audio codes, shape `[1, 16, ref_time]`.
    public let refCodes: MLXArray

    /// X-vector speaker embedding, shape `[1, enc_dim]`. Nil if the model
    /// has no speaker encoder (VoiceDesign models).
    public let speakerEmbedding: MLXArray?

    /// Transcript of the reference audio.
    public let refText: String

    /// Language code used for encoding.
    public let language: String

    /// Create a voice clone prompt from pre-computed data.
    ///
    /// - Parameters:
    ///   - refCodes: Encoded reference audio codes, shape `[1, 16, ref_time]`.
    ///   - speakerEmbedding: X-vector speaker embedding, shape `[1, enc_dim]` (or nil).
    ///   - refText: Transcript of the reference audio.
    ///   - language: Language code used for encoding (e.g. "en", "auto").
    public init(refCodes: MLXArray, speakerEmbedding: MLXArray?, refText: String, language: String) {
        self.refCodes = refCodes
        self.speakerEmbedding = speakerEmbedding
        self.refText = refText
        self.language = language
    }
}

// MARK: - Serialization

extension VoiceClonePrompt {
    /// Serialize to binary data for persistence (e.g. saving to disk).
    ///
    /// Format: `[4 bytes metadata length][JSON metadata][refCodes safetensors][speaker safetensors?]`
    ///
    /// - Returns: A self-contained binary ``Data`` blob that can be written to disk
    ///   and later restored with ``deserialize(from:)``.
    public func serialize() throws -> Data {
        // Save refCodes to safetensors data
        let refCodesData = try saveToData(arrays: ["ref_codes": refCodes])

        // Save speaker embedding if present
        var speakerData: Data? = nil
        if let embedding = speakerEmbedding {
            speakerData = try saveToData(arrays: ["speaker_embedding": embedding])
        }

        // Build metadata
        let metadata: [String: Any] = [
            "refText": refText,
            "language": language,
            "hasEmbedding": speakerEmbedding != nil,
            "refCodesSize": refCodesData.count,
            "speakerDataSize": speakerData?.count ?? 0
        ]
        let metadataJson = try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])

        // Pack: [4 bytes metadata length][metadata JSON][refCodes safetensors][speaker safetensors?]
        var result = Data()
        var metaLen = UInt32(metadataJson.count).littleEndian
        result.append(Data(bytes: &metaLen, count: 4))
        result.append(metadataJson)
        result.append(refCodesData)
        if let speakerData {
            result.append(speakerData)
        }

        return result
    }

    /// Deserialize a ``VoiceClonePrompt`` from binary data.
    ///
    /// - Parameter data: Binary data previously produced by ``serialize()``.
    /// - Returns: A reconstituted ``VoiceClonePrompt``.
    /// - Throws: ``VoiceClonePromptError/invalidData(_:)`` if the data is malformed.
    public static func deserialize(from data: Data) throws -> VoiceClonePrompt {
        guard data.count >= 4 else {
            throw VoiceClonePromptError.invalidData("Data too short")
        }

        // Read metadata length
        let metaLen = Int(data.withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian)
        guard data.count >= 4 + metaLen else {
            throw VoiceClonePromptError.invalidData("Metadata length exceeds data size")
        }

        // Parse metadata
        let metadataJson = data.subdata(in: 4 ..< (4 + metaLen))
        guard let metadata = try JSONSerialization.jsonObject(with: metadataJson) as? [String: Any],
              let refText = metadata["refText"] as? String,
              let language = metadata["language"] as? String,
              let hasEmbedding = metadata["hasEmbedding"] as? Bool,
              let refCodesSize = metadata["refCodesSize"] as? Int
        else {
            throw VoiceClonePromptError.invalidData("Invalid metadata format")
        }

        let refCodesStart = 4 + metaLen
        let refCodesEnd = refCodesStart + refCodesSize
        guard data.count >= refCodesEnd else {
            throw VoiceClonePromptError.invalidData("refCodes data truncated")
        }

        // Load refCodes from safetensors data
        let refCodesData = data.subdata(in: refCodesStart ..< refCodesEnd)
        let refCodesArrays = try loadArrays(data: refCodesData)
        guard let refCodes = refCodesArrays["ref_codes"] else {
            throw VoiceClonePromptError.invalidData("Missing ref_codes in safetensors")
        }

        // Load speaker embedding if present
        var speakerEmbedding: MLXArray? = nil
        if hasEmbedding {
            let speakerDataSize = (metadata["speakerDataSize"] as? Int) ?? 0
            let speakerStart = refCodesEnd
            let speakerEnd = speakerStart + speakerDataSize
            guard data.count >= speakerEnd else {
                throw VoiceClonePromptError.invalidData("Speaker embedding data truncated")
            }

            let speakerData = data.subdata(in: speakerStart ..< speakerEnd)
            let embArrays = try loadArrays(data: speakerData)
            speakerEmbedding = embArrays["speaker_embedding"]
        }

        return VoiceClonePrompt(
            refCodes: refCodes,
            speakerEmbedding: speakerEmbedding,
            refText: refText,
            language: language
        )
    }
}

// MARK: - Error type

/// Errors that can occur during ``VoiceClonePrompt`` serialization or deserialization.
public enum VoiceClonePromptError: Error, LocalizedError {
    /// The binary data is malformed or truncated.
    case invalidData(String)

    /// A human-readable description of the error.
    public var errorDescription: String? {
        switch self {
        case .invalidData(let msg): return "VoiceClonePrompt: \(msg)"
        }
    }
}

// MARK: - Factory method

extension Qwen3TTSModel {
    /// Create a VoiceClonePrompt by encoding reference audio and extracting
    /// speaker embedding. The resulting prompt can be reused for multiple
    /// generation calls without re-encoding.
    ///
    /// - Parameters:
    ///   - refAudio: Reference audio waveform `[samples]` or `[1, samples]`
    ///   - refText: Transcript of the reference audio
    ///   - language: Language code (e.g. "en", "chinese", "auto")
    /// - Returns: A VoiceClonePrompt ready for use with `generateWithClonePrompt()`
    public func createVoiceClonePrompt(
        refAudio: MLXArray,
        refText: String,
        language: String = "auto"
    ) throws -> VoiceClonePrompt {
        guard let speechTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }

        // Reshape audio for encoder: [batch, 1, samples]
        var audioForEncode = refAudio
        if audioForEncode.ndim == 1 {
            audioForEncode = audioForEncode.reshaped(1, 1, -1)
        } else if audioForEncode.ndim == 2 {
            audioForEncode = audioForEncode.expandedDimensions(axis: 1)
        }

        // Encode reference audio
        let refCodes = try speechTokenizer.encode(audioForEncode)
        eval(refCodes)

        // Extract speaker embedding if available
        var speakerEmbedding: MLXArray? = nil
        if speakerEncoder != nil {
            speakerEmbedding = try extractSpeakerEmbedding(audio: refAudio)
            eval(speakerEmbedding!)
        }

        return VoiceClonePrompt(
            refCodes: refCodes,
            speakerEmbedding: speakerEmbedding,
            refText: refText,
            language: language
        )
    }

    // MARK: - Generate with clone prompt

    /// Generate audio using a pre-computed VoiceClonePrompt.
    ///
    /// This is functionally identical to `generateICL()` but skips the expensive
    /// audio encoding and speaker embedding extraction steps by reusing cached data
    /// from the clone prompt.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - clonePrompt: Pre-computed voice clone prompt from `createVoiceClonePrompt()`
    ///   - language: Language code override (defaults to the prompt's language)
    ///   - temperature: Sampling temperature (default 0.9)
    ///   - topP: Nucleus sampling threshold (default 1.0)
    ///   - repetitionPenalty: Repetition penalty (minimum 1.5 for ICL, default 1.5)
    ///   - maxTokens: Maximum generation tokens (default 4096)
    /// - Returns: Generated audio waveform
    public func generateWithClonePrompt(
        text: String,
        clonePrompt: VoiceClonePrompt,
        language: String? = nil,
        instruct: String? = nil,
        temperature: Float = 0.9,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.5,
        maxTokens: Int = 4096
    ) throws -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer or text tokenizer not loaded")
        }

        let effectiveLanguage = language ?? clonePrompt.language
        let refCodes = clonePrompt.refCodes

        // Step 1: Prepare ICL inputs using pre-computed codes and speaker embedding
        let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = try prepareICLInputs(
            text: text,
            refCodes: refCodes,
            speakerEmbedding: clonePrompt.speakerEmbedding,
            refText: clonePrompt.refText,
            language: effectiveLanguage,
            instruct: instruct
        )

        // Step 2: Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Step 3: Apply minimum repetition penalty of 1.5 for ICL
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

}
