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

    public init(refCodes: MLXArray, speakerEmbedding: MLXArray?, refText: String, language: String) {
        self.refCodes = refCodes
        self.speakerEmbedding = speakerEmbedding
        self.refText = refText
        self.language = language
    }
}

// MARK: - Serialization

extension VoiceClonePrompt {
    /// Serialize to Data for persistence (e.g., saving to disk).
    ///
    /// Format: `[4 bytes metadata length][JSON metadata][refCodes safetensors][speaker safetensors?]`
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

    /// Deserialize from Data.
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

public enum VoiceClonePromptError: Error, LocalizedError {
    case invalidData(String)

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
}
