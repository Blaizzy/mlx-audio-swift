import Foundation
import MLX
import MLXNN
import HuggingFace
import MLXAudioCore

// MARK: - Fish S1 DAC Model

/// Fish S1 DAC audio codec used by Echo TTS.
/// A causal audio autoencoder with encoder, quantizer, and decoder.
public class FishS1DAC: Module {
    public let config: FishS1DACConfig
    public let sampleRate: Int
    public let hopLength: Int
    public let frameLength: Int

    @ModuleInfo(key: "encoder") var encoder: FishS1DACEncoder
    @ModuleInfo(key: "quantizer") var quantizer: FishDownsampleResidualVectorQuantize
    @ModuleInfo(key: "decoder") var decoder: FishS1DACDecoder

    public init(config: FishS1DACConfig) {
        self.config = config
        self.sampleRate = config.sampleRate
        self.hopLength = config.hopLength
        self.frameLength = config.frameLength

        self._encoder.wrappedValue = FishS1DACEncoder(config: config)
        self._quantizer.wrappedValue = FishDownsampleResidualVectorQuantize(config: config)
        self._decoder.wrappedValue = FishS1DACDecoder(config: config)
    }

    // MARK: - Padding

    private func padToFrameLength(_ audio: MLXArray) -> MLXArray {
        let length = audio.dim(-1)
        if length % frameLength != 0 {
            let padAmount = frameLength - (length % frameLength)
            return MLX.padded(audio, widths: [.init(0), .init(0), .init((0, padAmount))])
        }
        return audio
    }

    // MARK: - Encode

    /// Encode audio waveform to discrete indices.
    /// Input: [B, 1, T] (NCL). Returns (indices [B, nCodebooks+1, T_ds], indicesLens).
    public func encode(_ audio: MLXArray) -> (MLXArray, MLXArray) {
        let padded = padToFrameLength(audio)
        let z = encoder(padded)  // [B, latentDim, T_enc]
        let (_, indices) = quantizer(z)
        return (indices, MLXArray(indices.dim(-1)))
    }

    // MARK: - Decode

    /// Decode discrete indices to audio waveform.
    /// Input: indices [B, nCodebooks+1, T_ds]. Returns [B, 1, T].
    public func decode(_ indices: MLXArray) -> MLXArray {
        let zQ = quantizer.decodeFromIndices(indices)
        let zQProcessed = quantizer.postProcessAndUpsample(zQ)
        return decoder(zQProcessed)
    }

    // MARK: - encode_zq / decode_zq (Used by Echo TTS)

    /// Encode audio to quantized latent z_q (pre-upsample).
    /// This is the format Echo TTS works with.
    /// Input: [B, 1, T]. Returns [B, latentDim, T_ds] where T_ds = T / frameLength.
    public func encodeZQ(_ audio: MLXArray) -> MLXArray {
        let (indices, _) = encode(audio)

        // Clip indices to valid codebook ranges
        var clippedIndices = MLXArray.zeros(like: indices)

        // Semantic codebook (index 0): clip to semantic_codebook_size
        let semIdx = MLX.clip(
            indices[0..., ..<1, 0...],
            min: 0,
            max: config.semanticCodebookSize - 1
        )

        // Residual codebooks (index 1+): clip to codebook_size
        if indices.dim(1) > 1 {
            let resIdx = MLX.clip(
                indices[0..., 1..., 0...],
                min: 0,
                max: config.codebookSize - 1
            )
            clippedIndices = MLX.concatenated([semIdx, resIdx], axis: 1)
        } else {
            clippedIndices = semIdx
        }

        // Decode indices to continuous latent (pre-upsample)
        return quantizer.decodeFromIndices(clippedIndices)
    }

    /// Decode quantized latent z_q to audio waveform.
    /// Input: [B, latentDim, T_ds]. Returns [B, 1, T].
    public func decodeZQ(_ zQ: MLXArray) -> MLXArray {
        let zQProcessed = quantizer.postProcessAndUpsample(zQ)
        return decoder(zQProcessed)
    }

    // MARK: - Weight Sanitization

    /// Sanitize weights from PyTorch format.
    /// Converts weight normalization parametrization names.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // PyTorch weight normalization parametrization remapping
            // .conv.parametrizations.weight.original0 -> .conv.weight_g
            // .conv.parametrizations.weight.original1 -> .conv.weight_v
            // .parametrizations.weight.original0 -> .weight_g
            // .parametrizations.weight.original1 -> .weight_v

            if newKey.contains(".parametrizations.weight.original0") {
                newKey = newKey.replacingOccurrences(
                    of: ".parametrizations.weight.original0",
                    with: ".weight_g"
                )
            } else if newKey.contains(".parametrizations.weight.original1") {
                newKey = newKey.replacingOccurrences(
                    of: ".parametrizations.weight.original1",
                    with: ".weight_v"
                )
            }

            // Lift .conv.bias up when using WN wrapper
            // e.g., encoder.blocks.0.res_units.0.block.0.conv.bias
            // The Python WN wrapper has conv.bias, but in Swift the bias is at the WN level

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Load from Pretrained

    public static func fromPretrained(
        _ repoId: String,
        cache: HubCache = .default
    ) async throws -> FishS1DAC {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "FishS1DAC",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        var config = FishS1DACConfig()
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(FishS1DACConfig.self, from: configData)
        }

        // Create model
        let model = FishS1DAC(config: config)

        // Load weights
        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        var weights: [String: MLXArray]
        if FileManager.default.fileExists(atPath: weightsURL.path) {
            weights = try loadArrays(url: weightsURL)
        } else {
            let torchURL = modelDir.appendingPathComponent("pytorch_model.safetensors")
            weights = try loadArrays(url: torchURL)
        }

        let sanitizedWeights = sanitize(weights)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights),
            verify: .none
        )

        eval(model.parameters())
        return model
    }
}

// MARK: - AudioCodecModel Conformance

extension FishS1DAC: AudioCodecModel {
    public typealias EncodedAudio = MLXArray

    public var codecSampleRate: Double? { Double(sampleRate) }

    public func encodeAudio(_ waveform: MLXArray) -> MLXArray {
        encodeZQ(waveform)
    }

    public func decodeAudio(_ input: MLXArray) -> MLXArray {
        decodeZQ(input)
    }
}
