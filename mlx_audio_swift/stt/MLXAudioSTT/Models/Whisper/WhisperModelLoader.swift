import Foundation
import Hub
import MLX
import MLXNN

/// Loader for Whisper models from HuggingFace
public enum WhisperModelLoader {

    /// Result of loading a Whisper model
    public struct LoadedModel {
        public let encoder: AudioEncoder
        public let decoder: TextDecoder
        public let config: WhisperConfiguration
    }

    /// HuggingFace repository IDs for each model variant
    public static func repoId(for model: WhisperModel) -> String {
        switch model {
        case .tiny:
            return "mlx-community/whisper-tiny-mlx"
        case .base:
            return "mlx-community/whisper-base-mlx"
        case .small:
            return "mlx-community/whisper-small-mlx"
        case .medium:
            return "mlx-community/whisper-medium-mlx"
        case .largeV3:
            return "mlx-community/whisper-large-v3-mlx"
        case .largeTurbo:
            return "mlx-community/whisper-large-v3-turbo"
        }
    }

    /// Load a Whisper model from HuggingFace
    /// - Parameters:
    ///   - model: The Whisper model variant to load
    ///   - progressHandler: Optional callback for download progress
    /// - Returns: A tuple containing the encoder, decoder, and configuration
    public static func load(
        model: WhisperModel,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> LoadedModel {
        let repoId = repoId(for: model)
        let hub = HubApi()
        let repo = Hub.Repo(id: repoId)

        let modelDirectory = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "config.json"],
            progressHandler: progressHandler ?? { _ in }
        )

        let config = try loadConfiguration(from: modelDirectory, model: model)

        let encoder = AudioEncoder(config: config)
        let decoder = TextDecoder(config: config)

        try loadWeights(
            from: modelDirectory,
            encoder: encoder,
            decoder: decoder
        )

        eval(encoder, decoder)

        return LoadedModel(encoder: encoder, decoder: decoder, config: config)
    }

    /// Load a Whisper model from a local directory
    /// - Parameters:
    ///   - directory: Local directory containing model files
    ///   - model: The Whisper model variant (for alignment heads lookup)
    /// - Returns: A tuple containing the encoder, decoder, and configuration
    public static func load(
        from directory: URL,
        model: WhisperModel
    ) throws -> LoadedModel {
        let config = try loadConfiguration(from: directory, model: model)

        let encoder = AudioEncoder(config: config)
        let decoder = TextDecoder(config: config)

        try loadWeights(
            from: directory,
            encoder: encoder,
            decoder: decoder
        )

        eval(encoder, decoder)

        return LoadedModel(encoder: encoder, decoder: decoder, config: config)
    }

    // MARK: - Configuration Loading

    private static func loadConfiguration(
        from directory: URL,
        model: WhisperModel
    ) throws -> WhisperConfiguration {
        let configURL = directory.appending(path: "config.json")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw WhisperError.invalidModelFormat("config.json not found in \(directory.path)")
        }

        let configData = try Data(contentsOf: configURL)

        var config = try JSONDecoder().decode(WhisperConfiguration.self, from: configData)

        if config.alignmentHeads.isEmpty {
            config.alignmentHeads = WhisperAlignmentHeads.heads(for: model)
        }

        return config
    }

    // MARK: - Weight Loading

    private static func loadWeights(
        from directory: URL,
        encoder: AudioEncoder,
        decoder: TextDecoder
    ) throws {
        var allWeights = [String: MLXArray]()

        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        )

        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "safetensors" else { continue }
            let fileWeights = try loadArrays(url: url)
            for (key, value) in fileWeights {
                allWeights[key] = value
            }
        }

        guard !allWeights.isEmpty else {
            throw WhisperError.invalidModelFormat("No safetensors files found in \(directory.path)")
        }

        let (encoderWeights, decoderWeights) = splitAndSanitizeWeights(allWeights)

        let encoderParams = ModuleParameters.unflattened(encoderWeights)
        try encoder.update(parameters: encoderParams, verify: [.all])

        let decoderParams = ModuleParameters.unflattened(decoderWeights)
        try decoder.update(parameters: decoderParams, verify: [.all])
    }

    private static func splitAndSanitizeWeights(
        _ weights: [String: MLXArray]
    ) -> (encoder: [String: MLXArray], decoder: [String: MLXArray]) {
        var encoderWeights = [String: MLXArray]()
        var decoderWeights = [String: MLXArray]()

        let encoderPrefix = "model.encoder."
        let decoderPrefix = "model.decoder."

        for (key, value) in weights {
            if key.hasPrefix(encoderPrefix) {
                let strippedKey = String(key.dropFirst(encoderPrefix.count))
                let sanitizedKey = sanitizeKey(strippedKey)
                encoderWeights[sanitizedKey] = value
            } else if key.hasPrefix(decoderPrefix) {
                let strippedKey = String(key.dropFirst(decoderPrefix.count))
                let sanitizedKey = sanitizeKey(strippedKey)
                decoderWeights[sanitizedKey] = value
            }
        }

        return (encoderWeights, decoderWeights)
    }

    private static func sanitizeKey(_ key: String) -> String {
        var newKey = key

        newKey = newKey.replacingOccurrences(of: "attn_ln", with: "attnLn")
        newKey = newKey.replacingOccurrences(of: "cross_attn_ln", with: "crossAttnLn")
        newKey = newKey.replacingOccurrences(of: "cross_attn", with: "crossAttn")
        newKey = newKey.replacingOccurrences(of: "mlp_ln", with: "mlpLn")
        newKey = newKey.replacingOccurrences(of: "token_embedding", with: "tokenEmbedding")
        newKey = newKey.replacingOccurrences(of: "positional_embedding", with: "positionalEmbedding")

        if newKey.contains("mlp.0.") {
            newKey = newKey.replacingOccurrences(of: "mlp.0.", with: "mlp1.")
        } else if newKey.contains("mlp.2.") {
            newKey = newKey.replacingOccurrences(of: "mlp.2.", with: "mlp2.")
        }

        return newKey
    }
}
