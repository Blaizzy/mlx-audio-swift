import Foundation
import HuggingFace
import MLX
import MLXNN

/// Loader for Whisper models from HuggingFace
public enum WhisperModelLoader {

    /// Result of loading a Whisper model
    public struct LoadedModel {
        public let encoder: AudioEncoder
        public let decoder: TextDecoder
        public let config: WhisperConfiguration
        public let tokenizerDirectory: URL
    }

    /// Result of loading a model with quantization info
    public struct LoadResult {
        public let model: LoadedModel
        public let requestedQuantization: WhisperQuantization
        public let actualQuantization: WhisperQuantization

        public var didFallback: Bool { requestedQuantization != actualQuantization }
    }

    /// HuggingFace repository IDs for MLX weights
    public static func repoId(
        for model: WhisperModel,
        quantization: WhisperQuantization = .float16
    ) -> String {
        let base: String
        switch model {
        case .tiny:       base = "mlx-community/whisper-tiny-mlx"
        case .base:       base = "mlx-community/whisper-base-mlx"
        case .small:      base = "mlx-community/whisper-small-mlx"
        case .medium:     base = "mlx-community/whisper-medium-mlx"
        case .largeV3:    base = "mlx-community/whisper-large-v3-mlx"
        case .largeTurbo: base = "mlx-community/whisper-large-v3-turbo"
        }

        switch quantization {
        case .float16: return base
        case .int8:    return base + "-8bit"
        case .int4:    return base + "-4bit"
        }
    }

    /// HuggingFace repository IDs for tokenizer files (OpenAI repos have tokenizers)
    public static func tokenizerRepoId(for model: WhisperModel) -> String {
        switch model {
        case .tiny:
            return "openai/whisper-tiny"
        case .base:
            return "openai/whisper-base"
        case .small:
            return "openai/whisper-small"
        case .medium:
            return "openai/whisper-medium"
        case .largeV3:
            return "openai/whisper-large-v3"
        case .largeTurbo:
            return "openai/whisper-large-v3-turbo"
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
        let repoIdString = repoId(for: model)
        guard let repo = Repo.ID(rawValue: repoIdString) else {
            throw WhisperError.invalidModelFormat("Invalid repo ID: \(repoIdString)")
        }
        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        // Use the cache's snapshot directory structure (Python-compatible)
        let snapshotDir = cache.snapshotsDirectory(repo: repo, kind: .model)
            .appendingPathComponent("main")

        // Check if model files already exist (skip network request if cached)
        let configPath = snapshotDir.appendingPathComponent("config.json")
        let modelDirectory: URL

        if FileManager.default.fileExists(atPath: configPath.path) {
            // Check for safetensors files
            let contents = try? FileManager.default.contentsOfDirectory(at: snapshotDir, includingPropertiesForKeys: nil)
            let hasSafetensors = contents?.contains { $0.pathExtension == "safetensors" } ?? false
            if hasSafetensors {
                modelDirectory = snapshotDir
            } else {
                // Need to download
                try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
                modelDirectory = try await client.downloadSnapshot(
                    of: repo,
                    to: snapshotDir,
                    matching: ["*.safetensors", "config.json"],
                    progressHandler: progressHandler
                )
            }
        } else {
            // Need to download
            try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
            modelDirectory = try await client.downloadSnapshot(
                of: repo,
                to: snapshotDir,
                matching: ["*.safetensors", "config.json"],
                progressHandler: progressHandler
            )
        }

        // Download tokenizer files from OpenAI repo (cached separately)
        let tokenizerDirectory = try await downloadTokenizer(
            for: model,
            client: client,
            cache: cache,
            progressHandler: progressHandler
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

        return LoadedModel(
            encoder: encoder,
            decoder: decoder,
            config: config,
            tokenizerDirectory: tokenizerDirectory
        )
    }

    /// Load a Whisper model with quantization support
    /// - Parameters:
    ///   - model: The Whisper model variant to load
    ///   - quantization: Desired quantization level
    ///   - fallbackToFloat16: If true, falls back to float16 when quantized model unavailable
    ///   - progressHandler: Optional callback for download progress
    /// - Returns: LoadResult with model and quantization info
    public static func load(
        model: WhisperModel,
        quantization: WhisperQuantization,
        fallbackToFloat16: Bool = true,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> LoadResult {
        let repoIdString = repoId(for: model, quantization: quantization)

        do {
            let loaded = try await loadFromRepo(
                repoId: repoIdString,
                model: model,
                progressHandler: progressHandler
            )
            return LoadResult(
                model: loaded,
                requestedQuantization: quantization,
                actualQuantization: quantization
            )
        } catch {
            // If quantized model not found and fallback enabled, try float16
            if fallbackToFloat16 && quantization != .float16 {
                let fallbackRepoId = repoId(for: model, quantization: .float16)
                let loaded = try await loadFromRepo(
                    repoId: fallbackRepoId,
                    model: model,
                    progressHandler: progressHandler
                )
                return LoadResult(
                    model: loaded,
                    requestedQuantization: quantization,
                    actualQuantization: .float16
                )
            }
            throw WhisperError.quantizedModelNotAvailable(model, quantization)
        }
    }

    /// Internal helper to load from a specific repo
    private static func loadFromRepo(
        repoId repoIdString: String,
        model: WhisperModel,
        progressHandler: (@Sendable (Progress) -> Void)?
    ) async throws -> LoadedModel {
        guard let repo = Repo.ID(rawValue: repoIdString) else {
            throw WhisperError.invalidModelFormat("Invalid repo ID: \(repoIdString)")
        }

        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        let snapshotDir = cache.snapshotsDirectory(repo: repo, kind: .model)
            .appendingPathComponent("main")

        let configPath = snapshotDir.appendingPathComponent("config.json")
        let modelDirectory: URL

        if FileManager.default.fileExists(atPath: configPath.path) {
            let contents = try? FileManager.default.contentsOfDirectory(at: snapshotDir, includingPropertiesForKeys: nil)
            let hasSafetensors = contents?.contains { $0.pathExtension == "safetensors" } ?? false
            if hasSafetensors {
                modelDirectory = snapshotDir
            } else {
                try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
                modelDirectory = try await client.downloadSnapshot(
                    of: repo,
                    to: snapshotDir,
                    matching: ["*.safetensors", "config.json"],
                    progressHandler: progressHandler
                )
            }
        } else {
            try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
            modelDirectory = try await client.downloadSnapshot(
                of: repo,
                to: snapshotDir,
                matching: ["*.safetensors", "config.json"],
                progressHandler: progressHandler
            )
        }

        let tokenizerDirectory = try await downloadTokenizer(
            for: model,
            client: client,
            cache: cache,
            progressHandler: progressHandler
        )

        let config = try loadConfiguration(from: modelDirectory, model: model)

        let encoder = AudioEncoder(config: config)
        let decoder = TextDecoder(config: config)

        try loadWeights(from: modelDirectory, encoder: encoder, decoder: decoder)
        eval(encoder, decoder)

        return LoadedModel(
            encoder: encoder,
            decoder: decoder,
            config: config,
            tokenizerDirectory: tokenizerDirectory
        )
    }

    /// Load a Whisper model from a local directory
    /// - Parameters:
    ///   - directory: Local directory containing model files
    ///   - model: The Whisper model variant (for alignment heads lookup)
    ///   - tokenizerDirectory: Directory containing tokenizer files (defaults to model directory)
    /// - Returns: A tuple containing the encoder, decoder, and configuration
    public static func load(
        from directory: URL,
        model: WhisperModel,
        tokenizerDirectory: URL? = nil
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

        return LoadedModel(
            encoder: encoder,
            decoder: decoder,
            config: config,
            tokenizerDirectory: tokenizerDirectory ?? directory
        )
    }

    // MARK: - Tokenizer Download

    private static func downloadTokenizer(
        for model: WhisperModel,
        client: HubClient,
        cache: HubCache,
        progressHandler: (@Sendable (Progress) -> Void)?
    ) async throws -> URL {
        let tokenizerRepoIdString = tokenizerRepoId(for: model)
        guard let tokenizerRepo = Repo.ID(rawValue: tokenizerRepoIdString) else {
            throw WhisperError.invalidModelFormat("Invalid tokenizer repo ID: \(tokenizerRepoIdString)")
        }

        let tokenizerSnapshotDir = cache.snapshotsDirectory(repo: tokenizerRepo, kind: .model)
            .appendingPathComponent("main")

        // Check if tokenizer files already exist (skip network request if cached)
        let tokenizerJsonPath = tokenizerSnapshotDir.appendingPathComponent("tokenizer.json")

        if FileManager.default.fileExists(atPath: tokenizerJsonPath.path) {
            return tokenizerSnapshotDir
        }

        try FileManager.default.createDirectory(at: tokenizerSnapshotDir, withIntermediateDirectories: true)

        let tokenizerDirectory = try await client.downloadSnapshot(
            of: tokenizerRepo,
            to: tokenizerSnapshotDir,
            matching: [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json"
            ],
            progressHandler: progressHandler
        )

        return tokenizerDirectory
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
        decoder: TextDecoder,
        parallel: Bool
    ) async throws {
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

        if parallel {
            try await loadWeightsParallel(
                encoderWeights: encoderWeights,
                decoderWeights: decoderWeights,
                encoder: encoder,
                decoder: decoder
            )
        } else {
            let encoderParams = ModuleParameters.unflattened(encoderWeights)
            try encoder.update(parameters: encoderParams, verify: [.noUnusedKeys])

            let decoderParams = ModuleParameters.unflattened(decoderWeights)
            try decoder.update(parameters: decoderParams, verify: [.all])
        }
    }

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
        // Use .noUnusedKeys because encoder.positionalEmbedding is computed (sinusoidal), not loaded
        try encoder.update(parameters: encoderParams, verify: [.noUnusedKeys])

        let decoderParams = ModuleParameters.unflattened(decoderWeights)
        try decoder.update(parameters: decoderParams, verify: [.all])
    }

    /// Load weights in parallel for faster initialization
    private static func loadWeightsParallel(
        encoderWeights: [String: MLXArray],
        decoderWeights: [String: MLXArray],
        encoder: AudioEncoder,
        decoder: TextDecoder
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                let params = ModuleParameters.unflattened(encoderWeights)
                try encoder.update(parameters: params, verify: [.noUnusedKeys])
            }

            group.addTask {
                let params = ModuleParameters.unflattened(decoderWeights)
                try decoder.update(parameters: params, verify: [.all])
            }

            try await group.waitForAll()
        }
    }

    private static func splitAndSanitizeWeights(
        _ weights: [String: MLXArray]
    ) -> (encoder: [String: MLXArray], decoder: [String: MLXArray]) {
        var encoderWeights = [String: MLXArray]()
        var decoderWeights = [String: MLXArray]()

        let encoderPrefix = "encoder."
        let decoderPrefix = "decoder."

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

        if newKey.contains("mlp.0.") {
            newKey = newKey.replacingOccurrences(of: "mlp.0.", with: "mlp1.")
        } else if newKey.contains("mlp.2.") {
            newKey = newKey.replacingOccurrences(of: "mlp.2.", with: "mlp2.")
        }

        return newKey
    }
}
