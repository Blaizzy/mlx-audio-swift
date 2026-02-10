import Foundation
import HuggingFace
@preconcurrency import MLX

// MARK: - Float16 Preference for Apple Silicon

/// Controls dtype preference for model weights on Apple Silicon.
///
/// Apple Silicon's GPU ALUs are optimized for float16 (fp16) operations, making
/// them consistently faster than bfloat16 (bf16) for the same model. Since both
/// types are 16-bit, memory usage is identical — the benefit is purely in compute
/// throughput.
///
/// Reference: Apple's Recurrent Drafter research found that fp16 is consistently
/// faster than bf16 on Apple Silicon hardware (M1 through M4).
///
/// By default, `preferFloat16` is `true` — all bfloat16 weights are converted to
/// float16 at load time. Set to `false` to preserve the original weight dtype.
public enum DTypeConfiguration: Sendable {
    /// When `true`, bfloat16 model weights are converted to float16 at load time.
    /// Default is `true` because Apple Silicon GPUs have optimized fp16 ALUs.
    nonisolated(unsafe) public static var preferFloat16: Bool = true

    /// Converts bfloat16 values to float16 in a weight dictionary.
    ///
    /// This operates on the raw `[String: MLXArray]` dictionary that comes from
    /// loading safetensors files, **before** the weights are loaded into a Module.
    /// Only floating-point parameters with `.bfloat16` dtype are converted;
    /// integer types, quantization scales/biases, and float32 parameters are
    /// left untouched.
    ///
    /// - Parameters:
    ///   - weights: The weight dictionary to convert.
    ///   - force: If `true`, convert regardless of the `preferFloat16` setting.
    ///            Defaults to `false` (respects the global setting).
    /// - Returns: A new dictionary with bfloat16 values converted to float16,
    ///            or the original dictionary if conversion is not requested.
    public static func convertBFloat16ToFloat16(
        weights: [String: MLXArray],
        force: Bool = false
    ) -> [String: MLXArray] {
        guard force || preferFloat16 else { return weights }

        var converted = [String: MLXArray]()
        converted.reserveCapacity(weights.count)
        var convertedCount = 0

        for (key, value) in weights {
            if value.dtype == .bfloat16 {
                converted[key] = value.asType(.float16)
                convertedCount += 1
            } else {
                converted[key] = value
            }
        }

        if convertedCount > 0 {
            print("Converted \(convertedCount)/\(weights.count) weight tensors from bfloat16 to float16")
        }

        return converted
    }
}

// MARK: - ModelUtils

public enum ModelUtils {
    public static func resolveModelType(repoID: Repo.ID, hfToken: String? = nil) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let modelURL = try await resolveOrDownloadModel(repoID: repoID, requiredExtension: "safetensors", hfToken: hfToken)
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: modelURL.appendingPathComponent("config.json")))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String) ?? (config["architecture"] as? String) ?? modelNameComponents?.first?.lowercased()
        }
        return nil
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - string: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        hfToken: String? = nil
    ) async throws -> URL {
        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            client = HubClient.default
        }
        let cache = client.cache ?? HubCache.default
        return try await resolveOrDownloadModel(client: client, cache: cache, repoID: repoID, requiredExtension: requiredExtension)
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - client: The HuggingFace Hub client
    ///   - cache: The HuggingFace cache
    ///   - repoID: The repository ID
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache,
        repoID: Repo.ID,
        requiredExtension: String
    ) async throws -> URL {
        // Use a persistent cache directory based on repo ID
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = URL.cachesDirectory.appendingPathComponent("intrusive-memory/Models/Audio").appendingPathComponent(modelSubdir)

        // Check if model already exists with required files
        if FileManager.default.fileExists(atPath: modelDir.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let hasRequiredFiles = files?.contains { $0.pathExtension == requiredExtension } ?? false

            if hasRequiredFiles {
                // Validate that config.json is valid JSON
                let configPath = modelDir.appendingPathComponent("config.json")
                if FileManager.default.fileExists(atPath: configPath.path) {
                    if let configData = try? Data(contentsOf: configPath),
                       let _ = try? JSONSerialization.jsonObject(with: configData) {
                        print("Using cached model at: \(modelDir.path)")
                        return modelDir
                    } else {
                        print("Cached config.json is invalid, clearing cache...")
                        try? FileManager.default.removeItem(at: modelDir)
                    }
                }
            }
        }

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let allowedExtensions: Set<String> = ["*.\(requiredExtension)", "*.safetensors", "*.json", "*.txt"]

        print("Downloading model \(repoID)...")
        _ = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDir,
            revision: "main",
            matching: Array(allowedExtensions),
            progressHandler: { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }

    // MARK: - Wired Memory Integration

    /// Execute model inference with wired memory enabled to prevent page faults.
    ///
    /// This is a convenience wrapper around ``WiredMemoryManager/withPinnedMemory(limitBytes:_:)-1hqwp``
    /// that automatically sizes the wired limit based on current active memory (model weights)
    /// plus headroom for inference buffers.
    ///
    /// Typical usage after loading a model:
    /// ```swift
    /// let model = try await Qwen3TTSModel.fromPretrained(repo)
    /// let audio = try await ModelUtils.withWiredModelMemory {
    ///     try await model.generate(text: "Hello", voice: "A clear voice", language: "en")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - headroomFactor: Multiplier on current active memory to allow for inference buffers.
    ///     Defaults to 1.5 (50% headroom above model weight size).
    ///   - body: The async block to execute with wired memory.
    /// - Returns: The return value of `body`.
    /// - Throws: Rethrows any error from `body`.
    public static func withWiredModelMemory<R>(
        headroomFactor: Double = 1.5,
        _ body: () async throws -> R
    ) async rethrows -> R {
        try await WiredMemoryManager.withPinnedMemoryForCurrentModel(
            headroomFactor: headroomFactor,
            body
        )
    }
}
