import Foundation
import HuggingFace
import SwiftAcervo

// MARK: - ModelResolver

/// Central entry point for model resolution and download.
///
/// Replaces all scattered download mechanisms with a unified flow:
/// 1. Check Acervo cache for existing model
/// 2. If not cached, discover files via HuggingFace Hub API
/// 3. Download via Acervo to `~/Library/SharedModels/`
///
/// Models downloaded by any intrusive-memory app are shared across all apps.
public enum ModelResolver {

    private static let migrationKey = "ModelResolver.didMigrateLegacyPaths"

    /// Resolve HF token from environment variable or Info.plist.
    public static func resolveHFToken() -> String? {
        ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
    }

    /// Resolve a model directory, downloading via Acervo if not cached.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace repo ID (e.g., `"mlx-community/snac_24khz"`)
    ///   - requiredFiles: Specific file paths that must be present (e.g., `["tokenizer-e351c8d8-checkpoint125.safetensors"]`)
    ///   - extensions: File extensions to download when discovering repo contents. Defaults to safetensors, json, txt.
    ///   - hfToken: Optional HuggingFace bearer token for gated repos.
    /// - Returns: Local directory URL containing model files.
    public static func resolve(
        modelId: String,
        requiredFiles: [String] = [],
        extensions: Set<String> = ["safetensors", "json", "txt"],
        hfToken: String? = nil
    ) async throws -> URL {
        runMigrationIfNeeded()

        let token = hfToken ?? resolveHFToken()

        // Fast path: model already available with valid config.json
        if Acervo.isModelAvailable(modelId) {
            let dir = try Acervo.modelDirectory(for: modelId)
            let configPath = dir.appendingPathComponent("config.json")

            if FileManager.default.fileExists(atPath: configPath.path) {
                if let data = try? Data(contentsOf: configPath),
                   (try? JSONSerialization.jsonObject(with: data)) != nil {
                    print("Using cached model at: \(dir.path)")
                    return dir
                }
                // Invalid config.json — clear and re-download
                print("Cached config.json is invalid, clearing cache...")
                try? Acervo.deleteModel(modelId)
            }
        } else if !requiredFiles.isEmpty {
            // Model not marked available (no config.json), but check if
            // required files already exist (e.g., Mimi single-file models)
            let allPresent = requiredFiles.allSatisfy { fileName in
                Acervo.modelFileExists(modelId, fileName: fileName)
            }
            if allPresent {
                let dir = try Acervo.modelDirectory(for: modelId)
                print("Using cached model at: \(dir.path)")
                return dir
            }
        }

        // Discover repo files via HuggingFace Hub API
        let client: HubClient = if let token, !token.isEmpty {
            HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            HubClient.default
        }

        guard let repoID = Repo.ID(rawValue: modelId) else {
            throw AcervoError.invalidModelId(modelId)
        }

        let treeEntries = try await client.listFiles(
            in: repoID,
            kind: .model,
            revision: "main",
            recursive: true
        )

        // Build file list: required files + files matching extensions
        var filesToDownload = Set(requiredFiles)
        for entry in treeEntries where entry.type == .file {
            let ext = (entry.path as NSString).pathExtension
            if extensions.contains(ext) {
                filesToDownload.insert(entry.path)
            }
        }

        guard !filesToDownload.isEmpty else {
            throw AcervoError.downloadFailed(
                fileName: modelId,
                statusCode: 0
            )
        }

        print("Downloading model \(modelId) (\(filesToDownload.count) files)...")
        try await Acervo.download(
            modelId,
            files: Array(filesToDownload).sorted(),
            token: token,
            progress: { progress in
                print("  [\(progress.fileIndex + 1)/\(progress.totalFiles)] \(progress.fileName)")
            }
        )

        let modelDir = try Acervo.modelDirectory(for: modelId)
        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }

    /// Resolve a single specific file from a model repo.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace repo ID
    ///   - fileName: Specific file to resolve (e.g., `"tokenizer-e351c8d8-checkpoint125.safetensors"`)
    ///   - hfToken: Optional HuggingFace bearer token
    /// - Returns: URL to the specific file within the model directory.
    public static func resolveFile(
        modelId: String,
        fileName: String,
        hfToken: String? = nil
    ) async throws -> URL {
        let dir = try await resolve(
            modelId: modelId,
            requiredFiles: [fileName],
            hfToken: hfToken
        )
        return dir.appendingPathComponent(fileName)
    }

    // MARK: - Legacy Migration

    private static func runMigrationIfNeeded() {
        guard !UserDefaults.standard.bool(forKey: migrationKey) else { return }
        UserDefaults.standard.set(true, forKey: migrationKey)

        do {
            let migrated = try Acervo.migrateFromLegacyPaths()
            if !migrated.isEmpty {
                print("Migrated \(migrated.count) models to SharedModels:")
                for model in migrated {
                    print("  - \(model.id)")
                }
            }
        } catch {
            print("Warning: Legacy model migration failed: \(error)")
        }
    }
}
