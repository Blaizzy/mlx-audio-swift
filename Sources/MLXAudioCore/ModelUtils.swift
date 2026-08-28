import Foundation
import HuggingFace

public enum ModelUtils {
    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let modelURL = try await resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: modelURL.appendingPathComponent("config.json")))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String)
                ?? (config["architecture"] as? String)
                ?? (config["model_version"] as? String)
                ?? modelNameComponents?.first?.lowercased()
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
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns
        )
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
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let normalizedRequiredExtension = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension

        if let snapshot = existingHubSnapshot(
            repoID: repoID,
            cache: cache,
            requiredExtension: normalizedRequiredExtension
        ) {
            print("Using Hub cache at: \(snapshot.path)")
            return snapshot
        }

        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let mlxAudioDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
        if directoryLooksComplete(mlxAudioDir, requiredExtension: normalizedRequiredExtension) {
            print("Using cached model at: \(mlxAudioDir.path)")
            return mlxAudioDir
        }
        if FileManager.default.fileExists(atPath: mlxAudioDir.path) {
            print("Incomplete mlx-audio copy at \(mlxAudioDir.path), ignoring it.")
            try? FileManager.default.removeItem(at: mlxAudioDir)
        }

        var allowedExtensions: Set<String> = [
            "*.\(normalizedRequiredExtension)",
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.wav",
        ]
        allowedExtensions.formUnion(additionalMatchingPatterns)

        print("Downloading model \(repoID) into Hugging Face Hub cache...")
        let snapshot = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            revision: "main",
            matching: Array(allowedExtensions),
            progressHandler: progressHandler ?? { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        guard directoryLooksComplete(snapshot, requiredExtension: normalizedRequiredExtension) else {
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        print("Model downloaded to: \(snapshot.path)")
        return snapshot
    }

    /// Hugging Face Hub snapshot (`models--org--name/snapshots/<rev>`), if complete.
    public static func existingHubSnapshot(
        repoID: Repo.ID,
        cache: HubCache = .default,
        requiredExtension: String = "safetensors"
    ) -> URL? {
        let normalized = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension
        let slug = repoID.description.replacingOccurrences(of: "/", with: "--")
        var roots = [cache.repoDirectory(repo: repoID, kind: .model)]
        let defaultHub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--\(slug)")
        if defaultHub.standardizedFileURL.path != roots[0].standardizedFileURL.path {
            roots.append(defaultHub)
        }
        for repoDir in roots {
            if let snapshot = latestHubSnapshot(in: repoDir),
               directoryLooksComplete(snapshot, requiredExtension: normalized)
            {
                return snapshot
            }
        }
        return nil
    }

    public static func latestHubSnapshot(in repoDir: URL) -> URL? {
        let fm = FileManager.default
        let snapshotsDir = repoDir.appendingPathComponent("snapshots")
        let mainRef = repoDir.appendingPathComponent("refs").appendingPathComponent("main")
        if let revision = try? String(contentsOf: mainRef, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty
        {
            let snapshot = snapshotsDir.appendingPathComponent(revision)
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: snapshot.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return snapshot
            }
        }
        let snapshots = (try? fm.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return snapshots
            .filter { url in
                ((try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .last
    }

    public static func directoryLooksComplete(_ directory: URL, requiredExtension: String) -> Bool {
        var isDirectory: ObjCBool = false
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return false
        }
        let configPath = directory.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configPath.path),
              let configData = try? Data(contentsOf: configPath),
              (try? JSONSerialization.jsonObject(with: configData)) != nil
        else {
            return false
        }
        let files = (try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .isSymbolicLinkKey]
        )) ?? []
        return files.contains { file in
            guard file.pathExtension == requiredExtension else { return false }
            if (try? file.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true {
                return fm.fileExists(atPath: file.path)
            }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        }
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        }
    }
}
