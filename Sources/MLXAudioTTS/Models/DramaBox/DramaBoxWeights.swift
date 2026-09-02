import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

struct DramaBoxResolvedPaths: Sendable {
    let audioDirectory: URL
    let gemmaDirectory: URL
}

enum DramaBoxWeights {
    static let audioRequiredFiles = [dramaBoxDiTWeightFile, dramaBoxAudioComponentsFile]
    static let gemmaRequiredFileSuffixes = ["safetensors"]

    static func resolveGemmaRepository(_ spec: String?) -> String {
        let trimmed = spec?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty {
            return dramaBoxDefaultGemmaRepository
        }
        return trimmed
    }

    static func isLocalCheckpointDirectory(_ url: URL, requiredFiles: [String]) -> Bool {
        var isDirectory: ObjCBool = false
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        let config = url.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: config.path) else { return false }
        if requiredFiles.isEmpty {
            return directoryHasNonEmptySafetensors(url)
        }
        return requiredFiles.allSatisfy { name in
            let file = url.appendingPathComponent(name)
            return fileExistsAndNonEmpty(file)
        }
    }

    static func fileExistsAndNonEmpty(_ url: URL) -> Bool {
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey, .isSymbolicLinkKey]) else {
            return FileManager.default.fileExists(atPath: url.path)
        }
        if values.isRegularFile == true || values.isSymbolicLink == true {
            return (values.fileSize ?? 1) > 0 || FileManager.default.fileExists(atPath: url.path)
        }
        return FileManager.default.fileExists(atPath: url.path)
    }

    static func directoryHasNonEmptySafetensors(_ directory: URL) -> Bool {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return files.contains { file in
            file.pathExtension == "safetensors" && fileExistsAndNonEmpty(file)
        }
    }

    /// Local directory, then Hugging Face Hub snapshot, then Hub download.
    static func resolveDirectory(
        spec: String,
        requiredFiles: [String],
        cache: HubCache,
        hfToken: String?
    ) async throws -> URL {
        let expanded = (spec as NSString).expandingTildeInPath
        let localURL = URL(fileURLWithPath: expanded)
        if isLocalCheckpointDirectory(localURL, requiredFiles: requiredFiles) {
            return localURL
        }

        let repoName = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let repoID = Repo.ID(rawValue: repoName) else {
            throw DramaBoxError.invalidRepositoryID(spec)
        }

        if let cached = existingCachedDirectory(repoID: repoID, requiredFiles: requiredFiles, cache: cache) {
            return cached
        }

        return try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
    }

    static func resolveGemmaDirectory(
        spec: String,
        cache: HubCache,
        hfToken: String?
    ) async throws -> URL {
        let expanded = (spec as NSString).expandingTildeInPath
        let localURL = URL(fileURLWithPath: expanded)
        if isLocalCheckpointDirectory(localURL, requiredFiles: []) {
            let tokenizer = localURL.appendingPathComponent("tokenizer.json")
            if fileExistsAndNonEmpty(tokenizer) {
                return localURL
            }
        }

        guard let repoID = Repo.ID(rawValue: spec) else {
            throw DramaBoxError.invalidRepositoryID(spec)
        }

        if let cached = existingCachedDirectory(repoID: repoID, requiredFiles: [], cache: cache) {
            let tokenizer = cached.appendingPathComponent("tokenizer.json")
            if fileExistsAndNonEmpty(tokenizer) {
                return cached
            }
        }

        return try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["tokenizer.json", "tokenizer.model", "*.json"],
            hfToken: hfToken,
            cache: cache
        )
    }

    static func existingCachedDirectory(
        repoID: Repo.ID,
        requiredFiles: [String],
        cache: HubCache
    ) -> URL? {
        if let snapshot = ModelUtils.existingHubSnapshot(repoID: repoID, cache: cache),
           isLocalCheckpointDirectory(snapshot, requiredFiles: requiredFiles)
        {
            return snapshot
        }

        let mlxAudioDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(repoID.description.replacingOccurrences(of: "/", with: "_"))
        if isLocalCheckpointDirectory(mlxAudioDir, requiredFiles: requiredFiles) {
            return mlxAudioDir
        }
        return nil
    }

    static func latestHubSnapshot(repoDir: URL) -> URL? {
        ModelUtils.latestHubSnapshot(in: repoDir)
    }

    static func loadSafetensors(from directory: URL, files: [String]) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for name in files {
            let url = directory.appendingPathComponent(name)
            guard fileExistsAndNonEmpty(url) else {
                throw DramaBoxError.missingCheckpoint(directory, name)
            }
            let loaded = try MLX.loadArrays(url: url)
            weights.merge(loaded) { _, new in new }
        }
        return weights
    }

    static func loadGemmaShards(from directory: URL) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let shards = ((try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { url in
                let name = url.lastPathComponent
                return name.hasPrefix("model-") && name.hasSuffix(".safetensors")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let files = shards.isEmpty
            ? [directory.appendingPathComponent("model.safetensors")]
            : shards

        var weights: [String: MLXArray] = [:]
        for url in files {
            guard fileExistsAndNonEmpty(url) else {
                throw DramaBoxError.missingGemmaBackbone("No safetensors under \(directory.path)")
            }
            let loaded = try MLX.loadArrays(url: url)
            weights.merge(loaded) { _, new in new }
        }
        return weights
    }
}

func dramaBoxFilterPrefix(_ state: [String: MLXArray], prefix: String) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(state.count)
    for (key, value) in state where key.hasPrefix(prefix) {
        out[String(key.dropFirst(prefix.count))] = value
    }
    return out
}

func dramaBoxLoadModuleWeights(
    _ module: Module,
    state: [String: MLXArray],
    prefix: String,
    remap: ((String) -> String)? = nil,
    transform: ((String, MLXArray) -> MLXArray)? = nil
) throws {
    var sub = dramaBoxFilterPrefix(state, prefix: prefix)
    if let remap {
        sub = Dictionary(uniqueKeysWithValues: sub.map { (remap($0.key), $0.value) })
    }
    if let transform {
        sub = Dictionary(uniqueKeysWithValues: sub.map { ($0.key, transform($0.key, $0.value)) })
    }
    guard !sub.isEmpty else {
        throw DramaBoxError.generationFailed("No weights with prefix \(prefix)")
    }
    try module.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
}
