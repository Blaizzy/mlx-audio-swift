//  Tests for ModelUtils cache-completeness tracking (.mlx-audio-patterns).
//
//  Run this suite:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:'MLXAudioTests/ModelUtilsCacheTests' \
//      CODE_SIGNING_ALLOWED=NO

import Testing
import Foundation
import HuggingFace
@testable import MLXAudioCore

private func makeTemporaryCacheDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("modelutils-cache-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

/// Pre-populates the on-disk cache layout that `resolveOrDownloadModel`
/// inspects: `<cache>/mlx-audio/<repo with _>/` containing a valid
/// config.json and a non-empty weights file.
private func populateCachedModel(
    cache: HubCache,
    repoID: Repo.ID,
    patternsManifest: String?
) throws -> URL {
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = cache.cacheDirectory
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: modelDir.appendingPathComponent("config.json"))
    try Data([0x01]).write(to: modelDir.appendingPathComponent("model.safetensors"))
    if let patternsManifest {
        try Data(patternsManifest.utf8).write(
            to: modelDir.appendingPathComponent(".mlx-audio-patterns"))
    }
    return modelDir
}

@Suite("ModelUtils cache completeness")
struct ModelUtilsCacheTests {

    @Test func cachedModelWithoutManifestServesPlainLoad() async throws {
        let cacheDir = try makeTemporaryCacheDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }
        let cache = HubCache(cacheDirectory: cacheDir)
        let repoID = try #require(Repo.ID(rawValue: "mlx-audio-tests/cache-fixture"))
        let modelDir = try populateCachedModel(cache: cache, repoID: repoID, patternsManifest: nil)

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: HubClient(cache: cache),
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors"
        )
        #expect(resolved.standardizedFileURL == modelDir.standardizedFileURL)
    }

    @Test func cachedModelWithCoveringManifestServesPatternLoad() async throws {
        let cacheDir = try makeTemporaryCacheDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }
        let cache = HubCache(cacheDirectory: cacheDir)
        let repoID = try #require(Repo.ID(rawValue: "mlx-audio-tests/cache-fixture"))
        let modelDir = try populateCachedModel(
            cache: cache, repoID: repoID, patternsManifest: "*.mvn\n*.model")

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: HubClient(cache: cache),
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == modelDir.standardizedFileURL)
    }

    @Test func cachedModelMissingPatternsIsNotAccepted() async throws {
        let cacheDir = try makeTemporaryCacheDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }
        let cache = HubCache(cacheDirectory: cacheDir)
        // A repo that does not exist: if the incomplete cache is (wrongly)
        // accepted this returns instantly; the fixed behavior falls through
        // to a download which must fail for this repo.
        let repoID = try #require(Repo.ID(rawValue: "mlx-audio-tests/nonexistent-cache-fixture"))
        try populateCachedModel(cache: cache, repoID: repoID, patternsManifest: nil)

        await #expect(throws: (any Error).self) {
            _ = try await ModelUtils.resolveOrDownloadModel(
                client: HubClient(cache: cache),
                cache: cache,
                repoID: repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: ["*.mvn"]
            )
        }
    }
}
