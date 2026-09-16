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
import os
import HuggingFace
@testable import MLXAudioCore

private func makeTemporaryCacheDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("modelutils-cache-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

/// Counts network requests per host and always fails with `.notConnectedToInternet`;
/// used to assert that a cache hit performs zero network traffic.
private final class OfflineCacheProtocol: URLProtocol, @unchecked Sendable {
    static let requests = OSAllocatedUnfairLock(initialState: [String: Int]())

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.requests.withLock {
            $0[request.url!.host!, default: 0] += 1
        }
        client?.urlProtocol(
            self,
            didFailWithError: URLError(.notConnectedToInternet)
        )
    }

    override func stopLoading() {}
}

private struct OfflineCacheFixture {
    let root: URL
    let cache: HubCache
    let repoID: Repo.ID
    let modelDir: URL
    let host: URL
    let session: URLSession

    init() throws {
        root = try makeTemporaryCacheDirectory()
        cache = HubCache(cacheDirectory: root)
        repoID = try #require(
            Repo.ID(rawValue: "mlx-audio-tests/cache-fixture")
        )
        // Pre-populate the mlx-audio layout an earlier download produced.
        modelDir = try populateCachedModel(
            cache: cache,
            repoID: repoID,
            patternsManifest: ""
        )

        // Reproduce the initial partial download in BOTH caches so the
        // snapshot lookup below sees the shape a real offline re-entry
        // produces: the mlx-audio layout under `modelDir`, and a hub
        // snapshot under <cache>/models/<ns>/<name>/snapshots/<hash>
        // containing just config.json and model.safetensors.
        let commit = String(repeating: "a", count: 40)
        let snapshot = try cache.snapshotPath(
            repo: repoID,
            kind: .model,
            commitHash: commit
        )
        try FileManager.default.createDirectory(
            at: snapshot,
            withIntermediateDirectories: true
        )
        for name in ["config.json", "model.safetensors"] {
            try FileManager.default.copyItem(
                at: modelDir.appendingPathComponent(name),
                to: snapshot.appendingPathComponent(name)
            )
        }
        try cache.updateRef(
            repo: repoID,
            kind: .model,
            ref: "main",
            commit: commit
        )

        // Unique hosts isolate request counts when tests run concurrently.
        host = URL(
            string: "https://cache-\(UUID().uuidString.lowercased()).invalid"
        )!
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [OfflineCacheProtocol.self]
        session = URLSession(configuration: configuration)
    }

    var client: HubClient {
        HubClient(session: session, host: host, cache: cache)
    }

    var requestCount: Int {
        OfflineCacheProtocol.requests.withLock { $0[host.host!] ?? 0 }
    }

    func cleanUp() {
        session.invalidateAndCancel()
        OfflineCacheProtocol.requests.withLock {
            _ = $0.removeValue(forKey: host.host!)
        }
        try? FileManager.default.removeItem(at: root)
    }
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

    @Test func offlinePartialSnapshotDoesNotCertifyMissingPatterns() async throws {
        let fixture = try OfflineCacheFixture()
        defer { fixture.cleanUp() }

        await #expect(throws: (any Error).self) {
            _ = try await ModelUtils.resolveOrDownloadModel(
                client: fixture.client,
                cache: fixture.cache,
                repoID: fixture.repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: ["*.mvn"]
            )
        }

        #expect(fixture.requestCount > 0)

        let manifest = try? String(
            contentsOf: fixture.modelDir
                .appendingPathComponent(".mlx-audio-patterns"),
            encoding: .utf8
        )
        #expect(
            !(manifest ?? "")
                .split(separator: "\n")
                .contains("*.mvn")
        )
    }

    @Test func defaultPatternDoesNotRedownloadCachedSnapshot() async throws {
        let fixture = try OfflineCacheFixture()
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.json"]
        )

        #expect(
            resolved.standardizedFileURL
                == fixture.modelDir.standardizedFileURL
        )
        #expect(fixture.requestCount == 0)
    }
}
