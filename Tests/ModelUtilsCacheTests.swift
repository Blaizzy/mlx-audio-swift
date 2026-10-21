import Foundation
import HuggingFace
import os
import Testing
@testable import MLXAudioCore

private final class ModelCacheProtocol: URLProtocol, @unchecked Sendable {
    typealias Handler = @Sendable (URLRequest) throws -> (HTTPURLResponse, Data)
    static let handlers = OSAllocatedUnfairLock(initialState: [String: Handler]())

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            guard let handler = Self.handlers.withLock({ $0[request.url!.host!] }) else {
                throw URLError(.unsupportedURL)
            }
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private struct ModelCacheFixture {
    struct NetworkState {
        var offline = false
        var failTree = false
        var requests: [String] = []
    }

    static let commit = String(repeating: "a", count: 40)
    static let baseFiles = ["config.json": Data("{}".utf8), "model.safetensors": Data([1])]

    let root: URL
    let cache: HubCache
    let repoID: Repo.ID = "mlx-audio-tests/cache-fixture"
    let modelDir: URL
    let host: URL
    let session: URLSession
    let state = OSAllocatedUnfairLock(initialState: NetworkState())

    init(extraFiles: [String: Data] = [:], cachedFiles: [String]? = nil, cold: Bool = false) throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        cache = HubCache(cacheDirectory: root)
        modelDir = root.appendingPathComponent("mlx-audio/mlx-audio-tests_cache-fixture")
        host = URL(string: "https://cache-\(UUID().uuidString.lowercased()).invalid")!
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ModelCacheProtocol.self]
        session = URLSession(configuration: configuration)

        let files = Self.baseFiles.merging(extraFiles) { _, extra in extra }
        if !cold {
            let snapshot = try cache.snapshotPath(repo: repoID, kind: .model, commitHash: Self.commit)
            for name in cachedFiles ?? Array(Self.baseFiles.keys) {
                for directory in [modelDir, snapshot] {
                    let url = directory.appendingPathComponent(name)
                    try FileManager.default.createDirectory(
                        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try files[name]!.write(to: url)
                }
            }
            try cache.updateRef(repo: repoID, kind: .model, ref: "main", commit: Self.commit)
        }

        let state = state
        let repo = repoID.description
        let hostname = host.host!
        ModelCacheProtocol.handlers.withLock { handlers in
            handlers[hostname] = { request in
                let url = request.url!
                let path = url.path
                let fail = state.withLock { state in
                    state.requests.append(path)
                    return state.offline || (state.failTree && path.contains("/tree/"))
                }
                if fail { throw URLError(.notConnectedToInternet) }

                let body: Data
                var headers = ["Content-Type": "application/json"]
                if path.hasPrefix("/api/models/\(repo)/revision/") {
                    body = try JSONSerialization.data(withJSONObject: [
                        "id": repo, "sha": Self.commit,
                        "siblings": files.map { ["rfilename": $0.key, "size": $0.value.count] },
                    ])
                } else if path.contains("/tree/") {
                    body = try JSONSerialization.data(withJSONObject: files.map {
                        ["path": $0.key, "type": "file", "size": $0.value.count, "oid": $0.key] as [String: Any]
                    })
                } else if let range = path.range(of: "/resolve/\(Self.commit)/"),
                          let data = files[String(path[range.upperBound...])] {
                    body = request.httpMethod == "HEAD" ? Data() : data
                    headers = ["Content-Length": String(data.count), "X-Repo-Commit": Self.commit,
                               "ETag": "\"\(url.lastPathComponent)\""]
                } else {
                    throw URLError(.badURL)
                }
                return (HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!, body)
            }
        }
    }

    var manifestURL: URL { modelDir.appendingPathComponent(".mlx-audio-snapshot.json") }
    var requestCount: Int { state.withLock { $0.requests.count } }

    func resolve(_ patterns: [String] = [], requiredExtension: String = "safetensors") async throws -> URL {
        try await ModelUtils.resolveOrDownloadModel(
            client: HubClient(session: session, host: host, cache: cache),
            cache: cache, repoID: repoID, requiredExtension: requiredExtension,
            additionalMatchingPatterns: patterns
        )
    }

    func cleanUp() {
        session.invalidateAndCancel()
        ModelCacheProtocol.handlers.withLock { _ = $0.removeValue(forKey: host.host!) }
        try? FileManager.default.removeItem(at: root)
    }
}

@Suite("ModelUtils cache completeness")
struct ModelUtilsCacheTests {
    @Test(arguments: [[], ["*.json"]])
    func legacyDefaultPatternsWorkOffline(patterns: [String]) async throws {
        let fixture = try ModelCacheFixture()
        defer { fixture.cleanUp() }
        fixture.state.withLock { $0.offline = true }

        let resolved = try await fixture.resolve(patterns, requiredExtension: ".safetensors")
        #expect(resolved.path == fixture.modelDir.path)
        #expect(fixture.requestCount == 0)
    }

    @Test func offlinePartialSnapshotIsRejectedWithoutCertifyingFiles() async throws {
        let fixture = try ModelCacheFixture(extraFiles: ["am.mvn": Data([2])])
        defer { fixture.cleanUp() }
        fixture.state.withLock { $0.offline = true }

        await #expect(throws: (any Error).self) { try await fixture.resolve(["*.mvn"]) }
        #expect(fixture.requestCount > 0)
        #expect(!FileManager.default.fileExists(atPath: fixture.manifestURL.path))
        #expect(FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("model.safetensors").path))
    }

    @Test func laterLoadFetchesExtraFilesAndThenWorksOffline() async throws {
        let extraFiles = ["am.mvn": Data([2, 3]), "tokenizer/vocab.model": Data([4])]
        let fixture = try ModelCacheFixture(extraFiles: extraFiles)
        defer { fixture.cleanUp() }
        _ = try await fixture.resolve() // The initial model-type lookup.
        #expect(fixture.requestCount == 0)

        _ = try await fixture.resolve(["*.mvn", "tokenizer*"])
        for (name, data) in extraFiles {
            #expect(try Data(contentsOf: fixture.modelDir.appendingPathComponent(name)) == data)
        }
        let count = fixture.requestCount
        fixture.state.withLock { $0.offline = true }
        _ = try await fixture.resolve(["*.mvn", "*.model"])
        #expect(fixture.requestCount == count)
    }

    @Test func unmatchedPatternIsCompleteAfterOneListing() async throws {
        let fixture = try ModelCacheFixture()
        defer { fixture.cleanUp() }
        _ = try await fixture.resolve(["*.model"])
        let count = fixture.requestCount
        #expect(count > 0)

        fixture.state.withLock { $0.offline = true }
        _ = try await fixture.resolve(["*.model", "merges.txt"])
        #expect(fixture.requestCount == count)
    }

    @Test func snapshotOfflineFallbackCannotCertifyMissingFiles() async throws {
        let fixture = try ModelCacheFixture(extraFiles: ["am.mvn": Data([2])])
        defer { fixture.cleanUp() }
        fixture.state.withLock { $0.failTree = true }

        await #expect(throws: ModelUtilsError.self) { try await fixture.resolve(["*.mvn"]) }
        #expect(!FileManager.default.fileExists(atPath: fixture.manifestURL.path))
        fixture.state.withLock { $0.failTree = false }
        _ = try await fixture.resolve(["*.mvn"])
        #expect(try Data(contentsOf: fixture.modelDir.appendingPathComponent("am.mvn")) == Data([2]))
    }

    @Test func missingLocalFileCanBeRestoredFromHubCacheOffline() async throws {
        let fixture = try ModelCacheFixture(extraFiles: ["am.mvn": Data([2])])
        defer { fixture.cleanUp() }
        _ = try await fixture.resolve(["*.mvn"])
        let localFile = fixture.modelDir.appendingPathComponent("am.mvn")
        try FileManager.default.removeItem(at: localFile)
        let count = fixture.requestCount
        fixture.state.withLock { $0.offline = true }

        _ = try await fixture.resolve(["*.mvn"])
        #expect(try Data(contentsOf: localFile) == Data([2]))
        #expect(fixture.requestCount == count)
    }

    @Test func coldDownloadRecordsInventoryForLaterPatterns() async throws {
        let fixture = try ModelCacheFixture(extraFiles: ["am.mvn": Data([2])], cold: true)
        defer { fixture.cleanUp() }
        _ = try await fixture.resolve()
        #expect(FileManager.default.fileExists(atPath: fixture.manifestURL.path))
        #expect(!FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("am.mvn").path))

        _ = try await fixture.resolve(["*.mvn"])
        #expect(try Data(contentsOf: fixture.modelDir.appendingPathComponent("am.mvn")) == Data([2]))
        #expect(fixture.state.withLock { $0.requests.filter { $0.contains("/revision/") }.count } == 1)
    }

    @Test(arguments: [Data(), Data([9]), Data([9, 9, 9])])
    func damagedExtraFileIsNotAcceptedOffline(damaged: Data) async throws {
        let fixture = try ModelCacheFixture(extraFiles: ["am.mvn": Data([2, 3])])
        defer { fixture.cleanUp() }
        _ = try await fixture.resolve(["*.mvn"])
        try damaged.write(to: fixture.modelDir.appendingPathComponent("am.mvn"))
        let snapshot = try fixture.cache.snapshotPath(
            repo: fixture.repoID, kind: .model, commitHash: ModelCacheFixture.commit)
        try FileManager.default.removeItem(at: snapshot.appendingPathComponent("am.mvn"))
        fixture.state.withLock { $0.offline = true }

        await #expect(throws: ModelUtilsError.self) { try await fixture.resolve(["*.mvn"]) }
    }

    @Test func unsafeListedPathIsRejected() async throws {
        let fixture = try ModelCacheFixture(extraFiles: ["tokenizer/../../outside.model": Data([2])])
        defer { fixture.cleanUp() }
        await #expect(throws: ModelUtilsError.self) { try await fixture.resolve(["tokenizer*"]) }
        #expect(!FileManager.default.fileExists(atPath: fixture.manifestURL.path))
    }
}
