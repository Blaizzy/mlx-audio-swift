import Foundation
import HuggingFace
import MLX
import MLXNN

extension NemotronVoiceChatCodec {
    public static let defaultRepository = "mlx-community/NemotronLabs-VoiceChat-11B-4bit"

    /// Load a codec-only directory or the codec tensors from a VoiceChat artifact.
    /// Published Hub artifacts are cached as a tensor subset; unrelated language
    /// weights are neither downloaded nor instantiated. No precision is changed.
    public static func fromPretrained(
        _ source: String = defaultRepository, revision: String? = nil,
        hfToken: String? = nil, cache: HubCache = .default
    ) async throws -> NemotronVoiceChatCodec {
        let path = NSString(string: source).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue {
            return try fromModelDirectory(URL(fileURLWithPath: path))
        }
        guard !source.hasPrefix("/"), !source.hasPrefix("."), !source.hasPrefix("~"),
              let repo = Repo.ID(rawValue: source)
        else { throw NemotronVoiceChatCodecError.invalidArtifact("directory or repository not found: \(source)") }
        let client: HubClient
        if let hfToken, !hfToken.isEmpty {
            client = HubClient(host: HubClient.defaultHost, bearerToken: hfToken, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let directory = try await NemotronCodecHub.resolve(
            repo: repo, revision: revision, client: client, cache: cache)
        return try fromModelDirectory(directory)
    }

    /// Load original NeMo layouts, Python-prepared MLX layouts, or the published
    /// VoiceChat runtime-v2 packaging. Required codec parameters load strictly.
    public static func fromModelDirectory(_ directory: URL) throws -> NemotronVoiceChatCodec {
        let root = try artifactDictionary(directory.appendingPathComponent("config.json"))
        let config = try codecConfiguration(root)
        let model = NemotronVoiceChatCodec(config: config)
        let prepared = (root["weights_format"] as? String) == "mlx"
            || ((root["mlx_audio"] as? [String: Any])?["prepared_weights"] as? Bool == true)
        var weights: [String: MLXArray] = [:]
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [String]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try artifactDictionary(indexURL)
            guard let mapping = index["weight_map"] as? [String: String] else {
                throw NemotronVoiceChatCodecError.invalidArtifact("invalid safetensors index")
            }
            files = Array(Set(mapping.filter { codecKey($0.key) }.values)).sorted()
        } else {
            files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
                .filter { $0.hasSuffix(".safetensors") }.sorted()
        }
        guard !files.isEmpty else { throw NemotronVoiceChatCodecError.invalidWeights("no codec safetensors") }
        for file in files {
            try validatePath(file)
            let arrays = try loadArrays(url: directory.appendingPathComponent(file))
            for (key, value) in arrays where codecKey(key) {
                guard weights[key] == nil else {
                    throw NemotronVoiceChatCodecError.invalidWeights("duplicate tensor \(key)")
                }
                weights[key] = value
            }
        }
        weights = try model.sanitize(weights: weights, sourceLayout: !prepared)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model.parameters())
        return model
    }

    /// Adapt names and original Conv1d/ConvTranspose1d layouts in memory. Pass
    /// sourceLayout=false for Python-prepared weights; shape inference alone is
    /// ambiguous when kernel and channel dimensions happen to be equal.
    public func sanitize(weights: [String: MLXArray], sourceLayout: Bool = true) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (sourceKey, sourceValue) in weights {
            let key = Self.unprefixedKey(sourceKey)
                .replacingOccurrences(of: "prvq._variance_list.", with: "prvq.variance_list.")
            guard key.hasPrefix("encoder.") || key.hasPrefix("decoder.") || key.hasPrefix("prvq.") else { continue }
            guard result[key] == nil else {
                throw NemotronVoiceChatCodecError.invalidWeights("duplicate normalized tensor \(key)")
            }
            var value = sourceValue
            if sourceLayout, key.hasSuffix(".weight"), value.ndim == 3 {
                let parts = key.split(separator: ".")
                let layer = parts.count > 2 ? Int(parts[2]) : nil
                let width = config.blocksPerStage + 1
                let transposed = key.hasPrefix("decoder.layers.")
                    && layer.map { $0 < width * config.downsampleRates.count && $0 % width == 0 } == true
                    && !key.contains("dwconv")
                value = transposed ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            }
            result[key] = value
        }
        return result
    }

    static func unprefixedKey(_ key: String) -> String {
        for prefix in ["tts_model.audio_codec.", "audio_codec."] where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return key
    }

    static func codecKey(_ key: String) -> Bool {
        let key = unprefixedKey(key)
        return key.hasPrefix("encoder.") || key.hasPrefix("decoder.") || key.hasPrefix("prvq.")
    }

    static func artifactDictionary(_ url: URL) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NemotronVoiceChatCodecError.invalidArtifact("expected JSON object in \(url.lastPathComponent)")
        }
        return value
    }

    static func codecConfiguration(_ root: [String: Any]) throws -> NemotronVoiceChatCodecConfig {
        var values: [String: Any]
        if let codec = root["codec_config"] as? [String: Any] {
            values = codec
        } else if let model = root["model"] as? [String: Any],
                  let speechRoot = model["speech_generation"] as? [String: Any] {
            let speech = speechRoot["model"] as? [String: Any] ?? speechRoot
            guard let codec = speech["codec_config"] as? [String: Any] else {
                throw NemotronVoiceChatCodecError.invalidArtifact("missing speech codec configuration")
            }
            values = codec
            if values["sample_rate"] == nil {
                values["sample_rate"] = (root["data"] as? [String: Any])?["target_sample_rate"]
                    ?? (speechRoot["data"] as? [String: Any])?["target_sample_rate"] ?? 22_050
            }
        } else if root["base_channels"] != nil || root["base_hidden_size"] != nil {
            values = root
        } else {
            throw NemotronVoiceChatCodecError.invalidArtifact("missing codec configuration")
        }
        let aliases = ["base_channels": "base_hidden_size", "channel_multipliers": "channel_mult",
                       "downsample_rates": "rates", "blocks_per_stage": "num_blocks",
                       "block_kernel_size": "kernel_size", "latent_dim": "latent_size"]
        for (key, oldKey) in aliases where values[key] == nil { values[key] = values[oldKey] }
        return try JSONDecoder().decode(NemotronVoiceChatCodecConfig.self,
                                       from: JSONSerialization.data(withJSONObject: values))
    }

    static func validatePath(_ path: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\0"),
              !path.split(separator: "/").contains("..") else {
            throw NemotronVoiceChatCodecError.invalidArtifact("unsafe weight path")
        }
    }
}

enum NemotronCodecHub {
    struct Tensor: Sendable {
        let name: String
        let dtype: String
        let shape: [Int]
        let start: Int64
        let end: Int64
        let url: URL
    }

    static func resolve(repo: Repo.ID, revision: String?, client: HubClient, cache: HubCache) async throws -> URL {
        let fileManager = FileManager.default
        let root = cache.cacheDirectory.appendingPathComponent("mlx-audio-codecs/nemotron-voicechat")
            .appendingPathComponent(repo.description.replacingOccurrences(of: "/", with: "--"))
        let refURL = root.appendingPathComponent("resolved-revision.txt")
        let cachedRevision = revision ?? (try? String(contentsOf: refURL, encoding: .utf8))
        if let cachedRevision, cachedRevision.range(of: "^[a-fA-F0-9]{40}$", options: .regularExpression) != nil {
            let cached = root.appendingPathComponent(cachedRevision)
            if hasCachedArtifact(cached) { return cached }
        }
        let info = try await client.getModel(repo, revision: revision ?? "main", filesMetadata: true)
        guard let commit = info.sha, let files = info.siblings,
              commit.range(of: "^[a-fA-F0-9]{40}$", options: .regularExpression) != nil else {
            throw NemotronVoiceChatCodecError.invalidArtifact("Hub did not supply a pinned file listing")
        }
        let destination = root.appendingPathComponent(commit)
        // Explicit branch/tag revisions refresh metadata, then reuse the exact
        // commit's completed subset instead of downloading its tensor ranges again.
        if hasCachedArtifact(destination) {
            if revision == nil { try commit.write(to: refURL, atomically: true, encoding: .utf8) }
            return destination
        }
        let staging = root.appendingPathComponent("partial-\(UUID().uuidString)")
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }
        _ = try await client.downloadFile(at: "config.json", from: repo,
                                          to: staging.appendingPathComponent("config.json"), revision: commit)
        let config = try NemotronVoiceChatCodec.artifactDictionary(staging.appendingPathComponent("config.json"))
        _ = try NemotronVoiceChatCodec.codecConfiguration(config)
        let names: [String]
        if files.contains(where: { $0.relativeFilename == "model.safetensors.index.json" }) {
            let indexURL = staging.appendingPathComponent("model.safetensors.index.json")
            _ = try await client.downloadFile(at: "model.safetensors.index.json", from: repo, to: indexURL, revision: commit)
            guard let map = try NemotronVoiceChatCodec.artifactDictionary(indexURL)["weight_map"] as? [String: String] else {
                throw NemotronVoiceChatCodecError.invalidArtifact("missing weight map")
            }
            names = Array(Set(map.filter { NemotronVoiceChatCodec.codecKey($0.key) }.values)).sorted()
            try fileManager.removeItem(at: indexURL)
        } else {
            names = files.map(\.relativeFilename).filter { $0.hasSuffix(".safetensors") }.sorted()
        }
        guard !names.isEmpty else { throw NemotronVoiceChatCodecError.invalidWeights("no codec shards") }
        let token = await client.bearerToken
        var tensors: [Tensor] = []
        for name in names {
            try NemotronVoiceChatCodec.validatePath(name)
            guard files.contains(where: { $0.relativeFilename == name }),
                  let size = files.first(where: { $0.relativeFilename == name })?.size else {
                throw NemotronVoiceChatCodecError.invalidArtifact("codec shard absent from file listing")
            }
            let url = client.host.appendingPathComponent(repo.description)
                .appendingPathComponent("resolve").appendingPathComponent(commit).appendingPathComponent(name)
            let first = try await bytes(url: url, range: 0..<8, token: token)
            let count = first.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            guard count > 0, count <= 64 * 1_024 * 1_024, count + 8 < size else {
                throw NemotronVoiceChatCodecError.invalidArtifact("invalid safetensors header size")
            }
            let header = try await bytes(url: url, range: 8..<Int64(8 + count), token: token)
            tensors += try parseHeader(header, url: url, dataStart: Int64(8 + count), fileSize: Int64(size))
        }
        guard !tensors.isEmpty, Set(tensors.map(\.name)).count == tensors.count else {
            throw NemotronVoiceChatCodecError.invalidWeights("missing or duplicate codec tensors")
        }
        // A compact safetensors cache preserves original names, shapes, dtype and
        // layout. Filtering bytes is not checkpoint conversion or requantization.
        var header: [String: Any] = [:]
        var offsets: [Int64] = []
        var length: Int64 = 0
        for tensor in tensors {
            offsets.append(length)
            let next = length + tensor.end - tensor.start
            header[tensor.name] = ["dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [length, next]]
            length = next
        }
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        headerData.append(Data(repeating: 0x20, count: (8 - headerData.count % 8) % 8))
        var headerLength = UInt64(headerData.count).littleEndian
        let weightsURL = staging.appendingPathComponent("model.safetensors")
        fileManager.createFile(atPath: weightsURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: weightsURL)
        defer { try? handle.close() }
        try withUnsafeBytes(of: &headerLength) { try handle.write(contentsOf: Data($0)) }
        try handle.write(contentsOf: headerData)
        let dataStart = Int64(8 + headerData.count)
        try handle.truncate(atOffset: UInt64(dataStart + length))
        for start in stride(from: 0, to: tensors.count, by: 4) {
            try Task.checkCancellation()
            let end = min(start + 4, tensors.count)
            try await withThrowingTaskGroup(of: (Int, Data).self) { group in
                for index in start..<end {
                    let tensor = tensors[index]
                    group.addTask {
                        (index, try await bytes(url: tensor.url, range: tensor.start..<tensor.end, token: token))
                    }
                }
                for try await (index, data) in group {
                    try handle.seek(toOffset: UInt64(dataStart + offsets[index]))
                    try handle.write(contentsOf: data)
                }
            }
        }
        try handle.synchronize()
        try handle.close()
        if !hasCachedArtifact(destination) {
            if fileManager.fileExists(atPath: destination.path) { try fileManager.removeItem(at: destination) }
            try fileManager.moveItem(at: staging, to: destination)
        }
        if revision == nil { try commit.write(to: refURL, atomically: true, encoding: .utf8) }
        return destination
    }

    private static func hasCachedArtifact(_ directory: URL) -> Bool {
        for (name, minimumSize) in [("model.safetensors", 8), ("config.json", 0)] {
            let values = try? directory.appendingPathComponent(name)
                .resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values?.isRegularFile == true, (values?.fileSize ?? 0) > minimumSize else { return false }
        }
        return true
    }

    static func parseHeader(_ data: Data, url: URL, dataStart: Int64, fileSize: Int64) throws -> [Tensor] {
        guard let header = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NemotronVoiceChatCodecError.invalidArtifact("invalid safetensors header")
        }
        let widths: [String: Int64] = ["F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                                      "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1]
        var result: [Tensor] = []
        for (name, raw) in header where NemotronVoiceChatCodec.codecKey(name) {
            guard let item = raw as? [String: Any], let dtype = item["dtype"] as? String,
                  let width = widths[dtype], let shape = item["shape"] as? [Int],
                  let offsets = item["data_offsets"] as? [Int64], offsets.count == 2,
                  offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= fileSize - dataStart else {
                throw NemotronVoiceChatCodecError.invalidArtifact("invalid codec tensor header: \(name)")
            }
            var size = width
            for dimension in shape {
                let next = size.multipliedReportingOverflow(by: Int64(dimension))
                guard dimension > 0, !next.overflow else {
                    throw NemotronVoiceChatCodecError.invalidArtifact("invalid codec tensor shape")
                }
                size = next.partialValue
            }
            guard size == offsets[1] - offsets[0] else {
                throw NemotronVoiceChatCodecError.invalidArtifact("codec tensor byte count does not match shape")
            }
            result.append(Tensor(name: name, dtype: dtype, shape: shape,
                                 start: dataStart + offsets[0], end: dataStart + offsets[1], url: url))
        }
        let sorted = result.sorted { $0.start < $1.start }
        for (left, right) in zip(sorted, sorted.dropFirst()) where left.end > right.start {
            throw NemotronVoiceChatCodecError.invalidArtifact("overlapping codec tensor offsets")
        }
        return sorted
    }

    private static func bytes(url: URL, range: Range<Int64>, token: String?) async throws -> Data {
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 180)
        request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let validator = NemotronCodecRangeValidator(range: range)
        do {
            let (data, response) = try await URLSession.shared.data(for: request, delegate: validator)
            guard validator.accepts(response), data.count == range.count else {
                throw NemotronVoiceChatCodecError.invalidRangeResponse
            }
            return data
        } catch let error as URLError where error.code == .cancelled && !Task.isCancelled {
            // A response rejected by the delegate must not look like a user
            // cancellation, or encourage retrying a full unsupported download.
            throw NemotronVoiceChatCodecError.invalidRangeResponse
        }
    }
}

/// Cancel an unsupported Range response before URLSession receives its body.
final class NemotronCodecRangeValidator: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let range: Range<Int64>
    init(range: Range<Int64>) { self.range = range }
    func accepts(_ response: URLResponse) -> Bool {
        guard let http = response as? HTTPURLResponse, http.statusCode == 206,
              let value = http.value(forHTTPHeaderField: "Content-Range") else { return false }
        return value.hasPrefix("bytes \(range.lowerBound)-\(range.upperBound - 1)/")
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        completionHandler(accepts(response) ? .allow : .cancel)
    }
}
