import Foundation
import HuggingFace
import MLX

/// Prepares published containers once, without requiring the Python package or modifying the source checkpoint.
enum Phonon2Checkpoint {
    struct Metadata: Decodable {
        struct Artifact: Decodable {
            let filename: String
            let sha256: String
            let unpacked_bytes: Int
            let container: String?
            let container_sha256: String?
        }
        struct Fermion: Decodable {
            let container_format: String?
            let container: String?
            let container_sha256: String?
        }
        let model_type: String?
        let artifact: Artifact?
        let fermion: Fermion?
        let languages: [String]?

        var isContainer: Bool {
            model_type == "parakeet_tdt_five_value"
                || fermion?.container_format == Phonon2Container.format
        }
    }

    private static let preparationLock = NSLock()

    private struct Marker: Codable {
        let version: Int
        let configBytes: Int
        let weightBytes: Int
    }

    static func prepare(_ directory: URL, cache: HubCache) throws -> URL {
        preparationLock.lock()
        defer { preparationLock.unlock() }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let metadata = try JSONDecoder().decode(Metadata.self, from: data)
        guard metadata.isContainer else { return directory }
        let source: URL
        let digest: String
        if let artifact = metadata.artifact {
            try Phonon2Archive.validatePath(artifact.filename)
            guard artifact.filename.hasSuffix(".bps.tar.zst"), artifact.unpacked_bytes > 0,
                artifact.unpacked_bytes <= 1 << 30
            else { throw Phonon2Error.invalidCheckpoint("Invalid release metadata") }
            source = directory.appendingPathComponent(artifact.filename)
            digest = artifact.sha256
            try Phonon2Archive.verify(
                Data(contentsOf: source, options: .mappedIfSafe), sha256: digest)
        } else {
            let filename = metadata.fermion?.container ?? "model.fermion"
            try Phonon2Archive.validatePath(filename)
            source = directory.appendingPathComponent(filename)
            let container = try Data(contentsOf: source, options: .mappedIfSafe)
            digest = Phonon2Archive.digest(container)
            if let expected = metadata.fermion?.container_sha256 {
                try Phonon2Archive.verify(container, sha256: expected)
            }
        }
        let parent = cache.cacheDirectory.appendingPathComponent("mlx-audio/phonon2")
        let destination = parent.appendingPathComponent(digest)
        if isComplete(destination) { return destination }
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(".\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let container: URL
        let rawConfig: Data
        if let artifact = metadata.artifact {
            try Phonon2Archive.extract(
                source, to: temporary, expectedBytes: artifact.unpacked_bytes)
            let filename = artifact.container ?? "model.fermion"
            try Phonon2Archive.validatePath(filename)
            container = temporary.appendingPathComponent(filename)
            if let expected = artifact.container_sha256 {
                try Phonon2Archive.verify(
                    Data(contentsOf: container, options: .mappedIfSafe), sha256: expected)
            }
            rawConfig = try Data(contentsOf: temporary.appendingPathComponent("config.json"))
        } else {
            container = source
            rawConfig = data
        }
        let (weights, modules) = try Phonon2Container.read(container)
        guard let graph = try JSONSerialization.jsonObject(with: rawConfig) as? [String: Any] else {
            throw Phonon2Error.invalidConfiguration("Missing Parakeet configuration")
        }
        var runtime: [String: Any] = [:]
        for key in ["preprocessor", "encoder", "decoder", "joint", "decoding"] {
            guard let value = graph[key] else {
                throw Phonon2Error.invalidConfiguration("Missing \(key)")
            }
            runtime[key] = value
        }
        runtime["model_type"] = "phonon2"
        runtime["phonon2_modules"] = modules.map { ["name": $0.name, "shape": $0.shape] }
        runtime["languages"] = metadata.languages ?? graph["languages"]
        let config = try JSONSerialization.data(withJSONObject: runtime, options: [.sortedKeys])
        _ = try JSONDecoder().decode(Phonon2Config.self, from: config)
        try MLX.save(arrays: weights, url: temporary.appendingPathComponent("model.safetensors"))
        try config.write(to: temporary.appendingPathComponent("config.json"), options: .atomic)
        if metadata.artifact != nil { try FileManager.default.removeItem(at: container) }
        let marker = Marker(
            version: 1, configBytes: config.count,
            weightBytes: try size(temporary.appendingPathComponent("model.safetensors")))
        try JSONEncoder().encode(marker).write(
            to: temporary.appendingPathComponent(".complete.json"), options: .atomic)
        // Concurrent loaders may finish the same immutable artifact while this one prepares it.
        if isComplete(destination) { return destination }
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        do { try FileManager.default.moveItem(at: temporary, to: destination) } catch {
            if !isComplete(destination) { throw error }
        }
        return destination
    }

    private static func size(_ url: URL) throws -> Int {
        try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    }

    private static func isComplete(_ directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(".complete.json")),
            let marker = try? JSONDecoder().decode(Marker.self, from: data), marker.version == 1,
            marker.configBytes > 0, marker.weightBytes > 0
        else { return false }
        return (try? size(directory.appendingPathComponent("config.json"))) == marker.configBytes
            && (try? size(directory.appendingPathComponent("model.safetensors")))
                == marker.weightBytes
    }
}
