import Foundation
import HuggingFace
import MLXAudioCore

public final class MisakiTextProcessor: TextProcessor, @unchecked Sendable {
    private var usG2P: EnglishG2P?
    private var gbG2P: EnglishG2P?
    private let lock = NSLock()
    private nonisolated(unsafe) var resourceDirectory: URL?

    private static let g2pRepo = Repo.ID(namespace: "beshkenadze", name: "kitten-tts-g2p")

    public init() {}

    public func prepare() async throws {
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: Self.g2pRepo,
            requiredExtension: "safetensors"
        )
        lock.withLock {
            resourceDirectory = dir
        }
    }

    public func process(text: String, language: String?) throws -> String {
        let british = language?.lowercased().contains("gb") == true
        let g2p = try getG2P(british: british)
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }

    private func getG2P(british: Bool) throws -> EnglishG2P {
        lock.lock()
        defer { lock.unlock() }
        guard let dir = resourceDirectory else {
            throw MisakiError.resourcesNotDownloaded
        }
        if british {
            if let cached = gbG2P { return cached }
            let g2p = try EnglishG2P(british: true, directory: dir)
            gbG2P = g2p
            return g2p
        } else {
            if let cached = usG2P { return cached }
            let g2p = try EnglishG2P(british: false, directory: dir)
            usG2P = g2p
            return g2p
        }
    }

    enum MisakiError: Error, LocalizedError {
        case resourcesNotDownloaded

        var errorDescription: String? {
            switch self {
            case .resourcesNotDownloaded:
                return "G2P resources not downloaded. Call prepare() first."
            }
        }
    }
}
