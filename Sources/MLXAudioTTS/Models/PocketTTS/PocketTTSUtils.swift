import CryptoKit
import Foundation
import Hub
import HuggingFace
@preconcurrency import MLX

public enum PocketTTSUtils {
    public static let predefinedVoices: [String: String] = [
        "alba": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "marius": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/marius.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "javert": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/javert.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "jean": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/jean.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "fantine": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/fantine.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "cosette": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/cosette.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "eponine": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/eponine.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        "azelma": "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/azelma.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    ]

    public static func cacheDirectory() -> URL {
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("pocket_tts")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public static func downloadIfNecessary(
        _ path: String,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> URL {
        if path.hasPrefix("http://") || path.hasPrefix("https://") {
            let url = URL(string: path)!
            let ext = url.pathExtension.isEmpty ? "bin" : url.pathExtension
            let hash = SHA256.hash(data: Data(path.utf8)).compactMap { String(format: "%02x", $0) }.joined()
            let cached = cacheDirectory().appendingPathComponent("\(hash).\(ext)")
            if FileManager.default.fileExists(atPath: cached.path) {
                return cached
            }
            let (data, _) = try await URLSession.shared.data(from: url)
            try data.write(to: cached)
            return cached
        }
        if path.hasPrefix("hf://") {
            let trimmed = String(path.dropFirst(5))
            let parts = trimmed.split(separator: "/")
            guard parts.count >= 3 else {
                throw NSError(domain: "PocketTTSUtils", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid hf:// path: \(path)"])
            }
            let repoId = "\(parts[0])/\(parts[1])"
            let filenameWithRev = parts.dropFirst(2).joined(separator: "/")
            let filenameParts = filenameWithRev.split(separator: "@", maxSplits: 1, omittingEmptySubsequences: true)
            let filename = String(filenameParts[0])
            // NOTE: Hub.snapshot does not expose a revision parameter; we ignore revision for now.
            let snapshotURL = try await Hub.snapshot(from: repoId, matching: filename, progressHandler: progressHandler)
            return snapshotURL.appendingPathComponent(filename)
        }
        return URL(fileURLWithPath: path)
    }

    public static func loadPredefinedVoice(
        _ voiceName: String,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXArray {
        guard let path = predefinedVoices[voiceName] else {
            throw NSError(domain: "PocketTTSUtils", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unknown voice: \(voiceName)"])
        }
        let fileURL = try await downloadIfNecessary(path, progressHandler: progressHandler)
        let arrays = try MLX.loadArrays(url: fileURL)
        guard let prompt = arrays["audio_prompt"] else {
            throw NSError(domain: "PocketTTSUtils", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing audio_prompt in voice file: \(fileURL.path)"])
        }
        return prompt
    }
}
