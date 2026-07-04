import Foundation
import MLX
import MLXAudioCore

enum VoxCPM2WeightLoader {
    static func loadAllSafetensors(from directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !files.isEmpty else {
            throw AudioGenerationError.modelNotInitialized("No safetensors files found in \(directory.path)")
        }

        var weights: [String: MLXArray] = [:]
        for file in files {
            let arrays = try MLX.loadArrays(url: file)
            weights.merge(arrays) { _, new in new }
        }
        return weights
    }
}
