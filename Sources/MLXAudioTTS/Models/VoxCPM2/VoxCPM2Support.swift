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

enum VoxCPM2AudioResampler {
    static func resample(_ input: [Float], from sourceRate: Int, to targetRate: Int) -> [Float] {
        guard !input.isEmpty, sourceRate > 0, targetRate > 0, sourceRate != targetRate else {
            return input
        }

        let outputCount = max(1, Int((Double(input.count) * Double(targetRate) / Double(sourceRate)).rounded()))
        guard outputCount > 1, input.count > 1 else {
            return Array(repeating: input.first ?? 0, count: outputCount)
        }

        let ratio = Double(sourceRate) / Double(targetRate)
        return (0..<outputCount).map { index in
            let position = Double(index) * ratio
            let left = min(Int(position), input.count - 1)
            let right = min(left + 1, input.count - 1)
            let fraction = Float(position - Double(left))
            return input[left] * (1 - fraction) + input[right] * fraction
        }
    }
}
