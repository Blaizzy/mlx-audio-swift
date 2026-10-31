import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore

/// SpeechBrain VoxCeleb speaker embeddings from mono 16 kHz audio.
public final class EcapaSpeakerEncoder {
    public static let defaultRepository = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
    let model: EcapaTdnnBackbone

    public init() {
        model = EcapaTdnnBackbone(config: EcapaTdnnConfig(inputSize: 80, channels: 1024,
            embedDim: 192, globalContext: true, reflectPadding: true, centeredStatistics: true))
        model.train(false)
    }
    public static func fromPretrained(_ repository: String = defaultRepository, cache: HubCache = .default) async throws -> EcapaSpeakerEncoder {
        guard let id = Repo.ID(rawValue: repository) else { throw AudioGenerationError.invalidInput("Invalid speaker repository: \(repository)") }
        let file = try await HubClient(cache: cache).downloadFile(at: "model.safetensors", from: id)
        return try fromWeights(file)
    }
    public static func fromWeights(_ file: URL) throws -> EcapaSpeakerEncoder {
        let encoder = EcapaSpeakerEncoder()
        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: file) {
            var name = key.hasPrefix("embedding_model.") ? String(key.dropFirst("embedding_model.".count)) : key
            for index in 0..<4 {
                let prefix = "blocks.block\(index)."
                if name.hasPrefix(prefix) { name = "block\(index)." + name.dropFirst(prefix.count); break }
            }
            weights[name] = value
        }
        try encoder.model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(encoder.model)
        return encoder
    }
    public static func logMel(_ audio: MLXArray) -> MLXArray {
        let spectrum = stft(audio: audio.asType(.float32), window: hammingWindow(size: 400),
            nFft: 400, hopLength: 160, padMode: .constant)
        let power = abs(spectrum) * abs(spectrum)
        let melMax: Float = 2595 * log10(1 + 8000 / 700.0)
        let hz = 700 * (pow(10, MLXArray.linspace(0, melMax, count: 82) / 2595) - 1)
        let centers = hz[1..<81], bandwidths = hz[1..<81] - hz[..<80]
        let frequencies = MLXArray.linspace(0, 8000, count: 201)
        let slopes = (frequencies.expandedDimensions(axis: 1) - centers) / bandwidths
        let filterbank = maximum(0, minimum(slopes + 1, 1 - slopes))
        let mel = 10 * log10(maximum(matmul(power, filterbank), 1e-10))
        return maximum(mel, max(mel) - 80)
    }
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        guard audio.ndim == 1, audio.size >= 1600, all(isFinite(audio)).item(Bool.self) else {
            throw AudioGenerationError.invalidInput("ECAPA expects at least 0.1 seconds of finite mono 16 kHz audio")
        }
        let mel = Self.logMel(audio)
        let embedding = model((mel - mean(mel, axis: 0, keepDims: true)).expandedDimensions(axis: 0))[0]
        return embedding / maximum(sqrt(sum(embedding * embedding)), 1e-12)
    }
}
