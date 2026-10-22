import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

/// English Phonon-2 recognition using a Parakeet TDT graph and exact packed encoder layers.
public final class Phonon2Model: Module, STTGenerationModel {
    public let config: Phonon2Config
    @ModuleInfo var backbone: ParakeetModel

    public var computeDType: DType {
        get { backbone.computeDType }
        set { backbone.computeDType = newValue }
    }

    public var defaultGenerationParameters: STTGenerateParameters { backbone.defaultGenerationParameters }

    private init(config: Phonon2Config, backbone: ParakeetModel) {
        self.config = config
        self.backbone = backbone
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        // The common STT protocol cannot throw. Empty input produces an empty transcript.
        guard audio.size > 0 else { return STTOutput(text: "") }
        return backbone.generate(audio: audio, generationParameters: generationParameters)
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        guard audio.size > 0 else {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: STTError.invalidInput("Expected nonempty mono audio"))
            }
        }
        return backbone.generateStream(audio: audio, generationParameters: generationParameters)
    }

    public static func fromDirectory(_ directory: URL, computeDType: DType = .bfloat16) throws -> Phonon2Model {
        let config = try JSONDecoder().decode(Phonon2Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        return Phonon2Model(config: config, backbone: try ParakeetModel.fromDirectory(directory, computeDType: computeDType))
    }

    public static func fromPretrained(
        _ modelPath: String, computeDType: DType = .bfloat16, cache: HubCache = .default
    ) async throws -> Phonon2Model {
        let path = NSString(string: modelPath).expandingTildeInPath
        if FileManager.default.fileExists(atPath: path) {
            return try fromDirectory(URL(fileURLWithPath: path), computeDType: computeDType)
        }
        guard let repoID = Repo.ID(rawValue: modelPath) else { throw STTModelError.invalidRepositoryID(modelPath) }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: "safetensors",
            hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"], cache: cache
        )
        return try fromDirectory(directory, computeDType: computeDType)
    }
}

extension ParakeetModel {
    /// Installs already-prepared packed layers. It does not convert or requantize weights.
    func configurePhonon2(_ config: Phonon2Config) throws -> [Phonon2PackedLinear] {
        var replacements: [(String, Module)] = []
        var packed: [Phonon2PackedLinear] = []
        for row in config.modules {
            let bias = !row.name.hasSuffix("linear_pos") && config.parakeet.encoder.useBias
            let layer = Phonon2PackedLinear(inputDimensions: row.shape[1], outputDimensions: row.shape[0], bias: bias)
            replacements.append((row.name, layer))
            packed.append(layer)
        }
        try update(modules: ModuleChildren.unflattened(replacements), verify: .noUnusedKeys)
        return packed
    }
}
