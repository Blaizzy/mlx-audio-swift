import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

/// Granite Speech NAR refines a CTC hypothesis with a bidirectional transcript editor.
public final class GraniteSpeechNARModel: Module, STTGenerationModel {
    public let config: GraniteSpeechNARConfig
    let encoder: GraniteSpeechNAREncoder
    let projector: GraniteSpeechNARProjector
    let editor: GraniteSpeechLanguageModel
    public var tokenizer: Tokenizer?
    public var sampleRate: Int { 16000 }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 0, temperature: 0, topP: 1)
    }

    public init(_ config: GraniteSpeechNARConfig) {
        self.config = config
        encoder = GraniteSpeechNAREncoder(
            config.encoderConfig, layerIndices: config.encoderLayerIndices)
        projector = GraniteSpeechNARProjector(config.projectorConfig)
        editor = GraniteSpeechLanguageModel(config.textConfig, preserveDType: true)
    }

    static func collapse(_ tokens: [Int], blankId: Int) -> [Int] {
        var previous: Int?
        var result: [Int] = []
        for token in tokens {
            if token != previous && token != blankId { result.append(token) }
            previous = token
        }
        return result
    }

    static func insertionSlots(_ hypothesis: [Int], blankId: Int, minimumLength: Int) -> [Int] {
        var tokens = [Int](repeating: blankId, count: max(2 * hypothesis.count + 1, minimumLength))
        for (i, token) in hypothesis.enumerated() { tokens[2 * i + 1] = token }
        return tokens
    }

    /// Feature input is `[1, frames, 160]`; returns edited logits for the text tail.
    func editedLogits(_ features: MLXArray) -> (logits: MLXArray, audioTokens: Int) {
        let encoded = encoder(features.asType(.bfloat16))
        let predictions = argMax(encoded.bpeLogits[0], axis: -1).asType(.int32)
        let hypothesis = Self.collapse(
            predictions.asArray(Int32.self).map(Int.init), blankId: config.blankTokenId)
        var audioEmbeddings = projector(concatenated(encoded.hiddenStates, axis: -1))
        if config.scaleProjectedEmbeddings {
            audioEmbeddings = audioEmbeddings / config.textConfig.embeddingMultiplier
        }
        let ids = Self.insertionSlots(
            hypothesis, blankId: config.blankTokenId, minimumLength: config.minEditSequenceLength)
        let textEmbeddings = editor.embedTokens(MLXArray(ids.map(Int32.init))).asType(
            audioEmbeddings.dtype)
        let input = concatenated([audioEmbeddings[0], textEmbeddings], axis: 0).expandedDimensions(
            axis: 0)
        return (
            editor(inputEmbeddings: input, bidirectional: true, logitsStart: audioEmbeddings.dim(1))
                .asType(input.dtype), audioEmbeddings.dim(1)
        )
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        guard let tokenizer else {
            preconditionFailure("Granite Speech NAR tokenizer is not loaded")
        }
        precondition(audio.ndim == 1 && audio.size > 0, "Expected nonempty mono 16 kHz PCM")
        let start = Date()
        let waveform =
            audio.size < 320 ? padded(audio, widths: [.init((0, 320 - audio.size))]) : audio
        let features = GraniteSpeechNARFrontend.features(waveform).expandedDimensions(axis: 0)
        let (logits, audioTokens) = editedLogits(features)
        let predictions = argMax(logits[0], axis: -1).asType(.int32)
        let tokens = Self.collapse(
            predictions.asArray(Int32.self).map(Int.init), blankId: config.blankTokenId)
        let text = tokenizer.decode(tokens: tokens, skipSpecialTokens: true).trimmingCharacters(
            in: .whitespacesAndNewlines)
        let elapsed = Date().timeIntervalSince(start)
        let promptTokens = audioTokens + logits.dim(1)
        return STTOutput(
            text: text, promptTokens: promptTokens, generationTokens: tokens.count,
            totalTokens: promptTokens + tokens.count,
            generationTps: Double(tokens.count) / max(elapsed, 0.001),
            totalTime: elapsed, peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    /// NAR emits a completed transcript after its single editing pass.
    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = model.value.generate(
                        audio: input.value, generationParameters: generationParameters)
                    try Task.checkCancellation()
                    if !result.text.isEmpty { continuation.yield(.token(result.text)) }
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Accept both IBM's original safetensors names and the prepared MLX bundle.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in GraniteSpeechModel.sanitize(weights: weights) {
            var name = key
            if name.hasPrefix("language_model.model.") {
                name = "editor.model." + name.dropFirst("language_model.model.".count)
            } else if name.hasPrefix("editor.") && !name.hasPrefix("editor.model.") {
                name = "editor.model." + name.dropFirst("editor.".count)
            }
            name = name.replacingOccurrences(of: ".conv.bn.", with: ".conv.batch_norm.")
                .replacingOccurrences(
                    of: ".conv.depth_conv.weight", with: ".conv.depth_conv.conv.weight")
            mapped[name] = value
        }
        return mapped
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> GraniteSpeechNARModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> GraniteSpeechNARModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GraniteSpeechNARConfig.self, from: data)
        try config.validate()
        let model = GraniteSpeechNARModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter {
            $0.pathExtension == "safetensors" && $0.lastPathComponent != "out_llm.safetensors"
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No Granite Speech NAR safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights)
        if let quantization = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                // Prepared NAR bundles call the decoder `editor`, without the extra `model` level.
                let original = path.replacingOccurrences(of: "editor.model.", with: "editor.")
                return weights["\(path).scales"] == nil
                    ? nil
                    : (quantization.quantization(layer: original)
                        ?? quantization.quantization(layer: path))?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        model.train(false)
        eval(model)
        return model
    }
}
