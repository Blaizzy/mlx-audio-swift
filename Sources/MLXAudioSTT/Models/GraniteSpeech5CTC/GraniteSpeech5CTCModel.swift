import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore

/// Encoder-only English ASR. No language model, translation, or trained live-input protocol.
public final class GraniteSpeech5CTCModel: Module, STTGenerationModel {
    public let config: GraniteSpeech5CTCConfig
    @ModuleInfo public var encoder: GraniteSpeech5CTCEncoder
    public var tokenizer: GraniteSpeech5CTCTokenizer?
    public var sampleRate: Int { 16000 }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 0, temperature: 0, topP: 1, language: "en")
    }
    public init(_ config: GraniteSpeech5CTCConfig) {
        self.config = config
        _encoder.wrappedValue = GraniteSpeech5CTCEncoder(config.encoderConfig)
    }
    public func callAsFunction(_ features: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        encoder.out(encoder(features, attentionMask: attentionMask))
    }
    public static func collapse(_ tokenIds: [Int], blankId: Int = 0) -> [Int] {
        var result: [Int] = [], previous: Int?
        for id in tokenIds {
            if id != previous, id != blankId { result.append(id) }
            previous = id
        }
        return result
    }
    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        guard let tokenizer else { preconditionFailure("Granite TurboCTC tokenizer is not loaded") }
        precondition(audio.ndim == 1 && audio.size > 0, "Expected nonempty mono 16 kHz PCM")
        let start = Date()
        // Each strided block floors time/2; pad tiny clips so final time is nonzero.
        let minimum = max(320, 320 * (1 << config.encoderConfig.subsampleLayers.count))
        let waveform = audio.size < minimum ? padded(audio, widths: [.init((0,minimum - audio.size))]) : audio
        let features = GraniteSpeech5CTCFrontend.features(waveform, numMelBins: config.encoderConfig.numMelBins)
        let logits = self(features.expandedDimensions(axis: 0))
        let predictions = argMax(logits[0], axis: -1).asType(.int32)
        eval(predictions)
        let tokens = Self.collapse(predictions.asArray(Int32.self).map(Int.init), blankId: config.padTokenId)
        let text = tokenizer.decode(tokens), elapsed = Date().timeIntervalSince(start)
        return STTOutput(text: text, language: "en", generationTokens: tokens.count,
            totalTokens: tokens.count, generationTps: Double(tokens.count) / max(elapsed,0.001), totalTime: elapsed)
    }
    /// Full-buffer CTC result streaming, not incremental audio inference.
    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self), input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = model.value.generate(audio: input.value, generationParameters: generationParameters)
                    try Task.checkCancellation()
                    if !result.text.isEmpty { continuation.yield(.token(result.text)) }
                    continuation.yield(.result(result)); continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
    public static func sanitize(weights: [String:MLXArray]) -> [String:MLXArray] {
        var result: [String:MLXArray] = [:]
        for (key, var value) in weights where !key.hasSuffix("num_batches_tracked") {
            if key.hasSuffix("conv.depthwise_conv.weight"), value.ndim == 3, value.dim(2) > value.dim(1) {
                value = value.transposed(0,2,1)
            }
            result[key] = value
        }
        return result
    }
    public static func fromPretrained(_ modelPath: String, cache: HubCache = .default) async throws -> GraniteSpeech5CTCModel {
        let expanded = (modelPath as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            return try fromModelDirectory(URL(fileURLWithPath: expanded))
        }
        guard let repo = Repo.ID(rawValue: modelPath) else { throw STTModelError.invalidRepositoryID(modelPath) }
        let dir = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors",
            hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"], cache: cache)
        return try fromModelDirectory(dir)
    }
    public static func fromModelDirectory(_ directory: URL) throws -> GraniteSpeech5CTCModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GraniteSpeech5CTCConfig.self, from: data)
        try config.validate()
        let model = GraniteSpeech5CTCModel(config)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw STTError.modelNotInitialized("No Granite TurboCTC safetensors found") }
        var weights: [String:MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _,new in new } }
        weights = sanitize(weights: weights)
        if let quantization = try JSONDecoder().decode(BaseConfiguration.self, from: data).perLayerQuantization {
            quantize(model: model) { path, module in
                module is Linear && weights["\(path).scales"] != nil ? quantization.quantization(layer: path)?.asTuple : nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try GraniteSpeech5CTCTokenizer(directory: directory, vocabSize: config.vocabSize)
        model.train(false); eval(model)
        return model
    }
}
