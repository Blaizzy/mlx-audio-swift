import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Tokenizers

public struct MingOmniSpeechOptions: Sendable {
    /// Freeform or structured instructions supported by the original Ming Omni checkpoint.
    public var instruction: String?
    public var prompt: String?
    public var cfgScale: Float = 2
    public var sigma: Float = 0.25
    public var flowSteps: Int = 10
    /// Extract a speaker vector from reference audio. The Python-compatible default uses only its acoustic prefix.
    public var automaticSpeakerEmbedding: Bool = false
    public var zeroSpeakerEmbedding: Bool = false
    /// An optional 192-value CAMPPlus embedding, before the speaker projection.
    public var speakerEmbedding: [Float]?
    public init() {}
}

/// Ming Omni 16.8B/A3B speech synthesis with the original Bailing MoE backbone.
public final class MingOmniModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: MingOmniConfiguration
    @ModuleInfo(key: "model") var languageModel: MingOmniLanguageModel
    public let audio: MingAudioVAE
    @ModuleInfo(key: "linear_proj_audio") public var aggregator: MingLatentAggregator
    @ModuleInfo(key: "flowloss") public var flow: MingFlowLoss
    @ModuleInfo(key: "stop_head") var stopHead: Linear
    @ModuleInfo(key: "spk_head") var speakerHead: Linear
    private var tokenizer: (any Tokenizer)?
    private var speakerEncoder: MingCAMPPlus?
    public var options = MingOmniSpeechOptions()
    public var sampleRate: Int { configuration.audio.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 200, temperature: 0)
    }
    public init(_ configuration: MingOmniConfiguration) throws {
        try configuration.text.validate()
        func validBackbone(_ c: MingAudioBackboneConfiguration) -> Bool {
            let q = c.backbone
            return q.hiddenSize > 0 && q.hiddenLayers >= 0 && q.attentionHeads > 0
                && q.hiddenSize % q.attentionHeads == 0 && q.kvHeads > 0
                && q.attentionHeads % q.kvHeads == 0 && q.intermediateSize > 0
        }
        guard validBackbone(configuration.audio.encoder), validBackbone(configuration.audio.decoder),
              configuration.flow.depth >= 0, configuration.aggregator.depth >= 0,
              configuration.flow.patchSize > 0,
              configuration.flow.historyPatchSize >= configuration.flow.patchSize,
              configuration.flow.hiddenSize > 0, configuration.flow.numHeads > 0,
              configuration.flow.hiddenSize % configuration.flow.numHeads == 0,
              configuration.aggregator.hiddenSize > 0, configuration.aggregator.numHeads > 0,
              configuration.aggregator.hiddenSize % configuration.aggregator.numHeads == 0,
              configuration.audio.patchSize > 0, configuration.audio.sampleRate > 0,
              configuration.audio.encoder.hopSize > 0, configuration.audio.encoder.inputDim > 0,
              configuration.audio.decoder.outputDim > 0, configuration.audio.encoder.latentDim > 0,
              configuration.audio.encoder.latentDim == configuration.audio.decoder.latentDim else {
            throw AudioGenerationError.invalidInput("Invalid Ming Omni speech stack configuration")
        }
        self.configuration = configuration
        let dim = configuration.text.hiddenSize, latent = configuration.audio.encoder.latentDim
        _languageModel.wrappedValue = MingOmniLanguageModel(configuration.text)
        audio = MingAudioVAE(configuration.audio)
        _aggregator.wrappedValue = MingLatentAggregator(latentDim: latent, outputDim: dim, config: configuration.aggregator)
        _flow.wrappedValue = MingFlowLoss(latentDim: latent, conditionDim: dim, config: configuration.flow, ming3: false)
        _stopHead.wrappedValue = Linear(dim, 2)
        _speakerHead.wrappedValue = Linear(192, dim)
    }
    private var dtype: DType {
        (languageModel.model.embedding as? QuantizedEmbedding)?.scales.dtype ?? languageModel.model.embedding.weight.dtype
    }
    func encode(_ text: String) throws -> [Int] {
        guard let tokenizer else { throw AudioGenerationError.modelNotInitialized("Ming tokenizer is not loaded") }
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }
    func tokenID(_ text: String) throws -> Int {
        guard let value = tokenizer?.convertTokenToId(text) else {
            throw AudioGenerationError.invalidInput("Ming tokenizer is missing \(text)")
        }
        return value
    }
    func prepareInput(prompt: String, text: String, instruction: String?, promptLatent: MLXArray?,
                      promptText: String?, speaker: MLXArray?) throws -> (MLXArray, MLXArray) {
        var speakerTokens: [Int] = []
        if speaker != nil {
            speakerTokens = try encode("  speaker_1:") + encode("<spk>")
                + encode("<audioPatch>") + encode("</spk>\n")
        }
        var referenceEmbedding: MLXArray?
        if let promptLatent, promptText != nil {
            referenceEmbedding = aggregator(promptLatent.reshaped(-1, configuration.flow.patchSize,
                configuration.audio.encoder.latentDim)).reshaped(1, -1, configuration.text.hiddenSize)
        }
        let music = ["Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "].allSatisfy(text.contains)
        var ids = try encode("<role>HUMAN</role>")
        ids += try encode(prompt) + speakerTokens + (music ? [] : encode(" Text input:\n"))
        if referenceEmbedding != nil { ids += try encode(promptText!) }
        ids += try encode(text) + encode("<role>ASSISTANT</role>")
        if let instruction { ids += try encode(instruction) + encode("<|endoftext|>") }
        ids += try encode("<audio>")
        if let referenceEmbedding { ids += Array(repeating: try tokenID("<audioPatch>"), count: referenceEmbedding.dim(1)) }
        let tokens = MLXArray(ids.map(Int32.init)).reshaped(1, -1)
        var embedded = languageModel.model.embedding(tokens)
        if let speaker {
            let start = try tokenID("<spk>")
            guard let pos = ids.firstIndex(of: start) else { throw AudioGenerationError.invalidInput("Missing Ming speaker slot") }
            embedded = concatenated([embedded[0..., ..<(pos + 1), 0...], speaker.asType(embedded.dtype).reshaped(1, 1, -1),
                                     embedded[0..., (pos + 2)..., 0...]], axis: 1)
        }
        if let referenceEmbedding {
            let start = ids.firstIndex(of: try tokenID("<audio>"))! + 1
            embedded = concatenated([embedded[0..., ..<start, 0...], referenceEmbedding,
                                     embedded[0..., (start + referenceEmbedding.dim(1))..., 0...]], axis: 1)
        }
        return (tokens, embedded)
    }
    func padPrompt(_ waveform: MLXArray) -> MLXArray {
        let alignment = Int((1 / 12.5) * Double(configuration.flow.patchSize) * Double(sampleRate))
        let count = waveform.dim(1), amount = (alignment - count % alignment) % alignment
        guard amount > 0 else { return waveform }
        let zeros = MLXArray.zeros([waveform.dim(0), amount], dtype: waveform.dtype)
        return concatenated([waveform, zeros], axis: 1)
    }
    private func reference(_ ref: MLXArray?) throws -> MLXArray? {
        guard let ref else { return nil }
        let mono: MLXArray
        if ref.ndim == 1 { mono = ref }
        else if ref.ndim == 2, ref.dim(0) == 1 { mono = ref[0] }
        else if ref.ndim == 2, ref.dim(1) == 1 { mono = ref[0..., 0] }
        else { throw AudioGenerationError.invalidInput("Ming reference audio must be mono at \(sampleRate) Hz") }
        guard mono.size >= Int(Double(sampleRate) * 0.045), all(isFinite(mono)).item(Bool.self) else {
            throw AudioGenerationError.invalidInput("Ming reference audio must contain finite samples and last at least 45 ms")
        }
        return mono.asType(.float32)
    }
    func projectSpeaker(_ reference: MLXArray?, options: MingOmniSpeechOptions) throws -> MLXArray? {
        if options.zeroSpeakerEmbedding { return MLXArray.zeros([configuration.text.hiddenSize], dtype: dtype) }
        var embedding: MLXArray?
        if let supplied = options.speakerEmbedding {
            guard supplied.count == 192, supplied.allSatisfy(\.isFinite) else {
                throw AudioGenerationError.invalidInput("Ming speakerEmbedding must contain 192 finite values")
            }
            embedding = MLXArray(supplied).reshaped(1, 192)
        } else if options.automaticSpeakerEmbedding, let reference {
            guard let speakerEncoder else {
                throw AudioGenerationError.modelNotInitialized("CAMPPlus speaker weights are missing. Load with fromPretrained or supply speakerWeightsURL when loading a local model.")
            }
            let resampled = try resampleAudio(reference.asArray(Float.self), from: sampleRate, to: 16000)
            embedding = speakerEncoder(MingCAMPPlus.features(MLXArray(resampled))[.newAxis])
        }
        return embedding.map { speakerHead($0.asType(.float32))[0] }
    }
    private func run(text: String, voice: String?, refAudio: MLXArray?, refText: String?, language: String?,
                     parameters: GenerateParameters, emit: ((MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AudioGenerationError.invalidInput("Ming text is empty") }
        let controls = options
        guard controls.flowSteps > 0, controls.cfgScale >= 0, controls.sigma >= 0,
              parameters.temperature >= 0, controls.cfgScale.isFinite, controls.sigma.isFinite, parameters.temperature.isFinite else {
            throw AudioGenerationError.invalidInput("Ming flow settings must be finite and nonnegative; flowSteps must be positive")
        }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let waveform = try reference(refAudio)
        let speaker = try projectSpeaker(waveform, options: controls)
        let promptLatent = waveform.flatMap { wav in refText == nil ? nil : audio.encode(padPrompt(wav[.newAxis])) }
        let prompt = controls.prompt ?? "Please generate speech based on the following description.\n"
        let fragments = [text]
        var parts: [MLXArray] = []
        for fragment in fragments {
            try Task.checkCancellation()
            var (_, embedded) = try prepareInput(prompt: prompt, text: fragment, instruction: controls.instruction,
                promptLatent: promptLatent, promptText: refText, speaker: speaker)
            let cache: [KVCache] = (0..<configuration.text.numHiddenLayers).map { _ in KVCacheSimple() }
            let budget = parameters.maxTokens ?? 200
            guard budget > 0 else { throw AudioGenerationError.invalidInput("Ming generation has no positions left; shorten the prompt/reference or increase maxTokens") }
            var history = MLXArray.zeros([1, configuration.flow.historyPatchSize, configuration.audio.encoder.latentDim], dtype: dtype)
            if let promptLatent {
                let length = min(promptLatent.dim(1), history.dim(1))
                history = concatenated([history[0..., ..<(history.dim(1) - length), 0...],
                    promptLatent[0..., (promptLatent.dim(1) - length)..., 0...].asType(dtype)], axis: 1)
            }
            var chunks: [MLXArray] = []
            let decoderState = MingAudioDecoderState(configuration: configuration.audio)
            for step in 0..<budget {
                try Task.checkCancellation()
                let hidden = languageModel.model(embedded, cache: cache)
                let condition = hidden[0..., (hidden.dim(1) - 1)..., 0...]
                let sampled = flow.sample(condition, history: history, patchSize: configuration.flow.patchSize,
                    steps: controls.flowSteps, cfg: controls.cfgScale, sigma: controls.sigma, temperature: parameters.temperature)
                let stop = softmax(stopHead(condition), axis: -1)[0, 0, 1]
                eval(sampled, stop)
                let last = (step > 3 && stop.item(Float.self) > 0.5) || step == budget - 1
                let chunk = audio.decode(sampled, state: decoderState, lastChunk: last)[0]
                eval(chunk)
                if chunk.size > 0 {
                    guard all(isFinite(chunk)).item(Bool.self) else {
                        throw AudioGenerationError.audioDecodingFailed("Ming Omni produced non-finite audio")
                    }
                    if let emit { try emit(MLXArray(chunk.asArray(Float.self))) }
                    else { chunks.append(chunk) }
                }
                if last { break }
                embedded = aggregator(sampled)
                history = concatenated([history[0..., configuration.flow.patchSize..., 0...], sampled], axis: 1)
            }
            if emit == nil {
                guard !chunks.isEmpty else { throw AudioGenerationError.audioDecodingFailed("Ming Omni produced empty audio") }
                parts.append(chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0))
            } else { parts.append(MLXArray.zeros([0])) }
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
    }
    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try (run(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, parameters: generationParameters)).materialized()
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    _ = try self.run(text: text, voice: voice, refAudio: reference, refText: refText,
                        language: language, parameters: generationParameters) { continuation.yield(.audio($0)) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
    public static func fromModelDirectory(_ directory: URL, speakerWeightsURL: URL? = nil) async throws -> MingOmniModel {
        let config = try JSONDecoder().decode(MingOmniConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = try MingOmniModel(config)
        var weights: [String: MLXArray] = [:]
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            struct Index: Decodable { let weight_map: [String: String] }
            let names = Set(try JSONDecoder().decode(Index.self, from: Data(contentsOf: index)).weight_map.values).sorted()
            guard names.allSatisfy({ !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors") }) else {
                throw AudioGenerationError.invalidInput("Invalid Ming Omni weight index")
            }
            files = names.map { directory.appendingPathComponent($0) }
        } else {
            files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix("campplus") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        guard !files.isEmpty else { throw AudioGenerationError.modelNotInitialized("No Ming Omni safetensors found") }
        for file in files { weights.merge(try loadArrays(url: file)) { _, new in new } }
        weights = try model.sanitize(weights: weights)
        if config.text.tieWordEmbeddings {
            for suffix in ["weight", "scales", "biases"] { weights["model.lm_head." + suffix] = nil }
        }
        weights = weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        if let q = config.quantization {
            quantize(model: model) { path, _ in weights[path + ".scales"] == nil ? nil : q.asTuple }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        // BailingTokenizer publishes a complete standard BPE tokenizer.json.
        let tokenizerData = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [String: Any]
        guard (tokenizerData?["model"] as? [String: Any])?["type"] as? String == "BPE" else {
            throw AudioGenerationError.invalidInput("Ming Omni requires the published BPE tokenizer.json")
        }
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory, strict: false)
        for token in ["<audio>", "<audioPatch>", "<spk>", "</spk>", "<|endoftext|>"] {
            guard try model.encode(token) == [model.tokenID(token)] else {
                throw AudioGenerationError.invalidInput("Ming Omni tokenizer is missing the single-token control \(token)")
            }
        }
        let bundled = directory.appendingPathComponent("campplus.safetensors")
        if let speaker = speakerWeightsURL ?? (FileManager.default.fileExists(atPath: bundled.path) ? bundled : nil) {
            model.speakerEncoder = try MingCAMPPlus.fromWeights(loadArrays(url: speaker))
        }
        model.train(false); eval(model)
        return model
    }
    public static func fromPretrained(_ source: String, cache: HubCache = .default) async throws -> MingOmniModel {
        guard let repo = Repo.ID(rawValue: source) else { throw TTSModelError.invalidRepositoryID(source) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }
}
