import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Tokenizers

public struct MingSpeechOptions: Sendable {
    public var instruction: String?
    public var prompt: String?
    public var cfgScale: Float = 2
    public var sigma: Float = 0.25
    public var flowSteps: Int = 10
    public var rawText: Bool = false
    public var trimSilence: Bool = true
    public var fragmentLength: Int = 24
    public var zeroSpeakerEmbedding: Bool = false
    /// An optional precomputed 192-value CAMPPlus embedding, before spk_head projection.
    public var speakerEmbedding: [Float]?
    public init() {}
}

/// Qwen2.5 dense Ming speech synthesis with native latent and speaker conditioning.
public final class MingDenseModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: MingDenseConfiguration
    @ModuleInfo(key: "model") var languageModel: Qwen2Model
    public let audio: MingAudioVAE
    @ModuleInfo(key: "linear_proj_audio") public var aggregator: MingLatentAggregator
    @ModuleInfo(key: "flowloss") public var flow: MingFlowLoss
    @ModuleInfo(key: "stop_head") var stopHead: Linear
    @ModuleInfo(key: "spk_head") var speakerHead: Linear
    private var tokenizer: (any Tokenizer)?
    private var speakerEncoder: MingCAMPPlus?
    public var options = MingSpeechOptions()
    public var sampleRate: Int { configuration.audio.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: configuration.isMing3 ? 1000 : 200, temperature: 0)
    }
    public init(_ configuration: MingDenseConfiguration) {
        self.configuration = configuration
        let dim = configuration.text.hiddenSize, latent = configuration.audio.encoder.latentDim
        _languageModel.wrappedValue = Qwen2Model(configuration.text)
        audio = MingAudioVAE(configuration.audio)
        _aggregator.wrappedValue = MingLatentAggregator(latentDim: latent, outputDim: dim, config: configuration.aggregator)
        _flow.wrappedValue = MingFlowLoss(latentDim: latent, conditionDim: dim, config: configuration.flow, ming3: configuration.isMing3)
        _stopHead.wrappedValue = Linear(dim, 2)
        _speakerHead.wrappedValue = Linear(192, dim)
    }
    private var dtype: DType {
        (languageModel.parameters().flattened().first { $0.0 == "model.embed_tokens.scales" }
            ?? languageModel.parameters().flattened().first { $0.0 == "model.embed_tokens.weight" })!.1.dtype
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
            speakerTokens = try encode("  speaker_1:") + encode("<|vision_start|>")
                + encode("<|vision_pad|>") + encode("<|vision_end|>\n")
        }
        var referenceEmbedding: MLXArray?
        if let promptLatent, promptText != nil {
            referenceEmbedding = aggregator(promptLatent.asType(dtype).reshaped(-1, configuration.flow.patchSize,
                configuration.audio.encoder.latentDim)).reshaped(1, -1, configuration.text.hiddenSize)
        }
        let music = ["Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "].allSatisfy(text.contains)
        var ids = try encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n") + encode("<|im_start|>user\n")
        ids += try encode(prompt) + speakerTokens + (music ? [] : encode(" Text input:\n"))
        if referenceEmbedding != nil { ids += try encode(promptText!) }
        ids += try encode(text) + encode("<|im_end|>\n") + encode("<|im_start|>assistant\n")
        if let instruction { ids += try encode(instruction) + encode(configuration.isMing3 ? "<|im_end|>" : "<|endoftext|>") }
        ids += try encode("<audio>")
        if let referenceEmbedding { ids += Array(repeating: try tokenID("<audioPatch>"), count: referenceEmbedding.dim(1)) }
        let tokens = MLXArray(ids.map(Int32.init)).reshaped(1, -1)
        var embedded = languageModel.embedTokens(tokens)
        if let speaker {
            let start = try tokenID("<|vision_start|>")
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
        return concatenated(configuration.isMing3 ? [zeros, waveform] : [waveform, zeros], axis: 1)
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
    private func projectSpeaker(_ reference: MLXArray?, options: MingSpeechOptions) throws -> MLXArray? {
        if options.zeroSpeakerEmbedding { return MLXArray.zeros([configuration.text.hiddenSize], dtype: dtype) }
        var embedding: MLXArray?
        if let supplied = options.speakerEmbedding {
            guard supplied.count == 192, supplied.allSatisfy(\.isFinite) else {
                throw AudioGenerationError.invalidInput("Ming speakerEmbedding must contain 192 finite values")
            }
            embedding = MLXArray(supplied).reshaped(1, 192)
        } else if let reference {
            guard let speakerEncoder else {
                throw AudioGenerationError.modelNotInitialized("CAMPPlus speaker weights are missing. Load with fromPretrained or supply speakerWeightsURL when loading a local model.")
            }
            let resampled = try resampleAudio(reference.asArray(Float.self), from: sampleRate, to: 16000)
            embedding = speakerEncoder(MingCAMPPlus.features(MLXArray(resampled))[.newAxis])
        }
        return embedding.map { speakerHead($0.asType(.float32))[0] }
    }
    private func trim(_ audio: MLXArray) -> MLXArray {
        let frame = sampleRate / 10, tail = Int(Float(sampleRate) * 0.3)
        guard audio.size >= frame else { return audio }
        let count = audio.size / frame
        let means = abs(audio[..<(count * frame)].reshaped(count, frame)).mean(axis: 1).asArray(Float.self)
        guard let last = means.lastIndex(where: { $0 > 1e-3 }) else { return audio[..<min(tail, count * frame)] }
        return audio[..<min(count * frame, (last + 1) * frame + tail)]
    }
    private func run(text: String, voice: String?, refAudio: MLXArray?, refText: String?, language: String?,
                     parameters: GenerateParameters, emit: ((MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AudioGenerationError.invalidInput("Ming text is empty") }
        var controls = options
        if configuration.isMing3, let voice, refAudio == nil, MingText.isVoice(voice) {
            guard controls.instruction == nil else { throw AudioGenerationError.invalidInput("Pass either a Ming voice or an instruction") }
            controls.instruction = MingText.instruction(voice: voice); controls.zeroSpeakerEmbedding = true
        }
        guard controls.flowSteps > 0, controls.fragmentLength > 0, controls.cfgScale >= 0, controls.sigma >= 0,
              parameters.temperature >= 0, controls.cfgScale.isFinite, controls.sigma.isFinite, parameters.temperature.isFinite else {
            throw AudioGenerationError.invalidInput("Ming flow settings must be finite and nonnegative; flowSteps and fragmentLength must be positive")
        }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let waveform = try reference(refAudio)
        let speaker = try projectSpeaker(waveform, options: controls)
        let promptLatent = waveform.flatMap { wav in refText == nil ? nil : audio.encode(padPrompt(wav[.newAxis])) }
        var prompt = controls.prompt ?? "Please generate speech based on the following description.\n"
        if controls.prompt == nil, configuration.isMing3, let target = MingText.targetLanguage(language) { prompt += "Target language: \(target).\n" }
        let fragments: [String]
        if !configuration.isMing3 || controls.rawText { fragments = [text] }
        else if MingText.isStyle(controls.instruction) { fragments = [MingText.addPeriod(text.split(whereSeparator: \.isWhitespace).joined(separator: " "))] }
        else { fragments = MingText.fragments(MingText.addPeriod(text), maxLength: controls.fragmentLength) }
        var parts: [MLXArray] = []
        for fragment in fragments {
            try Task.checkCancellation()
            var (_, embedded) = try prepareInput(prompt: prompt, text: fragment, instruction: controls.instruction,
                promptLatent: promptLatent, promptText: refText, speaker: speaker)
            let cache: [KVCache] = (0..<configuration.text.hiddenLayers).map { _ in KVCacheSimple() }
            var budget = parameters.maxTokens ?? (configuration.isMing3 ? 1000 : 200)
            if configuration.isMing3 { budget = min(budget, (512 - embedded.dim(1)) / configuration.flow.patchSize) }
            guard budget > 0 else { throw AudioGenerationError.invalidInput("Ming generation has no positions left; shorten the prompt/reference or increase maxTokens") }
            var history = MLXArray.zeros([1, configuration.flow.historyPatchSize, configuration.audio.encoder.latentDim], dtype: dtype)
            if let promptLatent {
                let length = min(promptLatent.dim(1), history.dim(1))
                history = concatenated([history[0..., ..<(history.dim(1) - length), 0...],
                    promptLatent[0..., (promptLatent.dim(1) - length)..., 0...].asType(dtype)], axis: 1)
            }
            var patches: [MLXArray] = []
            for step in 0..<budget {
                try Task.checkCancellation()
                let hidden = languageModel.model(inputEmbeddings: embedded, cache: cache)
                let condition = hidden[0..., (hidden.dim(1) - 1)..., 0...]
                let sampled = flow.sample(condition, history: history, patchSize: configuration.flow.patchSize,
                    steps: controls.flowSteps, cfg: controls.cfgScale, sigma: controls.sigma, temperature: parameters.temperature)
                let stop = softmax(stopHead(condition), axis: -1)[0, 0, 1]
                eval(sampled, stop)
                patches.append(sampled)
                if step > 3 && stop.item(Float.self) > 0.5 { break }
                embedded = aggregator(sampled)
                history = concatenated([history[0..., configuration.flow.patchSize..., 0...], sampled], axis: 1)
            }
            var part = audio.decode(concatenated(patches, axis: 1))[0]
            if configuration.isMing3 && controls.trimSilence { part = trim(part) }
            eval(part)
            guard part.size > 0, all(isFinite(part)).item(Bool.self) else {
                throw AudioGenerationError.audioDecodingFailed("Ming produced empty or non-finite audio")
            }
            if let emit {
                // Disconnect the stream chunk from the decoder graph before transferring it.
                let chunk = MLXArray(part.asArray(Float.self))
                try emit(chunk)
                parts.append(MLXArray.zeros([0]))
            } else { parts.append(part) }
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
    }
    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try run(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, parameters: generationParameters).materialized()
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
    public static func fromModelDirectory(_ directory: URL, speakerWeightsURL: URL? = nil) async throws -> MingDenseModel {
        let config = try JSONDecoder().decode(MingDenseConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = MingDenseModel(config)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" && !file.lastPathComponent.hasPrefix("campplus") {
            weights.merge(try loadArrays(url: file)) { _, new in new }
        }
        weights = mingSanitizeSpeechWeights(weights)
        if config.text.tieWordEmbeddings { weights["model.lm_head.weight"] = nil }
        weights = weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        if let q = config.quantization {
            quantize(model: model) { path, _ in weights[path + ".scales"] == nil ? nil : q.asTuple }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let bundled = directory.appendingPathComponent("campplus.safetensors")
        if let speaker = speakerWeightsURL ?? (FileManager.default.fileExists(atPath: bundled.path) ? bundled : nil) {
            model.speakerEncoder = try MingCAMPPlus.fromWeights(loadArrays(url: speaker))
        }
        model.train(false); eval(model)
        return model
    }
    public static func fromPretrained(_ source: String, cache: HubCache = .default) async throws -> MingDenseModel {
        guard let repo = Repo.ID(rawValue: source) else { throw TTSModelError.invalidRepositoryID(source) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors", cache: cache)
        let bundled = directory.appendingPathComponent("campplus.safetensors")
        var speaker: URL?
        if !FileManager.default.fileExists(atPath: bundled.path) {
            // Byte-identical source speaker encoder used by the legacy dense Omni model.
            speaker = try await HubClient(cache: cache).downloadFile(
                at: "campplus.safetensors", from: Repo.ID(rawValue: "mlx-community/Step-Audio-2-token2wav")!,
                revision: "e21026b660069e6d1b9ac5c9a51e08ed0201f30e")
        }
        return try await fromModelDirectory(directory, speakerWeightsURL: speaker)
    }
}
