import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore
import Tokenizers

/// Native Higgs-Audio v3 transcription. Model instances are single-caller inference objects.
public final class HiggsAudio3Model: Module, STTGenerationModel {
    public static let defaultPrompt = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
    public let config: HiggsAudio3Config
    @ModuleInfo(key: "audio_tower") var audioTower: HiggsAudio3Encoder
    @ModuleInfo(key: "audio_encoder_proj") var projector: HiggsAudio3Projector
    @ModuleInfo(key: "model") var textModel: Qwen3ASRTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    public var tokenizer: (any Tokenizers.Tokenizer)?
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(maxTokens: 1024, topP: 1, chunkDuration: config.chunkSizeSeconds,
            repetitionContextSize: 100)
    }

    public init(_ config: HiggsAudio3Config) {
        self.config = config
        _audioTower.wrappedValue = HiggsAudio3Encoder(config.audioEncoderConfig)
        _projector.wrappedValue = HiggsAudio3Projector(config)
        _textModel.wrappedValue = Qwen3ASRTextModel(config.textConfig)
        // Higgs has a real independent text head even though the nested HF config says tied.
        _lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
    }

    public func callAsFunction(inputIds: MLXArray? = nil, inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil) -> MLXArray {
        lmHead(textModel(inputIds: inputIds, inputsEmbeds: inputsEmbeds, cache: cache))
    }

    public func audioFeatures(_ mel: MLXArray) -> MLXArray {
        // Quantized embeddings store uint32 weights; use the lookup output dtype.
        projector(audioTower(mel)).asType(textModel.embedTokens(MLXArray([Int32(0)])).dtype)
    }

    static func chunkRanges(sampleCount: Int, maxSamples: Int) -> [Range<Int>] {
        guard sampleCount > 0, maxSamples > 0 else { return [] }
        return stride(from: 0, to: sampleCount, by: maxSamples).map {
            $0..<min($0 + maxSamples, sampleCount)
        }
    }

    public func inputEmbeddings(audio: MLXArray, prompt: String = HiggsAudio3Model.defaultPrompt,
        chunkDuration: Float? = nil) throws -> (ids: [Int], embeddings: MLXArray) {
        guard let tokenizer else { throw STTError.modelNotInitialized("Higgs tokenizer not loaded") }
        guard audio.ndim == 1, audio.size > 0 else { throw STTError.invalidInput("Expected nonempty mono audio") }
        let seconds = chunkDuration ?? config.chunkSizeSeconds
        guard seconds.isFinite, seconds > 0 else { throw STTError.invalidInput("chunkDuration must be positive and finite") }
        // Check before Float-to-Int conversion, including arbitrarily large finite inputs.
        let maximumSeconds = Float(config.audioEncoderConfig.maxSourcePositions) * 320 / Float(sampleRate)
        guard seconds <= maximumSeconds else {
            throw STTError.invalidInput("Audio chunk exceeds the encoder positional context")
        }
        let maxSamples = max(1, Int(seconds * Float(sampleRate)))
        let ranges = Self.chunkRanges(sampleCount: audio.size, maxSamples: maxSamples)
        let longest = max(640, ranges.map(\.count).max() ?? 640)
        var encoded: [MLXArray] = []
        for range in ranges {
            var chunk = audio[range].asType(.float32)
            if chunk.size < longest { chunk = padded(chunk, widths: [.init((0, longest - chunk.size))]) }
            let mel = HiggsAudio3Frontend.features(chunk, numMelBins: config.audioEncoderConfig.numMelBins,
                sampleRate: sampleRate).asType(audioTower.conv1.weight.dtype)
            encoded.append(audioFeatures(mel))
        }
        let enc: (String) -> [Int] = { tokenizer.encode(text: $0, addSpecialTokens: false) }
        let prefix = enc("<|im_start|>user\n") + enc(prompt) + enc("<|audio_bos|>")
        let suffix = enc("<|audio_eos|>") + enc("<|im_end|>\n") + enc("<|im_start|>assistant\n")
        let ids = prefix + [Int](repeating: config.audioInTokenIndex, count: ranges.count) + suffix
        let embeds = textModel.embedTokens(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0))
        let pieces = [embeds[0...,0..<prefix.count,0...]] + encoded
            + [embeds[0...,(prefix.count + ranges.count)...,0...]]
        let result = concatenated(pieces, axis: 1)
        eval(result)
        return (ids,result)
    }

    public static func parseOutput(_ raw: String) -> String {
        var text = raw.replacingOccurrences(of: "(?s)<think>.*?</think>", with: "", options: .regularExpression)
        // Match Python's treatment of an unfinished thinking opener.
        if let range = text.range(of: "<think>") { text = String(text[range.upperBound...]) }
        text = text.replacingOccurrences(of: "<\\|.*?\\|>", with: "", options: .regularExpression)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Generate a transcription with an optional prompt, sampling controls and token callback.
    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String = HiggsAudio3Model.defaultPrompt, prefillStepSize: Int = 2048,
        minP: Float = 0, onToken: ((String) -> Void)? = nil) -> STTOutput {
        do {
            return try generateImpl(audio: audio, parameters: generationParameters ?? defaultGenerationParameters,
                prompt: prompt, prefillStepSize: prefillStepSize, minP: minP, onToken: onToken)
        } catch { preconditionFailure("\(error)") }
    }

    private func generateImpl(audio: MLXArray, parameters p: STTGenerateParameters,
        prompt: String, prefillStepSize: Int, minP: Float,
        onToken: ((String) -> Void)?) throws -> STTOutput {
        guard p.maxTokens >= 0, prefillStepSize > 0 else { throw STTError.invalidInput("Invalid token/prefill limits") }
        guard p.temperature >= 0, p.temperature.isFinite, p.repetitionPenalty > 0,
              p.repetitionPenalty.isFinite else { throw STTError.invalidInput("Invalid sampling parameters") }
        guard let tokenizer else { throw STTError.modelNotInitialized("Higgs tokenizer not loaded") }
        let start = Date()
        let inputs = try inputEmbeddings(audio: audio, prompt: prompt, chunkDuration: min(p.chunkDuration, config.chunkSizeSeconds))
        let promptTokens = inputs.embeddings.dim(1)
        let cacheConfiguration: KVCacheConfiguration?
        if let bits = p.kvBits {
            cacheConfiguration = KVCacheConfiguration(strategy: .affine(try AffineKVCacheConfiguration(
                bits: bits, groupSize: p.kvGroupSize, compressionStart: p.quantizedKVStart)))
        } else { cacheConfiguration = nil }
        let cache = try (0..<config.textConfig.numHiddenLayers).map { _ in
            try makeAttentionKVCache(configuration: cacheConfiguration)
        }
        var offset = 0
        while promptTokens - offset > prefillStepSize {
            let h = inputs.embeddings[0...,offset..<(offset + prefillStepSize),0...]
            eval(textModel(inputsEmbeds: h, cache: cache))
            offset += prefillStepSize
        }
        let hidden = textModel(inputsEmbeds: inputs.embeddings[0...,offset...,0...], cache: cache)
        var logits = lmHead(hidden[0..., -1, 0...].expandedDimensions(axis: 1))
        eval(logits)
        let prefillTime = Date().timeIntervalSince(start)
        let generationStart = Date()
        let sampler: any LogitSampler = p.temperature == 0 ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK, minP: minP)
        var tokens: [Int] = []
        for _ in 0..<p.maxTokens {
            if Task.isCancelled { break }
            var scores = logits[0,-1,0...]
            if p.repetitionPenalty != 1, !tokens.isEmpty {
                let ids = MLXArray(Array(Set(tokens.suffix(max(1,p.repetitionContextSize)))).map(Int32.init))
                let recent = scores[ids]
                scores[ids] = MLX.where(recent .> 0, recent / p.repetitionPenalty, recent * p.repetitionPenalty)
            }
            let token = sampler.sample(logits: scores).item(Int.self)
            if token == 151645 || token == 151643 { break }
            tokens.append(token)
            onToken?(tokenizer.decode(tokens: [token]))
            logits = callAsFunction(inputIds: MLXArray([Int32(token)]).expandedDimensions(axis: 0), cache: cache)
            eval(logits)
        }
        let total = Date().timeIntervalSince(start)
        let generationTime = Date().timeIntervalSince(generationStart)
        let text = Self.parseOutput(tokenizer.decode(tokens: tokens, skipSpecialTokens: false))
        return STTOutput(text: text, segments: [["start": 0.0, "end": Double(audio.size) / Double(sampleRate), "text": text]],
            promptTokens: promptTokens, generationTokens: tokens.count, totalTokens: promptTokens + tokens.count,
            promptTps: Double(promptTokens) / max(prefillTime, 1e-9),
            generationTps: Double(tokens.count) / max(generationTime, 1e-9), totalTime: total,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        generate(audio: audio, generationParameters: generationParameters, prompt: Self.defaultPrompt)
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(audio: audio, generationParameters: generationParameters, prompt: Self.defaultPrompt)
    }

    /// Stream generated text from a complete recording using optional model-specific controls.
    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String = HiggsAudio3Model.defaultPrompt, prefillStepSize: Int = 2048,
        minP: Float = 0) -> AsyncThrowingStream<STTGeneration, Error> {
        let parameters = generationParameters ?? defaultGenerationParameters
        let modelBox = UncheckedSendableBox(self), audioBox = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    let result = try modelBox.value.generateImpl(audio: audioBox.value, parameters: parameters,
                        prompt: prompt, prefillStepSize: prefillStepSize, minP: minP) {
                            continuation.yield(.token($0))
                        }
                    try Task.checkCancellation()
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Accept both HF tensors and already prepared MLX layouts, without converting checkpoints.
    public static func sanitize(weights: [String:MLXArray]) -> [String:MLXArray] {
        var result: [String:MLXArray] = [:]
        for (raw, original) in weights {
            if raw.hasPrefix("audio_codebook_embeddings") || raw.hasPrefix("audio_decoder_proj.audio_lm_head") { continue }
            var key = raw, value = original
            if raw == "audio_decoder_proj.text_lm_head.weight" { key = "lm_head.weight" }
            else if raw == "embed_tokens.weight" || raw == "norm.weight" || raw.hasPrefix("layers.") { key = "model." + raw }
            if (key.hasPrefix("audio_tower.conv") || key == "audio_encoder_proj.temporal.weight"),
                key.hasSuffix(".weight"), value.ndim == 3, value.dim(1) != 3, value.dim(2) == 3 {
                value = value.transposed(0,2,1)
            }
            result[key] = value
        }
        return result
    }

    public static func fromPretrained(_ modelPath: String, cache: HubCache = .default) async throws -> HiggsAudio3Model {
        let expanded = (modelPath as NSString).expandingTildeInPath
        let dir: URL
        if FileManager.default.fileExists(atPath: expanded) {
            dir = URL(fileURLWithPath: expanded)
        } else {
            guard let repo = Repo.ID(rawValue: modelPath) else { throw STTModelError.invalidRepositoryID(modelPath) }
            dir = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors",
                hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"], cache: cache)
        }
        return try await fromModelDirectory(dir)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> HiggsAudio3Model {
        let config = try JSONDecoder().decode(HiggsAudio3Config.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard config.sampleRate == 16000, config.chunkSizeSeconds > 0,
              config.audioEncoderConfig.dModel > 0, config.audioEncoderConfig.encoderAttentionHeads > 0,
              config.audioEncoderConfig.dModel % config.audioEncoderConfig.encoderAttentionHeads == 0,
              [1,2].contains(config.projectorTemporalDownsample), config.textConfig.ropeScaling == nil else {
            throw STTError.invalidInput("Unsupported Higgs model configuration")
        }
        let model = HiggsAudio3Model(config)
        var weights: [String:MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw STTError.modelNotInitialized("No safetensors in model directory") }
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        let sanitized = sanitize(weights: weights)
        if let quantization = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil ? quantization.quantization(layer: path)?.asTuple : nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        eval(model)
        return model
    }
}
