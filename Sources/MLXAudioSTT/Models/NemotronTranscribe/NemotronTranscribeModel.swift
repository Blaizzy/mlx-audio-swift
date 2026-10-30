import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

public struct NemotronTranscribeOptions: Sendable {
    public enum Task: String, Sendable {
        case transcribe, translate
        case speakerAttribution = "saa"
        case rich
        case questionAnswering = "qa"
    }
    public var task: Task?
    public var prompt: String?
    public var systemPrompt: String?
    public var hotwords: [String]
    public var enableThinking: Bool
    public var prefillStepSize: Int
    public var minP: Float
    public var wordTimestamps: Bool
    public init(
        task: Task? = nil, prompt: String? = nil, systemPrompt: String? = nil,
        hotwords: [String] = [], enableThinking: Bool = false,
        prefillStepSize: Int = 512, minP: Float = 0, wordTimestamps: Bool = false
    ) {
        self.task = task
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.hotwords = hotwords
        self.enableThinking = enableThinking
        self.prefillStepSize = prefillStepSize
        self.minP = minP
        self.wordTimestamps = wordTimestamps
    }
}

public final class NemotronTranscribeModel: Module, STTGenerationModel {
    public let config: NemotronTranscribeConfiguration
    public let llm: NemotronHLanguageModel
    let perception: NemotronPerception
    @ModuleInfo(key: "ctc_timestamp") var ctcTimestamp: NemotronCTCTimestamp?
    var alignmentTokenizer: UncheckedSendableBox<NemotronAlignmentTokenizer>?
    var alignmentSpecialTokens: [String] = []
    public var supportsWordTimestamps: Bool { ctcTimestamp != nil && alignmentTokenizer != nil }
    var _tokenizer: UncheckedSendableBox<any Tokenizer>?
    var tokenizer: (any Tokenizer)? { _tokenizer?.value }
    public var sampleRate: Int { config.preprocessor.sampleRate }
    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 1024, temperature: 0, topP: 1,
            repetitionContextSize: 64)
    }
    public init(_ config: NemotronTranscribeConfiguration) {
        self.config = config
        llm = NemotronHLanguageModel(config.text)
        perception = NemotronPerception(config)
        _ctcTimestamp.wrappedValue = config.timestamps.map(NemotronCTCTimestamp.init)
    }

    static let asrPrompt =
        "Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, numbers, dates, measurements, acronyms, and technical terms as clearly as possible. Keep the same language as spoken and do not translate."
    static let speakerPrompt =
        "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> <spk:4> <spk:5> <spk:6> <spk:7> on the transcription."
    static let richPrompt =
        "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Use <ev:*> and <sty:*> special tokens to recognize all non-speech verbatims and speaking style."

    static func resolvePrompt(
        _ options: NemotronTranscribeOptions,
        parameters: STTGenerateParameters, audioToken: String
    ) -> String? {
        let task =
            options.task
            ?? (parameters.task == "asr"
                ? .transcribe
                : NemotronTranscribeOptions.Task(rawValue: parameters.task))
        guard let task else { return nil }
        var prompt: String
        if let custom = options.prompt {
            prompt = custom
        } else {
            switch task {
            case .transcribe: prompt = asrPrompt
            case .translate:
                guard let language = parameters.language, !language.isEmpty else { return nil }
                prompt = "Translate the speech in this audio into \(language)."
            case .speakerAttribution: prompt = speakerPrompt
            case .rich: prompt = richPrompt
            case .questionAnswering: prompt = audioToken
            }
        }
        let words = options.hotwords.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !words.isEmpty { prompt += "\nContextual vocabulary: " + words.joined(separator: ", ") }
        let count = prompt.components(separatedBy: audioToken).count - 1
        guard count <= 1 else { return nil }
        return count == 0 ? prompt + " " + audioToken : prompt
    }

    static func turns(prompt: String, options: NemotronTranscribeOptions) -> [String] {
        let thinking = options.enableThinking ? "<think>\n" : "<think></think>"
        return [
            "<|im_start|>system\n\(options.systemPrompt ?? "")<|im_end|>\n",
            "<|im_start|>user\n\(prompt)<|im_end|>\n",
            "<|im_start|>assistant\n\(thinking)",
        ]
    }

    /// Native speaker turns, preserving event/style markup without inventing timestamps.
    public static func speakerSegments(_ text: String) -> [[String: Any]]? {
        let regex = try! NSRegularExpression(pattern: #"<spk:([0-7])>"#)
        let value = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: value.length))
        guard !matches.isEmpty else { return nil }
        return matches.enumerated().map { index, match in
            let first = NSMaxRange(match.range)
            let end = index + 1 < matches.count ? matches[index + 1].range.location : value.length
            return [
                "speaker": Int(value.substring(with: match.range(at: 1)))!,
                "text": value.substring(with: NSRange(location: first, length: end - first))
                    .trimmingCharacters(in: .whitespacesAndNewlines),
            ]
        }
    }

    func prepare(_ audio: MLXArray, prompt: String, options: NemotronTranscribeOptions)
        -> (
            tokens: [Int], embeddings: MLXArray,
            alignment: (states: MLXArray, probabilities: MLXArray)?
        )?
    {
        guard let tokenizer else { return nil }
        let ids = Self.turns(prompt: prompt, options: options)
            .flatMap { tokenizer.encode(text: $0, addSpecialTokens: false) }
        let audioIDs = tokenizer.encode(text: config.audioToken, addSpecialTokens: false)
        guard audioIDs.count == 1 else { return nil }
        let locations = ids.indices.filter { ids[$0] == audioIDs[0] }
        guard locations.count == 1, let index = locations.first else { return nil }
        let acoustic: MLXArray
        let alignment: (states: MLXArray, probabilities: MLXArray)?
        if options.wordTimestamps && supportsWordTimestamps {
            let output = perception.withAlignment(audio)
            acoustic = output.embeddings
            alignment = (output.states, output.probabilities)
        } else {
            acoustic = perception(audio)
            alignment = nil
        }
        let count = acoustic.dim(1)
        let expanded =
            Array(ids[..<index]) + Array(repeating: config.padTokenID, count: count)
            + Array(ids[(index + 1)...])
        let tokens = MLXArray(expanded.map(Int32.init)).expandedDimensions(axis: 0)
        let text = llm.backbone.embeddings(tokens)
        let embeddings = concatenated(
            [
                text[0..., ..<index, 0...], acoustic.asType(text.dtype),
                text[0..., (index + count)..., 0...],
            ], axis: 1)
        eval(embeddings)
        return (expanded, embeddings, alignment)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generate(audio: audio, generationParameters: generationParameters, options: .init())
    }
    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        options: NemotronTranscribeOptions
    ) -> STTOutput {
        generateImpl(
            audio, parameters: generationParameters ?? defaultGenerationParameters,
            options: options, emit: nil)
    }

    private func generateImpl(
        _ audio: MLXArray, parameters p: STTGenerateParameters,
        options: NemotronTranscribeOptions, emit: ((String) -> Void)?
    ) -> STTOutput {
        let started = Date()
        let waveform = audio.asType(.float32)
        guard waveform.ndim == 1, waveform.size >= config.preprocessor.hopLength,
            MLX.all(isFinite(waveform)).item(Bool.self), p.maxTokens >= 0,
            p.temperature.isFinite, p.temperature >= 0, p.repetitionPenalty > 0,
            options.prefillStepSize > 0, (0 ... 1).contains(options.minP),
            let tokenizer,
            let prompt = Self.resolvePrompt(options, parameters: p, audioToken: config.audioToken),
            let prepared = prepare(waveform, prompt: prompt, options: options)
        else {
            return STTOutput(text: "", totalTime: Date().timeIntervalSince(started))
        }
        let cache = llm.makeCache()
        let length = prepared.embeddings.dim(1)
        var offset = 0
        while length - offset > options.prefillStepSize && !Task.isCancelled {
            let end = offset + options.prefillStepSize
            let hidden = llm.backbone(
                embeddings: prepared.embeddings[0..., offset ..< end, 0...], cache: cache)
            eval([hidden] + cache.state)
            offset = end
        }
        let hidden = llm.backbone(
            embeddings: prepared.embeddings[0..., offset..., 0...], cache: cache)
        var logits = llm.head(hidden[0..., (-1)..., 0...])
        eval([logits] + cache.state)
        let prefillTime = Date().timeIntervalSince(started)
        let decodeStarted = Date()
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler()
            : TopPSampler(
                temperature: p.temperature, topP: p.topP, topK: p.topK, minP: options.minP)
        var generated: [Int] = []
        var emitted = ""
        let stops = Set(config.eosTokenIDs)
        for iteration in 0 ..< p.maxTokens {
            if Task.isCancelled { break }
            var scores = logits[0, -1, 0...]
            if p.repetitionPenalty != 1 {
                let history = prepared.tokens + generated
                let context =
                    p.repetitionContextSize > 0
                    ? Array(history.suffix(p.repetitionContextSize)) : history
                let indices = MLXArray(Array(Set(context)).map(Int32.init))
                let values = scores[indices]
                scores[indices] = MLX.where(
                    values .< 0, values * p.repetitionPenalty, values / p.repetitionPenalty)
            }
            let token = sampler.sample(logits: scores).item(Int.self)
            if stops.contains(token) { break }
            generated.append(token)
            if emit != nil || p.verbose {
                let text = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
                if !text.contains("\u{FFFD}"), text.hasPrefix(emitted) {
                    let delta = String(text.dropFirst(emitted.count))
                    if !delta.isEmpty {
                        emit?(delta)
                        if p.verbose { print(delta, terminator: "") }
                    }
                    emitted = text
                }
            }
            if iteration + 1 == p.maxTokens { break }
            logits = llm(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
            eval([logits] + cache.state)
        }
        let raw = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
        if raw.hasPrefix(emitted) {
            let tail = String(raw.dropFirst(emitted.count))
            if !tail.isEmpty {
                emit?(tail)
                if p.verbose { print(tail, terminator: "") }
            }
        }
        if p.verbose { print() }
        var segments = Self.speakerSegments(raw)
        let task = options.task ?? NemotronTranscribeOptions.Task(rawValue: p.task)
        if options.wordTimestamps, task != .translate, task != .questionAnswering,
            let alignment = prepared.alignment, let head = ctcTimestamp,
            let alignmentTokenizer = alignmentTokenizer?.value
        {
            var transcript = raw
            for token in alignmentSpecialTokens where !NemotronWordAlignment.isKeptTag(token) {
                transcript = transcript.replacingOccurrences(of: token, with: " ")
            }
            segments =
                (try? NemotronWordAlignment.align(
                    text: transcript, tokenizer: alignmentTokenizer,
                    head: head, states: alignment.states,
                    nativeProbabilities: alignment.probabilities,
                    duration: Double(waveform.size) / Double(sampleRate), configuration: config,
                    language: p.language)) ?? segments
        }
        let decodeTime = Date().timeIntervalSince(decodeStarted)
        return STTOutput(
            text: raw.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: segments, promptTokens: prepared.tokens.count,
            generationTokens: generated.count, totalTokens: prepared.tokens.count + generated.count,
            promptTps: Double(prepared.tokens.count) / max(prefillTime, 0.001),
            generationTps: Double(generated.count) / max(decodeTime, 0.001),
            totalTime: Date().timeIntervalSince(started),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        generateStream(audio: audio, generationParameters: generationParameters, options: .init())
    }
    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        options: NemotronTranscribeOptions
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        let parameters = generationParameters ?? defaultGenerationParameters
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let output = model.value.generateImpl(
                        input.value, parameters: parameters, options: options
                    ) {
                        continuation.yield(.token($0))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.result(output))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public static func sanitize(
        weights: [String: MLXArray], includeTimestamps: Bool = false,
        timestampUpsamplingFactor: Int = 1, expertCount: Int? = nil
    ) -> [String: MLXArray] {
        var result = remap(
            weights: weights, includeTimestamps: includeTimestamps,
            timestampUpsamplingFactor: timestampUpsamplingFactor)
        packExperts(&result, expertCount: expertCount, evaluate: false)
        return result
    }

    private static func remap(
        weights: [String: MLXArray], includeTimestamps: Bool, timestampUpsamplingFactor: Int
    ) -> [String: MLXArray] {
        let sourceLayout = weights.keys.contains {
            $0.hasPrefix("llm.model.") || $0.contains(".preprocessor.featurizer.")
        }
        var output: [String: MLXArray] = [:]
        for (key, tensor) in weights {
            if key.hasPrefix("llm.mtp.") || (!includeTimestamps && key.hasPrefix("ctc_timestamp."))
                || key.hasSuffix("._extra_state")
                || key == "ctc_timestamp.tokenizer.model_proto"
                || key == "ctc_timestamp.decoder.upsample_factor"
            {
                continue
            }
            let key = key.hasPrefix("backbone.") || key.hasPrefix("lm_head.") ? "llm." + key : key
            var name = key.replacingOccurrences(
                of: "llm.model.embed_tokens.", with: "llm.backbone.embeddings."
            )
            .replacingOccurrences(of: "llm.model.norm.", with: "llm.backbone.norm_f.")
            .replacingOccurrences(of: "llm.model.layers.", with: "llm.backbone.layers.")
            .replacingOccurrences(of: "._fp32_params.", with: ".")
            .replacingOccurrences(of: ".ffn.net.0.", with: ".ffn.linear1.")
            .replacingOccurrences(of: ".ffn.net.3.", with: ".ffn.linear2.")
            .replacingOccurrences(of: ".preprocessor.featurizer.", with: ".preprocessor.")
            var value = tensor
            if name.hasSuffix(".experts.gate_and_up_projs") {
                name = name.replacingOccurrences(
                    of: ".experts.gate_and_up_projs", with: ".switch_mlp.fc1.weight")
                value = value.transposed(0, 2, 1)
            } else if name.hasSuffix(".experts.down_projs") {
                name = name.replacingOccurrences(
                    of: ".experts.down_projs", with: ".switch_mlp.fc2.weight")
                value = value.transposed(0, 2, 1)
            } else if sourceLayout && name.hasSuffix(".conv1d.weight") && value.dim(1) == 1 {
                value = value.transposed(0, 2, 1)
            } else if sourceLayout && name.hasSuffix(".subpixel_upsample.weight")
                && value.dim(-1) == 3
            {
                value = value.transposed(0, 2, 1)
            } else if name == "ctc_timestamp.decoder.upsample.weight",
                value.dim(-1) == timestampUpsamplingFactor
            {
                value = value.transposed(1, 2, 0)
            } else if name == "ctc_timestamp.decoder.decoder_layers.0.weight", value.ndim == 3 {
                value = value.squeezed(axis: -1)
            }
            if name.hasPrefix("llm.") && name.hasSuffix(".gate.weight") {
                value = value.asType(.float32)
            }
            output[name] = value
        }
        return output
    }

    private static func packExperts(
        _ weights: inout [String: MLXArray], expertCount: Int?, evaluate: Bool
    ) {
        let pattern = try! NSRegularExpression(
            pattern:
                #"^(.+)\.experts\.0\.(up_proj|down_proj)\.(weight|scales|biases|weight_scale_2|weight_global_scale|pre_quant_scale)$"#
        )
        for key in weights.keys.sorted() {
            let text = key as NSString
            guard
                let match = pattern.firstMatch(
                    in: key, range: NSRange(location: 0, length: text.length))
            else { continue }
            let prefix = text.substring(with: match.range(at: 1))
            let projection = text.substring(with: match.range(at: 2))
            let suffix = text.substring(with: match.range(at: 3))
            let count =
                expertCount
                ?? weights.keys.filter {
                    $0.hasPrefix(prefix + ".experts.") && $0.hasSuffix(".\(projection).\(suffix)")
                }.count
            let keys = (0 ..< count).map { "\(prefix).experts.\($0).\(projection).\(suffix)" }
            guard count > 0, keys.allSatisfy({ weights[$0] != nil }) else { continue }
            let values = keys.map { weights.removeValue(forKey: $0)! }
            let packed = stacked(values)
            if evaluate { eval(packed) }
            weights["\(prefix).switch_mlp.\(projection == "up_proj" ? "fc1" : "fc2").\(suffix)"] =
                packed
            if evaluate { Memory.clearCache() }
        }
    }

    public static func fromPretrained(
        _ modelRepo: String = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        hfToken: String? = nil, cache: HubCache = .default
    ) async throws -> NemotronTranscribeModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["alignment_tokenizer/*"], hfToken: hfToken, cache: cache)
        return try await fromModelDirectory(directory)
    }
    public static func fromModelDirectory(_ directory: URL) async throws -> NemotronTranscribeModel
    {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(NemotronTranscribeConfiguration.self, from: data)
        try config.validate()
        let model = NemotronTranscribeModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No Nemotron Transcribe safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files {
            let shard = try MLX.loadArrays(url: file)
            guard Set(weights.keys).isDisjoint(with: shard.keys) else {
                throw STTError.modelNotInitialized("Duplicate Nemotron checkpoint tensors")
            }
            weights.merge(shard) { _, new in new }
        }
        weights = weights.filter { !$0.key.hasPrefix("llm.mtp.") }
        var source = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        try ModelOpt.prepare(weights: &weights, configuration: &source, directory: directory)
        let hasCTC = weights.keys.contains {
            $0.hasPrefix("ctc_timestamp.decoder.") && !$0.hasSuffix("upsample_factor")
        }
        let embeddedTokenizer = weights["ctc_timestamp.tokenizer.model_proto"].map {
            Data($0.asArray(UInt8.self))
        }
        if !hasCTC, model.ctcTimestamp != nil {
            try model.update(
                modules: ModuleChildren(values: ["ctc_timestamp": .none]), verify: .none)
        }
        weights = remap(
            weights: weights, includeTimestamps: hasCTC,
            timestampUpsamplingFactor: config.timestamps?.upsamplingFactor ?? 1)
        // Consume each source expert bank before materializing the next one.
        packExperts(&weights, expertCount: config.text.experts, evaluate: true)
        let base = try JSONDecoder().decode(
            BaseConfiguration.self, from: JSONSerialization.data(withJSONObject: source))
        if let q = base.perLayerQuantization {
            try ModelOpt.quantize(model: model) { path, _ in
                weights[path + ".scales"] == nil ? nil : q.quantization(layer: path)?.asTuple
            }
        }
        try ModelOpt.installScaledLinears(in: model, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let tokenizer = try await loadSTTTokenizer(from: directory)
        guard tokenizer.encode(text: config.audioToken, addSpecialTokens: false).count == 1 else {
            throw STTError.modelNotInitialized(
                "Nemotron tokenizer must encode the audio placeholder as one token")
        }
        model._tokenizer = UncheckedSendableBox(tokenizer)
        if let data = try? Data(
            contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
            let fields = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            var special: Set<String> = []
            for value in (fields["added_tokens_decoder"] as? [String: [String: Any]] ?? [:]).values
            {
                if value["special"] as? Bool == true, let token = value["content"] as? String {
                    special.insert(token)
                }
            }
            for name in [
                "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token",
                "mask_token",
            ] {
                if let token = fields[name] as? String { special.insert(token) }
                if let token = (fields[name] as? [String: Any])?["content"] as? String {
                    special.insert(token)
                }
            }
            special.formUnion(fields["additional_special_tokens"] as? [String] ?? [])
            model.alignmentSpecialTokens = special.sorted { $0.count > $1.count }
        }
        let alignmentDirectory = directory.appendingPathComponent("alignment_tokenizer")
        if hasCTC
            && FileManager.default.fileExists(
                atPath: alignmentDirectory.appendingPathComponent("tokenizer.json").path)
        {
            model.alignmentTokenizer = UncheckedSendableBox(
                .huggingFace(
                    try await loadSTTTokenizer(from: alignmentDirectory)))
        } else if hasCTC {
            let file = alignmentDirectory.appendingPathComponent("tokenizer.model")
            let proto =
                FileManager.default.fileExists(atPath: file.path)
                ? try Data(contentsOf: file) : embeddedTokenizer
            if let proto {
                model.alignmentTokenizer = UncheckedSendableBox(
                    .sentencePiece(
                        try SentencePieceTokenizer(sentencePieceModelData: proto)))
            }
        }
        model.train(false)
        eval(model)
        return model
    }
}
