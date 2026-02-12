import Foundation
import Hub
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
import MLXLMCommon
import MLXNN
import Tokenizers

public final class MarvisTTSModel: Module {
    public enum Voice: String, Sendable {
        case conversationalA = "conversational_a"
        case conversationalB = "conversational_b"
    }
    
    public enum QualityLevel: Int, Sendable {
        case low = 8
        case medium = 16
        case high = 24
        case maximum = 32
    }
    
    public let sampleRate: Int
    
    private let model: CSMModel
    private let _promptURLs: [URL]?
    private let _textTokenizer: Tokenizer
    private let _audio_tokenizer: MimiTokenizer
    private let _streamingDecoder: MimiStreamingDecoder
    
    public init(
        config: CSMModelArgs,
        repoId: String,
        promptURLs: [URL]? = nil,
        textTokenizer: Tokenizer,
        audioTokenizer: MimiTokenizer
    ) {
        _ = repoId
        self.model = CSMModel(config: config)
        self._promptURLs = promptURLs
        self._textTokenizer = textTokenizer
        self._audio_tokenizer = audioTokenizer
        self._streamingDecoder = MimiStreamingDecoder(audioTokenizer.codec)
        self.sampleRate = Int(audioTokenizer.codec.cfg.sampleRate)
        super.init()

        model.resetCaches()
    }

    public convenience init(
        config: CSMModelArgs,
        hub: HubApi = .shared,
        repoId: String,
        promptURLs: [URL]? = nil,
        progressHandler: @escaping (Progress) -> Void
    ) async throws {
        let textTokenizer = try await loadTokenizer(configuration: ModelConfiguration(id: repoId), hub: hub)
        let codec = try await Mimi.fromPretrained(progressHandler: progressHandler)
        let audioTokenizer = MimiTokenizer(codec)
        self.init(
            config: config,
            repoId: repoId,
            promptURLs: promptURLs,
            textTokenizer: textTokenizer,
            audioTokenizer: audioTokenizer
        )
    }
    
    private func tokenizeTextSegment(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        let frameW = K + 1
        
        let prompt = "[\(speaker)]" + text
        let ids = MLXArray(_textTokenizer.encode(text: prompt))
        
        let T = ids.shape[0]
        var frame = MLXArray.zeros([T, frameW], type: Int32.self)
        var mask = MLXArray.zeros([T, frameW], type: Bool.self)
        
        let lastCol = frameW - 1
        do {
            let left = split(frame, indices: [lastCol], axis: 1)[0]
            let right = split(frame, indices: [lastCol], axis: 1)[1]
            let tail = split(right, indices: [1], axis: 1)
            let after = (tail.count > 1) ? tail[1] : MLXArray.zeros([T, 0], type: Int32.self)
            frame = concatenated([left, ids.reshaped([T, 1]), after], axis: 1)
        }
        
        do {
            let ones = MLXArray.ones([T, 1], type: Bool.self)
            let left = split(mask, indices: [lastCol], axis: 1)[0]
            let right = split(mask, indices: [lastCol], axis: 1)[1]
            let tail = split(right, indices: [1], axis: 1)
            let after = (tail.count > 1) ? tail[1] : MLXArray.zeros([T, 0], type: Bool.self)
            mask = concatenated([left, ones, after], axis: 1)
        }
        
        return (frame, mask)
    }
    
    private func cacheURL(for audioURL: URL) -> URL {
        let cacheFilename = audioURL.deletingPathExtension().appendingPathExtension("npy").lastPathComponent
        let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.appending(component: "MarvisTTSModel/prompt_cache")
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        return cacheDirectory.appending(component: cacheFilename)
    }
    
    private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) throws -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        
        var codes: MLXArray
        let x = audio.reshaped([1, 1, audio.shape[0]])
        let encoded = encodeChunked(x) // [1, K, Tq]
        codes = split(encoded, indices: [1], axis: 0)[0].reshaped([K, encoded.shape[2]])
        
        let frameW = K + 1
        
        if addEOS {
            let eos = MLXArray.zeros([K, 1], type: Int32.self)
            codes = concatenated([codes, eos], axis: 1) // [K, Tq+1]
        }
        
        let T = codes.shape[1]
        var frame = MLXArray.zeros([T, frameW], type: Int32.self) // [T, K+1]
        var mask = MLXArray.zeros([T, frameW], type: Bool.self)
        
        let codesT = swappedAxes(codes, 0, 1) // [T, K]
        if K > 0 {
            let leftLen = K
            let right = split(frame, indices: [leftLen], axis: 1)[1] // [T, 1]
            frame = concatenated([codesT, right], axis: 1)
        }
        if K > 0 {
            let ones = MLXArray.ones([T, K], type: Bool.self)
            let right = MLXArray.zeros([T, 1], type: Bool.self)
            mask = concatenated([ones, right], axis: 1)
        }
        
        return (frame, mask)
    }
    
    private func tokenizeSegment(_ segment: Segment, addEOS: Bool = true) throws -> (MLXArray, MLXArray) {
        let (txt, txtMask) = tokenizeTextSegment(text: segment.text, speaker: segment.speaker)
        let (aud, audMask) = try tokenizeAudio(segment.audio, addEOS: addEOS)
        return (concatenated([txt, aud], axis: 0), concatenated([txtMask, audMask], axis: 0))
    }
}

public extension MarvisTTSModel {
    static func fromPretrained(
        _ modelRepo: String = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit",
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> MarvisTTSModel {
        Memory.cacheLimit = 100 * 1024 * 1024

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(
                domain: "MarvisTTSModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"]
            )
        }

        let modelDirectoryURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        let promptFileURLs = modelDirectoryURL.appendingPathComponent("prompts", isDirectory: true)
        var audioPromptURLs = [URL]()
        if FileManager.default.fileExists(atPath: promptFileURLs.path) {
            for promptURL in try FileManager.default.contentsOfDirectory(at: promptFileURLs, includingPropertiesForKeys: nil) {
                if promptURL.pathExtension == "wav" {
                    audioPromptURLs.append(promptURL)
                }
            }
        }

        let configFileURL = modelDirectoryURL.appendingPathComponent("config.json")
        let args = try JSONDecoder().decode(CSMModelArgs.self, from: Data(contentsOf: configFileURL))

        let textTokenizer = try await AutoTokenizer.from(modelFolder: modelDirectoryURL)
        let codec = try await Mimi.fromPretrained(progressHandler: progressHandler)
        let audioTokenizer = MimiTokenizer(codec)
        let model = MarvisTTSModel(
            config: args,
            repoId: modelRepo,
            promptURLs: audioPromptURLs,
            textTokenizer: textTokenizer,
            audioTokenizer: audioTokenizer
        )

        var weights = try marvisLoadWeights(from: modelDirectoryURL)
        
        if let quantization = args.quantization,
           let groupSize = quantization["group_size"],
           let bits = quantization["bits"] {
            if case let .number(g) = groupSize, case let .number(b) = bits {
                quantize(model: model, groupSize: Int(g), bits: Int(b)) { path, _ in
                    weights["\(path).scales"] != nil
                }
            }
            
        } else {
            weights = sanitize(weights: weights)
        }
        
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        
        eval(model)
        return model
    }

    static func fromPretrained(
        hub: HubApi = .shared,
        repoId: String = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit",
        progressHandler: @escaping (Progress) -> Void
    ) async throws -> MarvisTTSModel {
        _ = hub
        return try await fromPretrained(repoId, progressHandler: progressHandler)
    }
    
    private static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        
        for (rawKey, v) in weights {
            var k = rawKey
            
            if !k.hasPrefix("model.") {
                k = "model." + k
            }
            
            if k.contains("attn") && !k.contains("self_attn") {
                k = k.replacingOccurrences(of: "attn", with: "self_attn")
                k = k.replacingOccurrences(of: "output_proj", with: "o_proj")
            }
            
            if k.contains("mlp") {
                k = k.replacingOccurrences(of: "w1", with: "gate_proj")
                k = k.replacingOccurrences(of: "w2", with: "down_proj")
                k = k.replacingOccurrences(of: "w3", with: "up_proj")
            }
            
            if k.contains("sa_norm") || k.contains("mlp_norm") {
                k = k.replacingOccurrences(of: "sa_norm", with: "input_layernorm")
                k = k.replacingOccurrences(of: "scale", with: "weight")
                k = k.replacingOccurrences(of: "mlp_norm", with: "post_attention_layernorm")
                k = k.replacingOccurrences(of: "scale", with: "weight")
            }
            
            if k.contains("decoder.norm") || k.contains("backbone.norm") {
                k = k.replacingOccurrences(of: "scale", with: "weight")
            }
            
            out[k] = v
        }
        
        return out
    }
}

// MARK: - Loading Helpers

private func marvisLoadWeights(from directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let weightFileURL = directory.appendingPathComponent("model.safetensors")

    if fileManager.fileExists(atPath: weightFileURL.path) {
        return try MLX.loadArrays(url: weightFileURL)
    }

    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

private struct Segment {
    public let speaker: Int
    public let text: String
    public let audio: MLXArray
    
    public init(speaker: Int, text: String, audio: MLXArray) {
        self.speaker = speaker
        self.text = text
        self.audio = audio
    }
}

// MARK: -

enum MarvisTTSModelError: Error {
    case invalidArgument(String)
    case voiceNotFound
    case invalidRefAudio(String)
}

public extension MarvisTTSModel {
    struct GenerationResult: Sendable {
        public let audio: [Float]
        public let sampleRate: Int
        public let sampleCount: Int
        public let frameCount: Int
        public let audioDuration: TimeInterval
        public let realTimeFactor: Double
        public let processingTime: Double
    }
    
    func generate(
        text: [String],
        voice: Voice? = .conversationalA,
        qualityLevel: QualityLevel = .maximum,
        refAudio: MLXArray?,
        refText: String?,
        splitPattern: String? = #"(\n+)"#,
        streamingInterval: Double = 0.5
    ) async throws -> [Float] {
        var out: [GenerationResult] = []
        for try await chunk in generate(
            text: text,
            voice: voice,
            qualityLevel: qualityLevel,
            refAudio: refAudio,
            refText: refText,
            streamingInterval: streamingInterval
        ) {
            out.append(chunk)
        }
        return out.map(\.audio).flatMap { $0 }
    }
    
    func generate(
        text: String,
        voice: Voice? = .conversationalA,
        qualityLevel: QualityLevel = .maximum,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        splitPattern: String? = #"(\n+)"#,
        streamingInterval: Double = 0.5
    ) -> AsyncThrowingStream<GenerationResult, Error> {
        generate(
            text: Self.textPieces(text, splitPattern: splitPattern),
            voice: voice,
            qualityLevel: qualityLevel,
            refAudio: refAudio,
            refText: refText,
            streamingInterval: streamingInterval
        )
    }
    
    func generate(
        text: [String],
        voice: Voice? = .conversationalA,
        qualityLevel: QualityLevel = .maximum,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        streamingInterval: Double = 0.5
    ) -> AsyncThrowingStream<GenerationResult, Error> {
        let (stream, continuation) = AsyncThrowingStream<GenerationResult, Error>.makeStream()
        
        Task { @Sendable [weak self, continuation] in
            guard let self else { return }
            do {
                guard voice != nil || refAudio != nil else {
                    throw MarvisTTSModelError.invalidArgument("`voice` or `refAudio`/`refText` must be specified.")
                }
                
                let context: Segment
                if let refAudio, let refText {
                    context = Segment(speaker: 0, text: refText, audio: refAudio)
                } else if let voice {
                    var refAudioURL: URL?
                    for promptURL in _promptURLs ?? [] {
                        if promptURL.lastPathComponent == "\(voice.rawValue).wav" {
                            refAudioURL = promptURL
                        }
                    }
                    guard let refAudioURL else {
                        throw MarvisTTSModelError.voiceNotFound
                    }
                    
                    let refTextURL = refAudioURL.deletingPathExtension().appendingPathExtension("txt")
                    let refText = try String(data: Data(contentsOf: refTextURL), encoding: .utf8)
                    guard let refText else {
                        throw MarvisTTSModelError.voiceNotFound
                    }
                    
                    let (_, refAudio) = try loadAudioArray(from: refAudioURL)
                    context = Segment(speaker: 0, text: refText, audio: refAudio)
                } else {
                    throw MarvisTTSModelError.voiceNotFound
                }
                
                let maxAudioFrames = Int(60000 / 80.0) // 12.5 fps, 80 ms per frame
                let streamingIntervalTokens = Int(streamingInterval * 12.5)
                
            outerLoop:
                for prompt in text {
                    if Task.isCancelled { break outerLoop }
                    
                    let generationText = (context.text + " " + prompt).trimmingCharacters(in: .whitespaces)
                    let seg = Segment(speaker: 0, text: generationText, audio: context.audio)
                    
                    model.resetCaches()
                    _streamingDecoder.reset()
                    
                    let (toks, masks) = try tokenizeSegment(seg, addEOS: false)
                    let promptTokens = toks.asType(Int32.self) // [T, K+1]
                    let promptMask = masks.asType(Bool.self) // [T, K+1]
                    
                    var samplesFrames: [MLXArray] = [] // each is [B=1, K]
                    var currTokens = expandedDimensions(promptTokens, axis: 0) // [1, T, K+1]
                    var currMask = expandedDimensions(promptMask, axis: 0) // [1, T, K+1]
                    var currPos = expandedDimensions(MLXArray.arange(promptTokens.shape[0]), axis: 0) // [1, T]
                    var generatedCount = 0
                    var yieldedCount = 0
                    
                    let maxSeqLen = 2048 - maxAudioFrames
                    precondition(currTokens.shape[1] < maxSeqLen, "Inputs too long, must be below max_seq_len - max_audio_frames: \(maxSeqLen)")
                    
                    let sampleFn = TopPSampler(temperature: 0.9, topP: 0.8).sample
                    
                    var startTime = CFAbsoluteTimeGetCurrent()
                    
                    for _ in 0 ..< maxAudioFrames {
                        if Task.isCancelled { break outerLoop }
                        
                        let frame = model.generateFrame(
                            maxCodebooks: qualityLevel.rawValue,
                            tokens: currTokens,
                            tokensMask: currMask,
                            inputPos: currPos,
                            sampler: sampleFn
                        ) // [1, K]
                        
                        // EOS if every codebook is 0
                        if frame.sum().item(Int32.self) == 0 {
                            break
                        }
                        
                        samplesFrames.append(frame)
                        
                        let zerosText = MLXArray.zeros([1, 1], type: Int32.self)
                        let nextFrame = concatenated([frame, zerosText], axis: 1) // [1, K+1]
                        currTokens = expandedDimensions(nextFrame, axis: 1) // [1, 1, K+1]
                        
                        let onesK = ones([1, frame.shape[1]], type: Bool.self)
                        let zero1 = zeros([1, 1], type: Bool.self)
                        let nextMask = concatenated([onesK, zero1], axis: 1) // [1, K+1]
                        currMask = expandedDimensions(nextMask, axis: 1) // [1, 1, K+1]
                        
                        currPos = split(currPos, indices: [currPos.shape[1] - 1], axis: 1)[1] + MLXArray(1)
                        
                        generatedCount += 1
                        
                        if (generatedCount - yieldedCount) >= streamingIntervalTokens {
                            yieldedCount = generatedCount
                            let gr = generateResultChunk(samplesFrames, start: startTime)
                            continuation.yield(gr)
                            samplesFrames.removeAll(keepingCapacity: true)
                            startTime = CFAbsoluteTimeGetCurrent()
                        }
                    }
                    
                    if !samplesFrames.isEmpty {
                        let gr = generateResultChunk(samplesFrames, start: startTime)
                        continuation.yield(gr)
                    }
                }
                
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }
    
    private static func textPieces(_ text: String, splitPattern: String?) -> [String] {
        let pieces: [String]
        if let pat = splitPattern, let re = try? NSRegularExpression(pattern: pat) {
            let full = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let range = NSRange(full.startIndex ..< full.endIndex, in: full)
            let splits = re.split(full, range: range)
            pieces = splits.isEmpty ? [full] : splits
        } else {
            pieces = [text]
        }
        return pieces
    }
    
    private func generateResultChunk(_ frames: [MLXArray], start: CFTimeInterval) -> GenerationResult {
        let frameCount = frames.count
        
        var stacked = stacked(frames, axis: 0) // [F, 1, K]
        stacked = swappedAxes(stacked, 0, 1) // [1, F, K]
        stacked = swappedAxes(stacked, 1, 2) // [1, K, F]
        
        let audio1x1x = _streamingDecoder.decodeFrames(stacked) // [1, 1, S]
        let sampleCount = audio1x1x.shape[2]
        let audio = audio1x1x.reshaped([sampleCount]) // [S]
        
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let audioSeconds = Double(sampleCount) / Double(sampleRate)
        let rtf = (audioSeconds > 0) ? elapsed / audioSeconds : 0.0
        
        return GenerationResult(
            audio: audio.asArray(Float32.self),
            sampleRate: sampleRate,
            sampleCount: sampleCount,
            frameCount: frameCount,
            audioDuration: audioSeconds,
            realTimeFactor: (rtf * 100).rounded() / 100,
            processingTime: elapsed,
        )
    }
}

// MARK: - Mimi helpers

private extension MarvisTTSModel {
    func encodeChunked(_ xs: MLXArray, chunkSize: Int = 48_000) -> MLXArray {
        _audio_tokenizer.codec.resetState()

        var codes = [MLXArray]()
        for start in stride(from: 0, to: xs.shape[2], by: chunkSize) {
            let xsChunk = xs[.ellipsis, start ..< min(xs.shape[2], start + chunkSize)]
            let partialCodes = _audio_tokenizer.codec.encodeStep(xsChunk)
            codes.append(partialCodes)
        }
        return MLX.concatenated(codes, axis: 2)
    }
}

// MARK: - SpeechGenerationModel conformance

extension MarvisTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: Int(60000 / 80.0),
            temperature: 0.9,
            topP: 0.8
        )
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = generationParameters
        let resolvedVoice = try resolveVoice(from: voice)

        let audio = try await generate(
            text: [text],
            voice: resolvedVoice,
            qualityLevel: .maximum,
            refAudio: refAudio,
            refText: refText,
            streamingInterval: 0.5
        )

        return MLXArray(audio)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        
        Task { @Sendable [weak self, continuation] in
            guard let self else { return }
            
            do {
                _ = generationParameters
                let resolvedVoice = try resolveVoice(from: voice)
                
                for try await chunk in generate(
                    text: text,
                    voice: resolvedVoice,
                    qualityLevel: .maximum,
                    refAudio: refAudio,
                    refText: refText,
                    streamingInterval: 0.5
                ) {
                    continuation.yield(.audio(MLXArray(chunk.audio)))
                }
                
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        
        return stream
    }
}

private extension MarvisTTSModel {
    func resolveVoice(from voice: String?) throws -> Voice {
        guard let voice, !voice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .conversationalA
        }

        let normalized = voice.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let resolved = Voice(rawValue: normalized) else {
            throw AudioGenerationError.invalidInput("Unknown Marvis voice: \(voice)")
        }

        return resolved
    }
}

// MARK: -

private extension NSRegularExpression {
    func split(_ s: String, range: NSRange) -> [String] {
        var last = 0
        var parts: [String] = []
        enumerateMatches(in: s, options: [], range: range) { m, _, _ in
            guard let m else { return }
            let r = NSRange(location: last, length: m.range.location - last)
            if let rr = Range(r, in: s) {
                let piece = String(s[rr]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !piece.isEmpty { parts.append(piece) }
            }
            last = m.range.upperBound
        }
        let tailR = NSRange(location: last, length: range.upperBound - last)
        if let rr = Range(tailR, in: s) {
            let piece = String(s[rr]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { parts.append(piece) }
        }
        return parts
    }
}
