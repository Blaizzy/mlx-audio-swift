import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

private let defaultAudioPrompt: String = "alba"

public struct PocketTTSState {
    public var flowCache: [KVCacheSimple]
}

public final class PocketTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: PocketTTSModelConfig
    private let modelFolder: URL
    @ModuleInfo(key: "flow_lm") public var flow_lm: FlowLMModel
    @ModuleInfo(key: "mimi") public var mimi: MimiAdapter
    public var speaker_proj_weight: MLXArray

    public var temp: Float
    public var lsd_decode_steps: Int
    public var noise_clamp: Float?
    public var eos_threshold: Float

    public var sampleRate: Int { config.mimi.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: config.temp)
    }

    private init(config: PocketTTSModelConfig, modelFolder: URL, flowLM: FlowLMModel, mimi: MimiAdapter) {
        self.config = config
        self.modelFolder = modelFolder
        self._flow_lm = ModuleInfo(wrappedValue: flowLM)
        self._mimi = ModuleInfo(wrappedValue: mimi)
        self.speaker_proj_weight = MLXArray.zeros([config.flowLM.transformer.dModel, config.mimi.seanet.dimension])
        self.temp = config.temp
        self.lsd_decode_steps = config.lsdDecodeSteps
        self.noise_clamp = config.noiseClamp
        self.eos_threshold = config.eosThreshold
        super.init()
    }

    public static func fromConfig(
        _ config: PocketTTSModelConfig, modelFolder: URL, encoderEnabled: Bool? = nil
    ) async throws -> PocketTTSModel {
        guard config.fuser.cross.isEmpty, config.fuser.streamingSum.isEmpty,
              Set(config.fuser.sum).isSubset(of: ["num_speakers", "duration_delta"]) else {
            throw modelError("Unsupported Pocket TTS conditioning fuser")
        }
        guard ["rms", "rms_var"].contains(config.flowLM.timeRmsNorm),
              ["erf", "tanh"].contains(config.flowLM.transformer.gelu),
              ["erf", "tanh"].contains(config.mimi.gelu) else {
            throw modelError("Unsupported timestep normalization or GELU activation")
        }
        var numSpeakers: SpeakerCountConditioner?
        var durationDelta: ContinuousConditioner?
        for conditioner in config.conditioners {
            if conditioner.name == "num_speakers", conditioner.type == "lut", let lut = conditioner.lut {
                numSpeakers = SpeakerCountConditioner(
                    nBins: lut.nBins, dim: lut.dim, outputDim: config.flowLM.transformer.dModel)
            } else if conditioner.name == "duration_delta", conditioner.type == "continuous", let continuous = conditioner.continuous {
                guard continuous.dim >= 4, continuous.dim % 2 == 0 else {
                    throw modelError("Continuous conditioning requires an even dim >= 4")
                }
                durationDelta = ContinuousConditioner(
                    dim: continuous.dim, outputDim: config.flowLM.transformer.dModel,
                    scaleFactor: continuous.scaleFactor, maxPeriod: continuous.maxPeriod)
            } else {
                throw modelError("Unsupported conditioner: \(conditioner.name)")
            }
        }
        let flowLM = try await FlowLMModel.fromConfig(
            config.flowLM,
            latentDim: config.flowLM.ldim ?? config.mimi.quantizer.dimension,
            modelFolder: modelFolder,
            tokenizerPath: config.tokenizerName,
            numSpeakers: numSpeakers,
            durationDelta: durationDelta
        )
        let mimi = MimiAdapter.fromConfig(config.mimi, encoderEnabled: encoderEnabled)
        let model = PocketTTSModel(config: config, modelFolder: modelFolder, flowLM: flowLM, mimi: mimi)
        try model.validateSampling()
        return model
    }

    private static func modelError(_ message: String) -> NSError {
        NSError(domain: "PocketTTSModel", code: 2, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func validateSampling() throws {
        guard lsd_decode_steps > 0, temp.isFinite, temp >= 0,
              noise_clamp.map({ $0.isFinite && $0 >= 0 }) ?? true else {
            throw Self.modelError("Sampling requires positive lsd_decode_steps and finite, nonnegative temperature and noise_clamp")
        }
    }

    private static func evaluatedWaveform(_ audio: MLXArray) -> sending MLXArray {
        eval(audio)
        // Completed waveform tensors are never mutated by subsequent decoder
        // steps. Swift cannot infer this independence through MLX operations.
        nonisolated(unsafe) let output = audio
        return output
    }

    public func initState() -> PocketTTSState {
        PocketTTSState(flowCache: flow_lm.makeCache())
    }

    private func runFlowLM(
        _ state: inout PocketTTSState,
        textTokens: MLXArray,
        backboneInputLatents: MLXArray,
        audioConditioning: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let textEmb = textTokens.shape[1] > 0
            ? flow_lm.conditioner(TokenizedText(tokens: textTokens))
            : MLXArray.zeros([1, 0, flow_lm.dim])
        let combined = concatenated([textEmb, audioConditioning], axis: 1)
        if backboneInputLatents.shape[1] == 0 {
            if combined.shape[1] > 0 {
                _ = flow_lm.transformer(combined.asType(.float32), cache: state.flowCache)
            }
            // Match the reference RNG sequence without evaluating an unused latent.
            _ = MLXRandom.normal([1, flow_lm.ldim])
            return nil
        }
        let (out, isEos) = flow_lm(
            sequence: backboneInputLatents,
            textEmbeddings: combined,
            cache: state.flowCache,
            lsdDecodeSteps: lsd_decode_steps,
            temperature: temp,
            noiseClamp: noise_clamp,
            eosThreshold: eos_threshold
        )
        let outExpanded = out.expandedDimensions(axis: 1)
        return (outExpanded, isEos)
    }

    private func runFlowLMAndIncrementStep(
        _ state: inout PocketTTSState,
        textTokens: MLXArray? = nil,
        backboneInputLatents: MLXArray? = nil,
        audioConditioning: MLXArray? = nil
    ) -> (MLXArray, MLXArray)? {
        let tokens = textTokens ?? MLXArray.zeros([1, 0], type: Int32.self)
        let backbone = backboneInputLatents ?? MLXArray.zeros([1, 0, flow_lm.ldim])
        let conditioning = audioConditioning ?? MLXArray.zeros([1, 0, flow_lm.dim])
        return runFlowLM(&state, textTokens: tokens, backboneInputLatents: backbone, audioConditioning: conditioning)
    }

    private func encodeAudio(_ audio: MLXArray) -> MLXArray {
        let encoded = mimi.encodeToLatent(audio)
        let latents = encoded.transposed(0, 2, 1).asType(.float32)
        let projT = speaker_proj_weight.transposed(1, 0)
        let conditioning = matmul(latents, projT)
        return conditioning
    }

    private func normalizeAudio(_ audio: MLXArray) -> MLXArray {
        if audio.ndim == 1 {
            return audio.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        }
        if audio.ndim == 2 {
            var mono = audio
            if audio.shape[0] > 1 {
                mono = MLX.mean(audio, axis: 0, keepDims: true)
            }
            return mono.expandedDimensions(axis: 0)
        }
        return audio
    }

    private enum AudioPrompt {
        case embedding(MLXArray)
        case audio(MLXArray)
    }

    public static func loadPredefinedVoice(
        _ voiceName: String,
        modelFolder: URL,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXArray? {
        _ = progressHandler
        guard let fileURL = voiceFiles(in: modelFolder)[voiceName.lowercased()] else { return nil }
        let arrays = try MLX.loadArrays(url: fileURL)
        return arrays["audio_prompt"] ?? arrays["emb"] ?? arrays["speaker_wavs"]
    }

    private static func voiceFiles(in modelFolder: URL) -> [String: URL] {
        var voices: [String: URL] = [:]
        for directory in ["voices", "embeddings"] {
            let files = (try? FileManager.default.contentsOfDirectory(
                at: modelFolder.appendingPathComponent(directory), includingPropertiesForKeys: nil)) ?? []
            for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where file.pathExtension == "safetensors" {
                voices[file.deletingPathExtension().lastPathComponent.lowercased()] = file
            }
        }
        let defaultVoice = modelFolder.appendingPathComponent("default-voice.safetensors")
        if voices["default"] == nil, FileManager.default.fileExists(atPath: defaultVoice.path) {
            voices["default"] = defaultVoice
        }
        return voices
    }

    func voiceConditioning(_ arrays: [String: MLXArray]) throws -> MLXArray {
        if var latents = arrays["speaker_wavs"] {
            if latents.ndim == 2 { latents = latents.expandedDimensions(axis: 0) }
            guard latents.ndim == 3, latents.shape[0] == 1,
                  latents.shape[1] == speaker_proj_weight.shape[1] else {
                throw Self.modelError("Speaker latents must have shape [1, \(speaker_proj_weight.shape[1]), T]")
            }
            return matmul(latents.transposed(0, 2, 1).asType(.float32), speaker_proj_weight.T)
        }
        guard let embedding = arrays["emb"] ?? arrays["audio_prompt"] else {
            throw Self.modelError("Voice file must contain speaker_wavs, emb, or audio_prompt")
        }
        return embedding
    }

    private func resolveAudioPrompt(
        voice: String?,
        refAudio: MLXArray?,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> AudioPrompt {
        if let refAudio {
            guard mimi.encoder != nil else {
                throw Self.modelError("This checkpoint omits the speaker encoder; use a bundled voice embedding")
            }
            var audio = normalizeAudio(refAudio)
            guard audio.ndim == 3, audio.shape[0] == 1, audio.shape[1] == config.mimi.channels else {
                throw Self.modelError("Audio prompt must have shape [1, \(config.mimi.channels), T]")
            }
            guard Double(audio.shape[2]) >= config.audioPromptMinDuration * Double(sampleRate) else {
                throw Self.modelError("Audio prompt must be at least \(config.audioPromptMinDuration) seconds")
            }
            if let maximum = config.audioPromptMaxDuration {
                audio = audio[.ellipsis, ..<min(audio.shape[2], Int(maximum * Double(sampleRate)))]
            }
            if audio.shape[2] == 0 { return .embedding(MLXArray.zeros([1, 0, flow_lm.dim])) }
            return .audio(audio.asType(.float32))
        }

        if let voice, let file = try await resolveVoiceFile(voice) {
            return .embedding(try voiceConditioning(MLX.loadArrays(url: file)))
        }

        let voices = Self.voiceFiles(in: modelFolder)
        let defaultVoice = ["default", defaultAudioPrompt].first { voices[$0] != nil }
        let selected = voice?.lowercased()
            ?? defaultVoice ?? voices.keys.sorted().first ?? defaultAudioPrompt
        guard let file = voices[selected] else {
            throw Self.modelError("Unknown voice '\(selected)'. Available voices: \(voices.keys.sorted().joined(separator: ", "))")
        }
        return .embedding(try voiceConditioning(MLX.loadArrays(url: file)))
    }

    private func resolveVoiceFile(_ voice: String) async throws -> URL? {
        if voice.hasPrefix("hf://") {
            let parts = voice.dropFirst(5).split(separator: "/", maxSplits: 2)
            guard parts.count == 3, let repo = Repo.ID(rawValue: "\(parts[0])/\(parts[1])") else {
                throw Self.modelError("Invalid Hugging Face voice URL: \(voice)")
            }
            let fileAndRevision = parts[2].split(separator: "@", maxSplits: 1)
            guard let filename = fileAndRevision.first, filename.hasSuffix(".safetensors") else {
                throw Self.modelError("Voice URL must reference a safetensors file")
            }
            let revision = fileAndRevision.count == 2 ? String(fileAndRevision[1]) : "main"
            let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
            let client: HubClient
            if let token, !token.isEmpty {
                client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: .default)
            } else {
                client = HubClient(cache: .default)
            }
            return try await client.downloadFile(at: String(filename), from: repo, revision: revision)
        }
        let expanded = (voice as NSString).expandingTildeInPath
        let file = URL(fileURLWithPath: expanded)
        if file.pathExtension == "safetensors", FileManager.default.fileExists(atPath: file.path) {
            return file
        }
        return nil
    }

    private func getStateForAudioPrompt(_ prompt: AudioPrompt) throws -> PocketTTSState {
        var state = initState()
        var conditioning: MLXArray = switch prompt {
        case .embedding(let emb):
            emb
        case .audio(let audio):
            encodeAudio(audio)
        }
        if conditioning.ndim == 2 { conditioning = conditioning.expandedDimensions(axis: 0) }
        guard conditioning.ndim == 3, conditioning.shape[0] == 1, conditioning.shape[2] == flow_lm.dim else {
            throw Self.modelError("Voice embedding must have shape [1, T, \(flow_lm.dim)]")
        }
        if conditioning.shape[1] > 0 {
            _ = runFlowLMAndIncrementStep(&state, audioConditioning: conditioning)
        }
        sliceFlowCache(&state, to: conditioning.shape[1])
        return state
    }

    private func sliceFlowCache(_ state: inout PocketTTSState, to length: Int) {
        let targetLength = max(length, 0)
        for cache in state.flowCache {
            let s = cache.state
            guard s.count == 2 else { continue }
            let keys = s[0]
            let values = s[1]
            let end = min(targetLength, keys.shape[2])
            let slicedKeys = keys[.ellipsis, ..<end, 0...]
            let slicedValues = values[.ellipsis, ..<end, 0...]
            cache.state = [slicedKeys, slicedValues]
            cache.offset = min(cache.offset, end)
        }
    }

    private func getFlowCacheNumFrames(_ state: PocketTTSState) -> Int {
        for cache in state.flowCache {
            let s = cache.state
            guard s.count == 2 else { continue }
            let keys = s[0]
            return min(cache.offset, keys.shape[2])
        }
        return 0
    }

    public func generateAudio(
        state: PocketTTSState?,
        text: String,
        framesAfterEos: Int?,
        maxFrames: Int?
    ) throws -> MLXArray {
        var chunks: [MLXArray] = []
        for chunk in try generateAudioStream(state: state, text: text, framesAfterEos: framesAfterEos, maxFrames: maxFrames) {
            chunks.append(chunk)
        }
        if chunks.isEmpty { return MLXArray.zeros([0]) }
        return chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0)
    }

    public func generateAudioStream(
        state: PocketTTSState?,
        text: String,
        framesAfterEos: Int?,
        maxFrames: Int?
    ) throws -> [MLXArray] {
        var outputs: [MLXArray] = []
        try forEachAudioChunk(state: state, text: text, framesAfterEos: framesAfterEos, maxFrames: maxFrames) {
            outputs.append($0)
        }
        return outputs
    }

    private func forEachAudioChunk(
        state: PocketTTSState?, text: String, framesAfterEos: Int?, maxFrames: Int?,
        onChunk: (MLXArray) -> Void
    ) throws {
        guard var state else {
            throw NSError(domain: "PocketTTSModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing generation state"])
        }
        try validateSampling()
        let promptNumFrames = getFlowCacheNumFrames(state)
        let chunks = try PocketTTSTextUtils.splitIntoBestSentences(flow_lm.conditioner.tokenizer, text)
        for chunk in chunks {
            try Task.checkCancellation()
            sliceFlowCache(&state, to: promptNumFrames)
            let (_, guess) = try PocketTTSTextUtils.prepareTextPrompt(chunk)
            let frames = framesAfterEos ?? (guess + 2)
            try generateAudioStreamShortText(
                state: &state, text: chunk, framesAfterEos: frames, maxFrames: maxFrames, onChunk: onChunk)
        }
    }

    private func generateAudioStreamShortText(
        state: inout PocketTTSState,
        text: String,
        framesAfterEos: Int,
        maxFrames: Int?,
        onChunk: (MLXArray) -> Void
    ) throws {
        mimi.resetState()

        let words = text.split(separator: " ").count
        let genLenSec = Double(words) * 1.0 + 2.0
        let computedMax = Int(genLenSec * mimi.frameRate)
        let maxGenLen = max(0, maxFrames.map { min($0, computedMax) } ?? computedMax)

        let prepared = flow_lm.conditioner.prepare(text)
        _ = runFlowLMAndIncrementStep(&state, textTokens: prepared.tokens)

        var backboneInput = MLXArray.ones([1, 1, flow_lm.ldim]) * MLXArray(Float.nan)
        var eosStep: Int?

        for step in 0 ..< maxGenLen {
            try Task.checkCancellation()
            let (nextLatent, isEos) = runFlowLMAndIncrementStep(&state, backboneInputLatents: backboneInput)!
            try Task.checkCancellation()
            if eosStep == nil {
                let eos = isEos.asArray(Bool.self).first ?? false
                if eos { eosStep = step }
            }
            if let eosStep, step >= eosStep + framesAfterEos {
                break
            }

            let decodingInput = nextLatent * flow_lm.emb_std + flow_lm.emb_mean
            let quantized = mimi.quantizer(decodingInput.transposed(0, 2, 1))
            let audioChunk = mimi.decodeStep(quantized)
            let chunk = audioChunk[0, 0]
            eval(chunk)
            onChunk(chunk)
            backboneInput = nextLatent
        }

        try Task.checkCancellation()

    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: sending MLXArray? = nil,
        refText: String? = nil,
        language: String? = nil,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        _ = refText
        _ = language

        let prevTemp = temp
        temp = generationParameters.temperature
        defer { temp = prevTemp }
        try validateSampling()
        let prompt = try await resolveAudioPrompt(voice: voice, refAudio: refAudio)
        let state = try getStateForAudioPrompt(prompt)

        return try generateAudio(state: state, text: text, framesAfterEos: nil, maxFrames: generationParameters.maxTokens)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
            generationParameters: generationParameters, streamingInterval: 2.0)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        let refAudio = SendingBox(refAudio)
        let task = Task { @Sendable [weak self] in
            let refAudio = refAudio.take()
            guard let self else { continuation.finish(); return }
            do {
                guard streamingInterval.isFinite else { throw Self.modelError("Streaming interval must be finite") }
                let previousTemperature = self.temp
                self.temp = generationParameters.temperature
                defer { self.temp = previousTemperature }
                try self.validateSampling()
                let prompt = try await self.resolveAudioPrompt(voice: voice, refAudio: refAudio)
                let state = try self.getStateForAudioPrompt(prompt)
                let intervalSamples = max(1, Int(max(0, streamingInterval) * Double(self.sampleRate)))
                var buffer: [MLXArray] = []
                var bufferedSamples = 0
                func flush() {
                    guard !buffer.isEmpty else { return }
                    let audio = buffer.count == 1 ? buffer[0] : concatenated(buffer, axis: 0)
                    buffer.removeAll(keepingCapacity: true)
                    bufferedSamples = 0
                    continuation.yield(.audio(Self.evaluatedWaveform(audio)))
                }
                try self.forEachAudioChunk(
                    state: state, text: text, framesAfterEos: nil, maxFrames: generationParameters.maxTokens
                ) { chunk in
                    buffer.append(chunk)
                    bufferedSamples += chunk.size
                    if bufferedSamples >= intervalSamples { flush() }
                }
                flush()
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }

        return stream
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> PocketTTSModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "PocketTTSModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.model", "voices/*.safetensors", "embeddings/*.safetensors"],
            hfToken: hfToken,
            cache: cache
        )

        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> PocketTTSModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try PocketTTSModelConfig.load(from: configURL)

        let rawWeights = try loadPocketTTSWeights(modelDir: modelDir, weightsName: config.weightsName)
        let hasEncoder = config.mimi.encoderEnabled && rawWeights.keys.contains {
            $0.replacingOccurrences(of: "mimi.model.", with: "mimi.").hasPrefix("mimi.encoder.")
        }
        let model = try await PocketTTSModel.fromConfig(config, modelFolder: modelDir, encoderEnabled: hasEncoder)
        let weights = model.sanitize(weights: rawWeights)

        // Keep native MLX packed matrices quantized, including mixed precision
        // checkpoints. Only modules with scale tensors are replaced.
        var metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        metadata["model_type"] = config.modelType
        metadata["quantization"] = metadata["quantization"] ?? metadata["quantization_config"]
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: JSONSerialization.data(withJSONObject: metadata))
        if let quantization = base.perLayerQuantization {
            quantize(model: model) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                return quantization.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.flow_lm.cacheConditioning()
        eval(model)
        return model
    }
}

private func loadPocketTTSWeights(modelDir: URL, weightsName: String?) throws -> [String: MLXArray] {
    for name in [weightsName, "model.safetensors"].compactMap({ $0 }) {
        let file = modelDir.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: file.path) { return try MLX.loadArrays(url: file) }
    }
    let shards = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        .filter { $0.lastPathComponent.hasPrefix("model-") && $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !shards.isEmpty else {
        throw NSError(domain: "PocketTTSModel", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "Model safetensors not found in \(modelDir.path)"])
    }
    var weights: [String: MLXArray] = [:]
    for shard in shards { weights.merge(try MLX.loadArrays(url: shard)) { _, new in new } }
    return weights
}
