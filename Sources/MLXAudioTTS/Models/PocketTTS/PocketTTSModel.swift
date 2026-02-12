import Foundation
import Hub
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN

private let defaultTemperature: Float = 0.7
private let defaultLsdDecodeSteps: Int = 1
private let defaultNoiseClamp: Float? = nil
private let defaultEosThreshold: Float = -4.0
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

    public var temp: Float = defaultTemperature
    public var lsd_decode_steps: Int = defaultLsdDecodeSteps
    public var noise_clamp: Float? = defaultNoiseClamp
    public var eos_threshold: Float = defaultEosThreshold

    public var sampleRate: Int { config.mimi.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: defaultTemperature)
    }

    private init(config: PocketTTSModelConfig, modelFolder: URL, flowLM: FlowLMModel, mimi: MimiAdapter) {
        self.config = config
        self.modelFolder = modelFolder
        self._flow_lm = ModuleInfo(wrappedValue: flowLM)
        self._mimi = ModuleInfo(wrappedValue: mimi)
        self.speaker_proj_weight = MLXArray.zeros([config.flowLM.transformer.dModel, config.mimi.quantizer.outputDimension])
        super.init()
    }

    public static func fromConfig(_ config: PocketTTSModelConfig, modelFolder: URL) async throws -> PocketTTSModel {
        let flowLM = try await FlowLMModel.fromConfig(
            config.flowLM,
            latentDim: config.mimi.quantizer.dimension,
            modelFolder: modelFolder
        )
        let mimi = MimiAdapter.fromConfig(config.mimi)
        return PocketTTSModel(config: config, modelFolder: modelFolder, flowLM: flowLM, mimi: mimi)
    }

    public func initState() -> PocketTTSState {
        PocketTTSState(flowCache: flow_lm.makeCache())
    }

    private func runFlowLM(
        _ state: inout PocketTTSState,
        textTokens: MLXArray,
        backboneInputLatents: MLXArray,
        audioConditioning: MLXArray
    ) -> (MLXArray, MLXArray) {
        let textEmb = flow_lm.conditioner(TokenizedText(tokens: textTokens))
        let combined = concatenated([textEmb, audioConditioning], axis: 1)
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
    ) -> (MLXArray, MLXArray) {
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
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXArray? {
        _ = progressHandler
        let fileURL = modelFolder.appendingPathComponent("embeddings/\(voiceName).safetensors")
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            return nil
        }
        let arrays = try MLX.loadArrays(url: fileURL)
        guard let prompt = arrays["audio_prompt"] else {
            return nil
        }
        return prompt
    }

    private func resolveAudioPrompt(
        voice: String?,
        refAudio: MLXArray?,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> AudioPrompt {
        if let refAudio {
            return .audio(normalizeAudio(refAudio))
        }

        let voice = voice?.lowercased() ?? defaultAudioPrompt
        guard let emb = try await Self.loadPredefinedVoice(
            voice,
            modelFolder: modelFolder,
            progressHandler: progressHandler
        ) else {
            throw NSError(domain: "PocketTTSModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing audio prompt for voice: \(voice)"])
        }
        return .embedding(emb)
    }

    private func getStateForAudioPrompt(_ prompt: AudioPrompt) -> PocketTTSState {
        var state = initState()
        let conditioning: MLXArray = switch prompt {
        case .embedding(let emb):
            emb
        case .audio(let audio):
            encodeAudio(audio)
        }

        _ = runFlowLMAndIncrementStep(&state, audioConditioning: conditioning)
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
        guard var state else {
            throw NSError(domain: "PocketTTSModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing generation state"])
        }
        var outputs: [MLXArray] = []
        let promptNumFrames = getFlowCacheNumFrames(state)
        let chunks = try PocketTTSTextUtils.splitIntoBestSentences(flow_lm.conditioner.tokenizer, text)
        for chunk in chunks {
            sliceFlowCache(&state, to: promptNumFrames)
            let (_, guess) = try PocketTTSTextUtils.prepareTextPrompt(chunk)
            let frames = framesAfterEos ?? (guess + 2)
            let audioChunks = try generateAudioStreamShortText(state: &state, text: chunk, framesAfterEos: frames, maxFrames: maxFrames)
            outputs.append(contentsOf: audioChunks)
        }
        return outputs
    }

    private func generateAudioStreamShortText(
        state: inout PocketTTSState,
        text: String,
        framesAfterEos: Int,
        maxFrames: Int?
    ) throws -> [MLXArray] {
        mimi.resetState()
        var outputs: [MLXArray] = []

        let words = text.split(separator: " ").count
        let genLenSec = Double(words) * 1.0 + 2.0
        let computedMax = Int(genLenSec * mimi.frameRate)
        let maxGenLen = maxFrames.map { min($0, computedMax) } ?? computedMax

        let prepared = flow_lm.conditioner.prepare(text)
        _ = runFlowLMAndIncrementStep(&state, textTokens: prepared.tokens)

        var backboneInput = MLXArray.ones([1, 1, flow_lm.ldim]) * MLXArray(Float.nan)
        var eosStep: Int?

        for step in 0 ..< maxGenLen {
            let (nextLatent, isEos) = runFlowLMAndIncrementStep(&state, backboneInputLatents: backboneInput)
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
            outputs.append(audioChunk.squeezed())
            backboneInput = nextLatent
        }

        return outputs
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        language: String? = nil,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = refText
        _ = language

        let prompt = try await resolveAudioPrompt(voice: voice, refAudio: refAudio)
        let state = getStateForAudioPrompt(prompt)

        let prevTemp = temp
        let prevLsd = lsd_decode_steps
        let prevNoise = noise_clamp
        let prevEos = eos_threshold

        temp = generationParameters.temperature

        defer {
            temp = prevTemp
            lsd_decode_steps = prevLsd
            noise_clamp = prevNoise
            eos_threshold = prevEos
        }

        return try generateAudio(state: state, text: text, framesAfterEos: nil, maxFrames: generationParameters.maxTokens)
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

        Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    // MARK: - Loading

    public static func fromPretrained(_ modelRepo: String) async throws -> PocketTTSModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let client = if let token = hfToken, !token.isEmpty {
            HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            HubClient.default
        }
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "PocketTTSModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        let modelDir = try await resolveOrDownloadPocketTTSModel(client: client, cache: cache, repoID: repoID)
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try PocketTTSModelConfig.load(from: configURL)

        let model = try await PocketTTSModel.fromConfig(config, modelFolder: modelDir)
        let weights = try await loadPocketTTSWeights(modelDir: modelDir)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])

        eval(model)
        return model
    }
}

private func resolveOrDownloadPocketTTSModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID
) async throws -> URL {
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasConfig = files?.contains { $0.lastPathComponent == "config.json" } ?? false
        let hasWeights = files?.contains { $0.pathExtension == "safetensors" } ?? false
        if hasConfig, hasWeights {
            return modelDir
        }
    }

    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    _ = try await client.downloadSnapshot(
        of: repoID,
        kind: .model,
        to: modelDir,
        revision: "main",
        matching: ["*.json", "*.safetensors", "embeddings/*.safetensors"],
        progressHandler: { progress in
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        }
    )
    return modelDir
}

private func loadPocketTTSWeights(modelDir: URL) async throws -> [String: MLXArray] {
    let weightsURL = modelDir.appendingPathComponent("model.safetensors")
    if !FileManager.default.fileExists(atPath: weightsURL.path) {
        throw NSError(
            domain: "PocketTTSModel",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "model.safetensors not found at \(weightsURL.path)"]
        )
    }
    return try MLX.loadArrays(url: weightsURL)
}
