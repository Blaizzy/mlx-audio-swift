//
//  ChatterboxTurboTTS.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon
import HuggingFace
import Tokenizers
import MLXAudioCore

public typealias ChatterboxTurboError = AudioGenerationError
public typealias ChatterboxTurboGeneration = AudioGeneration
public typealias ChatterboxTurboGenerationInfo = AudioGenerationInfo

struct ChatterboxTurboConditionals {
    let t3: T3Cond
    let gen: S3GenReference
}

public final class ChatterboxTurboTTS: Module {
    public static let repoId = "ResembleAI/chatterbox-turbo"
    public static let s3TokenizerRepo = "mlx-community/S3TokenizerV2"

    public let sampleRate: Int = S3GenSampleRate

    private let encCondLen = 15 * S3SampleRate
    private let decCondLen = 10 * S3GenSampleRate

    @ModuleInfo(key: "t3") private var t3: T3
    @ModuleInfo(key: "s3gen") private var s3gen: S3Gen
    @ModuleInfo(key: "ve") private var voiceEncoder: VoiceEncoder

    public var tokenizer: Tokenizer?
    private var s3Tokenizer: S3TokenizerV2?
    private var conds: ChatterboxTurboConditionals?
    private var localPath: URL?

    public override init() {
        self._t3.wrappedValue = T3(.turbo())
        self._s3gen.wrappedValue = S3Gen(meanflow: true)
        self._voiceEncoder.wrappedValue = VoiceEncoder()
        super.init()
    }

    init(t3: T3, s3gen: S3Gen, voiceEncoder: VoiceEncoder) {
        self._t3.wrappedValue = t3
        self._s3gen.wrappedValue = s3gen
        self._voiceEncoder.wrappedValue = voiceEncoder
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var s3genWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("gen.") {
                continue
            }
            if key.hasPrefix("s3gen.") {
                let stripped = String(key.dropFirst(6))
                s3genWeights[stripped] = value
                continue
            }
            sanitized[key] = value
        }

        if !s3genWeights.isEmpty {
            let cleaned = s3gen.sanitize(s3genWeights)
            for (key, value) in cleaned {
                sanitized["s3gen.\(key)"] = value
            }
        }

        return sanitized
    }

    public func postLoadHook(modelDir: URL, client: HubClient) async throws {
        localPath = modelDir

        if tokenizer == nil {
            tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        }

        if s3Tokenizer == nil {
            let s3Repo = Repo.ID(rawValue: Self.s3TokenizerRepo)!
            let cache = client.cache ?? HubCache.default
            let s3Dir = try await resolveOrDownloadChatterboxModel(
                client: client,
                cache: cache,
                repoID: s3Repo,
                requiredExtension: "safetensors"
            )
            let s3WeightsPath = s3Dir.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: s3WeightsPath.path) {
                let weights = try MLX.loadArrays(url: s3WeightsPath)
                let tokenizer = S3TokenizerV2(name: "speech_tokenizer_v2_25hz")
                let sanitized = tokenizer.sanitize(weights: weights)
                try tokenizer.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
                eval(tokenizer)
                s3Tokenizer = tokenizer
            }
        }

        let condsPath = modelDir.appendingPathComponent("conds.safetensors")
        if FileManager.default.fileExists(atPath: condsPath.path) {
            let condsData = try MLX.loadArrays(url: condsPath)
            let speakerEmb = condsData["t3.speaker_emb"] ?? MLXArray.zeros([1, 256], type: Float.self)
            let condTokens = condsData["t3.cond_prompt_speech_tokens"]
            let t3Cond = T3Cond(
                speakerEmb: speakerEmb,
                clapEmb: nil,
                condPromptSpeechTokens: condTokens,
                condPromptSpeechEmb: nil,
                emotionAdv: nil
            )

            var genMap: [String: MLXArray] = [:]
            for (key, value) in condsData where key.hasPrefix("gen.") {
                genMap[String(key.dropFirst(4))] = value
            }

            guard
                let promptToken = genMap["prompt_token"],
                let promptTokenLen = genMap["prompt_token_len"],
                let promptFeat = genMap["prompt_feat"],
                let promptFeatLen = genMap["prompt_feat_len"],
                let embedding = genMap["embedding"]
            else {
                throw ChatterboxTurboError.invalidInput("conds.safetensors missing S3Gen reference keys")
            }

            let genRef = S3GenReference(
                promptToken: promptToken,
                promptTokenLen: promptTokenLen,
                promptFeat: promptFeat,
                promptFeatLen: promptFeatLen,
                embedding: embedding
            )

            conds = ChatterboxTurboConditionals(t3: t3Cond, gen: genRef)
        }
    }

    public static func fromPretrained(_ modelRepo: String = repoId) async throws -> ChatterboxTurboTTS {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            client = HubClient(host: HubClient.defaultHost, bearerToken: token)
        } else {
            client = HubClient.default
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw ChatterboxTurboError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let cache = client.cache ?? HubCache.default
        let modelDir = try await resolveOrDownloadChatterboxModel(
            client: client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors"
        )

        let model = ChatterboxTurboTTS()

        let weights = try loadChatterboxWeights(from: modelDir)
        let sanitized = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)

        try await model.postLoadHook(modelDir: modelDir, client: client)
        return model
    }

    public func prepareConditionals(
        refAudio: MLXArray,
        sampleRate: Int,
        exaggeration: Float = 0.0,
        normLoudness: Bool = true
    ) throws {
        var refWav24k = refAudio.asArray(Float.self)
        if sampleRate != S3GenSampleRate {
            refWav24k = s3ResampleLinear(refWav24k, from: sampleRate, to: S3GenSampleRate)
        }

        if Double(refWav24k.count) / Double(S3GenSampleRate) <= 5.0 {
            throw ChatterboxTurboError.invalidInput("Audio prompt must be longer than 5 seconds.")
        }

        if normLoudness {
            refWav24k = simpleRmsNormalize(refWav24k, targetDb: -27.0)
        }

        let refWav16k = s3ResampleLinear(refWav24k, from: S3GenSampleRate, to: S3SampleRate)
        let refWav24kTrimmed = Array(refWav24k.prefix(decCondLen))

        let (genRef, t3Tokens) = extractConditionals(
            refWav24k: refWav24kTrimmed,
            refWav16k: refWav16k
        )

        let veInput = Array(refWav16k.prefix(encCondLen))
        var veEmb = voiceEncoder.embedsFromWavs([veInput], sampleRate: S3SampleRate)
        veEmb = MLX.mean(veEmb, axis: 0, keepDims: true)

        let t3Cond = T3Cond(
            speakerEmb: veEmb,
            clapEmb: nil,
            condPromptSpeechTokens: t3Tokens,
            condPromptSpeechEmb: nil,
            emotionAdv: t3.hp.emotionAdv ? MLXArray([exaggeration]).reshaped([1, 1, 1]) : nil
        )

        conds = ChatterboxTurboConditionals(t3: t3Cond, gen: genRef)
    }

    public func generate(
        text: String,
        repetitionPenalty: Float = 1.2,
        topP: Float = 0.95,
        temperature: Float = 0.8,
        topK: Int = 1000,
        refAudio: MLXArray? = nil,
        sampleRate: Int? = nil,
        normLoudness: Bool = true,
        splitPattern: String? = "(?<=[.!?])\\s+",
        maxTokens: Int = 800
    ) throws -> MLXArray {
        if let refAudio, let sampleRate {
            try prepareConditionals(refAudio: refAudio, sampleRate: sampleRate, normLoudness: normLoudness)
        }

        guard let conds else {
            throw ChatterboxTurboError.modelNotInitialized("Conditionals not prepared")
        }

        let normalized = ChatterboxTurboTextUtils.puncNorm(text)
        let chunks = splitText(normalized, splitPattern: splitPattern, maxTokens: maxTokens)

        var audioChunks: [MLXArray] = []
        audioChunks.reserveCapacity(chunks.count)

        for chunk in chunks {
            let textTokens = tokenize(chunk)
            var speechTokens = t3.inferenceTurbo(
                cond: conds.t3,
                textTokens: textTokens,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxGenLen: maxTokens
            )

            speechTokens = speechTokens.reshaped([-1])
            let filtered = dropInvalidTokens(speechTokens)
            let silence = MLXArray([Int32(S3GenSilenceToken), Int32(S3GenSilenceToken), Int32(S3GenSilenceToken)])
            let combined = MLX.concatenated([filtered, silence], axis: 0).expandedDimensions(axis: 0)

            let (wav, _) = s3gen.inference(
                speechTokens: combined,
                refDict: conds.gen,
                nCfmTimesteps: 2
            )

            let flat = wav.ndim == 2 ? wav.squeezed(axis: 0) : wav
            audioChunks.append(flat)
            Memory.clearCache()
        }

        return MLX.concatenated(audioChunks, axis: 0)
    }

    public func generateStream(
        text: String,
        repetitionPenalty: Float = 1.2,
        topP: Float = 0.95,
        temperature: Float = 0.8,
        topK: Int = 1000,
        refAudio: MLXArray? = nil,
        sampleRate: Int? = nil,
        normLoudness: Bool = true,
        chunkSize: Int = 40,
        splitPattern: String? = "(?<=[.!?])\\s+",
        maxTokens: Int = 800
    ) -> AsyncThrowingStream<ChatterboxTurboGeneration, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    if let refAudio, let sampleRate {
                        try prepareConditionals(refAudio: refAudio, sampleRate: sampleRate, normLoudness: normLoudness)
                    }

                    guard let conds else {
                        throw ChatterboxTurboError.modelNotInitialized("Conditionals not prepared")
                    }

                    let normalized = ChatterboxTurboTextUtils.puncNorm(text)
                    let chunks = splitText(normalized, splitPattern: splitPattern, maxTokens: maxTokens)

                    for (chunkIndex, chunk) in chunks.enumerated() {
                        let isLastChunk = chunkIndex == chunks.count - 1
                        let textTokens = tokenize(chunk)

                        var accumulated = MLXArray.zeros([1, 0], type: Int32.self)
                        var prevSamples = 0

                        let stream = t3.inferenceTurboStream(
                            cond: conds.t3,
                            textTokens: textTokens,
                            temperature: temperature,
                            topK: topK,
                            topP: topP,
                            repetitionPenalty: repetitionPenalty,
                            maxGenLen: maxTokens,
                            chunkSize: chunkSize
                        )

                        for await (chunkTokens, isDone) in stream {
                            accumulated = MLX.concatenated([accumulated, chunkTokens], axis: 1)
                            let (audioChunk, totalSamples) = s3gen.inferenceStream(
                                speechTokens: accumulated,
                                refDict: conds.gen,
                                nCfmTimesteps: 2,
                                prevAudioSamples: prevSamples,
                                isFinal: isDone && isLastChunk
                            )
                            prevSamples = totalSamples
                            if audioChunk.shape[1] > 0 {
                                let flat = audioChunk.ndim == 2 ? audioChunk.squeezed(axis: 0) : audioChunk
                                continuation.yield(.audio(flat))
                            }
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func extractConditionals(
        refWav24k: [Float],
        refWav16k: [Float]
    ) -> (S3GenReference, MLXArray?) {
        if let s3Tokenizer {
            let maxLen16k = Int(Double(decCondLen) * Double(S3SampleRate) / Double(S3GenSampleRate))
            let s3Gen16k = Array(refWav16k.prefix(maxLen16k))
            let s3GenMel = s3LogMelSpectrogram(MLXArray(s3Gen16k))
            let s3GenMelBatch = s3GenMel.expandedDimensions(axis: 0)
            let s3GenMelLen = MLXArray([Int32(s3GenMel.shape[1])])
            let (s3Tokens, s3TokenLens) = s3Tokenizer(s3GenMelBatch, s3GenMelLen)

            let genRef = s3gen.embedRef(
                refWav: MLXArray(refWav24k).expandedDimensions(axis: 0),
                refSr: S3GenSampleRate,
                refSpeechTokens: s3Tokens,
                refSpeechTokenLens: s3TokenLens
            )

            let t3Wav = Array(refWav16k.prefix(encCondLen))
            let t3Mel = s3LogMelSpectrogram(MLXArray(t3Wav))
            let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
            let t3MelLen = MLXArray([Int32(t3Mel.shape[1])])
            let (t3Tokens, _) = s3Tokenizer(t3MelBatch, t3MelLen)

            let plen = t3.hp.speechCondPromptLen
            let clamped = min(plen, t3Tokens.shape[1])
            let t3Prompt = t3Tokens[0..., 0..<clamped]

            return (genRef, t3Prompt)
        }

        let genRef = s3gen.embedRef(
            refWav: MLXArray(refWav24k).expandedDimensions(axis: 0),
            refSr: S3GenSampleRate
        )
        let plen = t3.hp.speechCondPromptLen
        let t3Prompt = plen > 0 ? MLXArray.zeros([1, plen], type: Int32.self) : nil
        return (genRef, t3Prompt)
    }

    private func tokenize(_ text: String) -> MLXArray {
        if let tokenizer {
            let encoded = tokenizer.encode(text: text)
            let ids = encoded.map { Int32($0) }
            return MLXArray(ids).expandedDimensions(axis: 0)
        }
        let fallback = text.utf8.map { Int32($0) }
        return MLXArray(fallback).expandedDimensions(axis: 0)
    }

    private func splitText(_ text: String, splitPattern: String?, maxTokens: Int) -> [String] {
        let maxCharsPerChunk = (maxTokens / 8) * 4
        guard let splitPattern else { return [text] }

        let regex = try? NSRegularExpression(pattern: splitPattern, options: [])
        let range = NSRange(text.startIndex..., in: text)
        let sentences: [String]

        if let regex {
            let separated = regex.stringByReplacingMatches(in: text, range: range, withTemplate: "\u{0000}")
            sentences = separated
                .components(separatedBy: "\u{0000}")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        } else {
            sentences = [text]
        }

        var chunks: [String] = []
        var current = ""

        for sentence in sentences {
            if current.isEmpty {
                current = sentence
                continue
            }

            if current.count + sentence.count + 1 > maxCharsPerChunk {
                chunks.append(current)
                current = sentence
            } else {
                current += " " + sentence
            }
        }

        if !current.isEmpty {
            chunks.append(current)
        }

        return chunks.isEmpty ? [text] : chunks
    }

    private func dropInvalidTokens(_ tokens: MLXArray) -> MLXArray {
        let data = tokens.asArray(Int32.self)
        let filtered = data.filter { Int($0) < S3SpeechVocabSize }
        return MLXArray(filtered)
    }

    private func simpleRmsNormalize(_ wav: [Float], targetDb: Float) -> [Float] {
        guard !wav.isEmpty else { return wav }
        let rms = sqrt(wav.reduce(0) { $0 + $1 * $1 } / Float(wav.count))
        guard rms > 0 else { return wav }
        let target = powf(10.0, targetDb / 20.0)
        let gain = target / rms
        guard gain.isFinite, gain > 0 else { return wav }
        return wav.map { $0 * gain }
    }
}

private func loadChatterboxWeights(from directory: URL) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

private func resolveOrDownloadChatterboxModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID,
    requiredExtension: String
) async throws -> URL {
    if let cachedConfig = cache.cachedFilePath(
        repo: repoID,
        kind: .model,
        revision: "main",
        filename: "config.json"
    ) {
        let modelDir = cachedConfig.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: modelDir.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let hasRequiredFiles = files?.contains { $0.pathExtension == requiredExtension } ?? false
            if hasRequiredFiles {
                return modelDir
            }
        }
    }

    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasRequiredFiles = files?.contains { $0.pathExtension == requiredExtension } ?? false
        if hasRequiredFiles {
            return modelDir
        }
    }

    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    _ = try await client.downloadSnapshot(of: repoID, kind: .model, to: modelDir, revision: "main")
    return modelDir
}
