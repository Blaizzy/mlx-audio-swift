import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import MLXRandom

public enum DiaError: Error {
    case invalidRepo(String)
    case noAudio
}

public final class DiaTTSModel: SpeechGenerationModel, @unchecked Sendable {
    private let model: DiaModel
    private let dac: DescriptDAC
    private let config: DiaConfig

    public let sampleRate: Int

    private let cfgScale: Float = 3.0
    private let cfgFilterTopK: Int = 35

    init(model: DiaModel, dac: DescriptDAC, config: DiaConfig) {
        self.model = model
        self.dac = dac
        self.config = config
        self.sampleRate = config.model.sampleRate
    }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 3072, temperature: 1.3, topP: 0.95)
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> DiaTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else { throw DiaError.invalidRepo(modelRepo) }
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.json"], cache: cache)

        let config = try JSONDecoder().decode(
            DiaConfig.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let model = DiaModel(config)
        let weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)

        let dac = try await DescriptDAC.fromPretrained("mlx-community/descript-audio-codec-44khz", cache: cache)
        eval(model)
        return DiaTTSModel(model: model, dac: dac, config: config)
    }

    private func createAttnMask(_ qPad: MLXArray, _ kPad: MLXArray, causal: Bool) -> MLXArray {
        let pq = qPad.expandedDimensions(axis: 2)
        let pk = kPad.expandedDimensions(axis: 1)
        let bothNonPad = MLX.logicalAnd(pq, pk)
        let bothPad = MLX.logicalAnd(MLX.logicalNot(pq), MLX.logicalNot(pk))
        var mask = MLX.logicalOr(bothNonPad, bothPad)
        if causal {
            let tq = mask.shape[1], tk = mask.shape[2]
            let causal2d = MLX.tril(MLXArray.ones([tq, tk], dtype: .bool))
            mask = MLX.logicalAnd(mask, causal2d)
        }
        return mask.expandedDimensions(axis: 1)
    }

    private func prepareTextInput(_ text: String) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let maxLen = config.data.textLength
        let pad = Int32(config.data.textPadValue)
        let replaced = text.replacingOccurrences(of: "[S1]", with: "\u{01}")
            .replacingOccurrences(of: "[S2]", with: "\u{02}")
        var tokens = Array(replaced.utf8).map { Int32($0) }
        if tokens.count >= maxLen {
            tokens = Array(tokens.prefix(maxLen))
        } else {
            tokens += Array(repeating: pad, count: maxLen - tokens.count)
        }
        let src = MLXArray(tokens).reshaped([1, maxLen])
        let positions = MLXArray((0 ..< maxLen).map { Int32($0) }).reshaped([1, maxLen])
        let paddingMask = MLX.notEqual(src, MLXArray(pad))
        let encMask = createAttnMask(paddingMask, paddingMask, causal: false)
        return (src, positions, paddingMask, encMask)
    }

    private func sampleChannels(_ logits: MLXArray, temperature: Float, topK: Int) -> MLXArray {
        if temperature == 0 { return logits.argMax(axis: -1) }
        var l = logits
        let v = l.shape[l.ndim - 1]
        if topK > 0 && topK < v {
            let sorted = MLX.sorted(l, axis: -1)
            let kth = sorted[0..., (v - topK)].expandedDimensions(axis: -1)
            l = MLX.where(MLX.less(l, kth), MLXArray(-Float.infinity), l)
        }
        return MLXRandom.categorical(l * (1.0 / temperature), axis: -1)
    }

    private func generateCodes(_ text: String, temperature: Float, topK: Int, maxTokens: Int) -> MLXArray {
        let numChannels = config.data.channels
        let bos = Int32(config.data.audioBosValue)
        let eos = Int32(config.data.audioEosValue)
        let padV = Int32(config.data.audioPadValue)
        let delay = config.data.delayPattern
        let maxDelay = delay.max() ?? 0
        let extraStepsAfterEos = 30

        let (condSrc, condPos, condPad, condEncMask) = prepareTextInput(text)
        let uncondSrc = MLXArray.zeros(like: condSrc)
        let src = MLX.concatenated([uncondSrc, condSrc], axis: 0)
        let positions = MLX.concatenated([condPos, condPos], axis: 0)
        let srcPad = MLX.concatenated([condPad, condPad], axis: 0)
        let encMask = MLX.concatenated([condEncMask, condEncMask], axis: 0)

        let encoderOut = model.encoder(src, srcPositions: positions, attnMask: encMask)
        let crossCache = model.decoder.precomputeCrossAttentionKV(encoderOut: encoderOut, srcPositions: positions)
        let selfCache = (0 ..< model.decoder.numLayers).map { _ in DiaKVCache() }

        var frames = [[Int32]]()
        frames.append([Int32](repeating: bos, count: numChannels))

        let tgtPad = MLXArray.ones([2, 1], dtype: .bool)
        let crossAttnMask = createAttnMask(tgtPad, srcPad, causal: false)

        var eosDetected = false
        var eosCountdown = -1
        var lastStep = 0

        for step in 0 ..< maxTokens {
            lastStep = step
            let currentFrame = frames[step]
            let inputFrame = MLXArray(currentFrame + currentFrame).reshaped([2, 1, numChannels])
            let tgtPos = MLXArray([Int32(step), Int32(step)]).reshaped([2, 1])

            let logits = model.decoder.decodeStep(
                inputFrame, tgtPos: tgtPos, encoderOut: encoderOut, crossAttnMask: crossAttnMask,
                selfAttentionCache: selfCache, crossAttentionCache: crossCache)

            let last = logits[0..., -1, 0..., 0...]
            let uncond = last[0]
            let cond = last[1]
            var cfg = cond + cfgScale * (cond - uncond)
            let v = cfg.shape[1]
            cfg = MLX.concatenated(
                [cfg[0..., 0 ..< 1025], MLXArray.full([numChannels, v - 1025], values: MLXArray(-Float.infinity))],
                axis: 1)

            let predArr = sampleChannels(cfg, temperature: temperature, topK: topK)
            eval(predArr)
            var frame = predArr.asArray(Int32.self)

            for c in 0 ..< numChannels where step < delay[c] { frame[c] = bos }

            if !eosDetected && frame[0] == eos {
                eosDetected = true
                eosCountdown = extraStepsAfterEos
            }
            if eosCountdown > 0 {
                let stepAfterEos = maxDelay - eosCountdown
                for (i, d) in delay.enumerated() {
                    if stepAfterEos == d { frame[i] = eos } else if stepAfterEos > d { frame[i] = padV }
                }
                eosCountdown -= 1
            }

            frames.append(frame)
            if eosCountdown == 0 { break }
        }

        let outCount = max(lastStep, 0)
        var flat = [Int32]()
        flat.reserveCapacity(outCount * numChannels)
        for c in 0 ..< numChannels {
            for t in 0 ..< outCount { flat.append(frames[1 + t][c]) }
        }
        return MLXArray(flat).reshaped([numChannels, outCount])
    }

    public func generate(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let temperature = Float(generationParameters.temperature)
        let requested = generationParameters.maxTokens ?? config.data.audioLength
        let maxTokens = max(1, min(requested, config.data.audioLength))
        let codes = generateCodes(text, temperature: temperature, topK: cfgFilterTopK, maxTokens: maxTokens)
        let audio = diaCodebookToAudio(
            codes, dac: dac, delayPattern: config.data.delayPattern,
            maxT: config.data.audioLength, channels: config.data.channels)
        eval(audio)
        return audio.squeezed()
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language, generationParameters: generationParameters, streamingInterval: 2.0)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters, streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { continuation.finish(); return }
            do {
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio, refText: refText,
                    language: language, generationParameters: generationParameters)
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }
}
