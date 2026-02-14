//
//  VoxtralRealtime.swift
//  MLXAudioSTT
//

import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace

// MARK: - Constants

private let voxtralSampleRate = 16000
private let frameRate: Float = 12.5
private let rawAudioLengthPerTok = Int(Float(voxtralSampleRate) / frameRate)  // 1280
private let voxtralHopLength = 160
private let audioLengthPerTok = rawAudioLengthPerTok / voxtralHopLength  // 8

// MARK: - Streaming Helpers

private func numAudioTokens(_ audioLen: Int) -> Int {
    var len = audioLen
    if len % voxtralHopLength != 0 {
        len = Int(ceil(Double(len) / Double(voxtralHopLength) - 1))
    } else {
        len = len / voxtralHopLength
    }
    return Int(ceil(Double(len) / Double(audioLengthPerTok)))
}

private func numDelayTokens(_ delayMs: Int) -> Int {
    let delaySamples = Int(Double(delayMs) / 1000.0 * Double(voxtralSampleRate))
    return numAudioTokens(delaySamples)
}

private func padAudioStreaming(_ audio: [Float], nLeftPad: Int, nRightPad: Int) -> [Float] {
    let leftPad = nLeftPad * rawAudioLengthPerTok
    let alignPad = (rawAudioLengthPerTok - (audio.count % rawAudioLengthPerTok)) % rawAudioLengthPerTok
    let rightPad = alignPad + nRightPad * rawAudioLengthPerTok
    var padded = [Float](repeating: 0, count: leftPad)
    padded.append(contentsOf: audio)
    padded.append(contentsOf: [Float](repeating: 0, count: rightPad))
    return padded
}

// MARK: - Voxtral Mel Spectrogram

/// Compute mel spectrogram matching vLLM/Voxtral: periodic Hann, Slaney mel, drop last frame.
private func voxtralMelSpectrogram(
    audio: MLXArray, melFiltersBank: MLXArray,
    windowSize: Int, hopLen: Int, globalLogMelMax: Float
) -> MLXArray {
    // Periodic Hann window (divide by N, not N-1)
    let n = MLX.arange(windowSize).asType(.float32)
    let window = 0.5 * (1.0 - MLX.cos(2.0 * Float.pi * n / Float(windowSize)))

    let padSize = windowSize / 2
    let audioFlat = audio.flattened()
    let audioLen = audioFlat.dim(0)
    let prefixIndices = MLX.arange(padSize, 0, step: -1)
    let prefix = audioFlat[prefixIndices]
    let suffixIndices = MLX.arange(audioLen - 2, audioLen - padSize - 2, step: -1)
    let suffix = audioFlat[suffixIndices]
    let padded = MLX.concatenated([prefix, audioFlat, suffix])

    let nSamples = padded.dim(0)
    let nFrames = 1 + (nSamples - windowSize) / hopLen
    let frameOffsets = MLX.arange(nFrames) * hopLen
    let windowIndices = MLX.arange(windowSize)
    let indices = windowIndices[.newAxis, 0...] + frameOffsets[0..., .newAxis]
    let frames = padded[indices] * window[.newAxis, 0...]

    let spectrum = MLXFFT.rfft(frames, n: windowSize, axis: -1)
    let magnitudes = MLX.abs(spectrum).square()
    let magDropped = magnitudes[0..<(nFrames - 1), 0...].T
    let melSpec = MLX.matmul(melFiltersBank.T, magDropped)

    var logSpec = MLX.log10(MLX.maximum(melSpec, MLXArray(Float(1e-10))))
    let minVal = globalLogMelMax - 8.0
    logSpec = MLX.maximum(logSpec, MLXArray(minVal))
    logSpec = (logSpec + 4.0) / 4.0

    return logSpec
}

// MARK: - Voxtral Realtime Model

public class VoxtralRealtimeModel: Module {
    public let config: VoxtralRealtimeModelConfig

    @ModuleInfo(key: "encoder") var encoder: VoxtralAudioEncoder
    @ModuleInfo(key: "decoder") var decoder: VoxtralDecoder

    var tokenizer: TekkenTokenizer?
    var melFiltersBank: MLXArray?
    var adaScaleDelay: Int = -1

    public init(_ config: VoxtralRealtimeModelConfig) {
        self.config = config
        self._encoder.wrappedValue = VoxtralAudioEncoder(config.encoderArgs)
        self._decoder.wrappedValue = VoxtralDecoder(config.decoder)
    }

    // MARK: - Mel Filters

    private func ensureMelFilters() -> MLXArray {
        if let filters = melFiltersBank { return filters }
        let aec = config.audioEncodingArgs
        let filters = MLXAudioCore.melFilters(
            sampleRate: aec.samplingRate,
            nFft: aec.windowSize,
            nMels: aec.numMelBins,
            fMin: 0,
            fMax: 8000,
            norm: "slaney",
            melScale: .slaney
        )
        melFiltersBank = filters
        return filters
    }

    // MARK: - Audio Preprocessing

    private func prepareMel(_ audioArray: [Float], delayMs: Int? = nil) -> (MLXArray, Int) {
        let delay = delayMs ?? config.transcriptionDelayMs
        let nDelay = numDelayTokens(delay)
        let nLeft = config.nLeftPadTokens
        let nRight = (nDelay + 1) + 10

        let padded = padAudioStreaming(audioArray, nLeftPad: nLeft, nRightPad: nRight)

        let aec = config.audioEncodingArgs
        let filters = ensureMelFilters()
        let audioMx = MLXArray(padded)
        var mel = voxtralMelSpectrogram(
            audio: audioMx, melFiltersBank: filters,
            windowSize: aec.windowSize, hopLen: aec.hopLength,
            globalLogMelMax: aec.globalLogMelMax)

        if mel.dim(1) % 2 != 0 {
            mel = mel[0..., 1...]
        }

        return (mel, nDelay)
    }

    // MARK: - Ada Scales

    private func ensureAdaScales(delayMs: Int? = nil) {
        let delay = delayMs ?? config.transcriptionDelayMs
        let nDelay = numDelayTokens(delay)
        guard nDelay != adaScaleDelay else { return }

        let tCond = computeTimeEmbedding(tValue: Float(nDelay), dim: config.decoder.dim)
        decoder.precomputeAdaScales(tCond)
        if let scales = decoder.adaScales {
            for scale in scales {
                if let s = scale { MLX.eval(s) }
            }
        }
        adaScaleDelay = nDelay
    }

    // MARK: - Encode and Prefill

    private func encodeAndPrefill(_ audioArray: [Float], delayMs: Int? = nil) -> (
        MLXArray, Int, Int, MLXArray, [VoxtralKVCache], Date
    ) {
        let startTime = Date()
        ensureAdaScales(delayMs: delayMs)

        let (mel, nDelay) = prepareMel(audioArray, delayMs: delayMs)

        let convOut = encoder.convStem(mel)
        let ds = encoder.config.downsampleFactor
        let nAudioTotal = convOut.dim(0) / ds

        let nLeft = config.nLeftPadTokens
        let promptLen = 1 + nLeft + nDelay

        let adapterOut: MLXArray
        if convOut.dim(0) <= encoder.config.slidingWindow {
            adapterOut = encoder.encodeFull(convOut)
        } else {
            let encoded = encoder.encodeChunks(convOut)
            adapterOut = encoder.downsampleAndProject(encoded)
        }

        var promptIds = [config.bosTokenId]
        promptIds.append(
            contentsOf: [Int](repeating: config.streamingPadTokenId, count: nLeft + nDelay))

        let promptIdsMx = MLXArray(promptIds.map { Int32($0) })
        let promptTextEmbeds = decoder.embedTokens(promptIdsMx)
        let prefixEmbeds = adapterOut[0..<promptLen] + promptTextEmbeds

        let (h, cache) = decoder.forward(prefixEmbeds, startPos: 0, cache: nil)
        let logits = decoder.logits(h[h.dim(0) - 1])

        var toEvaluate = [logits]
        for kv in cache {
            if let k = kv.k { toEvaluate.append(k) }
            if let v = kv.v { toEvaluate.append(v) }
        }
        MLX.eval(toEvaluate)

        return (adapterOut, nAudioTotal, promptLen, logits, cache, startTime)
    }

    // MARK: - Token Sampling

    private func nextToken(_ logits: MLXArray, temperature: Float) -> MLXArray {
        if temperature == 0 {
            return argMax(logits)
        }
        return categorical(logits * (1.0 / temperature))
    }

    // MARK: - Public API

    /// Transcribe audio synchronously.
    public func generate(
        audio: MLXArray,
        maxTokens: Int = 4096,
        temperature: Float = 0.0
    ) -> STTOutput {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Use VoxtralRealtimeModel.fromPretrained().")
        }

        let audioArray = audio.flattened().asArray(Float.self)

        let (adapterOut, nAudio, promptLen, initialLogits, initialCache, startTime) =
            encodeAndPrefill(audioArray)

        let adapterLen = adapterOut.dim(0)
        var generated: [Int] = []
        var logits = initialLogits
        var cache = initialCache

        var nextTok = nextToken(logits, temperature: temperature)
        MLX.eval(nextTok)

        let decodeStart = Date()

        for pos in promptLen..<nAudio {
            let token = Int(nextTok.item(Int.self))
            generated.append(token)
            if token == config.eosTokenId || generated.count > maxTokens { break }

            let embed: MLXArray
            if pos < adapterLen {
                embed = adapterOut[pos] + decoder.embedToken(token)
            } else {
                embed = decoder.embedToken(token)
            }

            let (h, newCache) = decoder.forward(
                embed.expandedDimensions(axis: 0), startPos: pos, cache: cache)
            cache = newCache
            logits = decoder.logits(h.squeezed(axis: 0))
            nextTok = nextToken(logits, temperature: temperature)
            MLX.eval(nextTok)

            if generated.count % 256 == 0 {
                Memory.clearCache()
            }
        }

        // If loop ended normally (no break), read final pending token
        if generated.isEmpty || generated.last != config.eosTokenId {
            if generated.count <= maxTokens {
                let token = Int(nextTok.item(Int.self))
                if token != config.eosTokenId {
                    generated.append(token)
                }
            }
        }

        if let last = generated.last, last == config.eosTokenId {
            generated.removeLast()
        }

        let text = tokenizer.decode(generated).trimmingCharacters(in: .whitespacesAndNewlines)
        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let decodeTime = endTime.timeIntervalSince(decodeStart)

        Memory.clearCache()

        return STTOutput(
            text: text,
            promptTokens: promptLen,
            generationTokens: generated.count,
            totalTokens: promptLen + generated.count,
            promptTps: Double(promptLen) / max(totalTime, 0.001),
            generationTps: Double(generated.count) / max(decodeTime, 0.001),
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    /// Transcribe audio with streaming.
    public func generateStream(
        audio: MLXArray,
        maxTokens: Int = 4096,
        temperature: Float = 0.0
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            do {
                guard let tokenizer = self.tokenizer else {
                    throw STTError.modelNotInitialized("Tokenizer not loaded")
                }

                let audioArray = audio.flattened().asArray(Float.self)

                let (adapterOut, nAudio, promptLen, initialLogits, initialCache, startTime) =
                    self.encodeAndPrefill(audioArray)

                let prefillEndTime = Date()
                let prefillTime = prefillEndTime.timeIntervalSince(startTime)

                let adapterLen = adapterOut.dim(0)
                var generated: [Int] = []
                var prevText = ""
                var logits = initialLogits
                var cache = initialCache

                var nextTok = self.nextToken(logits, temperature: temperature)
                MLX.eval(nextTok)

                let generateStart = Date()

                for pos in promptLen..<nAudio {
                    let token = Int(nextTok.item(Int.self))
                    generated.append(token)

                    let nonEos = generated.filter { $0 != self.config.eosTokenId }
                    let textSoFar = tokenizer.decode(nonEos)
                    if textSoFar != prevText {
                        continuation.yield(.token(String(textSoFar.dropFirst(prevText.count))))
                        prevText = textSoFar
                    }

                    if token == self.config.eosTokenId || generated.count > maxTokens { break }

                    let embed: MLXArray
                    if pos < adapterLen {
                        embed = adapterOut[pos] + self.decoder.embedToken(token)
                    } else {
                        embed = self.decoder.embedToken(token)
                    }

                    let (h, newCache) = self.decoder.forward(
                        embed.expandedDimensions(axis: 0), startPos: pos, cache: cache)
                    cache = newCache
                    logits = self.decoder.logits(h.squeezed(axis: 0))
                    nextTok = self.nextToken(logits, temperature: temperature)
                    MLX.eval(nextTok)

                    if generated.count % 256 == 0 {
                        Memory.clearCache()
                    }
                }

                let endTime = Date()
                let generateTime = endTime.timeIntervalSince(generateStart)
                let totalTime = endTime.timeIntervalSince(startTime)

                if let last = generated.last, last == self.config.eosTokenId {
                    generated.removeLast()
                }

                Memory.clearCache()

                let tokensPerSecond =
                    generateTime > 0 ? Double(generated.count) / generateTime : 0
                let peakMemory = Double(Memory.peakMemory) / 1e9

                let info = STTGenerationInfo(
                    promptTokenCount: promptLen,
                    generationTokenCount: generated.count,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: tokensPerSecond,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.info(info))

                let text = tokenizer.decode(generated).trimmingCharacters(
                    in: .whitespacesAndNewlines)
                let output = STTOutput(
                    text: text,
                    promptTokens: promptLen,
                    generationTokens: generated.count,
                    totalTokens: promptLen + generated.count,
                    promptTps: Double(promptLen) / max(prefillTime, 0.001),
                    generationTps: tokensPerSecond,
                    totalTime: totalTime,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.result(output))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    // MARK: - Weight Sanitization

    /// Map weight names from consolidated.safetensors to module structure.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]

        let encPrefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
        let adapterPrefix = "mm_streams_embeddings.embedding_module"
        let tokEmbKey = "mm_streams_embeddings.embedding_module.tok_embeddings.weight"

        for (k, var v) in weights {
            var newKey: String?

            if k == tokEmbKey {
                newKey = "decoder.tok_embeddings.weight"
            } else if k == "norm.weight" {
                newKey = "decoder.norm.weight"
            } else if k.hasPrefix("\(encPrefix).conv_layers.") {
                let rest = String(k.dropFirst("\(encPrefix).conv_layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 2)
                let layerIdx = parts[0]
                let param = parts[2]
                newKey = "encoder.conv_layers_\(layerIdx)_conv.conv.\(param)"
                if param == "weight" && v.ndim == 3 {
                    v = v.transposed(0, 2, 1)
                }
            } else if k.hasPrefix("\(encPrefix).transformer.layers.") {
                let rest = String(k.dropFirst("\(encPrefix).transformer.layers.".count))
                let dotIdx = rest.firstIndex(of: ".")!
                let layerIdx = rest[..<dotIdx]
                var paramPath = String(rest[rest.index(after: dotIdx)...])
                paramPath = paramPath
                    .replacingOccurrences(of: "feed_forward.w1.", with: "feed_forward_w1.")
                    .replacingOccurrences(of: "feed_forward.w2.", with: "feed_forward_w2.")
                    .replacingOccurrences(of: "feed_forward.w3.", with: "feed_forward_w3.")
                newKey = "encoder.transformer_layers.\(layerIdx).\(paramPath)"
            } else if k.hasPrefix("\(encPrefix).transformer.norm.") {
                let rest = String(k.dropFirst("\(encPrefix).transformer.norm.".count))
                newKey = "encoder.transformer_norm.\(rest)"
            } else if k.hasPrefix("\(adapterPrefix).audio_language_projection.") {
                let rest = String(
                    k.dropFirst("\(adapterPrefix).audio_language_projection.".count))
                let parts = rest.split(separator: ".", maxSplits: 1)
                let idx = parts[0]
                let param = parts[1]
                newKey = "encoder.audio_language_projection_\(idx).\(param)"
            } else if k.hasPrefix("layers.") {
                let rest = String(k.dropFirst("layers.".count))
                let dotIdx = rest.firstIndex(of: ".")!
                let layerIdx = rest[..<dotIdx]
                var paramPath = String(rest[rest.index(after: dotIdx)...])
                paramPath = paramPath
                    .replacingOccurrences(of: "feed_forward.w1.", with: "feed_forward_w1.")
                    .replacingOccurrences(of: "feed_forward.w2.", with: "feed_forward_w2.")
                    .replacingOccurrences(of: "feed_forward.w3.", with: "feed_forward_w3.")
                    .replacingOccurrences(
                        of: "ada_rms_norm_t_cond.0.", with: "ada_rms_norm_t_cond.ada_down.")
                    .replacingOccurrences(
                        of: "ada_rms_norm_t_cond.2.", with: "ada_rms_norm_t_cond.ada_up.")
                newKey = "decoder.layers.\(layerIdx).\(paramPath)"
            }

            if let key = newKey {
                newWeights[key] = v
            } else {
                newWeights[k] = v
            }
        }
        return newWeights
    }

    /// Quantization predicate: skip norms, embeddings, convolutions, adapter projections.
    func quantizationPredicate(_ path: String, _ module: Module) -> Bool {
        let skipPatterns = [
            "norm", "ada_rms_norm", "tok_embeddings",
            "conv_layers", "audio_language_projection",
        ]
        return !skipPatterns.contains(where: { path.contains($0) })
    }

    // MARK: - From Pretrained

    /// Load model from HuggingFace Hub.
    public static func fromPretrained(_ modelPath: String) async throws -> VoxtralRealtimeModel {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "VoxtralRealtimeModel", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"])
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: "safetensors")

        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(VoxtralRealtimeModelConfig.self, from: configData)

        let perLayerQuantization = config.perLayerQuantization
        let model = VoxtralRealtimeModel(config)
        model.tokenizer = try TekkenTokenizer.fromModelPath(modelDir)

        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitizedWeights = model.sanitize(weights: weights)

        if let perLayerQuant = perLayerQuantization {
            quantize(model: model) { path, module in
                guard model.quantizationPredicate(path, module) else { return nil }
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuant.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])

        _ = model.ensureMelFilters()
        model.ensureAdaScales()

        MLX.eval(model)

        return model
    }
}
