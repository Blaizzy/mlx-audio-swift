import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace

private enum VoxtralRealtimeConstants {
    static let sampleRate = 16000
    static let frameRate: Float = 12.5
    static let rawAudioLengthPerToken = Int(Float(sampleRate) / frameRate) // 1280
    static let hopLength = 160
    static let audioLengthPerToken = rawAudioLengthPerToken / hopLength // 8
}

private struct VoxtralPrefillContext {
    let adapterOut: MLXArray
    let nAudioTotal: Int
    let promptLength: Int
    var logits: MLXArray
    var cache: [VoxtralRealtimeDecoderKVCache?]
    let startTime: Date
}

public final class VoxtralRealtimeModel: Module, STTGenerationModel {
    public let config: VoxtralRealtimeConfig

    @ModuleInfo(key: "encoder") var encoder: VoxtralRealtimeAudioEncoder
    @ModuleInfo(key: "decoder") var decoder: VoxtralRealtimeDecoder

    private var tokenizer: VoxtralRealtimeTokenizer?
    private var adaScaleDelay: Int = -1

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 4096,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            verbose: false,
            language: "en",
            chunkDuration: 1200.0,
            minChunkDuration: 1.0
        )
    }

    public init(_ config: VoxtralRealtimeConfig) {
        self.config = config
        self._encoder.wrappedValue = VoxtralRealtimeAudioEncoder(
            config.encoderArgs,
            decoderDim: config.decoder.dim
        )
        self._decoder.wrappedValue = VoxtralRealtimeDecoder(config.decoder)
    }

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        guard tokenizer != nil else {
            return STTOutput(text: "", language: generationParameters.language)
        }

        let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        var context = encodeAndPrefill(
            audio: audio1D,
            verbose: generationParameters.verbose,
            transcriptionDelayMs: nil
        )

        var generated: [Int] = []
        let decodeStart = Date()

        for pos in context.promptLength..<context.nAudioTotal {
            let token = sample(logits: context.logits, temperature: generationParameters.temperature)
            generated.append(token)

            if token == config.eosTokenId || generated.count > generationParameters.maxTokens {
                break
            }

            let tokenEmbed = decoder.embedToken(tokenId: token)
            let inputEmbed: MLXArray
            if pos < context.adapterOut.shape[0] {
                inputEmbed = context.adapterOut[pos] + tokenEmbed
            } else {
                inputEmbed = tokenEmbed
            }

            let next = decoder.forward(
                inputEmbed.expandedDimensions(axis: 0),
                startPos: pos,
                cache: context.cache
            )
            context.cache = next.1
            context.logits = decoder.logits(next.0[0])

            eval(context.logits)
            if generated.count % 256 == 0 {
                Memory.clearCache()
            }
        }

        if generated.last == config.eosTokenId {
            _ = generated.popLast()
        }

        let text = tokenizer?.decode(tokenIds: generated).trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        let end = Date()
        let totalTime = end.timeIntervalSince(context.startTime)
        let decodeTime = end.timeIntervalSince(decodeStart)

        Memory.clearCache()

        return STTOutput(
            text: text,
            language: generationParameters.language,
            promptTokens: context.promptLength,
            generationTokens: generated.count,
            totalTokens: context.promptLength + generated.count,
            promptTps: totalTime > 0 ? Double(context.promptLength) / totalTime : 0,
            generationTps: decodeTime > 0 ? Double(generated.count) / decodeTime : 0,
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            guard let tokenizer else {
                continuation.finish(throwing: STTError.modelNotInitialized("Tokenizer not loaded"))
                return
            }

            let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
            var context = encodeAndPrefill(
                audio: audio1D,
                verbose: generationParameters.verbose,
                transcriptionDelayMs: nil
            )

            var generated: [Int] = []
            var previousText = ""
            let decodeStart = Date()

            for pos in context.promptLength..<context.nAudioTotal {
                let token = sample(logits: context.logits, temperature: generationParameters.temperature)
                generated.append(token)

                let filtered = generated.filter { $0 != config.eosTokenId }
                let textSoFar = tokenizer.decode(tokenIds: filtered)
                if textSoFar != previousText {
                    let delta: String
                    if textSoFar.hasPrefix(previousText) {
                        delta = String(textSoFar.dropFirst(previousText.count))
                    } else {
                        delta = textSoFar
                    }
                    if !delta.isEmpty {
                        continuation.yield(.token(delta))
                    }
                    previousText = textSoFar
                }

                if token == config.eosTokenId || generated.count > generationParameters.maxTokens {
                    break
                }

                let tokenEmbed = decoder.embedToken(tokenId: token)
                let inputEmbed: MLXArray
                if pos < context.adapterOut.shape[0] {
                    inputEmbed = context.adapterOut[pos] + tokenEmbed
                } else {
                    inputEmbed = tokenEmbed
                }

                let next = decoder.forward(
                    inputEmbed.expandedDimensions(axis: 0),
                    startPos: pos,
                    cache: context.cache
                )
                context.cache = next.1
                context.logits = decoder.logits(next.0[0])

                eval(context.logits)
                if generated.count % 256 == 0 {
                    Memory.clearCache()
                }
            }

            if generated.last == self.config.eosTokenId {
                _ = generated.popLast()
            }

            let finalText = tokenizer.decode(tokenIds: generated).trimmingCharacters(in: .whitespacesAndNewlines)
            let end = Date()
            let totalTime = end.timeIntervalSince(context.startTime)
            let decodeTime = end.timeIntervalSince(decodeStart)

            let output = STTOutput(
                text: finalText,
                language: generationParameters.language,
                promptTokens: context.promptLength,
                generationTokens: generated.count,
                totalTokens: context.promptLength + generated.count,
                promptTps: totalTime > 0 ? Double(context.promptLength) / totalTime : 0,
                generationTps: decodeTime > 0 ? Double(generated.count) / decodeTime : 0,
                totalTime: totalTime,
                peakMemoryUsage: Double(Memory.peakMemory) / 1e9
            )

            continuation.yield(.result(output))
            continuation.finish()
        }
    }
}

private extension VoxtralRealtimeModel {
    func numAudioTokens(_ audioLength: Int) -> Int {
        let hop = VoxtralRealtimeConstants.hopLength
        let perTok = VoxtralRealtimeConstants.audioLengthPerToken

        let frames: Int
        if audioLength % hop != 0 {
            frames = Int(ceil(Double(audioLength) / Double(hop) - 1.0))
        } else {
            frames = audioLength / hop
        }
        return Int(ceil(Double(frames) / Double(perTok)))
    }

    func numDelayTokens(_ delayMs: Int) -> Int {
        let delayLen = Int(Double(delayMs) / 1000.0 * Double(VoxtralRealtimeConstants.sampleRate))
        return numAudioTokens(delayLen)
    }

    func padAudioStreaming(_ audio: MLXArray, leftTokens: Int, rightTokens: Int) -> MLXArray {
        let mult = VoxtralRealtimeConstants.rawAudioLengthPerToken
        let nSamples = audio.shape[0]

        let alignPad = (mult - (nSamples % mult)) % mult
        let rightPad = alignPad + rightTokens * mult
        let leftPad = leftTokens * mult

        return MLX.padded(audio, widths: [IntOrPair((leftPad, rightPad))])
    }

    func ensureMelFilters() -> MLXArray {
        let a = config.audioEncodingArgs
        return VoxtralRealtimeAudio.computeMelFilters(
            numMelBins: a.numMelBins,
            windowSize: a.windowSize,
            sampleRate: a.samplingRate
        ).asType(.float32)
    }

    func ensureAdaScales(transcriptionDelayMs: Int?) {
        let delayMs = transcriptionDelayMs ?? config.transcriptionDelayMs
        let delayTokens = numDelayTokens(delayMs)

        if delayTokens != adaScaleDelay {
            let tCond = voxtralComputeTimeEmbedding(
                tValue: Float(delayTokens),
                dim: config.decoder.dim
            )
            decoder.precomputeAdaScales(tCond)
            if let adaScales = decoder.adaScales {
                for scale in adaScales {
                    if let scale {
                        eval(scale)
                    }
                }
            }
            adaScaleDelay = delayTokens
        }
    }

    func prepareMel(_ audio: MLXArray, transcriptionDelayMs: Int?) -> (MLXArray, Int) {
        let delayMs = transcriptionDelayMs ?? config.transcriptionDelayMs
        let nDelay = numDelayTokens(delayMs)
        let nLeft = config.nLeftPadTokens
        let nRight = (nDelay + 1) + 10

        let padded = padAudioStreaming(audio, leftTokens: nLeft, rightTokens: nRight)
        let a = config.audioEncodingArgs
        let mel = VoxtralRealtimeAudio.computeMelSpectrogram(
            audio: padded,
            melFilters: ensureMelFilters(),
            windowSize: a.windowSize,
            hopLength: a.hopLength,
            globalLogMelMax: a.globalLogMelMax
        )

        if mel.shape[1] % 2 != 0 {
            return (mel[0..., 1...], nDelay)
        }
        return (mel, nDelay)
    }

    func encodeAndPrefill(
        audio: MLXArray,
        verbose: Bool,
        transcriptionDelayMs: Int?
    ) -> VoxtralPrefillContext {
        let start = Date()

        ensureAdaScales(transcriptionDelayMs: transcriptionDelayMs)
        let (mel, nDelay) = prepareMel(audio, transcriptionDelayMs: transcriptionDelayMs)

        let convOut = encoder.convStem(mel)
        let ds = config.encoderArgs.downsampleFactor
        let nAudioTotal = convOut.shape[0] / ds

        let promptLength = 1 + config.nLeftPadTokens + nDelay

        let adapterOut: MLXArray
        if convOut.shape[0] <= config.encoderArgs.slidingWindow {
            adapterOut = encoder.encodeFull(convOut)
        } else {
            adapterOut = encoder.encodeChunked(convOut)
        }

        let promptIds = [config.bosTokenId] + Array(
            repeating: config.streamingPadTokenId,
            count: config.nLeftPadTokens + nDelay
        )
        let promptIdsMX = MLXArray(promptIds.map(Int32.init))
        let promptTextEmbeds = decoder.embedTokens(promptIdsMX)

        let prefixEmbeds = adapterOut[0..<promptLength, 0...] + promptTextEmbeds

        let prefill = decoder.forward(prefixEmbeds, startPos: 0, cache: nil)
        let h = prefill.0
        let cache = prefill.1

        let logits = decoder.logits(h[h.shape[0] - 1])

        var cacheArrays: [MLXArray] = [logits]
        for layerCache in cache {
            if let layerCache {
                cacheArrays.append(layerCache.keys)
                cacheArrays.append(layerCache.values)
            }
        }
        eval(cacheArrays)

        if verbose {
            let seconds = Double(audio.shape[0]) / Double(VoxtralRealtimeConstants.sampleRate)
            print("Audio: \(audio.shape[0]) samples (\(String(format: "%.1f", seconds))s)")
            print("Prompt: \(promptLength) tokens, Audio span: \(nAudioTotal) tokens")
        }

        return VoxtralPrefillContext(
            adapterOut: adapterOut,
            nAudioTotal: nAudioTotal,
            promptLength: promptLength,
            logits: logits,
            cache: cache,
            startTime: start
        )
    }

    func sample(logits: MLXArray, temperature: Float) -> Int {
        let logits1D: MLXArray
        if logits.ndim > 1 {
            logits1D = logits.squeezed()
        } else {
            logits1D = logits
        }

        if temperature == 0 {
            return logits1D.argMax(axis: -1).item(Int.self)
        }

        let scaled = (logits1D / temperature).expandedDimensions(axis: 0)
        let sampled = categorical(scaled)
        return sampled.item(Int.self)
    }
}

public extension VoxtralRealtimeModel {
    static func fromDirectory(_ modelDir: URL) throws -> VoxtralRealtimeModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(VoxtralRealtimeConfig.self, from: configData)

        let quantConfig = try JSONDecoder().decode(VoxtralQuantizationConfig.self, from: configData)

        let model = VoxtralRealtimeModel(config)
        model.tokenizer = try VoxtralRealtimeTokenizer.fromModelDirectory(modelDir)
        _ = model.ensureMelFilters()

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensors = files.filter { $0.pathExtension == "safetensors" }

        var weights: [String: MLXArray] = [:]
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let sanitized = sanitize(weights: weights)

        if let perLayerQuantization = quantConfig.perLayerQuantization {
            quantize(model: model) { path, _ in
                guard model.shouldQuantize(path: path) else {
                    return nil
                }
                if sanitized["\(path).scales"] != nil {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        model.ensureAdaScales(transcriptionDelayMs: config.transcriptionDelayMs)
        eval(model)

        return model
    }

    static func fromPretrained(_ modelPath: String) async throws -> VoxtralRealtimeModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "VoxtralRealtimeModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        return try fromDirectory(modelDir)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        newWeights.reserveCapacity(weights.count)

        let encPrefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
        let adapterPrefix = "mm_streams_embeddings.embedding_module"
        let tokEmbKey = "mm_streams_embeddings.embedding_module.tok_embeddings.weight"

        for (key, value) in weights {
            var v = value
            var mapped: String?

            if key == tokEmbKey {
                mapped = "decoder.tok_embeddings.weight"
            } else if key == "norm.weight" {
                mapped = "decoder.norm.weight"
            } else if key.hasPrefix("\(encPrefix).conv_layers.") {
                let rest = String(key.dropFirst("\(encPrefix).conv_layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
                if parts.count == 3 {
                    let layerIdx = String(parts[0])
                    let param = String(parts[2])
                    mapped = "encoder.conv_layers_\(layerIdx)_conv.conv.\(param)"
                    if param == "weight" && v.ndim == 3 {
                        v = v.transposed(0, 2, 1)
                    }
                }
            } else if key.hasPrefix("\(encPrefix).transformer.layers.") {
                let rest = String(key.dropFirst("\(encPrefix).transformer.layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 {
                    let layerIdx = String(parts[0])
                    var paramPath = String(parts[1])
                    paramPath = paramPath.replacingOccurrences(of: "feed_forward.w1.", with: "feed_forward_w1.")
                    paramPath = paramPath.replacingOccurrences(of: "feed_forward.w2.", with: "feed_forward_w2.")
                    paramPath = paramPath.replacingOccurrences(of: "feed_forward.w3.", with: "feed_forward_w3.")
                    mapped = "encoder.transformer_layers.\(layerIdx).\(paramPath)"
                }
            } else if key.hasPrefix("\(encPrefix).transformer.norm.") {
                let rest = String(key.dropFirst("\(encPrefix).transformer.norm.".count))
                mapped = "encoder.transformer_norm.\(rest)"
            } else if key.hasPrefix("\(adapterPrefix).audio_language_projection.") {
                let rest = String(key.dropFirst("\(adapterPrefix).audio_language_projection.".count))
                let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 {
                    let idx = String(parts[0])
                    let param = String(parts[1])
                    mapped = "encoder.audio_language_projection_\(idx).\(param)"
                }
            } else if key.hasPrefix("layers.") {
                let rest = String(key.dropFirst("layers.".count))
                let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 {
                    let layerIdx = String(parts[0])
                    var paramPath = String(parts[1])
                    paramPath = paramPath.replacingOccurrences(of: "feed_forward.w1.", with: "feed_forward_w1.")
                    paramPath = paramPath.replacingOccurrences(of: "feed_forward.w2.", with: "feed_forward_w2.")
                    paramPath = paramPath.replacingOccurrences(of: "feed_forward.w3.", with: "feed_forward_w3.")
                    paramPath = paramPath.replacingOccurrences(of: "ada_rms_norm_t_cond.0.", with: "ada_rms_norm_t_cond.ada_down.")
                    paramPath = paramPath.replacingOccurrences(of: "ada_rms_norm_t_cond.2.", with: "ada_rms_norm_t_cond.ada_up.")
                    mapped = "decoder.layers.\(layerIdx).\(paramPath)"
                }
            }

            if let mapped {
                newWeights[mapped] = v
            } else {
                newWeights[key] = v
            }
        }

        return newWeights
    }

    func shouldQuantize(path: String) -> Bool {
        let skipPatterns = [
            "norm",
            "ada_rms_norm",
            "tok_embeddings",
            "conv_layers",
            "audio_language_projection",
        ]
        return !skipPatterns.contains(where: { path.contains($0) })
    }
}

private struct VoxtralQuantizationConfig: Decodable {
    let perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    init(from decoder: Decoder) throws {
        let base = try? BaseConfiguration(from: decoder)
        self.perLayerQuantization = base?.perLayerQuantization
    }
}
