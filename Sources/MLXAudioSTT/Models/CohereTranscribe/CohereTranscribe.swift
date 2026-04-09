import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon

private struct CoherePrefillContext {
    let adapterOut: MLXArray
    let promptLength: Int
    var logits: MLXArray
    var cache: CohereTranscribeDecoderKVCache?
    let startTime: Date
}

private func normalizeCohereWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var normalized = [String: MLXArray](minimumCapacity: weights.count)
    let replacements = [
        "encoder.subsampling.conv.0.": "encoder.subsampling.conv0.",
        "encoder.subsampling.conv.2.": "encoder.subsampling.conv2.",
        "encoder.subsampling.conv.3.": "encoder.subsampling.conv3.",
        "encoder.subsampling.conv.5.": "encoder.subsampling.conv5.",
        "encoder.subsampling.conv.6.": "encoder.subsampling.conv6.",
    ]
    let subsamplingKernelShapes = [
        "encoder.subsampling.conv0.weight": [3, 3],
        "encoder.subsampling.conv2.weight": [3, 3],
        "encoder.subsampling.conv3.weight": [1, 1],
        "encoder.subsampling.conv5.weight": [3, 3],
        "encoder.subsampling.conv6.weight": [1, 1],
    ]

    for (key, value) in weights {
        if key.hasSuffix(".num_batches_tracked")
            || key.hasPrefix("decoder.embedding.position_embedding")
            || key.hasPrefix("preprocessor.")
        {
            continue
        }

        let mappedKey = replacements.first { key.hasPrefix($0.key) }.map {
            key.replacingOccurrences(of: $0.key, with: $0.value)
        } ?? key

        if mappedKey.hasSuffix(".weight"), value.ndim == 4, let kernelShape = subsamplingKernelShapes[mappedKey] {
            if value.shape[1] == kernelShape[0], value.shape[2] == kernelShape[1] {
                normalized[mappedKey] = value
            } else if value.shape[2] == kernelShape[0], value.shape[3] == kernelShape[1] {
                normalized[mappedKey] = value.transposed(0, 2, 3, 1)
            } else {
                normalized[mappedKey] = value
            }
        } else if mappedKey.hasSuffix(".weight"), value.ndim == 3, mappedKey.contains(".conv.") {
            let likelyPyTorchLayout: Bool
            if mappedKey.contains("depthwise_conv") {
                likelyPyTorchLayout = value.shape[1] == 1 && value.shape[2] > 1
            } else {
                likelyPyTorchLayout = value.shape[2] == 1 && value.shape[1] > 1
            }
            normalized[mappedKey] = likelyPyTorchLayout ? value.transposed(0, 2, 1) : value
        } else {
            normalized[mappedKey] = value
        }
    }

    return normalized
}

public final class CohereTranscribeModel: Module, STTGenerationModel {
    public let config: CohereTranscribeConfig

    @ModuleInfo(key: "encoder") var encoder: ConformerEncoder
    @ModuleInfo(key: "decoder") var decoder: TransformerDecoderWrapper
    @ModuleInfo(key: "bridge_proj") var bridgeProj: Linear?
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    private var tokenizer: CohereTranscribeTokenizer?

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: config.decoder.maxSequenceLength,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            verbose: false,
            language: "en"
        )
    }

    public init(_ config: CohereTranscribeConfig) {
        self.config = config
        self._encoder.wrappedValue = ConformerEncoder(config.encoder)
        self._decoder.wrappedValue = TransformerDecoderWrapper(config: config)
        
        if config.encoder.dModel != config.decoder.hiddenSize {
            self._bridgeProj.wrappedValue = Linear(config.encoder.dModel, config.decoder.hiddenSize)
        }
        self._lmHead.wrappedValue = Linear(config.decoder.hiddenSize, config.vocabSize)
    }

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let chunks = splitAudioIntoChunks(
            audio1D,
            sampleRate: config.sampleRate,
            chunkDuration: generationParameters.chunkDuration,
            minChunkDuration: generationParameters.minChunkDuration
        )

        guard chunks.count > 1 else {
            return generateSingleChunk(audio: audio1D, generationParameters: generationParameters)
        }

        var outputs: [STTOutput] = []
        outputs.reserveCapacity(chunks.count)
        var remainingTokens = generationParameters.maxTokens

        for (chunkAudio, offsetSeconds) in chunks {
            if remainingTokens <= 0 {
                break
            }

            if generationParameters.verbose {
                print("Processing chunk at \(String(format: "%.1f", offsetSeconds))s")
            }

            let chunkParameters = chunkedParameters(
                from: generationParameters,
                maxTokens: remainingTokens
            )
            let output = generateSingleChunk(audio: chunkAudio, generationParameters: chunkParameters)
            outputs.append(output)
            remainingTokens = max(0, remainingTokens - output.generationTokens)
        }

        let combinedText = outputs
            .map(\.text)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")

        let promptTokens = outputs.reduce(0) { $0 + $1.promptTokens }
        let generationTokens = outputs.reduce(0) { $0 + $1.generationTokens }
        let totalTokens = outputs.reduce(0) { $0 + $1.totalTokens }
        let totalTime = outputs.reduce(0.0) { $0 + $1.totalTime }
        let peakMemoryUsage = outputs.map(\.peakMemoryUsage).max() ?? 0

        return STTOutput(
            text: combinedText,
            language: generationParameters.language,
            promptTokens: promptTokens,
            generationTokens: generationTokens,
            totalTokens: totalTokens,
            promptTps: totalTime > 0 ? Double(promptTokens) / totalTime : 0,
            generationTps: totalTime > 0 ? Double(generationTokens) / totalTime : 0,
            totalTime: totalTime,
            peakMemoryUsage: peakMemoryUsage
        )
    }

    private func generateSingleChunk(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        var context = encodeAndPrefill(
            audio: audio,
            generationParameters: generationParameters
        )

        var generated: [Int] = []
        let decodeStart = Date()
        
        let eosTokenId = tokenizer?.encode(text: "<|endoftext|>").first ?? 0

        let maxGenerationTokens = effectiveMaxGenerationTokens(
            promptLength: context.promptLength,
            requestedMaxTokens: generationParameters.maxTokens
        )

        for pos in context.promptLength..<(context.promptLength + maxGenerationTokens) {
            let token = sample(logits: context.logits, temperature: generationParameters.temperature)
            generated.append(token)

            if token == eosTokenId {
                break
            }

            let inputIds = MLXArray([Int32(token)]).expandedDimensions(axis: 0)
            let positions = MLXArray([Int32(pos)]).expandedDimensions(axis: 0)

            let next = decoder(
                inputIds: inputIds,
                positions: positions,
                encoderHiddenStates: context.adapterOut,
                selfAttentionMask: nil,
                crossAttentionMask: nil,
                cache: context.cache
            )
            
            context.cache = next.1
            context.logits = lmHead(next.0[0, -1])

            eval(context.logits)
            if generated.count % 256 == 0 {
                Memory.clearCache()
            }
        }

        if generated.last == eosTokenId {
            _ = generated.popLast()
        }

        let text = tokenizer?.decode(tokens: generated).trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

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
        let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let chunks = splitAudioIntoChunks(
            audio1D,
            sampleRate: config.sampleRate,
            chunkDuration: generationParameters.chunkDuration,
            minChunkDuration: generationParameters.minChunkDuration
        )

        if chunks.count > 1 {
            return AsyncThrowingStream { continuation in
                var outputs: [STTOutput] = []
                outputs.reserveCapacity(chunks.count)
                var remainingTokens = generationParameters.maxTokens
                var emittedAnyText = false

                for chunk in chunks {
                    if remainingTokens <= 0 {
                        break
                    }

                    let chunkParameters = chunkedParameters(
                        from: generationParameters,
                        maxTokens: remainingTokens
                    )
                    let output = generateSingleChunk(audio: chunk.0, generationParameters: chunkParameters)
                    outputs.append(output)
                    remainingTokens = max(0, remainingTokens - output.generationTokens)

                    let text = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !text.isEmpty {
                        let token = emittedAnyText ? "\n" + text : text
                        continuation.yield(STTGeneration.token(token))
                        emittedAnyText = true
                    }
                }

                let combinedText = outputs
                    .map(\.text)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n")

                let promptTokens = outputs.reduce(0) { $0 + $1.promptTokens }
                let generationTokens = outputs.reduce(0) { $0 + $1.generationTokens }
                let totalTokens = outputs.reduce(0) { $0 + $1.totalTokens }
                let totalTime = outputs.reduce(0.0) { $0 + $1.totalTime }
                let peakMemoryUsage = outputs.map(\.peakMemoryUsage).max() ?? 0

                continuation.yield(STTGeneration.result(STTOutput(
                    text: combinedText,
                    language: generationParameters.language,
                    promptTokens: promptTokens,
                    generationTokens: generationTokens,
                    totalTokens: totalTokens,
                    promptTps: totalTime > 0 ? Double(promptTokens) / totalTime : 0,
                    generationTps: totalTime > 0 ? Double(generationTokens) / totalTime : 0,
                    totalTime: totalTime,
                    peakMemoryUsage: peakMemoryUsage
                )))
                continuation.finish()
            }
        }

        return AsyncThrowingStream { continuation in
            var context = encodeAndPrefill(
                audio: audio1D,
                generationParameters: generationParameters
            )

            var generated: [Int] = []
            var previousText = ""
            let decodeStart = Date()
            
            let eosTokenId = tokenizer?.encode(text: "<|endoftext|>").first ?? 0

            let maxGenerationTokens = effectiveMaxGenerationTokens(
                promptLength: context.promptLength,
                requestedMaxTokens: generationParameters.maxTokens
            )

            for pos in context.promptLength..<(context.promptLength + maxGenerationTokens) {
                let token = sample(logits: context.logits, temperature: generationParameters.temperature)
                generated.append(token)

                let textSoFar = tokenizer?.decode(tokens: generated) ?? ""
                if textSoFar != previousText {
                    let delta: String
                    if textSoFar.hasPrefix(previousText) {
                        delta = String(textSoFar.dropFirst(previousText.count))
                    } else {
                        delta = textSoFar
                    }
                    if !delta.isEmpty {
                        continuation.yield(STTGeneration.token(delta))
                    }
                    previousText = textSoFar
                }

                if token == eosTokenId {
                    break
                }

                let inputIds = MLXArray([Int32(token)]).expandedDimensions(axis: 0)
                let positions = MLXArray([Int32(pos)]).expandedDimensions(axis: 0)

                let next = decoder(
                    inputIds: inputIds,
                    positions: positions,
                    encoderHiddenStates: context.adapterOut,
                    selfAttentionMask: nil,
                    crossAttentionMask: nil,
                    cache: context.cache
                )
                
                context.cache = next.1
                context.logits = lmHead(next.0[0, -1])

                eval(context.logits)
                if generated.count % 256 == 0 {
                    Memory.clearCache()
                }
            }

            if generated.last == eosTokenId {
                _ = generated.popLast()
            }

            let finalText = tokenizer?.decode(tokens: generated).trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
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

            Memory.clearCache()
            continuation.yield(.result(output))
            continuation.finish()
        }
    }
}

private extension CohereTranscribeModel {
    func encodeAndPrefill(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> CoherePrefillContext {
        let start = Date()
        guard let tokenizer else {
            fatalError("CohereTranscribeTokenizer must be loaded before generation")
        }

        let melFilters = CohereTranscribeAudio.computeMelFilters(
            sampleRate: config.sampleRate,
            nFft: 512,
            numMels: config.encoder.featIn
        )
        
        let features = CohereTranscribeAudio.computeFeatures(
            audio: audio,
            melFilters: melFilters,
            nFft: 512,
            winLength: 400,
            hopLength: 160
        )

        let (encOut, _) = encoder(features)
        
        let adapterOut: MLXArray
        if let bridgeProj = bridgeProj {
            adapterOut = bridgeProj(encOut)
        } else {
            adapterOut = encOut
        }

        let promptIds = tokenizer.buildPromptTokens(
            language: generationParameters.language ?? "en",
            usePunctuation: true,
            useTimestamps: false
        )
        
        let promptLength = promptIds.count
        let promptIdsMX = MLXArray(promptIds.map(Int32.init)).expandedDimensions(axis: 0)
        let positions = MLXArray((0..<promptLength).map(Int32.init)).expandedDimensions(axis: 0)
        let selfMask = MultiHeadAttention.createAdditiveCausalMask(promptLength).asType(adapterOut.dtype)

        let prefill = decoder(
            inputIds: promptIdsMX,
            positions: positions,
            encoderHiddenStates: adapterOut,
            selfAttentionMask: selfMask,
            crossAttentionMask: nil,
            cache: nil
        )
        
        let h = prefill.0
        let cache = prefill.1

        let logits = lmHead(h[0, -1])
        
        var cacheArrays: [MLXArray] = [logits]
        for layerCache in cache.layers {
            if let selfKeys = layerCache.selfKeys, let selfValues = layerCache.selfValues {
                cacheArrays.append(selfKeys)
                cacheArrays.append(selfValues)
            }
            if let crossKeys = layerCache.crossKeys, let crossValues = layerCache.crossValues {
                cacheArrays.append(crossKeys)
                cacheArrays.append(crossValues)
            }
        }
        eval(cacheArrays)

        if generationParameters.verbose {
            let seconds = Double(audio.shape[0]) / Double(config.sampleRate)
            print("Audio: \(audio.shape[0]) samples (\(String(format: "%.1f", seconds))s)")
            print("Prompt: \(promptLength) tokens")
        }

        return CoherePrefillContext(
            adapterOut: adapterOut,
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

    func effectiveMaxGenerationTokens(promptLength: Int, requestedMaxTokens: Int) -> Int {
        let availableTokens = max(0, config.decoder.maxSequenceLength - promptLength)
        return min(requestedMaxTokens, availableTokens)
    }

    func chunkedParameters(from generationParameters: STTGenerateParameters, maxTokens: Int) -> STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: maxTokens,
            temperature: generationParameters.temperature,
            topP: generationParameters.topP,
            topK: generationParameters.topK,
            verbose: generationParameters.verbose,
            language: generationParameters.language,
            chunkDuration: generationParameters.chunkDuration,
            minChunkDuration: generationParameters.minChunkDuration
        )
    }
}

public extension CohereTranscribeModel {
    static func fromDirectory(_ modelDir: URL) throws -> CohereTranscribeModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(CohereTranscribeConfig.self, from: configData)

        let model = CohereTranscribeModel(config)
        model.tokenizer = try CohereTranscribeTokenizer(modelDir: modelDir, config: config)

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensors = files.filter { $0.pathExtension == "safetensors" }

        var weights: [String: MLXArray] = [:]
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let sanitizedWeights = normalizeCohereWeightKeys(weights)

        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitizedWeights["\(path).scales"] != nil else {
                    return nil
                }

                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path) {
                    return layerQuant.asTuple
                }

                return config.quantization?.asTuple
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .all)
        model.train(false)
        eval(model)

        return model
    }

    static func fromPretrained(_ modelPath: String) async throws -> CohereTranscribeModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "CohereTranscribeModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.model"],
            hfToken: hfToken
        )

        let model = try fromDirectory(modelDir)
        return model
    }
}
