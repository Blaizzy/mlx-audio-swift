import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace
import Tokenizers

private let mossAudioPadToken = "<|audio_pad|>"
private let mossAudioStartToken = "<|audio_start|>"
private let mossAudioEndToken = "<|audio_end|>"
private let mossWhisperEncoderStride = 2

public let mossTranscribeDiarizeDefaultRepo = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

private let mossDefaultPrompt = """
Transcribe the audio into text. Start each segment with the start timestamp and speaker label ([S01], [S02], [S03], ...), write the corresponding spoken content, and end each segment with the ending timestamp to clearly mark the segment range.
"""

private struct MossTimestampTagOffsetter {
    let offsetSeconds: Double
    private var bufferedTag = ""
    private var isBufferingTag = false

    init(offsetSeconds: Double) {
        self.offsetSeconds = offsetSeconds
    }

    mutating func consume(_ text: String) -> String {
        guard offsetSeconds != 0 else { return text }

        var output = ""
        for character in text {
            if isBufferingTag {
                bufferedTag.append(character)
                if character == "]" {
                    output += offsetTag(bufferedTag)
                    bufferedTag = ""
                    isBufferingTag = false
                } else if bufferedTag.count > 24 {
                    output += bufferedTag
                    bufferedTag = ""
                    isBufferingTag = false
                }
            } else if character == "[" {
                bufferedTag = "["
                isBufferingTag = true
            } else {
                output.append(character)
            }
        }

        return output
    }

    mutating func finish() -> String {
        guard isBufferingTag else { return "" }
        defer {
            bufferedTag = ""
            isBufferingTag = false
        }
        return bufferedTag
    }

    private func offsetTag(_ tag: String) -> String {
        guard
            tag.hasPrefix("["),
            tag.hasSuffix("]"),
            let value = Double(tag.dropFirst().dropLast().replacingOccurrences(of: ",", with: "."))
        else {
            return tag
        }

        return String(format: "[%.2f]", locale: Locale(identifier: "en_US_POSIX"), value + offsetSeconds)
    }
}

public struct MossTranscribeDiarizeStreamingResult: Sendable {
    public let text: String
    public let isFinal: Bool
    public let startTime: Float
    public let endTime: Float
    public let language: String
    public let promptTokens: Int
    public let generationTokens: Int
}

public final class MossTranscribeDiarizeVQAdaptor: Module {
    @ModuleInfo(key: "layers") var layers: MLXNN.Sequential

    init(inputDim: Int, hiddenSize: Int, normEps: Float) {
        self._layers.wrappedValue = MLXNN.Sequential(layers: [
            Linear(inputDim, hiddenSize, bias: true),
            SiLU(),
            Linear(hiddenSize, hiddenSize, bias: true),
            LayerNorm(dimensions: hiddenSize, eps: normEps),
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers(x)
    }
}

public final class MossTranscribeDiarizeBackbone: Module {
    let config: MossTranscribeDiarizeConfig

    @ModuleInfo(key: "language_model") var languageModel: Qwen3ASRTextModel
    @ModuleInfo(key: "whisper_encoder") var whisperEncoder: WhisperEncoder
    @ModuleInfo(key: "vq_adaptor") var vqAdaptor: MossTranscribeDiarizeVQAdaptor

    init(_ config: MossTranscribeDiarizeConfig) {
        self.config = config
        self._languageModel.wrappedValue = Qwen3ASRTextModel(config.textConfig)
        self._whisperEncoder.wrappedValue = WhisperEncoder(config: config.audioConfig)
        self._vqAdaptor.wrappedValue = MossTranscribeDiarizeVQAdaptor(
            inputDim: config.adaptorInputDim ?? config.audioConfig.dModel * config.audioMergeSize,
            hiddenSize: config.textConfig.hiddenSize,
            normEps: config.textConfig.rmsNormEps
        )
    }

    func timeMerge(_ features: MLXArray) -> MLXArray {
        let batchSize = features.dim(0)
        let seqLen = features.dim(1)
        let dim = features.dim(2)
        let mergeSize = config.audioMergeSize
        let trimLen = (seqLen / mergeSize) * mergeSize
        return features[0..., 0..<trimLen, 0...].reshaped(
            batchSize,
            trimLen / mergeSize,
            dim * mergeSize
        )
    }

    func getAudioFeatures(
        inputFeatures: MLXArray,
        audioFeatureLengths: MLXArray,
        audioChunkMapping: MLXArray? = nil
    ) throws -> [MLXArray] {
        let whisperFeatures = whisperEncoder(inputFeatures)
        let lengths = audioFeatureLengths.asArray(Int32.self).map(Int.init)
        let mapping = audioChunkMapping?.asArray(Int32.self).map(Int.init)
            ?? [Int](repeating: 0, count: inputFeatures.dim(0))

        guard lengths.count == inputFeatures.dim(0) else {
            throw STTError.invalidInput("audio_feature_lengths must contain one length per input feature chunk.")
        }
        guard mapping.count == inputFeatures.dim(0) else {
            throw STTError.invalidInput("audio_chunk_mapping must contain one sample index per input feature chunk.")
        }

        let audioCount = (mapping.max() ?? -1) + 1
        var perAudioChunks = [[MLXArray]](repeating: [], count: audioCount)
        for chunkIndex in 0..<lengths.count {
            let sampleIndex = mapping[chunkIndex]
            let tokenLen = lengths[chunkIndex]
            let frameLen = tokenLen * config.audioMergeSize
            perAudioChunks[sampleIndex].append(
                whisperFeatures[chunkIndex..<(chunkIndex + 1), 0..<frameLen, 0...]
            )
        }

        return perAudioChunks.map { chunks in
            let features = MLX.concatenated(chunks, axis: 1)
            return vqAdaptor(timeMerge(features))
        }
    }

    func injectAudioFeatures(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray,
        inputFeatures: MLXArray,
        audioFeatureLengths: MLXArray,
        audioChunkMapping: MLXArray?
    ) throws -> MLXArray {
        let audioFeatures = try getAudioFeatures(
            inputFeatures: inputFeatures,
            audioFeatureLengths: audioFeatureLengths,
            audioChunkMapping: audioChunkMapping
        )
        let audioEmbeds = MLX.concatenated(audioFeatures.map { $0.squeezed(axis: 0) }, axis: 0)
            .asType(inputsEmbeds.dtype)

        let flatMask = (inputIds .== MLXArray(Int32(config.audioTokenId))).reshaped(-1)
        let maskValues = flatMask.asType(.int32).asArray(Int32.self)
        let audioTokenCount = maskValues.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        guard audioTokenCount == audioEmbeds.dim(0) else {
            throw STTError.invalidInput(
                "Audio features and audio tokens do not match: tokens \(audioTokenCount), features \(audioEmbeds.dim(0))."
            )
        }

        let batchSize = inputsEmbeds.dim(0)
        let seqLen = inputsEmbeds.dim(1)
        let hiddenDim = inputsEmbeds.dim(2)
        let flatEmbeds = inputsEmbeds.reshaped(-1, hiddenDim)

        var pieces: [MLXArray] = []
        var cursor = 0
        var audioIndex = 0
        for (position, value) in maskValues.enumerated() where value != 0 {
            if position > cursor {
                pieces.append(flatEmbeds[cursor..<position])
            }
            pieces.append(audioEmbeds[audioIndex..<(audioIndex + 1)])
            cursor = position + 1
            audioIndex += 1
        }
        if cursor < flatEmbeds.dim(0) {
            pieces.append(flatEmbeds[cursor..<flatEmbeds.dim(0)])
        }

        return MLX.concatenated(pieces, axis: 0).reshaped(batchSize, seqLen, hiddenDim)
    }

    public func callAsFunction(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputFeatures: MLXArray? = nil,
        audioFeatureLengths: MLXArray? = nil,
        audioChunkMapping: MLXArray? = nil
    ) -> MLXArray {
        var embeds = inputsEmbeds ?? languageModel.embedTokens(inputIds)
        if let inputFeatures,
           let audioFeatureLengths,
           cache == nil || cache?.first == nil || cache?.first?.offset == 0 {
            embeds = try! injectAudioFeatures(
                inputIds: inputIds,
                inputsEmbeds: embeds,
                inputFeatures: inputFeatures,
                audioFeatureLengths: audioFeatureLengths,
                audioChunkMapping: audioChunkMapping
            )
        }
        return languageModel(inputsEmbeds: embeds, cache: cache)
    }
}

public final class MossTranscribeDiarizeModel: Module, STTGenerationModel {
    public let config: MossTranscribeDiarizeConfig
    public let vocabSize: Int
    public let sampleRate: Int

    @ModuleInfo(key: "model") var model: MossTranscribeDiarizeBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var tokenizer: Tokenizers.Tokenizer?
    public var audioTokensPerSecond: Float = 12.5
    public var timeMarkerEverySeconds: Int = 5
    public var enableTimeMarker = true
    private var digitTokenIds: [Character: Int] = [:]

    public init(_ config: MossTranscribeDiarizeConfig) {
        self.config = config
        self.vocabSize = config.textConfig.vocabSize
        self.sampleRate = config.sampleRate
        self._model.wrappedValue = MossTranscribeDiarizeBackbone(config)
        if config.textConfig.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                config.textConfig.hiddenSize,
                config.textConfig.vocabSize,
                bias: false
            )
        }
    }

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 2048,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            verbose: false,
            language: nil,
            chunkDuration: 1800.0,
            minChunkDuration: 0.0,
            repetitionPenalty: 1.0,
            repetitionContextSize: 100
        )
    }

    public func makeCache() -> [KVCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    public func callAsFunction(
        inputIds: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        inputFeatures: MLXArray? = nil,
        audioFeatureLengths: MLXArray? = nil,
        audioChunkMapping: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let hiddenStates = model(
            inputIds: inputIds,
            inputsEmbeds: inputEmbeddings,
            cache: cache,
            inputFeatures: inputFeatures,
            audioFeatureLengths: audioFeatureLengths,
            audioChunkMapping: audioChunkMapping
        )
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.languageModel.embedTokens.asLinear(hiddenStates)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        generate(
            audio: audio,
            maxTokens: generationParameters.maxTokens,
            temperature: generationParameters.temperature,
            chunkDuration: generationParameters.chunkDuration,
            minChunkDuration: generationParameters.minChunkDuration,
            repetitionPenalty: generationParameters.repetitionPenalty,
            repetitionContextSize: generationParameters.repetitionContextSize,
            kvBits: generationParameters.kvBits,
            kvGroupSize: generationParameters.kvGroupSize,
            quantizedKVStart: generationParameters.quantizedKVStart
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let sendableModel = UncheckedSendableBox(self)
        let sendableAudio = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                let model = sendableModel.value
                let audio = sendableAudio.value
                do {
                    let start = Date()
                    let chunks = model.audioChunks(
                        audio,
                        chunkDuration: generationParameters.chunkDuration,
                        minChunkDuration: generationParameters.minChunkDuration
                    )
                    var outputs: [STTOutput] = []
                    outputs.reserveCapacity(chunks.count)
                    var emittedText = false

                    for (chunkAudio, offsetSeconds) in chunks {
                        try Task.checkCancellation()
                        var emittedChunkText = false

                        // MOSS emits verbose timestamped diarization text, so the UI's
                        // maxTokens setting is a per-chunk decode cap for long audio.
                        let output = try model.generateSingleChunk(
                            audio: chunkAudio,
                            maxTokens: generationParameters.maxTokens,
                            temperature: generationParameters.temperature,
                            repetitionPenalty: generationParameters.repetitionPenalty,
                            repetitionContextSize: generationParameters.repetitionContextSize,
                            prompt: nil,
                            offsetSeconds: Double(offsetSeconds),
                            kvBits: generationParameters.kvBits,
                            kvGroupSize: generationParameters.kvGroupSize,
                            quantizedKVStart: generationParameters.quantizedKVStart
                        ) { text in
                            guard !text.isEmpty else { return }
                            if !emittedText {
                                continuation.yield(.token(text))
                                emittedText = true
                            } else if !emittedChunkText {
                                continuation.yield(.token("\n" + text))
                            } else {
                                continuation.yield(.token(text))
                            }
                            emittedChunkText = true
                        }
                        outputs.append(output)
                        let elapsed = Date().timeIntervalSince(start)
                        continuation.yield(.info(Self.generationInfo(
                            for: outputs,
                            elapsedTime: elapsed
                        )))

                        Memory.clearCache()
                    }

                    let totalTime = Date().timeIntervalSince(start)
                    let combined = Self.combineChunkOutputs(outputs, totalTime: totalTime)
                    continuation.yield(.info(Self.generationInfo(
                        for: outputs,
                        elapsedTime: totalTime
                    )))
                    continuation.yield(.result(combined))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func generate(
        audio: MLXArray,
        maxTokens: Int = 2048,
        temperature: Float = 0.0,
        chunkDuration: Float = 1800.0,
        minChunkDuration: Float = 0.0,
        repetitionPenalty: Float = 1.0,
        repetitionContextSize: Int = 100,
        prompt: String? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0
    ) -> STTOutput {
        let start = Date()
        do {
            let chunks = audioChunks(
                audio,
                chunkDuration: chunkDuration,
                minChunkDuration: minChunkDuration
            )
            var outputs: [STTOutput] = []
            outputs.reserveCapacity(chunks.count)

            for (chunkAudio, offsetSeconds) in chunks {
                // MOSS emits verbose timestamped diarization text, so maxTokens is a
                // per-chunk decode cap instead of a whole-file budget.
                let output = try generateSingleChunk(
                    audio: chunkAudio,
                    maxTokens: maxTokens,
                    temperature: temperature,
                    repetitionPenalty: repetitionPenalty,
                    repetitionContextSize: repetitionContextSize,
                    prompt: prompt,
                    offsetSeconds: Double(offsetSeconds),
                    kvBits: kvBits,
                    kvGroupSize: kvGroupSize,
                    quantizedKVStart: quantizedKVStart
                )
                outputs.append(output)
                Memory.clearCache()
            }

            return Self.combineChunkOutputs(outputs, totalTime: Date().timeIntervalSince(start))
        } catch {
            fatalError("MOSS-Transcribe-Diarize generation failed: \(error)")
        }
    }
}

private extension MossTranscribeDiarizeModel {
    struct PreparedGenerationInputs {
        let promptIds: MLXArray
        let inputEmbeddings: MLXArray
        let promptTokenCount: Int
        let duration: Double
    }

    func audioToMono(_ audio: MLXArray) throws -> MLXArray {
        guard audio.shape.reduce(1, *) > 0 else {
            throw STTError.invalidInput("Audio must contain at least one sample.")
        }
        if audio.ndim == 1 {
            return audio.asType(.float32)
        }
        if audio.ndim == 2 {
            return audio.mean(axis: -1).asType(.float32)
        }
        return audio.reshaped(-1).asType(.float32)
    }

    func computeAudioTokenLength(numSamples: Int) -> Int {
        let stride = WhisperAudioConfig.hopLength * mossWhisperEncoderStride * config.audioMergeSize
        return (numSamples - 1) / stride + 1
    }

    func preprocessAudio(_ audio: MLXArray) throws -> (
        inputFeatures: MLXArray,
        audioLengths: MLXArray,
        chunkMapping: MLXArray,
        featureLengths: [Int],
        duration: Double
    ) {
        let wav = try audioToMono(audio)
        let sampleCount = wav.dim(0)
        let chunkSamples = WhisperAudioConfig.chunkLengthSamples
        var chunks: [MLXArray] = []
        var featureLengths: [Int] = []
        var chunkMapping: [Int32] = []

        var start = 0
        while start < sampleCount {
            let end = min(start + chunkSamples, sampleCount)
            let chunk = wav[start..<end]
            featureLengths.append(computeAudioTokenLength(numSamples: max(1, end - start)))
            chunks.append(WhisperAudio.encoderFeatures(audio: chunk, nMels: config.audioConfig.numMelBins))
            chunkMapping.append(0)
            start = end
        }

        if chunks.isEmpty {
            chunks.append(WhisperAudio.encoderFeatures(audio: wav, nMels: config.audioConfig.numMelBins))
            featureLengths.append(1)
            chunkMapping.append(0)
        }

        let inputFeatures = MLX.concatenated(chunks, axis: 0)
            .asType(model.whisperEncoder.conv1.weight.dtype)
        return (
            inputFeatures,
            MLXArray(featureLengths.map(Int32.init)),
            MLXArray(chunkMapping),
            featureLengths,
            Double(sampleCount) / Double(sampleRate)
        )
    }

    func audioSpanIds(audioTokenCount: Int) throws -> [Int] {
        guard enableTimeMarker,
              audioTokenCount > 0,
              timeMarkerEverySeconds > 0
        else {
            return [Int](repeating: config.audioTokenId, count: max(audioTokenCount, 0))
        }

        let tokensPerMarker = Int(audioTokensPerSecond * Float(timeMarkerEverySeconds))
        guard tokensPerMarker > 0 else {
            return [Int](repeating: config.audioTokenId, count: audioTokenCount)
        }
        guard !digitTokenIds.isEmpty else {
            throw STTError.modelNotInitialized("Digit token ids are not initialized.")
        }

        let duration = Float(audioTokenCount) / audioTokensPerSecond
        var output: [Int] = []
        var consumed = 0
        var seconds = timeMarkerEverySeconds
        while seconds <= Int(duration) {
            let position = (seconds / timeMarkerEverySeconds) * tokensPerMarker
            let segmentLength = position - consumed
            if segmentLength > 0 {
                output.append(contentsOf: [Int](repeating: config.audioTokenId, count: segmentLength))
                consumed += segmentLength
            }
            for digit in String(seconds) {
                if let token = digitTokenIds[digit] {
                    output.append(token)
                }
            }
            seconds += timeMarkerEverySeconds
        }
        let remainder = audioTokenCount - consumed
        if remainder > 0 {
            output.append(contentsOf: [Int](repeating: config.audioTokenId, count: remainder))
        }
        return output
    }

    func buildPrompt(audioTokenCount: Int, prompt: String?) throws -> MLXArray {
        guard let tokenizer else {
            throw STTError.modelNotInitialized("Tokenizer not loaded.")
        }

        let resolvedPrompt = (prompt?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
            ? prompt!
            : mossDefaultPrompt
        let rendered: String
        if resolvedPrompt.contains(mossAudioPadToken) {
            rendered = resolvedPrompt
        } else {
            rendered = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                + "<|im_start|>user\n"
                + "\(mossAudioStartToken)\(mossAudioPadToken)\(mossAudioEndToken)\n"
                + "\(resolvedPrompt)<|im_end|>\n"
                + "<|im_start|>assistant\n"
        }

        let parts = rendered.components(separatedBy: mossAudioPadToken)
        guard parts.count == 2 else {
            throw STTError.invalidInput("Expected exactly one \(mossAudioPadToken) token in the prompt.")
        }

        let tokenIds = tokenizer.encode(text: parts[0], addSpecialTokens: false)
            + (try audioSpanIds(audioTokenCount: audioTokenCount))
            + tokenizer.encode(text: parts[1], addSpecialTokens: false)
        return MLXArray(tokenIds.map(Int32.init)).expandedDimensions(axis: 0)
    }

    func prepareGenerationInputs(audio: MLXArray, prompt: String?) throws -> PreparedGenerationInputs {
        let (inputFeatures, audioLengths, chunkMapping, featureLengths, duration) = try preprocessAudio(audio)
        let audioTokenCount = featureLengths.reduce(0, +)
        let inputIds = try buildPrompt(audioTokenCount: audioTokenCount, prompt: prompt)
        let embeds = model.languageModel.embedTokens(inputIds)
        let inputsEmbeds = try model.injectAudioFeatures(
            inputIds: inputIds,
            inputsEmbeds: embeds,
            inputFeatures: inputFeatures,
            audioFeatureLengths: audioLengths,
            audioChunkMapping: chunkMapping
        )
        eval(inputsEmbeds)
        return PreparedGenerationInputs(
            promptIds: inputIds,
            inputEmbeddings: inputsEmbeds,
            promptTokenCount: inputIds.dim(1),
            duration: duration
        )
    }

    func audioChunks(
        _ audio: MLXArray,
        chunkDuration: Float,
        minChunkDuration: Float
    ) -> [(MLXArray, Float)] {
        let safeChunkDuration = chunkDuration > 0 ? chunkDuration : defaultGenerationParameters.chunkDuration
        return splitAudioIntoChunks(
            audio,
            sampleRate: sampleRate,
            chunkDuration: safeChunkDuration,
            minChunkDuration: max(0, minChunkDuration)
        )
    }

    func generateSingleChunk(
        audio: MLXArray,
        maxTokens: Int,
        temperature: Float,
        repetitionPenalty: Float,
        repetitionContextSize: Int,
        prompt: String?,
        offsetSeconds: Double,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        onText: ((String) -> Void)? = nil
    ) throws -> STTOutput {
        defer { Memory.clearCache() }

        let start = Date()
        let prefillStart = Date()
        let prepared = try prepareGenerationInputs(audio: audio, prompt: prompt)
        let prefillTime = Date().timeIntervalSince(prefillStart)
        let genStart = Date()
        var offsetter = MossTimestampTagOffsetter(offsetSeconds: offsetSeconds)
        let generatedTokens = try generateTokenIds(
            promptIds: prepared.promptIds,
            inputEmbeddings: prepared.inputEmbeddings,
            maxTokens: maxTokens,
            temperature: temperature,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        ) { token in
            let rawDelta = self.tokenizer?.decode(tokens: [token], skipSpecialTokens: true) ?? ""
            let shiftedDelta = offsetter.consume(rawDelta)
            if !shiftedDelta.isEmpty {
                onText?(shiftedDelta)
            }
        }
        let bufferedText = offsetter.finish()
        if !bufferedText.isEmpty {
            onText?(bufferedText)
        }
        let genTime = Date().timeIntervalSince(genStart)
        let rawText = tokenizer?
            .decode(tokens: generatedTokens, skipSpecialTokens: true)
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let text = Self.offsetTimestampTags(in: rawText, by: offsetSeconds)
        let totalTime = Date().timeIntervalSince(start)
        return STTOutput(
            text: text,
            segments: Self.parseSegments(
                text: rawText,
                fallbackEnd: prepared.duration,
                offsetSeconds: offsetSeconds
            ),
            promptTokens: prepared.promptTokenCount,
            generationTokens: generatedTokens.count,
            totalTokens: prepared.promptTokenCount + generatedTokens.count,
            promptTps: prefillTime > 0 ? Double(prepared.promptTokenCount) / prefillTime : 0,
            generationTps: genTime > 0 ? Double(generatedTokens.count) / genTime : 0,
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    func eosTokenIds() -> Set<Int> {
        [151643, 151645]
    }

    func generateTokenIds(
        promptIds: MLXArray,
        inputEmbeddings: MLXArray,
        maxTokens: Int,
        temperature: Float,
        repetitionPenalty: Float,
        repetitionContextSize: Int,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        onToken: ((Int) -> Void)? = nil
    ) throws -> [Int] {
        var cache = makeCache()
        // Quantized prefill attention is unfused; its transient scores tensor
        // scales with chunk size, so a smaller chunk bounds the memory spike.
        let prefillStepSize = kvBits == nil ? 2048 : 512
        let totalTokens = promptIds.dim(1)
        var processedTokens = 0

        while totalTokens - processedTokens > 1 {
            try Task.checkCancellation()

            let remaining = (totalTokens - processedTokens) - 1
            let n = min(prefillStepSize, remaining)
            let chunkIds = promptIds[0..., processedTokens..<(processedTokens + n)]
            let chunkEmbeds = inputEmbeddings[0..., processedTokens..<(processedTokens + n), 0...]
            let logits = callAsFunction(inputIds: chunkIds, inputEmbeddings: chunkEmbeds, cache: cache)
            eval(logits)
            // Quantize retained context as prefill progresses (as mlx-lm does):
            // a long prompt would otherwise peak at full-precision KV.
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart
            )
            Memory.clearCache()
            processedTokens += n
        }

        let lastIds = promptIds[0..., processedTokens..<totalTokens]
        let lastEmbeds = inputEmbeddings[0..., processedTokens..<totalTokens, 0...]
        var logits = callAsFunction(inputIds: lastIds, inputEmbeddings: lastEmbeds, cache: cache)
        var lastLogits = logits[0..., -1, 0...]
        if temperature > 0 {
            lastLogits = lastLogits / temperature
        }
        var nextTokenArray = lastLogits.argMax(axis: -1)
        asyncEval(nextTokenArray)

        // Covers prompts short enough that the chunk loop never ran.
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )

        var generated: [Int] = []
        let eos = eosTokenIds()

        for tokenIndex in 0..<maxTokens {
            try Task.checkCancellation()

            let token = nextTokenArray.item(Int.self)
            if eos.contains(token) {
                break
            }
            generated.append(token)
            onToken?(token)

            if repetitionPenalty == 1.0 && generated.count >= 24 {
                let tail = generated.suffix(24)
                if Set(tail).count <= 3 {
                    break
                }
            }
            if tokenIndex == maxTokens - 1 {
                break
            }

            let nextInput = MLXArray([Int32(token)]).expandedDimensions(axis: 0)
            logits = callAsFunction(inputIds: nextInput, cache: cache)
            lastLogits = logits[0..., -1, 0...]
            if temperature > 0 {
                lastLogits = lastLogits / temperature
            }
            if repetitionPenalty != 1.0 && !generated.isEmpty {
                let recent = Array(generated.suffix(max(1, repetitionContextSize))).map(Int32.init)
                let recentArray = MLXArray(recent)
                let logitsForRecent = lastLogits[0..., recentArray]
                let penalty = MLXArray(repetitionPenalty)
                lastLogits[0..., recentArray] = MLX.where(
                    logitsForRecent .> 0,
                    logitsForRecent / penalty,
                    logitsForRecent * penalty
                )
            }
            nextTokenArray = lastLogits.argMax(axis: -1)
            asyncEval(nextTokenArray)

            if tokenIndex > 0 && tokenIndex % 256 == 0 {
                Memory.clearCache()
            }
        }
        return generated
    }
}

extension MossTranscribeDiarizeModel {
    func streamingTranscribeWindow(
        audio: MLXArray,
        offsetSeconds: Double,
        config: StreamingConfig,
        maxTokens: Int? = nil,
        onText: ((String) -> Void)? = nil
    ) throws -> STTOutput {
        let defaults = defaultGenerationParameters
        let audioSeconds = Double(audio.dim(0)) / Double(sampleRate)
        let estimatedWindowTokens = max(96, Int(ceil(audioSeconds * 32.0)))
        let requestedMaxTokens = maxTokens ?? min(max(1, config.maxTokensPerPass), estimatedWindowTokens)
        return try generateSingleChunk(
            audio: audio,
            maxTokens: max(1, requestedMaxTokens),
            temperature: config.temperature,
            repetitionPenalty: defaults.repetitionPenalty,
            repetitionContextSize: defaults.repetitionContextSize,
            prompt: nil,
            offsetSeconds: offsetSeconds,
            onText: onText
        )
    }

    static func combineChunkOutputs(_ outputs: [STTOutput], totalTime: TimeInterval) -> STTOutput {
        let text = outputs
            .map(\.text)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        let segments = outputs.flatMap { $0.segments ?? [] }
        let promptTokens = outputs.reduce(0) { $0 + $1.promptTokens }
        let generationTokens = outputs.reduce(0) { $0 + $1.generationTokens }
        let totalTokens = outputs.reduce(0) { $0 + $1.totalTokens }
        let peakMemoryUsage = outputs.map(\.peakMemoryUsage).max() ?? 0

        return STTOutput(
            text: text,
            segments: segments.isEmpty ? nil : segments,
            promptTokens: promptTokens,
            generationTokens: generationTokens,
            totalTokens: totalTokens,
            promptTps: totalTime > 0 ? Double(promptTokens) / totalTime : 0,
            generationTps: totalTime > 0 ? Double(generationTokens) / totalTime : 0,
            totalTime: totalTime,
            peakMemoryUsage: peakMemoryUsage
        )
    }

    static func generationInfo(for outputs: [STTOutput], elapsedTime: TimeInterval) -> STTGenerationInfo {
        let promptTokens = outputs.reduce(0) { $0 + $1.promptTokens }
        let generationTokens = outputs.reduce(0) { $0 + $1.generationTokens }
        let peakMemoryUsage = outputs.map(\.peakMemoryUsage).max() ?? 0

        return STTGenerationInfo(
            promptTokenCount: promptTokens,
            generationTokenCount: generationTokens,
            prefillTime: 0,
            generateTime: elapsedTime,
            tokensPerSecond: elapsedTime > 0 ? Double(generationTokens) / elapsedTime : 0,
            peakMemoryUsage: peakMemoryUsage
        )
    }

    static func offsetTimestampTags(in text: String, by offsetSeconds: Double) -> String {
        guard offsetSeconds != 0 else { return text }
        let pattern = #"\[(\d+(?:[\.,]\d+)?)\]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let timestampLocale = Locale(identifier: "en_US_POSIX")

        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            options: [],
            range: NSRange(location: 0, length: nsText.length)
        )
        guard !matches.isEmpty else { return text }

        var output = ""
        var cursor = text.startIndex
        for match in matches {
            guard
                let fullRange = Range(match.range, in: text),
                let valueRange = Range(match.range(at: 1), in: text),
                let value = Self.timestampValue(String(text[valueRange]))
            else {
                continue
            }

            output += text[cursor..<fullRange.lowerBound]
            output += String(format: "[%.2f]", locale: timestampLocale, value + offsetSeconds)
            cursor = fullRange.upperBound
        }
        output += text[cursor..<text.endIndex]
        return output
    }

    static func parseSegments(
        text: String,
        fallbackEnd: Double,
        offsetSeconds: Double = 0
    ) -> [[String: Any]] {
        let pattern = #"\[(\d+(?:[\.,]\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:[\.,]\d+)?)\]"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else {
            return [["start": offsetSeconds, "end": offsetSeconds + max(fallbackEnd, 0.0), "text": text]]
        }

        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            options: [],
            range: NSRange(location: 0, length: nsText.length)
        )
        var segments: [[String: Any]] = []
        for match in matches {
            guard match.numberOfRanges == 5,
                  let start = Self.timestampValue(nsText.substring(with: match.range(at: 1))),
                  let end = Self.timestampValue(nsText.substring(with: match.range(at: 4))),
                  end >= start
            else {
                continue
            }
            let speaker = nsText.substring(with: match.range(at: 2))
            let segmentText = nsText.substring(with: match.range(at: 3))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !segmentText.isEmpty else {
                continue
            }
            segments.append([
                "start": start + offsetSeconds,
                "end": end + offsetSeconds,
                "text": "[\(speaker)] \(segmentText)",
                "speaker_id": speaker,
            ])
        }

        if !segments.isEmpty {
            return segments
        }
        return [["start": offsetSeconds, "end": offsetSeconds + max(fallbackEnd, 0.0), "text": text]]
    }

    private static func timestampValue(_ text: String) -> Double? {
        Double(text.replacingOccurrences(of: ",", with: "."))
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let alreadyConverted = weights.keys.contains { $0.contains("scales") }
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (rawKey, rawValue) in weights {
            if rawKey == "lm_head.weight" {
                continue
            }
            var key = rawKey
            var value = rawValue

            if key.hasPrefix("model.vq_adwaptor.") {
                key = key.replacingOccurrences(
                    of: "model.vq_adwaptor.",
                    with: "model.vq_adaptor.",
                    options: [.anchored]
                )
            }
            if key.hasPrefix("model.vq_adaptor.layers.") && !key.hasPrefix("model.vq_adaptor.layers.layers.") {
                key = key.replacingOccurrences(
                    of: "model.vq_adaptor.layers.",
                    with: "model.vq_adaptor.layers.layers.",
                    options: [.anchored]
                )
            }
            if key.hasPrefix("model.vq_adaptor.layers.layers.layers.") {
                key = key.replacingOccurrences(
                    of: "model.vq_adaptor.layers.layers.layers.",
                    with: "model.vq_adaptor.layers.layers.",
                    options: [.anchored]
                )
            }

            if !alreadyConverted,
               key.hasPrefix("model.whisper_encoder."),
               key.contains("conv"),
               key.hasSuffix(".weight"),
               value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
            sanitized[key] = value
        }
        return sanitized
    }

    public static func fromPretrained(
        _ modelPath: String = mossTranscribeDiarizeDefaultRepo,
        cache: HubCache = .default
    ) async throws -> MossTranscribeDiarizeModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "MossTranscribeDiarizeModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> MossTranscribeDiarizeModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MossTranscribeDiarizeConfig.self, from: configData)
        let model = MossTranscribeDiarizeModel(config)

        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        try model.loadProcessorConfig(from: modelDir)
        try model.initializeDigitTokenIds()

        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir,
            includingPropertiesForKeys: nil
        )
        let safetensors = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !safetensors.isEmpty else {
            throw NSError(
                domain: "MossTranscribeDiarizeModel",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No .safetensors files found in \(modelDir.path)."]
            )
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }
        let sanitized = sanitize(weights: weights)
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                // Only layers shipping quantized tensors are swapped.
                guard sanitized["\(path).scales"] != nil else {
                    return nil
                }
                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path) {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        model.train(false)
        eval(model)
        return model
    }

    func loadProcessorConfig(from modelDir: URL) throws {
        let processorURL = modelDir.appendingPathComponent("processor_config.json")
        guard FileManager.default.fileExists(atPath: processorURL.path) else {
            return
        }
        let data = try Data(contentsOf: processorURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        if let value = object["audio_tokens_per_second"] as? NSNumber {
            audioTokensPerSecond = value.floatValue
        }
        if let value = object["time_marker_every_seconds"] as? NSNumber {
            timeMarkerEverySeconds = value.intValue
        }
        if let value = object["enable_time_marker"] as? Bool {
            enableTimeMarker = value
        }
    }

    func initializeDigitTokenIds() throws {
        guard let tokenizer else {
            throw STTError.modelNotInitialized("Tokenizer not loaded.")
        }
        var ids: [Character: Int] = [:]
        for digit in "0123456789" {
            let encoded = tokenizer.encode(text: String(digit), addSpecialTokens: false)
            guard encoded.count == 1, let token = encoded.first else {
                throw STTError.invalidInput("Digit \(digit) is not a single token: \(encoded).")
            }
            ids[digit] = token
        }
        digitTokenIds = ids
    }
}
