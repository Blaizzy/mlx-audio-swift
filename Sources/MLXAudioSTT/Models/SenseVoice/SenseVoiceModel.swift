import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

final class SenseVoiceSinusoidalPositionEncoder {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape[0]
        let timesteps = x.shape[1]
        let inputDim = x.shape[2]
        let halfDim = max(inputDim / 2, 1)

        let positions = MLXArray((1...timesteps).map(Float.init))
        let logIncrement = Float(log(10000.0) / Double(max(halfDim - 1, 1)))
        let invTimescales = MLX.exp(MLX.arange(halfDim, dtype: .float32) * MLXArray(-logIncrement))
        let scaledTime = positions.expandedDimensions(axis: 1) * invTimescales.expandedDimensions(axis: 0)

        var encoding = MLX.concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
        if encoding.shape[1] > inputDim {
            encoding = encoding[0..., 0..<inputDim]
        } else if encoding.shape[1] < inputDim {
            let pad = MLXArray.zeros([timesteps, inputDim - encoding.shape[1]], type: Float.self)
            encoding = MLX.concatenated([encoding, pad], axis: 1)
        }

        let batched = MLX.broadcast(encoding.expandedDimensions(axis: 0), to: [batchSize, timesteps, inputDim])
        return x + batched.asType(x.dtype)
    }
}

final class SenseVoicePositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(idim: Int, hiddenUnits: Int) {
        self._w1.wrappedValue = Linear(idim, hiddenUnits)
        self._w2.wrappedValue = Linear(hiddenUnits, idim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(relu(w1(x)))
    }
}

final class SenseVoiceMultiHeadedAttentionSANM: Module {
    let dK: Int
    let heads: Int
    let nFeat: Int
    let leftPadding: Int
    let rightPadding: Int

    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_q_k_v") var linearQKV: Linear
    @ModuleInfo(key: "fsmn_block") var fsmnBlock: Conv1d

    init(
        nHead: Int,
        inFeat: Int,
        nFeat: Int,
        kernelSize: Int = 11,
        sanmShift: Int = 0
    ) {
        self.dK = nFeat / nHead
        self.heads = nHead
        self.nFeat = nFeat

        self._linearOut.wrappedValue = Linear(nFeat, nFeat)
        self._linearQKV.wrappedValue = Linear(inFeat, nFeat * 3)
        self._fsmnBlock.wrappedValue = Conv1d(
            inputChannels: nFeat,
            outputChannels: nFeat,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            groups: nFeat,
            bias: false
        )

        var leftPadding = (kernelSize - 1) / 2
        if sanmShift > 0 {
            leftPadding += sanmShift
        }
        self.leftPadding = leftPadding
        self.rightPadding = kernelSize - 1 - leftPadding
    }

    private func forwardFSMN(_ inputs: MLXArray) -> MLXArray {
        let padded = MLX.padded(
            inputs,
            widths: [.init(0), .init((leftPadding, rightPadding)), .init(0)]
        )
        let memory = fsmnBlock(padded)
        return memory + inputs
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]

        let qkv = linearQKV(x)
        let split = MLX.split(qkv, parts: 3, axis: -1)
        let q = split[0]
        let k = split[1]
        let v = split[2]

        let fsmnMemory = forwardFSMN(v)

        let qHeads = q.reshaped([batch, time, heads, dK]).transposed(0, 2, 1, 3)
        let kHeads = k.reshaped([batch, time, heads, dK]).transposed(0, 2, 1, 3)
        let vHeads = v.reshaped([batch, time, heads, dK]).transposed(0, 2, 1, 3)

        let attOut = MLXFast.scaledDotProductAttention(
            queries: qHeads,
            keys: kHeads,
            values: vHeads,
            scale: 1.0 / sqrt(Float(dK)),
            mask: nil
        )

        var merged = attOut.transposed(0, 2, 1, 3).reshaped([batch, time, nFeat])
        merged = linearOut(merged)
        return merged + fsmnMemory
    }
}

final class SenseVoiceEncoderLayerSANM: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SenseVoiceMultiHeadedAttentionSANM
    @ModuleInfo(key: "feed_forward") var feedForward: SenseVoicePositionwiseFeedForward
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    let inSize: Int
    let size: Int
    let normalizeBefore: Bool

    init(
        inSize: Int,
        size: Int,
        selfAttn: SenseVoiceMultiHeadedAttentionSANM,
        feedForward: SenseVoicePositionwiseFeedForward,
        normalizeBefore: Bool = true
    ) {
        self.inSize = inSize
        self.size = size
        self.normalizeBefore = normalizeBefore
        self._selfAttn.wrappedValue = selfAttn
        self._feedForward.wrappedValue = feedForward
        self._norm1.wrappedValue = LayerNorm(dimensions: inSize)
        self._norm2.wrappedValue = LayerNorm(dimensions: size)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = x
        if normalizeBefore {
            y = norm1(y)
        }

        let attnOut = selfAttn(y)
        if inSize == size {
            y = residual + attnOut
        } else {
            y = attnOut
        }

        let ffResidual = y
        if normalizeBefore {
            y = norm2(y)
        }

        y = ffResidual + feedForward(y)
        return y
    }
}

final class SenseVoiceEncoder: Module {
    let outputSize: Int
    let embed: SenseVoiceSinusoidalPositionEncoder

    @ModuleInfo(key: "encoders0") var encoders0: [SenseVoiceEncoderLayerSANM]
    @ModuleInfo(key: "encoders") var encoders: [SenseVoiceEncoderLayerSANM]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm
    @ModuleInfo(key: "tp_encoders") var tpEncoders: [SenseVoiceEncoderLayerSANM]
    @ModuleInfo(key: "tp_norm") var tpNorm: LayerNorm

    init(config: SenseVoiceConfig) {
        let enc = config.encoderConf
        self.outputSize = enc.outputSize
        self.embed = SenseVoiceSinusoidalPositionEncoder()

        self._encoders0.wrappedValue = [
            SenseVoiceEncoderLayerSANM(
                inSize: config.inputSize,
                size: enc.outputSize,
                selfAttn: SenseVoiceMultiHeadedAttentionSANM(
                    nHead: enc.attentionHeads,
                    inFeat: config.inputSize,
                    nFeat: enc.outputSize,
                    kernelSize: enc.kernelSize,
                    sanmShift: enc.sanmShift
                ),
                feedForward: SenseVoicePositionwiseFeedForward(
                    idim: enc.outputSize,
                    hiddenUnits: enc.linearUnits
                ),
                normalizeBefore: enc.normalizeBefore
            )
        ]

        self._encoders.wrappedValue = (0..<max(enc.numBlocks - 1, 0)).map { _ in
            SenseVoiceEncoderLayerSANM(
                inSize: enc.outputSize,
                size: enc.outputSize,
                selfAttn: SenseVoiceMultiHeadedAttentionSANM(
                    nHead: enc.attentionHeads,
                    inFeat: enc.outputSize,
                    nFeat: enc.outputSize,
                    kernelSize: enc.kernelSize,
                    sanmShift: enc.sanmShift
                ),
                feedForward: SenseVoicePositionwiseFeedForward(
                    idim: enc.outputSize,
                    hiddenUnits: enc.linearUnits
                ),
                normalizeBefore: enc.normalizeBefore
            )
        }

        self._afterNorm.wrappedValue = LayerNorm(dimensions: enc.outputSize)
        self._tpEncoders.wrappedValue = (0..<enc.tpBlocks).map { _ in
            SenseVoiceEncoderLayerSANM(
                inSize: enc.outputSize,
                size: enc.outputSize,
                selfAttn: SenseVoiceMultiHeadedAttentionSANM(
                    nHead: enc.attentionHeads,
                    inFeat: enc.outputSize,
                    nFeat: enc.outputSize,
                    kernelSize: enc.kernelSize,
                    sanmShift: enc.sanmShift
                ),
                feedForward: SenseVoicePositionwiseFeedForward(
                    idim: enc.outputSize,
                    hiddenUnits: enc.linearUnits
                ),
                normalizeBefore: enc.normalizeBefore
            )
        }
        self._tpNorm.wrappedValue = LayerNorm(dimensions: enc.outputSize)
    }

    func callAsFunction(_ xsPad: MLXArray) -> MLXArray {
        var hidden = xsPad * MLXArray(Float(sqrt(Double(outputSize))))
        hidden = embed(hidden)

        for layer in encoders0 {
            hidden = layer(hidden)
        }
        for layer in encoders {
            hidden = layer(hidden)
        }
        hidden = afterNorm(hidden)
        for layer in tpEncoders {
            hidden = layer(hidden)
        }
        return tpNorm(hidden)
    }
}

public final class SenseVoiceModel: Module, STTGenerationModel {
    public let config: SenseVoiceConfig
    let blankID = 0

    @ModuleInfo(key: "encoder") var encoder: SenseVoiceEncoder
    @ModuleInfo(key: "ctc_lo") var ctcLo: Linear
    @ModuleInfo(key: "embed") var embed: Embedding

    var cmvnMeans: MLXArray?
    var cmvnIstd: MLXArray?
    var tokenizer: SenseVoiceTokenizer?

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 0,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            verbose: false,
            language: "auto"
        )
    }

    public init(_ config: SenseVoiceConfig) {
        self.config = config
        self._encoder.wrappedValue = SenseVoiceEncoder(config: config)
        self._ctcLo.wrappedValue = Linear(config.encoderConf.outputSize, config.vocabSize)
        self._embed.wrappedValue = Embedding(embeddingCount: 16, dimensions: config.inputSize)
    }

    private var lidDict: [String: Int] {
        [
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        ]
    }

    private var textnormDict: [String: Int] {
        [
            "withitn": 14,
            "woitn": 15,
        ]
    }

    private func extractFeatures(_ audio: MLXArray) -> MLXArray {
        let frontend = config.frontendConf
        var feats = SenseVoiceAudio.computeFbank(
            audio,
            sampleRate: frontend.fs,
            nMels: frontend.nMels,
            frameLengthMS: frontend.frameLength,
            frameShiftMS: frontend.frameShift,
            window: frontend.window
        )
        feats = SenseVoiceAudio.applyLFR(feats, lfrM: frontend.lfrM, lfrN: frontend.lfrN)
        if let cmvnMeans, let cmvnIstd {
            feats = SenseVoiceAudio.applyCMVN(feats, means: cmvnMeans, istd: cmvnIstd)
        }
        return feats
    }

    private func normalizedLanguage(_ language: String?) -> String {
        guard let language else { return "auto" }
        switch language.lowercased() {
        case "zh", "chinese", "mandarin":
            return "zh"
        case "en", "english":
            return "en"
        case "yue", "cantonese":
            return "yue"
        case "ja", "japanese":
            return "ja"
        case "ko", "korean":
            return "ko"
        case "nospeech":
            return "nospeech"
        default:
            return "auto"
        }
    }

    private func buildQuery(batchSize: Int, language: String = "auto", useITN: Bool = false) -> (MLXArray, MLXArray) {
        let lid = lidDict[language] ?? 0
        var languageQuery = embed(MLXArray([Int32(lid)]).reshaped([1, 1]))

        let textnorm = useITN ? "withitn" : "woitn"
        let textnormID = textnormDict[textnorm] ?? 15
        var textnormQuery = embed(MLXArray([Int32(textnormID)]).reshaped([1, 1]))

        var eventEmoQuery = embed(MLXArray([Int32(1), Int32(2)]).reshaped([1, 2]))

        if batchSize > 1 {
            languageQuery = MLX.broadcast(languageQuery, to: [batchSize, languageQuery.shape[1], languageQuery.shape[2]])
            textnormQuery = MLX.broadcast(textnormQuery, to: [batchSize, textnormQuery.shape[1], textnormQuery.shape[2]])
            eventEmoQuery = MLX.broadcast(eventEmoQuery, to: [batchSize, eventEmoQuery.shape[1], eventEmoQuery.shape[2]])
        }

        let inputQuery = MLX.concatenated([languageQuery, eventEmoQuery], axis: 1)
        return (textnormQuery, inputQuery)
    }

    public func callAsFunction(
        _ feats: MLXArray,
        language: String = "auto",
        useITN: Bool = false
    ) -> MLXArray {
        let batch = feats.shape[0]
        let (textnormQuery, inputQuery) = buildQuery(batchSize: batch, language: language, useITN: useITN)
        var speech = MLX.concatenated([textnormQuery, feats], axis: 1)
        speech = MLX.concatenated([inputQuery, speech], axis: 1)

        let encoderOut = encoder(speech)
        let logits = ctcLo(encoderOut)
        return logSoftmax(logits, axis: -1)
    }

    private func greedyCTCDecode(_ logProbs: MLXArray) -> (tokenIDs: [Int], text: String) {
        let prediction = MLX.argMax(logProbs, axis: -1).asArray(Int32.self).map(Int.init)
        var deduped: [Int] = []
        var previous: Int?
        for token in prediction {
            if token != previous {
                deduped.append(token)
                previous = token
            }
        }

        let tokenIDs = deduped.filter { $0 != blankID }
        return (tokenIDs, tokenizer?.decode(tokenIDs) ?? tokenIDs.map(String.init).joined(separator: " "))
    }

    private func extractRichInfo(_ logProbs: MLXArray) -> [String: String] {
        let lidPred = logProbs[0].argMax(axis: -1).item(Int.self)
        let emoPred = logProbs[1].argMax(axis: -1).item(Int.self)
        let eventPred = logProbs[2].argMax(axis: -1).item(Int.self)

        let lidMap: [Int: String] = [
            24884: "zh",
            24885: "en",
            24888: "yue",
            24892: "ja",
            24896: "ko",
            24992: "nospeech",
        ]
        let emoMap: [Int: String] = [
            25001: "happy",
            25002: "sad",
            25003: "angry",
            25004: "neutral",
            25005: "fearful",
            25006: "disgusted",
            25007: "surprised",
            25008: "other",
            25009: "unk",
        ]
        let eventMap: [Int: String] = [
            24993: "Speech",
            24995: "BGM",
            24997: "Laughter",
            24999: "Applause",
        ]

        return [
            "language": lidMap[lidPred] ?? "unknown",
            "emotion": emoMap[emoPred] ?? "token_\(emoPred)",
            "event": eventMap[eventPred] ?? "token_\(eventPred)",
        ]
    }

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        generate(
            audio: audio,
            language: normalizedLanguage(generationParameters.language),
            useITN: false,
            verbose: generationParameters.verbose
        )
    }

    public func generate(
        audio: MLXArray,
        language: String = "auto",
        useITN: Bool = false,
        verbose: Bool = false
    ) -> STTOutput {
        let features = extractFeatures(audio).expandedDimensions(axis: 0)
        let logProbs = self(features, language: language, useITN: useITN)[0]
        let richInfo = extractRichInfo(logProbs[0..<4])
        let decoded = greedyCTCDecode(logProbs[4...])

        if verbose {
            print("Language: \(richInfo["language"] ?? "?")")
            print("Emotion: \(richInfo["emotion"] ?? "?")")
            print("Event: \(richInfo["event"] ?? "?")")
            print("Text: \(decoded.text)")
        }

        return STTOutput(
            text: decoded.text,
            segments: [[
                "text": decoded.text,
                "language": richInfo["language"] as Any,
                "emotion": richInfo["emotion"] as Any,
                "event": richInfo["event"] as Any,
            ]],
            language: richInfo["language"]
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let output = generate(audio: audio, generationParameters: generationParameters)
            continuation.yield(.result(output))
            continuation.finish()
        }
    }

    private func loadAssets(from modelDirectory: URL) throws {
        let mvnURL = modelDirectory.appendingPathComponent("am.mvn")
        if FileManager.default.fileExists(atPath: mvnURL.path) {
            let parsed = try SenseVoiceAudio.parseAMMVN(mvnURL)
            cmvnMeans = MLXArray(parsed.means)
            cmvnIstd = MLXArray(parsed.istd)
        } else if let means = config.cmvnMeans, let istd = config.cmvnIstd {
            cmvnMeans = MLXArray(means)
            cmvnIstd = MLXArray(istd)
        }

        let tokenizer = try? SenseVoiceTokenizer(modelDirectory: modelDirectory)
        if tokenizer?.tokenizer != nil || tokenizer?.tokenList != nil {
            self.tokenizer = tokenizer
        }
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            let newKey = key.replacingOccurrences(of: "ctc.ctc_lo.", with: "ctc_lo.")
            var newValue = value

            if newKey.contains("fsmn_block.weight"), newValue.ndim == 3 {
                newValue = newValue.transposed(0, 2, 1)
            }

            sanitized[newKey] = newValue
        }

        return sanitized
    }

    public static func fromDirectory(_ modelDirectory: URL) throws -> SenseVoiceModel {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(SenseVoiceConfig.self, from: configData)
        let model = SenseVoiceModel(config)

        let files = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        let safetensors = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var weights: [String: MLXArray] = [:]
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitize(weights: weights)),
            verify: .all
        )
        try model.loadAssets(from: modelDirectory)
        eval(model)
        return model
    }

    public static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> SenseVoiceModel {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "SenseVoiceModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDirectory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn", "*.model", "tokenizer*"],
            cache: cache
        )
        return try fromDirectory(modelDirectory)
    }
}
