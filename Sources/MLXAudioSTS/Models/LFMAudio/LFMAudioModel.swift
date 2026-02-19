import Foundation
import Hub
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN

// MARK: - Constants

public enum LFMModality: Int, Sendable {
    case text = 1
    case audioIn = 2
    case audioOut = 3
}

public let lfmAudioStartToken: Int = 128
public let lfmImEndToken: Int = 7
public let lfmTextEndToken: Int = 130
public let lfmAudioEOSToken: Int = 2048

// MARK: - Generation Config

public struct LFMGenerationConfig: Sendable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var audioTemperature: Float
    public var audioTopK: Int

    public init(
        maxNewTokens: Int = 512, temperature: Float = 1.0,
        topK: Int = 50, topP: Float = 1.0,
        audioTemperature: Float = 1.0, audioTopK: Int = 4
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.audioTemperature = audioTemperature
        self.audioTopK = audioTopK
    }
}

// MARK: - Generation Output

public enum LFMGenerationOutput: @unchecked Sendable {
    case text(MLXArray)
    case audio(MLXArray)
}

// MARK: - Audio Embedding

class AudioEmbedding: Module {
    let vocabSize: Int
    let dim: Int
    let numCodebooks: Int

    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "to_logits") var toLogits: Linear

    let codebookOffsets: MLXArray

    init(vocabSize: Int, dim: Int, numCodebooks: Int = 8, tie: Bool = false) {
        self.vocabSize = vocabSize
        self.dim = dim
        self.numCodebooks = numCodebooks

        let totalVocab = vocabSize * numCodebooks
        self._embedding.wrappedValue = Embedding(embeddingCount: totalVocab, dimensions: dim)
        self._embeddingNorm.wrappedValue = RMSNorm(dimensions: dim)
        self._toLogits.wrappedValue = Linear(dim, totalVocab, bias: false)

        self.codebookOffsets = MLXArray(
            (0..<numCodebooks).map { Int32($0 * vocabSize) }
        )
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        var c = codes
        if c.ndim == 1 { c = c.expandedDimensions(axis: 0) }
        let K = c.dim(1)
        let offsetCodes = c + codebookOffsets[..<K]
        let embedded = embedding(offsetCodes).sum(axis: 1)
        return c.ndim == 1 ? embedded.squeezed(axis: 0) : embedded
    }
}

// MARK: - Audio Embedding With Norm

class AudioEmbeddingWithNorm: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "to_logits") var toLogits: Linear

    init(vocabSize: Int, dim: Int) {
        self._embedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self._embeddingNorm.wrappedValue = RMSNorm(dimensions: dim)
        self._toLogits.wrappedValue = Linear(dim, vocabSize, bias: false)
    }

    func embed(_ x: MLXArray) -> MLXArray {
        embeddingNorm(embedding(x))
    }

    func embedRaw(_ x: MLXArray) -> MLXArray {
        embedding(x)
    }

    func logits(_ x: MLXArray) -> MLXArray {
        toLogits(x)
    }
}

// MARK: - Audio Head

class AudioHead: Module {
    let numCodebooks: Int
    let depthformerDim: Int

    @ModuleInfo(key: "depthformer") var depthformer: Depthformer

    init(inputDim: Int, config: DepthformerConfig, numCodebooks: Int = 8) {
        self.numCodebooks = numCodebooks
        self.depthformerDim = config.dim

        self._depthformer.wrappedValue = Depthformer(
            layers: config.layers, dim: config.dim,
            numHeads: config.numHeads, numKvHeads: config.numKvHeads,
            tie: config.tie
        )
    }

    func callAsFunction(
        _ x: MLXArray, cache: [(MLXArray, MLXArray)?]? = nil, useCache: Bool = false
    ) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        var h = x.reshaped(B, L, numCodebooks, depthformerDim)
        h = h.transposed(0, 2, 1, 3)
        h = h.reshaped(B * numCodebooks, L, depthformerDim)

        let (out, newCache) = depthformer(h, cache: cache, useCache: useCache)

        var result = out.reshaped(B, numCodebooks, L, depthformerDim)
        result = result.transposed(0, 2, 1, 3)
        return (result, newCache)
    }
}

// MARK: - LFM2 Audio Model

public class LFM2AudioModel: Module, STSModel, @unchecked Sendable {
    public let config: LFM2AudioConfig
    public var processor: LFM2AudioProcessor?
    public var modelDirectory: URL?

    @ModuleInfo(key: "audio_encoder") var audioEncoder: ConformerEncoder
    @ModuleInfo(key: "audio_adapter") var audioAdapter: AdapterMLP
    @ModuleInfo(key: "lfm") var lfm: Lfm2Model
    @ModuleInfo(key: "audio_embedding") var audioEmbedding: AudioEmbedding
    @ModuleInfo(key: "depth_embeddings") var depthEmbeddings: [AudioEmbeddingWithNorm]
    @ModuleInfo(key: "depth_linear") var depthLinear: Linear
    @ModuleInfo(key: "audio_head") var audioHead: AudioHead

    public init(_ config: LFM2AudioConfig) {
        self.config = config

        self._audioEncoder.wrappedValue = ConformerEncoder(config.encoder)
        self._audioAdapter.wrappedValue = AdapterMLP(
            inChannels: config.encoder.dModel,
            outChannels: config.lfm.hiddenSize,
            hiddenDims: config.adapterHiddenDims,
            useLayerNorm: config.adapterUseLayerNorm,
            dropout: config.adapterDropout
        )
        self._lfm.wrappedValue = Lfm2Model(config.lfm)
        self._audioEmbedding.wrappedValue = AudioEmbedding(
            vocabSize: config.audioVocabSize,
            dim: config.lfm.hiddenSize,
            numCodebooks: config.codebooks,
            tie: config.tieAudioEmbeddings
        )

        self._depthEmbeddings.wrappedValue = (0..<config.codebooks).map { _ in
            AudioEmbeddingWithNorm(vocabSize: config.audioVocabSize, dim: config.depthformer.dim)
        }

        self._depthLinear.wrappedValue = Linear(
            config.lfm.hiddenSize, config.codebooks * config.depthformer.dim
        )
        self._audioHead.wrappedValue = AudioHead(
            inputDim: config.lfm.hiddenSize, config: config.depthformer,
            numCodebooks: config.codebooks
        )
    }

    // MARK: - Encoding

    func encodeAudio(_ melFeatures: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (encoded, newLengths) = audioEncoder(melFeatures, lengths: lengths)
        let adapted = audioAdapter(encoded)
        return (adapted, newLengths)
    }

    func embedText(_ inputIds: MLXArray) -> MLXArray {
        lfm.embedTokens(inputIds)
    }

    func embedAudioIn(_ audioCodes: MLXArray) -> MLXArray {
        audioEmbedding(audioCodes)
    }

    func embedAudioOut(_ audioCodes: MLXArray) -> MLXArray {
        audioEmbedding(audioCodes)
    }

    // MARK: - Prefill

    func prefill(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> (MLXArray, [KVCache]) {

        let inputEmbeddings: MLXArray

        if let modalities = modalities {
            inputEmbeddings = buildInterleavedEmbeddings(
                textTokens: textTokens, audioFeatures: audioFeatures,
                audioCodes: audioCodes, modalities: modalities
            )
        } else {
            var embeddings: [MLXArray] = []
            if let t = textTokens { embeddings.append(embedText(t)) }
            if let a = audioFeatures { embeddings.append(encodeAudio(a).0) }
            if let ac = audioCodes {
                let (B, T, _) = (ac.dim(0), ac.dim(1), ac.dim(2))
                var audioOutEmb = MLXArray.zeros([B, T, config.lfm.hiddenSize])
                for t in 0..<T {
                    audioOutEmb = audioOutEmb.at[0..., t..<(t+1), 0...].add(
                        embedAudioOut(ac[0..., t, 0...]).expandedDimensions(axis: 1)
                    )
                }
                embeddings.append(audioOutEmb)
            }
            inputEmbeddings = embeddings.count > 1
                ? concatenated(embeddings, axis: 1)
                : embeddings[0]
        }

        let effectiveCache = cache ?? lfm.makeCache()
        let hiddenStates = lfm(inputEmbeddings: inputEmbeddings, cache: effectiveCache)
        return (hiddenStates, effectiveCache)
    }

    // MARK: - Interleaved Embeddings

    private func buildInterleavedEmbeddings(
        textTokens: MLXArray?, audioFeatures: MLXArray?,
        audioCodes: MLXArray?, modalities: MLXArray
    ) -> MLXArray {
        let B = modalities.dim(0)
        let TTotal = modalities.dim(1)
        let D = config.lfm.hiddenSize

        let modsFlat: [Int] = modalities[0].asArray(Int.self)
        let uniqueMods = Set(modsFlat)

        if uniqueMods == [LFMModality.text.rawValue], let t = textTokens {
            return embedText(t)
        }
        if uniqueMods == [LFMModality.audioIn.rawValue], let a = audioFeatures {
            return encodeAudio(a).0
        }

        let textEmb = textTokens.map { embedText($0) }
        let audioEmb = audioFeatures.map { encodeAudio($0).0 }
        var audioOutEmb: MLXArray? = nil
        if let ac = audioCodes {
            let (_, TAudio, _) = (ac.dim(0), ac.dim(1), ac.dim(2))
            var parts: [MLXArray] = []
            for t in 0..<TAudio {
                parts.append(embedAudioOut(ac[0..., t, 0...]))
            }
            audioOutEmb = MLX.stacked(parts, axis: 1)
        }

        var textPos: [Int] = [], audioInPos: [Int] = [], audioOutPos: [Int] = []
        for (pos, mod) in modsFlat.enumerated() {
            switch mod {
            case LFMModality.text.rawValue: textPos.append(pos)
            case LFMModality.audioIn.rawValue: audioInPos.append(pos)
            case LFMModality.audioOut.rawValue: audioOutPos.append(pos)
            default: break
            }
        }

        var embeddings = MLXArray.zeros([B, TTotal, D])

        if let te = textEmb, !textPos.isEmpty {
            let n = min(textPos.count, te.dim(1))
            for i in 0..<n {
                let pos = textPos[i]
                embeddings = embeddings.at[0..., pos..<(pos+1), 0...].add(te[0..., i..<(i+1), 0...])
            }
        }

        if let ae = audioEmb, !audioInPos.isEmpty {
            let n = min(audioInPos.count, ae.dim(1))
            for i in 0..<n {
                let pos = audioInPos[i]
                embeddings = embeddings.at[0..., pos..<(pos+1), 0...].add(ae[0..., i..<(i+1), 0...])
            }
        }

        if let aoe = audioOutEmb, !audioOutPos.isEmpty {
            let n = min(audioOutPos.count, aoe.dim(1))
            for i in 0..<n {
                let pos = audioOutPos[i]
                embeddings = embeddings.at[0..., pos..<(pos+1), 0...].add(aoe[0..., i..<(i+1), 0...])
            }
        }

        return embeddings
    }

    // MARK: - Sampling

    func sampleTextToken(logits: MLXArray, temperature: Float = 1.0, topK: Int = 50) -> MLXArray {
        if temperature == 0 { return MLX.argMax(logits, axis: -1) }

        var l = logits / MLXArray(temperature)

        if topK > 0 && topK < l.dim(-1) {
            let sortedIndices = MLX.argSort(-l, axis: -1)
            let kthPos = sortedIndices[0, topK - 1].item(Int.self)
            let kthValue = l[0, kthPos]
            l = MLX.which(l .>= kthValue, l, MLXArray(-Float.infinity))
        }

        return MLXRandom.categorical(l)
    }

    func sampleAudioFrame(
        hiddenState: MLXArray, audioCache: [(MLXArray, MLXArray)?]? = nil,
        temperature: Float = 1.0, topK: Int = 4
    ) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        let B = hiddenState.dim(0)
        let depthformerIn = depthLinear(hiddenState)
            .reshaped(B, 1, config.codebooks, audioHead.depthformerDim)

        var depthformerToken = MLXArray.zeros([B, audioHead.depthformerDim])
        var cache = audioCache ?? Array(repeating: nil, count: audioHead.depthformer.layersCount)
        var codes: [MLXArray] = []

        let greedy = temperature <= 0 || topK == 1

        for i in 0..<config.codebooks {
            var curInput = depthformerIn[0..., 0..., i, 0...]
            curInput = curInput + depthformerToken.expandedDimensions(axis: 1)

            let (out, newCache) = audioHead.depthformer(curInput, cache: cache, useCache: true)
            cache = newCache?.map { Optional($0) } ?? cache

            let logits = depthEmbeddings[i].logits(out[0..., (-1)..., 0...].squeezed(axis: 1))

            let code: MLXArray
            if greedy {
                code = MLX.argMax(logits, axis: -1, keepDims: true)
            } else {
                var l = logits / MLXArray(temperature)
                if topK > 0 && topK < l.dim(-1) {
                    let sortedIndices = MLX.argSort(-l, axis: -1)
                    let kthPos = sortedIndices[0, topK - 1].item(Int.self)
                    let kthValue = l[0, kthPos]
                    l = MLX.which(l .>= kthValue, l, MLXArray(-Float.infinity))
                }
                code = MLXRandom.categorical(l).expandedDimensions(axis: -1)
            }

            codes.append(code.squeezed(axis: -1))
            depthformerToken = depthEmbeddings[i].embedRaw(code.squeezed(axis: -1))
        }

        return (MLX.stacked(codes, axis: -1), cache.compactMap { $0 })
    }

    // MARK: - Interleaved Generation

    public func generateInterleaved(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        config genConfig: LFMGenerationConfig = LFMGenerationConfig()
    ) -> AsyncThrowingStream<(MLXArray, LFMModality), Error> {
        AsyncThrowingStream { continuation in
            let nText = config.interleavedNText
            let nAudio = config.interleavedNAudio

            let (hiddenStates, cache) = prefill(
                textTokens: textTokens, audioFeatures: audioFeatures,
                audioCodes: audioCodes, modalities: modalities
            )

            var lastHidden = hiddenStates[0..., (-1)..., 0...]
            var generated = 0
            var modalityLeft = nText
            var textDone = false
            var currentModality = LFMModality.text

            while generated < genConfig.maxNewTokens {
                if currentModality == .text {
                    let textLogits = lfm.embedTokens.asLinear(lastHidden)[0..., -1, 0...]
                    let textToken = sampleTextToken(
                        logits: textLogits, temperature: genConfig.temperature,
                        topK: genConfig.topK
                    )
                    let tokenId = textToken.item(Int.self)

                    if tokenId == lfmImEndToken { break }

                    continuation.yield((textToken, .text))

                    if tokenId == lfmTextEndToken { textDone = true }

                    let nextEmb = embedText(MLXArray([Int32(tokenId)]).reshaped(1, 1))
                    lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                    modalityLeft -= 1
                    generated += 1

                    if modalityLeft <= 0 || textDone {
                        modalityLeft = nAudio
                        currentModality = .audioOut
                    }
                } else {
                    let (audioFrame, _) = sampleAudioFrame(
                        hiddenState: lastHidden, audioCache: nil,
                        temperature: genConfig.audioTemperature,
                        topK: genConfig.audioTopK
                    )

                    if audioFrame[0, 0].item(Int.self) == lfmAudioEOSToken {
                        let eosFrame = MLX.full(audioFrame.shape, values: MLXArray(Int32(lfmAudioEOSToken)), type: Int32.self)
                        continuation.yield((eosFrame.squeezed(axis: 0), .audioOut))

                        // Embed EOS back into the model
                        let nextEmb = embedAudioOut(eosFrame).expandedDimensions(axis: 1)
                        lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                        generated += 1
                        currentModality = .text
                        if textDone { break }
                        continue
                    }

                    continuation.yield((audioFrame.squeezed(axis: 0), .audioOut))

                    let nextEmb = embedAudioOut(audioFrame).expandedDimensions(axis: 1)
                    lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                    modalityLeft -= 1
                    generated += 1

                    if modalityLeft <= 0 && !textDone {
                        modalityLeft = nText
                        currentModality = .text
                    }
                }
            }

            continuation.finish()
        }
    }

    // MARK: - Sequential Generation

    public func generateSequential(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        config genConfig: LFMGenerationConfig = LFMGenerationConfig()
    ) -> AsyncThrowingStream<(MLXArray, LFMModality), Error> {
        AsyncThrowingStream { continuation in
            let (hiddenStates, cache) = prefill(
                textTokens: textTokens, audioFeatures: audioFeatures,
                audioCodes: audioCodes, modalities: modalities
            )

            var lastHidden = hiddenStates[0..., (-1)..., 0...]

            var currentModality: LFMModality = .text
            if let t = textTokens, t[0, -1].item(Int.self) == lfmAudioStartToken {
                currentModality = .audioOut
            }

            var generated = 0

            while generated < genConfig.maxNewTokens {
                if currentModality == .text {
                    let textLogits = lfm.embedTokens.asLinear(lastHidden)[0..., -1, 0...]
                    let textToken = sampleTextToken(
                        logits: textLogits, temperature: genConfig.temperature,
                        topK: genConfig.topK
                    )
                    let tokenId = textToken.item(Int.self)

                    if tokenId == lfmImEndToken {
                        continuation.yield((textToken, .text))
                        break
                    }

                    if tokenId == lfmAudioStartToken {
                        currentModality = .audioOut
                        let nextEmb = embedText(MLXArray([Int32(tokenId)]).reshaped(1, 1))
                        lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)
                        continue
                    }

                    continuation.yield((textToken, .text))

                    let nextEmb = embedText(MLXArray([Int32(tokenId)]).reshaped(1, 1))
                    lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)

                } else {
                    let (audioFrame, _) = sampleAudioFrame(
                        hiddenState: lastHidden, audioCache: nil,
                        temperature: genConfig.audioTemperature,
                        topK: genConfig.audioTopK
                    )

                    if audioFrame[0, 0].item(Int.self) == lfmAudioEOSToken {
                        let eosFrame = MLX.full(audioFrame.shape, values: MLXArray(Int32(lfmAudioEOSToken)), type: Int32.self)
                        currentModality = .text
                        continuation.yield((eosFrame.squeezed(axis: 0), .audioOut))
                        let nextEmb = embedAudioOut(eosFrame).expandedDimensions(axis: 1)
                        lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)
                        generated += 1
                        continue
                    }

                    continuation.yield((audioFrame.squeezed(axis: 0), .audioOut))

                    let nextEmb = embedAudioOut(audioFrame).expandedDimensions(axis: 1)
                    lastHidden = lfm(inputEmbeddings: nextEmb, cache: cache)
                }

                generated += 1
            }

            continuation.finish()
        }
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        let skipKeys = [
            "audio_loss_weights", "codebook_offsets", "downsample.", "upsample.",
            ".num_batches_tracked", "pos_enc.pe", ".freqs",
        ]

        for (key, value) in weights {
            if skipKeys.contains(where: { key.contains($0) }) { continue }

            var newKey = key

            if key.hasPrefix("conformer.") {
                newKey = key.replacingOccurrences(of: "conformer.", with: "audio_encoder.")
                newKey = newKey.replacingOccurrences(of: ".norm_feed_forward1.", with: ".ff1_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_feed_forward2.", with: ".ff2_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_self_att.", with: ".attn_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_conv.", with: ".conv_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_out.", with: ".final_norm.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward1.", with: ".ff1.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward2.", with: ".ff2.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_q.", with: ".attn.q_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_k.", with: ".attn.k_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_v.", with: ".attn.v_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_out.", with: ".attn.out_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_pos.", with: ".attn.pos_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.pos_bias_u", with: ".attn.pos_bias_u")
                newKey = newKey.replacingOccurrences(of: ".self_attn.pos_bias_v", with: ".attn.pos_bias_v")
                newKey = newKey.replacingOccurrences(of: ".conv.batch_norm.", with: ".conv.norm.")
            }
            else if key.hasPrefix("audio_adapter.model.") {
                newKey = key.replacingOccurrences(of: "audio_adapter.model.", with: "audio_adapter.layers.")
            }
            else if key.hasPrefix("lfm.") {
                newKey = newKey.replacingOccurrences(of: ".feed_forward.linear1.", with: ".feed_forward.w1.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward.linear2.", with: ".feed_forward.w2.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward.linear3.", with: ".feed_forward.w3.")
            }
            else if key.hasPrefix("depthformer.") {
                if let range = key.range(of: #"depthformer\.layers\.(\d+)\.(.*)"#, options: .regularExpression) {
                    let matched = String(key[range])
                    let components = matched.split(separator: ".")
                    if components.count >= 4 {
                        let layerIdx = components[2]
                        let rest = components[3...].joined(separator: ".")

                        if rest == "operator.qkv_proj.weight" {
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.qkv_weight"
                        } else if rest == "operator.out_proj.weight" {
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.o_proj.weight"
                        } else if rest == "operator.bounded_attention.q_layernorm.weight" {
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.q_norm.weight"
                        } else if rest == "operator.bounded_attention.k_layernorm.weight" {
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.k_norm.weight"
                        } else if rest.hasPrefix("operator_norm.") {
                            let suffix = rest.split(separator: ".", maxSplits: 1).last.map(String.init) ?? ""
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).attn_norm.\(suffix)"
                        } else if rest.hasPrefix("feed_forward.") {
                            let suffix = rest.split(separator: ".", maxSplits: 1).last.map(String.init) ?? ""
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).ffn.\(suffix)"
                        } else if rest.hasPrefix("ffn_norm.") {
                            let suffix = rest.split(separator: ".", maxSplits: 1).last.map(String.init) ?? ""
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).ffn_norm.\(suffix)"
                        } else {
                            newKey = "audio_head.depthformer.blocks.\(layerIdx).\(rest)"
                        }
                    }
                }
            }

            sanitized[newKey] = value
        }

        var keysToRemove: [String] = []
        var keysToAdd: [String: MLXArray] = [:]

        for (key, value) in sanitized {
            if key.contains(".attn.qkv_weight") {
                let qDim = 1024
                let kvDim = 256
                let qWeight = value[..<qDim]
                let kWeight = value[qDim..<(qDim + kvDim)]
                let vWeight = value[(qDim + kvDim)...]

                let baseKey = key.replacingOccurrences(of: ".qkv_weight", with: "")
                keysToAdd["\(baseKey).q_proj.weight"] = qWeight
                keysToAdd["\(baseKey).k_proj.weight"] = kWeight
                keysToAdd["\(baseKey).v_proj.weight"] = vWeight
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove { sanitized.removeValue(forKey: key) }
        sanitized.merge(keysToAdd) { _, new in new }

        for (key, value) in sanitized {
            if key.contains("pointwise_conv") && key.contains("weight") && value.ndim == 3 {
                sanitized[key] = value.ndim == 2 ? value : value.squeezed(axis: -1)
            } else if (key.contains("depthwise_conv") || key.hasSuffix(".conv.weight"))
                        && value.ndim == 3 {
                if value.dim(2) > value.dim(1) {
                    sanitized[key] = value.transposed(0, 2, 1)
                }
            } else if key.contains("pre_encode.conv") && value.ndim == 4 {
            }
        }

        for (key, value) in sanitized {
            if key.hasPrefix("lfm.") && key.contains("conv.conv.weight") && value.ndim == 3 {
                if value.dim(value.ndim - 1) > value.dim(1) {
                    sanitized[key] = value.transposed(0, 2, 1)
                }
            }
        }


        var adapterRemap: [String: MLXArray] = [:]
        var adapterRemove: [String] = []
        var linearIdx = 0
        let adapterKeys = sanitized.keys.filter { $0.hasPrefix("audio_adapter.layers.") }
        let indices = Set(adapterKeys.compactMap { key -> Int? in
            let parts = key.dropFirst("audio_adapter.layers.".count).split(separator: ".", maxSplits: 1)
            return Int(parts.first ?? "")
        }).sorted()

        for idx in indices {
            let prefix = "audio_adapter.layers.\(idx)"
            let matchingKeys = adapterKeys.filter { $0.hasPrefix(prefix + ".") }
            if matchingKeys.isEmpty { continue }

            let isNorm = matchingKeys.contains { $0.hasSuffix(".weight") } &&
                         !matchingKeys.contains { $0.hasSuffix(".scales") } &&
                         sanitized["\(prefix).weight"]?.ndim == 1
            if isNorm {
                for key in matchingKeys {
                    let suffix = String(key.dropFirst(prefix.count + 1))
                    adapterRemap["audio_adapter.norm.\(suffix)"] = sanitized[key]
                    adapterRemove.append(key)
                }
            } else {
                for key in matchingKeys {
                    let suffix = String(key.dropFirst(prefix.count + 1))
                    adapterRemap["audio_adapter.linears.\(linearIdx).\(suffix)"] = sanitized[key]
                    adapterRemove.append(key)
                }
                linearIdx += 1
            }
        }
        for key in adapterRemove { sanitized.removeValue(forKey: key) }
        sanitized.merge(adapterRemap) { _, new in new }

        return sanitized
    }

    // MARK: - From Pretrained

    public static func fromPretrained(_ modelNameOrPath: String) async throws -> LFM2AudioModel {
        guard let repoID = Repo.ID(rawValue: modelNameOrPath) else {
            throw LFMAudioError.modelNotFound(modelNameOrPath)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors"
        )

        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(LFM2AudioConfig.self, from: configData)

        let model = LFM2AudioModel(config)

        // Load weights
        let files = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        let safetensorFiles = files.filter {
            $0.pathExtension == "safetensors" && !$0.lastPathComponent.contains("tokenizer")
        }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitizedWeights = sanitize(weights: weights)

        // Quantize if needed (follows Python model_quant_predicate: skip norm/conv)
        let perLayerQuantization = config.perLayerQuantization
        if perLayerQuantization != nil {
            quantize(model: model) { path, module in
                if path.contains("norm") || path.contains("conv") {
                    return nil
                }
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        // Cast float32 to float16 (except conv/norm) for non-quantized models
        var finalWeights = sanitizedWeights
        if perLayerQuantization == nil {
            for (key, value) in finalWeights {
                if value.dtype == .float32 {
                    if key.contains("conv") || key.contains("norm") { continue }
                    finalWeights[key] = value.asType(.float16)
                }
            }
        }

        let _ = try model.update(parameters: ModuleParameters.unflattened(finalWeights), verify: .noUnusedKeys)
        eval(model.parameters())

        model.modelDirectory = modelDir
        model.processor = try await LFM2AudioProcessor.fromPretrained(modelDir, config: config)

        return model
    }

    public var sampleRate: Int { config.sampleRate }
}
