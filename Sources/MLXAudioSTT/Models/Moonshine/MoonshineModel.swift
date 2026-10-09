import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

public final class MoonshineTokenizer {
    let idToToken: [Int: String]
    let specialTokenIds: Set<Int>

    public init(modelDir: URL) throws {
        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerURL)) as? [String: Any] ?? [:]
        let model = object["model"] as? [String: Any] ?? [:]
        guard let vocab = model["vocab"] as? [String: Int], !vocab.isEmpty else {
            throw STTError.modelNotInitialized("Moonshine tokenizer.json does not contain a BPE vocabulary.")
        }

        var idToToken: [Int: String] = [:]
        idToToken.reserveCapacity(vocab.count)
        for (token, id) in vocab {
            idToToken[id] = token
        }
        self.idToToken = idToToken

        var specialIds = Set<Int>()
        if let addedTokens = object["added_tokens"] as? [[String: Any]] {
            for token in addedTokens where (token["special"] as? Bool) == true {
                if let id = token["id"] as? Int {
                    specialIds.insert(id)
                }
            }
        }
        self.specialTokenIds = specialIds
    }

    public func decode(_ tokens: [Int]) -> String {
        var pieces: [String] = []
        var bytes: [UInt8] = []

        func flushBytes() {
            guard !bytes.isEmpty else { return }
            if let text = String(bytes: bytes, encoding: .utf8) {
                pieces.append(text)
            }
            bytes.removeAll()
        }

        for id in tokens {
            guard !specialTokenIds.contains(id), let token = idToToken[id] else {
                continue
            }
            if token.hasPrefix("<0x"), token.hasSuffix(">"), token.count == 6 {
                let hex = token.dropFirst(3).dropLast(1)
                if let byte = UInt8(hex, radix: 16) {
                    bytes.append(byte)
                    continue
                }
            }
            flushBytes()
            pieces.append(token)
        }
        flushBytes()

        return pieces.joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private func moonshineActivation(_ x: MLXArray, name: String) -> MLXArray {
    switch name.lowercased() {
    case "silu", "swish":
        return silu(x)
    default:
        return gelu(x)
    }
}

private func moonshineRotateHalf(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    let last = x.dim(-1)
    let paired = x.reshaped(shape.dropLast() + [last / 2, 2])
    let parts = MLX.split(paired, parts: 2, axis: -1)
    return MLX.concatenated([-parts[1], parts[0]], axis: -1).reshaped(shape)
}

private final class MoonshineRotaryEmbedding {
    let rotaryDim: Int
    let base: Float

    init(rotaryDim: Int, base: Float) {
        self.rotaryDim = rotaryDim
        self.base = base
    }

    func callAsFunction(sequenceLength: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let positions = MLX.arange(sequenceLength, dtype: .float32)
        let dimValues = stride(from: 0, to: rotaryDim, by: 2).map(Float.init)
        let invFreq = 1.0 / pow(MLXArray(base), MLXArray(dimValues) / Float(rotaryDim))
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let cos = MLX.repeated(MLX.cos(freqs), count: 2, axis: -1)
            .expandedDimensions(axes: [0, 1])
            .asType(dtype)
        let sin = MLX.repeated(MLX.sin(freqs), count: 2, axis: -1)
            .expandedDimensions(axes: [0, 1])
            .asType(dtype)
        return (cos, sin)
    }
}

private final class MoonshineAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let numKVGroups: Int
    let headDim: Int
    let rotaryDim: Int
    let scale: Float
    let isCausal: Bool
    let rotaryEmb: MoonshineRotaryEmbedding

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(
        hiddenSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        bias: Bool,
        isCausal: Bool,
        partialRotaryFactor: Float,
        ropeTheta: Float
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numKVGroups = max(1, numHeads / numKVHeads)
        self.headDim = hiddenSize / numHeads
        var rotaryDim = Int(Float(self.headDim) * partialRotaryFactor)
        rotaryDim -= rotaryDim % 2
        self.rotaryDim = max(2, rotaryDim)
        self.scale = pow(Float(self.headDim), -0.5)
        self.isCausal = isCausal
        self.rotaryEmb = MoonshineRotaryEmbedding(rotaryDim: self.rotaryDim, base: ropeTheta)
        _qProj.wrappedValue = Linear(hiddenSize, numHeads * self.headDim, bias: bias)
        _kProj.wrappedValue = Linear(hiddenSize, numKVHeads * self.headDim, bias: bias)
        _vProj.wrappedValue = Linear(hiddenSize, numKVHeads * self.headDim, bias: bias)
        _oProj.wrappedValue = Linear(numHeads * self.headDim, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let source = encoderHiddenStates ?? x
        let sourceTime = source.dim(1)

        var q = qProj(x).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(source).reshaped(batch, sourceTime, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(source).reshaped(batch, sourceTime, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if encoderHiddenStates == nil {
            let rotary = rotaryEmb(sequenceLength: time, dtype: q.dtype)
            let qRot = q[0..., 0..., 0..., ..<rotaryDim]
            let qPass = q[0..., 0..., 0..., rotaryDim...]
            let kRot = k[0..., 0..., 0..., ..<rotaryDim]
            let kPass = k[0..., 0..., 0..., rotaryDim...]
            q = MLX.concatenated([
                (qRot * rotary.0) + (moonshineRotateHalf(qRot) * rotary.1),
                qPass,
            ], axis: -1)
            k = MLX.concatenated([
                (kRot * rotary.0) + (moonshineRotateHalf(kRot) * rotary.1),
                kPass,
            ], axis: -1)
        }

        if numKVGroups > 1 {
            k = MLX.repeated(k, count: numKVGroups, axis: 1)
            v = MLX.repeated(v, count: numKVGroups, axis: 1)
        }

        let mask = isCausal && time > 1 ? MultiHeadAttention.createAdditiveCausalMask(time).asType(q.dtype) : nil
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped(batch, time, -1))
    }
}

private final class MoonshineEncoderMLP: Module {
    let activation: String
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(config: MoonshineConfig) {
        self.activation = config.encoderHiddenAct
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(moonshineActivation(fc1(x), name: activation))
    }
}

private final class MoonshineDecoderMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(config: MoonshineConfig) {
        _fc1.wrappedValue = Linear(config.hiddenSize, 2 * config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = fc1(x)
        let parts = MLX.split(projected, parts: 2, axis: -1)
        return fc2(silu(parts[1]) * parts[0])
    }
}

private final class MoonshineEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MoonshineAttention
    @ModuleInfo var mlp: MoonshineEncoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    init(config: MoonshineConfig) {
        _selfAttn.wrappedValue = MoonshineAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.encoderNumAttentionHeads,
            numKVHeads: config.encoderNumKeyValueHeads,
            bias: config.attentionBias,
            isCausal: false,
            partialRotaryFactor: config.partialRotaryFactor,
            ropeTheta: config.ropeTheta
        )
        _mlp.wrappedValue = MoonshineEncoderMLP(config: config)
        _inputLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, bias: false)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + selfAttn(inputLayerNorm(x))
        out = out + mlp(postAttentionLayerNorm(out))
        return out
    }
}

private final class MoonshineDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MoonshineAttention
    @ModuleInfo(key: "encoder_attn") var encoderAttn: MoonshineAttention
    @ModuleInfo var mlp: MoonshineDecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm

    init(config: MoonshineConfig) {
        _selfAttn.wrappedValue = MoonshineAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.decoderNumAttentionHeads,
            numKVHeads: config.decoderNumKeyValueHeads,
            bias: config.attentionBias,
            isCausal: true,
            partialRotaryFactor: config.partialRotaryFactor,
            ropeTheta: config.ropeTheta
        )
        _encoderAttn.wrappedValue = MoonshineAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.decoderNumAttentionHeads,
            numKVHeads: config.decoderNumKeyValueHeads,
            bias: config.attentionBias,
            isCausal: false,
            partialRotaryFactor: config.partialRotaryFactor,
            ropeTheta: config.ropeTheta
        )
        _mlp.wrappedValue = MoonshineDecoderMLP(config: config)
        _inputLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, bias: false)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, bias: false)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var out = x + selfAttn(inputLayerNorm(x))
        out = out + encoderAttn(postAttentionLayerNorm(out), encoderHiddenStates: encoderHiddenStates)
        out = out + mlp(finalLayerNorm(out))
        return out
    }
}

public final class MoonshineEncoder: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var groupnorm: GroupNorm
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var conv3: Conv1d
    @ModuleInfo fileprivate var layers: [MoonshineEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    public init(config: MoonshineConfig) {
        let dim = config.hiddenSize
        _conv1.wrappedValue = Conv1d(inputChannels: 1, outputChannels: dim, kernelSize: 127, stride: 64, bias: false)
        _groupnorm.wrappedValue = GroupNorm(groupCount: 1, dimensions: dim, pytorchCompatible: true)
        _conv2.wrappedValue = Conv1d(inputChannels: dim, outputChannels: 2 * dim, kernelSize: 7, stride: 3)
        _conv3.wrappedValue = Conv1d(inputChannels: 2 * dim, outputChannels: dim, kernelSize: 3, stride: 2)
        _layers.wrappedValue = (0..<config.encoderNumHiddenLayers).map { _ in MoonshineEncoderLayer(config: config) }
        _layerNorm.wrappedValue = LayerNorm(dimensions: dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var x = audio.ndim == 1 ? audio.expandedDimensions(axis: 0) : audio
        x = x.expandedDimensions(axis: -1)
        x = tanh(conv1(x))
        x = groupnorm(x)
        x = gelu(conv2(x))
        x = gelu(conv3(x))
        for layer in layers {
            x = layer(x)
        }
        return layerNorm(x)
    }
}

public final class MoonshineDecoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo fileprivate var layers: [MoonshineDecoderLayer]
    @ModuleInfo var norm: LayerNorm

    public init(config: MoonshineConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.decoderNumHiddenLayers).map { _ in MoonshineDecoderLayer(config: config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ tokens: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var x = embedTokens(tokens)
        for layer in layers {
            x = layer(x, encoderHiddenStates: encoderHiddenStates)
        }
        return norm(x)
    }
}

public final class MoonshineModel: Module, STTGenerationModel {
    public let config: MoonshineConfig
    private var tokenizer: MoonshineTokenizer?

    @ModuleInfo public var encoder: MoonshineEncoder
    @ModuleInfo public var decoder: MoonshineDecoder
    @ModuleInfo(key: "proj_out") public var projOut: Linear?

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(maxTokens: 200, temperature: 0, topP: 1, topK: 0, language: nil)
    }

    public init(config: MoonshineConfig, tokenizer: MoonshineTokenizer? = nil) {
        self.config = config
        self.tokenizer = tokenizer
        _encoder.wrappedValue = MoonshineEncoder(config: config)
        _decoder.wrappedValue = MoonshineDecoder(config: config)
        _projOut.wrappedValue = config.tieWordEmbeddings ? nil : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        let start = CFAbsoluteTimeGetCurrent()
        let waveform = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let encoderOut = encoder(waveform.asType(.float32))
        eval(encoderOut)

        var tokens = [config.decoderStartTokenId]
        var generated: [Int] = []
        for _ in 0..<generationParameters.maxTokens {
            let tokenIds = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count).asType(.int32)
            let hidden = decoder(tokenIds, encoderHiddenStates: encoderOut)
            let lastHidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
            let logits = logitsForHidden(lastHidden).squeezed(axis: 1)
            eval(logits)
            let nextToken: Int
            if generationParameters.temperature > 0 {
                nextToken = MLXRandom.categorical(logits / generationParameters.temperature).item(Int.self)
            } else {
                nextToken = logits.argMax(axis: -1).item(Int.self)
            }
            if nextToken == config.eosTokenId {
                break
            }
            tokens.append(nextToken)
            generated.append(nextToken)
        }

        let text = decode(tokens: generated).trimmingCharacters(in: .whitespacesAndNewlines)
        let totalTime = CFAbsoluteTimeGetCurrent() - start
        return STTOutput(
            text: text,
            segments: [["text": text, "start": 0.0, "end": 0.0]],
            generationTokens: generated.count,
            totalTokens: tokens.count,
            generationTps: Double(generated.count) / max(totalTime, 0.001),
            totalTime: totalTime
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let output = self.generate(audio: audio, generationParameters: generationParameters)
            if !output.text.isEmpty {
                continuation.yield(.token(output.text))
            }
            continuation.yield(.result(output))
            continuation.finish()
        }
    }

    public func logitsForHidden(_ hiddenStates: MLXArray) -> MLXArray {
        if config.tieWordEmbeddings {
            return decoder.embedTokens.asLinear(hiddenStates)
        }
        return projOut!(hiddenStates)
    }

    public func decode(tokens: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(tokens)
        }
        return tokens.map { token in
            token < 128 ? String(UnicodeScalar(token)!) : "<\(token)>"
        }.joined()
    }

    public static func sanitize(weights: [String: MLXArray], tieWordEmbeddings: Bool = true) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, var value) in weights {
            var newKey = key
            if key.hasPrefix("model.encoder.") || key.hasPrefix("model.decoder.") {
                newKey = String(key.dropFirst("model.".count))
            } else if key.hasPrefix("proj_out.") && tieWordEmbeddings {
                continue
            }

            if newKey.contains("conv") && newKey.hasSuffix(".weight") && value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
            sanitized[newKey] = value
        }
        return sanitized
    }

    public static func fromPretrained(_ modelName: String) async throws -> MoonshineModel {
        let expanded = (modelName as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            return try await fromModelDirectory(URL(fileURLWithPath: expanded))
        }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelName) else {
            throw STTError.invalidInput("Invalid Moonshine repository ID: \(modelName)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.json", "tokenizer.*"],
            hfToken: hfToken
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> MoonshineModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(MoonshineConfig.self, from: Data(contentsOf: configURL))
        let tokenizer = try? MoonshineTokenizer(modelDir: modelDir)
        let model = MoonshineModel(config: config, tokenizer: tokenizer)
        let weights = try loadMoonshineSafetensorWeights(from: modelDir)
        let sanitized = sanitize(weights: weights, tieWordEmbeddings: config.tieWordEmbeddings)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: Module.VerifyUpdate.noUnusedKeys)
        model.train(false)
        eval(model)
        return model
    }
}

private func loadMoonshineSafetensorWeights(from modelDir: URL) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !files.isEmpty else {
        throw STTError.modelNotInitialized("No safetensors files found in \(modelDir.path)")
    }

    var weights: [String: MLXArray] = [:]
    for file in files {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}
