import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

private func lasrActivation(_ x: MLXArray, name: String) -> MLXArray {
    switch name.lowercased() {
    case "relu":
        return relu(x)
    default:
        return silu(x)
    }
}

private func lasrRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., ..<half]
    let x2 = x[0..., 0..., 0..., half...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

private final class LasrRotaryEmbedding {
    let headDim: Int
    let base: Float

    init(config: LasrEncoderConfig) {
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.base = config.ropeTheta
    }

    func callAsFunction(sequenceLength: Int, dtype: DType = .float32) -> (MLXArray, MLXArray) {
        let positions = MLX.arange(sequenceLength, dtype: .float32)
        let dimValues = stride(from: 0, to: headDim, by: 2).map { Float($0) }
        let invFreq = 1.0 / pow(MLXArray(base), MLXArray(dimValues) / Float(headDim))
        let angles = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let emb = MLX.concatenated([angles, angles], axis: -1)
        let cos = MLX.cos(emb).expandedDimensions(axes: [0, 2]).asType(dtype)
        let sin = MLX.sin(emb).expandedDimensions(axes: [0, 2]).asType(dtype)
        return (cos, sin)
    }
}

private final class LasrEncoderSubsampling: Module {
    @ModuleInfo(key: "dense_0") var dense0: Linear
    @ModuleInfo(key: "conv_0") var conv0: Conv1d
    @ModuleInfo(key: "conv_1") var conv1: Conv1d
    @ModuleInfo(key: "dense_1") var dense1: Linear

    init(config: LasrEncoderConfig) {
        _dense0.wrappedValue = Linear(config.numMelBins, config.hiddenSize)
        _conv0.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.subsamplingConvKernelSize,
            stride: config.subsamplingConvStride
        )
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.subsamplingConvChannels,
            kernelSize: config.subsamplingConvKernelSize,
            stride: config.subsamplingConvStride
        )
        _dense1.wrappedValue = Linear(config.subsamplingConvChannels, config.hiddenSize)
    }

    func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        var hidden = relu(dense0(inputFeatures))
        hidden = relu(conv0(hidden))
        hidden = relu(conv1(hidden))
        return dense1(hidden)
    }
}

private final class LasrEncoderAttention: Module {
    let numHeads: Int
    let numKeyValueHeads: Int
    let numKeyValueGroups: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(config: LasrEncoderConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.numKeyValueGroups = max(1, config.numAttentionHeads / config.numKeyValueHeads)
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(config.numAttentionHeads * headDim, config.hiddenSize, bias: config.attentionBias)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let length = hiddenStates.dim(1)
        var q = qProj(hiddenStates).reshaped(batch, length, numHeads, headDim)
        var k = kProj(hiddenStates).reshaped(batch, length, numKeyValueHeads, headDim)
        var v = vProj(hiddenStates).reshaped(batch, length, numKeyValueHeads, headDim)

        q = (q * rotary.cos) + (lasrRotateHalf(q) * rotary.sin)
        k = (k * rotary.cos) + (lasrRotateHalf(k) * rotary.sin)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        if numKeyValueGroups > 1 {
            k = MLX.repeated(k, count: numKeyValueGroups, axis: 1)
            v = MLX.repeated(v, count: numKeyValueGroups, axis: 1)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class LasrEncoderConvolutionModule: Module {
    let activation: String
    let padLeft: Int
    let padRight: Int

    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(config: LasrEncoderConfig) {
        let channels = config.hiddenSize
        let kernel = config.convKernelSize
        self.activation = config.hiddenAct
        self.padLeft = (kernel - 1) / 2
        self.padRight = kernel - 1 - padLeft
        _pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: 2 * channels,
            kernelSize: 1,
            bias: config.convolutionBias
        )
        _depthwiseConv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernel,
            groups: channels,
            bias: config.convolutionBias
        )
        _norm.wrappedValue = BatchNorm(featureCount: channels, momentum: config.batchNormMomentum)
        _pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1,
            bias: config.convolutionBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hidden = pointwiseConv1(hiddenStates)
        let parts = MLX.split(hidden, parts: 2, axis: -1)
        hidden = parts[0] * sigmoid(parts[1])
        hidden = MLX.padded(hidden, widths: [.init((0, 0)), .init((padLeft, padRight)), .init((0, 0))])
        hidden = depthwiseConv(hidden)
        hidden = norm(hidden)
        hidden = lasrActivation(hidden, name: activation)
        return pointwiseConv2(hidden)
    }
}

private final class LasrEncoderFeedForward: Module {
    let activation: String
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(config: LasrEncoderConfig) {
        self.activation = config.hiddenAct
        _linear1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.attentionBias)
        _linear2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.attentionBias)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        linear2(lasrActivation(linear1(hiddenStates), name: activation))
    }
}

private final class LasrEncoderBlock: Module {
    let convResidualWeights: [Float]
    let feedForwardResidualWeights: [Float]

    @ModuleInfo(key: "feed_forward1") var feedForward1: LasrEncoderFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: LasrEncoderAttention
    @ModuleInfo var conv: LasrEncoderConvolutionModule
    @ModuleInfo(key: "feed_forward2") var feedForward2: LasrEncoderFeedForward
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(config: LasrEncoderConfig) {
        self.convResidualWeights = config.convResidualWeights
        self.feedForwardResidualWeights = config.feedForwardResidualWeights
        _feedForward1.wrappedValue = LasrEncoderFeedForward(config: config)
        _selfAttn.wrappedValue = LasrEncoderAttention(config: config)
        _conv.wrappedValue = LasrEncoderConvolutionModule(config: config)
        _feedForward2.wrappedValue = LasrEncoderFeedForward(config: config)
        _normFeedForward1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _normSelfAtt.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _normConv.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _normFeedForward2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _normOut.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil
    ) -> MLXArray {
        var residual = hiddenStates
        var hidden = feedForward1(normFeedForward1(hiddenStates))
        hidden = feedForwardResidualWeights[0] * residual + feedForwardResidualWeights[1] * hidden

        let attnOutput = selfAttn(normSelfAtt(hidden), rotary: rotary, mask: mask)
        hidden = hidden + attnOutput

        let convOutput = conv(normConv(hidden))
        hidden = convResidualWeights[0] * hidden + convResidualWeights[1] * convOutput

        residual = hidden
        hidden = feedForward2(normFeedForward2(hidden))
        hidden = feedForwardResidualWeights[0] * residual + feedForwardResidualWeights[1] * hidden
        return normOut(hidden)
    }
}

public final class LasrEncoder: Module {
    public let config: LasrEncoderConfig
    private let rotaryEmb: LasrRotaryEmbedding

    @ModuleInfo fileprivate var subsampler: LasrEncoderSubsampling
    @ModuleInfo fileprivate var layers: [LasrEncoderBlock]
    @ModuleInfo(key: "out_norm") var outNorm: LayerNorm

    public init(config: LasrEncoderConfig) {
        self.config = config
        self.rotaryEmb = LasrRotaryEmbedding(config: config)
        _subsampler.wrappedValue = LasrEncoderSubsampling(config: config)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in LasrEncoderBlock(config: config) }
        _outNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    public func callAsFunction(_ inputFeatures: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var hidden = subsampler(inputFeatures)
        let rotary = rotaryEmb(sequenceLength: hidden.dim(1), dtype: hidden.dtype)
        for layer in layers {
            hidden = layer(hidden, rotary: rotary, mask: mask)
        }
        return outNorm(hidden)
    }
}

public final class LasrCTCModel: Module, STTGenerationModel {
    public let config: LasrCTCConfig
    public var tokenizer: SentencePieceTokenizer?

    @ModuleInfo public var encoder: LasrEncoder
    @ModuleInfo(key: "ctc_head") public var ctcHead: Linear

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(maxTokens: 0, temperature: 0, topP: 1, topK: 0, language: nil)
    }

    public init(config: LasrCTCConfig, tokenizer: SentencePieceTokenizer? = nil) {
        self.config = config
        self.tokenizer = tokenizer
        _encoder.wrappedValue = LasrEncoder(config: config.encoderConfig)
        _ctcHead.wrappedValue = Linear(config.encoderConfig.hiddenSize, config.vocabSize)
        super.init()
    }

    public func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        ctcHead(encoder(inputFeatures))
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        let start = CFAbsoluteTimeGetCurrent()
        let features = prepareFeatures(audio)
        let logits = self(features)
        eval(logits)
        let tokens = Wav2Vec2CTCModel.greedyCTCTokens(logits: logits, blankTokenId: config.padTokenId).first ?? []
        let text = tokenizer?.decode(tokens) ?? tokens.map(String.init).joined(separator: " ")
        let totalTime = CFAbsoluteTimeGetCurrent() - start
        return STTOutput(
            text: text,
            segments: [["text": text, "start": 0.0, "end": 0.0]],
            language: generationParameters.language,
            generationTokens: tokens.count,
            totalTokens: tokens.count,
            generationTps: Double(tokens.count) / max(totalTime, 0.001),
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

    private func prepareFeatures(_ audio: MLXArray) -> MLXArray {
        if audio.ndim == 3 && audio.dim(2) == config.encoderConfig.numMelBins {
            return audio
        }
        if audio.ndim == 2 && audio.dim(1) == config.encoderConfig.numMelBins {
            return audio.expandedDimensions(axis: 0)
        }
        let waveform = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let preprocess = ParakeetPreprocessConfig(
            sampleRate: 16_000,
            normalize: "per_feature",
            windowSize: 0.025,
            windowStride: 0.01,
            window: "hann",
            features: config.encoderConfig.numMelBins,
            nFft: 512,
            dither: 0
        )
        return ParakeetAudio.logMelSpectrogram(waveform, config: preprocess)
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, var value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.hasSuffix("num_batches_tracked") {
                continue
            }
            if key.contains("conv") && key.hasSuffix(".weight") && value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }
            if key == "ctc_head.weight" && value.ndim == 3 {
                value = value.squeezed(axis: -1)
            }
            sanitized[key] = value
        }
        return sanitized
    }

    public static func fromPretrained(_ modelName: String) async throws -> LasrCTCModel {
        let expanded = (modelName as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            return try fromModelDirectory(URL(fileURLWithPath: expanded))
        }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelName) else {
            throw STTError.invalidInput("Invalid LASR repository ID: \(modelName)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.json"],
            hfToken: hfToken
        )
        return try fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) throws -> LasrCTCModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(LasrCTCConfig.self, from: Data(contentsOf: configURL))
        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")
        let tokenizer = FileManager.default.fileExists(atPath: tokenizerURL.path)
            ? try? SentencePieceTokenizer.from(tokenizerJSONURL: tokenizerURL)
            : nil
        let model = LasrCTCModel(config: config, tokenizer: tokenizer)
        let weights = try loadLasrSafetensorWeights(from: modelDir)
        let sanitized = sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: Module.VerifyUpdate.noUnusedKeys)
        model.train(false)
        eval(model)
        return model
    }
}

private func loadLasrSafetensorWeights(from modelDir: URL) throws -> [String: MLXArray] {
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
