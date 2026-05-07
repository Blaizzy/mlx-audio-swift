import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public let mossDefaultAudioTokenizerRepo = "mlx-community/MOSS-Audio-Tokenizer-Nano"

public struct MossAudioTokenizerConfig: Sendable {
    public var sampleRate: Int
    public var samplingRate: Int
    public var downsampleRate: Int
    public var causalTransformerContextDuration: Float
    public var numberChannels: Int
    public var enableChannelInterleave: Bool
    public var encoderKwargs: [[String: SendableValue]]
    public var decoderKwargs: [[String: SendableValue]]
    public var quantizerType: String
    public var quantizerKwargs: [String: SendableValue]

    public init(
        sampleRate: Int = 48_000,
        samplingRate: Int = 48_000,
        downsampleRate: Int = 3_840,
        causalTransformerContextDuration: Float = 10,
        numberChannels: Int = 2,
        enableChannelInterleave: Bool = true,
        encoderKwargs: [[String: SendableValue]] = [],
        decoderKwargs: [[String: SendableValue]] = [],
        quantizerType: String = "rlfq",
        quantizerKwargs: [String: SendableValue] = [:]
    ) {
        self.sampleRate = sampleRate
        self.samplingRate = samplingRate
        self.downsampleRate = downsampleRate
        self.causalTransformerContextDuration = causalTransformerContextDuration
        self.numberChannels = numberChannels
        self.enableChannelInterleave = enableChannelInterleave
        self.encoderKwargs = encoderKwargs
        self.decoderKwargs = decoderKwargs
        self.quantizerType = quantizerType
        self.quantizerKwargs = quantizerKwargs
    }

    public static func fromFile(_ url: URL) throws -> MossAudioTokenizerConfig {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        guard let json = object as? [String: Any] else {
            throw MossTTSNanoError.invalidInput("Invalid MOSS audio tokenizer config.")
        }
        return try fromDictionary(json)
    }

    static func fromDictionary(_ json: [String: Any]) throws -> MossAudioTokenizerConfig {
        let encoderKwargs = (json["encoder_kwargs"] as? [[String: Any]] ?? [])
            .map { $0.mapValues(SendableValue.init) }
        let decoderKwargs = (json["decoder_kwargs"] as? [[String: Any]] ?? [])
            .map { $0.mapValues(SendableValue.init) }
        let quantizerKwargs = (json["quantizer_kwargs"] as? [String: Any] ?? [:])
            .mapValues(SendableValue.init)

        return MossAudioTokenizerConfig(
            sampleRate: json.mossInt("sample_rate", fallback: json.mossInt("sampling_rate", fallback: 48_000)),
            samplingRate: json.mossInt("sampling_rate", fallback: json.mossInt("sample_rate", fallback: 48_000)),
            downsampleRate: json.mossInt("downsample_rate", fallback: 3_840),
            causalTransformerContextDuration: json.mossFloat("causal_transformer_context_duration", fallback: 10),
            numberChannels: json.mossInt("number_channels", fallback: 1),
            enableChannelInterleave: json.mossBool("enable_channel_interleave", fallback: true),
            encoderKwargs: encoderKwargs,
            decoderKwargs: decoderKwargs,
            quantizerType: json.mossString("quantizer_type", fallback: "rlfq"),
            quantizerKwargs: quantizerKwargs
        )
    }
}

public struct SendableValue: @unchecked Sendable {
    let rawValue: Any

    init(_ rawValue: Any) {
        self.rawValue = rawValue
    }
}

private extension Dictionary where Key == String, Value == Any {
    func mossInt(_ key: String, fallback: Int) -> Int {
        if let value = self[key] as? Int { return value }
        if let value = self[key] as? Double { return Int(value) }
        if let value = self[key] as? String, let parsed = Int(value) { return parsed }
        return fallback
    }

    func mossFloat(_ key: String, fallback: Float) -> Float {
        if let value = self[key] as? Float { return value }
        if let value = self[key] as? Double { return Float(value) }
        if let value = self[key] as? Int { return Float(value) }
        if let value = self[key] as? String, let parsed = Float(value) { return parsed }
        return fallback
    }

    func mossBool(_ key: String, fallback: Bool) -> Bool {
        if let value = self[key] as? Bool { return value }
        if let value = self[key] as? String { return ["true", "1", "yes"].contains(value.lowercased()) }
        return fallback
    }

    func mossString(_ key: String, fallback: String) -> String {
        (self[key] as? String) ?? fallback
    }
}

private extension Dictionary where Key == String, Value == SendableValue {
    func int(_ key: String, fallback: Int) -> Int {
        asAny.mossInt(key, fallback: fallback)
    }

    func optionalInt(_ key: String) -> Int? {
        let value = asAny[key]
        if let value = value as? Int { return value }
        if let value = value as? Double { return Int(value) }
        if let value = value as? String { return Int(value) }
        return nil
    }

    func float(_ key: String, fallback: Float) -> Float {
        asAny.mossFloat(key, fallback: fallback)
    }

    func optionalFloat(_ key: String) -> Float? {
        let value = asAny[key]
        if let value = value as? Float { return value }
        if let value = value as? Double { return Float(value) }
        if let value = value as? Int { return Float(value) }
        if let value = value as? String { return Float(value) }
        return nil
    }

    func bool(_ key: String, fallback: Bool) -> Bool {
        asAny.mossBool(key, fallback: fallback)
    }

    func string(_ key: String, fallback: String) -> String {
        asAny.mossString(key, fallback: fallback)
    }

    var asAny: [String: Any] {
        mapValues(\.rawValue)
    }
}

private func mossNormalizeWeightExceptOutput(_ weight: MLXArray) -> MLXArray {
    sqrt(sum(sum(square(weight), axis: 2, keepDims: true), axis: 1, keepDims: true))
}

private func mossL2Normalize(_ x: MLXArray, axis: Int = -1, eps: Float = 1e-12) -> MLXArray {
    x / maximum(sqrt(sum(square(x), axis: axis, keepDims: true)), MLXArray(eps))
}

private func mossExactGELU(_ x: MLXArray) -> MLXArray {
    MLXArray(0.5) * x * (MLXArray(1.0) + erf(x / MLXArray(Float(sqrt(2.0)))))
}

private final class MossWeightParam: Module {
    @ModuleInfo(key: "original0") var original0: MLXArray
    @ModuleInfo(key: "original1") var original1: MLXArray

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        _original0.wrappedValue = MLXArray.ones([outChannels, 1, 1])
        _original1.wrappedValue = MLXArray.zeros([outChannels, inChannels, kernelSize])
    }
}

private final class MossWeightParametrizations: Module {
    @ModuleInfo(key: "weight") var weight: MossWeightParam

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        _weight.wrappedValue = MossWeightParam(
            outChannels: outChannels,
            inChannels: inChannels,
            kernelSize: kernelSize
        )
    }
}

private final class MossWNConv1d: Module {
    let stride = 1
    let padding = 0
    let dilation = 1

    @ModuleInfo(key: "parametrizations") var parametrizations: MossWeightParametrizations
    @ModuleInfo var bias: MLXArray

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 1) {
        _parametrizations.wrappedValue = MossWeightParametrizations(
            outChannels: outChannels,
            inChannels: inChannels,
            kernelSize: kernelSize
        )
        _bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    private func sourceLayoutWeight() -> MLXArray {
        let weightG = parametrizations.weight.original0
        let weightV = parametrizations.weight.original1
        return weightG * weightV / mossNormalizeWeightExceptOutput(weightV)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = sourceLayoutWeight().transposed(0, 2, 1)
        let y = MLX.conv1d(
            x.transposed(0, 2, 1),
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation
        )
        return (y + bias).transposed(0, 2, 1)
    }
}

private final class MossLayerScale: Module {
    @ModuleInfo var scale: MLXArray

    init(channels: Int, initialValue: Float) {
        _scale.wrappedValue = MLXArray.full([channels], values: MLXArray(initialValue), dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * scale
    }
}

private final class MossAudioIdentity: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

@inline(__always)
private func mossAudioCallUnary(_ module: Module, _ x: MLXArray) -> MLXArray {
    (module as! UnaryLayer).callAsFunction(x)
}

private func mossApplyLayerScale(_ module: Module, to x: MLXArray) -> MLXArray {
    if let scale = module as? MossLayerScale {
        return scale(x)
    }
    return x
}

private protocol MossAudioTokenizerStage: AnyObject {
    var downsampleRatio: Int { get }
    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray) throws -> (MLXArray, MLXArray)
}

private func mossApplyAudioRoPE(
    q: MLXArray,
    k: MLXArray,
    maxPeriod: Float,
    offset: Int = 0
) -> (MLXArray, MLXArray) {
    let time = q.dim(2)
    let dim = q.dim(3)
    let freqs = exp(
        MLX.arange(dim / 2, dtype: .float32)
            * (MLXArray(-log(maxPeriod) * 2.0 / Float(dim)))
    )
    let positions = (MLX.arange(time, dtype: .float32) + Float(offset))
    let phase = positions.reshaped([1, 1, time, 1]) * freqs.reshaped([1, 1, 1, dim / 2])
    let cosPhase = cos(phase)
    let sinPhase = sin(phase)

    let qPairs = q.asType(.float32).reshaped(q.shape.dropLast() + [dim / 2, 2])
    let kPairs = k.asType(.float32).reshaped(k.shape.dropLast() + [dim / 2, 2])
    let qr = qPairs[0..., 0..., 0..., 0..., 0]
    let qi = qPairs[0..., 0..., 0..., 0..., 1]
    let kr = kPairs[0..., 0..., 0..., 0..., 0]
    let ki = kPairs[0..., 0..., 0..., 0..., 1]

    let qOut = MLX.stacked([qr * cosPhase - qi * sinPhase, qr * sinPhase + qi * cosPhase], axis: -1)
    let kOut = MLX.stacked([kr * cosPhase - ki * sinPhase, kr * sinPhase + ki * cosPhase], axis: -1)
    return (qOut.reshaped(q.shape).asType(q.dtype), kOut.reshaped(k.shape).asType(k.dtype))
}

private final class MossAudioMultiheadAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let causal: Bool
    let context: Int?
    let maxPeriod: Float
    let useRoPE: Bool
    let scale: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(
        embedDim: Int,
        numHeads: Int,
        causal: Bool,
        context: Int?,
        maxPeriod: Float,
        useRoPE: Bool
    ) {
        precondition(embedDim % numHeads == 0, "embed_dim must be divisible by num_heads")
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.causal = causal
        self.context = context
        self.maxPeriod = maxPeriod
        self.useRoPE = useRoPE
        self.scale = pow(Float(headDim), -0.5)
        _inProj.wrappedValue = Linear(embedDim, 3 * embedDim, bias: false)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: false)
    }

    private func mask(inputLengths: MLXArray, maxSequenceLength: Int, dtype: DType) -> MLXArray {
        let positions = MLXArray(0 ..< maxSequenceLength).asType(.int32)
        var allowed = positions.reshaped([1, 1, 1, maxSequenceLength])
            .< inputLengths.asType(.int32).reshaped([inputLengths.dim(0), 1, 1, 1])
        let delta = positions.reshaped([1, 1, maxSequenceLength, 1])
            - positions.reshaped([1, 1, 1, maxSequenceLength])
        if causal {
            allowed = logicalAnd(allowed, delta .>= MLXArray(Int32(0)))
        }
        if let context {
            allowed = logicalAnd(allowed, delta .< MLXArray(Int32(context)))
        }
        let minimum = MLXArray(Float(dtype.finfo?.min ?? -Double.greatestFiniteMagnitude))
        return MLX.where(allowed, MLXArray(0.0), minimum).asType(dtype)
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let qkv = inProj(x).reshaped(batch, time, 3, numHeads, headDim)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if useRoPE {
            (q, k) = mossApplyAudioRoPE(q: q, k: k, maxPeriod: maxPeriod)
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask(inputLengths: inputLengths, maxSequenceLength: time, dtype: x.dtype)
        )
        let validQuery = MLXArray(0 ..< time).asType(.int32).reshaped([1, 1, time, 1])
            .< inputLengths.asType(.int32).reshaped([inputLengths.dim(0), 1, 1, 1])
        out = MLX.where(validQuery, out, MLXArray.zeros([], dtype: out.dtype))
        out = out.transposed(0, 2, 1, 3).reshaped(batch, time, embedDim)
        return outProj(out)
    }
}

private final class MossAudioTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MossAudioMultiheadAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var ffn: [Module]
    @ModuleInfo(key: "layer_scale_1") var layerScale1: Module
    @ModuleInfo(key: "layer_scale_2") var layerScale2: Module

    init(
        dModel: Int,
        numHeads: Int,
        dimFeedforward: Int,
        causal: Bool,
        context: Int?,
        positionalEmbedding: String,
        maxPeriod: Float,
        layerScale: Float?,
        norm: String
    ) throws {
        guard norm == "layer_norm" else {
            throw MossTTSNanoError.invalidInput("Unsupported MOSS audio tokenizer norm: \(norm)")
        }
        _selfAttention.wrappedValue = MossAudioMultiheadAttention(
            embedDim: dModel,
            numHeads: numHeads,
            causal: causal,
            context: context,
            maxPeriod: maxPeriod,
            useRoPE: ["rope", "sin_rope"].contains(positionalEmbedding)
        )
        _norm1.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        _norm2.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        _ffn.wrappedValue = [
            Linear(dModel, dimFeedforward, bias: false),
            MossAudioIdentity(),
            Linear(dimFeedforward, dModel, bias: false),
        ]
        if let layerScale {
            _layerScale1.wrappedValue = MossLayerScale(channels: dModel, initialValue: layerScale)
            _layerScale2.wrappedValue = MossLayerScale(channels: dModel, initialValue: layerScale)
        } else {
            _layerScale1.wrappedValue = MossAudioIdentity()
            _layerScale2.wrappedValue = MossAudioIdentity()
        }
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray) -> MLXArray {
        let residual = x
        let attended = selfAttention(norm1(x), inputLengths: inputLengths)
        var hidden = residual + mossApplyLayerScale(layerScale1, to: attended)
        let mlpInput = norm2(hidden)
        let fcIn = ffn[0] as! Linear
        let fcOut = ffn[2] as! Linear
        let mlpOut = fcOut(mossExactGELU(fcIn(mlpInput)))
        hidden = hidden + mossApplyLayerScale(layerScale2, to: mlpOut)
        return hidden
    }
}

private final class MossAudioTransformer: Module {
    let positionalEmbedding: String
    let maxPeriod: Float
    let positionalScale: Float

    @ModuleInfo(key: "layers") var layers: [MossAudioTransformerLayer]

    init(
        dModel: Int,
        numHeads: Int,
        numLayers: Int,
        dimFeedforward: Int,
        causal: Bool,
        context: Int?,
        positionalEmbedding: String,
        maxPeriod: Float,
        positionalScale: Float,
        layerScale: Float?,
        norm: String,
        gating: String
    ) throws {
        guard gating == "none" else {
            throw MossTTSNanoError.invalidInput("Unsupported MOSS audio tokenizer gating: \(gating)")
        }
        self.positionalEmbedding = positionalEmbedding
        self.maxPeriod = maxPeriod
        self.positionalScale = positionalScale
        _layers.wrappedValue = try (0 ..< numLayers).map { _ in
            try MossAudioTransformerLayer(
                dModel: dModel,
                numHeads: numHeads,
                dimFeedforward: dimFeedforward,
                causal: causal,
                context: context,
                positionalEmbedding: positionalEmbedding,
                maxPeriod: maxPeriod,
                layerScale: layerScale,
                norm: norm
            )
        }
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray) -> MLXArray {
        var hidden = x
        if ["sin", "sin_rope"].contains(positionalEmbedding) {
            let positions = MLX.arange(hidden.dim(1), dtype: hidden.dtype)
            let half = hidden.dim(2) / 2
            let scale = pow(MLXArray(maxPeriod), MLX.arange(half, dtype: hidden.dtype) / MLXArray(Float(max(half - 1, 1))))
            let phase = positions.reshaped([hidden.dim(1), 1]) / scale.reshaped([1, half])
            let emb = MLX.concatenated([cos(phase), sin(phase)], axis: -1)
            hidden = hidden + MLXArray(positionalScale).asType(hidden.dtype) * emb.expandedDimensions(axis: 0)
        }
        for layer in layers {
            hidden = layer(hidden, inputLengths: inputLengths)
        }
        return hidden
    }
}

private final class MossProjectedTransformer: Module, MossAudioTokenizerStage {
    let downsampleRatio = 1

    @ModuleInfo(key: "input_proj") var inputProj: Module
    @ModuleInfo var transformer: MossAudioTransformer
    @ModuleInfo(key: "output_proj") var outputProj: Module

    init(kwargs: [String: SendableValue], context: Int) throws {
        let inputDimension = kwargs.int("input_dimension", fallback: 0)
        let outputDimension = kwargs.int("output_dimension", fallback: 0)
        let dModel = kwargs.int("d_model", fallback: 0)
        _inputProj.wrappedValue = inputDimension == dModel
            ? MossAudioIdentity()
            : Linear(inputDimension, dModel, bias: false)
        _transformer.wrappedValue = try MossAudioTransformer(
            dModel: dModel,
            numHeads: kwargs.int("num_heads", fallback: 1),
            numLayers: kwargs.int("num_layers", fallback: 1),
            dimFeedforward: kwargs.int("dim_feedforward", fallback: 4 * dModel),
            causal: kwargs.bool("causal", fallback: true),
            context: context,
            positionalEmbedding: kwargs.string("positional_embedding", fallback: "rope"),
            maxPeriod: kwargs.float("max_period", fallback: 10_000),
            positionalScale: kwargs.float("positional_scale", fallback: 1),
            layerScale: kwargs.optionalFloat("layer_scale"),
            norm: kwargs.string("norm", fallback: "layer_norm"),
            gating: kwargs.string("gating", fallback: "none")
        )
        _outputProj.wrappedValue = outputDimension == dModel
            ? MossAudioIdentity()
            : Linear(dModel, outputDimension, bias: false)
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray) throws -> (MLXArray, MLXArray) {
        var hidden = mossAudioCallUnary(inputProj, x.transposed(0, 2, 1))
        hidden = transformer(hidden, inputLengths: inputLengths)
        let output = mossAudioCallUnary(outputProj, hidden)
        return (output.transposed(0, 2, 1), inputLengths)
    }
}

private final class MossPatchedPretransform: Module, MossAudioTokenizerStage {
    let patchSize: Int
    let isDownsample: Bool
    var downsampleRatio: Int { patchSize }

    init(patchSize: Int, isDownsample: Bool) {
        self.patchSize = patchSize
        self.isDownsample = isDownsample
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray) throws -> (MLXArray, MLXArray) {
        let batch = x.dim(0)
        if isDownsample {
            let channels = x.dim(1)
            var hidden = x.reshaped(batch, channels, -1, patchSize)
            hidden = hidden.transposed(0, 1, 3, 2).reshaped(batch, channels * patchSize, -1)
            return (hidden, inputLengths / MLXArray(Int32(patchSize)))
        } else {
            let channelsPatch = x.dim(1)
            let length = x.dim(2)
            let channels = channelsPatch / patchSize
            var hidden = x.reshaped(batch, channels, patchSize, length)
            hidden = hidden.transposed(0, 1, 3, 2).reshaped(batch, channels, length * patchSize)
            return (hidden, inputLengths * MLXArray(Int32(patchSize)))
        }
    }
}

private final class MossLFQ: Module {
    let inputDim: Int
    let codebookSize: Int
    let codebookDim: Int

    @ModuleInfo(key: "in_proj") var inProj: MossWNConv1d
    @ModuleInfo(key: "out_proj") var outProj: MossWNConv1d
    @ModuleInfo var codebook: Embedding

    init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.inputDim = inputDim
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        _inProj.wrappedValue = MossWNConv1d(inChannels: inputDim, outChannels: codebookDim, kernelSize: 1)
        _outProj.wrappedValue = MossWNConv1d(inChannels: codebookDim, outChannels: inputDim, kernelSize: 1)
        _codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }

    func decodeCodeWithoutOutProjection(_ embedID: MLXArray) -> MLXArray {
        codebook(embedID).transposed(0, 2, 1)
    }

    func decodeCode(_ embedID: MLXArray) -> MLXArray {
        outProj(decodeCodeWithoutOutProjection(embedID).asType(.float32))
    }

    func decodeLatents(_ latents: MLXArray) -> (MLXArray, MLXArray) {
        var encodings = latents.transposed(0, 2, 1).reshaped(-1, latents.dim(1))
        var codebookWeight = codebook.weight.asType(.float32)
        encodings = mossL2Normalize(encodings.asType(.float32), axis: -1)
        codebookWeight = mossL2Normalize(codebookWeight, axis: -1)
        let dist = sum(square(encodings), axis: 1, keepDims: true)
            - 2.0 * matmul(encodings, codebookWeight.transposed(1, 0))
            + sum(square(codebookWeight), axis: 1, keepDims: true).transposed(1, 0)
        let indices = argMax(-dist, axis: 1).reshaped(latents.dim(0), -1)
        return (decodeCodeWithoutOutProjection(indices).asType(.float32), indices)
    }

    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let zE = inProj(z.asType(.float32)).asType(.float32)
        let decoded = decodeLatents(zE)
        let zQ = outProj(decoded.0.asType(.float32)).asType(.float32)
        return (zQ, decoded.1, zE)
    }
}

private final class MossResidualLFQ: Module {
    let inputDim: Int
    let rvqDim: Int
    let outputDim: Int
    let numQuantizers: Int
    let codebookSize: Int
    let codebookDim: Int

    @ModuleInfo(key: "input_proj") var inputProj: MossWNConv1d
    @ModuleInfo(key: "output_proj") var outputProj: MossWNConv1d
    @ModuleInfo var quantizers: [MossLFQ]

    init(kwargs: [String: SendableValue]) {
        let resolvedInputDim = kwargs.int("input_dim", fallback: 1_024)
        let resolvedRVQDim = kwargs.int("rvq_dim", fallback: resolvedInputDim)
        let resolvedOutputDim = kwargs.int("output_dim", fallback: resolvedInputDim)
        let resolvedNumQuantizers = kwargs.int("num_quantizers", fallback: 32)
        let resolvedCodebookSize = kwargs.int("codebook_size", fallback: 1_024)
        let resolvedCodebookDim = kwargs.int("codebook_dim", fallback: 8)
        self.inputDim = resolvedInputDim
        self.rvqDim = resolvedRVQDim
        self.outputDim = resolvedOutputDim
        self.numQuantizers = resolvedNumQuantizers
        self.codebookSize = resolvedCodebookSize
        self.codebookDim = resolvedCodebookDim
        _inputProj.wrappedValue = MossWNConv1d(inChannels: inputDim, outChannels: rvqDim, kernelSize: 1)
        _outputProj.wrappedValue = MossWNConv1d(inChannels: rvqDim, outChannels: outputDim, kernelSize: 1)
        _quantizers.wrappedValue = (0 ..< resolvedNumQuantizers).map { _ in
            MossLFQ(
                inputDim: resolvedRVQDim,
                codebookSize: resolvedCodebookSize,
                codebookDim: resolvedCodebookDim
            )
        }
    }

    func callAsFunction(
        _ z: MLXArray,
        inputLength: MLXArray,
        nQuantizers: Int? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let projected = inputProj(z.asType(.float32)).asType(.float32)
        let batch = projected.dim(0)
        let maxTime = projected.dim(2)
        let mask = MLXArray(0 ..< maxTime).asType(.int32).reshaped([1, maxTime])
            .< inputLength.asType(.int32).reshaped([inputLength.dim(0), 1])
        let updateMask = mask.expandedDimensions(axis: 1)
        var quantizedOut = MLXArray.zeros(projected.shape, dtype: .float32)
        var residual = projected
        var indices: [MLXArray] = []
        let activeQuantizers = min(nQuantizers ?? numQuantizers, numQuantizers)
        for quantizer in quantizers.prefix(activeQuantizers) {
            let (zQI, indicesI, _) = quantizer(residual * updateMask)
            quantizedOut = quantizedOut + zQI * updateMask
            residual = residual - zQI * updateMask
            indices.append(indicesI)
        }

        let allIndices = indices.isEmpty
            ? MLXArray.zeros([0, batch, maxTime], type: Int32.self)
            : MLX.stacked(indices, axis: 0).asType(.int32)
        return (outputProj(quantizedOut).asType(.float32), allIndices, inputLength)
    }

    func decodeCodes(_ codes: MLXArray) -> MLXArray {
        let nq = codes.dim(0)
        let batch = codes.dim(1)
        let time = codes.dim(2)
        var emb = MLXArray.zeros([batch, rvqDim, time], dtype: .float32)
        for (index, quantizer) in quantizers.prefix(nq).enumerated() {
            emb = emb + quantizer.decodeCode(codes[index]).asType(.float32)
        }
        return outputProj(emb).asType(.float32)
    }
}

public final class MLXMossAudioTokenizer: Module, MossAudioTokenizing, @unchecked Sendable {
    public let config: MossAudioTokenizerConfig
    public let sampleRate: Int
    public let samplingRate: Int
    public let downsampleRate: Int
    public let channels: Int
    public let enableChannelInterleave: Bool
    public let numQuantizers: Int

    @ModuleInfo var encoder: [Module]
    @ModuleInfo fileprivate var quantizer: MossResidualLFQ
    @ModuleInfo var decoder: [Module]

    public init(config: MossAudioTokenizerConfig) throws {
        self.config = config
        self.sampleRate = config.sampleRate
        self.samplingRate = config.samplingRate
        self.downsampleRate = config.downsampleRate
        self.channels = config.numberChannels
        self.enableChannelInterleave = config.enableChannelInterleave

        let channelFactor = enableChannelInterleave && channels > 1 ? channels : 1
        var currentFrameRate = Double(samplingRate * channelFactor)
        var encoderStages: [Module] = []
        for kwargs in config.encoderKwargs {
            let stage = try Self.makeStage(
                kwargs: kwargs,
                isDownsample: true,
                currentFrameRate: currentFrameRate,
                config: config
            )
            encoderStages.append(stage.module)
            currentFrameRate /= Double(stage.downsampleRatio)
        }

        let quantizerType = config.quantizerKwargs.string("quantizer_type", fallback: config.quantizerType)
        guard ["rlfq", "random_prefix_rlfq"].contains(quantizerType) else {
            throw MossTTSNanoError.invalidInput("Unsupported MOSS quantizer_type: \(quantizerType)")
        }
        let quantizer = MossResidualLFQ(kwargs: config.quantizerKwargs)
        self.numQuantizers = quantizer.numQuantizers

        var decoderStages: [Module] = []
        for kwargs in config.decoderKwargs {
            let stage = try Self.makeStage(
                kwargs: kwargs,
                isDownsample: false,
                currentFrameRate: currentFrameRate,
                config: config
            )
            decoderStages.append(stage.module)
            currentFrameRate *= Double(stage.downsampleRatio)
        }

        _encoder.wrappedValue = encoderStages
        _quantizer.wrappedValue = quantizer
        _decoder.wrappedValue = decoderStages
    }

    private static func makeStage(
        kwargs: [String: SendableValue],
        isDownsample: Bool,
        currentFrameRate: Double,
        config: MossAudioTokenizerConfig
    ) throws -> (module: Module, downsampleRatio: Int) {
        let moduleType = kwargs.string("module_type", fallback: "")
        switch moduleType {
        case "PatchedPretransform":
            let module = MossPatchedPretransform(
                patchSize: kwargs.int("patch_size", fallback: 1),
                isDownsample: isDownsample
            )
            return (module, module.downsampleRatio)
        case "Transformer":
            let contextDuration = kwargs.float(
                "context_duration",
                fallback: config.causalTransformerContextDuration
            )
            let context = Int(round(currentFrameRate * Double(contextDuration)))
            let module = try MossProjectedTransformer(kwargs: kwargs, context: context)
            return (module, module.downsampleRatio)
        default:
            throw MossTTSNanoError.invalidInput("Unsupported MOSS audio tokenizer module_type: \(moduleType)")
        }
    }

    public static func fromModelDirectory(_ modelDir: URL) throws -> MLXMossAudioTokenizer {
        let config = try MossAudioTokenizerConfig.fromFile(modelDir.appendingPathComponent("config.json"))
        let model = try MLXMossAudioTokenizer(config: config)
        let weights = try loadWeights(from: modelDir)
        try model.update(parameters: ModuleParameters.unflattened(sanitize(weights: weights)), verify: .all)
        eval(model)
        return model
    }

    private static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        for (key, value) in weights {
            var mappedKey: String?
            if key.contains(".self_attn.in_projs.0.") {
                mappedKey = key.replacingOccurrences(of: ".self_attn.in_projs.0.", with: ".self_attn.in_proj.")
            } else if key.contains(".self_attn.out_projs.0.") {
                mappedKey = key.replacingOccurrences(of: ".self_attn.out_projs.0.", with: ".self_attn.out_proj.")
            } else if key.contains(".transformer.layers.") && key.contains(".linear1.") {
                mappedKey = key.replacingOccurrences(of: ".linear1.", with: ".ffn.0.")
            } else if key.contains(".transformer.layers.") && key.contains(".linear2.") {
                mappedKey = key.replacingOccurrences(of: ".linear2.", with: ".ffn.2.")
            }

            if let mappedKey {
                result[mappedKey] = value
                result.removeValue(forKey: key)
            }
        }
        return result
    }

    public static func fromPretrained(
        _ source: String = mossDefaultAudioTokenizerRepo,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> MLXMossAudioTokenizer {
        guard let repoID = Repo.ID(rawValue: source) else {
            let url = URL(fileURLWithPath: NSString(string: source).expandingTildeInPath)
            return try fromModelDirectory(url)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.json", "*.index.json", "*.md"],
            hfToken: hfToken,
            cache: cache
        )
        return try fromModelDirectory(modelDir)
    }

    private func prepareAudioArray(_ audio: MLXArray) throws -> MLXArray {
        var shape = audio.shape
        var samples = audio.asType(.float32).asArray(Float.self)
        var sampleCount: Int
        var channelCount: Int
        if audio.ndim == 1 {
            sampleCount = audio.dim(0)
            channelCount = 1
        } else if audio.ndim == 2 {
            if audio.dim(0) <= 8 && audio.dim(0) < audio.dim(1) {
                channelCount = audio.dim(0)
                sampleCount = audio.dim(1)
                var transposed = Array(repeating: Float(0), count: samples.count)
                for ch in 0 ..< channelCount {
                    for i in 0 ..< sampleCount {
                        transposed[i * channelCount + ch] = samples[ch * sampleCount + i]
                    }
                }
                samples = transposed
                shape = [sampleCount, channelCount]
            } else {
                sampleCount = audio.dim(0)
                channelCount = audio.dim(1)
            }
        } else {
            throw MossTTSNanoError.invalidInput("Unsupported audio shape: \(audio.shape)")
        }

        if channelCount != channels {
            if channelCount == 1 && channels > 1 {
                var repeated = Array(repeating: Float(0), count: sampleCount * channels)
                for i in 0 ..< sampleCount {
                    for ch in 0 ..< channels {
                        repeated[i * channels + ch] = samples[i]
                    }
                }
                samples = repeated
                channelCount = channels
            } else if channelCount > 1 && channels == 1 {
                var mono = Array(repeating: Float(0), count: sampleCount)
                for i in 0 ..< sampleCount {
                    var sum: Float = 0
                    for ch in 0 ..< channelCount {
                        sum += samples[i * channelCount + ch]
                    }
                    mono[i] = sum / Float(channelCount)
                }
                samples = mono
                channelCount = 1
            } else {
                throw MossTTSNanoError.invalidInput(
                    "Unsupported reference audio channel conversion: \(channelCount) -> \(channels)"
                )
            }
            shape = [sampleCount, channelCount]
        }

        var channelFirst = Array(repeating: Float(0), count: sampleCount * channelCount)
        for i in 0 ..< sampleCount {
            for ch in 0 ..< channelCount {
                channelFirst[ch * sampleCount + i] = samples[i * channelCount + ch]
            }
        }
        _ = shape
        return MLXArray(channelFirst, [channelCount, sampleCount]).asType(.float32)
    }

    private func prepareWaveformBatch(_ waveforms: [MLXArray]) throws -> (MLXArray, MLXArray) {
        guard !waveforms.isEmpty else {
            throw MossTTSNanoError.invalidInput("waveforms must not be empty")
        }
        let lengths = waveforms.map { $0.dim($0.ndim - 1) }
        let maxLength = lengths.max() ?? 0
        var padded: [MLXArray] = []
        for waveform in waveforms {
            var current = waveform.ndim == 1 ? waveform.expandedDimensions(axis: 0) : waveform
            guard current.dim(0) == channels else {
                throw MossTTSNanoError.invalidInput("Expected waveform shape [\(channels), samples], got \(current.shape)")
            }
            let pad = maxLength - current.dim(1)
            if pad > 0 {
                current = MLX.padded(current, widths: [.init(0), .init((0, pad))])
            }
            padded.append(current)
        }
        return (
            MLX.stacked(padded, axis: 0),
            MLXArray(lengths.map(Int32.init)).asType(.int32)
        )
    }

    private func prepareCodesBatch(
        _ codesList: [MLXArray],
        numQuantizers: Int?
    ) throws -> (MLXArray, MLXArray, Int) {
        guard !codesList.isEmpty else {
            throw MossTTSNanoError.invalidInput("codesList must not be empty")
        }
        let available = codesList.map { $0.dim(0) }
        let effectiveNQ = numQuantizers ?? available[0]
        guard (available.min() ?? 0) >= effectiveNQ else {
            throw MossTTSNanoError.invalidInput("numQuantizers=\(effectiveNQ) exceeds available quantizers")
        }
        let lengths = codesList.map { $0.dim($0.ndim - 1) }
        let maxLength = lengths.max() ?? 0
        var flat = Array(repeating: Int32(0), count: effectiveNQ * codesList.count * maxLength)
        for (batchIndex, codes) in codesList.enumerated() {
            let time = codes.dim(1)
            let values = codes.asType(.int32).asArray(Int32.self)
            for q in 0 ..< effectiveNQ {
                for t in 0 ..< time {
                    flat[(q * codesList.count + batchIndex) * maxLength + t] = values[q * time + t]
                }
            }
        }
        return (
            MLXArray(flat, [effectiveNQ, codesList.count, maxLength]).asType(.int32),
            MLXArray(lengths.map(Int32.init)).asType(.int32),
            effectiveNQ
        )
    }

    private func flattenChannelsForCodec(
        inputValues: MLXArray,
        inputLengths: MLXArray
    ) -> (MLXArray, MLXArray) {
        var values = inputValues
        var lengths = inputLengths
        let remainder = values.dim(2) % downsampleRate
        if remainder != 0 {
            let padLength = downsampleRate - remainder
            values = MLX.padded(values, widths: [.init(0), .init(0), .init((0, padLength))])
        }
        if channels > 1 && enableChannelInterleave {
            values = values.transposed(0, 2, 1).reshaped(values.dim(0), 1, -1)
            lengths = lengths * MLXArray(Int32(channels))
        }
        return (values, lengths)
    }

    private func restoreChannelsFromCodec(
        outputValues: MLXArray,
        outputLengths: MLXArray
    ) -> (MLXArray, MLXArray) {
        guard channels > 1 && enableChannelInterleave else {
            return (outputValues.asType(.float32), outputLengths)
        }
        let batch = outputValues.dim(0)
        var values = outputValues[0..., 0, 0...].reshaped(batch, -1, channels)
        values = values.transposed(0, 2, 1).asType(.float32)
        return (values, outputLengths / MLXArray(Int32(channels)))
    }

    private func encodeFrame(
        inputValues: MLXArray,
        inputLengths: MLXArray? = nil,
        nQuantizers: Int? = nil
    ) throws -> (MLXArray, MLXArray, MLXArray) {
        var values = inputValues
        if values.ndim == 1 {
            values = values.reshaped(1, 1, -1)
        } else if values.ndim == 2 {
            values = channels == 1 ? values.expandedDimensions(axis: 1) : values.expandedDimensions(axis: 0)
        }
        let lengths = inputLengths ?? MLX.full(
            [values.dim(0)],
            values: Int32(values.dim(2)),
            type: Int32.self
        )
        var (hidden, hiddenLengths) = flattenChannelsForCodec(inputValues: values, inputLengths: lengths)
        for module in encoder {
            let stage = module as! MossAudioTokenizerStage
            (hidden, hiddenLengths) = try stage(hidden, inputLengths: hiddenLengths)
        }
        let result = quantizer(hidden.asType(.float32), inputLength: hiddenLengths, nQuantizers: nQuantizers)
        return (result.1, result.2, hidden.asType(.float32))
    }

    private func decodeFrame(
        codes: MLXArray,
        codesLengths: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray) {
        guard codes.ndim == 3 else {
            throw MossTTSNanoError.invalidInput("Expected codes shape [nq, batch, time], got \(codes.shape)")
        }
        let lengths = codesLengths ?? MLX.full([codes.dim(1)], values: Int32(codes.dim(2)), type: Int32.self)
        var audio = quantizer.decodeCodes(codes.asType(.int32))
        var audioLengths = lengths
        for module in decoder {
            let stage = module as! MossAudioTokenizerStage
            (audio, audioLengths) = try stage(audio, inputLengths: audioLengths)
        }
        return restoreChannelsFromCodec(outputValues: audio, outputLengths: audioLengths)
    }

    public func encodeAudio(_ audio: MLXArray, numQuantizers: Int) throws -> MLXArray {
        let waveform = try prepareAudioArray(audio)
        let batch = try prepareWaveformBatch([waveform])
        let encoded = try encodeFrame(
            inputValues: batch.0,
            inputLengths: batch.1,
            nQuantizers: numQuantizers
        )
        eval(encoded.0, encoded.1)
        let codeLength = encoded.1.asArray(Int32.self).first.map(Int.init) ?? encoded.0.dim(2)
        return encoded.0[0..., 0, 0..<codeLength].transposed(1, 0).asType(.int32)
    }

    public func decodeAudioCodes(_ audioTokenIDs: MLXArray, numQuantizers: Int) throws -> MLXArray {
        var codes = audioTokenIDs.asType(.int32)
        if codes.ndim == 3 {
            guard codes.dim(0) == 1 else {
                throw MossTTSNanoError.notImplemented("Batched MOSS audio-tokenizer decode is not implemented.")
            }
            codes = codes[0]
        }
        guard codes.ndim == 2 else {
            throw MossTTSNanoError.invalidInput("Expected codes shape [frames, nq], got \(codes.shape)")
        }
        guard codes.dim(0) > 0 else {
            return MLXArray.zeros([0, channels], dtype: .float32)
        }
        let effectiveNQ = min(numQuantizers, codes.dim(1))
        let prepared = try prepareCodesBatch(
            [codes[0..., 0..<effectiveNQ].transposed(1, 0)],
            numQuantizers: effectiveNQ
        )
        let decoded = try decodeFrame(codes: prepared.0, codesLengths: prepared.1)
        eval(decoded.0, decoded.1)
        let audioLength = decoded.1.asArray(Int32.self).first.map(Int.init) ?? decoded.0.dim(2)
        return decoded.0[0, 0..., 0..<audioLength].transposed(1, 0).asType(.float32)
    }
}
