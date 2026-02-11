import Foundation
@preconcurrency import MLX
import MLXAudioCodecs
import MLXLMCommon
import MLXNN

@inline(__always)
private func pocketGetExtraPaddingForConv1d(_ xs: MLXArray, ksize: Int, stride: Int, paddingTotal: Int = 0) -> Int {
    let len = xs.shape[2]
    let nframes = max(len + paddingTotal - ksize, 0)
    let nf = Double(nframes) / Double(stride) + 1.0
    let idealLen = (Int(ceil(nf)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLen - len)
}

@inline(__always)
private func pocketPadForConv1d(_ x: MLXArray, ksize: Int, stride: Int, paddingTotal: Int = 0) -> MLXArray {
    let extra = pocketGetExtraPaddingForConv1d(x, ksize: ksize, stride: stride, paddingTotal: paddingTotal)
    if extra <= 0 { return x }
    return MLX.padded(x, widths: [.init(0), .init(0), .init((0, extra))])
}

public final class DummyQuantizer: Module {
    @ModuleInfo(key: "output_proj") public var output_proj: MLXAudioCodecs.Conv1d

    public init(dimension: Int, outputDimension: Int) {
        self._output_proj = ModuleInfo(wrappedValue: MLXAudioCodecs.Conv1d(
            inChannels: dimension,
            outChannels: outputDimension,
            ksize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            dilation: 1,
            bias: false
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        output_proj(x)
    }
}

public final class MimiAdapter: Module {
    @ModuleInfo(key: "encoder") public var encoder: SeanetEncoder
    @ModuleInfo(key: "decoder") public var decoder: SeanetDecoder
    @ModuleInfo(key: "encoder_transformer") public var encoder_transformer: ProjectedTransformer
    @ModuleInfo(key: "decoder_transformer") public var decoder_transformer: ProjectedTransformer
    @ModuleInfo(key: "quantizer") public var quantizer: DummyQuantizer
    @ModuleInfo(key: "downsample") public var downsample: ConvDownsample1d?
    @ModuleInfo(key: "upsample") public var upsample: ConvTrUpsample1d?

    public let frameRate: Double
    public let sampleRate: Int
    public let channels: Int
    public let encoderFrameRate: Double
    public let dimension: Int

    public var encoderCache: [KVCacheSimple]
    public var decoderCache: [KVCacheSimple]

    public init(
        encoder: SeanetEncoder,
        decoder: SeanetDecoder,
        quantizer: DummyQuantizer,
        frameRate: Double,
        encoderFrameRate: Double,
        sampleRate: Int,
        channels: Int,
        encoderTransformer: ProjectedTransformer,
        decoderTransformer: ProjectedTransformer,
        dimension: Int
    ) {
        self._encoder = ModuleInfo(wrappedValue: encoder)
        self._decoder = ModuleInfo(wrappedValue: decoder)
        self._encoder_transformer = ModuleInfo(wrappedValue: encoderTransformer)
        self._decoder_transformer = ModuleInfo(wrappedValue: decoderTransformer)
        self._quantizer = ModuleInfo(wrappedValue: quantizer)
        self.frameRate = frameRate
        self.sampleRate = sampleRate
        self.channels = channels
        self.encoderFrameRate = encoderFrameRate
        self.dimension = dimension

        if encoderFrameRate != frameRate {
            let strideDouble = encoderFrameRate / frameRate
            precondition(strideDouble == floor(strideDouble), "Only integer strides are supported")
            let stride = Int(strideDouble)
            self._downsample = ModuleInfo(wrappedValue: ConvDownsample1d(stride: stride, dim: dimension, causal: true))
            self._upsample = ModuleInfo(wrappedValue: ConvTrUpsample1d(stride: stride, dim: dimension, causal: true))
        } else {
            self._downsample = ModuleInfo(wrappedValue: nil)
            self._upsample = ModuleInfo(wrappedValue: nil)
        }

        self.encoderCache = encoderTransformer.makeCache()
        self.decoderCache = decoderTransformer.makeCache()
        super.init()
    }

    public var frameSize: Int { Int(Double(sampleRate) / frameRate) }

    public func resetState() {
        encoder.resetState()
        decoder.resetState()
        downsample?.resetState()
        upsample?.resetState()
        for c in encoderCache { c.trim(c.offset) }
        for c in decoderCache { c.trim(c.offset) }
    }

    private func toFrameRate(_ x: MLXArray) -> MLXArray {
        guard let downsample else { return x }
        return downsample(x)
    }

    private func toEncoderFrameRate(_ x: MLXArray) -> MLXArray {
        guard let upsample else { return x }
        return upsample(x)
    }

    private func toEncoderFrameRateStep(_ x: MLXArray) -> MLXArray {
        guard let upsample else { return x }
        return upsample.step(x)
    }

    public func encodeToLatent(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 3, "encodeToLatent expects audio of shape [B,C,T]")
        encoder.resetState()
        for c in encoderCache { c.trim(c.offset) }
        downsample?.resetState()

        let frameSize = self.frameSize
        var xs = pocketPadForConv1d(x, ksize: frameSize, stride: frameSize)
        xs = encoder(xs)
        xs = encoder_transformer(xs, cache: encoderCache)[0]
        return toFrameRate(xs)
    }

    public func decodeFromLatent(_ latent: MLXArray) -> MLXArray {
        decoder.resetState()
        for c in decoderCache { c.trim(c.offset) }
        upsample?.resetState()

        var emb = toEncoderFrameRate(latent)
        emb = decoder_transformer(emb, cache: decoderCache)[0]
        return decoder(emb)
    }

    public func decodeStep(_ latent: MLXArray) -> MLXArray {
        var emb = toEncoderFrameRateStep(latent)
        emb = decoder_transformer(emb, cache: decoderCache)[0]
        return decoder.step(emb)
    }

    public static func fromConfig(_ config: PocketTTSMimiConfig) -> MimiAdapter {
        let padMode: PadMode = (config.seanet.padMode == "constant") ? .constant : .edge
        let seanetCfg = SeanetConfig(
            dimension: config.seanet.dimension,
            channels: config.seanet.channels,
            causal: true,
            nfilters: config.seanet.nFilters,
            nresidualLayers: config.seanet.nResidualLayers,
            ratios: config.seanet.ratios,
            ksize: config.seanet.kernelSize,
            residualKsize: config.seanet.residualKernelSize,
            lastKsize: config.seanet.lastKernelSize,
            dilationBase: config.seanet.dilationBase,
            padMode: padMode,
            trueSkip: true,
            compress: config.seanet.compress
        )

        let encoder = SeanetEncoder(cfg: seanetCfg)
        let decoder = SeanetDecoder(cfg: seanetCfg)

        let transformerCfg = TransformerConfig(
            dModel: config.transformer.dModel,
            numHeads: config.transformer.numHeads,
            numLayers: config.transformer.numLayers,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: Float(config.transformer.layerScale),
            positionalEmbedding: "rope",
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: false,
            gating: false,
            norm: "layer_norm",
            context: config.transformer.context,
            maxPeriod: Int(config.transformer.maxPeriod),
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedforward: config.transformer.dimFeedforward,
            convLayout: true
        )

        let encoderTransformer = ProjectedTransformer(
            cfg: transformerCfg,
            inputDim: config.transformer.inputDimension,
            outputDims: config.transformer.outputDimensions
        )
        let decoderTransformer = ProjectedTransformer(
            cfg: transformerCfg,
            inputDim: config.transformer.inputDimension,
            outputDims: config.transformer.outputDimensions
        )

        let quantizer = DummyQuantizer(dimension: config.quantizer.dimension, outputDimension: config.quantizer.outputDimension)
        let encoderFrameRate = Double(config.sampleRate) / Double(config.seanet.ratios.reduce(1, *))

        return MimiAdapter(
            encoder: encoder,
            decoder: decoder,
            quantizer: quantizer,
            frameRate: config.frameRate,
            encoderFrameRate: encoderFrameRate,
            sampleRate: config.sampleRate,
            channels: config.channels,
            encoderTransformer: encoderTransformer,
            decoderTransformer: decoderTransformer,
            dimension: config.transformer.dModel
        )
    }
}
