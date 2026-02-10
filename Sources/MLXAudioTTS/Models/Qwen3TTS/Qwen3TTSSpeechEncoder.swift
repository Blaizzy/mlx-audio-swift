// Port of encoder side from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
// Speech tokenizer encoder: audio -> SeanetEncoder -> ProjectedTransformer -> ConvDownsample1d -> SplitRVQ -> codes

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXAudioCodecs
import Foundation

// Type alias to disambiguate the Mimi codec SplitResidualVectorQuantizer (which has encode())
// from the local decoder-only SplitResidualVectorQuantizer in Qwen3TTSSpeechDecoder.swift.
typealias MimiSplitRVQ = MLXAudioCodecs.SplitResidualVectorQuantizer

// MARK: - Speech Tokenizer Encoder

/// Encodes raw audio waveforms into discrete codec tokens [batch, num_quantizers, time].
/// Required for ICL (in-context learning) voice cloning.
///
/// Architecture:
///   audio [batch, 1, samples]
///     -> SeanetEncoder (conv downsampling chain)
///     -> ProjectedTransformer (causal attention with RoPE, cache)
///     -> ConvDownsample1d (stride = encoder_frame_rate / frame_rate)
///     -> SplitResidualVectorQuantizer.encode() (nearest-neighbor codebook lookup)
///     -> codes [batch, valid_num_quantizers, time]
final class Qwen3TTSSpeechTokenizerEncoder: Module {
    let config: Qwen3TTSTokenizerEncoderConfig
    let validNumQuantizers: Int

    @ModuleInfo var encoder: SeanetEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo var downsample: ConvDownsample1d
    @ModuleInfo var quantizer: MimiSplitRVQ

    init(config: Qwen3TTSTokenizerEncoderConfig) {
        self.config = config
        self.validNumQuantizers = 16  // Only first 16 quantizers used for ICL

        // Build SeanetConfig from encoder config
        let seanetCfg = SeanetConfig(
            dimension: config.hiddenSize,
            channels: config.audioChannels,
            causal: config.useCausalConv,
            nfilters: config.numFilters,
            nresidualLayers: config.numResidualLayers,
            ratios: config.upsamplingRatios,
            ksize: config.kernelSize,
            residualKsize: config.residualKernelSize,
            lastKsize: config.lastKernelSize,
            dilationBase: config.dilationGrowthRate,
            padMode: .constant,
            trueSkip: !config.useConvShortcut,
            compress: config.compress
        )
        self._encoder.wrappedValue = SeanetEncoder(cfg: seanetCfg)

        // Build TransformerConfig for the projected transformer
        let kvRepeat = config.numAttentionHeads / config.numKeyValueHeads
        let transformerCfg = TransformerConfig(
            dModel: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            causal: config.useCausalConv,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: config.layerScaleInitialScale,
            positionalEmbedding: "rope",
            useConvBlock: false,
            crossAttention: false,
            convKernelSize: 3,
            useConvBias: true,
            gating: false,
            norm: "layer_norm",
            context: config.slidingWindow,
            maxPeriod: Int(config.ropeTheta),
            maxSeqLen: config.maxPositionEmbeddings,
            kvRepeat: kvRepeat,
            dimFeedforward: config.intermediateSize,
            convLayout: true
        )
        self._encoderTransformer.wrappedValue = ProjectedTransformer(
            cfg: transformerCfg,
            inputDim: config.hiddenSize,
            outputDims: [config.hiddenSize]
        )

        // ConvDownsample1d: stride = encoder_frame_rate / frame_rate
        let encoderFrameRate = Float(config.samplingRate) / Float(config.upsamplingRatios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / config.frameRate)
        self._downsample.wrappedValue = ConvDownsample1d(
            stride: downsampleStride,
            dim: config.hiddenSize,
            causal: config.useCausalConv
        )

        // SplitResidualVectorQuantizer (Mimi version with encode() support)
        self._quantizer.wrappedValue = MimiSplitRVQ(
            dim: config.codebookDim,
            inputDim: config.hiddenSize,
            outputDim: config.hiddenSize,
            nq: config.numQuantizers,
            bins: config.codebookSize
        )
    }

    /// Encode audio waveform to discrete codec tokens.
    ///
    /// - Parameter audio: Input audio waveform of shape `[batch, 1, samples]`
    /// - Returns: Codec tokens of shape `[batch, valid_num_quantizers, time]`
    func encode(_ audio: MLXArray) -> MLXArray {
        // Reset streaming state in the SeanetEncoder conv layers
        encoder.resetState()

        // Create fresh KV caches for the transformer (KVCacheSimple has no reset(),
        // so we allocate new ones each call to ensure offset == 0)
        let cache = encoderTransformer.makeCache()

        // SeanetEncoder: audio [B, 1, samples] -> features [B, hidden_size, time]
        var xs = encoder(audio)

        // ProjectedTransformer: features [B, hidden_size, time] -> features [B, hidden_size, time]
        // causal=true in the TransformerConfig causes the Attention layer to auto-generate
        // a causal mask via createAttentionMask(). convLayout=true handles NCL<->NLC conversion.
        let transformerOutputs = encoderTransformer(xs, cache: cache)
        xs = transformerOutputs[0]

        // ConvDownsample1d: reduce temporal resolution
        xs = downsample(xs)

        // SplitResidualVectorQuantizer: continuous features -> discrete codes
        var codes = quantizer.encode(xs)

        // Return only the first valid_num_quantizers
        codes = codes[0..., ..<validNumQuantizers]
        return codes
    }
}
