//
//  SpeechTokenizerEncoder.swift
//  MLXAudio
//
//  Top-level speech tokenizer encoder that composes all Mimi encoder components.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py:889-990
//
//  Pipeline: SeanetEncoder -> ProjectedTransformer -> ConvDownsample1d -> SplitRVQ.encode()
//

import Foundation
import MLX
import MLXNN

// MARK: - Qwen3TTSSpeechTokenizerEncoder

/// Encoder for the speech tokenizer using Mimi components.
///
/// Encodes raw audio waveform [batch, 1, samples] into discrete codes [batch, 16, time].
/// Used for ICL (In-Context Learning) voice cloning to encode reference audio.
public class Qwen3TTSSpeechTokenizerEncoder: Module {
    let config: Qwen3TTSTokenizerEncoderConfig
    let validNumQuantizers: Int = 16

    @ModuleInfo(key: "encoder") var encoder: MimiSeanetEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: MimiProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: MimiConvDownsample1d
    @ModuleInfo(key: "quantizer") var quantizer: MimiSplitResidualVectorQuantizer

    var encoderCache: [MimiKVCache]

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        self.config = config

        // SeanetEncoder
        self._encoder.wrappedValue = MimiSeanetEncoder(config: config)

        // ProjectedTransformer
        self._encoderTransformer.wrappedValue = MimiProjectedTransformer(config: config)

        // ConvDownsample1d: stride = encoder_frame_rate / frame_rate
        // encoder_frame_rate = sampling_rate / prod(upsampling_ratios) = 24000 / 960 = 25
        // stride = 25 / 12.5 = 2
        let encoderFrameRate = Float(config.samplingRate) / Float(config.upsamplingRatios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / config.frameRate)

        self._downsample.wrappedValue = MimiConvDownsample1d(
            stride: downsampleStride,
            dim: config.hiddenSize,
            causal: config.useCausalConv
        )

        // SplitResidualVectorQuantizer
        self._quantizer.wrappedValue = MimiSplitResidualVectorQuantizer(
            dim: config.codebookDim,
            inputDim: config.hiddenSize,
            outputDim: config.hiddenSize,
            nq: config.numQuantizers,
            bins: config.codebookSize
        )

        // Initialize cache
        self.encoderCache = []

        // Create cache after super.init
        super.init()
        self.encoderCache = encoderTransformer.makeCache()
    }

    /// Encode audio waveform to discrete codes.
    ///
    /// - Parameter audio: Raw audio [batch, 1, samples] at 24kHz
    /// - Returns: Codes [batch, 16, time] where 16 = validNumQuantizers
    public func encode(_ audio: MLXArray) -> MLXArray {
        // Reset encoder cache
        for c in encoderCache {
            c.reset()
        }

        // 1. SeanetEncoder: [batch, 1, samples] -> [batch, hidden, time/960]
        var xs = encoder(audio)
        eval(xs)

        // 2. Create causal attention mask for the transformer
        let seqLen = xs.shape[2]  // NCL format, time is last dim
        let mask = buildCausalAttentionMask(seqLen: seqLen, dtype: xs.dtype)

        // 3. ProjectedTransformer: [batch, hidden, time] -> [batch, hidden, time]
        xs = encoderTransformer(xs, cache: encoderCache, mask: mask)[0]
        eval(xs)

        // 4. ConvDownsample: [batch, hidden, time] -> [batch, hidden, time/2]
        xs = downsample(xs)
        eval(xs)

        // 5. Quantizer encode: [batch, hidden, time] -> [batch, nq, time]
        let codes = quantizer.encode(xs)
        eval(codes)

        // 6. Truncate to first 16 quantizers
        return codes[0..., 0..<validNumQuantizers, 0...]
    }

    /// Build a causal attention mask (upper triangle = -inf).
    private func buildCausalAttentionMask(seqLen: Int, dtype: DType) -> MLXArray {
        // Create [seqLen, seqLen] matrix filled with -inf
        var mask = MLXArray.full([seqLen, seqLen], values: MLXArray(-Float.infinity), type: Float.self)
        // triu with k=1: upper triangle (above diagonal) stays -inf, rest becomes 0
        mask = triu(mask, k: 1)
        // Reshape to [1, 1, seqLen, seqLen] for broadcasting over batch and heads
        return mask.reshaped([1, 1, seqLen, seqLen]).asType(dtype)
    }
}

/// Create upper triangular matrix (above k-th diagonal).
private func triu(_ x: MLXArray, k: Int = 0) -> MLXArray {
    let n = x.shape[0]
    let m = x.shape[1]
    // Create mask where row >= col + k (below or on k-th diagonal)
    let rows = MLXArray(Array(0..<n).map { Int32($0) }).reshaped([n, 1])
    let cols = MLXArray(Array(0..<m).map { Int32($0) }).reshaped([1, m])
    let lowerMask = rows .>= (cols + MLXArray(Int32(k)))
    return MLX.where(lowerMask, MLXArray(Float(0)), x)
}
