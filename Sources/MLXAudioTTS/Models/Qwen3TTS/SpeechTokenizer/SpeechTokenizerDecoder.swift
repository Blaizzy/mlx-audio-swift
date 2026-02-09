//
//  SpeechTokenizerDecoder.swift
//  MLXAudio
//
//  Top-level Speech Tokenizer Decoder that assembles all components.
//  Converts 16-codebook discrete tokens into continuous audio waveforms at 24kHz.
//
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - UpsampleBlock

/// Upsample block combining CausalTransposeConv1d and ConvNeXtBlock.
///
/// PyTorch structure - self.upsample[i] is a list:
/// - [0]: CausalTransposeConv1d
/// - [1]: ConvNeXtBlock
public class UpsampleBlock: Module, UnaryLayer {
    @ModuleInfo(key: "0") var transposeConv: CausalTransposeConv1d
    @ModuleInfo(key: "1") var convNeXt: ConvNeXtBlock

    public init(channels: Int, stride: Int) {
        self._transposeConv.wrappedValue = CausalTransposeConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: stride,  // kernel = stride for this upsample
            stride: stride
        )
        self._convNeXt.wrappedValue = ConvNeXtBlock(dim: channels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = transposeConv(x)
        out = convNeXt(out)
        return out
    }
}

// MARK: - Qwen3TTSSpeechTokenizerDecoder

/// Full decoder for speech tokenizer.
///
/// Converts discrete codes [batch, 16, time] to audio [batch, 1, samples].
///
/// Architecture:
/// 1. SplitResidualVectorQuantizer: codes -> quantized [B, 512, T]
/// 2. pre_conv (CausalConv1d): 512 -> 1024
/// 3. DecoderTransformer: 8 layers with input/output projections
/// 4. Upsample blocks: 2x (stride=2) -> 4x total
/// 5. Decoder blocks: initial conv + 4 DecoderBlocks + output snake + output conv
/// 6. Final clip(-1, 1)
///
/// Total upsampling: 4x (upsample) * 480x (decoder) = 1920x
public class Qwen3TTSSpeechTokenizerDecoder: Module {
    public let config: Qwen3TTSTokenizerDecoderConfig
    public let totalUpsample: Int

    // Quantizer
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    // Pre-conv: codebook_dim (512) -> latent_dim (1024)
    @ModuleInfo(key: "pre_conv") var preConv: CausalConv1d

    // Transformer
    @ModuleInfo(key: "pre_transformer") var preTransformer: DecoderTransformer

    // Upsample blocks - each is [CausalTransposeConv1d, ConvNeXtBlock]
    // PyTorch key: decoder.upsample.{idx}.{0|1}.{attr}
    @ModuleInfo(key: "upsample") var upsample: [UpsampleBlock]

    // Main decoder - matches PyTorch's self.decoder ModuleList
    // [0]: DecoderInitialConv
    // [1-4]: 4 DecoderBlocks
    // [5]: DecoderOutputSnake
    // [6]: DecoderOutputConv
    // PyTorch key: decoder.decoder.{idx}.{attr}
    @ModuleInfo(key: "decoder") var decoder: [Module]

    public init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config

        // Calculate total upsampling factor
        let allRatios = config.upsampleRates + config.upsamplingRatios
        self.totalUpsample = allRatios.reduce(1, *)

        // Initialize quantizer
        // dimension = codebook_dim / 2 = 256
        // input_dimension = output_dimension = codebook_dim = 512
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            nQ: config.numQuantizers,
            nQSemantic: config.numSemanticQuantizers,
            dimension: config.codebookDim / 2,
            inputDimension: config.codebookDim,
            outputDimension: config.codebookDim,
            bins: config.codebookSize
        )

        // Pre-conv: codebook_dim -> latent_dim, kernel=3
        self._preConv.wrappedValue = CausalConv1d(
            inChannels: config.codebookDim,
            outChannels: config.latentDim,
            kernelSize: 3
        )

        // Transformer
        self._preTransformer.wrappedValue = DecoderTransformer(config: config)

        // Upsample blocks (upsampling_ratios = [2, 2] -> 4x total)
        var upsampleBlocks: [UpsampleBlock] = []
        for factor in config.upsamplingRatios {
            upsampleBlocks.append(UpsampleBlock(
                channels: config.latentDim,
                stride: factor
            ))
        }
        self._upsample.wrappedValue = upsampleBlocks

        // Main decoder
        // Output dim = decoder_dim / 2^(num_blocks) = 1536 / 16 = 96
        let outputDim = config.decoderDim / (1 << config.upsampleRates.count)

        var decoderLayers: [Module] = []

        // [0]: Initial conv (latent_dim -> decoder_dim)
        decoderLayers.append(DecoderInitialConv(
            latentDim: config.latentDim,
            decoderDim: config.decoderDim,
            kernelSize: 7
        ))

        // [1-4]: 4 DecoderBlocks
        for i in 0..<config.upsampleRates.count {
            decoderLayers.append(DecoderBlock(config: config, layerIdx: i))
        }

        // [5]: Output snake activation
        decoderLayers.append(DecoderOutputSnake(channels: outputDim))

        // [6]: Output conv (outputDim -> 1)
        decoderLayers.append(DecoderOutputConv(channels: outputDim, kernelSize: 7))

        self._decoder.wrappedValue = decoderLayers
    }

    /// Decode speech codes to audio waveform.
    ///
    /// - Parameter codes: Discrete codes [batch, num_quantizers, time]
    /// - Returns: Audio waveform [batch, 1, samples] clipped to [-1, 1]
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        // Validate input shape
        guard codes.shape[1] == config.numQuantizers else {
            fatalError("Expected \(config.numQuantizers) layers of codes, got \(codes.shape[1])")
        }

        // Build computation graph in two phases to avoid excessively large lazy
        // graphs that cause numerical issues on iPhone Metal GPU.
        // Force float32 to avoid precision loss (float16 corrupts temporal data).

        // ── Phase 1: quantizer → pre_conv → transformer ──
        // Dequantize: [batch, 16, time] -> [batch, codebook_dim, time]
        var hidden = quantizer.decode(codes).asType(.float32)

        // Pre-conv: [batch, 512, time] -> [batch, 1024, time]
        hidden = preConv(hidden)

        // Transpose for transformer: [batch, 1024, time] -> [batch, time, 1024]
        hidden = hidden.transposed(0, 2, 1)

        // Transformer (no caching needed for decoder-only inference)
        hidden = preTransformer(hidden)

        // Back to conv format: [batch, time, 1024] -> [batch, 1024, time]
        hidden = hidden.transposed(0, 2, 1)

        // Materialize phase 1 to bound graph size on iPhone Metal GPU.
        eval(hidden)

        // ── Phase 2: upsample → decoder blocks → output (lazy) ──

        // Upsampling: [batch, 1024, time] -> [batch, 1024, time*4]
        for upsampleBlock in upsample {
            hidden = upsampleBlock(hidden)
        }

        // Main decoder: [batch, 1024, time*4] -> [batch, 1, samples]
        var wav = hidden
        for decoderLayer in decoder {
            if let initialConv = decoderLayer as? DecoderInitialConv {
                wav = initialConv(wav)
            } else if let block = decoderLayer as? DecoderBlock {
                wav = block(wav)
            } else if let snake = decoderLayer as? DecoderOutputSnake {
                wav = snake(wav)
            } else if let outputConv = decoderLayer as? DecoderOutputConv {
                wav = outputConv(wav)
            }
        }

        // Clip to valid audio range
        return clip(wav, min: -1.0, max: 1.0)
    }

    /// Decode in chunks to handle long sequences.
    ///
    /// - Parameters:
    ///   - codes: Discrete codes [batch, num_quantizers, time]
    ///   - chunkSize: Number of code frames per chunk (default 300)
    ///   - leftContextSize: Left context overlap for smooth transitions (default 25)
    /// - Returns: Audio waveform [batch, 1, samples]
    public func chunkedDecode(
        _ codes: MLXArray,
        chunkSize: Int = 300,
        leftContextSize: Int = 25
    ) -> MLXArray {
        var wavs: [MLXArray] = []
        var startIndex = 0

        while startIndex < codes.shape[2] {
            let endIndex = min(startIndex + chunkSize, codes.shape[2])
            let contextSize = startIndex == 0 ? 0 : leftContextSize

            // Extract chunk with context
            let chunkStart = max(0, startIndex - contextSize)
            let chunkCodes = codes[0..., 0..., chunkStart..<endIndex]

            // Decode chunk
            var wav = self.callAsFunction(chunkCodes)

            // Remove context samples from output
            if contextSize > 0 {
                let contextSamples = contextSize * totalUpsample
                wav = wav[0..., 0..., contextSamples...]
            }

            // Evaluate each chunk and free GPU memory (matches PR#23 pattern)
            eval(wav)
            wavs.append(wav)
            Memory.clearCache()
            startIndex = endIndex
        }

        // Concatenate all chunks
        return concatenated(wavs, axis: 2)
    }
}
