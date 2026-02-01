//
//  Qwen3TTSSpeechTokenizer.swift
//  MLXAudio
//
//  Top-level Speech Tokenizer with weight loading and sanitization.
//  Converts 16-codebook discrete tokens into continuous audio waveforms at 24kHz.
//
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Qwen3TTSSpeechTokenizer

/// Full speech tokenizer model.
///
/// This is the top-level class that wraps the decoder and provides
/// weight loading/sanitization functionality.
public class Qwen3TTSSpeechTokenizer: Module {
    public let config: Qwen3TTSTokenizerConfig
    public let encoderValidNumQuantizers: Int
    public let inputSampleRate: Int
    public let outputSampleRate: Int
    public let decodeUpsampleRate: Int
    public let encodeDownsampleRate: Int

    @ModuleInfo(key: "decoder") var decoder: Qwen3TTSSpeechTokenizerDecoder

    public init(config: Qwen3TTSTokenizerConfig) {
        self.config = config
        self.encoderValidNumQuantizers = config.encoderValidNumQuantizers
        self.inputSampleRate = config.inputSampleRate
        self.outputSampleRate = config.outputSampleRate
        self.decodeUpsampleRate = config.decodeUpsampleRate
        self.encodeDownsampleRate = config.encodeDownsampleRate

        // Use decoder config (default if not provided in config)
        let decoderConfig = config.decoderConfig ?? Qwen3TTSTokenizerDecoderConfig()
        self._decoder.wrappedValue = Qwen3TTSSpeechTokenizerDecoder(config: decoderConfig)
    }

    /// Decode audio codes to waveform.
    ///
    /// - Parameter audioCodes: Audio codes [batch, time, num_quantizers]
    /// - Returns: Tuple of (audio waveform [batch, samples], audio lengths [batch])
    public func decode(_ audioCodes: MLXArray) -> (audio: MLXArray, lengths: MLXArray) {
        // Transpose to [batch, num_quantizers, time]
        let codes = audioCodes.transposed(0, 2, 1)
        var wav = decoder.chunkedDecode(codes)
        wav = wav.squeezed(axis: 1)  // Remove channel dim -> [batch, samples]

        // Calculate audio lengths based on valid codes
        // Valid codes are where first quantizer > 0
        let validMask = audioCodes[0..., 0..., 0] .> 0  // [batch, time]
        let audioLengths = sum(validMask.asType(.int32), axis: 1) * Int32(decodeUpsampleRate)

        return (wav, audioLengths)
    }

    /// Decode with explicit chunking parameters.
    ///
    /// - Parameters:
    ///   - audioCodes: Audio codes [batch, time, num_quantizers]
    ///   - chunkSize: Number of code frames per chunk
    ///   - leftContextSize: Left context overlap for smooth transitions
    /// - Returns: Tuple of (audio waveform [batch, samples], audio lengths [batch])
    public func decode(
        _ audioCodes: MLXArray,
        chunkSize: Int = 300,
        leftContextSize: Int = 25
    ) -> (audio: MLXArray, lengths: MLXArray) {
        let codes = audioCodes.transposed(0, 2, 1)
        var wav = decoder.chunkedDecode(codes, chunkSize: chunkSize, leftContextSize: leftContextSize)
        wav = wav.squeezed(axis: 1)

        let validMask = audioCodes[0..., 0..., 0] .> 0
        let audioLengths = sum(validMask.asType(.int32), axis: 1) * Int32(decodeUpsampleRate)

        return (wav, audioLengths)
    }

    // MARK: - Weight Sanitization

    /// Sanitize weights from PyTorch/safetensors format to MLX format.
    ///
    /// This function handles:
    /// 1. Skipping encoder weights (we only implement decoder)
    /// 2. Transposing Conv1d weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
    /// 3. Transposing ConvTranspose1d weights: PyTorch [in, out, kernel] -> MLX [out, kernel, in]
    /// 4. Computing codebook embeddings from cluster_usage and embedding_sum
    ///
    /// - Parameter weights: Raw weights dictionary from safetensors
    /// - Returns: Sanitized weights ready for module.update()
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var codebookData: [String: [String: MLXArray]] = [:]

        for (key, value) in weights {
            // Skip encoder weights (we only implement decoder)
            if key.hasPrefix("encoder.") {
                continue
            }

            // Collect codebook cluster_usage and embedding_sum for later processing
            // PyTorch uses _codebook (with underscore), we need to compute embeddings
            if key.contains("_codebook.cluster_usage") || key.contains("_codebook.embedding_sum") {
                // Extract base path (everything before _codebook)
                if let range = key.range(of: "._codebook.") {
                    let basePath = String(key[..<range.lowerBound])
                    if codebookData[basePath] == nil {
                        codebookData[basePath] = [:]
                    }
                    if key.contains("cluster_usage") {
                        codebookData[basePath]!["cluster_usage"] = value
                    } else {
                        codebookData[basePath]!["embedding_sum"] = value
                    }
                }
                continue
            }

            // Remap depthwise conv keys: dwconv.conv -> dwconv.depthwiseConv
            // In Python, CausalConv1d uses "conv" for both regular and depthwise convs
            // In Swift, we use separate keys: "conv" for Conv1d, "depthwiseConv" for DepthwiseConvWeight
            var newKey = key
            if key.contains("dwconv.conv.") {
                newKey = key.replacingOccurrences(of: "dwconv.conv.", with: "dwconv.depthwiseConv.")
            }
            var newValue = value

            // Check if already in MLX format (heuristic: kernel dim is middle)
            let isMLXFormat = checkArrayShapeQwen3(newValue)

            // Identify ConvTranspose1d weights (for upsampling layers)
            // decoder.upsample.X.0.conv and decoder.decoder.X.block.1.conv are transpose convs
            let isTransposeConv = (key.contains("upsample") && key.contains(".0.conv.weight")) ||
                                  (key.contains("decoder.decoder") && key.contains("block.1.conv.weight"))

            if value.ndim == 3 {
                if isTransposeConv {
                    // ConvTranspose1d: PyTorch [in, out, kernel] -> MLX [out, kernel, in]
                    if !isMLXFormat {
                        newValue = value.transposed(1, 2, 0)
                    }
                } else if key.contains("conv.weight") || key.contains("_proj.weight") {
                    // Conv1d: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
                    // Note: For depthwise conv (dwconv), shape is [out, 1, kernel] which looks like MLX format
                    // due to middle dim=1, so we force transpose for dwconv weights
                    let isDepthwiseConv = key.contains("dwconv")
                    if !isMLXFormat || isDepthwiseConv {
                        newValue = value.transposed(0, 2, 1)
                    }
                }
            }

            sanitized[newKey] = newValue
        }

        // Compute embeddings from cluster_usage and embedding_sum
        let eps: Float = 1e-5
        for (basePath, data) in codebookData {
            if let clusterUsage = data["cluster_usage"],
               let embeddingSum = data["embedding_sum"] {
                // Compute normalized embedding: embedding_sum / cluster_usage
                // clusterUsage: [codebook_size], embeddingSum: [codebook_size, dim]
                let clampedUsage = clip(clusterUsage.expandedDimensions(axis: 1), min: eps, max: Float.infinity)
                let embedding = embeddingSum / clampedUsage

                // New key: base_path.codebook.embed.weight
                let newKey = "\(basePath).codebook.embed.weight"
                sanitized[newKey] = embedding
            }
        }

        return sanitized
    }

    /// Check if Conv1d weights are already in MLX format.
    ///
    /// MLX Conv1d expects [out_channels, kernel_size, in_channels]
    /// PyTorch Conv1d has [out_channels, in_channels, kernel_size]
    ///
    /// Heuristic: if middle dimension (kernel) is small (< 10), it's likely MLX format
    private static func checkArrayShapeQwen3(_ arr: MLXArray) -> Bool {
        guard arr.ndim == 3 else { return false }
        let shape = arr.shape
        // Kernel size is typically small (1, 3, 5, 7, etc.)
        // If middle dim is smaller than both outer dims, likely MLX format
        return shape[1] < shape[0] && shape[1] < shape[2]
    }

    // MARK: - Weight Loading

    /// Load weights from a safetensors file URL.
    ///
    /// Uses direct path-based weight assignment to handle the mismatch between
    /// Python's numeric indices (used for both arrays and module dicts) and
    /// Swift's strict array/dictionary distinction.
    ///
    /// - Parameters:
    ///   - url: URL to the safetensors file
    ///   - strict: Whether to verify all weights are used (currently ignored)
    /// - Throws: LoadSaveError if file cannot be loaded
    public func loadWeights(from url: URL, strict: Bool = false) throws {
        let rawWeights = try loadArrays(url: url)
        let sanitizedWeights = Self.sanitize(rawWeights)

        // Use direct path-based assignment instead of nested structure
        // This handles numeric keys correctly for both arrays and module dicts
        self.setWeightsByPath(sanitizedWeights)
    }

}
