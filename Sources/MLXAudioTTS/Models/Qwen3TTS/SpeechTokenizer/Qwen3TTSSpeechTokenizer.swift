//
//  Qwen3TTSSpeechTokenizer.swift
//  MLXAudio
//
//  Top-level Speech Tokenizer with weight loading and sanitization.
//  Supports both decoding (codes->audio) and encoding (audio->codes for ICL voice cloning).
//
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN
import RegexBuilder

// MARK: - Qwen3TTSSpeechTokenizer

/// Full speech tokenizer model.
///
/// This is the top-level class that wraps the decoder and optional encoder,
/// providing weight loading/sanitization functionality.
public class Qwen3TTSSpeechTokenizer: Module {
    public let config: Qwen3TTSTokenizerConfig
    public let encoderValidNumQuantizers: Int
    public let inputSampleRate: Int
    public let outputSampleRate: Int
    public let decodeUpsampleRate: Int
    public let encodeDownsampleRate: Int

    @ModuleInfo(key: "decoder") var decoder: Qwen3TTSSpeechTokenizerDecoder
    @ModuleInfo(key: "encoder_model") var encoderModel: Qwen3TTSSpeechTokenizerEncoder?

    /// Whether the encoder is available for audio-to-codes encoding (ICL voice cloning).
    public var hasEncoder: Bool { encoderModel != nil }

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

        // Create encoder if encoder config is available
        if let encoderConfig = config.encoderConfig {
            self._encoderModel.wrappedValue = Qwen3TTSSpeechTokenizerEncoder(config: encoderConfig)
        } else {
            self._encoderModel.wrappedValue = nil
        }
    }

    // MARK: - Encoding

    /// Encode audio waveform to discrete codes.
    ///
    /// - Parameter audio: Audio waveform [batch, 1, samples] at 24kHz
    /// - Returns: Codes [batch, num_quantizers, time]
    /// - Throws: Error if encoder is not available
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        guard let encoder = encoderModel else {
            throw Qwen3TTSError.invalidConfig("Encoder not available for this speech tokenizer")
        }
        return encoder.encode(audio)
    }

    // MARK: - Decoding

    /// Decode audio codes to waveform.
    ///
    /// - Parameter audioCodes: Audio codes [batch, time, num_quantizers]
    /// - Returns: Tuple of (audio waveform [batch, samples], audio lengths [batch])
    public func decode(_ audioCodes: MLXArray) -> (audio: MLXArray, lengths: MLXArray) {
        // Transpose to [batch, num_quantizers, time]
        let codes = audioCodes.transposed(0, 2, 1)
        // Use small chunks (15 steps) to avoid large lazy graphs that cause
        // position-dependent silence on iPhone Metal GPU.
        var wav = decoder.chunkedDecode(codes, chunkSize: 15, leftContextSize: 5)
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
    /// 1. Encoder weights: SeanetEncoder, ProjectedTransformer, ConvDownsample, SplitRVQ
    /// 2. Decoder weights: Conv1d transposition, ConvTranspose1d, codebook computation
    /// 3. Computing codebook embeddings from cluster_usage and embedding_sum
    ///
    /// - Parameter weights: Raw weights dictionary from safetensors
    /// - Returns: Sanitized weights ready for module.update()
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var codebookData: [String: [String: MLXArray]] = [:]

        // Encoder-specific collections
        var encoderTransformerQKV: [Int: [String: MLXArray]] = [:]
        var encoderCodebookData: [String: [String: MLXArray]] = [:]

        // SeanetEncoder layer mapping:
        // N=0: init_conv, N=3,6,9,12: downsample, N=14: final_conv
        // N=1,4,7,10: residual blocks
        let seanetConvMap: [Int: String] = [
            0: "encoder_model.encoder.init_conv1d",
            3: "encoder_model.encoder.layers.0.downsample",
            6: "encoder_model.encoder.layers.1.downsample",
            9: "encoder_model.encoder.layers.2.downsample",
            12: "encoder_model.encoder.layers.3.downsample",
            14: "encoder_model.encoder.final_conv1d",
        ]
        let seanetResidualMap: [Int: Int] = [1: 0, 4: 1, 7: 2, 10: 3]
        let seanetBlockMap: [Int: Int] = [1: 0, 3: 1]

        for (key, var value) in weights {
            if key.hasPrefix("encoder.") {
                // --- Handle encoder weights ---

                // SeanetEncoder convolutions
                if key.hasPrefix("encoder.encoder.layers.") {
                    let parts = key.split(separator: ".").map(String.init)
                    guard parts.count > 3, let n = Int(parts[3]) else { continue }

                    if key.contains("block") {
                        // Residual block: encoder.encoder.layers.{N}.block.{B}.conv.{w/b}
                        guard let layerIdx = seanetResidualMap[n],
                              parts.count > 5,
                              let blockIdx = Int(parts[5]),
                              let convIdx = seanetBlockMap[blockIdx] else { continue }
                        let basePath = "encoder_model.encoder.layers.\(layerIdx).residuals.0.block.\(convIdx)"
                        let suffix = parts[6...].joined(separator: ".")
                        let newKey = "\(basePath).conv.\(suffix)"
                        if suffix.contains("weight") && value.ndim == 3 {
                            value = value.transposed(0, 2, 1)  // PyTorch [out,in,k] -> MLX [out,k,in]
                        }
                        sanitized[newKey] = value
                    } else {
                        // Direct conv: encoder.encoder.layers.{N}.conv.{w/b}
                        guard let basePath = seanetConvMap[n] else { continue }
                        let suffix = parts[4...].joined(separator: ".")
                        let newKey = "\(basePath).conv.\(suffix)"
                        if suffix.contains("weight") && value.ndim == 3 {
                            value = value.transposed(0, 2, 1)
                        }
                        sanitized[newKey] = value
                    }
                }
                // Encoder transformer layers
                else if key.hasPrefix("encoder.encoder_transformer.layers.") {
                    let parts = key.split(separator: ".").map(String.init)
                    guard parts.count > 3, let layerIdx = Int(parts[3]) else { continue }
                    let rest = parts[4...].joined(separator: ".")

                    if rest.contains("self_attn.q_proj.weight") {
                        if encoderTransformerQKV[layerIdx] == nil { encoderTransformerQKV[layerIdx] = [:] }
                        encoderTransformerQKV[layerIdx]!["q"] = value
                    } else if rest.contains("self_attn.k_proj.weight") {
                        if encoderTransformerQKV[layerIdx] == nil { encoderTransformerQKV[layerIdx] = [:] }
                        encoderTransformerQKV[layerIdx]!["k"] = value
                    } else if rest.contains("self_attn.v_proj.weight") {
                        if encoderTransformerQKV[layerIdx] == nil { encoderTransformerQKV[layerIdx] = [:] }
                        encoderTransformerQKV[layerIdx]!["v"] = value
                    } else if rest.contains("self_attn.o_proj.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).self_attn.out_proj.weight"] = value
                    } else if rest.contains("mlp.fc1.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).gating.linear1.weight"] = value
                    } else if rest.contains("mlp.fc2.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).gating.linear2.weight"] = value
                    } else if rest.contains("input_layernorm.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm1.weight"] = value
                    } else if rest.contains("input_layernorm.bias") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm1.bias"] = value
                    } else if rest.contains("post_attention_layernorm.weight") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm2.weight"] = value
                    } else if rest.contains("post_attention_layernorm.bias") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm2.bias"] = value
                    } else if rest.contains("self_attn_layer_scale.scale") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).layer_scale_1.scale"] = value
                    } else if rest.contains("mlp_layer_scale.scale") {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).layer_scale_2.scale"] = value
                    }
                }
                // Encoder downsample conv
                // Raw key: encoder.downsample.conv.weight
                // Module path: encoder_model.downsample.conv.conv.conv.weight
                // (MimiConvDownsample1d -> MimiStreamableConv1d -> MimiNormConv1d -> MimiConv1d)
                else if key.hasPrefix("encoder.downsample.") {
                    let suffix = String(key.dropFirst("encoder.downsample.".count))
                    let newKey = "encoder_model.downsample.conv.conv.\(suffix)"
                    if suffix.contains("weight") && value.ndim == 3 {
                        value = value.transposed(0, 2, 1)
                    }
                    sanitized[newKey] = value
                }
                // Encoder quantizer
                else if key.hasPrefix("encoder.quantizer.") {
                    let rest = String(key.dropFirst("encoder.quantizer.".count))

                    // Codebook data (cluster_usage / embed_sum)
                    if rest.contains(".codebook.cluster_usage") || rest.contains(".codebook.embed_sum") {
                        let base: String
                        if let range = rest.range(of: ".codebook.") {
                            base = String(rest[..<range.lowerBound])
                        } else {
                            continue
                        }
                        if encoderCodebookData[base] == nil { encoderCodebookData[base] = [:] }
                        if rest.contains("cluster_usage") {
                            encoderCodebookData[base]!["cluster_usage"] = value
                        } else if rest.contains("embed_sum") {
                            encoderCodebookData[base]!["embedding_sum"] = value
                        }
                    } else if rest.contains(".codebook.initialized") {
                        // Skip
                    }
                    // Input/output projections
                    else if rest.contains("input_proj.weight") || rest.contains("output_proj.weight") {
                        let projType = rest.contains("input_proj") ? "input_proj" : "output_proj"
                        let rvqKey: String
                        if rest.contains("semantic_residual_vector_quantizer") {
                            rvqKey = "encoder_model.quantizer.rvq_first.\(projType).weight"
                        } else {
                            rvqKey = "encoder_model.quantizer.rvq_rest.\(projType).weight"
                        }
                        if value.ndim == 3 {
                            value = value.transposed(0, 2, 1)
                        }
                        sanitized[rvqKey] = value
                    }
                }
            } else {
                // --- Handle decoder weights (existing logic) ---

                // Collect codebook cluster_usage and embedding_sum for later processing
                if key.contains("_codebook.cluster_usage") || key.contains("_codebook.embedding_sum") {
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

                // Remap depthwise conv keys
                var newKey = key
                if key.contains("dwconv.conv.") {
                    newKey = key.replacingOccurrences(of: "dwconv.conv.", with: "dwconv.depthwiseConv.")
                }
                var newValue = value

                let isMLXFormat = checkArrayShapeQwen3(newValue)

                let isTransposeConv = (key.contains("upsample") && key.contains(".0.conv.weight")) ||
                                      (key.contains("decoder.decoder") && key.contains("block.1.conv.weight"))

                if value.ndim == 3 {
                    if isTransposeConv {
                        if !isMLXFormat {
                            newValue = value.transposed(1, 2, 0)
                        }
                    } else if key.contains("conv.weight") || key.contains("_proj.weight") {
                        let isDepthwiseConv = key.contains("dwconv")
                        if !isMLXFormat || isDepthwiseConv {
                            newValue = value.transposed(0, 2, 1)
                        }
                    }
                }

                sanitized[newKey] = newValue
            }
        }

        // Process encoder transformer q/k/v into combined in_proj weights
        for (layerIdx, qkv) in encoderTransformerQKV {
            if let q = qkv["q"], let k = qkv["k"], let v = qkv["v"] {
                let inProjWeight = concatenated([q, k, v], axis: 0)
                sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).self_attn.in_proj.weight"] = inProjWeight
            }
        }

        // Process encoder codebook data
        let eps: Float = 1e-5
        for (basePath, data) in encoderCodebookData {
            if let clusterUsage = data["cluster_usage"], let embeddingSum = data["embedding_sum"] {
                if basePath.contains("semantic_residual_vector_quantizer") {
                    // Extract layer index
                    if let layerIdx = extractLayerIndex(from: basePath) {
                        let prefix = "encoder_model.quantizer.rvq_first.vq.layers.\(layerIdx).codebook"
                        sanitized["\(prefix).embeddingSum"] = embeddingSum
                        sanitized["\(prefix).clusterUsage"] = clusterUsage
                    }
                } else if basePath.contains("acoustic_residual_vector_quantizer") {
                    if let layerIdx = extractLayerIndex(from: basePath) {
                        let prefix = "encoder_model.quantizer.rvq_rest.vq.layers.\(layerIdx).codebook"
                        sanitized["\(prefix).embeddingSum"] = embeddingSum
                        sanitized["\(prefix).clusterUsage"] = clusterUsage
                    }
                }
            }
        }

        // Compute decoder embeddings from cluster_usage and embedding_sum
        for (basePath, data) in codebookData {
            if let clusterUsage = data["cluster_usage"],
               let embeddingSum = data["embedding_sum"] {
                let clampedUsage = clip(clusterUsage.expandedDimensions(axis: 1), min: eps, max: Float.infinity)
                let embedding = embeddingSum / clampedUsage
                let newKey = "\(basePath).codebook.embed.weight"
                sanitized[newKey] = embedding
            }
        }

        return sanitized
    }

    /// Extract a layer index from a path like "...layers.3..."
    private static func extractLayerIndex(from path: String) -> Int? {
        // Find "layers.{N}" pattern
        let parts = path.split(separator: ".").map(String.init)
        for (i, part) in parts.enumerated() {
            if part == "layers" && i + 1 < parts.count, let idx = Int(parts[i + 1]) {
                return idx
            }
        }
        return nil
    }

    /// Check if Conv1d weights are already in MLX format.
    private static func checkArrayShapeQwen3(_ arr: MLXArray) -> Bool {
        guard arr.ndim == 3 else { return false }
        let shape = arr.shape
        return shape[1] < shape[0] && shape[1] < shape[2]
    }

    // MARK: - Weight Loading

    /// Load weights from a safetensors file URL.
    ///
    /// - Parameters:
    ///   - url: URL to the safetensors file
    ///   - strict: Whether to verify all weights are used (currently ignored)
    /// - Throws: LoadSaveError if file cannot be loaded
    public func loadWeights(from url: URL, strict: Bool = false) throws {
        let rawWeights = try loadArrays(url: url)
        let sanitizedWeights = Self.sanitize(rawWeights)

        // Use direct path-based assignment instead of nested structure
        self.setWeightsByPath(sanitizedWeights)
    }
}
