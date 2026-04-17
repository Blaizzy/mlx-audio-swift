//
//  ChatterboxModel.swift
//  MLXAudio
//
//  Top-level Chatterbox TTS model.
//  Two-stage pipeline: T3 (text→speech tokens) + S3Gen (speech tokens→audio).
//  Supports both Regular (LLaMA backbone) and Turbo (GPT-2 backbone) variants.
//  Ported from mlx-audio Python: chatterbox/chatterbox.py
//

import AVFoundation
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN
@preconcurrency import MLXLMCommon
import Tokenizers

// MARK: - Default Voice Conditioning

/// Pre-computed voice conditioning loaded from conds.safetensors (Turbo default voice).
public struct DefaultConditioning {
    /// T3 conditioning
    public var speakerEmb: MLXArray           // (1, 256)
    public var condPromptSpeechTokens: MLXArray // (1, T)
    public var emotionAdv: MLXArray           // (1, 1, 1)

    /// S3Gen conditioning
    public var xVector: MLXArray              // (1, 192)
    public var promptToken: MLXArray          // (1, T)
    public var promptTokenLen: MLXArray       // (1,)
    public var promptFeat: MLXArray           // (1, T, 80)
    public var promptFeatLen: MLXArray        // not stored — derived from promptFeat
}

// MARK: - Chatterbox Model

/// Chatterbox TTS: two-stage speech synthesis.
///
/// Stage 1 (T3): LLaMA or GPT-2 backbone converts text tokens → speech tokens,
/// conditioned on speaker embedding + optional prompt + emotion scalar.
///
/// Stage 2 (S3Gen): Flow matching decoder (Euler ODE) + HiFi-GAN vocoder converts
/// speech tokens → mel spectrogram → waveform at 24kHz.
///
/// Two variants:
/// - **Regular** (`Chatterbox-TTS-fp16`): LLaMA 520M, 500M params, 23 languages, emotion control
/// - **Turbo** (`chatterbox-turbo-fp16`): GPT-2 Medium, 350M params, English only, faster
public final class ChatterboxModel: Module, SpeechGenerationModel, @unchecked Sendable {

    // MARK: - Configuration

    public let config: ChatterboxConfiguration

    // MARK: - Sub-models

    /// Voice encoder: extracts 256-dim speaker embedding from reference audio.
    @ModuleInfo(key: "ve") var voiceEncoder: VoiceEncoder

    /// T3: text-to-speech-token model.
    /// Either T3Model (LLaMA) or T3GPT2Model (GPT-2), stored as Module for @ModuleInfo compatibility.
    @ModuleInfo(key: "t3") var t3: Module

    /// S3Gen: speech-token-to-audio model (Conformer + flow matching + HiFi-GAN).
    @ModuleInfo(key: "s3gen") var s3gen: CausalMaskedDiffWithXvec

    // MARK: - State

    /// Text tokenizer loaded from tokenizer.json.
    public var tokenizer: Tokenizers.Tokenizer?

    /// S3TokenizerV2: converts audio → speech token IDs (loaded separately).
    public var s3Tokenizer: S3TokenizerV2?

    /// Pre-computed default voice conditioning (from conds.safetensors).
    public var defaultConditioning: DefaultConditioning?

    /// Model directory (for loading auxiliary files).
    private var modelDir: URL?

    /// Configurable CFG weight for T3 inference (0.0-1.0). Only used by Regular model.
    /// Set from the host app to override the default 0.5.
    public var cfgWeightOverride: Float?

    /// Configurable emotion exaggeration (0.0-1.0). Only used by Regular model.
    /// Set from the host app to override the default 0.5.
    public var emotionAdvOverride: Float?

    // MARK: - Protocol conformance

    public var sampleRate: Int { config.s3genSr }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 0.8)
    }

    // MARK: - Special tokens

    private var sotToken: Int { config.t3Config.startTextToken }
    private var eotToken: Int { config.t3Config.stopTextToken }
    private var sosToken: Int { config.t3Config.startSpeechToken }
    private var eosToken: Int { config.t3Config.stopSpeechToken }
    private var speechVocabSize: Int { config.t3Config.speechTokensDictSize }

    // MARK: - Initialization

    public init(_ config: ChatterboxConfiguration = .default) {
        self.config = config

        self._voiceEncoder.wrappedValue = VoiceEncoder()

        // Create the appropriate T3 model based on config
        if config.isTurbo {
            let gpt2Config = config.gpt2Config ?? .medium
            self._t3.wrappedValue = T3GPT2Model(config.t3Config, gpt2Config: gpt2Config)
        } else {
            self._t3.wrappedValue = T3Model(config.t3Config)
        }

        self._s3gen.wrappedValue = CausalMaskedDiffWithXvec(
            decoderInChannels: config.decoderInChannels,
            meanflow: config.meanflow
        )
    }

    // MARK: - Weight Sanitization

    /// Route weights by prefix to the correct sub-model.
    ///
    /// Handles differences between Regular and Turbo weight key formats:
    /// - Regular S3Gen: `s3gen.flow.{decoder,encoder,...}` → strip `flow.` prefix
    /// - Both models: `s3gen.speaker_encoder.*` → dropped (loaded separately or unused)
    /// - Both models: `s3gen.tokenizer.*` → dropped (Turbo bakes it in, we don't use it)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var veWeights = [String: MLXArray]()
        var t3Weights = [String: MLXArray]()
        var s3genWeights = [String: MLXArray]()
        var result = [String: MLXArray]()

        for (key, value) in weights {
            if key.hasPrefix("ve.") {
                let subKey = String(key.dropFirst("ve.".count))
                veWeights[subKey] = value
            } else if key.hasPrefix("t3.") {
                let subKey = String(key.dropFirst("t3.".count))
                t3Weights[subKey] = value
            } else if key.hasPrefix("s3gen.") {
                var subKey = String(key.dropFirst("s3gen.".count))

                // Drop tokenizer weights — S3TokenizerV2 loaded separately from its own repo
                if subKey.hasPrefix("tokenizer.") { continue }

                // Regular model: strip `flow.` prefix so keys match module structure
                // e.g. s3gen.flow.decoder.* → s3gen.decoder.*
                if subKey.hasPrefix("flow.") {
                    subKey = String(subKey.dropFirst("flow.".count))
                }

                s3genWeights[subKey] = value
            }
            // Drop campplus.* (neither model has top-level campplus weights)
            // Drop any other unknown prefixes
        }

        // Sub-model sanitization
        let sanitizedVE = voiceEncoder.sanitize(weights: veWeights)
        let sanitizedT3: [String: MLXArray]
        if let t3llama = t3 as? T3Model {
            sanitizedT3 = t3llama.sanitize(weights: t3Weights)
        } else if let t3gpt2 = t3 as? T3GPT2Model {
            sanitizedT3 = t3gpt2.sanitize(weights: t3Weights)
        } else {
            sanitizedT3 = t3Weights
        }

        // Sanitize speaker_encoder weights via CAMPPlus
        var speakerEncoderWeights = [String: MLXArray]()
        var otherS3GenWeights = [String: MLXArray]()
        for (key, value) in s3genWeights {
            if key.hasPrefix("speaker_encoder.") {
                let subKey = String(key.dropFirst("speaker_encoder.".count))
                speakerEncoderWeights[subKey] = value
            } else {
                // Remap FeedForward net.0/net.1 keys to non-numeric names.
                // Python uses nn.ModuleList([GEGLU, Linear]) stored as net.0, net.1.
                // We use named submodules (gelu_gate, out_proj) to avoid [Module] array
                // issues with quantization (heterogeneous arrays cause mismatchedContainers).
                var remapped = key
                remapped = remapped.replacingOccurrences(of: ".net.0.", with: ".gelu_gate.")
                remapped = remapped.replacingOccurrences(of: ".net.1.", with: ".out_proj.")

                // Regular model: remap Python MLX weight keys to Swift module structure.
                //
                // The Regular model safetensors stores post-Python-sanitized keys which use
                // Python MLX module naming conventions. These differ from Swift's module
                // nesting in several ways:
                //
                // 1. Block arrays: Python uses setattr-based naming (down_blocks_0, transformer_0)
                //    while Swift uses [Module] arrays (down_blocks.0, transformer_blocks.0)
                // 2. Conv wrapping: Python uses nn.Conv1d directly, Swift wraps in S3GenConv1dPT
                // 3. CausalBlock1D: Python uses named attrs (self.conv, self.norm),
                //    Swift uses [Module] array (block.0, block.1)
                // 4. Transformer attention: Python post-sanitized keys use (attn.query_proj),
                //    Swift uses original PyTorch names (attn1.to_q)
                // 5. FeedForward: Python post-sanitized keys use (ff.layers.0),
                //    Swift uses (ff.gelu_gate.proj, ff.out_proj)
                //
                // Turbo weights use PyTorch-format keys that the existing remapping handles.
                if config.modelType == "chatterbox" {
                    remapped = Self.remapRegularS3GenKey(remapped)
                }

                otherS3GenWeights[remapped] = value
            }
        }

        // Run CAMPPlus sanitization on speaker_encoder weights
        let sanitizedSpeakerEncoder = CAMPPlus.sanitize(
            weights: speakerEncoderWeights,
            model: s3gen.speakerEncoder
        )

        // Reconstruct with prefixes
        for (key, value) in sanitizedVE {
            result["ve.\(key)"] = value
        }
        for (key, value) in sanitizedT3 {
            result["t3.\(key)"] = value
        }
        for (key, value) in otherS3GenWeights {
            result["s3gen.\(key)"] = value
        }
        for (key, value) in sanitizedSpeakerEncoder {
            result["s3gen.speaker_encoder.\(key)"] = value
        }

        return result
    }

    /// Remap a single S3Gen weight key from Python-MLX naming (post-sanitized) to Swift module naming.
    ///
    /// The Regular model's safetensors stores post-Python-sanitized keys which use Python MLX
    /// module naming conventions. These differ from Swift's module nesting in several ways:
    ///
    /// Python → Swift key mapping:
    /// 1. `down_blocks_0.` → `down_blocks.0.` (setattr vs [Module] array)
    /// 2. `transformer_0.` → `transformer_blocks.0.` (setattr vs array @ModuleInfo)
    /// 3. `.attn.query_proj.` → `.attn1.to_q.` (Python sanitized → Swift original PyTorch)
    /// 4. `.ff.layers.0.` → `.ff.gelu_gate.proj.` (Python LayerList → Swift named submodules)
    /// 5. `.block1.conv.conv.` → `.block1.block.0.conv.conv.` (named attrs → block array)
    /// 6. `.res_conv.weight` → `.res_conv.conv.weight` (bare Conv1d → S3GenConv1dPT wrapper)
    /// 7. `.final_proj.weight` → `.final_proj.conv.weight`
    /// 8. `.final_block.conv.` → `.final_block.block.0.conv.` (CausalBlock1D)
    /// 9. `.downsample.conv.weight` → `.downsample.conv.conv.weight` (Downsample1D wrapping)
    /// 10. `.mlp_linear.` → `.mlp.0.` (Python sanitized mlp.1→mlp_linear, Swift uses [Linear])
    static func remapRegularS3GenKey(_ key: String) -> String {
        var k = key

        // --- 1. Block array naming: setattr → Swift [Module] arrays ---
        // Python uses `down_blocks_0`, Swift uses `down_blocks.0`
        // Must handle multi-digit indices: down_blocks_12 → down_blocks.12
        k = k.replacingOccurrences(
            of: #"down_blocks_(\d+)\."#,
            with: "down_blocks.$1.",
            options: .regularExpression)
        k = k.replacingOccurrences(
            of: #"mid_blocks_(\d+)\."#,
            with: "mid_blocks.$1.",
            options: .regularExpression)
        k = k.replacingOccurrences(
            of: #"up_blocks_(\d+)\."#,
            with: "up_blocks.$1.",
            options: .regularExpression)

        // --- 2. Transformer array naming within blocks ---
        // Python: `transformer_0.` → Swift: `transformer_blocks.0.`
        k = k.replacingOccurrences(
            of: #"\.transformer_(\d+)\."#,
            with: ".transformer_blocks.$1.",
            options: .regularExpression)
        // Also handle top-level transformer_ (without leading dot)
        if k.hasPrefix("transformer_") {
            k = k.replacingOccurrences(
                of: #"^transformer_(\d+)\."#,
                with: "transformer_blocks.$1.",
                options: .regularExpression)
        }

        // --- 3. Attention naming: Python sanitized → Swift original PyTorch names ---
        // Python: .attn.query_proj. → Swift: .attn1.to_q.
        k = k.replacingOccurrences(of: ".attn.query_proj.", with: ".attn1.to_q.")
        k = k.replacingOccurrences(of: ".attn.key_proj.", with: ".attn1.to_k.")
        k = k.replacingOccurrences(of: ".attn.value_proj.", with: ".attn1.to_v.")
        k = k.replacingOccurrences(of: ".attn.out_proj.", with: ".attn1.to_out.0.")

        // --- 4. FeedForward naming: Python LayerList → Swift named submodules ---
        // Python: .ff.layers.0. (GELU proj) → Swift: .ff.gelu_gate.proj.
        // Python: .ff.layers.1. (output Linear) → Swift: .ff.out_proj.
        k = k.replacingOccurrences(of: ".ff.layers.0.", with: ".ff.gelu_gate.proj.")
        k = k.replacingOccurrences(of: ".ff.layers.1.", with: ".ff.out_proj.")

        // --- 5. CausalBlock1D: Python named attrs → Swift block array ---
        // Python CausalBlock1D stores: self.conv (CausalConv1d), self.norm (LayerNorm)
        // Swift S3GenCausalBlock1D stores: block[0] (CausalConv1d), block[1] (LayerNorm)
        //
        // Python key: .block1.conv.conv.weight → Swift: .block1.block.0.conv.conv.weight
        // Python key: .block1.norm.weight → Swift: .block1.block.1.weight
        // Same for block2 and final_block
        //
        // IMPORTANT: Must replace `conv.conv.` (CausalConv1d path) with `block.0.conv.conv.`
        // and `norm.` with `block.1.` — NOT just `conv.` → `block.0.` which would eat one
        // level of the CausalConv1d nesting.
        k = k.replacingOccurrences(of: ".block1.conv.conv.", with: ".block1.block.0.conv.conv.")
        k = k.replacingOccurrences(of: ".block1.norm.", with: ".block1.block.1.")
        k = k.replacingOccurrences(of: ".block2.conv.conv.", with: ".block2.block.0.conv.conv.")
        k = k.replacingOccurrences(of: ".block2.norm.", with: ".block2.block.1.")
        k = k.replacingOccurrences(of: ".final_block.conv.conv.", with: ".final_block.block.0.conv.conv.")
        k = k.replacingOccurrences(of: ".final_block.norm.", with: ".final_block.block.1.")

        // --- 6. res_conv: bare Conv1d → S3GenConv1dPT wrapper ---
        // Python: .res_conv.weight → Swift: .res_conv.conv.weight
        // Must NOT double-convert if already has .conv.
        // The Python key is `res_conv.weight` or `res_conv.bias` (bare nn.Conv1d).
        // Swift has S3GenConv1dPT which nests as res_conv.conv.weight.
        k = k.replacingOccurrences(
            of: #"\.res_conv\.(weight|bias)"#,
            with: ".res_conv.conv.$1",
            options: .regularExpression)

        // --- 7. final_proj: bare Conv1d → S3GenConv1dPT wrapper ---
        // Python: final_proj.weight → Swift: final_proj.conv.weight
        k = k.replacingOccurrences(
            of: #"\.final_proj\.(weight|bias)"#,
            with: ".final_proj.conv.$1",
            options: .regularExpression)
        // Also handle if final_proj is at the start of the key
        if k.hasPrefix("final_proj.") && !k.hasPrefix("final_proj.conv.") {
            k = k.replacingOccurrences(
                of: #"^final_proj\.(weight|bias)"#,
                with: "final_proj.conv.$1",
                options: .regularExpression)
        }

        // --- 8. Downsample/Upsample: bare Conv1d → S3GenConv1dPT/S3GenConvTranspose1dPT ---
        // Python Downsample1D: .downsample.conv.weight (nn.Conv1d is self.conv)
        // Swift S3GenDownsample1D: .downsample.conv.conv.weight (S3GenConv1dPT wraps Conv1d)
        // But: the last-layer downsample is a CausalConv1d which already has .conv.conv
        // So only add extra .conv when not already present.
        //
        // Match `.downsample.conv.weight` but NOT `.downsample.conv.conv.weight`
        k = k.replacingOccurrences(
            of: #"\.downsample\.conv\.(weight|bias)$"#,
            with: ".downsample.conv.conv.$1",
            options: .regularExpression)
        k = k.replacingOccurrences(
            of: #"\.upsample\.conv\.(weight|bias)$"#,
            with: ".upsample.conv.conv.$1",
            options: .regularExpression)

        // --- 9. MLP naming: Python sanitized mlp_linear → Swift mlp.0 ---
        // Python sanitize: .mlp.1. → .mlp_linear.
        // So safetensors has: .mlp_linear.weight
        // Swift has: @ModuleInfo(key: "mlp") var mlp: [Linear] → .mlp.0.weight
        k = k.replacingOccurrences(of: ".mlp_linear.", with: ".mlp.0.")

        // --- 10. mel2wav (HiFi-GAN vocoder): bare Conv1d → HiFiConv1d/HiFiConvTranspose1d ---
        //
        // The Regular model's Python HiFi-GAN uses bare nn.Conv1d/nn.ConvTranspose1d everywhere,
        // so safetensors keys have no `.conv` wrapping level:
        //   mel2wav.conv_pre.weight, mel2wav.resblocks.0.convs1.0.weight, etc.
        //
        // Swift's HiFiConv1d/HiFiConvTranspose1d wrap Conv1d with @ModuleInfo(key: "conv"),
        // adding a `.conv` level in the key path:
        //   mel2wav.conv_pre.conv.weight, mel2wav.resblocks.0.convs1.0.conv.weight, etc.
        //
        // Insert `.conv` before terminal `.weight`/`.bias` for all mel2wav Conv1d parameters.
        // Skip non-Conv1d parameters (Snake .alpha, Linear .weight/.bias in classifier/l_linear).
        if k.hasPrefix("mel2wav.") {
            k = Self.remapRegularMel2WavKey(k)
        }

        return k
    }

    /// Remap a single mel2wav (HiFi-GAN vocoder) weight key for the Regular model.
    ///
    /// The Regular model's Python HiFi-GAN uses bare nn.Conv1d / nn.ConvTranspose1d,
    /// producing flat keys like `mel2wav.conv_pre.weight`. Swift wraps these in
    /// HiFiConv1d / HiFiConvTranspose1d which add a `.conv` level via @ModuleInfo(key: "conv").
    ///
    /// This function inserts `.conv` before the terminal `.weight`/`.bias` for Conv1d parameters,
    /// while leaving non-Conv1d parameters unchanged (Snake alpha, Linear classifier, etc.).
    ///
    /// Affected paths:
    ///   - mel2wav.conv_pre.{w,b}           → mel2wav.conv_pre.conv.{w,b}
    ///   - mel2wav.conv_post.{w,b}          → mel2wav.conv_post.conv.{w,b}
    ///   - mel2wav.ups.N.{w,b}              → mel2wav.ups.N.conv.{w,b}
    ///   - mel2wav.source_downs.N.{w,b}     → mel2wav.source_downs.N.conv.{w,b}
    ///   - mel2wav.resblocks.N.convsK.M.{w,b}        → ...convsK.M.conv.{w,b}
    ///   - mel2wav.source_resblocks.N.convsK.M.{w,b} → ...convsK.M.conv.{w,b}
    ///   - mel2wav.f0_predictor.condnet.N.{w,b}      → ...condnet.N.conv.{w,b}
    ///
    /// Unchanged paths:
    ///   - mel2wav.resblocks.N.activationsK.M.alpha   (Snake, not Conv1d)
    ///   - mel2wav.f0_predictor.classifier.{w,b}      (Linear, not Conv1d)
    ///   - mel2wav.m_source.l_linear.{w,b}            (Linear, not Conv1d)
    ///   - mel2wav.m_source.l_sin_gen.*                (no parameters)
    static func remapRegularMel2WavKey(_ key: String) -> String {
        // Strip mel2wav. prefix, process, then re-add
        guard key.hasPrefix("mel2wav.") else { return key }
        let subKey = String(key.dropFirst("mel2wav.".count))

        // Match all Conv1d/ConvTranspose1d terminal patterns:
        //   conv_pre.{weight,bias}
        //   conv_post.{weight,bias}
        //   ups.N.{weight,bias}
        //   source_downs.N.{weight,bias}
        //   resblocks.N.convs1.M.{weight,bias}
        //   resblocks.N.convs2.M.{weight,bias}
        //   source_resblocks.N.convs1.M.{weight,bias}
        //   source_resblocks.N.convs2.M.{weight,bias}
        //   f0_predictor.condnet.N.{weight,bias}
        //
        // Pattern: these all end with either:
        //   (named_module).{weight,bias}  where named_module is conv_pre, conv_post
        //   (array_module).N.{weight,bias} where array_module is ups, source_downs, condnet, convs1, convs2
        //
        // We insert `.conv` before the terminal `.weight`/`.bias`.
        // Regex approach: match known Conv1d parent patterns and insert .conv

        // conv_pre / conv_post
        let remapped = subKey
            .replacingOccurrences(
                of: #"^(conv_pre|conv_post)\.(weight|bias)$"#,
                with: "$1.conv.$2",
                options: .regularExpression)
            // ups.N / source_downs.N
            .replacingOccurrences(
                of: #"^(ups|source_downs)\.(\d+)\.(weight|bias)$"#,
                with: "$1.$2.conv.$3",
                options: .regularExpression)
            // resblocks.N.convs{1,2}.M / source_resblocks.N.convs{1,2}.M
            .replacingOccurrences(
                of: #"^((?:source_)?resblocks\.\d+\.convs[12]\.\d+)\.(weight|bias)$"#,
                with: "$1.conv.$2",
                options: .regularExpression)
            // f0_predictor.condnet.N
            .replacingOccurrences(
                of: #"^(f0_predictor\.condnet\.\d+)\.(weight|bias)$"#,
                with: "$1.conv.$2",
                options: .regularExpression)

        return "mel2wav.\(remapped)"
    }

    // MARK: - Text Tokenization

    /// Tokenize text into token IDs for T3.
    func tokenizeText(_ text: String) throws -> MLXArray {
        guard let tokenizer = tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let encoded = tokenizer.encode(text: text)

        if config.meanflow {
            // Turbo: GPT-2 tokenizer — use raw token IDs, no SOT/EOT wrapping
            let ids = encoded.map { Int32($0) }
            return MLXArray(ids).reshaped([1, -1])
        } else {
            // Regular (LLaMA): custom EnTokenizer — wrap with SOT/EOT
            var ids = [Int32(sotToken)]
            ids.append(contentsOf: encoded.map { Int32($0) })
            ids.append(Int32(eotToken))
            return MLXArray(ids)
        }
    }

    // MARK: - Reference Audio Processing

    /// Process reference audio to extract speaker embedding and prompt tokens.
    /// Conditioning result from reference audio processing.
    struct RefAudioConditioning {
        let t3Cond: T3Cond
        /// S3Gen x-vector (speaker embedding for flow decoder)
        let xVector: MLXArray
        /// S3Gen prompt tokens (speech tokens for flow encoder input)
        let s3genPromptToken: MLXArray
        /// S3Gen prompt mel features (mel spectrogram for flow conditioning)
        let s3genPromptFeat: MLXArray
    }

    func prepareConditionals(
        refAudio: MLXArray,
        refAudioSR: Int = 24000
    ) throws -> RefAudioConditioning {
        var audio = refAudio
        if audio.ndim > 1 {
            audio = audio.mean(axis: 0)
        }

        // --- Loudness normalization (Turbo only, matches Python norm_loudness) ---
        // Python Chatterbox Turbo normalizes to -27 LUFS before any conditioning extraction.
        // This ensures consistent conditioning strength regardless of input volume.
        if config.isTurbo {
            audio = normalizeLoudness(audio, targetLUFS: -27.0)
            print("[Chatterbox]   Loudness normalized to -27 LUFS")
        }

        // --- Audio at different sample rates ---
        // 24kHz for S3Gen mel features (decoder conditioning)
        let audio24k: MLXArray
        if refAudioSR != ChatterboxConstants.s3genSampleRate {
            audio24k = resampleAudio(audio, fromSR: refAudioSR, toSR: ChatterboxConstants.s3genSampleRate)
        } else {
            audio24k = audio
        }
        let decCondLen = config.decCondLen
        let audio24kTrunc = audio24k.dim(0) > decCondLen ? audio24k[..<decCondLen] : audio24k

        // 16kHz for S3TokenizerV2, CAMPPlus, VoiceEncoder — single resample, derive all variants
        let audio16kFull: MLXArray
        if refAudioSR != ChatterboxConstants.s3SampleRate {
            audio16kFull = resampleAudio(audio, fromSR: refAudioSR, toSR: ChatterboxConstants.s3SampleRate)
        } else {
            audio16kFull = audio
        }
        let encCondLen = config.encCondLen
        let audio16kEnc = audio16kFull.dim(0) > encCondLen ? audio16kFull[..<encCondLen] : audio16kFull

        // Derive 16kHz for S3Gen tokenizer/CAMPPlus from the already-resampled 16k audio.
        // This is equivalent to resampling audio24kTrunc 24k→16k, but avoids a second
        // expensive polyphase resample. The 16k equivalent of decCondLen (24k) samples:
        let decCondLen16k = (decCondLen * ChatterboxConstants.s3SampleRate) / ChatterboxConstants.s3genSampleRate
        let audio16kFromDec = audio16kFull.dim(0) > decCondLen16k
            ? audio16kFull[..<decCondLen16k] : audio16kFull

        // --- 1. VoiceEncoder: 256-dim speaker embedding for T3 ---
        // Trim silence before computing speaker embedding (matches Python librosa.effects.trim)
        let audio16kTrimmed = trimSilence(audio16kEnc, topDb: 20.0)
        print("[Chatterbox]   VoiceEncoder: trimmed \(audio16kEnc.dim(0)) → \(audio16kTrimmed.dim(0)) samples")
        let veMels = voiceEncoderMelSpectrogram(audio16kTrimmed, isTurbo: config.isTurbo)
        let veMelsTransposed = veMels.transposed().expandedDimensions(axis: 0)
        let speakerEmb = voiceEncoder.inference(
            mels: veMelsTransposed,
            melLens: [veMelsTransposed.dim(1)]
        )
        // No eval() here — defer to single eval below for GPU pipelining

        // --- 2. S3TokenizerV2: speech token IDs for T3 and S3Gen ---
        let t3PromptSpeechTokens: MLXArray
        let s3genPromptToken: MLXArray

        if let s3tok = s3Tokenizer {
            // T3 tokens: from encCondLen audio at 16kHz
            let t3Mel = s3TokenizerLogMelSpectrogram(audio16kEnc) // (128, T)
            let t3MelBatch = t3Mel.expandedDimensions(axis: 0) // (1, 128, T)
            let t3MelLen = MLXArray([Int32(t3Mel.dim(1))])
            let (t3Tokens, _) = s3tok(t3MelBatch, melLen: t3MelLen)
            // No eval() here — defer to single eval below

            // Truncate to speechCondPromptLen (150 for regular, 375 for turbo)
            let plen = config.t3Config.speechCondPromptLen
            t3PromptSpeechTokens = t3Tokens[0..., ..<min(plen, t3Tokens.dim(1))]

            // S3Gen tokens: from decCondLen audio
            let s3genMel = s3TokenizerLogMelSpectrogram(audio16kFromDec)
            let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
            let s3genMelLen = MLXArray([Int32(s3genMel.dim(1))])
            let (s3genTokens, _) = s3tok(s3genMelBatch, melLen: s3genMelLen)
            // No eval() here — defer to single eval below
            s3genPromptToken = s3genTokens

            print("[Chatterbox]   S3TokenizerV2: T3 tokens \(t3PromptSpeechTokens.shape) (plen=\(plen)), S3Gen tokens \(s3genPromptToken.shape)")
        } else {
            // Fallback: use default conditioning tokens
            print("[Chatterbox]   S3TokenizerV2 not loaded — falling back to defaults")
            if let defaultTokens = defaultConditioning?.condPromptSpeechTokens,
               defaultTokens.dim(1) > 0 {
                t3PromptSpeechTokens = defaultTokens
            } else {
                t3PromptSpeechTokens = MLXArray.zeros([1, 0]).asType(.int32)
            }
            s3genPromptToken = defaultConditioning?.promptToken ?? MLXArray.zeros([1, 0]).asType(.int32)
        }

        // --- 3. CAMPPlus: 192-dim x-vector for S3Gen flow decoder ---
        let xVector = s3gen.speakerEncoder.inference([audio16kFromDec])
        // No eval() here — defer to single eval below

        // --- 4. S3Gen prompt mel features ---
        let s3genPromptFeat = s3genMelSpectrogram(
            y: audio24kTrunc.expandedDimensions(axis: 0),
            samplingRate: ChatterboxConstants.s3genSampleRate
        ) // (1, 80, T_mel)

        // --- Single eval: evaluate entire computation graph at once ---
        // This lets MLX pipeline all independent branches (VoiceEncoder, S3Tokenizer,
        // CAMPPlus, mel) without GPU sync barriers between them.
        eval(speakerEmb, s3genPromptFeat, xVector)
        if s3Tokenizer != nil {
            eval(t3PromptSpeechTokens, s3genPromptToken)
        }
        print("[Chatterbox]   VoiceEncoder: speakerEmb \(speakerEmb.shape)")
        print("[Chatterbox]   CAMPPlus: xVector \(xVector.shape)")

        // --- 5. Align token count and mel frame count ---
        // Invariant: mel_frames = 2 * num_tokens (token_mel_ratio = 2)
        var promptToken = s3genPromptToken
        var promptFeat = s3genPromptFeat
        let tokenMelRatio = 2

        if promptToken.dim(1) > 0 && promptFeat.dim(2) > 0 {
            let numTokens = promptToken.dim(1)
            let melFrames = promptFeat.dim(2)
            let expectedMel = numTokens * tokenMelRatio

            if expectedMel < melFrames {
                // Truncate mel to match tokens
                promptFeat = promptFeat[0..., 0..., ..<expectedMel]
            } else if expectedMel > melFrames {
                // Truncate tokens to match mel
                let maxTokens = melFrames / tokenMelRatio
                if maxTokens > 0 {
                    promptToken = promptToken[0..., ..<maxTokens]
                }
            }
            print("[Chatterbox]   Aligned: promptToken \(promptToken.shape), promptFeat \(promptFeat.shape)")
        }

        // Build T3Cond
        let t3Cond = T3Cond(
            speakerEmb: speakerEmb,
            condPromptSpeechTokens: t3PromptSpeechTokens.dim(1) > 0 ? t3PromptSpeechTokens : nil,
            condPromptSpeechEmb: nil,
            emotionAdv: MLXArray(Float(emotionAdvOverride ?? 0.5))
        )

        return RefAudioConditioning(
            t3Cond: t3Cond,
            xVector: xVector,
            s3genPromptToken: promptToken,
            s3genPromptFeat: promptFeat
        )
    }

    // MARK: - Speech Token Post-processing

    /// Drop invalid speech tokens (out of vocab range).
    /// Matches Python: `mask = np.where(np.array(speech_tokens) < 6561)[0]`
    /// Only keeps actual speech tokens (< startSpeechToken), not control tokens.
    func dropInvalidTokens(_ tokens: MLXArray) -> MLXArray {
        let flat = tokens.reshaped([-1])
        let count = flat.dim(0)
        // Python filters with `< 6561` (SPEECH_VOCAB_SIZE), which excludes control tokens
        // (start=6561, stop=6562). Use the startSpeechToken as the threshold.
        let threshold = config.t3Config.startSpeechToken
        var validIds = [Int32]()

        for i in 0 ..< count {
            let id = flat[i].item(Int.self)
            if id >= 0 && id < threshold {
                validIds.append(Int32(id))
            }
        }

        if validIds.isEmpty {
            return MLXArray([Int32(0)])
        }
        return MLXArray(validIds)
    }

    // MARK: - Generation

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // Use reference audio, or fall back to default conditioning
        let t3Cond: T3Cond
        let xVector: MLXArray
        let promptTokens: MLXArray
        let promptFeat: MLXArray

        if let refAudio = refAudio {
            print("[Chatterbox] Processing reference audio (shape: \(refAudio.shape))...")
            let condStart = CFAbsoluteTimeGetCurrent()
            let refCond = try prepareConditionals(refAudio: refAudio)
            print("[Chatterbox] Conditioning prepared in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - condStart))s")
            print("[Chatterbox]   speakerEmb: \(refCond.t3Cond.speakerEmb.shape), s3genPromptToken: \(refCond.s3genPromptToken.shape), s3genPromptFeat: \(refCond.s3genPromptFeat.shape), xVector: \(refCond.xVector.shape)")
            t3Cond = refCond.t3Cond
            xVector = refCond.xVector
            promptTokens = refCond.s3genPromptToken
            promptFeat = refCond.s3genPromptFeat
        } else if let defaults = defaultConditioning {
            // Use pre-computed default voice
            t3Cond = T3Cond(
                speakerEmb: defaults.speakerEmb,
                condPromptSpeechTokens: defaults.condPromptSpeechTokens,
                condPromptSpeechEmb: nil,
                emotionAdv: defaults.emotionAdv
            )
            xVector = defaults.xVector
            promptTokens = defaults.promptToken
            promptFeat = defaults.promptFeat
        } else {
            throw AudioGenerationError.invalidInput(
                "Chatterbox requires reference audio for voice cloning. Pass refAudio parameter."
            )
        }

        // Tokenize text
        let textTokens = try tokenizeText(text)
        print("[Chatterbox] Text tokenized: \(textTokens.shape)")

        let temperature = generationParameters.temperature
        let topP = generationParameters.topP

        // Cap max tokens: when using reference audio without prompt speech tokens,
        // the model may not generate EOS reliably, so use a smaller limit.
        // ~10 speech tokens per text token is a reasonable heuristic.
        let hasPromptTokens = t3Cond.condPromptSpeechTokens != nil
            && (t3Cond.condPromptSpeechTokens?.dim(1) ?? 0) > 0
        let maxTokens: Int
        if hasPromptTokens {
            maxTokens = config.t3Config.maxSpeechTokens
        } else {
            // Estimate: ~10 speech tokens per text token, with min 200, max 768
            let textLen = textTokens.dim(textTokens.ndim - 1)
            maxTokens = min(768, max(200, textLen * 10))
            print("[Chatterbox] No prompt speech tokens — capping maxTokens to \(maxTokens) (text length: \(textLen))")
        }

        print("[Chatterbox] Stage 1: T3 text→speech tokens (maxTokens=\(maxTokens))")
        let t3Start = CFAbsoluteTimeGetCurrent()
        // Stage 1: T3 — generate speech tokens
        var t3CondMut = t3Cond
        let speechTokens: MLXArray

        if let t3gpt2 = t3 as? T3GPT2Model {
            // Turbo: GPT-2 inference (no CFG)
            speechTokens = t3gpt2.inference(
                t3Cond: &t3CondMut,
                textTokens: textTokens,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topK: 1000,
                topP: topP,
                repetitionPenalty: 1.2
            )
        } else if let t3llama = t3 as? T3Model {
            // Regular: LLaMA inference (with CFG + min_p filtering)
            // Python reference uses topP=1.0 (disabled) + minP=0.05 for the Regular model,
            // relying on min-p filtering instead of nucleus sampling.
            speechTokens = t3llama.inference(
                t3Cond: &t3CondMut,
                textTokens: textTokens,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topP: 1.0,
                minP: 0.05,
                repetitionPenalty: 1.2,
                cfgWeight: cfgWeightOverride ?? 0.5
            )
        } else {
            throw AudioGenerationError.modelNotInitialized("Unknown T3 model type")
        }
        eval(speechTokens)
        print("[Chatterbox] Stage 1 complete: \(speechTokens.shape) in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - t3Start))s")

        // Post-process: remove OOV tokens and append silence
        let cleanTokens = dropInvalidTokens(speechTokens)
        // Append 3 silence tokens (S3GEN_SIL = 4299) to reduce artifacts at end
        let silenceTokens = MLXArray([Int32(4299), Int32(4299), Int32(4299)])
        let finalTokens = MLX.concatenated([cleanTokens, silenceTokens])

        // Stage 2: S3Gen — speech tokens → mel → waveform
        let tokenArr = finalTokens.reshaped([1, -1])
        let tokenLen = MLXArray([Int32(tokenArr.dim(1))])
        let promptTokenLen = MLXArray([Int32(promptTokens.dim(1))])

        // Prepare prompt mel features for S3Gen conditioning
        // s3genMelSpectrogram returns (B, 80, T') — transpose to (B, T', 80) for inference
        let promptFeatTransposed: MLXArray
        if promptFeat.dim(1) == 80 && promptFeat.ndim == 3 {
            // Shape is (B, 80, T') — transpose to (B, T', 80)
            promptFeatTransposed = promptFeat.transposed(0, 2, 1)
        } else {
            // Already (B, T', 80) or some other shape
            promptFeatTransposed = promptFeat
        }

        // Run flow matching inference with raw int32 token IDs
        // Meanflow (turbo distilled): 2 steps. Non-meanflow (regular): 10 steps.
        let nTimesteps = config.meanflow ? 2 : 10
        print("[Chatterbox] Stage 2: S3Gen speech tokens→mel (nTimesteps=\(nTimesteps), tokens=\(tokenArr.shape))")
        let s3Start = CFAbsoluteTimeGetCurrent()
        let mel = s3gen.inference(
            token: tokenArr,
            tokenLen: tokenLen,
            promptToken: promptTokens,
            promptTokenLen: promptTokenLen,
            promptFeat: promptFeatTransposed,
            embedding: xVector,
            nTimesteps: nTimesteps
        )
        eval(mel)
        print("[Chatterbox] Stage 2 complete: mel \(mel.shape) in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - s3Start))s")

        print("[Chatterbox] Stage 3: Vocoder mel→waveform")
        let vocStart = CFAbsoluteTimeGetCurrent()
        // Vocoder: mel → waveform via HiFi-GAN
        var (waveform, _) = s3gen.vocoder(mel)

        // Apply fade-in window to reduce vocoder spillover artifacts at start (matches Python trim_fade)
        // 480 samples silence + 480 samples cosine ramp = 960 samples (40ms at 24kHz)
        let nTrim = sampleRate / 50  // 480
        let fadeLen = nTrim * 2      // 960
        if waveform.dim(waveform.ndim - 1) >= fadeLen {
            // Build fade: [zeros(480), cosine_ramp(480)] — matches Python trim_fade
            let zeros = MLXArray.zeros([nTrim])
            let cosValues = (0..<nTrim).map { i -> Float in
                let t = Float.pi * (1.0 - Float(i) / Float(nTrim - 1))
                return (cos(t) + 1) / 2
            }
            let cosineRamp = MLXArray(cosValues)
            let trimFade = MLX.concatenated([zeros, cosineRamp])

            // Multiply first fadeLen samples of waveform
            let fadePart = waveform[0..., ..<fadeLen] * trimFade
            let restPart = waveform[0..., fadeLen...]
            waveform = MLX.concatenated([fadePart, restPart], axis: -1)
        }

        // Peak-normalize output to ensure audible volume.
        // The vocoder output can be quiet depending on conditioning strength.
        // Scale so peak amplitude is ~0.95 (standard headroom for speech).
        let peak = MLX.abs(waveform).max().item(Float.self)
        if peak > 1e-6 {
            let targetPeak: Float = 0.95
            waveform = waveform * MLXArray(targetPeak / peak)
        }

        print("[Chatterbox] Stage 3 complete: waveform \(waveform.shape) in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - vocStart))s")
        return waveform
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let startTime = Date()
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                let generateTime = Date().timeIntervalSince(startTime)

                continuation.yield(.audio(audio))

                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: audio.dim(audio.ndim - 1),
                    prefillTime: 0,
                    generateTime: generateTime,
                    tokensPerSecond: Double(audio.dim(audio.ndim - 1)) / max(generateTime, 0.001),
                    peakMemoryUsage: 0
                )
                continuation.yield(.info(info))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    // MARK: - Factory

    /// Load Chatterbox model from HuggingFace repository.
    ///
    /// Supports both Regular (`Chatterbox-TTS-fp16`) and Turbo (`chatterbox-turbo-fp16`).
    /// Automatically detects model variant from config.json.
    public static func fromPretrained(_ modelRepo: String) async throws -> ChatterboxModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        return try await fromModelDirectory(modelDir, hfToken: hfToken)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        hfToken: String?
    ) async throws -> ChatterboxModel {
        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let config: ChatterboxConfiguration
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ChatterboxConfiguration.self, from: configData)
        } else {
            config = .default
        }

        // Create model (T3 variant selected by config)
        let model = ChatterboxModel(config)
        model.modelDir = modelDir

        // Load main weights
        let weights = try loadChatterboxWeights(modelDir: modelDir)

        // Sanitize weights
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantization
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitizedWeights["\(path).scales"] != nil else { return nil }
                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path)
                {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }

        // Update model parameters — allow unused keys since we drop tokenizer weights
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [])

        eval(model)

        // Ensure tokenizer files exist before loading
        // Turbo: ships with slow tokenizer only (vocab.json + merges.txt) — generate fast tokenizer.json
        let tokenizerJsonPath = modelDir.appendingPathComponent("tokenizer.json")
        if !FileManager.default.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            if FileManager.default.fileExists(atPath: vocabPath.path),
               FileManager.default.fileExists(atPath: mergesPath.path)
            {
                try generateTokenizerJson(
                    vocabPath: vocabPath,
                    mergesPath: mergesPath,
                    tokenizerConfigPath: modelDir.appendingPathComponent("tokenizer_config.json"),
                    outputPath: tokenizerJsonPath
                )
            }
        }

        // Regular: has tokenizer.json but missing tokenizer_config.json — generate minimal config
        let tokenizerConfigPath = modelDir.appendingPathComponent("tokenizer_config.json")
        if !FileManager.default.fileExists(atPath: tokenizerConfigPath.path) {
            let minimalConfig: [String: Any] = ["tokenizer_class": "GPT2Tokenizer"]
            let data = try JSONSerialization.data(withJSONObject: minimalConfig)
            try data.write(to: tokenizerConfigPath)
        }

        // Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            print("Warning: Could not load tokenizer from model folder: \(error)")
        }

        // Load default voice conditioning (conds.safetensors)
        let condsURL = modelDir.appendingPathComponent("conds.safetensors")
        if FileManager.default.fileExists(atPath: condsURL.path) {
            do {
                let condsWeights = try MLX.loadArrays(url: condsURL)
                model.defaultConditioning = DefaultConditioning(
                    speakerEmb: condsWeights["t3.speaker_emb"] ?? MLXArray.zeros([1, 256]),
                    condPromptSpeechTokens: condsWeights["t3.cond_prompt_speech_tokens"] ?? MLXArray.zeros([1, 0]).asType(.int32),
                    emotionAdv: condsWeights["t3.emotion_adv"] ?? MLXArray(Float(0.5)).reshaped([1, 1, 1]),
                    xVector: condsWeights["gen.embedding"] ?? MLXArray.zeros([1, 192]),
                    promptToken: condsWeights["gen.prompt_token"] ?? MLXArray.zeros([1, 0]).asType(.int32),
                    promptTokenLen: condsWeights["gen.prompt_token_len"] ?? MLXArray([Int32(0)]),
                    promptFeat: condsWeights["gen.prompt_feat"] ?? MLXArray.zeros([1, 0, 80]),
                    promptFeatLen: MLXArray([Int32(0)])
                )
                eval(model.defaultConditioning!.speakerEmb)
                eval(model.defaultConditioning!.xVector)
            } catch {
                print("Warning: Could not load default conditioning from conds.safetensors: \(error)")
            }
        }

        // Load S3TokenizerV2 from separate HuggingFace repo (needed for voice cloning)
        let s3TokenizerRepo = "mlx-community/S3TokenizerV2"
        do {
            guard let s3RepoID = Repo.ID(rawValue: s3TokenizerRepo) else {
                throw AudioGenerationError.invalidInput("Invalid S3Tokenizer repo ID")
            }
            let s3Dir = try await ModelUtils.resolveOrDownloadModel(
                repoID: s3RepoID,
                requiredExtension: "safetensors",
                hfToken: hfToken
            )
            let s3WeightsURL = s3Dir.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: s3WeightsURL.path) {
                let s3Tokenizer = S3TokenizerV2()
                var s3Weights = try MLX.loadArrays(url: s3WeightsURL)
                s3Weights = S3TokenizerV2.sanitize(weights: s3Weights, model: s3Tokenizer)
                try s3Tokenizer.update(
                    parameters: ModuleParameters.unflattened(s3Weights), verify: []
                )
                eval(s3Tokenizer)
                model.s3Tokenizer = s3Tokenizer
                print("[Chatterbox] Loaded S3TokenizerV2 from \(s3TokenizerRepo)")
            }
        } catch {
            print("Warning: Could not load S3TokenizerV2: \(error)")
            print("  Voice cloning will fall back to default conditioning tokens.")
        }

        return model
    }
}

// MARK: - Weight Loading

/// Load safetensors weights for Chatterbox.
///
/// Handles both single `model.safetensors` and sharded patterns.
/// Excludes `conds.safetensors` (loaded separately for default voice).
private func loadChatterboxWeights(modelDir: URL) throws -> [String: MLXArray] {
    let singleWeightsURL = modelDir.appendingPathComponent("model.safetensors")
    if FileManager.default.fileExists(atPath: singleWeightsURL.path) {
        return try MLX.loadArrays(url: singleWeightsURL)
    }

    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
    let safetensorFiles = files
        .filter {
            $0.pathExtension == "safetensors"
                && $0.lastPathComponent != "conds.safetensors"
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !safetensorFiles.isEmpty else {
        throw AudioGenerationError.modelNotInitialized("No .safetensors files found in \(modelDir.path)")
    }

    var allWeights = [String: MLXArray]()
    for file in safetensorFiles {
        let shardWeights = try MLX.loadArrays(url: file)
        for (key, value) in shardWeights {
            allWeights[key] = value
        }
    }

    return allWeights
}

// MARK: - Audio Resampling

/// Polyphase FIR audio resampling with proper anti-aliasing.
///
/// Matches the quality of `scipy.signal.resample_poly` / `librosa.resample`:
/// designs a windowed-sinc lowpass FIR filter, upsamples, filters, and downsamples.
/// This prevents aliasing artifacts that occur with naive linear interpolation.
///
/// Uses a polyphase decomposition for efficiency: the FIR filter is split into
/// `up` sub-filters (phases), and only the relevant phase is evaluated per output sample.
/// Audio samples are bulk-extracted via `asArray()` for fast CPU-side computation.
private func resampleAudio(_ audio: MLXArray, fromSR: Int, toSR: Int) -> MLXArray {
    guard fromSR != toSR else { return audio }

    func gcd(_ a: Int, _ b: Int) -> Int {
        var a = a, b = b
        while b != 0 { (a, b) = (b, a % b) }
        return a
    }

    let g = gcd(fromSR, toSR)
    let up = toSR / g
    let down = fromSR / g

    let inputLength = audio.dim(0)
    let outputLength = (inputLength * up + down - 1) / down

    // Design lowpass FIR filter: windowed sinc with Kaiser window
    let nZeroCrossings = 10
    let filterHalfLen = nZeroCrossings * max(up, down)
    let filterLen = 2 * filterHalfLen + 1

    // Cutoff frequency: min(1/up, 1/down) to prevent aliasing
    let fc = 1.0 / Float(max(up, down))

    var h = [Float](repeating: 0, count: filterLen)
    let beta: Float = 5.0
    let i0Beta = besselI0(beta)
    for i in 0 ..< filterLen {
        let n = Float(i - filterHalfLen)
        // Sinc
        let sincVal: Float = (n == 0) ? fc : fc * sin(Float.pi * fc * n) / (Float.pi * fc * n)
        // Kaiser window
        let x = 2.0 * Float(i) / Float(filterLen - 1) - 1.0
        let arg = beta * sqrt(max(0, 1.0 - x * x))
        h[i] = sincVal * besselI0(arg) / i0Beta
    }

    // Normalize filter so that passband gain = up
    let filterSum = h.reduce(0, +)
    let normFactor = Float(up) / filterSum
    for i in 0 ..< filterLen { h[i] *= normFactor }

    // Decompose filter into polyphase components: phase p has taps at indices p, p+up, p+2*up, ...
    // For each output sample i: phase = (i * down) % up
    // The convolution sum becomes a dot product of the phase's taps with input samples.
    let tapsPerPhase = (filterLen + up - 1) / up
    var polyphase = [[Float]](repeating: [Float](repeating: 0, count: tapsPerPhase), count: up)
    for p in 0 ..< up {
        var t = 0
        var idx = p
        while idx < filterLen {
            polyphase[p][t] = h[idx]
            t += 1
            idx += up
        }
    }

    // Bulk-extract audio samples (single memcpy, not per-element)
    let flatAudio = audio.reshaped([-1]).asType(.float32)
    eval(flatAudio)
    let audioSamples = flatAudio.asArray(Float.self)

    // Polyphase resampling: for each output sample, pick the right phase and dot-product
    var output = [Float](repeating: 0, count: outputLength)
    for i in 0 ..< outputLength {
        let n = i * down  // Position in upsampled signal
        let phase = n % up
        let taps = polyphase[phase]

        // First input sample that contributes: ceil((n - filterHalfLen) / up)
        // But also the polyphase offset: startOrig = (n - phase) / up - (filterHalfLen - phase) / up
        // Simplified: the k-th tap of this phase corresponds to input index (n / up) - k + correction
        let baseOrig = (n - phase) / up  // Input index for tap 0 (before centering)
        let centerOffset = filterHalfLen / up  // Centering shift

        var sum: Float = 0
        for t in 0 ..< tapsPerPhase {
            let origIdx = baseOrig - centerOffset + t
            if origIdx >= 0 && origIdx < inputLength {
                sum += audioSamples[origIdx] * taps[t]
            }
        }
        output[i] = sum
    }

    return MLXArray(output)
}

/// Modified Bessel function of the first kind, order 0.
/// Used for Kaiser window computation.
private func besselI0(_ x: Float) -> Float {
    var sum: Float = 1.0
    var term: Float = 1.0
    let halfX = x / 2.0
    for k in 1 ... 25 {
        term *= (halfX / Float(k)) * (halfX / Float(k))
        sum += term
        if term < 1e-10 * sum { break }
    }
    return sum
}

// MARK: - LUFS Loudness Normalization

/// Normalize audio loudness to a target LUFS level.
///
/// Matches Python Chatterbox Turbo's `norm_loudness()` which uses `pyloudnorm`
/// to normalize reference audio to -27 LUFS before conditioning extraction.
/// Uses an RMS-based approximation of integrated loudness (accurate for speech).
///
/// - Parameters:
///   - audio: 1D audio waveform
///   - targetLUFS: Target loudness in LUFS (default: -27)
/// - Returns: Loudness-normalized audio
private func normalizeLoudness(_ audio: MLXArray, targetLUFS: Float = -27.0) -> MLXArray {
    // Compute RMS (approximation of integrated loudness for speech signals)
    let rms = MLX.sqrt(MLX.mean(audio * audio)).item(Float.self)
    guard rms > 1e-10 else { return audio }

    // Convert RMS to approximate LUFS (RMS dBFS ≈ LUFS for speech with K-weighting ≈ unity)
    let currentLUFS = 20.0 * log10(rms)
    let gainDB = targetLUFS - currentLUFS
    let gainLinear = pow(10.0, gainDB / 20.0)

    guard gainLinear.isFinite && gainLinear > 0 else { return audio }
    return audio * MLXArray(gainLinear)
}

// MARK: - Silence Trimming

/// Trim leading and trailing silence from audio.
///
/// Matches Python's `librosa.effects.trim(wav, top_db=20)` used by both
/// Regular and Turbo VoiceEncoder before computing speaker embeddings.
/// Uses vectorized MLX operations (asStrided framing) for fast computation.
///
/// - Parameters:
///   - audio: 1D audio waveform
///   - topDb: Threshold in dB below peak RMS to consider as silence (default: 20)
///   - frameLength: Analysis frame length in samples (default: 2048)
///   - hopLength: Hop between frames (default: 512)
/// - Returns: Trimmed audio with silence removed from both ends
private func trimSilence(
    _ audio: MLXArray,
    topDb: Float = 20.0,
    frameLength: Int = 2048,
    hopLength: Int = 512
) -> MLXArray {
    let inputLen = audio.dim(0)
    let nFrames = max(0, 1 + (inputLen - frameLength) / hopLength)
    guard nFrames > 0 else { return audio }

    // Frame the audio using strided view (vectorized, no per-sample loop)
    let floatAudio = audio.reshaped([-1]).asType(.float32)
    let frames = asStrided(floatAudio, [nFrames, frameLength], strides: [hopLength, 1], offset: 0)

    // Compute per-frame RMS energy in dB (fully vectorized)
    let rmsEnergy = MLX.sqrt(MLX.mean(frames * frames, axis: 1))  // (nFrames,)
    let rmsDb = 20.0 * MLX.log10(MLX.maximum(rmsEnergy, MLXArray(Float(1e-10))))
    eval(rmsDb)

    let maxDb = rmsDb.max().item(Float.self)
    let threshold = maxDb - topDb

    // Find first and last frames above threshold
    let aboveThreshold = rmsDb .>= MLXArray(threshold)
    eval(aboveThreshold)
    let mask = aboveThreshold.asArray(Bool.self)

    var startFrame = 0
    var endFrame = nFrames
    for i in 0 ..< nFrames {
        if mask[i] { startFrame = i; break }
    }
    for i in stride(from: nFrames - 1, through: 0, by: -1) {
        if mask[i] { endFrame = i + 1; break }
    }

    let startSample = startFrame * hopLength
    let endSample = min(endFrame * hopLength + frameLength, inputLen)

    guard endSample > startSample else { return audio }
    return audio[startSample ..< endSample]
}

// MARK: - Tokenizer Generation

/// Generate `tokenizer.json` (fast tokenizer format) from `vocab.json` + `merges.txt`.
///
/// Chatterbox Turbo ships with a slow tokenizer (vocab.json + merges.txt) but
/// swift-transformers requires tokenizer.json. This builds the fast tokenizer JSON
/// from the available files, using GPT-2 style BPE with ByteLevel pre-tokenizer.
///
/// Pattern reused from Qwen3TTS which has the same requirement.
private func generateTokenizerJson(
    vocabPath: URL,
    mergesPath: URL,
    tokenizerConfigPath: URL,
    outputPath: URL
) throws {
    // Read vocab
    let vocabData = try Data(contentsOf: vocabPath)
    let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

    // Read merges (skip header line "#version: ...")
    let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
    let mergeLines = mergesText.components(separatedBy: .newlines)
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }

    // Read added_tokens from tokenizer_config.json (if available)
    var addedTokens = [[String: Any]]()
    if let configData = try? Data(contentsOf: tokenizerConfigPath),
       let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
       let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]]
    {
        for (idStr, tokenInfo) in addedTokensDecoder {
            guard let tokenId = Int(idStr),
                  let content = tokenInfo["content"] as? String else { continue }
            let entry: [String: Any] = [
                "id": tokenId,
                "content": content,
                "single_word": tokenInfo["single_word"] as? Bool ?? false,
                "lstrip": tokenInfo["lstrip"] as? Bool ?? false,
                "rstrip": tokenInfo["rstrip"] as? Bool ?? false,
                "normalized": tokenInfo["normalized"] as? Bool ?? false,
                "special": tokenInfo["special"] as? Bool ?? true,
            ]
            addedTokens.append(entry)
        }
        addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
    }

    // Build tokenizer.json — GPT-2 style BPE with ByteLevel pre-tokenizer
    let tokenizerJson: [String: Any] = [
        "version": "1.0",
        "truncation": NSNull(),
        "padding": NSNull(),
        "added_tokens": addedTokens,
        "normalizer": NSNull(),
        "pre_tokenizer": [
            "type": "Sequence",
            "pretokenizers": [
                [
                    "type": "Split",
                    "pattern": [
                        "Regex":
                            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    ],
                    "behavior": "Isolated",
                    "invert": false,
                ] as [String: Any],
                [
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false,
                ] as [String: Any],
            ] as [[String: Any]],
        ] as [String: Any],
        "post_processor": NSNull(),
        "decoder": [
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        ] as [String: Any],
        "model": [
            "type": "BPE",
            "dropout": NSNull(),
            "unk_token": NSNull(),
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocabDict,
            "merges": mergeLines,
        ] as [String: Any],
    ]

    let jsonData = try JSONSerialization.data(withJSONObject: tokenizerJson, options: [.sortedKeys])
    try jsonData.write(to: outputPath)
}
