// Ported from Python mlx-audio codec/models/s3/model_v2.py
// S3TokenizerV2: Supervised Semantic Speech Tokenizer
// Converts 16kHz audio waveform → discrete speech token IDs (6561 vocab, 25 tokens/sec)

import Foundation
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

// MARK: - Configuration

/// S3TokenizerV2 model configuration.
public struct S3TokenizerConfig {
    public var nMels: Int = 128
    public var nAudioCtx: Int = 1500
    public var nAudioState: Int = 1280
    public var nAudioHead: Int = 20
    public var nAudioLayer: Int = 6
    public var nCodebookSize: Int = 6561 // 3^8

    public init() {}
}

// MARK: - Rotary Embeddings

/// Precompute rotary position embeddings (cos, sin) for the attention blocks.
private func precomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000.0) -> (MLXArray, MLXArray) {
    let halfDim = dim / 2
    let freqs = 1.0 / MLX.pow(
        MLXArray(theta),
        MLXArray(0 ..< halfDim).asType(.float32)[..<halfDim] / Float(dim)
    )
    let t = MLXArray(0 ..< end).asType(.float32)
    let outerProduct = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)

    let cosFreqs = MLX.cos(outerProduct) // (end, halfDim)
    let sinFreqs = MLX.sin(outerProduct)
    // Duplicate along last dim: (end, dim)
    let cosOut = MLX.concatenated([cosFreqs, cosFreqs], axis: -1)
    let sinOut = MLX.concatenated([sinFreqs, sinFreqs], axis: -1)
    return (cosOut, sinOut)
}

/// Apply rotary position embeddings to query and key tensors.
/// q, k shape: (B, T, nHead, headDim)
/// cos, sin shape: (maxLen, headDim)
private func applyRotaryEmb(
    q: MLXArray, k: MLXArray,
    cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    let T = q.dim(1)
    // Slice to sequence length and reshape for broadcasting: (1, T, 1, headDim)
    let cosSlice = cos[..<T].expandedDimensions(axes: [0, 2])
    let sinSlice = sin[..<T].expandedDimensions(axes: [0, 2])

    let D = q.dim(-1)
    let halfD = D / 2

    // Rotate: [-x_right, x_left]
    let qRotated = MLX.concatenated([-q[0..., 0..., 0..., halfD...], q[0..., 0..., 0..., ..<halfD]], axis: -1)
    let kRotated = MLX.concatenated([-k[0..., 0..., 0..., halfD...], k[0..., 0..., 0..., ..<halfD]], axis: -1)

    let qOut = q * cosSlice + qRotated * sinSlice
    let kOut = k * cosSlice + kRotated * sinSlice
    return (qOut, kOut)
}

// MARK: - FSQ Codebook

/// Finite Scalar Quantization codebook.
/// Projects hidden states to 8 dimensions, discretizes via round(tanh(x)),
/// then encodes 8 ternary digits into a single integer (codebook size = 3^8 = 6561).
class FSQCodebook: Module {
    @ModuleInfo(key: "project_down") var projectDown: Linear
    let level: Int

    init(dim: Int, level: Int = 3) {
        self.level = level
        self._projectDown.wrappedValue = Linear(dim, 8)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let xShape = x.shape // (B, T, D)
        let B = xShape[0], T = xShape[1]
        let flat = x.reshaped([-1, xShape[xShape.count - 1]])

        var h = projectDown(flat).asType(.float32)
        h = MLX.tanh(h)
        h = h * Float(0.9990000128746033)
        h = MLX.round(h) + 1 // Each dimension in {0, 1, 2}

        // Base-3 encoding: powers = [1, 3, 9, 27, 81, 243, 729, 2187]
        let nDims = 1 << level // 2^3 = 8
        let powers = MLXArray((0 ..< nDims).map { Float(pow(Float(level), Float($0))) })
        let mu = (h * powers.expandedDimensions(axis: 0)).sum(axis: -1)
        return mu.reshaped([B, T]).asType(.int32)
    }
}

/// FSQ Vector Quantization wrapper.
class FSQVectorQuantization: Module {
    @ModuleInfo(key: "fsq_codebook") var fsqCodebook: FSQCodebook
    let codebookSize: Int

    init(dim: Int, codebookSize: Int) {
        self.codebookSize = codebookSize
        self._fsqCodebook.wrappedValue = FSQCodebook(dim: dim, level: 3)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        return fsqCodebook.encode(x)
    }
}

// MARK: - FSMN Multi-Head Attention

/// Multi-head attention with Feedforward Sequential Memory Network (FSMN).
/// Adds a depthwise Conv1d memory path on the value tensor.
class FSMNMultiHeadAttention: Module {
    let nHead: Int
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "fsmn_block") var fsmnBlock: Conv1d

    let leftPadding: Int
    let rightPadding: Int

    init(nState: Int, nHead: Int, kernelSize: Int = 31) {
        self.nHead = nHead
        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
        self._fsmnBlock.wrappedValue = Conv1d(
            inputChannels: nState, outputChannels: nState,
            kernelSize: kernelSize, stride: 1, padding: 0,
            groups: nState, bias: false
        )
        self.leftPadding = (kernelSize - 1) / 2
        self.rightPadding = kernelSize - 1 - leftPadding
    }

    /// FSMN memory: depthwise conv1d on value tensor with residual connection.
    func forwardFSMN(_ inputs: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, t, n, d) = (inputs.dim(0), inputs.dim(1), inputs.dim(2), inputs.dim(3))
        var x = inputs.reshaped([b, t, n * d]) // (B, T, nState)

        if let mask = mask, mask.dim(2) > 0 {
            x = x * mask
        }

        // Manual padding (left and right)
        let padLeft = MLXArray.zeros([b, leftPadding, x.dim(2)])
        let padRight = MLXArray.zeros([b, rightPadding, x.dim(2)])
        let xPadded = MLX.concatenated([padLeft, x, padRight], axis: 1)

        var result = fsmnBlock(xPadded)
        result = result + x // residual

        if let mask = mask {
            result = result * mask
        }
        return result
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        maskPad: MLXArray? = nil,
        freqsCis: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray?) {
        let (B, T, D) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = D / nHead
        let scale = pow(Float(headDim), -0.25)

        var q = query(x) // (B, T, D)
        var k = key(x)
        var v = value(x)

        // Reshape to multi-head: (B, T, nHead, headDim)
        q = q.reshaped([B, T, nHead, headDim])
        k = k.reshaped([B, T, nHead, headDim])
        v = v.reshaped([B, T, nHead, headDim])

        // Apply rotary embeddings
        if let (cos, sin) = freqsCis {
            (q, k) = applyRotaryEmb(q: q, k: k, cos: cos, sin: sin)
        }

        // FSMN memory from value
        let fsmMemory = forwardFSMN(v, mask: maskPad)

        // Standard multi-head attention
        q = q.transposed(0, 2, 1, 3) * scale // (B, nHead, T, headDim)
        k = k.transposed(0, 2, 1, 3) * scale
        v = v.transposed(0, 2, 1, 3)

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask
        )
        let output = attnOutput.transposed(0, 2, 1, 3).reshaped([B, T, D])

        return (out(output) + fsmMemory, nil)
    }
}

// MARK: - Residual Attention Block

/// Transformer block with FSMN attention and MLP.
class S3ResidualAttentionBlock: Module {
    @ModuleInfo(key: "attn") var attn: FSMNMultiHeadAttention
    @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Sequential
    @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

    init(nState: Int, nHead: Int, kernelSize: Int = 31) {
        let nMlp = nState * 4
        self._attn.wrappedValue = FSMNMultiHeadAttention(nState: nState, nHead: nHead, kernelSize: kernelSize)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState, eps: 1e-6)
        self._mlp.wrappedValue = Sequential {
            Linear(nState, nMlp)
            GELU()
            Linear(nMlp, nState)
        }
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        maskPad: MLXArray? = nil,
        freqsCis: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let (attnOut, _) = attn(attnLn(x), mask: mask, maskPad: maskPad, freqsCis: freqsCis)
        let x2 = x + attnOut
        return x2 + mlp(mlpLn(x2))
    }
}

// MARK: - Audio Encoder V2

/// Audio encoder with two strided Conv1d layers and transformer blocks.
/// Total 4x downsampling: stride-2 conv1 + stride-2 conv2.
class AudioEncoderV2: Module {
    let stride: Int
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [S3ResidualAttentionBlock]

    let freqsCis: (MLXArray, MLXArray)

    init(nMels: Int, nState: Int, nHead: Int, nLayer: Int, stride: Int = 2) {
        self.stride = stride
        self._conv1.wrappedValue = Conv1d(
            inputChannels: nMels, outputChannels: nState,
            kernelSize: 3, stride: stride, padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: nState, outputChannels: nState,
            kernelSize: 3, stride: 2, padding: 1
        )
        self._blocks.wrappedValue = (0 ..< nLayer).map { _ in
            S3ResidualAttentionBlock(nState: nState, nHead: nHead, kernelSize: 31)
        }
        // Precompute rotary embeddings: dim = headDim = nState / nHead
        self.freqsCis = precomputeFreqsCis(dim: nState / nHead, end: 1024 * 2)
    }

    func callAsFunction(_ x: MLXArray, xLen: MLXArray) -> (MLXArray, MLXArray) {
        // x: (B, nMels, T) — transpose to (B, T, nMels) for Conv1d channels-last
        var out = x.transposed(0, 2, 1) // (B, T, nMels)
        var outLen = xLen

        // Mask and apply conv1
        // Conv1d length formula: (L + 2*pad - dilation*(kernel-1) - 1) / stride + 1
        // With kernel=3, stride=stride, pad=1, dilation=1: (L + 2 - 2 - 1) / stride + 1 = (L - 1) / stride + 1
        let mask1 = makeNonPadMask(outLen, maxLen: out.dim(1))
        out = out * mask1.expandedDimensions(axis: -1)
        out = gelu(conv1(out))
        let strideVal = Int32(stride)
        outLen = (outLen + MLXArray(Int32(0))) / strideVal + MLXArray(Int32(1))

        // Mask and apply conv2
        // Same formula with stride=2: (L - 1) / 2 + 1
        let mask2 = makeNonPadMask(outLen, maxLen: out.dim(1))
        out = out * mask2.expandedDimensions(axis: -1)
        out = gelu(conv2(out))
        outLen = (outLen + MLXArray(Int32(0))) / Int32(2) + MLXArray(Int32(1))

        // Attention mask
        let mask3 = makeNonPadMask(outLen, maxLen: out.dim(1))
        let maskPad = mask3.expandedDimensions(axis: -1) // (B, T, 1)
        let maskBias = maskToBias(mask3, dtype: out.dtype)
            .expandedDimensions(axis: 1) // (B, 1, T)

        for block in blocks {
            out = block(out, mask: maskBias, maskPad: maskPad, freqsCis: freqsCis)
        }

        return (out, outLen)
    }
}

// MARK: - S3TokenizerV2

/// S3TokenizerV2: Converts mel spectrograms → discrete speech token IDs.
///
/// Architecture: AudioEncoderV2 (Conv1d + Transformer with FSMN + RoPE) → FSQ quantizer.
/// Input: mel spectrogram (B, 128, T) at 100 frames/sec (from 16kHz audio, hop=160)
/// Output: token IDs (B, T') at 25 tokens/sec (4x downsampling), vocabulary size 6561.
public class S3TokenizerV2: Module {
    let config: S3TokenizerConfig
    @ModuleInfo(key: "encoder") var encoder: AudioEncoderV2
    @ModuleInfo(key: "quantizer") var quantizer: FSQVectorQuantization

    public init(config: S3TokenizerConfig = S3TokenizerConfig()) {
        self.config = config
        self._encoder.wrappedValue = AudioEncoderV2(
            nMels: config.nMels,
            nState: config.nAudioState,
            nHead: config.nAudioHead,
            nLayer: config.nAudioLayer,
            stride: 2
        )
        self._quantizer.wrappedValue = FSQVectorQuantization(
            dim: config.nAudioState,
            codebookSize: config.nCodebookSize
        )
    }

    /// Tokenize mel spectrogram to speech tokens.
    /// - Parameters:
    ///   - mel: Mel spectrogram (B, nMels, T)
    ///   - melLen: Length of each mel in the batch (B,)
    /// - Returns: (tokens, tokenLens) where tokens is (B, T') int32, tokenLens is (B,)
    public func callAsFunction(_ mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
        let (hidden, codeLen) = encoder(mel, xLen: melLen)
        let code = quantizer.encode(hidden)
        return (code, codeLen)
    }

    /// Weight sanitization for loading from safetensors.
    ///
    /// Handles two weight formats:
    /// - **MLX-community** (pre-converted): Conv1d weights already in MLX layout (outCh, kernel, inCh)
    /// - **PyTorch raw**: Conv1d weights in PyTorch layout (outCh, inCh, kernel) — needs swapAxes(1,2)
    ///
    /// Detects format by comparing weight shapes against the model's expected parameter shapes.
    public static func sanitize(weights: [String: MLXArray], model: S3TokenizerV2) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        // Flatten model parameters to a dict of "a.b.c.weight" → MLXArray for shape comparison
        let flatParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        for (key, var value) in weights {
            var newKey = key

            // Skip precomputed arrays
            if newKey.contains("freqs_cis") || newKey.contains("_mel_filters") { continue }
            if newKey.hasPrefix("onnx::") { continue }

            // Rename quantizer codebook
            newKey = newKey.replacingOccurrences(of: "quantizer._codebook.", with: "quantizer.fsq_codebook.")
            newKey = newKey.replacingOccurrences(of: "quantizer.codebook.", with: "quantizer.fsq_codebook.")

            // Sequential MLP indexing: mlp.0 -> mlp.layers.0
            if let range = newKey.range(of: #"\.mlp\.(\d+)\."#, options: .regularExpression) {
                let match = String(newKey[range])
                let replacement = match
                    .replacingOccurrences(of: ".mlp.", with: ".mlp.layers.")
                newKey = newKey.replacingCharacters(in: range, with: replacement)
            }

            // Conv weight transposition: Conv1d (3D) may need axes 1,2 swapped.
            // MLX Conv1d expects (outCh, kernelSize, inCh).
            // PyTorch Conv1d stores (outCh, inCh, kernelSize).
            // Only transpose if the weight is NOT already in MLX format.
            // Detect by flattening model parameters and comparing shapes.
            if newKey.contains(".conv1.") || newKey.contains(".conv2.") || newKey.contains(".fsmn_block.") {
                if newKey.hasSuffix(".weight") && value.ndim == 3 {
                    if let expectedWeight = flatParams[newKey] {
                        // Only transpose if shapes don't match (PyTorch format)
                        if value.shape != expectedWeight.shape {
                            value = value.swappedAxes(1, 2)
                        }
                    }
                    // If no model reference found, leave weight as-is (assume MLX format)
                }
            }

            newWeights[newKey] = value
        }
        return newWeights
    }
}

// MARK: - Helper: Log Mel Spectrogram for S3TokenizerV2

/// Compute log mel spectrogram for S3TokenizerV2 input.
/// Uses Whisper-style normalization: log10, clamp to max-8, then (x+4)/4.
///
/// - Parameters:
///   - audio: 1D audio waveform at 16kHz
///   - sampleRate: Sample rate (default 16000)
///   - nMels: Number of mel bins (default 128)
///   - nFft: FFT size (default 400)
///   - hopLength: Hop length (default 160)
/// - Returns: Mel spectrogram (nMels, T) — note: NOT batched
public func s3TokenizerLogMelSpectrogram(
    _ audio: MLXArray,
    sampleRate: Int = 16000,
    nMels: Int = 128,
    nFft: Int = 400,
    hopLength: Int = 160
) -> MLXArray {
    let window = hanningWindow(size: nFft)

    // STFT (center=true, matching Python default)
    let freqs = stft(audio: audio, window: window, nFft: nFft, hopLength: hopLength)
    // freqs shape: (T', nFft/2+1)

    // Power spectrum — drop last frame to match Python's `spec[:, :-1, :]` behavior
    // Python drops the last STFT frame to match PyTorch torch.stft convention
    let numFrames = freqs.dim(0)
    let magnitudes = abs(freqs[..<(numFrames - 1)]).square()

    // Mel filterbank: (nFreqs, nMels)
    let filters = melFilters(
        sampleRate: sampleRate, nFft: nFft, nMels: nMels,
        norm: "slaney", melScale: .slaney
    )

    // Apply: (T', nFreqs) @ (nFreqs, nMels) -> (T', nMels)
    // Then we need (nMels, T') for the S3Tokenizer convention.
    // But Python does filters @ magnitudes where filters is (nMels, nFreqs) and magnitudes is (nFreqs, T')
    // Our melFilters returns (nFreqs, nMels), so: magnitudes.T @ melFilters = (T', nFreqs) @ (nFreqs, nMels) = (T', nMels)
    // Then transpose to (nMels, T')
    // Actually Python does: filters @ magnitudes where filters=(nMels, nFreqs), magnitudes=(nFreqs, T')
    // Result is (nMels, T'). Python's stft returns (nFreqs, T') orientation.
    // Our stft returns (T', nFreqs). So: melFilters.T @ magnitudes.T = (nMels, nFreqs) @ (nFreqs, T') doesn't work directly.
    // Simplest: magnitudes @ melFilters = (T', nFreqs) @ (nFreqs, nMels) = (T', nMels)
    // Then transpose.

    let melSpec = matmul(magnitudes, filters) // (T', nMels)
    var logSpec = MLX.log10(MLX.maximum(melSpec, MLXArray(Float(1e-10))))
    let maxVal = logSpec.max()
    logSpec = MLX.maximum(logSpec, maxVal - 8.0)
    logSpec = (logSpec + 4.0) / 4.0

    // Transpose to (nMels, T') to match Python convention
    return logSpec.transposed()
}

// MARK: - Mask Utilities

/// Create non-pad mask: 1 for valid positions, 0 for padding.
/// - Parameters:
///   - lengths: (B,) tensor of sequence lengths
///   - maxLen: Maximum length (0 = use max from lengths)
/// - Returns: Boolean mask (B, maxLen)
func makeNonPadMask(_ lengths: MLXArray, maxLen: Int = 0) -> MLXArray {
    let maxLength: Int
    if maxLen > 0 {
        maxLength = maxLen
    } else {
        eval(lengths)
        maxLength = lengths.max().item(Int.self)
    }

    let seqRange = MLXArray(0 ..< maxLength).asType(.int32)
    let seqRangeExpanded = seqRange.expandedDimensions(axis: 0) // (1, maxLen)
    let lengthsExpanded = lengths.expandedDimensions(axis: -1) // (B, 1)
    return seqRangeExpanded .< lengthsExpanded // (B, maxLen)
}

/// Convert boolean mask to attention bias (0 for valid, -1e10 for masked).
func maskToBias(_ mask: MLXArray, dtype: DType = .float32) -> MLXArray {
    let floatMask = mask.asType(dtype)
    return (1.0 - floatMask) * Float(-1.0e10)
}
