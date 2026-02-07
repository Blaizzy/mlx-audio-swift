//
//  Qwen3ASR.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 06/02/2026.
//

import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace
import Tokenizers

// MARK: - Helper Functions

private func floorDiv(_ a: MLXArray, _ b: Int) -> MLXArray {
    return floor(a.asType(.float32) / Float(b)).asType(.int32)
}

func getFeatExtractOutputLengths(_ inputLengths: MLXArray) -> MLXArray {
    let inputLengthsLeave = inputLengths % 100
    let featLengths = floorDiv(inputLengthsLeave - 1, 2) + 1
    let outputLengths = (
        floorDiv(floorDiv(featLengths - 1, 2) + 1 - 1, 2)
        + 1
        + (inputLengths / 100) * 13
    )
    return outputLengths
}

// MARK: - Sinusoidal Position Embedding

class Qwen3ASRSinusoidalPE: Module {
    let _positionalEmbedding: MLXArray

    init(length: Int, channels: Int, maxTimescale: Float = 10000.0) {
        precondition(channels % 2 == 0, "SinusoidalPE channels must be even")

        let logTimescaleIncrement = log(maxTimescale) / Float(channels / 2 - 1)
        let invTimescales = MLX.exp(
            -logTimescaleIncrement * MLXArray(0..<(channels / 2)).asType(.float32)
        )
        let positions = MLXArray(0..<length).asType(.float32).reshaped(-1, 1)
        let scaledTime = positions * invTimescales.reshaped(1, -1)
        self._positionalEmbedding = MLX.concatenated(
            [MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1
        )
        super.init()
    }

    func callAsFunction(_ seqLen: Int) -> MLXArray {
        return _positionalEmbedding[0..<seqLen]
    }
}

// MARK: - Audio Encoder Attention

class Qwen3ASRAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen3AudioEncoderConfig) {
        self.embedDim = config.dModel
        self.numHeads = config.encoderAttentionHeads
        self.headDim = embedDim / numHeads
        self.scaling = pow(Float(headDim), -0.5)

        precondition(headDim * numHeads == embedDim,
            "embed_dim must be divisible by num_heads")

        self._qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._kProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask != nil ? .array(mask!) : .none
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scaling,
            mask: maskMode
        )

        let output = attnOutput.transposed(0, 2, 1, 3).reshaped(B, L, embedDim)
        return outProj(output)
    }
}

// MARK: - Audio Encoder Layer

class Qwen3ASRAudioEncoderLayer: Module {
    let embedDim: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3ASRAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: Qwen3AudioEncoderConfig) {
        self.embedDim = config.dModel

        self._selfAttn.wrappedValue = Qwen3ASRAttention(config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
        self._fc1.wrappedValue = Linear(embedDim, config.encoderFfnDim)
        self._fc2.wrappedValue = Linear(config.encoderFfnDim, embedDim)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Pre-norm attention
        var residual = hiddenStates
        var h = selfAttnLayerNorm(hiddenStates)
        h = selfAttn(h, mask: mask)
        h = residual + h

        // Pre-norm FFN
        residual = h
        h = finalLayerNorm(h)
        h = gelu(fc1(h))
        h = fc2(h)
        h = residual + h

        return h
    }
}

// MARK: - Audio Encoder

public class Qwen3ASRAudioEncoder: Module {
    let config: Qwen3AudioEncoderConfig
    let nWindow: Int
    let nWindowInfer: Int

    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen3ASRAudioEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    let positionalEmbedding: Qwen3ASRSinusoidalPE

    public init(_ config: Qwen3AudioEncoderConfig) {
        self.config = config
        let embedDim = config.dModel
        self.nWindow = config.nWindow
        self.nWindowInfer = config.nWindowInfer

        // Conv2d frontend: input is [batch, mel_bins, time, 1]
        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        // Frequency dimension after 3 conv layers with stride 2
        let freqAfterConv = ((((config.numMelBins + 1) / 2) + 1) / 2 + 1) / 2
        self._convOut.wrappedValue = Linear(
            config.downsampleHiddenSize * freqAfterConv, embedDim, bias: false
        )

        self.positionalEmbedding = Qwen3ASRSinusoidalPE(
            length: config.maxSourcePositions, channels: embedDim
        )

        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            Qwen3ASRAudioEncoderLayer(config)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: embedDim)
        self._proj1.wrappedValue = Linear(embedDim, embedDim)
        self._proj2.wrappedValue = Linear(embedDim, config.outputDim)
    }

    private func createBlockAttentionMask(
        seqLen: Int, cuSeqlens: [Int], dtype: DType
    ) -> MLXArray {
        var maskValues = [Float](repeating: -1e9, count: seqLen * seqLen)
        for i in 0..<(cuSeqlens.count - 1) {
            let start = cuSeqlens[i]
            let end = min(cuSeqlens[i + 1], seqLen)
            for r in start..<end {
                for c in start..<end {
                    maskValues[r * seqLen + c] = 0.0
                }
            }
        }
        return MLXArray(maskValues).reshaped(seqLen, seqLen).asType(dtype)
    }

    public func callAsFunction(
        _ inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        // inputFeatures shape: [batch, n_mels, n_frames]
        let batchSize = inputFeatures.dim(0)
        let nFrames = inputFeatures.dim(2)

        // Determine feature lengths
        let featureLens: [Int]
        if let mask = featureAttentionMask {
            let lens = mask.sum(axis: -1).asType(.int32)
            featureLens = (0..<batchSize).map { Int(lens[$0].item(Int32.self)) }
        } else {
            featureLens = [Int](repeating: nFrames, count: batchSize)
        }

        let featureLensArray = MLXArray(featureLens.map { Int32($0) })
        let aftercnnLens = getFeatExtractOutputLengths(featureLensArray)
        let chunkSize = nWindow * 2

        // Split features into chunks
        var chunkLengths: [Int] = []
        var chunks: [MLXArray] = []

        for i in 0..<batchSize {
            let featLen = featureLens[i]
            let numChunks = Int(ceil(Double(featLen) / Double(chunkSize)))
            let feat = inputFeatures[i]  // [n_mels, n_frames]

            var pos = 0
            for j in 0..<numChunks {
                let clen: Int
                if j == numChunks - 1 {
                    let remainder = featLen % chunkSize
                    clen = remainder == 0 ? chunkSize : remainder
                } else {
                    clen = chunkSize
                }
                let chunk = feat[0..., pos..<(pos + clen)]  // [n_mels, clen]
                chunks.append(chunk)
                chunkLengths.append(clen)
                pos += clen
            }
        }

        let maxChunkLen = chunkLengths.max() ?? 0

        // Pad chunks to max length
        var paddedChunks: [MLXArray] = []
        for (idx, chunk) in chunks.enumerated() {
            let clen = chunkLengths[idx]
            if clen < maxChunkLen {
                let padWidth = maxChunkLen - clen
                let padded = MLX.padded(chunk, widths: [IntOrPair((0, 0)), IntOrPair((0, padWidth))])
                paddedChunks.append(padded)
            } else {
                paddedChunks.append(chunk)
            }
        }

        let paddedFeature = MLX.stacked(paddedChunks, axis: 0)  // [numChunks, n_mels, maxChunkLen]

        // Compute output lengths after CNN for each chunk
        let chunkLensArray = MLXArray(chunkLengths.map { Int32($0) })
        let featureLensAfterCnn = getFeatExtractOutputLengths(chunkLensArray)
        let featureLensAfterCnnValues = (0..<chunkLengths.count).map {
            Int(featureLensAfterCnn[$0].item(Int32.self))
        }
        let maxLenAfterCnn = featureLensAfterCnnValues.max() ?? 0

        // Apply Conv2d layers: input [numChunks, n_mels, maxChunkLen, 1]
        var x = paddedFeature.expandedDimensions(axis: -1)  // Add channel dim
        x = gelu(conv2d1(x))
        x = gelu(conv2d2(x))
        x = gelu(conv2d3(x))

        // Reshape: [b, f, t, c] -> [b, t, c*f]
        let b = x.dim(0)
        let f = x.dim(1)
        let t = x.dim(2)
        let c = x.dim(3)
        x = x.transposed(0, 2, 3, 1).reshaped(b, t, c * f)
        x = convOut(x)  // [b, t, d_model]

        // Add positional embeddings
        let posEmb = positionalEmbedding(x.dim(1))
        x = x + posEmb.expandedDimensions(axis: 0)

        // Extract valid-length hidden states and concatenate
        var hiddenList: [MLXArray] = []
        for i in 0..<x.dim(0) {
            let validLen = featureLensAfterCnnValues[i]
            hiddenList.append(x[i, 0..<validLen])
        }
        var hiddenStates = MLX.concatenated(hiddenList, axis: 0)  // [totalValidLen, d_model]

        // Build block attention mask
        let aftercnnLensValues = (0..<batchSize).map {
            Int(aftercnnLens[$0].item(Int32.self))
        }
        let windowAftercnn = maxLenAfterCnn * (nWindowInfer / (nWindow * 2))

        var cuChunkLens: [Int] = [0]
        for cnnLen in aftercnnLensValues {
            let numFullWindows = cnnLen / windowAftercnn
            for _ in 0..<numFullWindows {
                cuChunkLens.append(windowAftercnn)
            }
            let remainder = cnnLen % windowAftercnn
            if remainder != 0 {
                cuChunkLens.append(remainder)
            }
        }

        var cuSeqlens: [Int] = []
        var cumSum = 0
        for len in cuChunkLens {
            cumSum += len
            cuSeqlens.append(cumSum)
        }

        let seqLen = hiddenStates.dim(0)
        var attentionMask = createBlockAttentionMask(
            seqLen: seqLen, cuSeqlens: cuSeqlens, dtype: hiddenStates.dtype
        )
        // [1, 1, seqLen, seqLen]
        attentionMask = attentionMask.expandedDimensions(axes: [0, 1])

        // [1, seqLen, d_model]
        hiddenStates = hiddenStates.expandedDimensions(axis: 0)

        // Apply transformer layers
        for layer in layers {
            hiddenStates = layer(hiddenStates, mask: attentionMask)
        }

        // Post-processing
        hiddenStates = hiddenStates[0]  // Remove batch dim
        hiddenStates = lnPost(hiddenStates)
        hiddenStates = gelu(proj1(hiddenStates))
        hiddenStates = proj2(hiddenStates)

        return hiddenStates  // [seqLen, outputDim]
    }
}

// MARK: - Text Decoder Attention

class Qwen3ASRTextAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ config: Qwen3TextConfig, layerIdx: Int) {
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(B, L, numHeads, headDim)
        keys = keys.reshaped(B, L, numKvHeads, headDim)
        values = values.reshaped(B, L, numKvHeads, headDim)

        // Apply Q/K normalization before transpose
        queries = qNorm(queries)
        keys = kNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE
        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Text Decoder MLP

class Qwen3ASRTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Qwen3TextConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Text Decoder Layer

class Qwen3ASRTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3ASRTextAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3ASRTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: Qwen3TextConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = Qwen3ASRTextAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Qwen3ASRTextMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var residual = hiddenStates
        var h = inputLayernorm(hiddenStates)
        h = selfAttn(h, mask: mask, cache: cache)
        h = residual + h

        residual = h
        h = postAttentionLayernorm(h)
        h = mlp(h)
        h = residual + h

        return h
    }
}

// MARK: - Text Model

public class Qwen3ASRTextModel: Module {
    let config: Qwen3TextConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3ASRTextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: Qwen3TextConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { i in
            Qwen3ASRTextDecoderLayer(config, layerIdx: i)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let embeds = inputsEmbeds {
            h = embeds
        } else if let ids = inputIds {
            h = embedTokens(ids)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let mask = createAttentionMask(h: h, cache: cache?.first)

        let caches = cache ?? [KVCache?](repeating: nil, count: layers.count)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches[i])
        }

        return norm(h)
    }
}

// MARK: - Qwen3 ASR Model

public class Qwen3ASRModel: Module {
    public let config: Qwen3ASRConfig

    @ModuleInfo(key: "audio_tower") var audioTower: Qwen3ASRAudioEncoder
    @ModuleInfo(key: "model") var model: Qwen3ASRTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var tokenizer: Tokenizer?

    public init(_ config: Qwen3ASRConfig) {
        self.config = config

        self._audioTower.wrappedValue = Qwen3ASRAudioEncoder(config.audioConfig)
        self._model.wrappedValue = Qwen3ASRTextModel(config.textConfig)

        if config.textConfig.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                config.textConfig.hiddenSize,
                config.textConfig.vocabSize,
                bias: false
            )
        }
    }

    // MARK: - Audio Features

    public func getAudioFeatures(
        _ inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        return audioTower(inputFeatures, featureAttentionMask: featureAttentionMask)
    }

    // MARK: - Forward Pass

    public func callAsFunction(
        inputIds: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        inputFeatures: MLXArray? = nil,
        featureAttentionMask: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var inputsEmbeds: MLXArray
        if let embeddings = inputEmbeddings {
            inputsEmbeds = embeddings
        } else {
            inputsEmbeds = model.embedTokens(inputIds)
        }

        // Encode and merge audio features on first pass
        if let features = inputFeatures,
           cache == nil || cache?.first == nil || (cache?.first as? KVCacheSimple)?.offset == 0 {
            let audioFeatures = getAudioFeatures(features, featureAttentionMask: featureAttentionMask)
                .asType(inputsEmbeds.dtype)

            inputsEmbeds = mergeAudioFeatures(
                inputsEmbeds: inputsEmbeds,
                audioFeatures: audioFeatures,
                inputIds: inputIds
            )
        }

        let hiddenStates = model(inputsEmbeds: inputsEmbeds, cache: cache)

        if let lmHead = lmHead {
            return lmHead(hiddenStates)
        } else {
            return model.embedTokens.asLinear(hiddenStates)
        }
    }

    // MARK: - Audio-Text Merging

    private func mergeAudioFeatures(
        inputsEmbeds: MLXArray,
        audioFeatures: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let audioTokenMask = inputIds .== MLXArray(Int32(config.audioTokenId))

        // Find audio token positions
        let flatMask = audioTokenMask.reshaped(-1)
        let batchSize = inputsEmbeds.dim(0)
        let seqLen = inputsEmbeds.dim(1)
        let hiddenDim = inputsEmbeds.dim(2)

        // Count audio tokens
        let numAudioTokens = Int(flatMask.asType(.int32).sum().item(Int32.self))
        guard numAudioTokens > 0 && audioFeatures.dim(0) > 0 else {
            return inputsEmbeds
        }

        let numToReplace = min(numAudioTokens, audioFeatures.dim(0))
        let flatEmbeds = inputsEmbeds.reshaped(-1, hiddenDim)

        // Build indices for replacement
        var resultList: [MLXArray] = []
        var audioIdx = 0
        let totalLen = flatEmbeds.dim(0)

        for i in 0..<totalLen {
            let isAudioToken = Int(flatMask[i].item(Int32.self)) != 0
            if audioIdx < numToReplace && isAudioToken {
                resultList.append(audioFeatures[audioIdx])
                audioIdx += 1
            } else {
                resultList.append(flatEmbeds[i])
            }
        }

        return MLX.stacked(resultList, axis: 0).reshaped(batchSize, seqLen, hiddenDim)
    }

    // MARK: - Audio Preprocessing

    public func preprocessAudio(_ audio: MLXArray) -> (MLXArray, MLXArray, Int) {
        // Compute mel spectrogram
        let melSpec = MLXAudioCore.computeMelSpectrogram(
            audio: audio,
            sampleRate: 16000,
            nFft: 400,
            hopLength: 160,
            nMels: config.audioConfig.numMelBins
        )

        // melSpec shape: [numFrames, nMels] -> need [1, nMels, numFrames]
        let transposed = melSpec.transposed(1, 0)
        let inputFeatures = transposed.expandedDimensions(axis: 0)

        // Create attention mask (all ones for single audio)
        let numFrames = melSpec.dim(0)
        let featureAttentionMask = MLX.ones([1, numFrames]).asType(.int32)

        // Compute number of audio tokens after CNN
        let audioLengths = featureAttentionMask.sum(axis: -1).asType(.int32)
        let aftercnnLens = getFeatExtractOutputLengths(audioLengths)
        let numAudioTokens = Int(aftercnnLens[0].item(Int32.self))

        return (inputFeatures, featureAttentionMask, numAudioTokens)
    }

    // MARK: - Prompt Building

    public func buildPrompt(numAudioTokens: Int, language: String = "English") -> MLXArray {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded")
        }

        let supported = config.supportLanguages
        let supportedLower = Dictionary(uniqueKeysWithValues: supported.map { ($0.lowercased(), $0) })
        let langName = supportedLower[language.lowercased()] ?? language

        let prompt = "<|im_start|>system\n<|im_end|>\n"
            + "<|im_start|>user\n<|audio_start|>"
            + String(repeating: "<|audio_pad|>", count: numAudioTokens)
            + "<|audio_end|><|im_end|>\n"
            + "<|im_start|>assistant\nlanguage \(langName)<asr_text>"

        let tokenIds = tokenizer.encode(text: prompt)
        return MLXArray(tokenIds.map { Int32($0) }).expandedDimensions(axis: 0)
    }

    // MARK: - Cache Creation

    public func makeCache() -> [KVCache] {
        return (0..<config.textConfig.numHiddenLayers).map { _ in
            KVCacheSimple()
        }
    }

    // MARK: - Generation

    public func generate(
        audio: MLXArray,
        maxTokens: Int = 8192,
        temperature: Float = 0.0,
        language: String = "English"
    ) -> STTOutput {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded")
        }

        let startTime = Date()
        let eosTokenIds = [151645, 151643]

        // Preprocess audio
        let (inputFeatures, featureAttentionMask, numAudioTokens) = preprocessAudio(audio)

        // Build prompt
        let inputIds = buildPrompt(numAudioTokens: numAudioTokens, language: language)
        let promptTokenCount = inputIds.dim(1)

        // Encode audio
        let audioFeatures = getAudioFeatures(inputFeatures, featureAttentionMask: featureAttentionMask)
        eval(audioFeatures)

        // Build input embeddings (use embeds.dtype to get correct float type even for quantized models)
        let embeds = model.embedTokens(inputIds)
        let inputsEmbeds = mergeAudioFeatures(
            inputsEmbeds: embeds,
            audioFeatures: audioFeatures.asType(embeds.dtype),
            inputIds: inputIds
        )

        // Create cache and run prefill
        let cache = makeCache()
        var logits = callAsFunction(
            inputIds: inputIds,
            inputEmbeddings: inputsEmbeds,
            cache: cache
        )
        eval(logits)

        let prefillTime = Date()

        // Generate tokens
        var generatedTokens: [Int] = []

        for _ in 0..<maxTokens {
            var lastLogits = logits[0..., -1, 0...]
            if temperature > 0 {
                lastLogits = lastLogits / temperature
            }
            let nextToken = lastLogits.argMax(axis: -1).item(Int.self)

            if eosTokenIds.contains(nextToken) {
                break
            }

            generatedTokens.append(nextToken)

            // Step forward
            let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            logits = callAsFunction(inputIds: nextTokenArray, cache: cache)
            eval(logits)
        }

        let endTime = Date()
        Memory.clearCache()

        let text = tokenizer.decode(tokens: generatedTokens)
        let totalTime = endTime.timeIntervalSince(startTime)
        let prefillDuration = prefillTime.timeIntervalSince(startTime)
        let generateTime = endTime.timeIntervalSince(prefillTime)

        return STTOutput(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            promptTokens: promptTokenCount,
            generationTokens: generatedTokens.count,
            totalTokens: promptTokenCount + generatedTokens.count,
            promptTps: Double(promptTokenCount) / max(prefillDuration, 0.001),
            generationTps: Double(generatedTokens.count) / max(generateTime, 0.001),
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    /// Generate transcription with streaming output.
    public func generateStream(
        audio: MLXArray,
        maxTokens: Int = 8192,
        temperature: Float = 0.0,
        language: String = "English"
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            do {
                guard let tokenizer = self.tokenizer else {
                    throw STTError.modelNotInitialized("Tokenizer not loaded")
                }

                let startTime = Date()
                let eosTokenIds = [151645, 151643]

                // Preprocess audio
                let (inputFeatures, featureAttentionMask, numAudioTokens) = self.preprocessAudio(audio)

                // Build prompt
                let inputIds = self.buildPrompt(numAudioTokens: numAudioTokens, language: language)
                let promptTokenCount = inputIds.dim(1)

                // Encode audio
                let audioFeatures = self.getAudioFeatures(
                    inputFeatures, featureAttentionMask: featureAttentionMask
                )
                eval(audioFeatures)

                // Build input embeddings (use embeds.dtype to get correct float type even for quantized models)
                let embeds = self.model.embedTokens(inputIds)
                let inputsEmbeds = self.mergeAudioFeatures(
                    inputsEmbeds: embeds,
                    audioFeatures: audioFeatures.asType(embeds.dtype),
                    inputIds: inputIds
                )

                // Create cache and prefill
                let cache = self.makeCache()
                var logits = self.callAsFunction(
                    inputIds: inputIds,
                    inputEmbeddings: inputsEmbeds,
                    cache: cache
                )
                eval(logits)

                let prefillEndTime = Date()
                let prefillTime = prefillEndTime.timeIntervalSince(startTime)

                // Generate tokens
                var generatedTokens: [Int] = []

                for _ in 0..<maxTokens {
                    var lastLogits = logits[0..., -1, 0...]
                    if temperature > 0 {
                        lastLogits = lastLogits / temperature
                    }
                    let nextToken = lastLogits.argMax(axis: -1).item(Int.self)

                    if eosTokenIds.contains(nextToken) {
                        break
                    }

                    generatedTokens.append(nextToken)

                    // Emit token
                    let tokenText = tokenizer.decode(tokens: [nextToken])
                    continuation.yield(.token(tokenText))

                    // Step forward
                    let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
                    logits = self.callAsFunction(inputIds: nextTokenArray, cache: cache)
                    eval(logits)
                }

                let endTime = Date()
                let generateTime = endTime.timeIntervalSince(prefillEndTime)
                let totalTime = endTime.timeIntervalSince(startTime)

                Memory.clearCache()

                // Emit generation info
                let tokensPerSecond = generateTime > 0 ? Double(generatedTokens.count) / generateTime : 0
                let peakMemory = Double(Memory.peakMemory) / 1e9
                let info = STTGenerationInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generatedTokens.count,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: tokensPerSecond,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.info(info))

                // Emit final result
                let text = tokenizer.decode(tokens: generatedTokens)
                let output = STTOutput(
                    text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                    promptTokens: promptTokenCount,
                    generationTokens: generatedTokens.count,
                    totalTokens: promptTokenCount + generatedTokens.count,
                    promptTps: Double(promptTokenCount) / max(prefillTime, 0.001),
                    generationTps: tokensPerSecond,
                    totalTime: totalTime,
                    peakMemoryUsage: peakMemory
                )
                continuation.yield(.result(output))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray], skipLmHead: Bool = true) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        let isFormatted = !weights.keys.contains { $0.hasPrefix("thinker.") }

        for (key, var value) in weights {
            var newKey = key

            // Strip thinker prefix
            if newKey.hasPrefix("thinker.") {
                newKey = String(newKey.dropFirst("thinker.".count))
            }

            // Skip lm_head for ASR (tied to embeddings)
            if skipLmHead && newKey == "lm_head.weight" {
                continue
            }

            // Transpose Conv2d weights from PyTorch format
            if !isFormatted && newKey.contains("conv2d") && newKey.contains("weight") && value.ndim == 4 {
                value = value.transposed(0, 2, 3, 1)
            }

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - Tokenizer JSON Generation

    /// Generate `tokenizer.json` from `vocab.json` + `merges.txt` + `tokenizer_config.json`
    static func generateTokenizerJSONIfMissing(in modelDir: URL) throws {
        let tokenizerJSONPath = modelDir.appendingPathComponent("tokenizer.json")
        guard !FileManager.default.fileExists(atPath: tokenizerJSONPath.path) else { return }

        let vocabURL = modelDir.appendingPathComponent("vocab.json")
        let mergesURL = modelDir.appendingPathComponent("merges.txt")
        let tokenizerConfigURL = modelDir.appendingPathComponent("tokenizer_config.json")

        guard FileManager.default.fileExists(atPath: vocabURL.path),
              FileManager.default.fileExists(atPath: mergesURL.path) else {
            return  // Can't generate without vocab + merges
        }

        // Read vocab.json as raw JSON
        let vocabData = try Data(contentsOf: vocabURL)

        // Read merges.txt, skip header line
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let mergeLines = mergesText.components(separatedBy: "\n")
            .filter { !$0.hasPrefix("#") && !$0.isEmpty }

        // Build merges JSON array (legacy string format: "token1 token2")
        let mergesJSON = mergeLines.map { line -> String in
            let escaped = line
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "\"\(escaped)\""
        }.joined(separator: ",")

        // Read added_tokens_decoder from tokenizer_config.json
        var addedTokensJSON = "[]"
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            let configData = try Data(contentsOf: tokenizerConfigURL)
            if let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: Any] {
                var tokens: [(Int, [String: Any])] = []
                for (idStr, value) in addedTokensDecoder {
                    if let id = Int(idStr), let tokenDict = value as? [String: Any] {
                        let entry: [String: Any] = [
                            "id": id,
                            "content": tokenDict["content"] ?? "",
                            "single_word": tokenDict["single_word"] ?? false,
                            "lstrip": tokenDict["lstrip"] ?? false,
                            "rstrip": tokenDict["rstrip"] ?? false,
                            "normalized": tokenDict["normalized"] ?? false,
                            "special": tokenDict["special"] ?? false,
                        ]
                        tokens.append((id, entry))
                    }
                }
                tokens.sort { $0.0 < $1.0 }
                let tokenData = try JSONSerialization.data(
                    withJSONObject: tokens.map { $0.1 }, options: [])
                addedTokensJSON = String(data: tokenData, encoding: .utf8) ?? "[]"
            }
        }

        // Qwen2 BPE pre-tokenizer pattern
        let preTokenizerPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

        // Escape for JSON embedding
        let escapedPattern = preTokenizerPattern
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")

        let vocabString = String(data: vocabData, encoding: .utf8) ?? "{}"

        // Construct tokenizer.json
        let tokenizerJSON = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": \(addedTokensJSON),
          "normalizer": {"type": "NFC"},
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {
                "type": "Split",
                "pattern": {"Regex": "\(escapedPattern)"},
                "behavior": "Isolated",
                "invert": false
              },
              {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false
              }
            ]
          },
          "post_processor": null,
          "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
          },
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": \(vocabString),
            "merges": [\(mergesJSON)]
          }
        }
        """

        try tokenizerJSON.write(to: tokenizerJSONPath, atomically: true, encoding: .utf8)
        print("Generated tokenizer.json at: \(tokenizerJSONPath.path)")
    }

    // MARK: - Model Loading

    public static func fromPretrained(_ modelPath: String) async throws -> Qwen3ASRModel {
        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "Qwen3ASRModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await resolveOrDownloadModel(client: client, cache: cache, repoID: repoID)

        // Load config
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: configData)

        // Get per-layer quantization
        let perLayerQuantization = config.perLayerQuantization

        // Create model
        let model = Qwen3ASRModel(config)

        // Generate tokenizer.json if missing (Qwen3 ASR models don't ship it)
        try generateTokenizerJSONIfMissing(in: modelDir)

        // Load tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        // Load weights
        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        // Sanitize weights
        let skipLmHead = config.textConfig.tieWordEmbeddings
        let sanitizedWeights = Qwen3ASRModel.sanitize(weights: weights, skipLmHead: skipLmHead)

        // Quantize if needed
        if perLayerQuantization != nil {
            quantize(model: model) { path, module in
                // Don't quantize audio tower
                if path.hasPrefix("audio_tower") {
                    return nil
                }
                // Check if scales exist for this layer in sanitized weights
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        // Load weights into model
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])
        eval(model)

        return model
    }

    static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache,
        repoID: Repo.ID
    ) async throws -> URL {
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        let configPath = modelDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configPath.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let hasSafetensors = files?.contains { $0.pathExtension == "safetensors" } ?? false
            let hasTokenizer = FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("tokenizer.json").path)
            let hasVocab = FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("vocab.json").path)

            if hasSafetensors && (hasTokenizer || hasVocab) {
                print("Using cached model at: \(modelDir.path)")
                return modelDir
            }
        }

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        print("Downloading model \(repoID)...")
        _ = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDir,
            revision: "main",
            progressHandler: { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }
}
