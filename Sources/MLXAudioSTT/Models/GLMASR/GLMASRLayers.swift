//
//  GLMASRLayers.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN

// MARK: - Whisper Attention

/// Whisper attention layer with optional Rotary Position Embeddings.
public class WhisperAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scaling: Float
    let useRope: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "rope") var rope: RoPE?

    public init(config: WhisperConfig, useRope: Bool = false) {
        self.embedDim = config.dModel
        self.numHeads = config.encoderAttentionHeads
        self.headDim = embedDim / numHeads
        self.scaling = pow(Float(headDim), -0.5)
        self.useRope = useRope

        self._qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._kProj.wrappedValue = Linear(embedDim, embedDim, bias: false)
        self._vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)

        if useRope {
            self._rope.wrappedValue = RoPE(dimensions: headDim / 2, traditional: config.ropeTraditional)
        } else {
            self._rope.wrappedValue = nil
        }
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (bsz, tgtLen, _) = (hiddenStates.shape[0], hiddenStates.shape[1], hiddenStates.shape[2])

        var queryStates = qProj(hiddenStates)
        var keyStates = kProj(hiddenStates)
        var valueStates = vProj(hiddenStates)

        // Reshape to (bsz, numHeads, tgtLen, headDim)
        queryStates = queryStates.reshaped([bsz, tgtLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        keyStates = keyStates.reshaped([bsz, tgtLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        valueStates = valueStates.reshaped([bsz, tgtLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        if useRope, let rope = rope {
            queryStates = rope(queryStates)
            keyStates = rope(keyStates)
        }

        // Scaled dot product attention
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queryStates,
            keys: keyStates,
            values: valueStates,
            scale: scaling,
            mask: nil
        )

        // Reshape back to (bsz, tgtLen, embedDim)
        let output = attnOutput.transposed(0, 2, 1, 3).reshaped([bsz, tgtLen, embedDim])

        return outProj(output)
    }
}

// MARK: - Whisper Encoder Layer

/// Whisper encoder layer with optional RoPE support.
public class WhisperEncoderLayer: Module {
    let embedDim: Int

    @ModuleInfo(key: "self_attn") var selfAttn: WhisperAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(config: WhisperConfig, useRope: Bool = false) {
        self.embedDim = config.dModel

        self._selfAttn.wrappedValue = WhisperAttention(config: config, useRope: useRope)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
        self._fc1.wrappedValue = Linear(embedDim, config.encoderFfnDim)
        self._fc2.wrappedValue = Linear(config.encoderFfnDim, embedDim)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Pre-norm attention
        var residual = hiddenStates
        var h = selfAttnLayerNorm(hiddenStates)
        h = selfAttn(h)
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

// MARK: - Whisper Encoder

/// Whisper encoder with optional rotary position embeddings.
public class WhisperEncoder: Module {
    let config: WhisperConfig
    let useRope: Bool

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layers") var layers: [WhisperEncoderLayer]

    public init(config: WhisperConfig, useRope: Bool = false) {
        self.config = config
        self.useRope = useRope
        let embedDim = config.dModel

        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins,
            outputChannels: embedDim,
            kernelSize: 3,
            padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: embedDim,
            outputChannels: embedDim,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        // Always create for weight loading compatibility (only used when not using RoPE)
        self._embedPositions.wrappedValue = Embedding(embeddingCount: config.maxSourcePositions, dimensions: embedDim)

        var encoderLayers: [WhisperEncoderLayer] = []
        for _ in 0..<config.encoderLayers {
            encoderLayers.append(WhisperEncoderLayer(config: config, useRope: useRope))
        }
        self._layers.wrappedValue = encoderLayers
    }

    public func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        var hiddenStates = gelu(conv1(inputFeatures))
        hiddenStates = gelu(conv2(hiddenStates))

        // Add position embeddings if not using RoPE
        if !useRope {
            let seqLen = hiddenStates.shape[1]
            let embedPos = embedPositions.weight[0..<seqLen]
            hiddenStates = hiddenStates + embedPos
        }

        for layer in layers {
            hiddenStates = layer(hiddenStates)
        }

        return hiddenStates
    }
}

// MARK: - Adapting MLP

/// MLP adapter for audio-to-LM projection.
public class AdaptingMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(inputDim: Int, intermediateDim: Int, outputDim: Int) {
        self._fc1.wrappedValue = Linear(inputDim, intermediateDim, bias: true)
        self._fc2.wrappedValue = Linear(intermediateDim, outputDim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = gelu(h)
        h = fc2(h)
        return h
    }
}

// MARK: - Audio Encoder

/// Audio encoder with Whisper backbone and MLP adapter.
///
/// This matches the HuggingFace weight structure:
/// - audio_encoder.whisper.*
/// - audio_encoder.layer_norm.*
/// - audio_encoder.proj.*
/// - audio_encoder.adapting.*
/// - audio_encoder.audio_bos_eos_token.*
public class AudioEncoder: Module {
    let config: GLMASRModelConfig

    @ModuleInfo(key: "whisper") var whisper: WhisperEncoder
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "adapting") var adapting: AdaptingMLP
    @ModuleInfo(key: "audio_bos_eos_token") var audioBosEosToken: Embedding

    public init(config: GLMASRModelConfig) {
        self.config = config
        let whisperConfig = config.whisperConfig
        let lmHiddenSize = config.lmConfig.hiddenSize

        // Whisper encoder
        self._whisper.wrappedValue = WhisperEncoder(config: whisperConfig, useRope: config.useRope)

        // Layer norm after whisper encoder
        self._layerNorm.wrappedValue = LayerNorm(dimensions: whisperConfig.dModel)

        // Projection from whisper dim to LM hidden size
        self._proj.wrappedValue = Linear(whisperConfig.dModel, lmHiddenSize, bias: true)

        // MLP adapter: merged_dim -> intermediate -> lm_hidden
        let mergedDim = whisperConfig.dModel * config.mergeFactor
        let intermediateDim = lmHiddenSize * 2

        self._adapting.wrappedValue = AdaptingMLP(
            inputDim: mergedDim,
            intermediateDim: intermediateDim,
            outputDim: lmHiddenSize
        )

        // Begin/End of audio token embeddings
        self._audioBosEosToken.wrappedValue = Embedding(embeddingCount: 2, dimensions: lmHiddenSize)
    }

    public func callAsFunction(_ inputFeatures: MLXArray) -> (MLXArray, Int) {
        // Whisper encoding
        var audioFeatures = whisper(inputFeatures)

        // Layer norm
        audioFeatures = layerNorm(audioFeatures)

        // Merge audio features by merge_factor
        let batchSize = audioFeatures.shape[0]
        let seqLen = audioFeatures.shape[1]
        let mergeFactor = config.mergeFactor

        var newSeqLen = (seqLen - mergeFactor) / mergeFactor + 1
        let maxLen = config.maxWhisperLength / mergeFactor
        newSeqLen = min(newSeqLen, maxLen)

        var mergedFeatures: [MLXArray] = []
        for i in 0..<newSeqLen {
            let startIdx = i * mergeFactor
            let endIdx = startIdx + mergeFactor
            let chunk = audioFeatures[0..., startIdx..<endIdx, 0...]
            let chunkReshaped = chunk.reshaped([batchSize, -1])
            mergedFeatures.append(chunkReshaped)
        }

        let mergedAudio = MLX.stacked(mergedFeatures, axis: 1)

        // Project through MLP adapter
        let audioEmbeds = adapting(mergedAudio)

        return (audioEmbeds, newSeqLen)
    }

    /// Get begin-of-audio and end-of-audio token embeddings.
    public func getBoaEoaTokens() -> (MLXArray, MLXArray) {
        let boa = audioBosEosToken(MLXArray([0]))
        let eoa = audioBosEosToken(MLXArray([1]))
        return (boa, eoa)
    }
}
