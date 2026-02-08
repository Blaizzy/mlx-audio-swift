//
//  VocosBackbone.swift
//  MLXAudioCodecs
//
//  Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN

// MARK: - ConvNeXt Block

/// ConvNeXt block for the Vocos backbone.
///
/// Uses depthwise convolution followed by pointwise convolutions with GELU activation.
/// Supports optional adaptive normalization for conditional generation.
public class ConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dwconv: MLXNN.Conv1d
    @ModuleInfo(key: "norm") var norm: Module
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    let gamma: MLXArray?
    let useAdaNorm: Bool

    public init(
        dim: Int,
        intermediateDim: Int,
        layerScaleInitValue: Float = 0.125,
        adanormNumEmbeddings: Int? = nil,
        dwKernelSize: Int = 7
    ) {
        self.useAdaNorm = adanormNumEmbeddings != nil

        // Depthwise convolution with groups=dim
        self._dwconv.wrappedValue = MLXNN.Conv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: dwKernelSize,
            padding: dwKernelSize / 2,
            groups: dim
        )

        // Normalization (either LayerNorm or AdaLayerNorm)
        if let numEmbeddings = adanormNumEmbeddings {
            let adaNorm = AdaLayerNorm(numEmbeddings: numEmbeddings, embeddingDim: dim, eps: 1e-6)
            self._norm.wrappedValue = adaNorm
        } else {
            let norm = LayerNorm(dimensions: dim, eps: 1e-6)
            self._norm.wrappedValue = norm
        }

        // Pointwise/1x1 convs, implemented with linear layers
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim)

        // Layer scale parameter
        if layerScaleInitValue > 0 {
            self.gamma = layerScaleInitValue * MLXArray.ones([dim])
        } else {
            self.gamma = nil
        }
    }

    public func callAsFunction(_ x: MLXArray, condEmbeddingId: MLXArray? = nil) -> MLXArray {
        let residual = x

        // Depthwise conv
        var h = dwconv(x)

        // Normalization (conditional or regular)
        if useAdaNorm {
            guard let condEmbeddingId else {
                fatalError("AdaLayerNorm requires condEmbeddingId")
            }
            guard let adaNorm = norm as? AdaLayerNorm else {
                fatalError("Expected AdaLayerNorm for ConvNeXtBlock.norm")
            }
            h = adaNorm(h, condEmbedding: condEmbeddingId)
        } else {
            guard let layerNorm = norm as? LayerNorm else {
                fatalError("Expected LayerNorm for ConvNeXtBlock.norm")
            }
            h = layerNorm(h)
        }

        // Pointwise convs with GELU
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)

        // Layer scale
        if let gamma = gamma {
            h = gamma * h
        }

        // Residual connection
        return residual + h
    }
}

// MARK: - Vocos Backbone

/// Vocos backbone using ConvNeXt blocks.
///
/// Processes input features through an embedding conv layer followed by
/// a stack of ConvNeXt blocks. Supports optional adaptive normalization
/// for conditional generation (e.g., bandwidth-conditioned).
public class VocosBackbone: Module {
    let inputChannels: Int
    let useAdaNorm: Bool
    @ModuleInfo(key: "embed") var embed: MLXNN.Conv1d
    @ModuleInfo(key: "norm") var norm: Module
    @ModuleInfo(key: "convnext") var convnext: [ConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(
        inputChannels: Int,
        dim: Int,
        intermediateDim: Int,
        numLayers: Int,
        layerScaleInitValue: Float? = nil,
        adanormNumEmbeddings: Int? = nil,
        bias: Bool = true,
        inputKernelSize: Int = 7,
        dwKernelSize: Int = 7
    ) {
        self.inputChannels = inputChannels
        self.useAdaNorm = adanormNumEmbeddings != nil

        // Embedding convolution
        self._embed.wrappedValue = MLXNN.Conv1d(
            inputChannels: inputChannels,
            outputChannels: dim,
            kernelSize: inputKernelSize,
            padding: inputKernelSize / 2
        )

        // Initial normalization (either LayerNorm or AdaLayerNorm)
        if let numEmbeddings = adanormNumEmbeddings {
            let adaNorm = AdaLayerNorm(numEmbeddings: numEmbeddings, embeddingDim: dim, eps: 1e-6)
            self._norm.wrappedValue = adaNorm
        } else {
            let norm = LayerNorm(dimensions: dim, eps: 1e-6)
            self._norm.wrappedValue = norm
        }

        // Calculate layer scale init value
        let scaleValue = layerScaleInitValue ?? (1.0 / Float(numLayers))

        // Stack of ConvNeXt blocks
        self._convnext.wrappedValue = (0..<numLayers).map { _ in
            ConvNeXtBlock(
                dim: dim,
                intermediateDim: intermediateDim,
                layerScaleInitValue: scaleValue,
                adanormNumEmbeddings: adanormNumEmbeddings,
                dwKernelSize: dwKernelSize
            )
        }

        // Final layer norm (always regular LayerNorm)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
    }

    public func callAsFunction(_ x: MLXArray, bandwidthId: MLXArray? = nil) -> MLXArray {
        var h = x

        // Transpose if input is not in (B, L, C) format
        // Input should be (B, L, C) where C is input_channels
        if h.shape.last != inputChannels {
            h = h.transposed(0, 2, 1)
        }

        // Embedding conv
        h = embed(h)

        // Initial norm (conditional or regular)
        if useAdaNorm {
            guard let bandwidthId else {
                fatalError("AdaLayerNorm requires bandwidthId")
            }
            guard let adaNorm = norm as? AdaLayerNorm else {
                fatalError("Expected AdaLayerNorm for VocosBackbone.norm")
            }
            h = adaNorm(h, condEmbedding: bandwidthId)
        } else {
            guard let layerNorm = norm as? LayerNorm else {
                fatalError("Expected LayerNorm for VocosBackbone.norm")
            }
            h = layerNorm(h)
        }

        // ConvNeXt blocks
        for block in convnext {
            h = block(h, condEmbeddingId: bandwidthId)
        }

        // Final norm
        h = finalLayerNorm(h)

        return h
    }
}
