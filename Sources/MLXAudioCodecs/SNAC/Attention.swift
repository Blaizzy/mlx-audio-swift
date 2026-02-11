//
//  Attention.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/25.
//

import Foundation
import MLX
import MLXNN

// MARK: - Local Multi-Head Attention

public class LocalMHA: Module, UnaryLayer {
    let norm: LayerNorm
    let heads: Int
    let windowSize: Int
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "rel_pos") var relPos: SinusoidalEmbeddings?
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(dim: Int = 1024, windowSize: Int = 32, dimHead: Int = 64, useRotaryPosEmb: Bool = true) {
        self.norm = LayerNorm(dimensions: dim)
        self.heads = dim / dimHead
        self.windowSize = windowSize
        self._toQKV.wrappedValue = Linear(dim, dim * 3, bias: false)
        self._relPos.wrappedValue = useRotaryPosEmb ? SinusoidalEmbeddings(dim: dimHead, scaleBase: Float(windowSize / 2)) : nil
        self._toOut.wrappedValue = Linear(dim, dim, bias: false)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let T = x.shape[2]
        
        let residual = x
        let x = norm(x.swappedAxes(1, 2))  // [B, T, C]
        let windows = T / windowSize
        
        let qkv = toQKV(x)
        let qkvSplit = qkv.split(parts: 3, axis: -1)
        var q = qkvSplit[0]
        var k = qkvSplit[1]
        var v = qkvSplit[2]
        
        // Rearrange: "b (w n) (h d) -> b h w n d"
        q = rearrangeForAttention(q, windows: windows, heads: heads)
        k = rearrangeForAttention(k, windows: windows, heads: heads)
        v = rearrangeForAttention(v, windows: windows, heads: heads)
        
        if let relPos = relPos {
            let (posEmb, scale) = relPos(k)
            (q, k) = applyRotaryPosEmb(q: q, k: k, freqs: posEmb, scale: scale)
        }
        
        let scale = sqrt(MLXArray(Float(q.shape[q.ndim - 1])))
        let scores = matmul(q, k.transposed(0, 1, 2, 4, 3)) / scale
        let attnWeights = softmax(scores, axis: -1)
        var out = matmul(attnWeights, v)
        
        // Rearrange: "b h w n d -> b (w n) (h d)"
        out = rearrangeFromAttention(out, windows: windows, heads: heads)
        out = toOut(out)
        
        return out.swappedAxes(1, 2) + residual
    }
    
    private func rearrangeForAttention(_ x: MLXArray, windows: Int, heads: Int) -> MLXArray {
        // Input: [B, W*N, H*D]
        // Output: [B, H, W, N, D]
        let B = x.shape[0]
        let WN = x.shape[1]
        let HD = x.shape[2]
        let N = WN / windows
        let D = HD / heads
        
        // Reshape to [B, W, N, H, D]
        let reshaped = x.reshaped([B, windows, N, heads, D])
        // Transpose to [B, H, W, N, D]
        return reshaped.transposed(0, 3, 1, 2, 4)
    }
    
    private func rearrangeFromAttention(_ x: MLXArray, windows: Int, heads: Int) -> MLXArray {
        // Input: [B, H, W, N, D]
        // Output: [B, W*N, H*D]
        let B = x.shape[0]
        let H = x.shape[1]
        let W = x.shape[2]
        let N = x.shape[3]
        let D = x.shape[4]
        
        // Transpose to [B, W, N, H, D]
        let transposed = x.transposed(0, 2, 3, 1, 4)
        // Reshape to [B, W*N, H*D]
        return transposed.reshaped([B, W * N, H * D])
    }
}

// MARK: - Sinusoidal Embeddings

public class SinusoidalEmbeddings: Module {
    @ModuleInfo(key: "inv_freq") var invFreq: MLXArray
    let useXPos: Bool
    let scaleBase: Float?
    var scale: MLXArray?

    public init(dim: Int, scaleBase: Float? = nil, useXPos: Bool = false) {
        let arange = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        self._invFreq.wrappedValue = 1.0 / pow(MLXArray(10000.0), arange / Float(dim))

        self.useXPos = useXPos
        self.scaleBase = scaleBase

        assert(!(useXPos && scaleBase == nil), "scale base must be defined if using xpos")

        if useXPos {
            let arange = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
            self.scale = (arange + 0.4 * Float(dim)) / (1.4 * Float(dim))
        } else {
            self.scale = nil
        }
    }
    
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = x.shape[x.ndim - 2]
        let t = MLXArray(stride(from: 0, to: seqLen, by: 1).map { Float($0) })
        
        // einsum "i,j->ij"
        let freqs = matmul(t.reshaped([-1, 1]), invFreq.reshaped([1, -1]))
        let freqsConcat = concatenated([freqs, freqs], axis: -1)
        
        if !useXPos {
            return (freqsConcat, ones([1]))
        }
        
        guard let scaleBase = scaleBase, let scale = scale else {
            return (freqsConcat, ones([1]))
        }
        
        let power = (t - Float(seqLen / 2)) / scaleBase
        let powerReshaped = power.reshaped([-1, 1])
        let scaleResult = pow(scale, powerReshaped)
        let scaleConcat = concatenated([scaleResult, scaleResult], axis: -1)
        
        return (freqsConcat, scaleConcat)
    }
}

// MARK: - Helper Functions

func rotateHalf(_ x: MLXArray) -> MLXArray {
    // Split last dimension in half: [..., d] -> [..., 2, d/2]
    let shape = x.shape
    let lastDim = shape[shape.count - 1]
    let halfDim = lastDim / 2
    
    var newShape = Array(shape[0..<shape.count-1])
    newShape.append(2)
    newShape.append(halfDim)
    
    let reshaped = x.reshaped(newShape)
    let split = reshaped.split(parts: 2, axis: -2)
    let x1 = split[0]
    let x2 = split[1]
    
    // Concatenate [-x2, x1] along last dimension
    let negX2 = -x2
    return concatenated([negX2, x1], axis: -1)
}

func applyRotaryPosEmb(q: MLXArray, k: MLXArray, freqs: MLXArray, scale: MLXArray) -> (MLXArray, MLXArray) {
    let qLen = q.shape[q.ndim - 2]
    
    // Get last q_len positions from freqs
    let qFreqs = freqs[.ellipsis, (-qLen)..., 0...]
    let invScale = pow(scale, MLXArray(-1.0))
    
    var scaleSliced = scale
    if scale.ndim == 2 {
        scaleSliced = scale[(-qLen)..., 0...]
    }
    
    let qRotated = (q * cos(qFreqs) * scaleSliced) + (rotateHalf(q) * sin(qFreqs) * scaleSliced)
    let kRotated = (k * cos(freqs) * invScale) + (rotateHalf(k) * sin(freqs) * invScale)
    
    return (qRotated, kRotated)
}
