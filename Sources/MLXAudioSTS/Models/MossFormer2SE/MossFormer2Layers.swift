import Foundation
import MLX
import MLXFast
import MLXNN

public class ScaleNorm: Module {
    let scale: Float
    let eps: Float

    @ModuleInfo(key: "g") var g: MLXArray

    public init(dim: Int, eps: Float = 1e-8) {
        self.scale = pow(Float(dim), -0.5)
        self.eps = eps
        self._g.wrappedValue = MLXArray.ones([1])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var norm = MLX.sqrt(MLX.sum(x * x, axis: -1, keepDims: true)) * scale
        norm = MLX.maximum(norm, MLXArray(eps))
        return x * (g / norm)
    }
}

public class GlobalLayerNorm: Module {
    let eps: Float
    let shape: Int
    let elementwiseAffine: Bool

    @ModuleInfo(key: "weight") var weight: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    public init(dim: Int, shape: Int = 3, eps: Float = 1e-8, elementwiseAffine: Bool = true) {
        self.eps = eps
        self.shape = shape
        self.elementwiseAffine = elementwiseAffine

        if elementwiseAffine {
            if shape == 3 {
                self._weight.wrappedValue = MLXArray.ones([dim, 1])
                self._bias.wrappedValue = MLXArray.zeros([dim, 1])
            } else {
                self._weight.wrappedValue = MLXArray.ones([dim])
                self._bias.wrappedValue = MLXArray.zeros([dim])
            }
        } else {
            self._weight.wrappedValue = nil
            self._bias.wrappedValue = nil
        }
    }

    public convenience init(_ dim: Int, _ shape: Int = 3, eps: Float = 1e-8, elementwiseAffine: Bool = true) {
        self.init(dim: dim, shape: shape, eps: eps, elementwiseAffine: elementwiseAffine)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(MLX.mean(x, axis: 2, keepDims: true), axis: 1, keepDims: true)
        let centered = x - mean
        let varValue = MLX.mean(MLX.mean(centered * centered, axis: 2, keepDims: true), axis: 1, keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(varValue + eps)

        guard elementwiseAffine, let weight, let bias else {
            return normalized
        }

        let w = weight.squeezed().reshaped([1, -1, 1])
        let b = bias.squeezed().reshaped([1, -1, 1])
        return w * normalized + b
    }
}

public class CLayerNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    public init(normalizedShape: Int, eps: Float = 1e-8) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([normalizedShape])
        self._bias.wrappedValue = MLXArray.zeros([normalizedShape])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let varValue = MLX.variance(x, axis: -1, keepDims: true)
        return (x - mean) / MLX.sqrt(varValue + eps) * weight + bias
    }
}

public class ScaledSinuEmbedding: Module {
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "inv_freq") var invFreq: MLXArray

    public init(dim: Int) {
        self._scale.wrappedValue = MLXArray.ones([1])
        let positions = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        self._invFreq.wrappedValue = 1.0 / MLX.pow(MLXArray(10000.0), positions / Float(dim))
    }

    public convenience init(_ dim: Int) {
        self.init(dim: dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.shape[1]
        let positions = MLXArray(stride(from: 0, to: seqLen, by: 1).map { Float($0) })
        let sinusoids = MLX.matmul(positions.reshaped([-1, 1]), invFreq.reshaped([1, -1]))
        let emb = MLX.concatenated([MLX.sin(sinusoids), MLX.cos(sinusoids)], axis: -1)
        return emb * scale
    }
}

public class OffsetScale: Module {
    let heads: Int

    @ModuleInfo(key: "gamma") var gamma: MLXArray
    @ModuleInfo(key: "beta") var beta: MLXArray

    public init(dim: Int, heads: Int = 1) {
        self.heads = heads
        self._gamma.wrappedValue = MLXRandom.normal([heads, dim], scale: 0.02) + 1.0
        self._beta.wrappedValue = MLXArray.zeros([heads, dim])
    }

    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let expanded = x.expandedDimensions(axis: -2)
        let out = expanded * gamma + beta
        return out.split(parts: heads, axis: -2).map { $0.squeezed(axis: -2) }
    }
}

public class ConvModule: Module {
    let inChannels: Int
    let padding: Int

    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(inChannels: Int, kernelSize: Int = 17) {
        self.inChannels = inChannels
        self.padding = (kernelSize - 1) / 2
        self._weight.wrappedValue = MLXArray.zeros([inChannels, kernelSize, 1])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let convOut = MLX.conv1d(x, weight, stride: 1, padding: padding, groups: inChannels)
        return x + convOut
    }
}

public class FFConvM: Module {
    @ModuleInfo(key: "norm") var norm: Module
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "conv_module") var convModule: ConvModule

    public init(dimIn: Int, dimOut: Int, normType: String = "layernorm") {
        if normType == "scalenorm" {
            self._norm.wrappedValue = ScaleNorm(dim: dimIn)
        } else {
            self._norm.wrappedValue = LayerNorm(dimensions: dimIn)
        }
        self._linear.wrappedValue = Linear(dimIn, dimOut)
        self._convModule.wrappedValue = ConvModule(inChannels: dimOut)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized: MLXArray
        if let scaleNorm = norm as? ScaleNorm {
            normalized = scaleNorm(x)
        } else if let layerNorm = norm as? LayerNorm {
            normalized = layerNorm(x)
        } else {
            normalized = x
        }

        var y = linear(normalized)
        y = silu(y)
        return convModule(y)
    }
}

public class UniDeepFsmnDepthwiseConv2d: Module {
    let channels: Int

    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(channels: Int, kernelSize: Int) {
        self.channels = channels
        self._weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1, 1])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1d = x.squeezed(axis: 2)
        let w1d = weight.squeezed(axis: 3)
        let y1d = MLX.conv1d(x1d, w1d, stride: 1, padding: 0, groups: channels)
        return y1d.expandedDimensions(axis: 2)
    }
}

public class UniDeepFsmn: Module {
    let inputDim: Int
    let outputDim: Int
    let lorder: Int

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "project") var project: Linear
    @ModuleInfo(key: "conv1") var conv1: UniDeepFsmnDepthwiseConv2d

    public init(inputDim: Int, outputDim: Int, lorder: Int, hiddenSize: Int) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.lorder = lorder
        self._linear.wrappedValue = Linear(inputDim, hiddenSize)
        self._project.wrappedValue = Linear(hiddenSize, outputDim, bias: false)
        self._conv1.wrappedValue = UniDeepFsmnDepthwiseConv2d(channels: outputDim, kernelSize: lorder + lorder - 1)
    }

    public func callAsFunction(_ inputTensor: MLXArray) -> MLXArray {
        var f1 = linear(inputTensor)
        f1 = MLX.maximum(f1, MLXArray(0.0))
        let p1 = project(f1)

        let x = p1.expandedDimensions(axis: 2)
        let padLeft = lorder - 1
        let padRight = lorder - 1
        let padded = MLX.padded(x, widths: [.init(0), .init((padLeft, padRight)), .init(0), .init(0)])

        let out = x + conv1(padded)
        let enhanced = out.squeezed(axis: 2)

        if inputDim == outputDim {
            return inputTensor + enhanced
        }
        return enhanced
    }
}

public class GatedFSMN: Module {
    @ModuleInfo(key: "to_u") var toU: FFConvM
    @ModuleInfo(key: "to_v") var toV: FFConvM
    @ModuleInfo(key: "fsmn") var fsmn: UniDeepFsmn

    public init(inChannels: Int, outChannels: Int, lorder: Int, hiddenSize: Int) {
        self._toU.wrappedValue = FFConvM(dimIn: inChannels, dimOut: hiddenSize, normType: "layernorm")
        self._toV.wrappedValue = FFConvM(dimIn: inChannels, dimOut: hiddenSize, normType: "layernorm")
        self._fsmn.wrappedValue = UniDeepFsmn(inputDim: inChannels, outputDim: outChannels, lorder: lorder, hiddenSize: hiddenSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputResidual = x
        var xu = toU(x)
        let xv = toV(x)
        xu = fsmn(xu)
        return xv * xu + inputResidual
    }
}

public class PReLU: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(initValue: Float = 0.25) {
        self._weight.wrappedValue = MLXArray(initValue)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pos = MLX.maximum(x, MLXArray(0.0))
        let neg = MLX.minimum(x, MLXArray(0.0))
        return pos + weight * neg
    }
}

public class GatedFSMNBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "norm1") var norm1: CLayerNorm
    @ModuleInfo(key: "norm2") var norm2: CLayerNorm
    @ModuleInfo(key: "gated_fsmn") var gatedFsmn: GatedFSMN
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    public init(dim: Int, innerChannels: Int = 256, groupSize: Int = 256, normType: String = "scalenorm") {
        self._conv1.wrappedValue = Conv1d(inputChannels: dim, outputChannels: innerChannels, kernelSize: 1, bias: true)
        self._prelu.wrappedValue = PReLU()
        self._norm1.wrappedValue = CLayerNorm(normalizedShape: innerChannels)
        self._norm2.wrappedValue = CLayerNorm(normalizedShape: innerChannels)
        self._gatedFsmn.wrappedValue = GatedFSMN(inChannels: innerChannels, outChannels: innerChannels, lorder: 20, hiddenSize: innerChannels)
        self._conv2.wrappedValue = Conv1d(inputChannels: innerChannels, outputChannels: dim, kernelSize: 1, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = conv1(x)
        y = prelu(y)
        y = norm1(y)
        y = gatedFsmn(y)
        y = norm2(y)
        y = conv2(y)
        return y + residual
    }
}

public enum FlashAttention {
    public static func simpleKernel(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, groupSize: Int? = nil) -> MLXArray {
        let g = groupSize ?? q.shape[2]
        let scale = 1.0 / Float(g)
        let sim = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let relu = MLX.maximum(sim, MLXArray(0.0))
        let attn = relu * relu
        return MLX.matmul(attn, v)
    }
}

public class FLASH_ShareA_FFConvM: Module {
    let dim: Int
    let groupSize: Int
    let queryKeyDim: Int
    let expansionFactor: Float
    let causal: Bool
    let shiftTokens: Bool

    var rotaryPosEmb: RoPE?

    @ModuleInfo(key: "to_hidden") var toHidden: FFConvM
    @ModuleInfo(key: "to_qk") var toQk: FFConvM
    @ModuleInfo(key: "qk_offset_scale") var qkOffsetScale: OffsetScale
    @ModuleInfo(key: "to_out") var toOut: FFConvM

    public init(
        dim: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        dropout: Float = 0.1,
        rotaryPosEmb: RoPE? = nil,
        normType: String = "scalenorm",
        shiftTokens: Bool = true
    ) {
        self.dim = dim
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.causal = causal
        self.shiftTokens = shiftTokens
        self.rotaryPosEmb = rotaryPosEmb

        let hiddenDim = Int(Float(dim) * expansionFactor)
        self._toHidden.wrappedValue = FFConvM(dimIn: dim, dimOut: hiddenDim, normType: normType)
        self._toQk.wrappedValue = FFConvM(dimIn: dim, dimOut: queryKeyDim, normType: normType)
        self._qkOffsetScale.wrappedValue = OffsetScale(dim: queryKeyDim, heads: 4)
        self._toOut.wrappedValue = FFConvM(dimIn: dim * 2, dimOut: dim, normType: normType)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var normedX = x

        if shiftTokens {
            let split = normedX.split(parts: 2, axis: -1)
            var xShift = split[0]
            let xPass = split[1]
            let seqLen = xShift.shape[1]
            if seqLen > 1 {
                let padding = MLXArray.zeros([xShift.shape[0], 1, xShift.shape[2]])
                xShift = MLX.concatenated([padding, xShift[0..., 0..<(seqLen - 1), 0...]], axis: 1)
            }
            normedX = MLX.concatenated([xShift, xPass], axis: -1)
        }

        let hiddenOutput = toHidden(normedX)
        let hiddenSplit = hiddenOutput.split(parts: 2, axis: -1)
        let v = hiddenSplit[0]
        let u = hiddenSplit[1]

        let qk = toQk(normedX)
        let heads = qkOffsetScale(qk)
        let quadQ = heads[0]
        let linQ = heads[1]
        let quadK = heads[2]
        let linK = heads[3]

        let (attV, attU) = calAttention(x: x, quadQ: quadQ, linQ: linQ, quadK: quadK, linK: linK, v: v, u: u, mask: mask)
        let out = (attU * v) * MLX.sigmoid(attV * u)
        return x + toOut(out)
    }

    private func calAttention(
        x: MLXArray,
        quadQ: MLXArray,
        linQ: MLXArray,
        quadK: MLXArray,
        linK: MLXArray,
        v: MLXArray,
        u: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let b = x.shape[0]
        let n = x.shape[1]
        let g = groupSize

        var quadQWork = quadQ
        var linQWork = linQ
        var quadKWork = quadK
        var linKWork = linK
        var vWork = v
        var uWork = u

        if let rope = rotaryPosEmb {
            quadQWork = rope(quadQWork)
            linQWork = rope(linQWork)
            quadKWork = rope(quadKWork)
            linKWork = rope(linKWork)
        }

        var maskWork = mask
        let padding = (g - n % g) % g
        if padding > 0 {
            let widths: [IntOrPair] = [.init(0), .init((0, padding)), .init(0)]
            quadQWork = MLX.padded(quadQWork, widths: widths)
            linQWork = MLX.padded(linQWork, widths: widths)
            quadKWork = MLX.padded(quadKWork, widths: widths)
            linKWork = MLX.padded(linKWork, widths: widths)
            vWork = MLX.padded(vWork, widths: widths)
            uWork = MLX.padded(uWork, widths: widths)
            if let currentMask = maskWork {
                maskWork = MLX.padded(currentMask, widths: [.init(0), .init((0, padding))])
            }
        }

        if let maskWork {
            let maskExpanded = maskWork.asType(quadQWork.dtype).expandedDimensions(axis: -1)
            quadKWork = quadKWork * maskExpanded
            linKWork = linKWork * maskExpanded
            vWork = vWork * maskExpanded
            uWork = uWork * maskExpanded
        }

        let newSeq = quadQWork.shape[1]
        let numGroups = newSeq / g

        let quadQGrouped = quadQWork.reshaped([b, numGroups, g, quadQWork.shape[2]])
        let quadKGrouped = quadKWork.reshaped([b, numGroups, g, quadKWork.shape[2]])
        let linQGrouped = linQWork.reshaped([b, numGroups, g, linQWork.shape[2]])
        let linKGrouped = linKWork.reshaped([b, numGroups, g, linKWork.shape[2]])
        let vGrouped = vWork.reshaped([b, numGroups, g, vWork.shape[2]])
        let uGrouped = uWork.reshaped([b, numGroups, g, uWork.shape[2]])

        let quadOutV = FlashAttention.simpleKernel(quadQGrouped, quadKGrouped, vGrouped, groupSize: g)
        let quadOutU = FlashAttention.simpleKernel(quadQGrouped, quadKGrouped, uGrouped, groupSize: g)

        let linKFlat = linKGrouped.reshaped([b, newSeq, queryKeyDim])
        let vDim = vGrouped.shape[3]
        let uDim = uGrouped.shape[3]
        let vFlat = vGrouped.reshaped([b, newSeq, vDim])
        let uFlat = uGrouped.reshaped([b, newSeq, uDim])

        let linKV = MLX.matmul(linKFlat.transposed(0, 2, 1), vFlat) / Float(n)
        let linKU = MLX.matmul(linKFlat.transposed(0, 2, 1), uFlat) / Float(n)

        let linQFlat = linQGrouped.reshaped([b, newSeq, queryKeyDim])
        let linOutVFlat = MLX.matmul(linQFlat, linKV)
        let linOutUFlat = MLX.matmul(linQFlat, linKU)

        var outV = quadOutV.reshaped([b, newSeq, vDim]) + linOutVFlat
        var outU = quadOutU.reshaped([b, newSeq, uDim]) + linOutUFlat

        if padding > 0 {
            outV = outV[0..., 0..<n, 0...]
            outU = outU[0..., 0..<n, 0...]
        }

        return (outV, outU)
    }
}

public class MossFormerBlock_GFSMN: Module {
    let depth: Int

    @ModuleInfo(key: "fsmn") var fsmn: [GatedFSMNBlock]
    @ModuleInfo(key: "layers") var layers: [FLASH_ShareA_FFConvM]

    public init(
        dim: Int,
        depth: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        attnDropout: Float = 0.1,
        normType: String = "scalenorm",
        shiftTokens: Bool = true
    ) {
        self.depth = depth

        let rope = RoPE(dimensions: min(32, queryKeyDim), traditional: true, base: 10000.0)

        self._fsmn.wrappedValue = (0..<depth).map { _ in
            GatedFSMNBlock(dim: dim, innerChannels: 256, groupSize: groupSize, normType: normType)
        }

        self._layers.wrappedValue = (0..<depth).map { _ in
            FLASH_ShareA_FFConvM(
                dim: dim,
                groupSize: groupSize,
                queryKeyDim: queryKeyDim,
                expansionFactor: expansionFactor,
                causal: causal,
                dropout: attnDropout,
                rotaryPosEmb: rope,
                normType: normType == "scalenorm" ? "scalenorm" : "layernorm",
                shiftTokens: shiftTokens
            )
        }
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var y = x
        for i in 0..<depth {
            y = layers[i](y, mask: mask)
            y = fsmn[i](y)
        }
        return y
    }
}
