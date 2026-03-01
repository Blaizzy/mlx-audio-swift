import Foundation
import MLX
import MLXAudioCore
import MLXNN

// MARK: - Fused Embedding

class FusedEmbedding: Module {
    let numCodebooks: Int
    let vocabSize: Int
    let dim: Int

    @ModuleInfo(key: "emb") var emb: Embedding

    init(numCodebooks: Int, vocabSize: Int, dim: Int) {
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        self.dim = dim
        self._emb.wrappedValue = Embedding(embeddingCount: numCodebooks * vocabSize, dimensions: dim)
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let K = codes.dim(1)
        let offsets = (MLXArray(0..<Int32(K)).expandedDimensions(axes: [0, 2])) * MLXArray(Int32(vocabSize))
        let offsetCodes = codes + offsets
        let embeddings = emb(offsetCodes)
        return embeddings.mean(axis: 1)
    }
}

// MARK: - Detokenizer RMSNorm

class DetokRMSNorm: Module {
    let eps: Float
    var weight: MLXArray

    init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = MLX.sqrt(MLX.mean(x * x, axis: -1, keepDims: true) + MLXArray(eps))
        return x / rms * weight
    }
}

// MARK: - Detokenizer Conv Layer

class DetokenizerConvLayer: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int) {
        self._inProj.wrappedValue = Linear(dim, dim * 3, bias: false)
        self._conv.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: 3, padding: 2, groups: dim, bias: false
        )
        self._outProj.wrappedValue = Linear(dim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let seqlen = x.dim(1)
        let BCx = inProj(x).split(parts: 3, axis: -1)
        let bGate = BCx[0], cGate = BCx[1], xProj = BCx[2]
        let Bx = bGate * xProj
        let convOut = conv(Bx)[0..., ..<seqlen, 0...]
        return outProj(cGate * convOut)
    }
}

// MARK: - Detokenizer Sliding Window Attention

class DetokenizerSlidingWindowAttention: Module {
    let dim: Int
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayernorm: DetokRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayernorm: DetokRMSNorm

    init(dim: Int, numHeads: Int, numKvHeads: Int, slidingWindow: Int, ropeTheta: Float = 1000000.0) {
        self.dim = dim
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.headDim = dim / numHeads
        self.slidingWindow = slidingWindow
        self.scale = pow(Float(self.headDim), -0.5)
        self.ropeTheta = ropeTheta

        self._qProj.wrappedValue = Linear(dim, dim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKvHeads * self.headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKvHeads * self.headDim, bias: false)
        self._outProj.wrappedValue = Linear(dim, dim, bias: false)
        self._qLayernorm.wrappedValue = DetokRMSNorm(dim: self.headDim)
        self._kLayernorm.wrappedValue = DetokRMSNorm(dim: self.headDim)
    }

    private func applyRoPE(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let (B, H, T, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let invFreq = 1.0 / MLX.pow(
            MLXArray(ropeTheta),
            MLXArray(stride(from: 0, to: D, by: 2).map { Float($0) / Float(D) })
        )
        let positions = MLXArray(Int32(offset)..<Int32(offset + T)).asType(.float32)
        let angles = MLX.outer(positions, invFreq)

        let cosHalf = MLX.cos(angles)
        let sinHalf = MLX.sin(angles)
        let cosF = concatenated([cosHalf, cosHalf], axis: -1).expandedDimensions(axes: [0, 1])
        let sinF = concatenated([sinHalf, sinHalf], axis: -1).expandedDimensions(axes: [0, 1])

        let x1 = x[0..., 0..., 0..., ..<(D / 2)]
        let x2 = x[0..., 0..., 0..., (D / 2)...]

        let cosFirst = cosF[0..., 0..., 0..., ..<(D / 2)]
        let sinFirst = sinF[0..., 0..., 0..., ..<(D / 2)]
        let cosSecond = cosF[0..., 0..., 0..., (D / 2)...]
        let sinSecond = sinF[0..., 0..., 0..., (D / 2)...]
        let part1 = x1 * cosFirst - x2 * sinFirst
        let part2 = x2 * cosSecond + x1 * sinSecond
        return concatenated([part1, part2], axis: -1)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, T, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)

        q = qLayernorm(q)
        k = kLayernorm(k)
        q = applyRoPE(q)
        k = applyRoPE(k)

        // GQA expansion
        if numKvHeads < numHeads {
            let nRep = numHeads / numKvHeads
            k = MLX.repeated(k, count: nRep, axis: 1)
            v = MLX.repeated(v, count: nRep, axis: 1)
        }

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let mask = mask {
            scores = scores + mask
        }
        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, v).transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return outProj(out)
    }
}

// MARK: - Detokenizer SwiGLU

class DetokenizerSwiGLU: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Detokenizer Block

class DetokenizerBlock: Module {
    let layerType: String

    @ModuleInfo(key: "operator_norm") var operatorNorm: DetokRMSNorm
    @ModuleInfo(key: "conv") var conv: DetokenizerConvLayer?
    @ModuleInfo(key: "self_attn") var selfAttn: DetokenizerSlidingWindowAttention?
    @ModuleInfo(key: "ffn_norm") var ffnNorm: DetokRMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: DetokenizerSwiGLU

    init(
        dim: Int, hiddenDim: Int, layerType: String,
        numHeads: Int = 16, numKvHeads: Int = 8,
        slidingWindow: Int = 30, normEps: Float = 1e-5, ropeTheta: Float = 1000000.0
    ) {
        self.layerType = layerType
        self._operatorNorm.wrappedValue = DetokRMSNorm(dim: dim, eps: normEps)

        if layerType == "conv" {
            self._conv.wrappedValue = DetokenizerConvLayer(dim: dim)
        } else {
            self._selfAttn.wrappedValue = DetokenizerSlidingWindowAttention(
                dim: dim, numHeads: numHeads, numKvHeads: numKvHeads,
                slidingWindow: slidingWindow, ropeTheta: ropeTheta
            )
        }

        self._ffnNorm.wrappedValue = DetokRMSNorm(dim: dim, eps: normEps)
        self._feedForward.wrappedValue = DetokenizerSwiGLU(dim: dim, hiddenDim: hiddenDim)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = operatorNorm(x)
        let r: MLXArray
        if layerType == "conv" {
            r = conv!(h, mask: mask)
        } else {
            r = selfAttn!(h, mask: mask)
        }
        var out = x + r
        out = out + feedForward(ffnNorm(out))
        return out
    }
}

// MARK: - LFM Detokenizer Model

class LFMDetokenizerModel: Module {
    let config: DetokenizerConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: DetokRMSNorm
    let layers: [DetokenizerBlock]

    init(_ config: DetokenizerConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(embeddingCount: 65536, dimensions: config.hiddenSize)
        self._embeddingNorm.wrappedValue = DetokRMSNorm(dim: config.hiddenSize, eps: config.normEps)

        self.layers = config.layerTypes.map { layerType in
            DetokenizerBlock(
                dim: config.hiddenSize, hiddenDim: config.intermediateSize,
                layerType: layerType, numHeads: config.numAttentionHeads,
                numKvHeads: config.numKeyValueHeads, slidingWindow: config.slidingWindow,
                normEps: config.normEps, ropeTheta: config.ropeTheta
            )
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return embeddingNorm(h)
    }
}

// MARK: - LFM2 Audio Detokenizer

public class LFM2AudioDetokenizer: Module {
    let config: DetokenizerConfig

    @ModuleInfo(key: "emb") var emb: FusedEmbedding
    @ModuleInfo(key: "lfm") var lfm: LFMDetokenizerModel
    @ModuleInfo(key: "lin") var lin: Linear

    let nFft: Int
    let hopLength: Int
    var istftWindow: MLXArray?

    public init(_ config: DetokenizerConfig) {
        self.config = config
        self.nFft = config.nFft
        self.hopLength = config.hopLength

        self._emb.wrappedValue = FusedEmbedding(
            numCodebooks: config.numCodebooks, vocabSize: config.vocabSize,
            dim: config.hiddenSize
        )
        self._lfm.wrappedValue = LFMDetokenizerModel(config)
        self._lin.wrappedValue = Linear(config.hiddenSize, config.outputSize, bias: true)
    }

    private var window: MLXArray {
        if let w = istftWindow { return w }
        let n = nFft
        return MLXArray(
            (0..<n).map { Float(0.5 - 0.5 * cos(2.0 * Float.pi * Float($0) / Float(n))) }
        )
    }

    private func createSlidingWindowMask(_ T: Int) -> MLXArray {
        let idx = MLXArray(0..<Int32(T))
        let dIdx = idx.expandedDimensions(axis: 1) - idx.expandedDimensions(axis: 0)
        let geZero = dIdx .>= MLXArray(Int32(0))
        let ltWindow = MLXArray(Int32(config.slidingWindow)) .> dIdx
        let valid = geZero .&& ltWindow
        let mask = MLX.which(valid, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        return mask.expandedDimensions(axes: [0, 1])
    }

    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let (B, K, T) = (codes.dim(0), codes.dim(1), codes.dim(2))

        let clampedCodes = MLX.clip(codes, min: 0, max: config.vocabSize - 1)
        var x = emb(clampedCodes)
        x = MLX.repeated(x, count: config.upsampleFactor, axis: 1)

        let mask = createSlidingWindowMask(x.dim(1))
        x = lfm(x, mask: mask)
        x = lin(x)

        let nBins = nFft / 2 + 1
        let logMag = x[0..., 0..., ..<nBins]
        let phase = x[0..., 0..., nBins...]
        let mag = MLX.exp(logMag)
        return performISTFT(mag: mag, phase: phase)
    }

    private func performISTFT(mag: MLXArray, phase: MLXArray) -> MLXArray {
        let B = mag.dim(0)
        let TFrames = mag.dim(1)
        let win = window
        let pad = (nFft - hopLength) / 2

        let real = mag * MLX.cos(phase)
        let imag = mag * MLX.sin(phase)
        let stftComplex = real + MLXArray(real: Float(0), imaginary: Float(1)) * imag

        var outputs: [MLXArray] = []
        for b in 0..<B {
            let spec = stftComplex[b].transposed(1, 0)
            let framesFreq = MLXFFT.irfft(spec, axis: 0)
            let framesTime = framesFreq.transposed(1, 0)
            let windowedFrames = framesTime * win

            let outputLength = (TFrames - 1) * hopLength + nFft
            var audioSamples = [Float](repeating: 0, count: outputLength)
            var windowSum = [Float](repeating: 0, count: outputLength)

            let windowArray = win.asArray(Float.self)

            for i in 0..<TFrames {
                let start = i * hopLength
                let frameData = windowedFrames[i].asArray(Float.self)
                for j in 0..<min(nFft, frameData.count) {
                    if start + j < outputLength {
                        audioSamples[start + j] += frameData[j]
                        windowSum[start + j] += windowArray[j] * windowArray[j]
                    }
                }
            }

            for i in 0..<outputLength {
                if windowSum[i] != 0 {
                    audioSamples[i] /= windowSum[i]
                }
            }

            let trimmed: [Float]
            if pad > 0 && outputLength > 2 * pad {
                trimmed = Array(audioSamples[pad..<(outputLength - pad)])
            } else {
                trimmed = audioSamples
            }

            outputs.append(MLXArray(trimmed))
        }

        return MLX.stacked(outputs, axis: 0)
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, var value) in weights {
            if key.contains("conv.conv.weight") {
                if value.ndim == 3 {
                    if value.dim(2) > value.dim(1) {
                        value = value.transposed(0, 2, 1)
                    }
                }
            }
            mapped[key] = value
        }
        return mapped
    }

    // MARK: - From Pretrained

    public static func fromPretrained(modelPath: URL) throws -> LFM2AudioDetokenizer {
        let configURL = modelPath.appendingPathComponent("audio_detokenizer/config.json")
        let weightsURL = modelPath.appendingPathComponent("audio_detokenizer/model.safetensors")

        let configData = try Data(contentsOf: configURL)
        var config = try JSONDecoder().decode(DetokenizerConfig.self, from: configData)

        var weights = try MLX.loadArrays(url: weightsURL)

        if let ffnWeight = weights["lfm.layers.0.feed_forward.w1.weight"] {
            config.intermediateSize = ffnWeight.dim(0)
        }

        let model = LFM2AudioDetokenizer(config)

        let istftWindow = weights.removeValue(forKey: "istft.window")

        let sanitized = sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys])

        if let w = istftWindow {
            model.istftWindow = w
        }

        return model
    }
}
