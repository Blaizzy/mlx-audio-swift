import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXFast
import MLXAudioCore
import MLXNN

public let stepAudio2SampleRate = 24_000

public struct StepAudio2Prompt {
    public var promptToken: MLXArray
    public var promptTokenLen: MLXArray
    public var promptFeat: MLXArray
    public var promptFeatLen: MLXArray?
    public var embedding: MLXArray

    public init(
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        promptFeat: MLXArray,
        promptFeatLen: MLXArray? = nil,
        embedding: MLXArray
    ) {
        self.promptToken = promptToken
        self.promptTokenLen = promptTokenLen
        self.promptFeat = promptFeat
        self.promptFeatLen = promptFeatLen
        self.embedding = embedding
    }
}

public enum StepAudio2Error: Error, LocalizedError {
    case missingWeights(URL)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeights(let url):
            "Missing StepAudio2 weights at \(url.path)"
        case .invalidInput(let message):
            message
        }
    }
}

private func stepAudio2ApproxGELU(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1.0 + tanh(Float(sqrt(2.0 / Float.pi)) * (x + 0.044715 * pow(x, MLXArray(3)))))
}

private func stepAudio2Modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

private final class StepAudio2LayerNormNoAffine: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
    }
}

private final class StepAudio2MLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(inFeatures: Int, hiddenFeatures: Int? = nil, outFeatures: Int? = nil) {
        let hidden = hiddenFeatures ?? inFeatures
        let output = outFeatures ?? inFeatures
        _fc1.wrappedValue = Linear(inFeatures, hidden)
        _fc2.wrappedValue = Linear(hidden, output)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(stepAudio2ApproxGELU(fc1(x)))
    }
}

private final class StepAudio2Attention: Module {
    let numHeads: Int
    let headDim: Int
    let innerDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo var proj: Linear

    init(dim: Int, numHeads: Int = 8, headDim: Int = 64, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.innerDim = numHeads * headDim
        self.scale = pow(Float(headDim), -0.5)
        _toQ.wrappedValue = Linear(dim, innerDim, bias: qkvBias)
        _toK.wrappedValue = Linear(dim, innerDim, bias: qkvBias)
        _toV.wrappedValue = Linear(dim, innerDim, bias: qkvBias)
        _qNorm.wrappedValue = LayerNorm(dimensions: headDim)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        _proj.wrappedValue = Linear(innerDim, dim)
    }

    private func toHeads(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let c = x.dim(2)
        return x.reshaped(b, t, numHeads, c / numHeads).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray, attnMask: MLXArray?) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        var q = toHeads(toQ(x))
        var k = toHeads(toK(x))
        let v = toHeads(toV(x))

        q = qNorm(q)
        k = kNorm(k)

        var scores = matmul(q, k.swappedAxes(-1, -2)) * scale
        if var mask = attnMask {
            if mask.ndim == 3 {
                mask = mask.expandedDimensions(axis: 1)
            }
            scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        }
        let attn = softmax(scores, axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped(b, t, innerDim)
        return proj(out)
    }
}

private final class StepAudio2TimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int
    let scale: Float

    @ModuleInfo var mlp: StepAudio2TimestepMLP

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self.scale = 1000
        _mlp.wrappedValue = StepAudio2TimestepMLP(inputSize: frequencyEmbeddingSize, hiddenSize: hiddenSize)
    }

    private func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10_000) -> MLXArray {
        let half = dim / 2
        let freqs = exp(
            -Float(log(Float(maxPeriod))) * MLXArray(0..<half).asType(.float32) / Float(half)
        ).asType(t.dtype)
        let args = t.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
        var embedding = MLX.concatenated([cos(args), sin(args)], axis: -1)
        if dim % 2 != 0 {
            embedding = MLX.concatenated([embedding, MLXArray.zeros(like: embedding[0..., ..<1])], axis: -1)
        }
        return embedding
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        mlp(timestepEmbedding(t * scale, dim: frequencyEmbeddingSize))
    }
}

private final class StepAudio2TimestepMLP: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(inputSize: Int, hiddenSize: Int) {
        _linear1.wrappedValue = Linear(inputSize, hiddenSize)
        _linear2.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

private final class StepAudio2AdaLNModulation: Module {
    @ModuleInfo var linear: Linear

    init(hiddenSize: Int, outputSize: Int) {
        _linear.wrappedValue = Linear(hiddenSize, outputSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(silu(x))
    }
}

private final class StepAudio2CausalConvBlockModules: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var conv2: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int) {
        _conv1.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize)
        _norm.wrappedValue = LayerNorm(dimensions: outChannels)
        _conv2.wrappedValue = Conv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: kernelSize)
    }
}

private final class StepAudio2CausalConvBlock: Module {
    let kernelSize: Int

    @ModuleInfo var block: StepAudio2CausalConvBlockModules

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 3) {
        self.kernelSize = kernelSize
        _block.wrappedValue = StepAudio2CausalConvBlockModules(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize
        )
    }

    private func causalConv(_ x: MLXArray, _ conv: Conv1d) -> MLXArray {
        conv(MLX.padded(x, widths: [.init(0), .init((kernelSize - 1, 0)), .init(0)]))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x
        if let mask {
            out = out * mask
        }
        out = causalConv(out, block.conv1)
        out = block.norm(out)
        out = mish(out)
        out = causalConv(out, block.conv2)
        if let mask {
            out = out * mask
        }
        return out
    }
}

private final class StepAudio2DiTBlock: Module {
    @ModuleInfo var norm1: StepAudio2LayerNormNoAffine
    @ModuleInfo var attn: StepAudio2Attention
    @ModuleInfo var norm2: StepAudio2LayerNormNoAffine
    @ModuleInfo var mlp: StepAudio2MLP
    @ModuleInfo var norm3: StepAudio2LayerNormNoAffine
    @ModuleInfo var conv: StepAudio2CausalConvBlock
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: StepAudio2AdaLNModulation

    init(hiddenSize: Int, numHeads: Int, headDim: Int, mlpRatio: Float = 4.0) {
        _norm1.wrappedValue = StepAudio2LayerNormNoAffine(eps: 1e-6)
        _attn.wrappedValue = StepAudio2Attention(dim: hiddenSize, numHeads: numHeads, headDim: headDim)
        _norm2.wrappedValue = StepAudio2LayerNormNoAffine(eps: 1e-6)
        _mlp.wrappedValue = StepAudio2MLP(inFeatures: hiddenSize, hiddenFeatures: Int(Float(hiddenSize) * mlpRatio))
        _norm3.wrappedValue = StepAudio2LayerNormNoAffine(eps: 1e-6)
        _conv.wrappedValue = StepAudio2CausalConvBlock(inChannels: hiddenSize, outChannels: hiddenSize, kernelSize: 3)
        _adaLNModulation.wrappedValue = StepAudio2AdaLNModulation(hiddenSize: hiddenSize, outputSize: 9 * hiddenSize)
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray, attnMask: MLXArray?) -> MLXArray {
        let mod = adaLNModulation(c)
        let pieces = mod.split(parts: 9, axis: -1)
        var out = x
        out = out + pieces[2] * attn(stepAudio2Modulate(norm1(out), shift: pieces[0], scale: pieces[1]), attnMask: attnMask)
        out = out + pieces[8] * conv(stepAudio2Modulate(norm3(out), shift: pieces[6], scale: pieces[7]))
        out = out + pieces[5] * mlp(stepAudio2Modulate(norm2(out), shift: pieces[3], scale: pieces[4]))
        return out
    }
}

private final class StepAudio2FinalLayer: Module {
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: StepAudio2AdaLNModulation
    @ModuleInfo(key: "norm_final") var normFinal: StepAudio2LayerNormNoAffine
    @ModuleInfo var linear: Linear

    init(hiddenSize: Int, outChannels: Int) {
        _adaLNModulation.wrappedValue = StepAudio2AdaLNModulation(hiddenSize: hiddenSize, outputSize: 2 * hiddenSize)
        _normFinal.wrappedValue = StepAudio2LayerNormNoAffine(eps: 1e-6)
        _linear.wrappedValue = Linear(hiddenSize, outChannels)
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let mod = adaLNModulation(c)
        let pieces = mod.split(parts: 2, axis: -1)
        return linear(stepAudio2Modulate(normFinal(x), shift: pieces[0], scale: pieces[1]))
    }
}

private final class StepAudio2DiT: Module {
    let inChannels: Int
    let outChannels: Int

    @ModuleInfo(key: "t_embedder") var tEmbedder: StepAudio2TimestepEmbedder
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo var blocks: [StepAudio2DiTBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: StepAudio2FinalLayer

    init(
        inChannels: Int = 320,
        outChannels: Int = 80,
        mlpRatio: Float = 4.0,
        depth: Int = 16,
        numHeads: Int = 8,
        headDim: Int = 64,
        hiddenSize: Int = 512
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        _tEmbedder.wrappedValue = StepAudio2TimestepEmbedder(hiddenSize: hiddenSize)
        _inProj.wrappedValue = Linear(inChannels, hiddenSize)
        _blocks.wrappedValue = (0..<depth).map { _ in
            StepAudio2DiTBlock(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim, mlpRatio: mlpRatio)
        }
        _finalLayer.wrappedValue = StepAudio2FinalLayer(hiddenSize: hiddenSize, outChannels: outChannels)
    }

    func callAsFunction(
        x: MLXArray,
        mask: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        let time = tEmbedder(t).expandedDimensions(axis: 1)
        var pieces = [x, mu]
        if let spks {
            let expanded = spks.expandedDimensions(axis: -1)
            pieces.append(MLX.broadcast(expanded, to: [spks.dim(0), spks.dim(1), x.dim(-1)]))
        }
        if let cond {
            pieces.append(cond)
        }
        let combined = MLX.concatenated(pieces, axis: 1)
        return blocksForward(combined, t: time, mask: mask.asType(.bool))
    }

    private func blocksForward(_ x: MLXArray, t: MLXArray, mask: MLXArray?) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        out = inProj(out)
        for block in blocks {
            out = block(out, c: t, attnMask: mask)
        }
        out = finalLayer(out, c: t)
        return out.transposed(0, 2, 1)
    }
}

private final class StepAudio2CausalConditionalCFM: Module {
    let inferenceCFGRate: Float
    let outChannels: Int

    @ModuleInfo var estimator: StepAudio2DiT
    @ParameterInfo(key: "rand_noise") var randNoise: MLXArray

    init(estimator: StepAudio2DiT = StepAudio2DiT(), inferenceCFGRate: Float = 0.7) {
        self.inferenceCFGRate = inferenceCFGRate
        self.outChannels = estimator.outChannels
        _estimator.wrappedValue = estimator
        _randNoise.wrappedValue = MLXRandom.normal([1, estimator.outChannels, 50 * 600])
    }

    private func solveEuler(
        x initialX: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray,
        cond: MLXArray
    ) -> MLXArray {
        var x = initialX
        var t = tSpan[0].expandedDimensions(axis: 0)
        var dt = tSpan[1] - tSpan[0]
        let nSteps = tSpan.dim(0) - 1

        let maskIn = MLX.concatenated([mask, mask], axis: 0)
        let muIn = MLX.concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
        let spksIn = MLX.concatenated([spks, MLXArray.zeros(like: spks)], axis: 0)
        let condIn = MLX.concatenated([cond, MLXArray.zeros(like: cond)], axis: 0)

        for step in 1...nSteps {
            let xIn = MLX.concatenated([x, x], axis: 0)
            let tIn = MLX.concatenated([t, t], axis: 0)
            var dphiDt = estimator(x: xIn, mask: maskIn, mu: muIn, t: tIn, spks: spksIn, cond: condIn)
            let split = dphiDt.split(parts: 2, axis: 0)
            dphiDt = (1.0 + inferenceCFGRate) * split[0] - inferenceCFGRate * split[1]
            x = x + dt * dphiDt
            t = t + dt
            if step < nSteps {
                dt = tSpan[step + 1] - t
            }
        }
        return x
    }

    func callAsFunction(
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray,
        cond: MLXArray,
        nTimesteps: Int = 10,
        temperature: Float = 1.0,
        noise: MLXArray? = nil
    ) -> MLXArray {
        let z = (noise ?? randNoise[0..., 0..., ..<mu.dim(2)]) * temperature
        let linear = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1).asType(mu.dtype)
        let tSpan = 1 - cos(linear * Float32(Float.pi / 2))
        return solveEuler(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    }
}

private final class StepAudio2Flow: Module {
    let outputSize: Int
    let vocabSize: Int
    let preLookaheadLen: Int
    let upRate: Int

    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo var decoder: StepAudio2CausalConditionalCFM

    init(inputSize: Int = 512, outputSize: Int = 80, spkEmbedDim: Int = 192, vocabSize: Int = 6_561) {
        self.outputSize = outputSize
        self.vocabSize = vocabSize
        self.preLookaheadLen = 3
        self.upRate = 2
        _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
        _spkEmbedAffineLayer.wrappedValue = Linear(spkEmbedDim, outputSize)
        _encoder.wrappedValue = UpsampleConformerEncoder(
            inputSize: inputSize,
            outputSize: inputSize,
            attentionHeads: 8,
            linearUnits: 2048,
            numBlocks: 6,
            numUpBlocks: 4,
            dropoutRate: 0.1,
            positionalDropoutRate: 0.1,
            attentionDropoutRate: 0.1,
            posEncLayerType: "rel_pos_espnet",
            normalizeBefore: true,
            selfattentionLayerType: "rel_selfattn",
            keyBias: true,
            preLookaheadLen: 3,
            upsampleStride: 2
        )
        _encoderProj.wrappedValue = Linear(inputSize, outputSize)
        _decoder.wrappedValue = StepAudio2CausalConditionalCFM()
    }

    func inference(
        token: MLXArray,
        tokenLen: MLXArray,
        prompt: StepAudio2Prompt,
        nTimesteps: Int = 10
    ) throws -> MLXArray {
        if token.dim(0) != 1 {
            throw StepAudio2Error.invalidInput("StepAudio2 flow inference currently supports batch size 1")
        }

        let embNorm = sqrt((prompt.embedding * prompt.embedding).sum(axis: 1, keepDims: true))
        let embedding = spkEmbedAffineLayer(prompt.embedding / (embNorm + 1e-8))

        let combinedToken = MLX.concatenated([prompt.promptToken, token], axis: 1)
        let combinedLen = prompt.promptTokenLen + tokenLen
        var mask = MLX.logicalNot(s3genMakePadMask(lengths: combinedLen, maxLen: combinedToken.dim(1)))
        mask = mask.expandedDimensions(axis: -1).asType(embedding.dtype)
        let clipped = MLX.clip(combinedToken, min: 0, max: inputEmbedding.weight.dim(0) - 1)
        let embedded = inputEmbedding(clipped) * mask

        let (encoderOut, _) = encoder(xs: embedded, xsLens: combinedLen)
        let h = encoderProj(encoderOut)

        let promptMelLen = prompt.promptFeat.dim(1)
        let generatedMelLen = h.dim(1) - promptMelLen
        let conds = MLX.concatenated(
            [
                prompt.promptFeat,
                MLXArray.zeros([h.dim(0), generatedMelLen, outputSize]).asType(h.dtype),
            ],
            axis: 1
        ).transposed(0, 2, 1)

        let totalLen = promptMelLen + generatedMelLen
        var decoderMask = MLX.logicalNot(s3genMakePadMask(lengths: MLXArray([Int32(totalLen)]), maxLen: totalLen))
        decoderMask = decoderMask.asType(h.dtype).expandedDimensions(axis: 1)

        let feat = decoder(
            mu: h.transposed(0, 2, 1),
            mask: decoderMask,
            spks: embedding,
            cond: conds,
            nTimesteps: nTimesteps
        )
        let generated = feat[0..., 0..., promptMelLen...]
        if generated.dim(2) != generatedMelLen {
            throw StepAudio2Error.invalidInput("Unexpected generated mel length: \(generated.dim(2)) != \(generatedMelLen)")
        }
        return generated
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (key, value) in weights {
            var newKey = key
            newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.mlp.linear1.")
            newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.mlp.linear2.")
            newKey = newKey.replacingOccurrences(of: ".adaLN_modulation.1.", with: ".adaLN_modulation.linear.")
            newKey = newKey.replacingOccurrences(of: ".conv.block.1.", with: ".conv.block.conv1.")
            newKey = newKey.replacingOccurrences(of: ".conv.block.3.", with: ".conv.block.norm.")
            newKey = newKey.replacingOccurrences(of: ".conv.block.6.", with: ".conv.block.conv2.")
            result[newKey] = value
        }
        return result
    }
}

private final class StepAudio2HiFTGenerator: HiFTGenerator {
    init() {
        super.init(
            samplingRate: stepAudio2SampleRate,
            upsampleRates: [8, 5, 3],
            upsampleKernelSizes: [16, 11, 7],
            sourceResblockKernelSizes: [7, 7, 11],
            sourceResblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            useInterpolation: true
        )
    }

    override func decode(x: MLXArray, s: MLXArray) -> MLXArray {
        let squeezedS = s.squeezed(axis: 1)
        let nFft = istftParams["n_fft"]!
        let hopLen = istftParams["hop_len"]!
        let (sReal, sImag) = hifigan_stft(x: squeezedS, nFft: nFft, hopLength: hopLen, window: stftWindow)
        let sStft = MLX.concatenated([sReal, sImag], axis: 1)

        var h = convPre(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        for i in 0..<numUpsamples {
            h = leakyRelu(h, negativeSlope: lreluSlope)
            h = ups[i](h.transposed(0, 2, 1)).transposed(0, 2, 1)
            if i == numUpsamples - 1 {
                h = MLX.concatenated([h[0..., 0..., 1..<2], h], axis: 2)
            }

            var si = sourceDowners[i](sStft.transposed(0, 2, 1)).transposed(0, 2, 1)
            si = sourceResblocks[i](si)
            h = h + si

            var sum: MLXArray?
            let startIndex = i * numKernels
            for j in 0..<numKernels {
                let value = resblocks[startIndex + j](h)
                sum = sum == nil ? value : sum! + value
            }
            h = sum! / Float(numKernels)
        }

        h = leakyRelu(h)
        h = convPost(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        let nFftHalf = nFft / 2 + 1
        let magnitude = exp(h[0..., ..<nFftHalf, 0...])
        let phase = sin(h[0..., nFftHalf..., 0...])
        return MLX.clip(
            hifigan_istft(magnitude: magnitude, phase: phase, nFft: nFft, hopLength: hopLen, window: stftWindow),
            min: -audioLimit,
            max: audioLimit
        )
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (key, value) in weights {
            var newKey = key
            if key == "stft_window" { continue }
            if key.hasSuffix(".weight") || key.hasSuffix(".bias") {
                if key.hasPrefix("conv_pre.")
                    || key.hasPrefix("conv_post.")
                    || key.hasPrefix("ups.")
                    || key.hasPrefix("source_downs.")
                    || key.contains(".convs1.")
                    || key.contains(".convs2.")
                    || key.hasPrefix("f0_predictor.condnet.")
                {
                    let suffix = key.hasSuffix(".weight") ? ".weight" : ".bias"
                    newKey = String(key.dropLast(suffix.count)) + ".conv" + suffix
                }
            }
            result[newKey] = value
        }
        return result
    }
}

public final class StepAudio2Token2Wav: Module {
    public static let defaultRepository = "mlx-community/Step-Audio-2-token2wav"

    @ModuleInfo private var flow: StepAudio2Flow
    @ModuleInfo private var hift: StepAudio2HiFTGenerator

    public override init() {
        _flow.wrappedValue = StepAudio2Flow()
        _hift.wrappedValue = StepAudio2HiFTGenerator()
        super.init()
    }

    public func decodeToMel(_ speechTokens: MLXArray, prompt: StepAudio2Prompt, nTimesteps: Int = 10) throws -> MLXArray {
        let batchedTokens = speechTokens.ndim == 1 ? speechTokens.expandedDimensions(axis: 0) : speechTokens
        let tokens = batchedTokens.asType(.int32)
        let tokenLen = MLXArray([Int32(tokens.dim(1))])
        return try flow.inference(token: tokens, tokenLen: tokenLen, prompt: prompt, nTimesteps: nTimesteps)
    }

    public func vocode(_ mel: MLXArray) -> MLXArray {
        let (wav, _) = hift(mel)
        return wav
    }

    public func decode(_ speechTokens: MLXArray, prompt: StepAudio2Prompt, nTimesteps: Int = 10) throws -> MLXArray {
        try vocode(decodeToMel(speechTokens, prompt: prompt, nTimesteps: nTimesteps))
    }

    public static func fromModelDirectory(_ directory: URL) throws -> StepAudio2Token2Wav {
        let flowURL = directory.appendingPathComponent("flow.safetensors")
        let hiftURL = directory.appendingPathComponent("hift.safetensors")
        guard FileManager.default.fileExists(atPath: flowURL.path) else {
            throw StepAudio2Error.missingWeights(flowURL)
        }
        guard FileManager.default.fileExists(atPath: hiftURL.path) else {
            throw StepAudio2Error.missingWeights(hiftURL)
        }

        let model = StepAudio2Token2Wav()
        let flowWeights = StepAudio2Flow.sanitize(weights: try loadArrays(url: flowURL))
        try model.flow.update(parameters: ModuleParameters.unflattened(flowWeights), verify: .all)
        let hiftWeights = StepAudio2HiFTGenerator.sanitize(weights: try loadArrays(url: hiftURL))
        try model.hift.update(parameters: ModuleParameters.unflattened(hiftWeights), verify: .noUnusedKeys)
        eval(model)
        return model
    }

    public static func fromPretrained(
        _ source: String = defaultRepository,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> StepAudio2Token2Wav {
        guard let repoID = Repo.ID(rawValue: source) else {
            return try fromModelDirectory(URL(fileURLWithPath: NSString(string: source).expandingTildeInPath))
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.yaml"],
            hfToken: hfToken,
            cache: cache
        )
        return try fromModelDirectory(modelDir)
    }
}
