import Foundation
import MLX
import MLXAudioCore
import MLXNN
import MLXLMCommon
import HuggingFace

private struct UncheckedSendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

// MARK: - FastConformer Encoder Components

/// Depthwise-striding convolutional subsampling (factor=8).
private class ConvSubsampling: Module {
    @ModuleInfo var layers_0: Conv2d
    @ModuleInfo var layers_2: Conv2d
    @ModuleInfo var layers_3: Conv2d
    @ModuleInfo var layers_5: Conv2d
    @ModuleInfo var layers_6: Conv2d
    @ModuleInfo var linear: Linear

    init(_ config: FCEncoderConfig) {
        let convChannels = config.subsamplingConvChannels
        let featOut = config.hiddenSize
        let ks = config.subsamplingConvKernelSize
        let stride = config.subsamplingConvStride
        let pad = (ks - 1) / 2
        let ksPair = IntOrPair((ks, ks))
        let stridePair = IntOrPair((stride, stride))
        let padPair = IntOrPair((pad, pad))

        self._layers_0.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: convChannels,
            kernelSize: ksPair, stride: stridePair, padding: padPair
        )
        self._layers_2.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: ksPair, stride: stridePair, padding: padPair,
            groups: convChannels
        )
        self._layers_3.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: 1
        )
        self._layers_5.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: ksPair, stride: stridePair, padding: padPair,
            groups: convChannels
        )
        self._layers_6.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: 1
        )

        let featIn = config.numMelBins
        let linearIn = convChannels * Int(ceil(Double(featIn) / 8.0))
        self._linear.wrappedValue = Linear(linearIn, featOut)
    }

    /// - Parameters:
    ///   - x: `(batch, featDim, time)` mel spectrogram
    ///   - lengths: `(batch,)` frame lengths
    /// - Returns: `(x: (batch, time/8, hiddenSize), lengths: (batch,))`
    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        // (batch, feat, time) → NHWC: (batch, time, feat, 1)
        var h = x.transposed(0, 2, 1).expandedDimensions(axis: -1)

        h = relu(layers_0(h))
        h = relu(layers_3(layers_2(h)))
        h = relu(layers_6(layers_5(h)))

        // NHWC → (b, t, c, f) for flatten
        let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        h = h.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
        h = linear(h)

        // floor((L - 1) / 2) + 1 per stride-2 stage
        var outLengths = lengths.asType(.float32)
        for _ in 0..<3 {
            outLengths = MLX.floor((outLengths - 1) / 2).asType(.int32) + 1
        }

        return (h, outLengths)
    }
}

/// Relative positional encoding for Conformer (Transformer-XL style).
private class RelPositionalEncoding: Module {
    let dModel: Int

    init(dModel: Int) {
        self.dModel = dModel
    }

    /// Generate relative positional encoding.
    /// - Parameter x: `(batch, time, dModel)`
    /// - Returns: `(1, 2*time-1, dModel)`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.dim(1)
        let positions = MLXArray(stride(from: seqLen - 1, through: -(seqLen - 1), by: -1).map { Float($0) })

        let dim = MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
        let divTerm = MLX.exp(dim * Float(-log(10000.0) / Double(dModel)))

        let angles = positions.expandedDimensions(axis: 1) * divTerm.expandedDimensions(axis: 0)
        // Build PE: interleave sin/cos
        let sinAngles = MLX.sin(angles)
        let cosAngles = MLX.cos(angles)
        // (posLen, dModel/2, 2) → (posLen, dModel)
        let pe = MLX.stacked([sinAngles, cosAngles], axis: -1)
            .reshaped(positions.dim(0), dModel)
        return pe.expandedDimensions(axis: 0).asType(x.dtype)
    }
}

/// Multi-head attention with relative positional encoding (Transformer-XL).
private class RelPositionMultiHeadAttention: Module {
    let h: Int
    let dK: Int
    let sDK: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear

    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray

    init(_ config: FCEncoderConfig) {
        let nFeat = config.hiddenSize
        let nHead = config.numAttentionHeads
        h = nHead
        dK = nFeat / nHead
        sDK = sqrt(Float(dK))

        self._qProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._relativeKProj.wrappedValue = Linear(nFeat, nFeat, bias: false)

        self._biasU.wrappedValue = MLXArray.zeros([nHead, dK])
        self._biasV.wrappedValue = MLXArray.zeros([nHead, dK])
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let (b, headCount, qlen, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Pad left
        var padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((1, 0))])
        padded = padded.reshaped(b, headCount, posLen + 1, qlen)
        padded = padded[0..., 0..., 1..., 0...].reshaped(b, headCount, qlen, posLen)
        return padded
    }

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray,
        mask: MLXArray? = nil, posEmb: MLXArray? = nil
    ) -> MLXArray {
        let nBatch = query.dim(0)

        let q = qProj(query).reshaped(nBatch, -1, h, dK).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(nBatch, -1, h, dK).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(nBatch, -1, h, dK).transposed(0, 2, 1, 3)

        let qT = q.transposed(0, 2, 1, 3)

        let p = relativeKProj(posEmb!).reshaped(1, -1, h, dK).transposed(0, 2, 1, 3)

        let qWithBiasU = (qT + biasU).transposed(0, 2, 1, 3)
        let qWithBiasV = (qT + biasV).transposed(0, 2, 1, 3)

        let matrixAC = MLX.matmul(qWithBiasU, k.transposed(0, 1, 3, 2))
        var matrixBD = MLX.matmul(qWithBiasV, p.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., ..<matrixAC.dim(3)]

        var scores = (matrixAC + matrixBD) / sDK

        if let mask {
            scores = MLX.where(mask, MLXArray(-1e4).asType(scores.dtype), scores)
        }

        var attn = softmax(scores, axis: -1)
        if let mask {
            attn = MLX.where(mask, MLXArray(Float(0)).asType(scores.dtype), attn)
        }

        let out = MLX.matmul(attn, v)
        let reshaped = out.transposed(0, 2, 1, 3).reshaped(nBatch, -1, h * dK)
        return oProj(reshaped)
    }
}

/// Conformer feed-forward module.
private class ConformerFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(dModel: Int, dFf: Int) {
        self._linear1.wrappedValue = Linear(dModel, dFf)
        self._linear2.wrappedValue = Linear(dFf, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// Batch normalization using stored running statistics (inference mode only).
private class BatchNorm1d: Module {
    let eps: Float
    var weight: MLXArray
    var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(numFeatures: Int, eps: Float = 1e-5) {
        self.eps = eps
        weight = MLXArray.ones([numFeatures])
        bias = MLXArray.zeros([numFeatures])
        self._runningMean.wrappedValue = MLXArray.zeros([numFeatures])
        self._runningVar.wrappedValue = MLXArray.ones([numFeatures])
    }

    /// Apply batch norm using running stats.
    /// - Parameter x: `(batch, time, features)`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) / MLX.sqrt(runningVar + eps) * weight + bias
    }
}

/// Conformer convolution module with GLU, depthwise conv, and batch norm.
private class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo var norm: BatchNorm1d
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(_ config: FCEncoderConfig) {
        let dModel = config.hiddenSize
        let kernelSize = config.convKernelSize

        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel * 2,
            kernelSize: 1, bias: true
        )
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: kernelSize,
            padding: (kernelSize - 1) / 2,
            groups: dModel,
            bias: true
        )
        self._norm.wrappedValue = BatchNorm1d(numFeatures: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: 1, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pointwiseConv1(x)

        // GLU
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * sigmoid(parts[1])

        h = depthwiseConv(h)
        h = norm(h)
        h = silu(h)
        h = pointwiseConv2(h)
        return h
    }
}

/// Single Conformer encoder layer: FF1 → Self-Attn → Conv → FF2 → LN
private class ConformerLayer: Module {
    let fcFactor: Float = 0.5

    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo var conv: ConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: FCEncoderConfig) {
        let dModel = config.hiddenSize
        let dFf = config.intermediateSize

        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward1.wrappedValue = ConformerFeedForward(dModel: dModel, dFf: dFf)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(config)
        self._normConv.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ConformerConvolution(config)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward2.wrappedValue = ConformerFeedForward(dModel: dModel, dFf: dFf)
        self._normOut.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var residual = x
        var h = normFeedForward1(x)
        h = feedForward1(h)
        residual = residual + h * fcFactor

        h = normSelfAtt(residual)
        h = selfAttn(query: h, key: h, value: h, mask: mask, posEmb: posEmb)
        residual = residual + h

        h = normConv(residual)
        h = conv(h)
        residual = residual + h

        h = normFeedForward2(residual)
        h = feedForward2(h)
        residual = residual + h * fcFactor

        return normOut(residual)
    }
}

/// FastConformer encoder with conv subsampling and Conformer layers.
private class FastConformerEncoder: Module {
    let scaleInput: Bool
    let hiddenSize: Int

    @ModuleInfo var subsampling: ConvSubsampling
    var layers: [ConformerLayer]
    @ModuleInfo(key: "pos_enc") var posEnc: RelPositionalEncoding

    init(_ config: FCEncoderConfig) {
        scaleInput = config.scaleInput
        hiddenSize = config.hiddenSize

        self._subsampling.wrappedValue = ConvSubsampling(config)
        layers = (0..<config.numHiddenLayers).map { _ in ConformerLayer(config) }
        self._posEnc.wrappedValue = RelPositionalEncoding(dModel: config.hiddenSize)
    }

    /// Run ConvSubsampling only (for streaming pre-encode).
    func preEncode(_ audioSignal: MLXArray, length: MLXArray) -> (MLXArray, MLXArray) {
        subsampling(audioSignal, lengths: length)
    }

    /// Run Conformer layers on pre-encoded embeddings.
    /// - Returns: `(batch, hiddenSize, time)` channels-first
    func encode(_ embeddings: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var x = embeddings
        if scaleInput {
            x = x * Float(sqrt(Double(hiddenSize)))
        }

        let posEmb = posEnc(x)
        for layer in layers {
            x = layer(x, posEmb: posEmb)
        }

        x = x.transposed(0, 2, 1)
        return (x, lengths)
    }

    /// Full forward: ConvSubsampling + Conformer layers.
    func callAsFunction(_ audioSignal: MLXArray, length: MLXArray) -> (MLXArray, MLXArray) {
        let (x, lengths) = preEncode(audioSignal, length: length)
        return encode(x, lengths: lengths)
    }
}

// MARK: - Transformer Encoder Components (BART-style)

/// Standard multi-head attention for the Transformer encoder.
private class TransformerAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: TFEncoderConfig) {
        embedDim = config.dModel
        numHeads = config.encoderAttentionHeads
        headDim = embedDim / numHeads
        scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._kProj.wrappedValue = Linear(embedDim, embedDim, bias: config.kProjBias)
        self._vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let (b, t, _) = (query.dim(0), query.dim(1), query.dim(2))

        let q = qProj(query).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(b, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(b, -1, numHeads, headDim).transposed(0, 2, 1, 3)

        var scores = MLX.matmul(q * scale, k.transposed(0, 1, 3, 2))

        if let mask {
            scores = scores + mask
        }

        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, v).transposed(0, 2, 1, 3).reshaped(b, t, embedDim)
        return outProj(out)
    }
}

/// Single Transformer encoder layer (post-LN, BART-style).
private class TransformerEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TransformerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: TFEncoderConfig) {
        self._selfAttn.wrappedValue = TransformerAttention(config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        self._fc1.wrappedValue = Linear(config.dModel, config.encoderFfnDim)
        self._fc2.wrappedValue = Linear(config.encoderFfnDim, config.dModel)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
    }

    /// Post-LN: Attn → Add → LN → FFN(ReLU) → Add → LN
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var residual = x
        var h = selfAttn(query: x, key: x, value: x, mask: mask)
        h = residual + h
        h = selfAttnLayerNorm(h)

        residual = h
        h = relu(fc1(h))
        h = fc2(h)
        h = residual + h
        h = finalLayerNorm(h)

        return h
    }
}

/// Transformer encoder with learned positional embeddings.
private class TransformerEncoder: Module {
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    var layers: [TransformerEncoderLayer]

    init(_ config: TFEncoderConfig) {
        self._embedPositions.wrappedValue = Embedding(embeddingCount: config.maxSourcePositions, dimensions: config.dModel)
        layers = (0..<config.encoderLayers).map { _ in TransformerEncoderLayer(config) }
    }

    func callAsFunction(_ encoderStates: MLXArray, encoderMask: MLXArray? = nil) -> MLXArray {
        let seqLen = encoderStates.dim(1)
        let positions = MLXArray(0..<seqLen)
        var x = encoderStates + embedPositions(positions)

        var attnMask: MLXArray? = nil
        if let encoderMask {
            // Invert mask: True where valid → large negative where invalid
            let inverted = (1 - encoderMask.asType(.float32)) * -1e4
            attnMask = inverted.expandedDimensions(axes: [1, 2])
        }

        for layer in layers {
            x = layer(x, mask: attnMask)
        }

        return x
    }
}

// MARK: - Sortformer Modules

/// Sortformer output modules: projection + feedforward + speaker sigmoid.
private class SortformerModules: Module {
    let nSpk: Int

    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "first_hidden_to_hidden") var firstHiddenToHidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var singleHiddenToSpks: Linear
    @ModuleInfo(key: "hidden_to_spks") var hiddenToSpks: Linear

    init(_ config: ModulesConfig) {
        nSpk = config.numSpeakers

        self._encoderProj.wrappedValue = Linear(config.fcDModel, config.tfDModel)
        self._firstHiddenToHidden.wrappedValue = Linear(config.tfDModel, config.tfDModel)
        self._singleHiddenToSpks.wrappedValue = Linear(config.tfDModel, config.numSpeakers)
        self._hiddenToSpks.wrappedValue = Linear(2 * config.tfDModel, config.numSpeakers)
    }

    func forwardSpeakerSigmoids(_ hiddenOut: MLXArray) -> MLXArray {
        var h = relu(hiddenOut)
        h = firstHiddenToHidden(h)
        h = relu(h)
        let spkPreds = singleHiddenToSpks(h)
        return sigmoid(spkPreds)
    }

    static func lengthToMask(_ lengths: MLXArray, maxLength: Int) -> MLXArray {
        let arange = MLXArray(0..<maxLength)
        return arange.expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)
    }
}

// MARK: - Main Model

public class SortformerModel: Module {
    public let config: SortformerConfig

    @ModuleInfo(key: "fc_encoder") private var fcEncoder: FastConformerEncoder
    @ModuleInfo(key: "tf_encoder") private var tfEncoder: TransformerEncoder
    @ModuleInfo(key: "sortformer_modules") private var sortformerModules: SortformerModules

    public init(_ config: SortformerConfig) {
        self.config = config
        self._fcEncoder.wrappedValue = FastConformerEncoder(config.fcEncoderConfig)
        self._tfEncoder.wrappedValue = TransformerEncoder(config.tfEncoderConfig)
        self._sortformerModules.wrappedValue = SortformerModules(config.modulesConfig)
    }

    public var modelDtype: DType {
        sortformerModules.encoderProj.weight.dtype
    }

    /// Full forward pass.
    /// - Parameters:
    ///   - audioSignal: `(batch, nMels, time)` mel features
    ///   - audioSignalLength: `(batch,)` feature lengths
    /// - Returns: `(batch, diarFrameCount, numSpeakers)`
    public func callAsFunction(_ audioSignal: MLXArray, audioSignalLength: MLXArray) -> MLXArray {
        let signal = audioSignal.asType(modelDtype)
        var (embSeq, embSeqLength) = fcEncoder(signal, length: audioSignalLength)
        embSeq = embSeq.transposed(0, 2, 1)

        embSeq = sortformerModules.encoderProj(embSeq)

        let encoderMask = SortformerModules.lengthToMask(embSeqLength, maxLength: embSeq.dim(1))
        let transEmbSeq = tfEncoder(embSeq, encoderMask: encoderMask)
        let preds = sortformerModules.forwardSpeakerSigmoids(transEmbSeq)
        return preds * encoderMask.expandedDimensions(axis: 2)
    }

    // MARK: - Offline Inference

    public func generate(
        audio: MLXArray,
        sampleRate: Int = 16000,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0,
        verbose: Bool = false
    ) async throws -> DiarizationOutput {
        let sendableModel = UncheckedSendableBox(self)
        let sendableAudio = UncheckedSendableBox(audio)
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                let model = sendableModel.value
                let audio = sendableAudio.value
                let startTime = CFAbsoluteTimeGetCurrent()
                let proc = model.config.processorConfig

                var waveform = audio.asType(.float32)
                if waveform.ndim > 1 {
                    waveform = MLX.mean(waveform, axis: -1)
                }

                let (trimmed, trimOffset) = trimSilence(waveform, sampleRate: proc.samplingRate)
                waveform = trimmed
                let trimOffsetSec = Float(trimOffset) / Float(proc.samplingRate)

                // Peak normalize
                waveform = (1.0 / (MLX.abs(waveform).max() + 1e-3)) * waveform

                let features = extractMelFeatures(
                    waveform,
                    sampleRate: proc.samplingRate,
                    nFft: proc.nFft,
                    hopLength: proc.hopLength,
                    winLength: proc.winLength,
                    nMels: proc.featureSize,
                    preemphasisCoeff: proc.preemphasis
                )
                let featureLengths = MLXArray([Int32(features.dim(2))])

                if verbose {
                    print("Audio: \(String(format: "%.2f", Float(waveform.dim(0)) / Float(proc.samplingRate)))s")
                    if trimOffset > 0 {
                        print("Trimmed \(String(format: "%.2f", trimOffsetSec))s leading silence")
                    }
                    print("Features: \(features.shape)")
                }

                let preds = model(features, audioSignalLength: featureLengths)
                eval(preds)

                let subsamplingFactor = model.config.fcEncoderConfig.subsamplingFactor
                let frameDuration = Float(proc.hopLength * subsamplingFactor) / Float(proc.samplingRate)

                var segments = Self.predsToSegments(
                    preds[0],
                    frameDuration: frameDuration,
                    threshold: threshold,
                    minDuration: minDuration,
                    mergeGap: mergeGap
                )

                if trimOffset > 0 {
                    segments = segments.map {
                        DiarizationSegment(
                            start: $0.start + trimOffsetSec,
                            end: $0.end + trimOffsetSec,
                            speaker: $0.speaker
                        )
                    }
                }

                let activeSpeakers = Set(segments.map { $0.speaker })
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime

                if verbose {
                    print("Found \(segments.count) segments with \(activeSpeakers.count) speakers")
                    print("Processing time: \(String(format: "%.2f", elapsed))s")
                }

                continuation.resume(returning: DiarizationOutput(
                    segments: segments,
                    speakerProbs: preds[0],
                    numSpeakers: activeSpeakers.count,
                    totalTime: elapsed
                ))
            }
        }
    }

    // MARK: - Streaming API

    public func initStreamingState() -> StreamingState {
        let embDim = config.fcEncoderConfig.hiddenSize
        let nSpk = config.modulesConfig.numSpeakers
        let emptyEmb = MLXArray.zeros([1, 0, embDim])
        let emptyPred = MLXArray.zeros([1, 0, nSpk])
        return StreamingState(
            spkcache: emptyEmb,
            spkcachePreds: emptyPred,
            fifo: emptyEmb,
            fifoPreds: emptyPred,
            framesProcessed: 0,
            meanSilEmb: MLXArray.zeros([1, embDim]),
            nSilFrames: MLXArray.zeros([1])
        )
    }

    /// Process one chunk of mel features through the streaming pipeline.
    public func streamingStep(
        chunkFeatures: MLXArray,
        chunkLength: MLXArray,
        state: StreamingState,
        rightContextEmbs: MLXArray? = nil
    ) -> (MLXArray, StreamingState) {
        let mc = config.modulesConfig
        let useContext = mc.useAosc
        let lc = useContext ? mc.chunkLeftContext : 0
        let rc = useContext ? mc.chunkRightContext : 0

        // Pre-encode chunk through ConvSubsampling
        let chunkFeat = chunkFeatures.asType(modelDtype)
        var (chunkEmbs, chunkEmbLengths) = fcEncoder.preEncode(chunkFeat, length: chunkLength)
        let chunkDiarLen = Int(chunkEmbLengths[0].item(Int32.self))
        chunkEmbs = chunkEmbs[0..., ..<chunkDiarLen, 0...]

        // Build left context from end of FIFO
        var leftCtx: MLXArray? = nil
        var leftCtxLen = 0
        if lc > 0 && state.fifoLen > 0 {
            let take = min(lc, state.fifoLen)
            leftCtx = state.fifo[0..., (state.fifoLen - take)..., 0...]
            leftCtxLen = take
        }

        // Right context
        var rightCtxLen = 0
        if let rightContextEmbs, rc > 0 {
            rightCtxLen = rightContextEmbs.dim(1)
        }

        // Concatenate [spkcache, fifo, left_ctx, chunk, right_ctx]
        var parts = [MLXArray]()
        if state.spkcacheLen > 0 { parts.append(state.spkcache) }
        if state.fifoLen > 0 { parts.append(state.fifo) }
        if let leftCtx { parts.append(leftCtx) }
        parts.append(chunkEmbs)
        if let rightContextEmbs, rightCtxLen > 0 { parts.append(rightContextEmbs) }

        let allEmbs = MLX.concatenated(parts, axis: 1)
        let totalLen = allEmbs.dim(1)
        let allLengths = MLXArray([Int32(totalLen)])

        // Full encoder pass
        var (fcOut, _) = fcEncoder.encode(allEmbs, lengths: allLengths)
        fcOut = fcOut.transposed(0, 2, 1)
        fcOut = sortformerModules.encoderProj(fcOut)

        let encoderMask = SortformerModules.lengthToMask(allLengths, maxLength: totalLen)
        let transOut = tfEncoder(fcOut, encoderMask: encoderMask)
        var allPreds = sortformerModules.forwardSpeakerSigmoids(transOut)
        allPreds = allPreds * encoderMask.expandedDimensions(axis: 2)

        // Extract predictions for the new chunk only
        let chunkStart = state.spkcacheLen + state.fifoLen + leftCtxLen
        let chunkPreds = allPreds[0..., chunkStart..<(chunkStart + chunkDiarLen), 0...]
        let updatedCachePreds = allPreds[0..., ..<state.spkcacheLen, 0...]
        let updatedFifoPreds = allPreds[0..., state.spkcacheLen..<(state.spkcacheLen + state.fifoLen), 0...]

        eval(chunkPreds, chunkEmbs, updatedCachePreds, updatedFifoPreds)

        let newState = Self.updateStreamingState(
            state,
            chunkEmbs: chunkEmbs,
            chunkPreds: chunkPreds,
            updatedCachePreds: updatedCachePreds,
            updatedFifoPreds: updatedFifoPreds
        )

        return (chunkPreds[0], newState)
    }

    /// Feed a single audio chunk and get diarization results.
    public func feed(
        chunk: MLXArray,
        state: StreamingState,
        sampleRate: Int = 16000,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0,
        spkcacheMax: Int = 188,
        fifoMax: Int = 188
    ) async throws -> (DiarizationOutput, StreamingState) {
        let sendableModel = UncheckedSendableBox(self)
        let sendableChunk = UncheckedSendableBox(chunk)
        let sendableState = UncheckedSendableBox(state)
        return try await withCheckedThrowingContinuation { continuation in
            Task.detached {
                let model = sendableModel.value
                let chunk = sendableChunk.value
                let state = sendableState.value
                let proc = model.config.processorConfig
                let subsamplingFactor = model.config.fcEncoderConfig.subsamplingFactor
                let frameDuration = Float(proc.hopLength * subsamplingFactor) / Float(proc.samplingRate)

                var chunkMx = chunk.asType(.float32)
                if chunkMx.ndim > 1 {
                    chunkMx = MLX.mean(chunkMx, axis: -1)
                }

                let chunkTimeOffset = Float(state.framesProcessed) * frameDuration

                let useV2Feats = model.config.modulesConfig.useAosc
                if !useV2Feats {
                    chunkMx = (1.0 / (MLX.abs(chunkMx).max() + 1e-3)) * chunkMx
                }

                let features = extractMelFeatures(
                    chunkMx,
                    sampleRate: proc.samplingRate,
                    nFft: proc.nFft,
                    hopLength: proc.hopLength,
                    winLength: proc.winLength,
                    nMels: proc.featureSize,
                    preemphasisCoeff: proc.preemphasis,
                    normalize: useV2Feats ? nil : "per_feature",
                    padTo: 0
                )
                let featureLengths = MLXArray([Int32(features.dim(2))])

                var (chunkPreds, newState) = model.streamingStep(
                    chunkFeatures: features,
                    chunkLength: featureLengths,
                    state: state
                )

                var segments = Self.predsToSegments(
                    chunkPreds,
                    frameDuration: frameDuration,
                    threshold: threshold,
                    minDuration: minDuration,
                    mergeGap: mergeGap
                )

                segments = segments.map {
                    DiarizationSegment(
                        start: $0.start + chunkTimeOffset,
                        end: $0.end + chunkTimeOffset,
                        speaker: $0.speaker
                    )
                }

                newState = Self.maybeCompressState(
                    newState,
                    spkcacheMax: spkcacheMax,
                    fifoMax: fifoMax,
                    modulesCfg: model.config.modulesConfig
                )

                let activeSpeakers = Set(segments.map { $0.speaker })
                let output = DiarizationOutput(
                    segments: segments,
                    speakerProbs: chunkPreds,
                    numSpeakers: activeSpeakers.count
                )
                continuation.resume(returning: (output, newState))
            }
        }
    }

    /// Process audio in chunks, yielding diarization results incrementally.
    public func generateStream(
        audio: MLXArray,
        sampleRate: Int = 16000,
        chunkDuration: Float = 5.0,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0,
        spkcacheMax: Int = 188,
        fifoMax: Int = 188,
        verbose: Bool = false
    ) -> AsyncThrowingStream<DiarizationOutput, Error> {
        let sendableModel = UncheckedSendableBox(self)
        let sendableAudio = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            Task.detached {
                let model = sendableModel.value
                let audio = sendableAudio.value
                let proc = model.config.processorConfig
                let mc = model.config.modulesConfig

                var waveform = audio.asType(.float32)
                if waveform.ndim > 1 {
                    waveform = MLX.mean(waveform, axis: -1)
                }

                let useV2Feats = mc.useAosc

                var trimOffsetSec: Float = 0
                if !useV2Feats {
                    let (trimmed, trimOffset) = trimSilence(waveform, sampleRate: proc.samplingRate)
                    waveform = trimmed
                    trimOffsetSec = Float(trimOffset) / Float(proc.samplingRate)
                    waveform = (1.0 / (MLX.abs(waveform).max() + 1e-3)) * waveform
                }

                let features = extractMelFeatures(
                    waveform,
                    sampleRate: proc.samplingRate,
                    nFft: proc.nFft,
                    hopLength: proc.hopLength,
                    winLength: proc.winLength,
                    nMels: proc.featureSize,
                    preemphasisCoeff: proc.preemphasis,
                    normalize: useV2Feats ? nil : "per_feature",
                    padTo: useV2Feats ? 0 : 16
                )

                let totalMelFrames = features.dim(2)
                let subsamplingFactor = model.config.fcEncoderConfig.subsamplingFactor
                let frameDuration = Float(proc.hopLength * subsamplingFactor) / Float(proc.samplingRate)

                var chunkMel = Int(round(
                    chunkDuration * Float(proc.samplingRate) / Float(proc.hopLength) / Float(subsamplingFactor)
                )) * subsamplingFactor
                chunkMel = max(chunkMel, subsamplingFactor)

                // For v2.1: pre-encode all features for right context
                let rc = mc.chunkRightContext
                var allPreEmbs: MLXArray? = nil
                if useV2Feats && rc > 0 {
                    let (preEmbs, _) = model.fcEncoder.preEncode(features, length: MLXArray([Int32(totalMelFrames)]))
                    eval(preEmbs)
                    allPreEmbs = preEmbs
                }

                if verbose {
                    let audioDur = Float(waveform.dim(0)) / Float(proc.samplingRate)
                    let nChunks = Int(ceil(Double(totalMelFrames) / Double(chunkMel)))
                    print("Streaming: \(String(format: "%.2f", audioDur))s audio in \(nChunks) chunks (\(String(format: "%.1f", chunkDuration))s each)")
                }

                var state = model.initStreamingState()
                var offsetMel = 0
                var chunkIdx = 0
                var embOffset = 0

                while offsetMel < totalMelFrames {
                    try Task.checkCancellation()

                    let endMel = min(offsetMel + chunkMel, totalMelFrames)
                    let chunkFeat = features[0..., 0..., offsetMel..<endMel]
                    let chunkLen = MLXArray([Int32(chunkFeat.dim(2))])

                    // Compute right context embeddings for file mode
                    var rightCtx: MLXArray? = nil
                    if let allPreEmbs, rc > 0 {
                        let chunkMelFrames = chunkFeat.dim(2)
                        var dLen = Float(chunkMelFrames)
                        for _ in 0..<3 {
                            dLen = floor((dLen - 1) / 2) + 1
                        }
                        let chunkEmbLen = Int(dLen)
                        let rcStart = embOffset + chunkEmbLen
                        let rcEnd = min(rcStart + rc, allPreEmbs.dim(1))
                        if rcEnd > rcStart {
                            rightCtx = allPreEmbs[0..., rcStart..<rcEnd, 0...]
                        }
                        embOffset += chunkEmbLen
                    }

                    let (chunkPreds, newState) = model.streamingStep(
                        chunkFeatures: chunkFeat,
                        chunkLength: chunkLen,
                        state: state,
                        rightContextEmbs: rightCtx
                    )
                    state = newState

                    let chunkTimeOffset = Float(offsetMel * proc.hopLength) / Float(proc.samplingRate)

                    var segments = Self.predsToSegments(
                        chunkPreds,
                        frameDuration: frameDuration,
                        threshold: threshold,
                        minDuration: minDuration,
                        mergeGap: mergeGap
                    )

                    segments = segments.map {
                        DiarizationSegment(
                            start: $0.start + chunkTimeOffset + trimOffsetSec,
                            end: $0.end + chunkTimeOffset + trimOffsetSec,
                            speaker: $0.speaker
                        )
                    }

                    let activeSpeakers = Set(segments.map { $0.speaker })

                    if verbose {
                        chunkIdx += 1
                        let t0 = chunkTimeOffset + trimOffsetSec
                        let t1 = t0 + Float(chunkPreds.dim(0)) * frameDuration
                        print("  Chunk \(chunkIdx): \(String(format: "%.2f", t0))s-\(String(format: "%.2f", t1))s  \(segments.count) segments, context=\(state.spkcacheLen)+\(state.fifoLen) frames")
                    }

                    continuation.yield(DiarizationOutput(
                        segments: segments,
                        speakerProbs: chunkPreds,
                        numSpeakers: activeSpeakers.count
                    ))

                    state = Self.maybeCompressState(
                        state,
                        spkcacheMax: spkcacheMax,
                        fifoMax: fifoMax,
                        modulesCfg: model.config.modulesConfig
                    )

                    offsetMel = endMel
                }

                continuation.finish()
            }
        }
    }

    // MARK: - State Management

    private static func updateStreamingState(
        _ state: StreamingState,
        chunkEmbs: MLXArray,
        chunkPreds: MLXArray,
        updatedCachePreds: MLXArray,
        updatedFifoPreds: MLXArray
    ) -> StreamingState {
        let spkcache = state.spkcache
        let spkcachePreds = state.spkcacheLen > 0 ? updatedCachePreds : state.spkcachePreds
        let fifoPreds = state.fifoLen > 0 ? updatedFifoPreds : state.fifoPreds

        let newFifo = MLX.concatenated([state.fifo, chunkEmbs], axis: 1)
        let newFifoPreds = MLX.concatenated([fifoPreds, chunkPreds], axis: 1)
        eval(newFifo, newFifoPreds)

        return StreamingState(
            spkcache: spkcache,
            spkcachePreds: spkcachePreds,
            fifo: newFifo,
            fifoPreds: newFifoPreds,
            framesProcessed: state.framesProcessed + chunkPreds.dim(1),
            meanSilEmb: state.meanSilEmb,
            nSilFrames: state.nSilFrames
        )
    }

    private static func maybeCompressState(
        _ state: StreamingState,
        spkcacheMax: Int,
        fifoMax: Int,
        modulesCfg: ModulesConfig
    ) -> StreamingState {
        if state.fifoLen <= fifoMax {
            return state
        }

        let useAosc = modulesCfg.useAosc

        var popLen = state.fifoLen - fifoMax
        if useAosc {
            popLen = min(popLen, modulesCfg.spkcacheUpdatePeriod)
        }

        let poppedEmbs = state.fifo[0..., ..<popLen, 0...]
        let poppedPreds = state.fifoPreds[0..., ..<popLen, 0...]

        var meanSilEmb = state.meanSilEmb
        var nSilFrames = state.nSilFrames
        if useAosc {
            (meanSilEmb, nSilFrames) = getSilenceProfile(
                meanSilEmb: meanSilEmb,
                nSilFrames: nSilFrames,
                embs: poppedEmbs,
                preds: poppedPreds,
                silThreshold: modulesCfg.silThreshold
            )
        }

        var newCache = MLX.concatenated([state.spkcache, poppedEmbs], axis: 1)
        var newCachePreds = MLX.concatenated([state.spkcachePreds, poppedPreds], axis: 1)

        if newCache.dim(1) > spkcacheMax {
            if useAosc {
                (newCache, newCachePreds) = compressSpkcacheAosc(
                    embs: newCache,
                    preds: newCachePreds,
                    meanSilEmb: meanSilEmb,
                    modulesCfg: modulesCfg
                )
            } else {
                (newCache, newCachePreds) = compressSpkcacheSimple(
                    embs: newCache,
                    preds: newCachePreds,
                    targetLen: spkcacheMax
                )
            }
        }

        let newFifo = state.fifo[0..., popLen..., 0...]
        let newFifoPreds = state.fifoPreds[0..., popLen..., 0...]

        eval(newCache, newCachePreds, newFifo, newFifoPreds, meanSilEmb, nSilFrames)

        return StreamingState(
            spkcache: newCache,
            spkcachePreds: newCachePreds,
            fifo: newFifo,
            fifoPreds: newFifoPreds,
            framesProcessed: state.framesProcessed,
            meanSilEmb: meanSilEmb,
            nSilFrames: nSilFrames
        )
    }

    // MARK: - AOSC Compression

    private static func getSilenceProfile(
        meanSilEmb: MLXArray, nSilFrames: MLXArray,
        embs: MLXArray, preds: MLXArray, silThreshold: Float
    ) -> (MLXArray, MLXArray) {
        let isSil = MLX.sum(preds, axis: 2) .< silThreshold
        let silCount = MLX.sum(isSil.asType(.float32), axis: 1)

        let silEmbSum = MLX.sum(
            embs * isSil.asType(.float32).expandedDimensions(axis: -1),
            axis: 1
        )

        let updNSil = nSilFrames + silCount
        let oldSilSum = meanSilEmb * nSilFrames.expandedDimensions(axis: -1)
        let totalSilSum = oldSilSum + silEmbSum
        let updMean = totalSilSum / MLX.clip(updNSil.expandedDimensions(axis: -1), min: 1)

        return (updMean, updNSil)
    }

    private static func getLogPredScores(_ preds: MLXArray, threshold: Float) -> MLXArray {
        let logProbs = MLX.log(MLX.clip(preds, min: threshold))
        let log1Probs = MLX.log(MLX.clip(1.0 - preds, min: threshold))
        let log1ProbsSum = MLX.sum(log1Probs, axis: 2, keepDims: true)
        let log1ProbsSumBroadcast = MLX.broadcast(log1ProbsSum, to: preds.shape)
        return logProbs - log1Probs + log1ProbsSumBroadcast - Float(log(0.5))
    }

    private static func disableLowScores(
        preds: MLXArray, scores: MLXArray, minPosScoresPerSpk: Int
    ) -> MLXArray {
        let negInf = MLXArray(Float.infinity * -1)
        let isSpeech = preds .> 0.5
        var result = MLX.where(isSpeech, scores, negInf)

        let isPos = result .> 0
        let posCount = MLX.sum(isPos.asType(.float32), axis: 1, keepDims: true)
        let hasEnough = posCount .>= Float(minPosScoresPerSpk)
        let isNonposReplace = (.!isPos) & isSpeech & hasEnough
        result = MLX.where(isNonposReplace, negInf, result)
        return result
    }

    private static func boostTopkScores(
        scores: MLXArray, nBoostPerSpk: Int, scaleFactor: Float = 1.0
    ) -> MLXArray {
        if nBoostPerSpk <= 0 { return scores }
        let (_, nFrames, nSpk) = (scores.dim(0), scores.dim(1), scores.dim(2))
        let k = min(nBoostPerSpk, nFrames)
        let boostVal = -scaleFactor * Float(log(0.5))

        var resultSlices = [MLXArray]()
        for spk in 0..<nSpk {
            let flat = scores[0..., 0..., spk]  // (batch, nFrames)
            let topkIdx = MLX.argPartition(-flat, kth: k - 1, axis: 1)[0..., ..<k]
            let isFinite = flat .> (Float.infinity * -1)

            var mask = MLXArray.zeros(like: flat)
            let ones = MLXArray.ones(like: topkIdx).asType(.float32)
            // Scatter ones at top-k positions
            let batchIdx = MLXArray(0..<scores.dim(0)).expandedDimensions(axis: 1)
            let batchIdxBroadcast = MLX.broadcast(batchIdx, to: topkIdx.shape)
            mask = mask.at[batchIdxBroadcast, topkIdx].add(ones)

            let boostAmount = mask * boostVal * isFinite.asType(.float32)
            resultSlices.append(flat + boostAmount)
        }

        return MLX.stacked(resultSlices, axis: -1)
    }

    private static func getTopkIndices(
        scores: MLXArray, spkcacheLen: Int, spkcacheSilFramesPerSpk: Int, maxIndex: Int
    ) -> (MLXArray, MLXArray) {
        let (batchSize, nFrames, _) = (scores.dim(0), scores.dim(1), scores.dim(2))
        let nFramesNoSil = nFrames - spkcacheSilFramesPerSpk

        // Flatten: (batch, nSpk, nFrames) → (batch, nSpk * nFrames)
        let scoresFlat = scores.transposed(0, 2, 1).reshaped(batchSize, -1)

        let k = min(spkcacheLen, scoresFlat.dim(1))
        var topkIndices = MLX.argPartition(-scoresFlat, kth: k - 1, axis: 1)[0..., ..<k]
        let topkValues = MLX.takeAlong(scoresFlat, topkIndices, axis: 1)

        let validMask = topkValues .> (Float.infinity * -1)
        topkIndices = MLX.where(validMask, topkIndices, MLXArray(Int32(maxIndex)))

        var topkIndicesSorted = sorted(topkIndices, axis: 1)

        var isDisabled = topkIndicesSorted .== Int32(maxIndex)
        topkIndicesSorted = topkIndicesSorted % Int32(nFrames)
        isDisabled = isDisabled | (topkIndicesSorted .>= Int32(nFramesNoSil))
        topkIndicesSorted = MLX.where(isDisabled, MLXArray(Int32(0)), topkIndicesSorted)

        return (topkIndicesSorted, isDisabled)
    }

    private static func gatherSpkcacheAndPreds(
        embs: MLXArray, preds: MLXArray,
        topkIndices: MLXArray, isDisabled: MLXArray,
        meanSilEmb: MLXArray, spkcacheLen: Int
    ) -> (MLXArray, MLXArray) {
        let embDim = embs.dim(2)
        let nSpk = preds.dim(2)

        let idxEmb = MLX.broadcast(
            topkIndices.expandedDimensions(axis: -1),
            to: [topkIndices.dim(0), topkIndices.dim(1), embDim]
        )
        var gatheredEmbs = MLX.takeAlong(embs, idxEmb, axis: 1)

        let silExpanded = MLX.broadcast(
            meanSilEmb.expandedDimensions(axis: 1),
            to: [topkIndices.dim(0), spkcacheLen, embDim]
        )
        let disabledMask = isDisabled.expandedDimensions(axis: -1)
        gatheredEmbs = MLX.where(disabledMask, silExpanded, gatheredEmbs)

        let idxSpk = MLX.broadcast(
            topkIndices.expandedDimensions(axis: -1),
            to: [topkIndices.dim(0), topkIndices.dim(1), nSpk]
        )
        var gatheredPreds = MLX.takeAlong(preds, idxSpk, axis: 1)
        gatheredPreds = MLX.where(disabledMask, MLXArray(Float(0)), gatheredPreds)

        return (gatheredEmbs, gatheredPreds)
    }

    private static func compressSpkcacheAosc(
        embs: MLXArray, preds: MLXArray,
        meanSilEmb: MLXArray, modulesCfg: ModulesConfig
    ) -> (MLXArray, MLXArray) {
        let nSpk = modulesCfg.numSpeakers
        let spkcacheLen = modulesCfg.spkcacheLen
        let silPerSpk = modulesCfg.spkcacheSilFramesPerSpk
        let spkcacheLenPerSpk = spkcacheLen / nSpk - silPerSpk
        let strongBoost = Int(floor(Float(spkcacheLenPerSpk) * modulesCfg.strongBoostRate))
        let weakBoost = Int(floor(Float(spkcacheLenPerSpk) * modulesCfg.weakBoostRate))
        let minPos = Int(floor(Float(spkcacheLenPerSpk) * modulesCfg.minPosScoresRate))

        var scores = getLogPredScores(preds, threshold: modulesCfg.predScoreThreshold)
        scores = disableLowScores(preds: preds, scores: scores, minPosScoresPerSpk: minPos)

        // Boost newly added frames
        if modulesCfg.scoresBoostLatest > 0 && scores.dim(1) > spkcacheLen {
            let boostMask = MLX.concatenated([
                MLXArray.zeros([scores.dim(0), spkcacheLen, nSpk]),
                MLX.full([scores.dim(0), scores.dim(1) - spkcacheLen, nSpk], values: modulesCfg.scoresBoostLatest)
            ], axis: 1)
            scores = scores + boostMask
        }

        scores = boostTopkScores(scores: scores, nBoostPerSpk: strongBoost, scaleFactor: 2.0)
        scores = boostTopkScores(scores: scores, nBoostPerSpk: weakBoost, scaleFactor: 1.0)

        // Append silence padding with +inf scores
        if silPerSpk > 0 {
            let batchSize = scores.dim(0)
            let pad = MLX.full([batchSize, silPerSpk, nSpk], values: Float.infinity)
            scores = MLX.concatenated([scores, pad], axis: 1)
        }

        let (topkIndices, isDisabled) = getTopkIndices(
            scores: scores,
            spkcacheLen: spkcacheLen,
            spkcacheSilFramesPerSpk: silPerSpk,
            maxIndex: modulesCfg.maxIndex
        )

        let (compressedEmbs, compressedPreds) = gatherSpkcacheAndPreds(
            embs: embs, preds: preds,
            topkIndices: topkIndices, isDisabled: isDisabled,
            meanSilEmb: meanSilEmb, spkcacheLen: spkcacheLen
        )
        eval(compressedEmbs, compressedPreds)
        return (compressedEmbs, compressedPreds)
    }

    private static func compressSpkcacheSimple(
        embs: MLXArray, preds: MLXArray, targetLen: Int
    ) -> (MLXArray, MLXArray) {
        let logPreds = MLX.log(MLX.clip(preds[0], min: 1e-7, max: 1.0))
        let frameScores = MLX.sum(logPreds, axis: -1)

        var topIndices = MLX.argSort(-frameScores)[..<targetLen]
        topIndices = sorted(topIndices)

        let compressedEmbs = embs[0..., topIndices, 0...]
        let compressedPreds = preds[0..., topIndices, 0...]
        eval(compressedEmbs, compressedPreds)

        return (compressedEmbs, compressedPreds)
    }

    // MARK: - Postprocessing

    public static func predsToSegments(
        _ preds: MLXArray,
        frameDuration: Float,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0
    ) -> [DiarizationSegment] {
        let numSpeakers = preds.dim(1)
        var segments = [DiarizationSegment]()

        for spk in 0..<numSpeakers {
            let activity = preds[0..., spk] .> threshold
            if !MLX.any(activity).item(Bool.self) {
                continue
            }

            let padded = MLX.concatenated([
                MLXArray.zeros([1]).asType(DType.bool),
                activity,
                MLXArray.zeros([1]).asType(DType.bool)
            ])
            let changes = padded[1...].asType(DType.int32) - padded[..<(-1)].asType(DType.int32)
            eval(changes)

            let changesList: [Int32] = (0..<changes.dim(0)).map { changes[$0].item(Int32.self) }

            let starts = changesList.enumerated().compactMap { $0.element == 1 ? $0.offset : nil }
            let ends = changesList.enumerated().compactMap { $0.element == -1 ? $0.offset : nil }

            var spkSegments = [DiarizationSegment]()
            for (s, e) in zip(starts, ends) {
                let startTime = Float(s) * frameDuration
                let endTime = Float(e) * frameDuration
                let duration = endTime - startTime

                if duration >= minDuration {
                    spkSegments.append(DiarizationSegment(
                        start: startTime, end: endTime, speaker: spk
                    ))
                }
            }

            if mergeGap > 0 && spkSegments.count > 1 {
                var merged = [spkSegments[0]]
                for seg in spkSegments.dropFirst() {
                    if seg.start - merged.last!.end <= mergeGap {
                        merged[merged.count - 1] = DiarizationSegment(
                            start: merged.last!.start, end: seg.end, speaker: seg.speaker
                        )
                    } else {
                        merged.append(seg)
                    }
                }
                spkSegments = merged
            }

            segments.append(contentsOf: spkSegments)
        }

        segments.sort { $0.start < $1.start }
        return segments
    }

    // MARK: - Weight Sanitization & Loading

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        let skipKeys: Set<String> = ["num_batches_tracked"]

        let alreadyConverted = weights.keys.contains { $0.contains("subsampling.layers_") }

        for (k, var v) in weights {
            if skipKeys.contains(where: { k.contains($0) }) { continue }

            var newK = k

            if !alreadyConverted {
                if newK.contains("fc_encoder.subsampling.layers.") {
                    newK = newK.replacingOccurrences(of: "subsampling.layers.", with: "subsampling.layers_")
                }

                // Conv2d: PyTorch (O,I,H,W) → MLX (O,H,W,I)
                if newK.contains("subsampling") && newK.contains("weight") && !newK.contains("linear") {
                    if v.ndim == 4 {
                        v = v.transposed(0, 2, 3, 1)
                    }
                }

                // Conv1d: PyTorch (O,I,K) → MLX (O,K,I)
                if (newK.contains("pointwise_conv1") || newK.contains("pointwise_conv2") || newK.contains("depthwise_conv"))
                    && newK.contains("weight") {
                    if v.ndim == 3 {
                        v = v.transposed(0, 2, 1)
                    }
                }
            }

            sanitized[newK] = v
        }

        return sanitized
    }

    /// Load model from a HuggingFace repository.
    public static func fromPretrained(
        _ repoId: String,
        cache: HubCache = .default
    ) async throws -> SortformerModel {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "SortformerModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }

        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        // Load config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(SortformerConfig.self, from: configData)

        let model = SortformerModel(config)

        // Load weights
        let weightFiles = try FileManager.default.contentsOfDirectory(
            at: modelURL, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }

        var allWeights = [String: MLXArray]()
        for file in weightFiles {
            let weights = try loadArrays(url: file)
            for (k, v) in weights {
                allWeights[k] = v
            }
        }

        let sanitized = sanitize(allWeights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)
        eval(model.parameters())

        return model
    }
}
