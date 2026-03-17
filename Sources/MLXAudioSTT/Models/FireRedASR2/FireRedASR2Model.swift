import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN

final class FireRedASR2Conv2dSubsampling: Module {
    let subsampling: Int
    let context: Int

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "out") var out: Linear

    init(idim: Int, dModel: Int, outChannels: Int = 32) {
        self.subsampling = 4
        self.context = 7

        let subsampleIdim = ((idim - 1) / 2 - 1) / 2

        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: outChannels,
            kernelSize: 3,
            stride: 2
        )
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            stride: 2
        )
        self._out.wrappedValue = Linear(outChannels * subsampleIdim, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.expandedDimensions(axis: 3)
        y = relu(conv1(y))
        y = relu(conv2(y))

        let batch = y.shape[0]
        let time = y.shape[1]
        let freq = y.shape[2]
        let channels = y.shape[3]

        y = y.transposed(0, 1, 3, 2)
        y = y.reshaped([batch, time, channels * freq])
        return out(y)
    }
}

final class FireRedASR2RelPositionalEncoding {
    let pe: MLXArray

    init(dModel: Int, maxLen: Int = 5000) {
        var positive = [Float](repeating: 0, count: maxLen * dModel)
        var negative = [Float](repeating: 0, count: maxLen * dModel)

        let halfDim = dModel / 2
        var divTerm = [Float](repeating: 0, count: max(halfDim, 1))
        for i in 0..<halfDim {
            divTerm[i] = exp(Float(2 * i) * (-log(10000.0) / Float(dModel)))
        }

        for position in 0..<maxLen {
            for i in 0..<halfDim {
                let value = Float(position) * divTerm[i]
                let posBase = position * dModel + 2 * i
                positive[posBase] = sin(value)
                if posBase + 1 < positive.count {
                    positive[posBase + 1] = cos(value)
                }
                negative[posBase] = sin(-value)
                if posBase + 1 < negative.count {
                    negative[posBase + 1] = cos(-value)
                }
            }
        }

        var merged = [Float]()
        merged.reserveCapacity((2 * maxLen - 1) * dModel)

        for position in (0..<maxLen).reversed() {
            let start = position * dModel
            merged.append(contentsOf: positive[start..<(start + dModel)])
        }
        if maxLen > 1 {
            for position in 1..<maxLen {
                let start = position * dModel
                merged.append(contentsOf: negative[start..<(start + dModel)])
            }
        }

        self.pe = MLXArray(merged).reshaped([1, 2 * maxLen - 1, dModel])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let total = pe.shape[1]
        let length = x.shape[1]
        let start = total / 2 - length + 1
        let end = total / 2 + length
        return pe[0..., start..<end, 0...]
    }
}

final class FireRedASR2ConformerFeedForward: Module {
    @ModuleInfo(key: "net_0") var net0: LayerNorm
    @ModuleInfo(key: "net_1") var net1: Linear
    @ModuleInfo(key: "net_4") var net4: Linear

    init(dModel: Int) {
        self._net0.wrappedValue = LayerNorm(dimensions: dModel)
        self._net1.wrappedValue = Linear(dModel, dModel * 4)
        self._net4.wrappedValue = Linear(dModel * 4, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = net0(x)
        y = net1(y)
        y = y * sigmoid(y)
        y = net4(y)
        return y + residual
    }
}

final class FireRedASR2ConformerConvolution: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(dModel: Int, kernelSize: Int = 33) {
        let padding = (kernelSize - 1) / 2

        self._preLayerNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel * 4,
            kernelSize: 1,
            bias: false
        )
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel * 2,
            outputChannels: dModel * 2,
            kernelSize: kernelSize,
            padding: padding,
            groups: dModel * 2,
            bias: false
        )
        self._batchNorm.wrappedValue = LayerNorm(dimensions: dModel * 2)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel * 2,
            outputChannels: dModel,
            kernelSize: 1,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = preLayerNorm(x)
        y = pointwiseConv1(y)
        let gluParts = MLX.split(y, parts: 2, axis: -1)
        y = gluParts[0] * sigmoid(gluParts[1])
        y = depthwiseConv(y)
        y = batchNorm(y)
        y = y * sigmoid(y)
        y = pointwiseConv2(y)
        return y + residual
    }
}

final class FireRedASR2RelPosMultiHeadAttention: Module {
    let nHead: Int
    let dK: Int
    let scale: Float

    @ModuleInfo(key: "w_qs") var wQs: Linear
    @ModuleInfo(key: "w_ks") var wKs: Linear
    @ModuleInfo(key: "w_vs") var wVs: Linear
    @ModuleInfo(key: "layer_norm_q") var layerNormQ: LayerNorm
    @ModuleInfo(key: "layer_norm_k") var layerNormK: LayerNorm
    @ModuleInfo(key: "layer_norm_v") var layerNormV: LayerNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(nHead: Int, dModel: Int) {
        self.nHead = nHead
        self.dK = dModel / nHead
        self.scale = 1.0 / sqrt(Float(self.dK))

        self._wQs.wrappedValue = Linear(dModel, nHead * dK, bias: false)
        self._wKs.wrappedValue = Linear(dModel, nHead * dK, bias: false)
        self._wVs.wrappedValue = Linear(dModel, nHead * dK, bias: false)
        self._layerNormQ.wrappedValue = LayerNorm(dimensions: dModel)
        self._layerNormK.wrappedValue = LayerNorm(dimensions: dModel)
        self._layerNormV.wrappedValue = LayerNorm(dimensions: dModel)
        self._fc.wrappedValue = Linear(nHead * dK, dModel, bias: false)
        self._linearPos.wrappedValue = Linear(dModel, nHead * dK, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, dK], type: Float.self)
        self._posBiasV.wrappedValue = MLXArray.zeros([nHead, dK], type: Float.self)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let heads = x.shape[1]
        let t1 = x.shape[2]
        let t2 = x.shape[3]

        let zeroPad = MLXArray.zeros([batch, heads, t1, 1], type: Float.self)
        var padded = MLX.concatenated([zeroPad, x], axis: -1)
        padded = padded.reshaped([batch, heads, t2 + 1, t1])

        var shifted = padded[0..., 0..., 1..<(t2 + 1), 0...]
        shifted = shifted.reshaped([batch, heads, t1, t2])
        return shifted[0..., 0..., 0..., 0..<(t2 / 2 + 1)]
    }

    func callAsFunction(
        _ q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        posEmb: MLXArray
    ) -> MLXArray {
        let batch = q.shape[0]
        let time = q.shape[1]
        let residual = q

        var qProj = wQs(layerNormQ(q)).reshaped([batch, time, nHead, dK]).transposed(0, 2, 1, 3)
        let kProj = wKs(layerNormK(k)).reshaped([batch, k.shape[1], nHead, dK]).transposed(0, 2, 1, 3)
        let vProj = wVs(layerNormV(v)).reshaped([batch, v.shape[1], nHead, dK]).transposed(0, 2, 1, 3)
        let pProj = linearPos(posEmb).reshaped([posEmb.shape[0], posEmb.shape[1], nHead, dK]).transposed(0, 2, 1, 3)

        let qTransposed = qProj.transposed(0, 2, 1, 3)
        let qWithBiasU = (qTransposed + posBiasU).transposed(0, 2, 1, 3)
        let qWithBiasV = (qTransposed + posBiasV).transposed(0, 2, 1, 3)

        let matrixAC = qWithBiasU.matmul(kProj.transposed(0, 1, 3, 2))
        var matrixBD = qWithBiasV.matmul(pProj.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)

        let attention = softmax((matrixAC + matrixBD) * scale, axis: -1)
        var output = attention.matmul(vProj)
        output = output.transposed(0, 2, 1, 3).reshaped([batch, time, nHead * dK])
        qProj = fc(output)
        return qProj + residual
    }
}

final class FireRedASR2ConformerBlock: Module {
    @ModuleInfo(key: "ffn1") var ffn1: FireRedASR2ConformerFeedForward
    @ModuleInfo(key: "mhsa") var mhsa: FireRedASR2RelPosMultiHeadAttention
    @ModuleInfo(key: "conv") var conv: FireRedASR2ConformerConvolution
    @ModuleInfo(key: "ffn2") var ffn2: FireRedASR2ConformerFeedForward
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(dModel: Int, nHead: Int, kernelSize: Int = 33) {
        self._ffn1.wrappedValue = FireRedASR2ConformerFeedForward(dModel: dModel)
        self._mhsa.wrappedValue = FireRedASR2RelPosMultiHeadAttention(nHead: nHead, dModel: dModel)
        self._conv.wrappedValue = FireRedASR2ConformerConvolution(dModel: dModel, kernelSize: kernelSize)
        self._ffn2.wrappedValue = FireRedASR2ConformerFeedForward(dModel: dModel)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        var y = MLXArray(Float(0.5)) * x + MLXArray(Float(0.5)) * ffn1(x)
        y = mhsa(y, k: y, v: y, posEmb: posEmb)
        y = conv(y)
        y = MLXArray(Float(0.5)) * y + MLXArray(Float(0.5)) * ffn2(y)
        return layerNorm(y)
    }
}

final class FireRedASR2Encoder: Module {
    @ModuleInfo(key: "input_preprocessor") var inputPreprocessor: FireRedASR2Conv2dSubsampling
    @ModuleInfo(key: "layer_stack") var layerStack: [FireRedASR2ConformerBlock]

    let positionalEncoding: FireRedASR2RelPositionalEncoding

    init(config: FireRedASR2Config) {
        self._inputPreprocessor.wrappedValue = FireRedASR2Conv2dSubsampling(
            idim: config.idim,
            dModel: config.encoder.dModel
        )
        self._layerStack.wrappedValue = (0..<config.encoder.nLayers).map { _ in
            FireRedASR2ConformerBlock(
                dModel: config.encoder.dModel,
                nHead: config.encoder.nHead,
                kernelSize: config.encoder.kernelSize
            )
        }
        self.positionalEncoding = FireRedASR2RelPositionalEncoding(
            dModel: config.encoder.dModel,
            maxLen: config.encoder.peMaxlen
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let padRight = MLXArray.zeros(
            [batch, inputPreprocessor.context - 1, x.shape[2]],
            type: Float.self
        )
        var y = MLX.concatenated([x, padRight], axis: 1)
        y = inputPreprocessor(y)
        let posEmb = positionalEncoding(y)
        for layer in layerStack {
            y = layer(y, posEmb: posEmb)
        }
        return y
    }
}

final class FireRedASR2PositionalEncoding {
    let pe: MLXArray

    init(dModel: Int, maxLen: Int = 5000) {
        var values = [Float](repeating: 0, count: maxLen * dModel)
        let halfDim = dModel / 2
        var divTerm = [Float](repeating: 0, count: max(halfDim, 1))
        for i in 0..<halfDim {
            divTerm[i] = exp(Float(2 * i) * (-log(10000.0) / Float(dModel)))
        }

        for position in 0..<maxLen {
            for i in 0..<halfDim {
                let value = Float(position) * divTerm[i]
                let base = position * dModel + 2 * i
                values[base] = sin(value)
                if base + 1 < values.count {
                    values[base + 1] = cos(value)
                }
            }
        }

        self.pe = MLXArray(values).reshaped([1, maxLen, dModel])
    }

    func callAsFunction(length: Int) -> MLXArray {
        pe[0..., 0..<length, 0...]
    }
}

final class FireRedASR2DecoderMultiHeadAttention: Module {
    let nHead: Int
    let dK: Int
    let scale: Float

    @ModuleInfo(key: "w_qs") var wQs: Linear
    @ModuleInfo(key: "w_ks") var wKs: Linear
    @ModuleInfo(key: "w_vs") var wVs: Linear
    @ModuleInfo(key: "fc") var fc: Linear

    init(dModel: Int, nHead: Int) {
        self.nHead = nHead
        self.dK = dModel / nHead
        self.scale = 1.0 / sqrt(Float(self.dK))

        self._wQs.wrappedValue = Linear(dModel, nHead * dK)
        self._wKs.wrappedValue = Linear(dModel, nHead * dK, bias: false)
        self._wVs.wrappedValue = Linear(dModel, nHead * dK)
        self._fc.wrappedValue = Linear(nHead * dK, dModel)
    }

    func callAsFunction(
        _ q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batch = q.shape[0]

        let qProj = wQs(q).reshaped([batch, q.shape[1], nHead, dK]).transposed(0, 2, 1, 3)
        let kProj = wKs(k).reshaped([batch, k.shape[1], nHead, dK]).transposed(0, 2, 1, 3)
        let vProj = wVs(v).reshaped([batch, v.shape[1], nHead, dK]).transposed(0, 2, 1, 3)

        let additiveMask = mask.map {
            let zero = MLXArray.zeros($0.shape, dtype: qProj.dtype)
            let negInf = MLXArray.full($0.shape, values: MLXArray(Float(-1e9)), dtype: qProj.dtype)
            return MLX.where($0, zero, negInf).expandedDimensions(axis: 1)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: qProj,
            keys: kProj,
            values: vProj,
            scale: scale,
            mask: additiveMask
        )
        var output = attended
        output = output.transposed(0, 2, 1, 3).reshaped([batch, q.shape[1], nHead * dK])
        return fc(output)
    }
}

final class FireRedASR2PositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(dModel: Int, dFF: Int) {
        self._w1.wrappedValue = Linear(dModel, dFF)
        self._w2.wrappedValue = Linear(dFF, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(gelu(w1(x)))
    }
}

final class FireRedASR2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: FireRedASR2DecoderMultiHeadAttention
    @ModuleInfo(key: "cross_attn_norm") var crossAttnNorm: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: FireRedASR2DecoderMultiHeadAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FireRedASR2PositionwiseFeedForward

    init(dModel: Int, nHead: Int) {
        self._selfAttnNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = FireRedASR2DecoderMultiHeadAttention(dModel: dModel, nHead: nHead)
        self._crossAttnNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._crossAttn.wrappedValue = FireRedASR2DecoderMultiHeadAttention(dModel: dModel, nHead: nHead)
        self._mlpNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._mlp.wrappedValue = FireRedASR2PositionwiseFeedForward(dModel: dModel, dFF: dModel * 4)
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        selfAttnMask: MLXArray? = nil,
        cache: MLXArray? = nil
    ) -> MLXArray {
        var residual = x
        let xNorm = selfAttnNorm(x)

        let query: MLXArray
        let mask: MLXArray?
        if cache != nil {
            query = xNorm[0..., (xNorm.shape[1] - 1)..<xNorm.shape[1], 0...]
            residual = residual[0..., (residual.shape[1] - 1)..<residual.shape[1], 0...]
            if let selfAttnMask {
                mask = selfAttnMask[0..., (selfAttnMask.shape[1] - 1)..<selfAttnMask.shape[1], 0...]
            } else {
                mask = nil
            }
        } else {
            query = xNorm
            mask = selfAttnMask
        }

        var y = residual + selfAttn(query, k: xNorm, v: xNorm, mask: mask)
        residual = y
        y = residual + crossAttn(crossAttnNorm(y), k: encoderOutput, v: encoderOutput)

        residual = y
        y = residual + mlp(mlpNorm(y))

        if let cache {
            y = MLX.concatenated([cache, y], axis: 1)
        }
        return y
    }
}

final class FireRedASR2TransformerDecoder: Module {
    let sosID: Int
    let eosID: Int
    let padID: Int
    let nLayers: Int
    let scale: Float

    @ModuleInfo(key: "tgt_word_emb") var tgtWordEmb: Embedding
    @ModuleInfo(key: "layer_stack") var layerStack: [FireRedASR2DecoderLayer]
    @ModuleInfo(key: "layer_norm_out") var layerNormOut: LayerNorm
    @ModuleInfo(key: "tgt_word_prj") var tgtWordPrj: Linear

    let positionalEncoding: FireRedASR2PositionalEncoding

    init(config: FireRedASR2Config) {
        self.sosID = config.sosID
        self.eosID = config.eosID
        self.padID = config.padID
        self.nLayers = config.decoder.nLayers
        self.scale = sqrt(Float(config.decoder.dModel))

        self._tgtWordEmb.wrappedValue = Embedding(
            embeddingCount: config.odim,
            dimensions: config.decoder.dModel
        )
        self.positionalEncoding = FireRedASR2PositionalEncoding(
            dModel: config.decoder.dModel,
            maxLen: config.decoder.peMaxlen
        )
        self._layerStack.wrappedValue = (0..<config.decoder.nLayers).map { _ in
            FireRedASR2DecoderLayer(
                dModel: config.decoder.dModel,
                nHead: config.decoder.nHead
            )
        }
        self._layerNormOut.wrappedValue = LayerNorm(dimensions: config.decoder.dModel)
        self._tgtWordPrj.wrappedValue = Linear(config.decoder.dModel, config.odim, bias: false)
    }

    private func causalMask(length: Int) -> MLXArray {
        let positions = MLXArray((0..<length).map(Int32.init))
        let row = positions.expandedDimensions(axis: 1)
        let col = positions.expandedDimensions(axis: 0)
        return (row .>= col).expandedDimensions(axis: 0)
    }

    private func topK(_ x: MLXArray, k: Int) -> (MLXArray, MLXArray) {
        let count = min(max(k, 1), x.shape[x.ndim - 1])
        let sortedIndices = MLX.argSort(-x, axis: -1)
        if x.ndim == 1 {
            let topIndices = sortedIndices[0..<count]
            return (take(x, topIndices, axis: -1), topIndices)
        }
        let topIndices = sortedIndices[0..., 0..<count]
        return (take(x, topIndices, axis: -1), topIndices)
    }

    private func decoderState(
        tokens: MLXArray,
        encoderOutput: MLXArray,
        cache: [MLXArray?]? = nil
    ) -> (MLXArray, [MLXArray?]) {
        let seqLen = tokens.shape[1]
        let causal = causalMask(length: seqLen)

        var hiddenStates = tgtWordEmb(tokens) * MLXArray(scale) + positionalEncoding(length: seqLen)
        var newCaches: [MLXArray?] = []
        newCaches.reserveCapacity(layerStack.count)

        for (index, layer) in layerStack.enumerated() {
            let next = layer(
                hiddenStates,
                encoderOutput: encoderOutput,
                selfAttnMask: causal,
                cache: cache?[safe: index] ?? nil
            )
            hiddenStates = next
            newCaches.append(next)
        }

        hiddenStates = layerNormOut(hiddenStates)
        return (hiddenStates, newCaches)
    }

    func callAsFunction(
        _ tokens: MLXArray,
        encoderOutput: MLXArray,
        cache: [MLXArray?]? = nil
    ) -> (MLXArray, [MLXArray?]) {
        let (hiddenStates, newCaches) = decoderState(
            tokens: tokens,
            encoderOutput: encoderOutput,
            cache: cache
        )
        return (tgtWordPrj(hiddenStates), newCaches)
    }

    func decodeOneStep(
        _ tokens: MLXArray,
        encoderOutput: MLXArray,
        cache: [MLXArray?]
    ) -> (MLXArray, [MLXArray?]) {
        let (hiddenStates, newCaches) = decoderState(
            tokens: tokens,
            encoderOutput: encoderOutput,
            cache: cache
        )
        let logits = tgtWordPrj(hiddenStates[0..., (-1)..., 0...].squeezed(axis: 1))
        return (logits, newCaches)
    }

    func beamSearch(
        encoderOutput: MLXArray,
        beamSize: Int = 3,
        maxLen: Int = 0,
        softmaxSmoothing: Float = 1.25,
        lengthPenalty: Float = 0.6,
        eosPenalty: Float = 1.0
    ) -> ([Int32], [Float]) {
        let beamCount = max(1, beamSize)
        let maxDecode = maxLen > 0 ? maxLen : encoderOutput.shape[1]
        let expandedEncoder = beamCount > 1
            ? MLX.repeated(encoderOutput, count: beamCount, axis: 0)
            : encoderOutput

        var tokens = MLX.full([beamCount, 1], values: Int32(sosID), type: Int32.self)
        var scores = Array(repeating: -Float.greatestFiniteMagnitude, count: beamCount)
        scores[0] = 0
        var isFinished = Array(repeating: false, count: beamCount)
        var caches = Array<MLXArray?>(repeating: nil, count: nLayers)
        var confidences = Array(repeating: [Float](), count: beamCount)

        for _ in 0..<maxDecode {
            let (logits, newCaches) = decodeOneStep(tokens, encoderOutput: expandedEncoder, cache: caches)
            caches = newCaches

            var stepScores = MLX.log(softmax(logits / MLXArray(softmaxSmoothing), axis: -1) + MLXArray(Float(1e-10)))
            if eosPenalty != 1.0 {
                let scaledEOS = stepScores[0..., eosID..<(eosID + 1)] * MLXArray(eosPenalty)
                var pieces: [MLXArray] = []
                if eosID > 0 {
                    pieces.append(stepScores[0..., 0..<eosID])
                }
                pieces.append(scaledEOS)
                if eosID + 1 < stepScores.shape[1] {
                    pieces.append(stepScores[0..., (eosID + 1)...])
                }
                stepScores = MLX.concatenated(pieces, axis: -1)
            }

            let (topScores, topTokens) = topK(stepScores, k: beamCount)
            let topScoreValues = topScores.asArray(Float.self)
            let topTokenValues = topTokens.asArray(Int32.self)

            struct Candidate {
                let totalScore: Float
                let beamIndex: Int
                let token: Int32
                let tokenScore: Float
            }

            var candidates: [Candidate] = []
            candidates.reserveCapacity(beamCount * beamCount)

            for beam in 0..<beamCount {
                if isFinished[beam] {
                    candidates.append(
                        Candidate(
                            totalScore: scores[beam],
                            beamIndex: beam,
                            token: Int32(eosID),
                            tokenScore: 0
                        )
                    )
                    for _ in 1..<beamCount {
                        candidates.append(
                            Candidate(
                                totalScore: -Float.greatestFiniteMagnitude,
                                beamIndex: beam,
                                token: Int32(eosID),
                                tokenScore: -Float.greatestFiniteMagnitude
                            )
                        )
                    }
                    continue
                }

                for rank in 0..<beamCount {
                    let flatIndex = beam * beamCount + rank
                    let tokenScore = topScoreValues[flatIndex]
                    candidates.append(
                        Candidate(
                            totalScore: scores[beam] + tokenScore,
                            beamIndex: beam,
                            token: topTokenValues[flatIndex],
                            tokenScore: tokenScore
                        )
                    )
                }
            }

            candidates.sort { lhs, rhs in
                lhs.totalScore > rhs.totalScore
            }
            let chosen = Array(candidates.prefix(beamCount))

            let beamIndices = MLXArray(chosen.map { Int32($0.beamIndex) })
            let newTokenArray = MLXArray(chosen.map(\.token)).reshaped([beamCount, 1])

            tokens = take(tokens, beamIndices, axis: 0)
            tokens = MLX.concatenated([tokens, newTokenArray], axis: 1)
            caches = caches.map { cache in
                cache.map { take($0, beamIndices, axis: 0) }
            }

            var nextScores = Array(repeating: -Float.greatestFiniteMagnitude, count: beamCount)
            var nextFinished = Array(repeating: false, count: beamCount)
            var nextConfidences = Array(repeating: [Float](), count: beamCount)
            for (index, candidate) in chosen.enumerated() {
                nextScores[index] = candidate.totalScore
                nextFinished[index] = Int(candidate.token) == eosID
                nextConfidences[index] = confidences[candidate.beamIndex]
                nextConfidences[index].append(exp(candidate.tokenScore))
            }

            scores = nextScores
            isFinished = nextFinished
            confidences = nextConfidences

            if isFinished.allSatisfy({ $0 }) {
                break
            }
        }

        let tokenValues = tokens.asArray(Int32.self)
        let seqLen = tokens.shape[1]

        var bestBeam = 0
        var bestScore = -Float.greatestFiniteMagnitude
        for beam in 0..<beamCount {
            let start = beam * seqLen
            let beamTokens = Array(tokenValues[start..<(start + seqLen)])
            let length = beamTokens.reduce(into: 0) { partialResult, token in
                if Int(token) != eosID {
                    partialResult += 1
                }
            }

            let finalScore: Float
            if lengthPenalty > 0 {
                let penalty = pow((5.0 + Float(length)) / 6.0, lengthPenalty)
                finalScore = scores[beam] / penalty
            } else {
                finalScore = scores[beam]
            }

            if finalScore > bestScore {
                bestScore = finalScore
                bestBeam = beam
            }
        }

        let start = bestBeam * seqLen
        let bestTokens = Array(tokenValues[start..<(start + seqLen)])
        let sequence = Array(bestTokens.dropFirst())
        return (sequence, confidences[bestBeam])
    }
}

private struct FireRedASR2CMVN: Decodable {
    let means: [Float]
    let istd: [Float]
}

public final class FireRedASR2Model: Module, STTGenerationModel {
    public let config: FireRedASR2Config

    @ModuleInfo(key: "encoder") var encoder: FireRedASR2Encoder
    @ModuleInfo(key: "decoder") var decoder: FireRedASR2TransformerDecoder

    var tokenizer: FireRedASR2Tokenizer?
    var cmvnMeans: MLXArray?
    var cmvnIstd: MLXArray?

    public var vocabulary: [String] {
        tokenizer?.vocabulary ?? []
    }

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 0,
            temperature: 0.0,
            topP: 0.95,
            topK: 0,
            verbose: false,
            language: "English"
        )
    }

    public init(_ config: FireRedASR2Config) {
        self.config = config
        self._encoder.wrappedValue = FireRedASR2Encoder(config: config)
        self._decoder.wrappedValue = FireRedASR2TransformerDecoder(config: config)
    }

    public func encode(_ features: MLXArray) -> MLXArray {
        encoder(features)
    }

    public func callAsFunction(_ features: MLXArray, tokens: MLXArray) -> MLXArray {
        let encoderOutput = encode(features)
        return decoder(tokens, encoderOutput: encoderOutput).0
    }

    public func decodeOneStep(
        _ tokens: MLXArray,
        encoderOutput: MLXArray,
        cache: [MLXArray?] = []
    ) -> (MLXArray, [MLXArray?]) {
        let preparedCache = cache.count == decoder.nLayers
            ? cache
            : Array(repeating: Optional<MLXArray>.none, count: decoder.nLayers)
        return decoder.decodeOneStep(tokens, encoderOutput: encoderOutput, cache: preparedCache)
    }

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        generate(
            audio: audio,
            beamSize: 3,
            softmaxSmoothing: 1.25,
            lengthPenalty: 0.6,
            eosPenalty: 1.0,
            maxLen: max(generationParameters.maxTokens, 0),
            language: generationParameters.language
        )
    }

    public func generate(
        audio: MLXArray,
        beamSize: Int = 3,
        softmaxSmoothing: Float = 1.25,
        lengthPenalty: Float = 0.6,
        eosPenalty: Float = 1.0,
        maxLen: Int = 0,
        language: String? = nil
    ) -> STTOutput {
        let startTime = CFAbsoluteTimeGetCurrent()

        var features = FireRedASR2Audio.extractFbank(audio)
        if let cmvnMeans, let cmvnIstd {
            features = FireRedASR2Audio.applyCMVN(features, means: cmvnMeans, istd: cmvnIstd)
        }

        let batchedFeatures = features.expandedDimensions(axis: 0)
        let encoderOutput = encoder(batchedFeatures)
        eval(encoderOutput)

        let (sequence, confidenceScores) = decoder.beamSearch(
            encoderOutput: encoderOutput,
            beamSize: beamSize,
            maxLen: maxLen,
            softmaxSmoothing: softmaxSmoothing,
            lengthPenalty: lengthPenalty,
            eosPenalty: eosPenalty
        )

        let trimmedSequence: [Int32]
        if let eosIndex = sequence.firstIndex(where: { Int($0) == config.eosID }) {
            trimmedSequence = Array(sequence[..<eosIndex])
        } else {
            trimmedSequence = sequence
        }

        let text = tokenizer?.decode(tokenIds: trimmedSequence.map(Int.init)) ?? ""
        let confidence: Float
        if trimmedSequence.isEmpty {
            confidence = 0
        } else {
            let count = min(trimmedSequence.count, confidenceScores.count)
            confidence = confidenceScores.prefix(count).reduce(0, +) / Float(max(count, 1))
        }

        let roundedConfidence = (confidence * 1000).rounded() / 1000

        return STTOutput(
            text: text,
            segments: text.isEmpty ? nil : [[
                "text": text,
                "confidence": Double(roundedConfidence),
            ]],
            language: language,
            generationTokens: trimmedSequence.count,
            totalTokens: trimmedSequence.count,
            totalTime: CFAbsoluteTimeGetCurrent() - startTime
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let output = generate(audio: audio, generationParameters: generationParameters)
            continuation.yield(.result(output))
            continuation.finish()
        }
    }

    private func loadAssets(from modelDirectory: URL) throws {
        let cmvnURL = modelDirectory.appendingPathComponent("cmvn.json")
        if FileManager.default.fileExists(atPath: cmvnURL.path) {
            let data = try Data(contentsOf: cmvnURL)
            let cmvn = try JSONDecoder().decode(FireRedASR2CMVN.self, from: data)
            cmvnMeans = MLXArray(cmvn.means)
            cmvnIstd = MLXArray(cmvn.istd)
        }

        let dictURL = modelDirectory.appendingPathComponent("dict.txt")
        if FileManager.default.fileExists(atPath: dictURL.path) {
            tokenizer = try FireRedASR2Tokenizer(modelDirectory: modelDirectory)
        }
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            newKey = newKey.replacingOccurrences(
                of: "encoder.input_preprocessor.conv.0.",
                with: "encoder.input_preprocessor.conv1."
            )
            newKey = newKey.replacingOccurrences(
                of: "encoder.input_preprocessor.conv.2.",
                with: "encoder.input_preprocessor.conv2."
            )
            newKey = newKey.replacingOccurrences(
                of: #"\.net\.(\d+)\."#,
                with: ".net_$1.",
                options: .regularExpression
            )

            if (
                newKey.contains("pointwise_conv1.weight")
                    || newKey.contains("pointwise_conv2.weight")
                    || newKey.contains("depthwise_conv.weight")
            ) && newValue.ndim == 3 {
                newValue = newValue.transposed(0, 2, 1)
            } else if newKey.contains("input_preprocessor.conv"), newKey.contains("weight"), newValue.ndim == 4 {
                newValue = newValue.transposed(0, 2, 3, 1)
            }

            sanitized[newKey] = newValue
        }

        if sanitized["decoder.tgt_word_prj.weight"] == nil,
           let tied = sanitized["decoder.tgt_word_emb.weight"]
        {
            sanitized["decoder.tgt_word_prj.weight"] = tied
        }

        return sanitized
    }

    public static func fromDirectory(_ modelDirectory: URL) throws -> FireRedASR2Model {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(FireRedASR2Config.self, from: configData)

        let model = FireRedASR2Model(config)
        try model.loadAssets(from: modelDirectory)

        let files = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        let safetensors = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var weights: [String: MLXArray] = [:]
        for file in safetensors {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }

        let sanitized = sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        return model
    }

    public static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> FireRedASR2Model {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "FireRedASR2Model",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDirectory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            cache: cache
        )
        return try fromDirectory(modelDirectory)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}
