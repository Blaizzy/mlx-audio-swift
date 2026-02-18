import Foundation
import MLX
import MLXNN

class ParakeetMultiHeadAttention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float
    let nFeat: Int

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(nHead: Int, nFeat: Int, bias: Bool = true) {
        self.nHead = nHead
        self.headDim = nFeat / nHead
        self.scale = pow(Float(headDim), -0.5)
        self.nFeat = nFeat

        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: bias)
    }

    func callAsFunction(
        _ q: MLXArray,
        _ k: MLXArray,
        _ v: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let qProj = linearQ(q)
        let kProj = linearK(k)
        let vProj = linearV(v)

        let batch = qProj.shape[0]
        let qSeq = qProj.shape[1]
        let kSeq = kProj.shape[1]

        let qHeads = qProj.reshaped(batch, qSeq, nHead, headDim).transposed(0, 2, 1, 3)
        let kHeads = kProj.reshaped(batch, kSeq, nHead, headDim).transposed(0, 2, 1, 3)
        let vHeads = vProj.reshaped(batch, kSeq, nHead, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask != nil ? .array(mask!) : .none
        let attended = MLXFast.scaledDotProductAttention(
            queries: qHeads,
            keys: kHeads,
            values: vHeads,
            scale: scale,
            mask: maskMode
        )

        let out = attended.transposed(0, 2, 1, 3).reshaped(batch, qSeq, nFeat)
        return linearOut(out)
    }
}

final class ParakeetRelPositionMultiHeadAttention: ParakeetMultiHeadAttention {
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    var posBiasU: MLXArray
    var posBiasV: MLXArray

    init(
        nHead: Int,
        nFeat: Int,
        bias: Bool = true,
        posBiasU: MLXArray? = nil,
        posBiasV: MLXArray? = nil
    ) {
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self.posBiasU = posBiasU ?? MLXArray.zeros([nHead, nFeat / nHead], type: Float.self)
        self.posBiasV = posBiasV ?? MLXArray.zeros([nHead, nFeat / nHead], type: Float.self)
        super.init(nHead: nHead, nFeat: nFeat, bias: bias)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let h = x.shape[1]
        let tq = x.shape[2]
        let posLen = x.shape[3]

        let padded = MLX.padded(x, widths: [.init(0), .init(0), .init(0), .init((1, 0))])
        let reshaped = padded.reshaped([b, h, posLen + 1, tq])
        let shifted = reshaped[0..., 0..., 1..., 0...]
        return shifted.reshaped([b, h, tq, posLen])
    }

    func callAsFunction(
        _ q: MLXArray,
        _ k: MLXArray,
        _ v: MLXArray,
        posEmb: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let qProj = linearQ(q)
        let kProj = linearK(k)
        let vProj = linearV(v)
        let pProj = linearPos(posEmb)

        let batch = qProj.shape[0]
        let qSeq = qProj.shape[1]
        let kSeq = kProj.shape[1]
        let posLen = pProj.shape[1]

        let qHeads = qProj.reshaped(batch, qSeq, nHead, headDim)
        let qU = (qHeads + posBiasU).transposed(0, 2, 1, 3)
        let qV = (qHeads + posBiasV).transposed(0, 2, 1, 3)

        let kHeads = kProj.reshaped(batch, kSeq, nHead, headDim).transposed(0, 2, 1, 3)
        let vHeads = vProj.reshaped(batch, kSeq, nHead, headDim).transposed(0, 2, 1, 3)
        let pHeads = pProj.reshaped(batch, posLen, nHead, headDim).transposed(0, 2, 1, 3)

        var matrixBD = MLX.matmul(qV, pHeads.swappedAxes(-2, -1))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., ..<kSeq] * scale

        if let mask {
            matrixBD = matrixBD + mask
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: qU,
            keys: kHeads,
            values: vHeads,
            scale: scale,
            mask: .array(matrixBD)
        )
        let out = attended.transposed(0, 2, 1, 3).reshaped(batch, qSeq, -1)
        return linearOut(out)
    }
}

final class ParakeetRelPositionalEncoding {
    let dModel: Int
    var maxLen: Int
    let scaleInput: Float
    var pe: MLXArray

    init(dModel: Int, maxLen: Int = 5000, scaleInput: Bool = true) {
        self.dModel = dModel
        self.maxLen = maxLen
        self.scaleInput = scaleInput ? sqrt(Float(dModel)) : 1.0
        self.pe = MLXArray.zeros([1, max(1, 2 * maxLen - 1), dModel], type: Float.self)
        calculatePE()
    }

    private func calculatePE() {
        let rows = 2 * maxLen - 1
        var values = [Float](repeating: 0, count: rows * dModel)
        let logDiv = Float(log(10000.0)) / Float(dModel)

        for r in 0..<rows {
            let pos = Float(maxLen - 1 - r)
            for c in stride(from: 0, to: dModel, by: 2) {
                let div = exp(-Float(c) * logDiv)
                let angle = pos * div
                values[r * dModel + c] = sin(angle)
                if c + 1 < dModel {
                    values[r * dModel + c + 1] = cos(angle)
                }
            }
        }

        pe = MLXArray(values).reshaped([1, rows, dModel])
        eval(pe)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let inputLength = x.shape[1] + offset
        if inputLength > maxLen {
            maxLen = inputLength + 1
            calculatePE()
        }

        let scaledX = x * scaleInput
        let bufferLen = pe.shape[1]
        let start = bufferLen / 2 - (inputLength - 1)
        let end = bufferLen / 2 + (inputLength - 1) + 1
        let posEmb = pe[0..., start..<end, 0...].asType(x.dtype)
        return (scaledX, posEmb)
    }
}
