import Accelerate
import MLX
import MLXNN

// MARK: - Network Forward (V2/V3)

extension DeepFilterNetModel {

    func forward(
        spec: MLXArray,
        featErb: MLXArray,
        featSpec5D: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let featSpec = featSpec5D
            .squeezed(axis: 1)
            .transposed(0, 3, 1, 2)

        let featErbShift = applyLookahead(feature: featErb, lookahead: config.convLookahead)
        let featSpecShift = applyLookahead(feature: featSpec, lookahead: config.convLookahead)

        let (e0, e1, e2, e3, emb, c0, lsnr) = try encode(featErb: featErbShift, featSpec: featSpecShift)

        let mask = try decodeErb(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        let specMasked = applyMask(spec: spec, mask: mask)

        let dfCoefs = try decodeDf(emb: emb, c0: c0)
        let b = dfCoefs.shape[0]
        let t = dfCoefs.shape[1]
        let dfCoefs5 = dfCoefs
            .reshaped([b, t, config.nbDf, config.dfOrder, 2])
            .transposed(0, 3, 1, 2, 4)

        let specEnhanced: MLXArray
        if config.encConcat {
            specEnhanced = deepFilter(spec: specMasked, coefs: dfCoefs5)
        } else {
            let specDf = deepFilter(spec: spec, coefs: dfCoefs5)
            let low = specDf[0..., 0..., 0..., 0..<config.nbDf, 0...]
            let high = specMasked[0..., 0..., 0..., config.nbDf..., 0...]
            specEnhanced = MLX.concatenated([low, high], axis: 3)
        }

        return (specEnhanced, mask, lsnr, dfCoefs5)
    }

    private func encode(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyEncoderConv(featErb, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
        let e1 = try applyEncoderConv(e0, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e2 = try applyEncoderConv(e1, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e3 = try applyEncoderConv(e2, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

        let c0 = try applyEncoderConv(featSpec, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
        let c1 = try applyEncoderConv(c0, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

        let b = c1.shape[0]
        let t = c1.shape[2]
        var cemb = c1.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        cemb = relu(groupedLinear(cemb, weight: try w("enc.df_fc_emb.0.weight")))

        var emb = e3.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        emb = config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)

        emb = try squeezedGRU(
            emb,
            prefix: "enc.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let lsnr = sigmoid(linear(
            emb,
            weight: try w("enc.lsnr_fc.0.weight"),
            bias: try w("enc.lsnr_fc.0.bias")
        )) * MLXArray(Float(config.lsnrMax - config.lsnrMin)) + MLXArray(Float(config.lsnrMin))

        return (e0, e1, e2, e3, emb, c0, lsnr)
    }

    private func decodeErb(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        var embDec = try squeezedGRU(
            emb,
            prefix: "erb_dec.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let b = embDec.shape[0]
        let t = embDec.shape[1]
        let f8 = e3.shape[3]
        embDec = embDec.reshaped([b, t, f8, -1]).transposed(0, 3, 1, 2)

        var d3 = relu(try applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
        d3 = relu(try applyRegularBlock(d3, prefix: "erb_dec.convt3"))
        var d2 = relu(try applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
        d2 = relu(try applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
        var d1 = relu(try applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
        d1 = relu(try applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
        let d0 = relu(try applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
        let out = try applyOutputConv(d0, prefix: "erb_dec.conv0_out")
        return sigmoid(out)
    }

    private func decodeDf(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
        var c = try squeezedGRU(
            emb,
            prefix: "df_dec.df_gru",
            hiddenSize: config.dfHiddenDim,
            linearOut: false
        )

        if weights["df_dec.df_skip.weight"] != nil {
            c = c + groupedLinear(emb, weight: try w("df_dec.df_skip.weight"))
        }

        var c0p = try conv2dLayer(
            c0,
            weightKey: "df_dec.df_convp.1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = try conv2dLayer(
            c0p,
            weightKey: "df_dec.df_convp.2.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = relu(try batchNorm(c0p, prefix: "df_dec.df_convp.3"))
        c0p = c0p.transposed(0, 2, 3, 1)

        let b = c.shape[0]
        let t = c.shape[1]
        let dfOut = tanh(groupedLinear(c, weight: try w("df_dec.df_out.0.weight")))
            .reshaped([b, t, config.nbDf, config.dfOrder * 2])

        return dfOut + c0p
    }

    func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
        let b = mask.shape[0]
        let t = mask.shape[2]
        let e = mask.shape[3]
        let flat = mask.reshaped([b * t, e])
        let gains = MLX.matmul(flat, erbInvFB).reshaped([b, 1, t, config.freqBins, 1])
        return spec * gains
    }

    func deepFilter(spec: MLXArray, coefs: MLXArray, alpha: MLXArray? = nil) -> MLXArray {
        let t = spec.shape[2]
        let padLeft = config.dfOrder - 1 - config.dfLookahead
        let padRight = config.dfLookahead

        let specLow = spec[0..., 0, 0..., 0..<config.nbDf, 0...]
        let padded = MLX.padded(
            specLow,
            widths: [
                .init(0),
                .init((padLeft, padRight)),
                .init(0),
                .init(0),
            ],
            mode: .constant
        )

        let b = spec.shape[0]
        var outR = MLXArray.zeros([b, t, config.nbDf], dtype: spec.dtype)
        var outI = MLXArray.zeros([b, t, config.nbDf], dtype: spec.dtype)
        for k in 0..<config.dfOrder {
            let window = padded[0..., k..<(k + t), 0..., 0...]
            let coef = coefs[0..., k, 0..., 0..., 0...]
            let sr = window[0..., 0..., 0..., 0]
            let si = window[0..., 0..., 0..., 1]
            let cr = coef[0..., 0..., 0..., 0]
            let ci = coef[0..., 0..., 0..., 1]

            outR = outR + (sr * cr - si * ci)
            outI = outI + (sr * ci + si * cr)
        }

        var low = MLX.stacked([outR, outI], axis: -1).expandedDimensions(axis: 1)
        if let alpha {
            let b = spec.shape[0]
            let a = alpha.reshaped([b, 1, t, 1, 1])
            let origLow = spec[0..., 0..., 0..., 0..<config.nbDf, 0...]
            low = low * a + origLow * (MLXArray(Float(1.0)) - a)
        }
        let high = spec[0..., 0..., 0..., config.nbDf..., 0...]
        return MLX.concatenated([low, high], axis: 3)
    }
}

// MARK: - Network Forward (V1)

extension DeepFilterNetModel {

    func forwardV1(
        spec: MLXArray,
        featErb: MLXArray,
        featSpec5D: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let featSpec = featSpec5D
            .squeezed(axis: 1)
            .transposed(0, 3, 1, 2)

        let (e0, e1, e2, e3, emb, c0, lsnr) = try encodeV1(featErb: featErb, featSpec: featSpec)
        eval(emb, c0, lsnr)

        var mask = try decodeErbV1(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        mask = alignTimeAxis(mask, target: spec.shape[2], fillValue: 1.0, axis: 2)
        let specMasked = applyMask(spec: spec, mask: mask)
        eval(mask, specMasked)

        var (dfCoefsBTOF2, dfAlpha) = try decodeDfV1(emb: emb, c0: c0)
        dfCoefsBTOF2 = alignTimeAxis(dfCoefsBTOF2, target: spec.shape[2], fillValue: 0.0, axis: 1)
        dfAlpha = alignTimeAxis(dfAlpha, target: spec.shape[2], fillValue: 0.0, axis: 1)
        let dfCoefs5 = dfCoefsBTOF2.transposed(0, 2, 1, 3, 4)
        eval(dfCoefs5, dfAlpha)

        let specEnhanced = deepFilter(spec: specMasked, coefs: dfCoefs5, alpha: dfAlpha)
        return (specEnhanced, mask, lsnr, dfCoefs5)
    }

    private func encodeV1(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyV1ConvKxF(featErb, prefix: "enc.erb_conv0", fstride: 1, lookahead: config.convLookahead > 0 ? 1 : 0, activation: .relu)
        let e1 = try applyV1ConvKxF(e0, prefix: "enc.erb_conv1", fstride: 2, lookahead: config.convLookahead > 1 ? 1 : 0, activation: .relu)
        let e2 = try applyV1ConvKxF(e1, prefix: "enc.erb_conv2", fstride: 2, lookahead: config.convLookahead > 2 ? 1 : 0, activation: .relu)
        let e3 = try applyV1ConvKxF(e2, prefix: "enc.erb_conv3", fstride: 1, lookahead: 0, activation: .relu)

        let c0 = try applyV1ConvKxF(featSpec, prefix: "enc.clc_conv0", fstride: 1, lookahead: config.convLookahead, activation: .relu)
        let c1 = try applyV1ConvKxF(c0, prefix: "enc.clc_conv1", fstride: 2, lookahead: 0, activation: .relu)

        let t = c1.shape[2]
        let b = c1.shape[0]
        var cemb = c1.transposed(2, 0, 1, 3).reshaped([t, b, -1])
        cemb = relu(try groupedLinearV1(cemb, prefix: "enc.clc_fc_emb.layers"))

        var emb = e3.transposed(2, 0, 1, 3).reshaped([t, b, -1])
        emb = emb + cemb
        emb = try groupedGRUV1(
            emb,
            prefix: "enc.emb_gru.grus",
            numLayers: config.embNumLayers,
            addOutputs: true,
            shuffleBetweenLayers: config.groupShuffle
        )
        let embBT = emb.transposed(1, 0, 2)

        let lsnr = sigmoid(linear(
            embBT,
            weight: try w("enc.lsnr_fc.0.weight"),
            bias: try w("enc.lsnr_fc.0.bias")
        )) * MLXArray(Float(config.lsnrMax - config.lsnrMin)) + MLXArray(Float(config.lsnrMin))

        return (e0, e1, e2, e3, embBT, c0, lsnr)
    }

    private func decodeErbV1(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        let b = emb.shape[0]
        let t = emb.shape[1]
        let f8 = e3.shape[3]

        var embProj = relu(try groupedLinearV1(emb, prefix: "erb_dec.fc_emb.0.layers"))
        embProj = embProj.reshaped([b, t, -1, f8]).transposed(0, 2, 1, 3)

        let p3 = try applyV1ConvKxF(e3, prefix: "erb_dec.conv3p", fstride: 1, lookahead: 0, activation: .relu)
        var d3 = alignAndAdd(p3, embProj)
        d3 = try applyV1ConvKxF(d3, prefix: "erb_dec.convt3", fstride: 1, lookahead: 0, activation: .relu)

        let p2 = try applyV1ConvKxF(e2, prefix: "erb_dec.conv2p", fstride: 1, lookahead: 0, activation: .relu)
        var d2 = alignAndAdd(p2, d3)
        d2 = try applyV1ConvKxF(d2, prefix: "erb_dec.convt2", fstride: 2, lookahead: 0, activation: .relu)

        let p1 = try applyV1ConvKxF(e1, prefix: "erb_dec.conv1p", fstride: 1, lookahead: 0, activation: .relu)
        var d1 = alignAndAdd(p1, d2)
        d1 = try applyV1ConvKxF(d1, prefix: "erb_dec.convt1", fstride: 2, lookahead: 0, activation: .relu)

        let p0 = try applyV1ConvKxF(e0, prefix: "erb_dec.conv0p", fstride: 1, lookahead: 0, activation: .relu)
        let d0 = alignAndAdd(p0, d1)

        return try applyV1ConvKxF(
            d0,
            prefix: "erb_dec.conv0_out",
            fstride: 1,
            lookahead: 0,
            activation: .sigmoid,
            applyBatchNorm: false,
            allowBias: true
        )
    }

    private func decodeDfV1(emb: MLXArray, c0: MLXArray) throws -> (MLXArray, MLXArray) {
        let cTBI = try groupedGRUV1(
            emb.transposed(1, 0, 2),
            prefix: "clc_dec.clc_gru.grus",
            numLayers: config.dfNumLayers,
            addOutputs: true,
            shuffleBetweenLayers: config.groupShuffle
        )
        let c = cTBI.transposed(1, 0, 2)

        var c0p = try applyV1ConvKxF(c0, prefix: "clc_dec.clc_convp", fstride: 1, lookahead: 0, activation: .relu)
        c0p = c0p.transposed(0, 2, 1, 3)

        let b = c.shape[0]
        let t = c.shape[1]
        let alpha = sigmoid(linear(
            c,
            weight: try w("clc_dec.clc_fc_a.0.weight"),
            bias: try w("clc_dec.clc_fc_a.0.bias")
        ))

        var coefs = tanh(linear(
            c,
            weight: try w("clc_dec.clc_fc_out.0.weight"),
            bias: try w("clc_dec.clc_fc_out.0.bias")
        ))
        coefs = coefs.reshaped([b, t, config.dfOrder * 2, config.nbDf]) + c0p
        coefs = coefs.reshaped([b, t, config.dfOrder, 2, config.nbDf]).transposed(0, 1, 2, 4, 3)
        return (coefs, alpha)
    }
}

// MARK: - V1 Layer Helpers

extension DeepFilterNetModel {

    enum V1Activation {
        case none
        case relu
        case sigmoid
    }

    func applyV1ConvKxF(
        _ x: MLXArray,
        prefix: String,
        fstride: Int,
        lookahead: Int,
        activation: V1Activation,
        applyBatchNorm: Bool = true,
        allowBias: Bool = false
    ) throws -> MLXArray {
        let mainKey: String
        let transposed: Bool
        if weights["\(prefix).sconvt.weight"] != nil {
            mainKey = "\(prefix).sconvt.weight"
            transposed = true
        } else {
            mainKey = "\(prefix).sconv.weight"
            transposed = false
        }

        let mainWeight = try w(mainKey)
        var y: MLXArray
        if transposed {
            let groups = max(1, mainWeight.shape[0] / max(1, mainWeight.shape[1]))
            y = try convTranspose2dLayer(
                x,
                weight: mainWeight,
                fstride: fstride,
                groups: groups
            )
        } else {
            let bias = allowBias ? weights["\(prefix).sconv.bias"] : nil
            y = try conv2dLayer(
                x,
                weight: mainWeight,
                bias: bias,
                fstride: fstride,
                lookahead: lookahead
            )
        }

        if weights["\(prefix).1x1conv.weight"] != nil {
            y = try conv2dLayer(
                y,
                weightKey: "\(prefix).1x1conv.weight",
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }

        if applyBatchNorm, weights["\(prefix).norm.running_mean"] != nil {
            y = try batchNorm(y, prefix: "\(prefix).norm")
        }

        switch activation {
        case .none:
            return y
        case .relu:
            return relu(y)
        case .sigmoid:
            return sigmoid(y)
        }
    }

    // convTranspose2dLayer that takes a raw weight (not from cache) - needed for V1
    private func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))

        if groups <= 1 {
            let w = weight.transposed(1, 2, 3, 0)
            x = MLX.convTransposed2d(
                x,
                w,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }

        let inPerGroup = max(1, x.shape[3] / groups)
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)

        for g in 0..<groups {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]

            let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
            let wT = wg.transposed(1, 2, 3, 0)
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }

        return MLX.concatenated(ys, axis: 3).transposed(0, 3, 1, 2)
    }

    func groupedLinearV1(
        _ x: MLXArray,
        prefix: String
    ) throws -> MLXArray {
        if let pack = v1GroupedLinearPacks[prefix] {
            let b = x.shape[0]
            let t = x.shape[1]
            var y = MLX.einsum(
                "btgi,gio->btgo",
                x.reshaped([b, t, pack.groups, pack.inputPerGroup]),
                pack.weightGIO
            ) + pack.biasGO.reshaped([1, 1, pack.groups, pack.outputPerGroup])
            y = y.reshaped([b, t, pack.groups * pack.outputPerGroup])
            if config.groupShuffle, pack.groups > 1 {
                let hiddenPerGroup = y.shape[2] / pack.groups
                y = y
                    .reshaped([y.shape[0], y.shape[1], hiddenPerGroup, pack.groups])
                    .transposed(0, 1, 3, 2)
                    .reshaped([y.shape[0], y.shape[1], -1])
            }
            return y
        }

        let groups = max(1, config.linearGroups)
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)
        for g in 0..<groups {
            let wg = try w("\(prefix).\(g).weight")
            let bg = try w("\(prefix).\(g).bias")
            let inPerGroup = wg.shape[1]
            let start = g * inPerGroup
            let stop = start + inPerGroup
            let xg = x[0..., 0..., start..<stop]
            let x2 = xg.reshaped([-1, inPerGroup])
            let y2 = MLX.addMM(bg, x2, wg.transposed())
            ys.append(y2.reshaped([x.shape[0], x.shape[1], wg.shape[0]]))
        }
        var y = MLX.concatenated(ys, axis: 2)
        if config.groupShuffle, groups > 1 {
            let hiddenPerGroup = y.shape[2] / groups
            y = y
                .reshaped([y.shape[0], y.shape[1], hiddenPerGroup, groups])
                .transposed(0, 1, 3, 2)
                .reshaped([y.shape[0], y.shape[1], -1])
        }
        return y
    }

    func groupedGRUV1(
        _ xTBI: MLXArray,
        prefix: String,
        numLayers: Int,
        addOutputs: Bool,
        shuffleBetweenLayers: Bool
    ) throws -> MLXArray {
        if let pack = v1GroupedGRUPacks[prefix], pack.layers.count >= numLayers {
            return groupedGRUV1Packed(
                xTBI,
                pack: pack,
                numLayers: numLayers,
                addOutputs: addOutputs,
                shuffleBetweenLayers: shuffleBetweenLayers
            )
        }

        let groups = max(1, config.gruGroups)
        var cur = xTBI
        var out = xTBI

        for layer in 0..<numLayers {
            let base = "\(prefix).\(layer).layers"
            let w0 = try w("\(base).0.weight_ih_l0")
            let inPerGroup = w0.shape[1]
            let hiddenPerGroup = w0.shape[0] / 3
            var ys = [MLXArray]()
            ys.reserveCapacity(groups)
            for g in 0..<groups {
                let start = g * inPerGroup
                let stop = start + inPerGroup
                let xg = cur[0..., 0..., start..<stop]
                let yg = try gruCellSequenceV1(
                    xg,
                    weightIH: w("\(base).\(g).weight_ih_l0"),
                    weightHH: w("\(base).\(g).weight_hh_l0"),
                    biasIH: w("\(base).\(g).bias_ih_l0"),
                    biasHH: w("\(base).\(g).bias_hh_l0"),
                    hiddenSize: hiddenPerGroup
                )
                ys.append(yg)
            }

            var layerOut = MLX.concatenated(ys, axis: 2)
            if shuffleBetweenLayers && layer < numLayers - 1 && groups > 1 {
                let hidden = layerOut.shape[2] / groups
                layerOut = layerOut
                    .reshaped([layerOut.shape[0], layerOut.shape[1], hidden, groups])
                    .transposed(0, 1, 3, 2)
                    .reshaped([layerOut.shape[0], layerOut.shape[1], -1])
            }

            if addOutputs {
                out = (layer == 0) ? layerOut : (out + layerOut)
            } else {
                out = layerOut
            }
            cur = layerOut
        }

        return out
    }

    private func groupedGRUV1Packed(
        _ xTBI: MLXArray,
        pack: V1GroupedGRUPack,
        numLayers: Int,
        addOutputs: Bool,
        shuffleBetweenLayers: Bool
    ) -> MLXArray {
        let groups = pack.groups
        let t = xTBI.shape[0]
        let b = xTBI.shape[1]

        var curMLX = xTBI
        var outFlat: [Float]? = nil

        for layer in 0..<numLayers {
            let p = pack.layers[layer]
            let h = p.hiddenPerGroup
            let inpg = p.inputPerGroup
            let h3 = 3 * h
            let totalH = groups * h

            let x4 = curMLX.reshaped([t, b, groups, inpg])
            let gxAllMLX = MLX.einsum("tbgi,gio->tbgo", x4, p.weightIHGI3H)
                + p.biasIHG3H.reshaped([1, 1, groups, h3])
            eval(gxAllMLX)

            let gxAll = gxAllMLX.asType(.float32).reshaped([-1]).asArray(Float.self)
            let wHH = p.weightHHGH3H.asType(.float32).reshaped([-1]).asArray(Float.self)
            let bHH = p.biasHHG3H.asType(.float32).reshaped([-1]).asArray(Float.self)

            var output = Array<Float>(repeating: 0, count: t * b * totalH)
            var state = Array<Float>(repeating: 0, count: b * totalH)
            var gh = Array<Float>(repeating: 0, count: h3)

            for ti in 0..<t {
                for bi in 0..<b {
                    for g in 0..<groups {
                        let stOff = bi * totalH + g * h
                        let gxOff = ((ti * b + bi) * groups + g) * h3
                        let wHHOff = g * h * h3
                        let bHHOff = g * h3

                        state.withUnsafeBufferPointer { sBuf in
                            wHH.withUnsafeBufferPointer { wBuf in
                                gh.withUnsafeMutableBufferPointer { gBuf in
                                    vDSP_mmul(
                                        sBuf.baseAddress! + stOff, 1,
                                        wBuf.baseAddress! + wHHOff, 1,
                                        gBuf.baseAddress!, 1,
                                        vDSP_Length(1), vDSP_Length(h3), vDSP_Length(h)
                                    )
                                }
                            }
                        }

                        for k in 0..<h {
                            let xr = gxAll[gxOff + k]
                            let xz = gxAll[gxOff + h + k]
                            let xn = gxAll[gxOff + 2 * h + k]
                            let hr = gh[k] + bHH[bHHOff + k]
                            let hz = gh[h + k] + bHH[bHHOff + h + k]
                            let hn = gh[2 * h + k] + bHH[bHHOff + 2 * h + k]

                            let r = 1.0 / (1.0 + expf(-(xr + hr)))
                            let z = 1.0 / (1.0 + expf(-(xz + hz)))
                            let n = tanhf(xn + r * hn)
                            let prev = state[stOff + k]
                            state[stOff + k] = n + z * (prev - n)
                        }

                        let outOff = (ti * b + bi) * totalH + g * h
                        output.replaceSubrange(outOff..<(outOff + h), with: state[stOff..<(stOff + h)])
                    }
                }

            }

            if shuffleBetweenLayers && layer < numLayers - 1 && groups > 1 {
                var shuffled = Array<Float>(repeating: 0, count: output.count)
                for ti in 0..<t {
                    for bi in 0..<b {
                        let base = (ti * b + bi) * totalH
                        for i in 0..<totalH {
                            let dest = (i % groups) * h + (i / groups)
                            shuffled[base + dest] = output[base + i]
                        }
                    }
                }
                output = shuffled
            }

            if addOutputs {
                if layer == 0 {
                    outFlat = output
                } else {
                    for i in 0..<output.count {
                        outFlat![i] += output[i]
                    }
                }
            } else {
                outFlat = output
            }

            if layer < numLayers - 1 {
                curMLX = MLXArray(output).reshaped([t, b, totalH])
            }
        }

        return MLXArray(outFlat!).reshaped([t, b, groups * pack.layers[0].hiddenPerGroup])
    }

    private func gruCellSequenceV1(
        _ xTBI: MLXArray,
        weightIH: @autoclosure () throws -> MLXArray,
        weightHH: @autoclosure () throws -> MLXArray,
        biasIH: @autoclosure () throws -> MLXArray,
        biasHH: @autoclosure () throws -> MLXArray,
        hiddenSize: Int
    ) throws -> MLXArray {
        let wihT = try weightIH().transposed()
        let whhT = try weightHH().transposed()
        let bih = try biasIH()
        let bhh = try biasHH()

        var h = MLXArray.zeros([xTBI.shape[1], hiddenSize], type: Float.self)
        var states = [MLXArray]()
        states.reserveCapacity(xTBI.shape[0])

        for i in 0..<xTBI.shape[0] {
            let xt = xTBI[i, 0..., 0...]
            let gx = MLX.addMM(bih, xt, wihT)
            let gh = MLX.addMM(bhh, h, whhT)

            let xr = gx[0..., 0..<hiddenSize]
            let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
            let xn = gx[0..., (2 * hiddenSize)...]
            let hr = gh[0..., 0..<hiddenSize]
            let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
            let hn = gh[0..., (2 * hiddenSize)...]

            let r = sigmoid(xr + hr)
            let z = sigmoid(xz + hz)
            let n = tanh(xn + r * hn)
            h = n + z * (h - n)
            states.append(h)
        }
        return MLX.stacked(states, axis: 0)
    }

    private func alignAndAdd(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let t = min(a.shape[2], b.shape[2])
        let f = min(a.shape[3], b.shape[3])
        return a[0..., 0..., 0..<t, 0..<f] + b[0..., 0..., 0..<t, 0..<f]
    }

    private func alignTimeAxis(
        _ x: MLXArray,
        target: Int,
        fillValue: Float,
        axis: Int
    ) -> MLXArray {
        let t = x.shape[axis]
        if t == target { return x }
        if t > target {
            switch axis {
            case 1:
                if x.ndim == 3 {
                    return x[0..., 0..<target, 0...]
                }
                return x[0..., 0..<target, 0..., 0..., 0...]
            case 2:
                if x.ndim == 4 {
                    return x[0..., 0..., 0..<target, 0...]
                }
                return x[0..., 0..., 0..<target, 0..., 0...]
            default:
                return x
            }
        }
        switch axis {
        case 1:
            if x.ndim == 3 {
                let pad = MLX.full([x.shape[0], target - t, x.shape[2]], values: fillValue)
                return MLX.concatenated([x, pad], axis: 1)
            }
            let pad = MLX.full([x.shape[0], target - t, x.shape[2], x.shape[3], x.shape[4]], values: fillValue)
            return MLX.concatenated([x, pad], axis: 1)
        case 2:
            if x.ndim == 4 {
                let pad = MLX.full([x.shape[0], x.shape[1], target - t, x.shape[3]], values: fillValue)
                return MLX.concatenated([x, pad], axis: 2)
            }
            let pad = MLX.full([x.shape[0], x.shape[1], target - t, x.shape[3], x.shape[4]], values: fillValue)
            return MLX.concatenated([x, pad], axis: 2)
        default:
            return x
        }
    }
}
