import Accelerate
import MLX
import MLXNN

// MARK: - Compound Layer Helpers

extension DeepFilterNetModel {

    func applyEncoderConv(
        _ x: MLXArray,
        prefix: String,
        main: Int,
        pointwise: Int?,
        bn: Int,
        fstride: Int
    ) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).\(main).weight",
            bias: nil,
            fstride: fstride,
            lookahead: 0
        )
        if let pointwise {
            y = try conv2dLayer(
                y,
                weightKey: "\(prefix).\(pointwise).weight",
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }
        y = try batchNorm(y, prefix: "\(prefix).\(bn)")
        return relu(y)
    }

    func applyPathwayConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return relu(y)
    }

    func applyTransposeBlock(_ x: MLXArray, prefix: String, fstride: Int) throws -> MLXArray {
        var y = try convTranspose2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            fstride: fstride,
            groups: config.convCh
        )
        y = try conv2dLayer(
            y,
            weightKey: "\(prefix).1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    func applyRegularBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try conv2dLayer(
            y,
            weightKey: "\(prefix).1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    func applyOutputConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return y
    }
}

// MARK: - Conv2d

extension DeepFilterNetModel {

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weightKey: String,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        if let wOHWI = conv2dWeightsOHWI[weightKey] {
            return conv2dLayer(
                xBCHW,
                weightOHWI: wOHWI,
                bias: bias,
                fstride: fstride,
                lookahead: lookahead
            )
        }
        return try conv2dLayer(
            xBCHW,
            weight: w(weightKey),
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        conv2dLayer(
            xBCHW,
            weightOHWI: weight.transposed(0, 2, 3, 1),
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weightOHWI: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) -> MLXArray {
        let kT = weightOHWI.shape[1]
        let kF = weightOHWI.shape[2]
        let inPerGroup = weightOHWI.shape[3]
        let inChannels = xBCHW.shape[1]
        let groups = max(1, inChannels / max(1, inPerGroup))

        let rawLeft = kT - 1 - lookahead
        let timeCrop = max(0, -rawLeft)
        let timePadLeft = max(0, rawLeft)
        let timePadRight = max(0, lookahead)
        let freqPad = kF / 2

        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        if timeCrop > 0, x.shape[1] > timeCrop {
            x = x[0..., timeCrop..., 0..., 0...]
        }
        x = MLX.padded(
            x,
            widths: [
                .init(0),
                .init((timePadLeft, timePadRight)),
                .init((freqPad, freqPad)),
                .init(0),
            ],
            mode: .constant
        )

        var y = MLX.conv2d(x, weightOHWI, stride: [1, fstride], padding: [0, 0], groups: groups)
        if let bias {
            y = y + bias.reshaped([1, 1, 1, bias.shape[0]])
        }
        return y.transposed(0, 3, 1, 2)
    }
}

// MARK: - ConvTranspose2d

extension DeepFilterNetModel {

    func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weightKey: String,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        if groups > 1, let denseWeight = convTransposeDenseWeights[weightKey] {
            var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
            let kT = denseWeight.shape[1]
            let kF = denseWeight.shape[2]
            let padding = IntOrPair((kT - 1, kF / 2))
            let outputPadding = IntOrPair((0, kF / 2))
            x = MLX.convTransposed2d(
                x,
                denseWeight,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }
        if groups > 1, let groupedWeights = convTransposeGroupWeights[weightKey], groupedWeights.count == groups {
            return convTranspose2dLayer(
                xBCHW,
                groupedWeights: groupedWeights,
                fstride: fstride
            )
        }
        return try convTranspose2dLayer(
            xBCHW,
            weight: w(weightKey),
            fstride: fstride,
            groups: groups
        )
    }

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

    private func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        groupedWeights: [MLXArray],
        fstride: Int
    ) -> MLXArray {
        let groups = groupedWeights.count
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = groupedWeights[0].shape[1]
        let kF = groupedWeights[0].shape[2]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))
        let inPerGroup = max(1, x.shape[3] / groups)

        var ys = [MLXArray]()
        ys.reserveCapacity(groups)
        for (g, wT) in groupedWeights.enumerated() {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]
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
        x = MLX.concatenated(ys, axis: 3)
        return x.transposed(0, 3, 1, 2)
    }
}

// MARK: - BatchNorm, GRU, Linear

extension DeepFilterNetModel {

    func batchNorm(_ x: MLXArray, prefix: String) throws -> MLXArray {
        if let scale = bnScale[prefix], let bias = bnBias[prefix] {
            return x * scale + bias
        }

        let gamma = try w("\(prefix).weight")
        let beta = try w("\(prefix).bias")
        let mean = try w("\(prefix).running_mean")
        let variance = try w("\(prefix).running_var")
        let scale = (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5))))
            .reshaped([1, gamma.shape[0], 1, 1])
        let shift = (beta - mean * (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5)))))
            .reshaped([1, beta.shape[0], 1, 1])
        return x * scale + shift
    }

    func squeezedGRU(
        _ x: MLXArray,
        prefix: String,
        hiddenSize: Int,
        linearOut: Bool
    ) throws -> MLXArray {
        var y = relu(groupedLinear(x, weight: try w("\(prefix).linear_in.0.weight")))

        var layer = 0
        while weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
            y = try pytorchGRULayer(y, prefix: "\(prefix).gru", layer: layer, hiddenSize: hiddenSize)
            layer += 1
        }

        if linearOut, weights["\(prefix).linear_out.0.weight"] != nil {
            y = relu(groupedLinear(y, weight: try w("\(prefix).linear_out.0.weight")))
        }
        return y
    }

    func groupedLinear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let groups = weight.shape[0]
        let ws = weight.shape[1]
        let hs = weight.shape[2]
        let b = x.shape[0]
        let t = x.shape[1]
        let reshaped = x.reshaped([b, t, groups, ws])
        let out = MLX.einsum("btgi,gih->btgh", reshaped, weight)
        return out.reshaped([b, t, groups * hs])
    }

    func pytorchGRULayer(
        _ x: MLXArray,
        prefix: String,
        layer: Int,
        hiddenSize: Int
    ) throws -> MLXArray {
        let wihKey = "\(prefix).weight_ih_l\(layer)"
        let whhKey = "\(prefix).weight_hh_l\(layer)"
        let wihT: MLXArray
        if let cached = gruTransposedWeights[wihKey] {
            wihT = cached
        } else {
            wihT = try w(wihKey).transposed()
        }
        let whhT: MLXArray
        if let cached = gruTransposedWeights[whhKey] {
            whhT = cached
        } else {
            whhT = try w(whhKey).transposed()
        }
        let bih = try w("\(prefix).bias_ih_l\(layer)")
        let bhh = try w("\(prefix).bias_hh_l\(layer)")

        let batchSize = x.shape[0]
        let t = x.shape[1]
        let h3 = 3 * hiddenSize

        // Batch input projection on GPU: gxAll[B, T, 3H] = x @ wihT + bih
        let x2 = x.reshaped([batchSize * t, x.shape[2]])
        let gxAllMLX = (MLX.matmul(x2, wihT) + bih).reshaped([batchSize, t, h3])
        eval(gxAllMLX)

        // Extract to CPU
        let gxAll = gxAllMLX.asType(.float32).reshaped([-1]).asArray(Float.self)
        let wHH = whhT.asType(.float32).reshaped([-1]).asArray(Float.self)
        let biasHH = bhh.asType(.float32).reshaped([-1]).asArray(Float.self)

        let totalOut = batchSize * t * hiddenSize
        var output = Array<Float>(repeating: 0, count: totalOut)
        var state = Array<Float>(repeating: 0, count: batchSize * hiddenSize)
        var ghBuf = Array<Float>(repeating: 0, count: h3)

        for ti in 0..<t {
            for bi in 0..<batchSize {
                let stOff = bi * hiddenSize
                let gxOff = (bi * t + ti) * h3

                // gh = state @ wHH (hidden projection via Accelerate)
                state.withUnsafeBufferPointer { sBuf in
                    wHH.withUnsafeBufferPointer { wBuf in
                        ghBuf.withUnsafeMutableBufferPointer { gBuf in
                            vDSP_mmul(
                                sBuf.baseAddress! + stOff, 1,
                                wBuf.baseAddress!, 1,
                                gBuf.baseAddress!, 1,
                                vDSP_Length(1), vDSP_Length(h3), vDSP_Length(hiddenSize)
                            )
                        }
                    }
                }

                // GRU gates (PyTorch convention: (1-z)*n + z*h)
                for k in 0..<hiddenSize {
                    let xr = gxAll[gxOff + k]
                    let xz = gxAll[gxOff + hiddenSize + k]
                    let xn = gxAll[gxOff + 2 * hiddenSize + k]
                    let hr = ghBuf[k] + biasHH[k]
                    let hz = ghBuf[hiddenSize + k] + biasHH[hiddenSize + k]
                    let hn = ghBuf[2 * hiddenSize + k] + biasHH[2 * hiddenSize + k]

                    let r = 1.0 / (1.0 + expf(-(xr + hr)))
                    let z = 1.0 / (1.0 + expf(-(xz + hz)))
                    let n = tanhf(xn + r * hn)
                    state[stOff + k] = (1.0 - z) * n + z * state[stOff + k]
                }

                let outOff = (bi * t + ti) * hiddenSize
                output.replaceSubrange(outOff..<(outOff + hiddenSize), with: state[stOff..<(stOff + hiddenSize)])
            }
        }

        return MLXArray(output).reshaped([batchSize, t, hiddenSize])
    }

    func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let x2 = x.reshaped([b * t, x.shape[2]])
        var y = MLX.matmul(x2, weight.transposed())
        y = y + bias
        return y.reshaped([b, t, weight.shape[0]])
    }
}
