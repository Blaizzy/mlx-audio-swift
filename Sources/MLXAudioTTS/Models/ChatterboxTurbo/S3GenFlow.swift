//
//  S3GenFlow.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom

class ConditionalCFM: Module {
    let inChannels: Int
    let sigmaMin: Float
    let tScheduler: String
    let inferenceCfgRate: Float
    @ModuleInfo(key: "estimator") var estimator: ConditionalDecoder

    init(
        inChannels: Int = 240,
        sigmaMin: Float = 1e-6,
        tScheduler: String = "cosine",
        inferenceCfgRate: Float = 0.7,
        estimator: ConditionalDecoder
    ) {
        self.inChannels = inChannels
        self.sigmaMin = sigmaMin
        self.tScheduler = tScheduler
        self.inferenceCfgRate = inferenceCfgRate
        self._estimator.wrappedValue = estimator
    }

    func callAsFunction(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        noisedMels: MLXArray? = nil,
        meanflow: Bool = false
    ) -> (MLXArray, MLXArray?) {
        var z = MLXRandom.normal(mu.shape) * MLXArray(temperature)
        if let noisedMels {
            let promptLen = mu.shape[2] - noisedMels.shape[2]
            let prefix = z[0..., 0..., 0..<promptLen]
            z = MLX.concatenated([prefix, noisedMels], axis: 2)
        }

        var tSpan = linspace(Float(0.0), Float(1.0), count: nTimesteps + 1)
        if !meanflow, tScheduler == "cosine" {
            tSpan = MLXArray(Float(1.0)) - MLX.cos(tSpan * MLXArray(Float(0.5 * Float.pi)))
        }

        if meanflow {
            return (basicEuler(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond), nil)
        }

        return (solveEulerCFG(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond, meanflow: meanflow), nil)
    }

    private func basicEuler(
        x: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        var state = x
        let count = tSpan.shape[0]
        for i in 0..<(count - 1) {
            let t = tSpan[i..<(i + 1)]
            let r = tSpan[(i + 1)..<(i + 2)]
            let dxdt = estimator(
                x: state,
                mask: mask,
                mu: mu,
                t: t,
                spks: spks,
                cond: cond,
                r: r
            )
            let dt = r - t
            state = state + dt * dxdt
        }
        return state
    }

    private func solveEulerCFG(
        x: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?,
        meanflow: Bool
    ) -> MLXArray {
        var state = x
        let batch = mu.shape[0]

        let count = tSpan.shape[0]
        for i in 0..<(count - 1) {
            let t = tSpan[i..<(i + 1)]
            let r = tSpan[(i + 1)..<(i + 2)]

            let xIn = MLX.concatenated([state, state], axis: 0)
            let maskIn = MLX.concatenated([mask, mask], axis: 0)
            let muIn = MLX.concatenated([mu, MLXArray.zeros(mu.shape, type: Float.self)], axis: 0)
            let tIn = t.broadcasted(to: [batch * 2])
            let rIn = r.broadcasted(to: [batch * 2])

            let spksIn = spks != nil ? MLX.concatenated([spks!, MLXArray.zeros(spks!.shape, type: Float.self)], axis: 0) : nil
            let condIn = cond != nil ? MLX.concatenated([cond!, MLXArray.zeros(cond!.shape, type: Float.self)], axis: 0) : nil

            let dxdt = estimator(
                x: xIn,
                mask: maskIn,
                mu: muIn,
                t: tIn,
                spks: spksIn,
                cond: condIn,
                r: meanflow ? rIn : nil
            )

            let splits = dxdt.split(parts: 2, axis: 0)
            let dxdtCond = splits[0]
            let dxdtUncond = splits[1]

            let guided = (MLXArray(Float(1.0 + inferenceCfgRate)) * dxdtCond)
                - MLXArray(Float(inferenceCfgRate)) * dxdtUncond

            let dt = r - t
            state = state + dt * guided
        }

        return state
    }
}

final class CausalConditionalCFM: ConditionalCFM {
    override func callAsFunction(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        noisedMels: MLXArray? = nil,
        meanflow: Bool = false
    ) -> (MLXArray, MLXArray?) {
        super.callAsFunction(
            mu: mu,
            mask: mask,
            nTimesteps: nTimesteps,
            temperature: temperature,
            spks: spks,
            cond: cond,
            noisedMels: noisedMels,
            meanflow: meanflow
        )
    }
}
