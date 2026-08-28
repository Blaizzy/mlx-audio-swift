import Foundation
@preconcurrency import MLX

struct DramaBoxGuiderParams: Sendable {
    var cfgScale: Float
    var stgScale: Float
    var stgBlocks: [Int]
    var rescaleScale: Float
    var modalityScale: Float

    init(
        cfgScale: Float = 2.5,
        stgScale: Float = 1.5,
        stgBlocks: [Int] = [29],
        rescaleScale: Float = 0,
        modalityScale: Float = 1.0
    ) {
        self.cfgScale = cfgScale
        self.stgScale = stgScale
        self.stgBlocks = stgBlocks
        self.rescaleScale = rescaleScale
        self.modalityScale = modalityScale
    }

    var needsUncond: Bool { abs(cfgScale - 1.0) > 1e-6 }
    var needsPtb: Bool { abs(stgScale) > 1e-6 }
    var needsModality: Bool { abs(modalityScale - 1.0) > 1e-6 }

    var stgBlockSet: Set<Int> { Set(stgBlocks) }
}

struct DramaBoxMultiModalGuider {
    var params: DramaBoxGuiderParams

    func callAsFunction(
        cond: MLXArray,
        uncond: MLXArray? = nil,
        ptb: MLXArray? = nil,
        modality: MLXArray? = nil
    ) -> MLXArray {
        var pred = cond
        if params.needsUncond {
            guard let uncond else {
                fatalError("uncond required for non-unit cfgScale")
            }
            pred = pred + (params.cfgScale - 1.0) * (cond - uncond)
        }
        if params.needsPtb {
            guard let ptb else {
                fatalError("ptb required for non-zero stgScale")
            }
            pred = pred + params.stgScale * (cond - ptb)
        }
        if params.needsModality {
            guard let modality else {
                fatalError("modality required for non-unit modalityScale")
            }
            pred = pred + (params.modalityScale - 1.0) * (cond - modality)
        }
        if params.rescaleScale != 0 {
            let condStd = MLX.std(cond.asType(.float32))
            let predStd = MLX.std(pred.asType(.float32)) + 1e-8
            var factor = condStd / predStd
            factor = params.rescaleScale * factor + (1.0 - params.rescaleScale)
            pred = pred * factor.asType(pred.dtype)
        }
        return pred
    }
}
