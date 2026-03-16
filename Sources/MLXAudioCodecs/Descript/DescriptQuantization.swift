import Foundation
import MLX
import MLXNN

public typealias DescriptWNConv1d = BigVGANWNConv1d
public typealias DescriptWNConvTranspose1d = BigVGANWNConvTranspose1d

func descriptNormalize(_ x: MLXArray, dim: Int = 1, eps: Float = 1e-12) -> MLXArray {
    let norm = MLX.sqrt(MLX.sum(x * x, axis: dim, keepDims: true))
    return x / MLX.maximum(norm, MLXArray(eps))
}

public final class DescriptSnake1d: Module, UnaryLayer {
    public var alpha: MLXArray

    public init(channels: Int) {
        self.alpha = MLXArray.ones([1, 1, channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let recip = 1.0 / (alpha + 1e-9)
        let sine = MLX.sin(alpha * x)
        return x + recip * (sine * sine)
    }
}

public final class DescriptVectorQuantize: Module {
    public let codebookSize: Int
    public let codebookDim: Int

    @ModuleInfo(key: "in_proj") public var inProj: DescriptWNConv1d
    @ModuleInfo(key: "out_proj") public var outProj: DescriptWNConv1d
    @ModuleInfo(key: "codebook") public var codebook: Embedding

    public init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self._inProj = ModuleInfo(wrappedValue: DescriptWNConv1d(
            inChannels: inputDim,
            outChannels: codebookDim,
            kernelSize: 1
        ))
        self._outProj = ModuleInfo(wrappedValue: DescriptWNConv1d(
            inChannels: codebookDim,
            outChannels: inputDim,
            kernelSize: 1
        ))
        self._codebook = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: codebookSize,
            dimensions: codebookDim
        ))
    }

    public func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let zE = inProj(z.transposed(0, 2, 1)).transposed(0, 2, 1)
        let (zQ, indices) = decodeLatents(zE)

        let diffEQ = zE - zQ
        let commitmentLoss = MLX.mean(diffEQ * diffEQ, axes: [1, 2])
        let diffQE = zQ - zE
        let codebookLoss = MLX.mean(diffQE * diffQE, axes: [1, 2])

        let zQST = zE + MLX.stopGradient(zQ - zE)
        let projected = outProj(zQST.transposed(0, 2, 1)).transposed(0, 2, 1)
        return (projected, commitmentLoss, codebookLoss, indices, zE)
    }

    public func embedCode(_ embedId: MLXArray) -> MLXArray {
        codebook.weight[embedId]
    }

    public func decodeCode(_ embedId: MLXArray) -> MLXArray {
        embedCode(embedId).transposed(0, 2, 1)
    }

    public func decodeLatents(_ latents: MLXArray) -> (MLXArray, MLXArray) {
        let batch = latents.shape[0]
        let dim = latents.shape[1]
        let time = latents.shape[2]

        let encodings = latents.transposed(0, 2, 1).reshaped([batch * time, dim])
        let codebookWeights = codebook.weight

        let encNorm = descriptNormalize(encodings)
        let codeNorm = descriptNormalize(codebookWeights)

        let dist = MLX.sum(encNorm * encNorm, axis: 1, keepDims: true)
            - 2 * MLX.matmul(encNorm, codeNorm.T)
            + MLX.sum(codeNorm * codeNorm, axis: 1, keepDims: true).T

        let indices = MLX.argMax(-dist, axis: 1).reshaped([batch, time])
        return (decodeCode(indices), indices)
    }
}

public final class DescriptResidualVectorQuantize: Module {
    public let nCodebooks: Int
    public let codebookDim: [Int]
    public let codebookSize: Int

    @ModuleInfo(key: "quantizers") public var quantizers: [DescriptVectorQuantize]

    public init(
        inputDim: Int = 512,
        nCodebooks: Int = 9,
        codebookSize: Int = 1024,
        codebookDim: Int = 8
    ) {
        self.nCodebooks = nCodebooks
        self.codebookDim = Array(repeating: codebookDim, count: nCodebooks)
        self.codebookSize = codebookSize
        self._quantizers = ModuleInfo(wrappedValue: (0..<nCodebooks).map { _ in
            DescriptVectorQuantize(
                inputDim: inputDim,
                codebookSize: codebookSize,
                codebookDim: codebookDim
            )
        })
    }

    public func callAsFunction(_ z: MLXArray, nQuantizers: Int? = nil) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let activeQuantizers = nQuantizers ?? self.nCodebooks

        var zQ = MLXArray(0.0)
        var residual = z
        var commitmentLoss = MLXArray(0.0)
        var codebookLoss = MLXArray(0.0)
        var codebookIndices: [MLXArray] = []
        var latents: [MLXArray] = []

        for (index, quantizer) in quantizers.enumerated() where index < activeQuantizers {
            let (zQI, commitmentI, codebookI, indicesI, zEI) = quantizer(residual)
            zQ = zQ + zQI
            residual = residual - zQI
            commitmentLoss = commitmentLoss + MLX.mean(commitmentI)
            codebookLoss = codebookLoss + MLX.mean(codebookI)
            codebookIndices.append(indicesI)
            latents.append(zEI)
        }

        return (
            zQ,
            MLX.stacked(codebookIndices, axis: 1),
            MLX.concatenated(latents, axis: 1),
            commitmentLoss,
            codebookLoss
        )
    }

    public func fromCodes(_ codes: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        var zQ = MLXArray(0.0)
        var zP: [MLXArray] = []
        let codebookCount = codes.shape[1]

        for index in 0..<codebookCount {
            let zPI = quantizers[index].decodeCode(codes[0..., index, 0...])
            zP.append(zPI)
            let projected = quantizers[index].outProj(zPI.transposed(0, 2, 1)).transposed(0, 2, 1)
            zQ = zQ + projected
        }

        return (zQ, MLX.concatenated(zP, axis: 1), codes)
    }

    public func fromLatents(_ latents: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        var zQ = MLXArray(0.0)
        var zP: [MLXArray] = []
        var codes: [MLXArray] = []

        var dims = [0]
        for dim in codebookDim {
            dims.append(dims.last! + dim)
        }

        let availableCodebooks = dims.indices.dropLast().filter { dims[$0 + 1] <= latents.shape[1] }
        for index in availableCodebooks {
            let start = dims[index]
            let end = dims[index + 1]
            let slice = latents[0..., start..<end, 0...]
            let (zPI, codesI) = quantizers[index].decodeLatents(slice)
            zP.append(zPI)
            codes.append(codesI)
            let projected = quantizers[index].outProj(zPI.transposed(0, 2, 1)).transposed(0, 2, 1)
            zQ = zQ + projected
        }

        return (zQ, MLX.concatenated(zP, axis: 1), MLX.stacked(codes, axis: 1))
    }
}
