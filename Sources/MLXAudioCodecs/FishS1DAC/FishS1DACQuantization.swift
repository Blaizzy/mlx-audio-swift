import Foundation
import MLX
import MLXNN

func fishS1NormalizeRows(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
    let norm = MLX.sqrt(MLX.sum(x * x, axis: 1, keepDims: true))
    return x / MLX.maximum(norm, MLXArray(eps))
}

public struct FishS1VQResult {
    public let z: MLXArray
    public let codes: MLXArray
    public let latents: MLXArray
    public let codebookLoss: MLXArray
    public let commitmentLoss: MLXArray

    public init(
        z: MLXArray,
        codes: MLXArray,
        latents: MLXArray,
        codebookLoss: MLXArray,
        commitmentLoss: MLXArray
    ) {
        self.z = z
        self.codes = codes
        self.latents = latents
        self.codebookLoss = codebookLoss
        self.commitmentLoss = commitmentLoss
    }
}

public final class FishS1VectorQuantize: Module {
    public let codebookSize: Int
    public let codebookDim: Int

    @ModuleInfo(key: "in_proj") public var inProj: FishS1WNConv1d
    @ModuleInfo(key: "out_proj") public var outProj: FishS1WNConv1d
    @ModuleInfo(key: "codebook") public var codebook: Embedding

    public init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self._inProj = ModuleInfo(wrappedValue: FishS1WNConv1d(
            inChannels: inputDim,
            outChannels: codebookDim,
            kernelSize: 1
        ))
        self._outProj = ModuleInfo(wrappedValue: FishS1WNConv1d(
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
        let zE = inProj(z)
        let (zQ, indices) = decodeLatents(zE)

        let commitmentLoss = MLX.mean(MLX.square(zE - zQ), axes: [1, 2])
        let codebookLoss = MLX.mean(MLX.square(zQ - zE), axes: [1, 2])

        let zQST = zE + stopGradient(zQ - zE)
        let projected = outProj(zQST)
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

        let encNorm = fishS1NormalizeRows(encodings)
        let codeNorm = fishS1NormalizeRows(codebookWeights)

        let dist = MLX.sum(encNorm * encNorm, axis: 1, keepDims: true)
            - 2 * MLX.matmul(encNorm, codeNorm.T)
            + MLX.sum(codeNorm * codeNorm, axis: 1, keepDims: true).T

        let indices = MLX.argMax(-dist, axis: 1).reshaped([batch, time])
        return (decodeCode(indices), indices)
    }
}

public final class FishS1ResidualVectorQuantize: Module {
    public let nCodebooks: Int
    public let codebookDim: [Int]
    public let codebookSize: Int

    @ModuleInfo(key: "quantizers") public var quantizers: [FishS1VectorQuantize]

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
            FishS1VectorQuantize(
                inputDim: inputDim,
                codebookSize: codebookSize,
                codebookDim: codebookDim
            )
        })
    }

    public func callAsFunction(_ z: MLXArray, nQuantizers: Int? = nil) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let active = nQuantizers ?? self.nCodebooks

        var zQ = MLXArray(0.0)
        var residual = z
        var commitmentLoss = MLXArray(0.0)
        var codebookLoss = MLXArray(0.0)
        var codebookIndices: [MLXArray] = []
        var latents: [MLXArray] = []

        for (index, quantizer) in quantizers.enumerated() where index < active {
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
        let count = codes.shape[1]

        for index in 0..<count {
            let zPI = quantizers[index].decodeCode(codes[0..., index, 0...])
            zP.append(zPI)
            zQ = zQ + quantizers[index].outProj(zPI)
        }

        return (zQ, MLX.concatenated(zP, axis: 1), codes)
    }
}

public final class FishS1DownsampleStage: Module, UnaryLayer {
    @ModuleInfo(key: "0") public var conv: FishS1CausalConvNet
    @ModuleInfo(key: "1") public var block: FishS1ConvNeXtBlock

    public init(inputDim: Int, outputDim: Int, factor: Int) {
        self._conv = ModuleInfo(wrappedValue: FishS1CausalConvNet(
            inChannels: inputDim,
            outChannels: outputDim,
            kernelSize: factor,
            stride: factor
        ))
        self._block = ModuleInfo(wrappedValue: FishS1ConvNeXtBlock(dim: outputDim))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        block(conv(x))
    }
}

public final class FishS1UpsampleStage: Module, UnaryLayer {
    @ModuleInfo(key: "0") public var conv: FishS1CausalTransConvNet
    @ModuleInfo(key: "1") public var block: FishS1ConvNeXtBlock

    public init(inputDim: Int, outputDim: Int, factor: Int) {
        self._conv = ModuleInfo(wrappedValue: FishS1CausalTransConvNet(
            inChannels: inputDim,
            outChannels: outputDim,
            kernelSize: factor,
            stride: factor
        ))
        self._block = ModuleInfo(wrappedValue: FishS1ConvNeXtBlock(dim: outputDim))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        block(conv(x))
    }
}

public final class FishS1DownsampleResidualVectorQuantize: Module {
    public let nCodebooks: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let semanticCodebookSize: Int
    public let downsampleFactor: [Int]

    @ModuleInfo(key: "semantic_quantizer") public var semanticQuantizer: FishS1ResidualVectorQuantize
    @ModuleInfo(key: "quantizer") public var quantizer: FishS1ResidualVectorQuantize
    @ModuleInfo(key: "downsample") public var downsample: [FishS1DownsampleStage]
    @ModuleInfo(key: "upsample") public var upsample: [FishS1UpsampleStage]
    @ModuleInfo(key: "pre_module") public var preModule: Module
    @ModuleInfo(key: "post_module") public var postModule: Module

    public init(
        inputDim: Int = 1024,
        nCodebooks: Int = 9,
        codebookDim: Int = 8,
        codebookSize: Int = 1024,
        semanticCodebookSize: Int = 4096,
        downsampleFactor: [Int] = [2, 2],
        downsampleDims: [Int]? = nil,
        preModule: Module? = nil,
        postModule: Module? = nil
    ) {
        let resolvedDims = downsampleDims ?? Array(repeating: inputDim, count: downsampleFactor.count)
        let allDims = [inputDim] + resolvedDims

        self.nCodebooks = nCodebooks
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.semanticCodebookSize = semanticCodebookSize
        self.downsampleFactor = downsampleFactor

        self._semanticQuantizer = ModuleInfo(wrappedValue: FishS1ResidualVectorQuantize(
            inputDim: inputDim,
            nCodebooks: 1,
            codebookSize: semanticCodebookSize,
            codebookDim: codebookDim
        ))
        self._quantizer = ModuleInfo(wrappedValue: FishS1ResidualVectorQuantize(
            inputDim: inputDim,
            nCodebooks: nCodebooks,
            codebookSize: codebookSize,
            codebookDim: codebookDim
        ))
        self._downsample = ModuleInfo(wrappedValue: downsampleFactor.enumerated().map { index, factor in
            FishS1DownsampleStage(
                inputDim: allDims[index],
                outputDim: allDims[index + 1],
                factor: factor
            )
        })
        self._upsample = ModuleInfo(wrappedValue: downsampleFactor.enumerated().reversed().map { index, factor in
            FishS1UpsampleStage(
                inputDim: allDims[index + 1],
                outputDim: allDims[index],
                factor: factor
            )
        })
        self._preModule = ModuleInfo(wrappedValue: preModule ?? FishS1Identity())
        self._postModule = ModuleInfo(wrappedValue: postModule ?? FishS1Identity())
    }

    public func callAsFunction(_ z: MLXArray, nQuantizers: Int? = nil) -> FishS1VQResult {
        let originalLength = z.shape[2]
        var hidden = z

        for stage in downsample {
            hidden = stage(hidden)
        }

        hidden = fishS1CallUnary(preModule, hidden)

        let (semanticZ, semanticCodes, semanticLatents, semanticCommitmentLoss, semanticCodebookLoss) = semanticQuantizer(hidden)
        let residualHidden = hidden - semanticZ
        let (residualZ, residualCodes, residualLatents, commitmentLoss, codebookLoss) = quantizer(
            residualHidden,
            nQuantizers: nQuantizers
        )

        hidden = semanticZ + residualZ
        hidden = fishS1CallUnary(postModule, hidden)
        for stage in upsample {
            hidden = stage(hidden)
        }

        let diff = originalLength - hidden.shape[2]
        if diff > 0 {
            hidden = MLX.padded(
                hidden,
                widths: [
                    IntOrPair(0),
                    IntOrPair(0),
                    IntOrPair((diff, 0))
                ]
            )
        } else if diff < 0 {
            hidden = hidden[0..., 0..., abs(diff)...]
        }

        return FishS1VQResult(
            z: hidden,
            codes: MLX.concatenated([semanticCodes, residualCodes], axis: 1),
            latents: MLX.concatenated([semanticLatents, residualLatents], axis: 1),
            codebookLoss: codebookLoss + semanticCodebookLoss,
            commitmentLoss: commitmentLoss + semanticCommitmentLoss
        )
    }

    public func decode(_ indices: MLXArray) -> MLXArray {
        let semanticIndices = MLX.clip(
            indices[0..., 0..<1, 0...],
            min: 0,
            max: semanticQuantizer.codebookSize - 1
        )
        let zQSemantic = semanticQuantizer.fromCodes(semanticIndices).0

        let zQResidual: MLXArray
        if indices.shape[1] > 1 {
            let residualIndices = MLX.clip(
                indices[0..., 1..., 0...],
                min: 0,
                max: quantizer.codebookSize - 1
            )
            zQResidual = quantizer.fromCodes(residualIndices).0
        } else {
            zQResidual = MLXArray.zeros(zQSemantic.shape, dtype: zQSemantic.dtype)
        }

        var zQ = fishS1CallUnary(postModule, zQSemantic + zQResidual)
        for stage in upsample {
            zQ = stage(zQ)
        }
        return zQ
    }
}
