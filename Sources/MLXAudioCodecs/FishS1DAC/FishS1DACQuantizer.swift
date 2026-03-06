import Foundation
import MLX
import MLXNN

// MARK: - Vector Quantize

/// Single codebook vector quantization with cosine distance.
public class FishVectorQuantize: Module {
    let codebookSize: Int
    let codebookDim: Int

    @ModuleInfo(key: "in_proj") var inProj: FishWNConv1d
    @ModuleInfo(key: "out_proj") var outProj: FishWNConv1d
    @ModuleInfo(key: "codebook") var codebook: Embedding

    public init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim

        self._inProj.wrappedValue = FishWNConv1d(
            inChannels: inputDim, outChannels: codebookDim, kernelSize: 1
        )
        self._outProj.wrappedValue = FishWNConv1d(
            inChannels: codebookDim, outChannels: inputDim, kernelSize: 1
        )
        self._codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }

    /// Encode input to nearest codebook indices using cosine distance.
    /// Input: NCL format. Returns (z_q in NCL, indices).
    public func encode(_ z: MLXArray) -> (MLXArray, MLXArray) {
        // Project to codebook dim
        let zE = inProj(z)  // [B, codebookDim, T]

        // NCL -> NLC for distance computation
        let zFlat = zE.transposed(0, 2, 1)  // [B, T, codebookDim]

        // Cosine distance: normalize both
        let zNorm = zFlat / (MLX.sqrt(MLX.sum(zFlat * zFlat, axis: -1, keepDims: true)) + 1e-8)
        let cbWeight = codebook.weight
        let cbNorm = cbWeight / (MLX.sqrt(MLX.sum(cbWeight * cbWeight, axis: -1, keepDims: true)) + 1e-8)

        // Distance: 1 - cosine_similarity
        // cosine_sim = z_norm @ cb_norm.T
        let sim = MLX.matmul(zNorm, cbNorm.transposed(0, 1))
        let indices = MLX.argMax(sim, axis: -1)  // [B, T]

        // Look up codebook
        let zQ = codebook(indices)  // [B, T, codebookDim]

        // NLC -> NCL
        let zQNCL = zQ.transposed(0, 2, 1)

        return (zQNCL, indices)
    }

    /// Decode codebook indices back to input space.
    /// Returns (decoded NCL, latent NCL).
    public func decodeCode(_ indices: MLXArray) -> (MLXArray, MLXArray) {
        // indices: [B, T]
        let zQ = codebook(indices)  // [B, T, codebookDim]
        let zQNCL = zQ.transposed(0, 2, 1)  // [B, codebookDim, T]
        let decoded = outProj(zQNCL)  // [B, inputDim, T]
        return (decoded, zQNCL)
    }
}

// MARK: - Residual Vector Quantize

/// Stacks multiple VectorQuantize codebooks in sequence.
public class FishResidualVectorQuantize: Module {
    let nCodebooks: Int
    let codebookSize: Int

    @ModuleInfo(key: "quantizers") var quantizers: [FishVectorQuantize]

    public init(inputDim: Int, nCodebooks: Int, codebookSize: Int, codebookDim: Int) {
        self.nCodebooks = nCodebooks
        self.codebookSize = codebookSize

        self._quantizers.wrappedValue = (0..<nCodebooks).map { _ in
            FishVectorQuantize(
                inputDim: inputDim, codebookSize: codebookSize, codebookDim: codebookDim
            )
        }
    }

    /// Encode with residual quantization.
    /// Input: NCL. Returns (quantized NCL, indices [B, nCodebooks, T]).
    public func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        var residual = z
        var allIndices: [MLXArray] = []
        var zQ = MLXArray.zeros(like: z)

        for quantizer in quantizers {
            let (zQi, indices) = quantizer.encode(residual)
            residual = residual - stopGradient(zQi)
            zQ = zQ + zQi
            allIndices.append(indices.expandedDimensions(axis: 1))
        }

        let indices = MLX.concatenated(allIndices, axis: 1)  // [B, nCodebooks, T]
        return (zQ, indices)
    }

    /// Decode from indices: [B, nCodebooks, T] -> NCL.
    public func fromCodes(_ codes: MLXArray) -> (MLXArray, MLXArray) {
        var zQ = MLXArray.zeros([codes.dim(0), quantizers[0].outProj.weightG.dim(0), codes.dim(2)])
        var allLatents: [MLXArray] = []

        for (i, quantizer) in quantizers.enumerated() {
            let codeSlice = codes[0..., i, 0...]  // [B, T]
            let (decoded, latent) = quantizer.decodeCode(codeSlice)
            zQ = zQ + decoded
            allLatents.append(latent)
        }

        // Stack latents: [B, nCodebooks, codebookDim, T]
        let latents = MLX.stacked(allLatents, axis: 1)
        return (zQ, latents)
    }
}

// MARK: - Downsample Residual Vector Quantize

/// Two-tier quantizer with downsample/upsample paths and transformer modules.
public class FishDownsampleResidualVectorQuantize: Module {
    let inputDim: Int

    @ModuleInfo(key: "semantic_quantizer") var semanticQuantizer: FishResidualVectorQuantize
    @ModuleInfo(key: "quantizer") var quantizer: FishResidualVectorQuantize
    @ModuleInfo(key: "downsample") var downsample: [[Module]]
    @ModuleInfo(key: "upsample") var upsample: [[Module]]
    @ModuleInfo(key: "pre_module") var preModule: FishWindowLimitedTransformer
    @ModuleInfo(key: "post_module") var postModule: FishWindowLimitedTransformer

    // Keep typed references for forward pass
    let downsampleBlocks: [(FishCausalConvNet, FishConvNeXtBlock)]
    let upsampleBlocks: [(FishCausalTransConvNet, FishConvNeXtBlock)]

    public init(config: FishS1DACConfig) {
        self.inputDim = config.latentDim

        // Semantic quantizer: 1 codebook, larger codebook size
        self._semanticQuantizer.wrappedValue = FishResidualVectorQuantize(
            inputDim: config.latentDim,
            nCodebooks: 1,
            codebookSize: config.semanticCodebookSize,
            codebookDim: config.codebookDim
        )

        // Residual quantizer: multiple codebooks
        self._quantizer.wrappedValue = FishResidualVectorQuantize(
            inputDim: config.latentDim,
            nCodebooks: config.nCodebooks,
            codebookSize: config.codebookSize,
            codebookDim: config.codebookDim
        )

        // Downsample stages
        var dsBlocks: [(FishCausalConvNet, FishConvNeXtBlock)] = []
        var dsModules: [[Module]] = []
        for factor in config.downsampleFactor {
            let conv = FishCausalConvNet(
                inChannels: config.latentDim, outChannels: config.latentDim,
                kernelSize: factor, stride: factor
            )
            let cnb = FishConvNeXtBlock(dim: config.latentDim)
            dsBlocks.append((conv, cnb))
            dsModules.append([conv, cnb])
        }
        self.downsampleBlocks = dsBlocks
        self._downsample.wrappedValue = dsModules

        // Upsample stages (reversed)
        var usBlocks: [(FishCausalTransConvNet, FishConvNeXtBlock)] = []
        var usModules: [[Module]] = []
        for factor in config.downsampleFactor.reversed() {
            let conv = FishCausalTransConvNet(
                inChannels: config.latentDim, outChannels: config.latentDim,
                kernelSize: factor, stride: factor
            )
            let cnb = FishConvNeXtBlock(dim: config.latentDim)
            usBlocks.append((conv, cnb))
            usModules.append([conv, cnb])
        }
        self.upsampleBlocks = usBlocks
        self._upsample.wrappedValue = usModules

        // Transformer modules for pre/post processing
        var qConfig = FishS1DACConfig()
        qConfig.dim = config.quantizerDim
        qConfig.nHead = config.quantizerNHead
        qConfig.nLocalHeads = config.quantizerNHead
        qConfig.nLayer = config.quantizerNLayer
        qConfig.intermediateSize = config.quantizerIntermediateSize
        qConfig.headDim = config.quantizerDim / config.quantizerNHead
        qConfig.blockSize = config.blockSize

        self._preModule.wrappedValue = FishWindowLimitedTransformer(
            config: qConfig,
            inputDim: config.latentDim,
            outputDim: config.latentDim,
            channelsFirst: true,
            windowSize: config.quantizerWindowSize
        )

        self._postModule.wrappedValue = FishWindowLimitedTransformer(
            config: qConfig,
            inputDim: config.latentDim,
            outputDim: config.latentDim,
            channelsFirst: true,
            windowSize: config.quantizerWindowSize
        )
    }

    /// Full forward pass: downsample -> pre_module -> quantize -> post_module -> upsample.
    /// Input: NCL. Returns (quantized NCL, indices [B, 1+nCodebooks, T_ds]).
    public func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        let originalLength = z.dim(-1)

        // Downsample
        var h = z
        for (conv, cnb) in downsampleBlocks {
            h = conv(h)
            h = cnb(h)
        }

        // Pre-transformer
        h = preModule(h)

        // Semantic quantization
        let (semZQ, semIndices) = semanticQuantizer(h)

        // Residual quantization
        let residual = h - stopGradient(semZQ)
        let (resZQ, resIndices) = quantizer(residual)

        // Combine
        h = semZQ + resZQ

        // Post-transformer
        h = postModule(h)

        // Upsample
        for (conv, cnb) in upsampleBlocks {
            h = conv(h)
            h = cnb(h)
        }

        // Trim to original length
        if h.dim(-1) > originalLength {
            h = h[0..., 0..., ..<originalLength]
        }

        // Combine indices: [B, 1 + nCodebooks, T_ds]
        let indices = MLX.concatenated([semIndices, resIndices], axis: 1)
        return (h, indices)
    }

    /// Decode from indices (used by encode_zq).
    /// Returns pre-upsample z_q in NCL format.
    public func decodeFromIndices(_ indices: MLXArray) -> MLXArray {
        let semIndices = indices[0..., ..<1, 0...]
        let resIndices = indices[0..., 1..., 0...]

        let (semZQ, _) = semanticQuantizer.fromCodes(semIndices)

        var resZQ: MLXArray
        if resIndices.dim(1) > 0 {
            let (r, _) = quantizer.fromCodes(resIndices)
            resZQ = r
        } else {
            resZQ = MLXArray.zeros(like: semZQ)
        }

        return semZQ + resZQ
    }

    /// Apply post-module and upsample to z_q (used by decode_zq).
    public func postProcessAndUpsample(_ zQ: MLXArray) -> MLXArray {
        var h = postModule(zQ)
        for (conv, cnb) in upsampleBlocks {
            h = conv(h)
            h = cnb(h)
        }
        return h
    }
}
