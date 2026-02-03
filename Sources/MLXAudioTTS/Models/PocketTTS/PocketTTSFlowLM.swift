import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon

@inline(__always)
private func pocketLsdDecode(
    flowNet: SimpleMLPAdaLN,
    condition: MLXArray,
    x0: MLXArray,
    numSteps: Int
) -> MLXArray {
    var current = x0
    let batch = x0.shape[0]
    for i in 0..<numSteps {
        let s = Float(i) / Float(numSteps)
        let t = Float(i + 1) / Float(numSteps)
        let sT = MLXArray.zeros([batch, 1]) + MLXArray(s)
        let tT = MLXArray.zeros([batch, 1]) + MLXArray(t)
        let flowDir = flowNet(condition, sT, tT, current)
        current = current + flowDir / MLXArray(Float(numSteps))
    }
    return current
}

public final class FlowLMModel: Module {
    @ModuleInfo(key: "conditioner") public var conditioner: LUTConditioner
    @ModuleInfo(key: "flow_net") public var flow_net: SimpleMLPAdaLN
    public var emb_std: MLXArray
    public var emb_mean: MLXArray
    public var bos_emb: MLXArray
    @ModuleInfo(key: "input_linear") public var input_linear: Linear
    @ModuleInfo(key: "transformer") public var transformer: PocketStreamingTransformer
    @ModuleInfo(key: "out_norm") public var out_norm: PocketLayerNorm
    @ModuleInfo(key: "out_eos") public var out_eos: Linear

    public let ldim: Int
    public let dim: Int

    public init(
        conditioner: LUTConditioner,
        flowNet: SimpleMLPAdaLN,
        transformer: PocketStreamingTransformer,
        dim: Int,
        ldim: Int
    ) {
        self._conditioner = ModuleInfo(wrappedValue: conditioner)
        self._flow_net = ModuleInfo(wrappedValue: flowNet)
        self.emb_std = MLXArray.ones([ldim])
        self.emb_mean = MLXArray.zeros([ldim])
        self.bos_emb = MLXRandom.normal([ldim])
        self._input_linear = ModuleInfo(wrappedValue: Linear(ldim, dim, bias: false))
        self._transformer = ModuleInfo(wrappedValue: transformer)
        self._out_norm = ModuleInfo(wrappedValue: PocketLayerNorm(channels: dim, eps: 1e-5))
        self._out_eos = ModuleInfo(wrappedValue: Linear(dim, 1, bias: true))
        self.ldim = ldim
        self.dim = dim
        super.init()
    }

    public func makeCache() -> [KVCacheSimple] { transformer.makeCache() }

    private func backbone(input: MLXArray, textEmbeddings: MLXArray, sequence: MLXArray, cache: [KVCacheSimple]) -> MLXArray {
        let combined = concatenated([textEmbeddings, input], axis: 1)
        var out = transformer(combined, cache: cache)
        out = out_norm(out)
        let seqLen = sequence.shape[1]
        if seqLen == 0 {
            return out
        }
        let splitIdx = out.shape[1] - seqLen
        let parts = split(out, indices: [splitIdx], axis: 1)
        return parts.count > 1 ? parts[1] : out
    }

    public func callAsFunction(
        sequence: MLXArray,
        textEmbeddings: MLXArray,
        cache: [KVCacheSimple],
        lsdDecodeSteps: Int,
        temperature: Float,
        noiseClamp: Float?,
        eosThreshold: Float
    ) -> (MLXArray, MLXArray) {
        let bos = bos_emb.reshaped([1, 1, ldim])
        let nanMask = sequence .!= sequence
        let seq = MLX.where(nanMask, bos, sequence)

        let input = input_linear(seq)
        var transformerOut = backbone(input: input, textEmbeddings: textEmbeddings, sequence: seq, cache: cache)
        transformerOut = transformerOut.asType(.float32)

        precondition(lsdDecodeSteps > 0, "lsd_decode_steps must be > 0")

        let last = split(transformerOut, indices: [transformerOut.shape[1] - 1], axis: 1)[1]
        let lastToken = last.squeezed(axis: 1)

        let eosLogits = out_eos(lastToken)
        let isEos = eosLogits .> MLXArray(eosThreshold)

        let noiseShape = [lastToken.shape[0], ldim]
        var noise = MLXRandom.normal(noiseShape) * MLXArray(Float(sqrt(temperature)))
        if let noiseClamp {
            let minVal = MLXArray(-noiseClamp)
            let maxVal = MLXArray(noiseClamp)
            noise = MLX.minimum(MLX.maximum(noise, minVal), maxVal)
        }

        let nextLatent = pocketLsdDecode(flowNet: flow_net, condition: lastToken, x0: noise, numSteps: lsdDecodeSteps)
        return (nextLatent, isEos)
    }

    public static func fromConfig(_ config: PocketTTSFlowLMConfig, latentDim: Int, modelFolder: URL) async throws -> FlowLMModel {
        let dModel = config.transformer.dModel
        let flowNet = SimpleMLPAdaLN(
            inChannels: latentDim,
            modelChannels: config.flow.dim,
            outChannels: latentDim,
            condChannels: dModel,
            numResBlocks: config.flow.depth,
            numTimeConds: 2
        )

        let conditioner = try await LUTConditioner(
            nBins: config.lookupTable.nBins,
            modelFolder: modelFolder,
            dim: config.lookupTable.dim,
            outputDim: dModel
        )

        let transformer = PocketStreamingTransformer(
            dModel: dModel,
            numHeads: config.transformer.numHeads,
            numLayers: config.transformer.numLayers,
            dimFeedforward: Int(config.transformer.hiddenScale * dModel),
            maxPeriod: config.transformer.maxPeriod,
            layerScale: nil
        )

        return FlowLMModel(
            conditioner: conditioner,
            flowNet: flowNet,
            transformer: transformer,
            dim: dModel,
            ldim: latentDim
        )
    }
}
