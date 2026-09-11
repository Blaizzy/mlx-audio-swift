import Foundation
import MLXAudioCodecs
@preconcurrency import MLX
import MLXNN

/// Weightless SamplingBlock at the checkpoint's `sample_ratios == [1, 1]`
/// (both scales 1): it reduces to a factor of 3.
fileprivate final class SparkSamplingBlock: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { 3 * x }
}

fileprivate func makeDownsample(_ ratios: [Int], dim: Int, intermediateDim: Int) -> [[Module]] {
    ratios.map { _ in
        [
            SparkSamplingBlock(),
            VocosBackbone(inputChannels: dim, dim: dim, intermediateDim: intermediateDim, numLayers: 2),
        ]
    }
}

fileprivate func runDownsample(_ stages: [[Module]], _ x: MLXArray) -> MLXArray {
    var h = x
    for stage in stages {
        h = (stage[0] as! SparkSamplingBlock)(h)
        h = (stage[1] as! VocosBackbone)(h)
    }
    return h
}

/// Feature encoder: Vocos backbone -> downsample stages -> linear projection.
public final class SparkFeatEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: VocosBackbone
    @ModuleInfo(key: "downsample") fileprivate var downsample: [[Module]]
    @ModuleInfo(key: "project") var project: Linear

    public init(
        inputChannels: Int, vocosDim: Int, vocosIntermediateDim: Int,
        vocosNumLayers: Int, outChannels: Int, sampleRatios: [Int]
    ) {
        self._encoder = ModuleInfo(
            wrappedValue: VocosBackbone(
                inputChannels: inputChannels, dim: vocosDim,
                intermediateDim: vocosIntermediateDim, numLayers: vocosNumLayers),
            key: "encoder")
        self._downsample = ModuleInfo(
            wrappedValue: makeDownsample(sampleRatios, dim: vocosDim, intermediateDim: vocosIntermediateDim),
            key: "downsample")
        self._project = ModuleInfo(wrappedValue: Linear(vocosDim, outChannels), key: "project")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = encoder(x)
        h = runDownsample(downsample, h)
        h = project(h)
        return h.transposed(0, 2, 1)
    }
}

/// Feature decoder (prenet/postnet): linear_pre -> downsample -> conditioned
/// Vocos backbone -> linear. `conditionDim` enables AdaLayerNorm (prenet).
public final class SparkFeatDecoder: Module {
    @ModuleInfo(key: "linear_pre") var linearPre: Linear
    @ModuleInfo(key: "downsample") fileprivate var downsample: [[Module]]
    @ModuleInfo(key: "vocos_backbone") var vocosBackbone: VocosBackbone
    @ModuleInfo(key: "linear") var linear: Linear

    private let useTanhAtFinal: Bool

    public init(
        inputChannels: Int, vocosDim: Int, vocosIntermediateDim: Int,
        vocosNumLayers: Int, outChannels: Int, conditionDim: Int?,
        sampleRatios: [Int], useTanhAtFinal: Bool
    ) {
        self.useTanhAtFinal = useTanhAtFinal
        self._linearPre = ModuleInfo(wrappedValue: Linear(inputChannels, vocosDim), key: "linear_pre")
        self._downsample = ModuleInfo(
            wrappedValue: makeDownsample(sampleRatios, dim: vocosDim, intermediateDim: vocosIntermediateDim),
            key: "downsample")
        self._vocosBackbone = ModuleInfo(
            wrappedValue: VocosBackbone(
                inputChannels: vocosDim, dim: vocosDim,
                intermediateDim: vocosIntermediateDim, numLayers: vocosNumLayers,
                adanormNumEmbeddings: conditionDim),
            key: "vocos_backbone")
        self._linear = ModuleInfo(wrappedValue: Linear(vocosDim, outChannels), key: "linear")
    }

    public func callAsFunction(_ x: MLXArray, condition c: MLXArray? = nil) -> MLXArray {
        var h = linearPre(x.transposed(0, 2, 1))
        h = runDownsample(downsample, h)
        h = vocosBackbone(h.transposed(0, 2, 1), bandwidthId: c)
        h = linear(h).transposed(0, 2, 1)
        return useTanhAtFinal ? MLX.tanh(h) : h
    }
}
