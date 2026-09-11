//
//  SparkFeatCoders.swift
//  MLXAudio
//
//  Feature encoder/decoder for Spark-TTS BiCodec (ports
//  spark/modules/encoder_decoder/feat_encoder.py and feat_decoder.py).
//  Both wrap the shared Vocos backbone. The `SamplingBlock` at the checkpoint's
//  `sample_ratios == [1, 1]` carries no weights and reduces to a factor of 3,
//  so only the per-stage Vocos backbones are registered (under key "1", matching
//  the Python `downsample.N.1.*` keys).
//

import Foundation
import MLXAudioCodecs
@preconcurrency import MLX
import MLXNN

/// One `downsample` stage: a weightless SamplingBlock (scale 1 -> x3) followed by
/// a 2-layer Vocos backbone. The backbone is keyed "1" to match `downsample.N.1.*`.
private final class SparkDownStage: Module {
    @ModuleInfo(key: "1") var backbone: VocosBackbone

    init(dim: Int, intermediateDim: Int) {
        self._backbone = ModuleInfo(
            wrappedValue: VocosBackbone(
                inputChannels: dim, dim: dim, intermediateDim: intermediateDim, numLayers: 2),
            key: "1")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        backbone(3 * x)
    }
}

/// Encoder: Vocos backbone -> downsample stages -> linear projection.
/// Input [B, input_channels, T] -> [B, out_channels, T'].
public final class SparkFeatEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: VocosBackbone
    @ModuleInfo(key: "downsample") fileprivate var downsample: [SparkDownStage]
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
            wrappedValue: sampleRatios.map { _ in
                SparkDownStage(dim: vocosDim, intermediateDim: vocosIntermediateDim)
            }, key: "downsample")
        self._project = ModuleInfo(wrappedValue: Linear(vocosDim, outChannels), key: "project")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = encoder(x)                 // [B, T, vocosDim]
        for stage in downsample { h = stage(h) }
        h = project(h)                     // [B, T, outChannels]
        return h.transposed(0, 2, 1)       // [B, outChannels, T]
    }
}

/// Decoder (used as prenet and postnet): linear_pre -> downsample stages ->
/// conditioned Vocos backbone -> linear. `conditionDim` enables AdaLayerNorm
/// (prenet); postnet passes nil.
public final class SparkFeatDecoder: Module {
    @ModuleInfo(key: "linear_pre") var linearPre: Linear
    @ModuleInfo(key: "downsample") fileprivate var downsample: [SparkDownStage]
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
            wrappedValue: sampleRatios.map { _ in
                SparkDownStage(dim: vocosDim, intermediateDim: vocosIntermediateDim)
            }, key: "downsample")
        self._vocosBackbone = ModuleInfo(
            wrappedValue: VocosBackbone(
                inputChannels: vocosDim, dim: vocosDim,
                intermediateDim: vocosIntermediateDim, numLayers: vocosNumLayers,
                adanormNumEmbeddings: conditionDim),
            key: "vocos_backbone")
        self._linear = ModuleInfo(wrappedValue: Linear(vocosDim, outChannels), key: "linear")
    }

    /// `x`: [B, input_channels, T]; `c`: optional AdaLayerNorm condition id.
    public func callAsFunction(_ x: MLXArray, condition c: MLXArray? = nil) -> MLXArray {
        var h = linearPre(x.transposed(0, 2, 1))   // [B, T, vocosDim]
        for stage in downsample { h = stage(h) }
        h = vocosBackbone(h.transposed(0, 2, 1), bandwidthId: c)  // [B, T, vocosDim]
        h = linear(h).transposed(0, 2, 1)          // [B, outChannels, T]
        return useTanhAtFinal ? MLX.tanh(h) : h
    }
}
