//
//  SparkBiCodec.swift
//  MLXAudio
//
//  BiCodec detokenizer for Spark-TTS: semantic + global tokens -> waveform.
//  The encode path (voice cloning: mel + Wav2Vec2 + ECAPA + perceiver) is not
//  included; only the synthesis path used by controllable TTS is implemented.
//

import Foundation
@preconcurrency import MLX
import MLXNN

public final class SparkBiCodec: Module {
    @ModuleInfo(key: "quantizer") public var quantizer: SparkFactorizedVectorQuantize
    @ModuleInfo(key: "speaker_encoder") public var speakerEncoder: SparkSpeakerEncoder
    @ModuleInfo(key: "prenet") public var prenet: SparkFeatDecoder
    @ModuleInfo(key: "decoder") public var decoder: SparkWaveGenerator

    public init(_ config: BiCodecConfiguration) {
        self._quantizer = ModuleInfo(
            wrappedValue: SparkFactorizedVectorQuantize(
                inputDim: config.quantizer.inputDim,
                codebookSize: config.quantizer.codebookSize,
                codebookDim: config.quantizer.codebookDim),
            key: "quantizer")
        self._speakerEncoder = ModuleInfo(
            wrappedValue: SparkSpeakerEncoder(
                latentDim: config.speakerEncoder.latentDim,
                outDim: config.speakerEncoder.outDim,
                tokenNum: config.speakerEncoder.tokenNum,
                fsqLevels: config.speakerEncoder.fsqLevels),
            key: "speaker_encoder")
        self._prenet = ModuleInfo(
            wrappedValue: SparkFeatDecoder(
                inputChannels: config.prenet.inputChannels,
                vocosDim: config.prenet.vocosDim,
                vocosIntermediateDim: config.prenet.vocosIntermediateDim,
                vocosNumLayers: config.prenet.vocosNumLayers,
                outChannels: config.prenet.outChannels,
                conditionDim: config.prenet.conditionDim,
                sampleRatios: config.prenet.sampleRatios ?? [1, 1],
                useTanhAtFinal: config.prenet.useTanhAtFinal ?? false),
            key: "prenet")
        self._decoder = ModuleInfo(
            wrappedValue: SparkWaveGenerator(
                inputChannel: config.decoder.inputChannel,
                channels: config.decoder.channels,
                rates: config.decoder.rates,
                kernelSizes: config.decoder.kernelSizes),
            key: "decoder")
    }

    /// `semanticTokens`: [B, T], `globalTokens`: [B, tokenNum] -> waveform [B*samples].
    public func detokenize(semanticTokens: MLXArray, globalTokens: MLXArray) -> MLXArray {
        let global = globalTokens.expandedDimensions(axis: 1)
        let zq = quantizer.detokenize(semanticTokens).transposed(0, 2, 1)
        let dVector = speakerEncoder.detokenize(global)
        var x = prenet(zq, condition: dVector)
        x = x + dVector.expandedDimensions(axis: -1)
        let wav = decoder(x)
        return wav.squeezed()
    }

    /// Reorder PyTorch checkpoint weights to the MLX layout each module expects,
    /// dropping the encode-only speaker weights and the quantizer EMA buffer.
    /// Conv weights differ by layout (standard vs transpose), so 3-D weights are
    /// transposed to whichever permutation matches the target parameter shape.
    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let expected = Dictionary(
            self.parameters().flattened().map { ($0.0, $0.1.shape) },
            uniquingKeysWith: { a, _ in a })
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key == "quantizer.cluster_size" { continue }
            if key.hasPrefix("encoder.") || key.hasPrefix("postnet.")
                || key.hasPrefix("quantizer.in_project.")
                || key.hasPrefix("speaker_encoder.speaker_encoder.")
                || key.hasPrefix("speaker_encoder.perceiver_sampler.")
                || key.hasPrefix("speaker_encoder.quantizer.project_in.") { continue }
            var v = value
            if let want = expected[key], v.ndim == 3, v.shape != want {
                for perm in [[0, 2, 1], [1, 2, 0], [2, 1, 0], [2, 0, 1], [1, 0, 2]] {
                    let t = v.transposed(perm[0], perm[1], perm[2])
                    if t.shape == want { v = t; break }
                }
            }
            out[key] = v
        }
        return out
    }
}
