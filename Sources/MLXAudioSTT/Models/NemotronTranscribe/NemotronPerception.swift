// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Adapted from NeMo Speech's parallel expert encoder (Apache-2.0).

import Foundation
import MLX
import MLXAudioCore
import MLXAudioVAD
import MLXFFT
import MLXFast
import MLXNN

final class NemotronMelFeatures: Module {
    let window: MLXArray
    let fb: MLXArray
    let config: NemotronMelConfiguration
    init(_ c: NemotronMelConfiguration) {
        config = c
        window = MLXArray(
            (0 ..< c.windowLength).map {
                Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / Double(c.windowLength - 1))))
            })
        fb = melFilters(
            sampleRate: c.sampleRate, nFft: c.fftSize, nMels: c.features,
            norm: "slaney", melScale: .slaney
        ).T.expandedDimensions(axis: 0)
    }
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let c = config
        let count = audio.size / c.hopLength
        if count == 0 { return .zeros([1, c.features, 0]) }
        let positions =
            MLXArray(0 ..< count).expandedDimensions(axis: -1) * c.hopLength
            + MLXArray(0 ..< c.fftSize).expandedDimensions(axis: 0) - c.fftSize / 2
        func gather(_ indices: MLXArray) -> MLXArray {
            let values = audio[clip(indices, min: 0, max: audio.size - 1)]
            return MLX.where((indices .>= 0) & (indices .< audio.size), values, MLXArray(Float(0)))
        }
        let frames = MLX.where(
            (positions .>= 0) & (positions .< audio.size),
            gather(positions) - c.preemphasis * gather(positions - 1), MLXArray(Float(0)))
        let pad = c.fftSize - c.windowLength
        let weights = padded(window.asType(.float32), widths: [.init((pad / 2, pad - pad / 2))])
        let spectrum = abs(MLXFFT.rfft(frames * weights))
        let features = log(
            matmul(spectrum * spectrum, fb[0].asType(.float32).T) + Float(pow(2.0, -24)))
        return features.T.expandedDimensions(axis: 0)
    }
}

final class NemotronFeatureStacking: Module {
    let proj: Linear
    let factor: Int
    init(_ c: NemotronSpeechEncoderConfiguration) {
        factor = c.factor
        proj = Linear(c.features * c.factor, c.width, bias: false)
    }
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = features.transposed(0, 2, 1)
        let length = x.dim(1)
        let padding = (factor - length % factor) % factor
        if padding > 0 { x = padded(x, widths: [.init(0), .init((0, padding)), .init(0)]) }
        return proj(x.reshaped(x.dim(0), -1, x.dim(2) * factor))
    }
}

final class NemotronSpeechAttention: Module {
    @ModuleInfo(key: "w_qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?
    let rope: RoPE
    let config: NemotronSpeechEncoderConfiguration
    init(_ c: NemotronSpeechEncoderConfiguration) {
        config = c
        _qkv.wrappedValue = Linear(c.width, 3 * c.width, bias: c.qkvBias)
        _output.wrappedValue = Linear(c.width, c.width)
        _qNorm.wrappedValue = c.qkNorm ? LayerNorm(dimensions: c.width / c.heads) : nil
        _kNorm.wrappedValue = c.qkNorm ? LayerNorm(dimensions: c.width / c.heads) : nil
        rope = RoPE(
            dimensions: Int(Float(c.width / c.heads) * c.rotaryFraction),
            traditional: false, base: c.rotaryBase)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let c = config
        let batch = x.dim(0)
        let length = x.dim(1)
        let headDim = c.width / c.heads
        let packed = qkv(x).reshaped(batch, length, 3, c.heads, headDim)
        var q = packed[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = packed[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = packed[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        q = rope(q)
        k = rope(k)
        let y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(headDim), -0.5), mask: .array(mask))
        return output(y.transposed(0, 2, 1, 3).reshaped(batch, length, c.width))
    }
}

final class NemotronSpeechFeedForward: Module {
    let linear1: Linear
    let linear2: Linear
    init(_ c: NemotronSpeechEncoderConfiguration) {
        let intermediate = Int(Float(c.width) * c.expansion)
        linear1 = Linear(c.width, intermediate)
        linear2 = Linear(intermediate, c.width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

final class NemotronSpeechBlock: Module {
    let norm1: LayerNorm
    let attn: NemotronSpeechAttention
    let norm2: LayerNorm
    let ffn: NemotronSpeechFeedForward
    init(_ c: NemotronSpeechEncoderConfiguration) {
        norm1 = LayerNorm(dimensions: c.width)
        norm2 = LayerNorm(dimensions: c.width)
        attn = NemotronSpeechAttention(c)
        ffn = NemotronSpeechFeedForward(c)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let x = x + attn(norm1(x), mask: mask)
        return x + ffn(norm2(x))
    }
}

final class NemotronSpeechEncoder: Module {
    @ModuleInfo(key: "pre_encode") var stacking: NemotronFeatureStacking
    @ModuleInfo(key: "embed_norm") var embeddingNorm: LayerNorm?
    let layers: [NemotronSpeechBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    let config: NemotronSpeechEncoderConfiguration
    init(_ c: NemotronSpeechEncoderConfiguration) {
        config = c
        _stacking.wrappedValue = NemotronFeatureStacking(c)
        _embeddingNorm.wrappedValue = c.preBlockNorm ? LayerNorm(dimensions: c.width) : nil
        layers = (0 ..< c.layers).map { _ in NemotronSpeechBlock(c) }
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.width)
    }
    func callAsFunction(_ input: MLXArray, length: Int) -> MLXArray {
        let mask = (MLXArray(0 ..< input.dim(1)) .< length).reshaped(1, 1, 1, -1)
        var x = config.scaleInput ? input * sqrt(Float(config.width)) : input
        if let embeddingNorm { x = embeddingNorm(x) }
        for layer in layers { x = layer(x, mask: mask) }
        return finalNorm(x)
    }
}

final class NemotronSpeakerHead: Module {
    @ModuleInfo(key: "encoder_proj") var projection: Linear
    @ModuleInfo(key: "first_hidden_to_hidden") var hidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var speakers: Linear
    @ModuleInfo(key: "hidden_to_spks") var pairedSpeakers: Linear
    @ModuleInfo(key: "subpixel_upsample") var upsample: Conv1d
    @ParameterInfo(key: "learnable_sil_emb") var silence: MLXArray
    init(_ c: ModulesConfig) {
        _projection.wrappedValue = Linear(c.fcDModel, c.tfDModel)
        _hidden.wrappedValue = Linear(c.tfDModel, c.tfDModel)
        _speakers.wrappedValue = Linear(c.tfDModel, c.numSpeakers)
        _pairedSpeakers.wrappedValue = Linear(2 * c.tfDModel, c.numSpeakers)
        _upsample.wrappedValue = Conv1d(
            inputChannels: c.tfDModel,
            outputChannels: c.tfDModel * c.subsamplingFactor, kernelSize: 3, padding: 1)
        _silence.wrappedValue = .zeros([c.fcDModel])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = projection(x)
        let high = upsample(projected).reshaped(x.dim(0), -1, projected.dim(-1))
        return sigmoid(speakers(relu(hidden(relu(high)))))
    }
}

final class NemotronSpeakerState {
    var cache: MLXArray
    var cachePredictions: MLXArray
    var fifo: MLXArray
    var fifoPredictions: MLXArray
    var compressed = false
    init(width: Int, speakers: Int, dtype: DType) {
        cache = .zeros([1, 0, width], dtype: dtype)
        fifo = .zeros([1, 0, width], dtype: dtype)
        cachePredictions = .zeros([1, 0, speakers])
        fifoPredictions = .zeros([1, 0, speakers])
    }
}

final class NemotronSpeakerEncoder: Module {
    let encoder: NemotronSpeechEncoder
    @ModuleInfo(key: "sortformer_modules") var head: NemotronSpeakerHead
    let preprocessor: NemotronMelFeatures
    let config: NemotronSpeakerConfiguration
    init(_ c: NemotronSpeakerConfiguration) {
        config = c
        encoder = NemotronSpeechEncoder(c.encoder)
        _head.wrappedValue = NemotronSpeakerHead(c.modules)
        preprocessor = NemotronMelFeatures(c.preprocessor)
    }
    func makeState() -> NemotronSpeakerState {
        NemotronSpeakerState(
            width: config.encoder.width, speakers: config.modules.numSpeakers,
            dtype: encoder.stacking.proj.weight.dtype)
    }
    func step(
        _ features: MLXArray, state: NemotronSpeakerState, centralFrames: Int,
        featureLength: Int, leftContextFrames: Int
    ) -> MLXArray {
        let c = config.modules
        let factor = config.encoder.factor
        let chunk = encoder.stacking(features.asType(encoder.stacking.proj.weight.dtype))
        let length = (featureLength + factor - 1) / factor
        let cacheLength = state.cache.dim(1)
        let fifoLength = state.fifo.dim(1)
        let combined = concatenated([state.cache, state.fifo, chunk], axis: 1)
        var high = head(encoder(combined, length: length + cacheLength + fifoLength))
        let valid = MLXArray(0 ..< high.dim(1)) .< ((length + cacheLength + fifoLength) * factor)
        high = high * valid.reshaped(1, -1, 1)
        let low = high.asType(.float32).reshaped(1, -1, factor, c.numSpeakers).mean(axis: 2)
        let count = (centralFrames + factor - 1) / factor
        let left = leftContextFrames / factor
        let start = cacheLength + fifoLength + left
        let result = high[0, (start * factor) ..< (start * factor + centralFrames), 0...].asType(
            .float32)
        state.fifo = concatenated(
            [state.fifo, chunk[0..., left ..< (left + count), 0...]], axis: 1)
        state.fifoPredictions = concatenated(
            [
                low[0..., cacheLength ..< (cacheLength + fifoLength), 0...],
                low[0..., start ..< (start + count), 0...],
            ], axis: 1)
        if state.fifo.dim(1) > c.fifoLen {
            let pop = min(
                state.fifo.dim(1), max(c.spkcacheUpdatePeriod, state.fifo.dim(1) - c.fifoLen))
            state.cache = concatenated([state.cache, state.fifo[0..., ..<pop, 0...]], axis: 1)
            let previous =
                state.compressed ? state.cachePredictions : low[0..., ..<cacheLength, 0...]
            state.cachePredictions = concatenated(
                [previous, state.fifoPredictions[0..., ..<pop, 0...]], axis: 1)
            state.fifo = state.fifo[0..., pop..., 0...]
            state.fifoPredictions = state.fifoPredictions[0..., pop..., 0...]
            if state.cache.dim(1) > c.spkcacheLen {
                let compressed = SortformerModel.compressSpeakerCache(
                    embeddings: state.cache,
                    probabilities: state.cachePredictions,
                    silenceEmbedding: head.silence.expandedDimensions(axis: 0),
                    configuration: c)
                state.cache = compressed.embeddings
                state.cachePredictions = compressed.probabilities
                state.compressed = true
            }
        }
        eval([result, state.cache, state.cachePredictions, state.fifo, state.fifoPredictions])
        return result
    }
}

final class NemotronParallelEncoder: Module {
    @ModuleInfo(key: "asr_encoder") var speech: NemotronSpeechEncoder
    @ModuleInfo(key: "diarization_model") var speaker: NemotronSpeakerEncoder
    @ModuleInfo(key: "asr_norm") var speechNorm: LayerNorm
    @ModuleInfo(key: "diar_norm") var speakerNorm: LayerNorm
    let config: NemotronTranscribeConfiguration
    private let _speakerKernel: MLXArray
    init(_ c: NemotronTranscribeConfiguration) {
        config = c
        _speech.wrappedValue = NemotronSpeechEncoder(c.speech)
        _speaker.wrappedValue = NemotronSpeakerEncoder(c.speaker)
        _speechNorm.wrappedValue = LayerNorm(dimensions: c.speech.width)
        _speakerNorm.wrappedValue = LayerNorm(dimensions: c.speaker.modules.numSpeakers)
        let positions = MLXArray(0 ..< c.speaker.modules.numSpeakers).asType(.float32)
            .expandedDimensions(axis: -1)
        let frequencies = exp(
            MLXArray(stride(from: 0, to: c.speech.width, by: 2)).asType(.float32)
                * (-log(Float(10000)) / Float(c.speech.width)))
        let angles = positions * frequencies
        _speakerKernel = stacked([sin(angles), cos(angles)], axis: -1).reshaped(-1, c.speech.width)
    }
    static func normalize(_ features: MLXArray) -> MLXArray {
        let reference = features[0..., 0..., ..<1]
        let mean = reference + (features - reference).mean(axis: -1, keepDims: true)
        let centered = features - mean
        let variance =
            (centered * centered).sum(axis: -1, keepDims: true)
            / Float(max(1, features.dim(-1) - 1))
        return centered / (sqrt(variance) + 1e-5)
    }
    func fuse(_ states: MLXArray, probabilities: MLXArray) -> MLXArray {
        let c = config
        var probabilities = probabilities.asType(states.dtype)
        if c.speakerMode == "thresholded" {
            probabilities = (probabilities .> c.speakerThreshold!).asType(states.dtype)
        }
        let targets = speakerNorm(probabilities)
        return speechNorm(states) + c.speakerScale
            * matmul(targets, _speakerKernel.asType(targets.dtype))
    }
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let c = config
        let factor = c.speech.factor
        let size = c.windowLength * factor
        let left = c.leftContext * factor
        let right = c.rightContext * factor
        let normalized = Self.normalize(features).asType(speechNorm.weight!.dtype)
        let state = speaker.makeState()
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: features.dim(-1), by: size) {
            let end = min(start + size, features.dim(-1))
            let contextStart = max(0, start - left)
            let contextEnd = min(features.dim(-1), end + right)
            let length = contextEnd - contextStart
            let x = speech.stacking(normalized[0..., 0..., contextStart ..< contextEnd])
            let encoded = speech(x, length: (length + factor - 1) / factor)
            let first = (start - contextStart) / factor
            let count = (end + factor - 1) / factor - (start + factor - 1) / factor
            let probabilities = speaker.step(
                features[0..., 0..., contextStart ..< contextEnd],
                state: state, centralFrames: count * factor, featureLength: length,
                leftContextFrames: start - contextStart
            )
            .reshaped(1, count, factor, -1).mean(axis: 2)
            let y = fuse(
                encoded[0..., first ..< (first + count), 0...], probabilities: probabilities)
            eval(y)
            outputs.append(y)
        }
        return concatenated(outputs, axis: 1)
    }
}

final class NemotronPerception: Module {
    let preprocessor: NemotronMelFeatures
    let encoder: NemotronParallelEncoder
    let proj: Linear
    init(_ c: NemotronTranscribeConfiguration) {
        preprocessor = NemotronMelFeatures(c.preprocessor)
        encoder = NemotronParallelEncoder(c)
        proj = Linear(c.speech.width, c.outputDimension)
    }
    func callAsFunction(_ audio: MLXArray) -> MLXArray { proj(encoder(preprocessor(audio))) }
}
