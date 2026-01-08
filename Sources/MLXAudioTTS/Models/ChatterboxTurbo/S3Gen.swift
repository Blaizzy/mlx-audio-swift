//
//  S3Gen.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import MLXAudioCore

let S3GenSampleRate = 24_000
let S3GenSilenceToken = 4_299

struct S3GenReference {
    let promptToken: MLXArray
    let promptTokenLen: MLXArray
    let promptFeat: MLXArray
    let promptFeatLen: MLXArray
    let embedding: MLXArray
}

func s3ResampleLinear(_ wav: [Float], from: Int, to: Int) -> [Float] {
    guard from != to, wav.count > 1 else { return wav }
    let ratio = Float(to) / Float(from)
    let newLength = Int(round(Float(wav.count) * ratio))
    guard newLength > 1 else { return wav }

    var output = [Float](repeating: 0, count: newLength)
    for i in 0..<newLength {
        let pos = Float(i) / ratio
        let idx = Int(floor(pos))
        let frac = pos - Float(idx)
        let idxNext = min(idx + 1, wav.count - 1)
        output[i] = wav[idx] * (1 - frac) + wav[idxNext] * frac
    }
    return output
}

class S3Token2Mel: Module {
    let meanflow: Bool
    let tokenMelRatio: Int = 2
    let preLookaheadLen: Int = 3

    @ModuleInfo(key: "input_embedding") private var inputEmbedding: Embedding
    @ModuleInfo(key: "speaker_encoder") fileprivate var speakerEncoder: CAMPPlus
    @ModuleInfo(key: "spk_embed_affine_layer") private var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "encoder") private var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") private var encoderProj: Linear
    @ModuleInfo(key: "decoder") private var decoder: CausalConditionalCFM

    init(meanflow: Bool = false) {
        self.meanflow = meanflow
        self._inputEmbedding.wrappedValue = Embedding(embeddingCount: S3SpeechVocabSize, dimensions: 512)
        self._speakerEncoder.wrappedValue = CAMPPlus(
            featDim: 80,
            embeddingSize: 192,
            growthRate: 32,
            bnSize: 4,
            initChannels: 128
        )
        self._spkEmbedAffineLayer.wrappedValue = Linear(192, 80)
        self._encoder.wrappedValue = UpsampleConformerEncoder(
            inputSize: 512,
            outputSize: 512,
            attentionHeads: 8,
            linearUnits: 2048,
            numBlocks: 6,
            dropoutRate: 0.1
        )
        self._encoderProj.wrappedValue = Linear(512, 80)
        let estimator = ConditionalDecoder(
            inChannels: 320,
            outChannels: 80,
            causal: true,
            channels: [256],
            attentionHeadDim: 64,
            nBlocks: 4,
            numMidBlocks: 12,
            numHeads: 8,
            meanflow: meanflow
        )
        self._decoder.wrappedValue = CausalConditionalCFM(
            inChannels: 240,
            sigmaMin: 1e-6,
            tScheduler: "cosine",
            inferenceCfgRate: 0.7,
            estimator: estimator
        )
    }

    func embedRef(
        refWav: MLXArray,
        refSr: Int,
        refSpeechTokens: MLXArray? = nil,
        refSpeechTokenLens: MLXArray? = nil
    ) -> S3GenReference {
        var wav = refWav
        if wav.ndim == 1 {
            wav = wav.expandedDimensions(axis: 0)
        }

        var refWav24k = wav
        if refSr != S3GenSampleRate {
            let data = wav[0].asArray(Float.self)
            let resampled = s3ResampleLinear(data, from: refSr, to: S3GenSampleRate)
            refWav24k = MLXArray(resampled).expandedDimensions(axis: 0)
        }

        var refMels = s3genMelSpectrogram(refWav24k, numMels: 80, samplingRate: S3GenSampleRate)
        refMels = refMels.transposed(0, 2, 1)

        var promptTokens = refSpeechTokens
        if promptTokens == nil {
            let tokenLen = max(refMels.shape[1] / 2, 1)
            promptTokens = MLXArray.zeros([1, tokenLen], type: Int32.self)
        }

        var promptTokenLen = refSpeechTokenLens
        if promptTokenLen == nil {
            promptTokenLen = MLXArray([Int32(promptTokens!.shape[1])])
        }

        let expectedTokenLen = refMels.shape[1] / 2
        let actualTokenLen = promptTokens!.shape[1]
        if actualTokenLen != expectedTokenLen {
            if actualTokenLen < expectedTokenLen {
                let expectedMelLen = actualTokenLen * 2
                refMels = refMels[0..., 0..<expectedMelLen, 0...]
            } else {
                promptTokens = promptTokens![0..., 0..<expectedTokenLen]
                promptTokenLen = MLXArray([Int32(expectedTokenLen)])
            }
        }

        var refWav16k = wav
        if refSr != S3SampleRate {
            let data = wav[0].asArray(Float.self)
            let resampled = s3ResampleLinear(data, from: refSr, to: S3SampleRate)
            refWav16k = MLXArray(resampled).expandedDimensions(axis: 0)
        }

        let refXVector = speakerEncoder.inference([refWav16k[0]])
        return S3GenReference(
            promptToken: promptTokens!,
            promptTokenLen: promptTokenLen!,
            promptFeat: refMels,
            promptFeatLen: MLXArray([Int32(refMels.shape[1])]),
            embedding: refXVector
        )
    }

    func callAsFunction(
        _ speechTokens: MLXArray,
        refDict: S3GenReference,
        nCfmTimesteps: Int? = nil,
        finalize: Bool = true
    ) -> MLXArray {
        let batch = speechTokens.shape[0]

        var promptToken = refDict.promptToken
        var promptTokenLen = refDict.promptTokenLen
        var promptFeat = refDict.promptFeat
        var embedding = refDict.embedding

        if promptToken.shape[0] != batch {
            promptToken = promptToken.broadcasted(to: [batch, promptToken.shape[1]])
            promptTokenLen = promptTokenLen.broadcasted(to: [batch])
        }
        if embedding.shape[0] != batch {
            embedding = embedding.broadcasted(to: [batch, embedding.shape[1]])
        }
        if promptFeat.shape[0] != batch {
            promptFeat = promptFeat.broadcasted(to: [batch, promptFeat.shape[1], promptFeat.shape[2]])
        }

        let norm = MLX.sqrt(MLX.sum(embedding.square(), axis: -1, keepDims: true)) + 1e-8
        embedding = embedding / norm
        embedding = spkEmbedAffineLayer(embedding)

        let tokenLen = MLXArray(Array(repeating: Int32(speechTokens.shape[1]), count: batch))
        let token = MLX.concatenated([promptToken, speechTokens], axis: 1)
        let totalTokenLen = promptTokenLen + tokenLen

        let maxLen = token.shape[1]
        let seqRange = MLXArray.arange(0, maxLen, dtype: .int32)
        let seqExpand = seqRange.expandedDimensions(axis: 0).broadcasted(to: [batch, maxLen])
        var mask = seqExpand .< totalTokenLen.expandedDimensions(axis: -1)
        mask = mask.expandedDimensions(axis: 2).asType(.float32)

        var tokenEmb = inputEmbedding(token.asType(.int32))
        tokenEmb = tokenEmb * mask

        var (h, hMasks) = encoder(tokenEmb, xsLens: totalTokenLen)

        if !finalize {
            let trim = preLookaheadLen * tokenMelRatio
            if h.shape[1] > trim {
                h = h[0..., 0..<(h.shape[1] - trim), 0...]
                hMasks = hMasks[0..., 0..., 0..<(hMasks.shape[2] - trim)]
            }
        }

        let hLengths = MLX.sum(hMasks[0..., 0, 0...].asType(.int32), axis: -1)
        let melLen1 = promptFeat.shape[1]
        let melLen2 = max(h.shape[1] - melLen1, 0)

        h = encoderProj(h)
        let zerosPadding = MLXArray.zeros([batch, melLen2, 80], type: Float.self)
        var conds = MLX.concatenated([promptFeat, zerosPadding], axis: 1)
        conds = conds.transposed(0, 2, 1)

        let seqRangeH = MLXArray.arange(0, h.shape[1], dtype: .int32)
        let seqExpandH = seqRangeH.expandedDimensions(axis: 0).broadcasted(to: [batch, h.shape[1]])
        var outMask = seqExpandH .< hLengths.expandedDimensions(axis: -1)
        outMask = outMask.expandedDimensions(axis: 1).asType(.float32)

        let timesteps = nCfmTimesteps ?? (meanflow ? 2 : 10)
        var noisedMels: MLXArray? = nil
        if meanflow {
            noisedMels = MLXRandom.normal([batch, 80, speechTokens.shape[1] * 2])
        }

        let (feat, _) = decoder(
            mu: h.transposed(0, 2, 1),
            mask: outMask,
            nTimesteps: timesteps,
            spks: embedding,
            cond: conds,
            noisedMels: noisedMels,
            meanflow: meanflow
        )

        return feat[0..., 0..., melLen1...]
    }
}

final class S3Token2Wav: S3Token2Mel {
    @ModuleInfo(key: "mel2wav") private var mel2wav: HiFTGenerator
    private let trimFade: [Float]

    override init(meanflow: Bool = false) {
        self._mel2wav.wrappedValue = HiFTGenerator(
            samplingRate: S3GenSampleRate,
            upsampleRates: [8, 5, 3],
            upsampleKernelSizes: [16, 11, 7],
            sourceResblockKernelSizes: [7, 7, 11],
            sourceResblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0Predictor: F0Predictor()
        )
        let nTrim = S3GenSampleRate / 50
        var fade = [Float](repeating: 0, count: 2 * nTrim)
        for i in 0..<nTrim {
            fade[nTrim + i] = (cosf(Float.pi * (Float(nTrim - i) / Float(nTrim))) + 1) / 2
        }
        self.trimFade = fade
        super.init(meanflow: meanflow)
    }

    func inference(
        speechTokens: MLXArray,
        refDict: S3GenReference? = nil,
        refWav: MLXArray? = nil,
        refSr: Int? = nil,
        nCfmTimesteps: Int? = nil
    ) -> (MLXArray, MLXArray) {
        var refDict = refDict
        if refDict == nil {
            guard let refWav, let refSr else {
                fatalError("Must provide either refDict or refWav/refSr")
            }
            refDict = embedRef(refWav: refWav, refSr: refSr)
        }

        let timesteps = nCfmTimesteps ?? (meanflow ? 2 : 10)
        let outputMels = callAsFunction(speechTokens, refDict: refDict!, nCfmTimesteps: timesteps, finalize: true)
        let melForVocoder = outputMels.transposed(0, 2, 1)
        var (wav, src) = mel2wav.inference(melForVocoder, nil)

        let fadeLen = trimFade.count
        if wav.shape[1] >= fadeLen {
            let fade = MLXArray(trimFade).expandedDimensions(axis: 0)
            let fadedStart = wav[0..., 0..<fadeLen] * fade
            wav = MLX.concatenated([fadedStart, wav[0..., fadeLen...]], axis: 1)
        }

        return (wav, src)
    }

    func inferenceStream(
        speechTokens: MLXArray,
        refDict: S3GenReference,
        nCfmTimesteps: Int? = nil,
        prevAudioSamples: Int = 0,
        isFinal: Bool = false
    ) -> (MLXArray, Int) {
        let timesteps = nCfmTimesteps ?? (meanflow ? 2 : 10)
        let outputMels = callAsFunction(speechTokens, refDict: refDict, nCfmTimesteps: timesteps, finalize: isFinal)
        let melForVocoder = outputMels.transposed(0, 2, 1)
        var (wav, _) = mel2wav.inference(melForVocoder, nil)

        if prevAudioSamples == 0 {
            let fadeLen = trimFade.count
            if wav.shape[1] >= fadeLen {
                let fade = MLXArray(trimFade).expandedDimensions(axis: 0)
                let fadedStart = wav[0..., 0..<fadeLen] * fade
                wav = MLX.concatenated([fadedStart, wav[0..., fadeLen...]], axis: 1)
            }
        }

        let totalSamples = wav.shape[1]
        let newAudio: MLXArray
        if prevAudioSamples > 0 && prevAudioSamples < totalSamples {
            newAudio = wav[0..., prevAudioSamples...]
        } else if prevAudioSamples == 0 {
            newAudio = wav
        } else {
            newAudio = wav[0..., 0..<0]
        }

        return (newAudio, totalSamples)
    }

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let flattenedParams = self.parameters().flattened()
        let paramKeys = Set(flattenedParams.map { $0.0 })
        let expectedShapes = Dictionary(uniqueKeysWithValues: flattenedParams.map { ($0.0, $0.1.shape) })
        let paramValues = Dictionary(uniqueKeysWithValues: flattenedParams.map { ($0.0, $0.1) })
        let hasContiguousCondnet = weights.keys.contains { $0.contains("condnet.1.") }
        var pending: [(String, MLXArray)] = []
        var weightNorm: [String: (g: MLXArray?, v: MLXArray?)] = [:]

        for (key, value) in weights {
            if key.contains("num_batches_tracked") {
                continue
            }
            if key.hasPrefix("tokenizer.") {
                continue
            }

            if key.contains(".parametrizations.weight.original0")
                || key.contains(".parametrizations.weight.original1")
            {
                let base = key
                    .replacingOccurrences(of: ".parametrizations.weight.original0", with: "")
                    .replacingOccurrences(of: ".parametrizations.weight.original1", with: "")
                var entry = weightNorm[base] ?? (g: nil, v: nil)
                if key.contains("original0") {
                    entry.g = value
                } else {
                    entry.v = value
                }
                weightNorm[base] = entry
                continue
            }

            pending.append((key, value))
        }

        for (base, pair) in weightNorm {
            guard let g = pair.g, let v = pair.v else {
                continue
            }
            if v.ndim <= 1 {
                pending.append((base + ".weight", v))
                continue
            }

            var vNorm = v.square()
            for axis in 1..<v.ndim {
                vNorm = MLX.sum(vNorm, axis: axis, keepDims: true)
            }
            vNorm = MLX.sqrt(vNorm)
            pending.append((base + ".weight", v * (g / vNorm)))
        }

        func mapKey(_ key: String) -> String {
            var components = key.split(separator: ".").map(String.init)

            func mapBlocks(_ root: String, mapping: [String: String]) {
                for index in 0..<components.count where components[index] == root {
                    let mappedIndex = index + 2
                    if components.indices.contains(mappedIndex),
                       let mapped = mapping[components[mappedIndex]]
                    {
                        components[mappedIndex] = mapped
                    }
                }
            }

            func mapEmbed(_ name: String) {
                var index = 0
                while index + 2 < components.count {
                    if components[index] == name && components[index + 1] == "out" {
                        let outIndex = components[index + 2]
                        if outIndex == "0" {
                            components[index + 1] = "linear"
                            components.remove(at: index + 2)
                        } else if outIndex == "1" {
                            components[index + 1] = "norm"
                            components.remove(at: index + 2)
                        }
                    }
                    index += 1
                }
            }

            mapBlocks("down_blocks", mapping: ["0": "resnet", "1": "transformer_blocks", "2": "downsample"])
            mapBlocks("mid_blocks", mapping: ["0": "resnet", "1": "transformer_blocks"])
            mapBlocks("up_blocks", mapping: ["0": "resnet", "1": "transformer_blocks", "2": "upsample"])

            mapEmbed("embed")
            mapEmbed("up_embed")

            var index = 0
            while index + 2 < components.count {
                if components[index] == "ff",
                   components[index + 1] == "net",
                   components[index + 2] == "2"
                {
                    components[index + 2] = "1"
                }
                index += 1
            }

            index = 0
            while index + 1 < components.count {
                if components[index] == "mlp", components[index + 1] == "1" {
                    components[index + 1] = "0"
                }
                index += 1
            }

            if !hasContiguousCondnet {
                let condnetMap: [String: String] = ["0": "0", "2": "1", "4": "2", "6": "3", "8": "4"]
                index = 0
                while index + 1 < components.count {
                    if components[index] == "condnet",
                       let mapped = condnetMap[components[index + 1]]
                    {
                        components[index + 1] = mapped
                    }
                    index += 1
                }
            }

            index = 0
            while index + 2 < components.count {
                let name = components[index]
                if (name == "block1" || name == "block2" || name == "final_block"),
                   components[index + 1] == "block"
                {
                    if components[index + 2] == "0" {
                        components[index + 1] = "conv"
                        let nextIndex = index + 3
                        if components.indices.contains(nextIndex), components[nextIndex] == "conv" {
                            components.remove(at: index + 2)
                        } else {
                            components[index + 2] = "conv"
                        }
                    } else if components[index + 2] == "1" || components[index + 2] == "2" {
                        components[index + 1] = "norm"
                        components.remove(at: index + 2)
                    }
                }
                index += 1
            }

            return components.joined(separator: ".")
        }

        func resolveKey(_ key: String) -> String? {
            if paramKeys.contains(key) {
                return key
            }
            guard key.hasSuffix(".weight") || key.hasSuffix(".bias") else {
                return nil
            }
            let suffix = key.hasSuffix(".weight") ? ".weight" : ".bias"
            let base = String(key.dropLast(suffix.count))
            let candidates = [
                base + ".conv" + suffix,
                base + ".conv.conv" + suffix
            ]
            for candidate in candidates where paramKeys.contains(candidate) {
                return candidate
            }
            return nil
        }

        func adjustToExpectedShape(_ value: MLXArray, key: String) -> MLXArray {
            guard let expected = expectedShapes[key] else {
                return value
            }
            if value.shape == expected {
                return value
            }
            if value.ndim == 4 {
                let t = value.transposed(0, 2, 3, 1)
                if t.shape == expected {
                    return t
                }
            } else if value.ndim == 3 {
                let t1 = value.transposed(0, 2, 1)
                if t1.shape == expected {
                    return t1
                }
                let t2 = value.transposed(1, 2, 0)
                if t2.shape == expected {
                    return t2
                }
            }
            return value
        }

        var sanitized: [String: MLXArray] = [:]

        for (rawKey, value) in pending {
            if rawKey.hasPrefix("speaker_encoder.") {
                let stripped = String(rawKey.dropFirst("speaker_encoder.".count))
                let mapped = speakerEncoder.sanitize(weights: [stripped: value])
                for (mappedKey, mappedValue) in mapped {
                    let fullKey = "speaker_encoder.\(mappedKey)"
                    if let resolved = resolveKey(fullKey) {
                        sanitized[resolved] = adjustToExpectedShape(mappedValue, key: resolved)
                    }
                }
                continue
            }

            var newKey = rawKey
            if newKey.hasPrefix("flow.") {
                newKey = String(newKey.dropFirst("flow.".count))
            }

            newKey = mapKey(newKey)
            if let resolved = resolveKey(newKey) {
                sanitized[resolved] = adjustToExpectedShape(value, key: resolved)
            }
        }

        if sanitized["mel2wav.stftWindow"] == nil, let stftWindow = paramValues["mel2wav.stftWindow"] {
            sanitized["mel2wav.stftWindow"] = stftWindow
        }

        return sanitized
    }
}

typealias S3Gen = S3Token2Wav
