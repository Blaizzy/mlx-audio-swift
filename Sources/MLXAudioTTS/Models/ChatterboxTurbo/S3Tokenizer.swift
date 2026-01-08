//
//  S3Tokenizer.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCodecs

struct S3TokenizerConfig: Sendable {
    var nMels: Int = 128
    var nAudioCtx: Int = 1500
    var nAudioState: Int = 1280
    var nAudioHead: Int = 20
    var nAudioLayer: Int = 6
    var nCodebookSize: Int = 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3
}

private func precomputeFreqsCis(
    dim: Int,
    end: Int,
    theta: Float = 10_000.0,
    scaling: Float? = nil
) -> (MLXArray, MLXArray) {
    let inv = MLXArray.arange(0.0, Double(dim), step: 2.0, dtype: .float32)
    let freqs = MLXArray(Float(1.0)) / MLX.pow(MLXArray(theta), inv / Float(dim))
    var t = MLXArray.arange(0.0, Double(end), step: 1.0, dtype: .float32)
    if let scaling {
        t = t * scaling
    }

    let freqsMat = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    let cosFreqs = MLX.concatenated([MLX.cos(freqsMat), MLX.cos(freqsMat)], axis: -1)
    let sinFreqs = MLX.concatenated([MLX.sin(freqsMat), MLX.sin(freqsMat)], axis: -1)
    return (cosFreqs, sinFreqs)
}

private func applyRotaryEmb(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cosExpanded = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let sinExpanded = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)

    let qParts = q.split(parts: 2, axis: -1)
    let kParts = k.split(parts: 2, axis: -1)

    let qRotated = MLX.concatenated([-qParts[1], qParts[0]], axis: -1)
    let kRotated = MLX.concatenated([-kParts[1], kParts[0]], axis: -1)

    let qOut = q * cosExpanded + qRotated * sinExpanded
    let kOut = k * cosExpanded + kRotated * sinExpanded

    return (qOut, kOut)
}

final class GELULayer: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(x)
    }
}

class S3MultiHeadAttention: Module {
    let nHead: Int

    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(nState: Int, nHead: Int) {
        self.nHead = nHead
        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        let q = query(x)
        let k = key(x)
        let v = value(x)
        let (wv, qk) = qkvAttention(q: q, k: k, v: v, mask: mask)
        return (out(wv), qk)
    }

    func qkvAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        let batch = q.dim(0)
        let length = q.dim(1)
        let dim = q.dim(2)
        let scale = pow(Float(dim / nHead), -0.25)
        let headDim = dim / nHead

        var q = q.reshaped(batch, length, nHead, headDim).transposed(0, 2, 1, 3) * scale
        var k = k.reshaped(batch, length, nHead, headDim).transposed(0, 2, 1, 3) * scale
        let v = v.reshaped(batch, length, nHead, headDim).transposed(0, 2, 1, 3)

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2))
        if let mask {
            scores = scores + mask
        }

        let attn = MLX.softmax(scores, axis: -1)
        let output = MLX.matmul(attn, v)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, dim)

        return (output, nil)
    }
}

final class S3FSMNMultiHeadAttention: S3MultiHeadAttention {
    @ModuleInfo(key: "fsmn_block") var fsmnBlock: MLXNN.Conv1d
    private let leftPadding: Int
    private let rightPadding: Int

    init(nState: Int, nHead: Int, kernelSize: Int = 31) {
        self.leftPadding = (kernelSize - 1) / 2
        self.rightPadding = kernelSize - 1 - leftPadding
        self._fsmnBlock.wrappedValue = MLXNN.Conv1d(
            inputChannels: nState,
            outputChannels: nState,
            kernelSize: kernelSize,
            padding: 0,
            groups: nState,
            bias: false
        )
        super.init(nState: nState, nHead: nHead)
    }

    private func forwardFsmn(_ inputs: MLXArray, maskPad: MLXArray?) -> MLXArray {
        let batch = inputs.dim(0)
        let length = inputs.dim(1)

        var merged = inputs.reshaped(batch, length, -1)
        let channels = merged.dim(2)
        if let maskPad {
            merged = merged * maskPad
        }

        let padLeft = MLXArray.zeros([batch, leftPadding, channels], type: Float.self)
        let padRight = MLXArray.zeros([batch, rightPadding, channels], type: Float.self)
        let padded = MLX.concatenated([padLeft, merged, padRight], axis: 1)
        var out = fsmnBlock(padded)
        out = out + merged

        if let maskPad {
            out = out * maskPad
        }

        return out
    }

    func qkvAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        mask: MLXArray? = nil,
        maskPad: MLXArray? = nil,
        freqsCis: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray?) {
        let batch = q.dim(0)
        let length = q.dim(1)
        let dim = q.dim(2)
        let scale = pow(Float(dim / nHead), -0.25)
        let headDim = dim / nHead

        var q = q.reshaped(batch, length, nHead, headDim)
        var k = k.reshaped(batch, length, nHead, headDim)
        var v = v.reshaped(batch, length, nHead, headDim)

        if let freqsCis {
            let cos = freqsCis.0[0..<length, 0...]
            let sin = freqsCis.1[0..<length, 0...]
            (q, k) = applyRotaryEmb(q: q, k: k, cos: cos, sin: sin)
        }

        let fsmMemory = forwardFsmn(v, maskPad: maskPad)

        q = q.transposed(0, 2, 1, 3) * scale
        k = k.transposed(0, 2, 1, 3) * scale
        v = fsmMemory.reshaped(batch, length, nHead, headDim).transposed(0, 2, 1, 3)

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2))
        if let mask {
            scores = scores + mask
        }

        let attn = MLX.softmax(scores, axis: -1)
        let output = MLX.matmul(attn, v)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, dim)

        return (output, nil)
    }
}

final class S3ResidualAttentionBlock: Module {
    @ModuleInfo(key: "attn") var attn: S3FSMNMultiHeadAttention
    @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLXNN.Sequential
    @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

    init(nState: Int, nHead: Int, kernelSize: Int = 31) {
        self._attn.wrappedValue = S3FSMNMultiHeadAttention(nState: nState, nHead: nHead, kernelSize: kernelSize)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState, eps: 1e-6)
        self._mlp.wrappedValue = MLXNN.Sequential(layers: [
            Linear(nState, nState * 4),
            GELULayer(),
            Linear(nState * 4, nState)
        ])
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        maskPad: MLXArray? = nil,
        freqsCis: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let normed = attnLn(x)
        let q = attn.query(normed)
        let k = attn.key(normed)
        let v = attn.value(normed)

        let (attnOut, _) = attn.qkvAttention(
            q: q,
            k: k,
            v: v,
            mask: mask,
            maskPad: maskPad,
            freqsCis: freqsCis
        )

        var h = x + attn.out(attnOut)
        h = h + mlp(mlpLn(h))
        return h
    }
}

final class S3AudioEncoderV2: Module {
    let stride: Int

    @ModuleInfo(key: "conv1") var conv1: MLXNN.Conv1d
    @ModuleInfo(key: "conv2") var conv2: MLXNN.Conv1d

    let freqsCis: (MLXArray, MLXArray)
    let blocks: [S3ResidualAttentionBlock]

    init(
        nMels: Int,
        nState: Int,
        nHead: Int,
        nLayer: Int,
        stride: Int
    ) {
        self.stride = stride
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: nMels,
            outputChannels: nState,
            kernelSize: 3,
            stride: stride,
            padding: 1
        )
        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: nState,
            outputChannels: nState,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        self.freqsCis = precomputeFreqsCis(dim: 64, end: 2048)
        self.blocks = (0..<nLayer).map { _ in S3ResidualAttentionBlock(nState: nState, nHead: nHead) }
    }

    func callAsFunction(_ x: MLXArray, xLen: MLXArray) -> (MLXArray, MLXArray) {
        var lengths = xLen.asArray(Int32.self).map { Int($0) }
        var mask = s3MakeNonPadMask(lengths: xLen)
        mask = mask.expandedDimensions(axis: 1)

        var h = x.transposed(0, 2, 1)
        let maskTransposed = mask.transposed(0, 2, 1)
        h = conv1(h * maskTransposed)
        h = gelu(h)

        lengths = lengths.map { ($0 + 2 - (3 - 1) - 1) / stride + 1 }
        var xLenUpdated = MLXArray(lengths.map { Int32($0) })

        var mask2 = s3MakeNonPadMask(lengths: xLenUpdated)
        let mask2Transposed = mask2.expandedDimensions(axis: -1)
        h = conv2(h * mask2Transposed)
        h = gelu(h)

        lengths = lengths.map { ($0 + 2 - (3 - 1) - 1) / 2 + 1 }
        xLenUpdated = MLXArray(lengths.map { Int32($0) })

        let maskPad = s3MakeNonPadMask(lengths: xLenUpdated).expandedDimensions(axis: -1)
        var maskBias = s3MaskToBias(s3MakeNonPadMask(lengths: xLenUpdated), dtype: h.dtype)
        maskBias = maskBias.expandedDimensions(axis: 1)

        for block in blocks {
            h = block(h, mask: maskBias.expandedDimensions(axis: 1), maskPad: maskPad, freqsCis: freqsCis)
        }

        return (h, xLenUpdated)
    }
}

final class S3FSQCodebook: Module {
    let level: Int

    @ModuleInfo(key: "project_down") var projectDown: Linear

    init(dim: Int, level: Int = 3) {
        self.level = level
        self._projectDown.wrappedValue = Linear(dim, 8)
    }

    func preprocess(_ x: MLXArray) -> MLXArray {
        x.reshaped(-1, x.dim(2))
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let xShape = x.shape
        var h = preprocess(x)
        h = projectDown(h).asType(.float32)
        h = MLX.tanh(h)
        h = h * MLXArray(Float(0.9990000128746033))
        h = MLX.round(h) + MLXArray(Float(1.0))

        let powerCount = 1 << level
        let powers = MLX.pow(MLXArray(Float(level)), MLXArray.arange(0, powerCount, dtype: .float32))
        let mu = MLX.sum(h * powers.expandedDimensions(axis: 0), axis: -1)
        return mu.reshaped(xShape[0], xShape[1]).asType(.int32)
    }
}

final class S3FSQVectorQuantization: Module {
    @ModuleInfo(key: "fsq_codebook") var fsqCodebook: S3FSQCodebook
    let codebookSize: Int

    init(dim: Int, codebookSize: Int) {
        self.codebookSize = codebookSize
        self._fsqCodebook.wrappedValue = S3FSQCodebook(dim: dim, level: 3)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        fsqCodebook.encode(x)
    }
}

final class S3TokenizerV2: Module {
    let config: S3TokenizerConfig

    @ModuleInfo(key: "encoder") var encoder: S3AudioEncoderV2
    @ModuleInfo(key: "quantizer") var quantizer: S3FSQVectorQuantization

    init(name: String, config: S3TokenizerConfig = S3TokenizerConfig()) {
        var config = config
        if !name.contains("v1") {
            precondition(name.contains("v2"))
            config.nCodebookSize = 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3
        }
        self.config = config
        self._encoder.wrappedValue = S3AudioEncoderV2(
            nMels: config.nMels,
            nState: config.nAudioState,
            nHead: config.nAudioHead,
            nLayer: config.nAudioLayer,
            stride: 2
        )
        self._quantizer.wrappedValue = S3FSQVectorQuantization(
            dim: config.nAudioState,
            codebookSize: config.nCodebookSize
        )
    }

    func callAsFunction(_ mel: MLXArray, _ melLen: MLXArray) -> (MLXArray, MLXArray) {
        quantize(mel, melLen)
    }

    func quantize(_ mel: MLXArray, _ melLen: MLXArray) -> (MLXArray, MLXArray) {
        let maxFrames = 3000
        let lengths = melLen.asArray(Int32.self).map { Int($0) }
        let hasLong = lengths.contains { $0 > maxFrames }

        if hasLong {
            return quantizeMixedBatch(mel, melLen, maxFrames: maxFrames)
        }

        let (hidden, codeLen) = encoder(mel, xLen: melLen)
        let code = quantizer.encode(hidden)
        return (code, codeLen)
    }

    private struct SegmentInfo {
        let batchIndex: Int
        let isLongAudio: Bool
        let segmentIndex: Int
        let totalSegments: Int
    }

    private func quantizeMixedBatch(
        _ mel: MLXArray,
        _ melLen: MLXArray,
        maxFrames: Int
    ) -> (MLXArray, MLXArray) {
        let batchSize = mel.shape[0]
        let lengths = melLen.asArray(Int32.self).map { Int($0) }

        let sampleRate = 16_000
        let hopLength = 160
        let windowSize = 30
        let overlap = 4

        let framesPerWindow = windowSize * sampleRate / hopLength
        let framesPerOverlap = overlap * sampleRate / hopLength
        let framesPerStride = framesPerWindow - framesPerOverlap

        var allSegments: [MLXArray] = []
        var allSegmentsLen: [Int] = []
        var segmentInfo: [SegmentInfo] = []

        for batchIndex in 0..<batchSize {
            let audioLen = lengths[batchIndex]
            let isLong = audioLen > maxFrames
            let audioMel = mel[batchIndex, 0..., 0...]

            if !isLong {
                var segment = audioMel[0..<config.nMels, 0..<audioLen]
                var segLen = audioLen
                if segLen < framesPerWindow {
                    let padSize = framesPerWindow - segLen
                    segment = MLX.padded(segment, widths: [.init(0), .init((0, padSize))])
                }
                allSegments.append(segment)
                allSegmentsLen.append(segLen)
                segmentInfo.append(SegmentInfo(batchIndex: batchIndex, isLongAudio: false, segmentIndex: 0, totalSegments: 1))
            } else {
                var start = 0
                var segIndex = 0
                while start < audioLen {
                    let end = min(start + framesPerWindow, audioLen)
                    var segment = audioMel[0..<config.nMels, start..<end]
                    let segLen = segment.shape[1]
                    if segLen < framesPerWindow {
                        let padSize = framesPerWindow - segLen
                        segment = MLX.padded(segment, widths: [.init(0), .init((0, padSize))])
                    }
                    allSegments.append(segment)
                    allSegmentsLen.append(segLen)
                    segmentInfo.append(SegmentInfo(batchIndex: batchIndex, isLongAudio: true, segmentIndex: segIndex, totalSegments: 0))
                    segIndex += 1
                    start += framesPerStride
                }

                for i in 0..<segmentInfo.count where segmentInfo[i].batchIndex == batchIndex && segmentInfo[i].isLongAudio {
                    segmentInfo[i] = SegmentInfo(batchIndex: batchIndex, isLongAudio: true, segmentIndex: segmentInfo[i].segmentIndex, totalSegments: segIndex)
                }
            }
        }

        if allSegments.isEmpty {
            return (
                MLXArray.zeros([batchSize, 0], type: Int32.self),
                MLXArray.zeros([batchSize], type: Int32.self)
            )
        }

        let unifiedBatch = MLX.stacked(allSegments, axis: 0)
        let unifiedLens = MLXArray(allSegmentsLen.map { Int32($0) })
        let (hidden, codeLen) = encoder(unifiedBatch, xLen: unifiedLens)
        let codes = quantizer.encode(hidden)

        var results: [Int: ([Int], Int)] = [:]
        var longResults: [Int: [[Int]]] = [:]

        for (segIndex, info) in segmentInfo.enumerated() {
            let segLen = Int(codeLen[segIndex].item(Int32.self))
            let tokenArray = codes[segIndex, 0..<segLen].asArray(Int32.self).map { Int($0) }

            if !info.isLongAudio {
                results[info.batchIndex] = (tokenArray, tokenArray.count)
            } else {
                longResults[info.batchIndex, default: []].append(tokenArray)
            }
        }

        for batchIndex in 0..<batchSize {
            if let segments = longResults[batchIndex] {
                let merged = s3MergeTokenizedSegments(segments, overlap: overlap, tokenRate: S3TokenRate)
                results[batchIndex] = (merged, merged.count)
            }
        }

        let maxCodeLen = results.values.map { $0.1 }.max() ?? 0
        var outputList: [MLXArray] = []
        var lenList: [Int32] = []

        for batchIndex in 0..<batchSize {
            let (tokens, lenVal) = results[batchIndex] ?? ([], 0)
            var codeTensor = MLXArray(tokens.map { Int32($0) })
            if codeTensor.shape[0] < maxCodeLen {
                let padSize = maxCodeLen - codeTensor.shape[0]
                codeTensor = MLX.padded(codeTensor, widths: [.init((0, padSize))])
            }
            outputList.append(codeTensor)
            lenList.append(Int32(lenVal))
        }

        let outputCodes = MLX.stacked(outputList, axis: 0)
        let outputLens = MLXArray(lenList)
        return (outputCodes, outputLens)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("freqs_cis") || key.contains("_mel_filters") {
                continue
            }
            if key.hasPrefix("onnx::") {
                continue
            }

            var newKey = key
            newKey = newKey.replacingOccurrences(of: "quantizer._codebook.", with: "quantizer.fsq_codebook.")
            newKey = newKey.replacingOccurrences(of: "quantizer.codebook.", with: "quantizer.fsq_codebook.")

            if let regex = try? NSRegularExpression(pattern: "\\.mlp\\.(\\d+)\\.") {
                let range = NSRange(location: 0, length: newKey.utf16.count)
                newKey = regex.stringByReplacingMatches(in: newKey, range: range, withTemplate: ".mlp.layers.$1.")
            }

            if (newKey.contains(".conv1.") || newKey.contains(".conv2.") || newKey.contains(".fsmn_block.")),
               newKey.contains("weight"),
               value.ndim == 3,
               value.shape[1] > value.shape[2]
            {
                sanitized[newKey] = value.transposed(0, 2, 1)
            } else {
                sanitized[newKey] = value
            }
        }

        return sanitized
    }
}
