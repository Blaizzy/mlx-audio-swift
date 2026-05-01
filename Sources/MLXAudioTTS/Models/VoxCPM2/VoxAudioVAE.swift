//
//  VoxAudioVAE.swift
//  MLXAudio
//
//  VoxCPM2 AudioVAE: causal convolutional encoder/decoder with Snake activations.
//  Asymmetric: encodes at 16kHz, decodes to 48kHz with sample-rate conditioning.
//  Ported from mlx-audio Python: voxcpm2/audio_vae.py
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Causal Conv1d

class VoxCausalConv1d: Module {
    @ModuleInfo var conv: Conv1d
    let padVal: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.padVal = padding
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if padVal > 0 {
            let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padVal * 2, 0)), IntOrPair((0, 0))])
            return conv(padded)
        }
        return conv(x)
    }
}

// MARK: - Causal Transpose Conv1d

class VoxCausalTransposeConv1d: Module {
    @ModuleInfo var conv: ConvTransposed1d
    let padVal: Int
    let outputPadding: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        bias: Bool = true
    ) {
        self.padVal = padding
        self.outputPadding = outputPadding
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        let trim = padVal * 2 - outputPadding
        if trim > 0 {
            y = y[0..., ..<(-trim), 0...]
        }
        return y
    }
}

// MARK: - Snake Activation

class VoxSnake1d: Module {
    let alpha: MLXArray

    init(channels: Int) {
        self.alpha = MLXArray.ones([1, 1, channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1.0 / (alpha + 1e-9)) * MLX.pow(MLX.sin(alpha * x), 2)
    }
}

// MARK: - Causal Residual Unit

class VoxCausalResidualUnit: Module {
    @ModuleInfo var snake1: VoxSnake1d
    @ModuleInfo var conv1: VoxCausalConv1d
    @ModuleInfo var snake2: VoxSnake1d
    @ModuleInfo var conv2: VoxCausalConv1d

    init(dim: Int = 16, dilation: Int = 1, kernel: Int = 7, groups: Int = 1) {
        let pad = ((kernel - 1) * dilation) / 2

        self._snake1.wrappedValue = VoxSnake1d(channels: dim)
        self._conv1.wrappedValue = VoxCausalConv1d(
            inChannels: dim, outChannels: dim,
            kernelSize: kernel, dilation: dilation, padding: pad, groups: groups
        )
        self._snake2.wrappedValue = VoxSnake1d(channels: dim)
        self._conv2.wrappedValue = VoxCausalConv1d(
            inChannels: dim, outChannels: dim, kernelSize: 1
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let res = x
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return res + h
    }
}

// MARK: - Encoder Block

class VoxCausalEncoderBlock: Module {
    @ModuleInfo var res1: VoxCausalResidualUnit
    @ModuleInfo var res2: VoxCausalResidualUnit
    @ModuleInfo var res3: VoxCausalResidualUnit
    @ModuleInfo var snake: VoxSnake1d
    @ModuleInfo var conv: VoxCausalConv1d

    init(outputDim: Int = 16, inputDim: Int? = nil, stride: Int = 1, groups: Int = 1) {
        let inDim = inputDim ?? outputDim / 2

        self._res1.wrappedValue = VoxCausalResidualUnit(dim: inDim, dilation: 1, groups: groups)
        self._res2.wrappedValue = VoxCausalResidualUnit(dim: inDim, dilation: 3, groups: groups)
        self._res3.wrappedValue = VoxCausalResidualUnit(dim: inDim, dilation: 9, groups: groups)
        self._snake.wrappedValue = VoxSnake1d(channels: inDim)
        self._conv.wrappedValue = VoxCausalConv1d(
            inChannels: inDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride,
            padding: Int(ceil(Double(stride) / 2.0))
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = res1(x)
        h = res2(h)
        h = res3(h)
        h = snake(h)
        h = conv(h)
        return h
    }
}

// MARK: - Causal Encoder

class VoxCausalEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: VoxCausalConv1d
    let blocks: [VoxCausalEncoderBlock]
    @ModuleInfo(key: "fc_mu") var fcMu: VoxCausalConv1d

    init(
        dModel: Int = 64,
        latentDim: Int = 32,
        strides: [Int] = [2, 4, 8, 8],
        depthwise: Bool = false
    ) {
        self._convIn.wrappedValue = VoxCausalConv1d(
            inChannels: 1, outChannels: dModel, kernelSize: 7, padding: 3
        )

        var blockList: [VoxCausalEncoderBlock] = []
        var currDim = dModel
        for stride in strides {
            let nextDim = currDim * 2
            let groups = depthwise ? nextDim / 2 : 1
            blockList.append(VoxCausalEncoderBlock(
                outputDim: nextDim, inputDim: currDim, stride: stride, groups: groups
            ))
            currDim = nextDim
        }
        self.blocks = blockList

        self._fcMu.wrappedValue = VoxCausalConv1d(
            inChannels: currDim, outChannels: latentDim, kernelSize: 3, padding: 1
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in blocks {
            h = block(h)
        }
        return fcMu(h)
    }
}

// MARK: - Noise Block

class VoxNoiseBlock: Module {
    @ModuleInfo var linear: VoxCausalConv1d

    init(dim: Int) {
        self._linear.wrappedValue = VoxCausalConv1d(
            inChannels: dim, outChannels: dim, kernelSize: 1, bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let noise = MLXRandom.normal([x.dim(0), x.dim(1), 1]).asType(x.dtype)
        let h = linear(x)
        return x + noise * h
    }
}

// MARK: - Decoder Block

class VoxCausalDecoderBlock: Module {
    let inputChannels: Int

    @ModuleInfo var snake: VoxSnake1d
    @ModuleInfo(key: "conv_t") var convT: VoxCausalTransposeConv1d
    var noise: VoxNoiseBlock?
    @ModuleInfo var res1: VoxCausalResidualUnit
    @ModuleInfo var res2: VoxCausalResidualUnit
    @ModuleInfo var res3: VoxCausalResidualUnit

    init(
        inputDim: Int = 16,
        outputDim: Int = 8,
        stride: Int = 1,
        groups: Int = 1,
        useNoiseBlock: Bool = false
    ) {
        self.inputChannels = inputDim

        self._snake.wrappedValue = VoxSnake1d(channels: inputDim)
        self._convT.wrappedValue = VoxCausalTransposeConv1d(
            inChannels: inputDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride,
            padding: Int(ceil(Double(stride) / 2.0)),
            outputPadding: stride % 2
        )

        if useNoiseBlock {
            self.noise = VoxNoiseBlock(dim: outputDim)
        }

        self._res1.wrappedValue = VoxCausalResidualUnit(dim: outputDim, dilation: 1, groups: groups)
        self._res2.wrappedValue = VoxCausalResidualUnit(dim: outputDim, dilation: 3, groups: groups)
        self._res3.wrappedValue = VoxCausalResidualUnit(dim: outputDim, dilation: 9, groups: groups)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = convT(h)
        if let noise {
            h = noise(h)
        }
        h = res1(h)
        h = res2(h)
        h = res3(h)
        return h
    }
}

// MARK: - Sample Rate Conditioning Layer

class VoxSRCondLayer: Module {
    let condType: String

    @ModuleInfo(key: "scale_embed") var scaleEmbed: Embedding?
    @ModuleInfo(key: "bias_embed") var biasEmbed: Embedding?
    @ModuleInfo(key: "cond_embed") var condEmbed: Embedding?
    @ModuleInfo(key: "out_snake") var outSnake: VoxSnake1d?
    @ModuleInfo(key: "out_conv") var outConv: VoxCausalConv1d?
    let hasOutLayer: Bool

    init(
        inputDim: Int,
        srBinBuckets: Int,
        condType: String = "scale_bias",
        condDim: Int = 128,
        outLayer: Bool = false
    ) {
        self.condType = condType
        self.hasOutLayer = outLayer

        if condType == "scale_bias" || condType == "scale_bias_init" {
            self._scaleEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: inputDim)
            self._biasEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: inputDim)
        } else if condType == "add" {
            self._condEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: inputDim)
        } else if condType == "concat" {
            self._condEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: condDim)
        }

        if outLayer {
            let outDim = condType == "concat" ? inputDim + condDim : inputDim
            self._outSnake.wrappedValue = VoxSnake1d(channels: outDim)
            self._outConv.wrappedValue = VoxCausalConv1d(
                inChannels: outDim, outChannels: inputDim, kernelSize: 1
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray, srCond: MLXArray) -> MLXArray {
        var h = x
        if condType == "scale_bias" || condType == "scale_bias_init", let scaleEmbed, let biasEmbed {
            let scale = scaleEmbed(srCond)[0..., .newAxis, 0...]
            let bias = biasEmbed(srCond)[0..., .newAxis, 0...]
            h = h * scale + bias
        } else if condType == "add", let condEmbed {
            h = h + condEmbed(srCond)[0..., .newAxis, 0...]
        } else if condType == "concat", let condEmbed {
            var cond = condEmbed(srCond)[0..., .newAxis, 0...]
            cond = MLX.broadcast(cond, to: [h.dim(0), h.dim(1), cond.dim(-1)])
            h = MLX.concatenated([h, cond], axis: -1)
        }

        if hasOutLayer, let outSnake, let outConv {
            h = outSnake(h)
            h = outConv(h)
        }

        return h
    }
}

// MARK: - Causal Decoder

class VoxCausalDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Module
    let blocks: [VoxCausalDecoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: VoxSnake1d
    @ModuleInfo(key: "conv_out") var convOut: VoxCausalConv1d
    let srCondLayers: [VoxSRCondLayer?]
    var srBoundaries: MLXArray?

    init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        depthwise: Bool = false,
        dOut: Int = 1,
        useNoiseBlock: Bool = false,
        srBinBoundaries: [Int]? = nil,
        condType: String = "scale_bias",
        condDim: Int = 128,
        condOutLayer: Bool = false
    ) {
        // Conv in (depthwise: 2-layer sequential, else single conv)
        if depthwise {
            let depthConv = VoxCausalConv1d(
                inChannels: inputChannel, outChannels: inputChannel,
                kernelSize: 7, padding: 3, groups: inputChannel
            )
            let pointConv = VoxCausalConv1d(
                inChannels: inputChannel, outChannels: channels, kernelSize: 1
            )
            self._convIn.wrappedValue = VoxSequential(depthConv, pointConv)
        } else {
            self._convIn.wrappedValue = VoxCausalConv1d(
                inChannels: inputChannel, outChannels: channels, kernelSize: 7, padding: 3
            )
        }

        var blockList: [VoxCausalDecoderBlock] = []
        for (i, stride) in rates.enumerated() {
            let inDim = channels / (1 << i)
            let outDim = channels / (1 << (i + 1))
            let groups = depthwise ? outDim : 1
            blockList.append(VoxCausalDecoderBlock(
                inputDim: inDim, outputDim: outDim, stride: stride,
                groups: groups, useNoiseBlock: useNoiseBlock
            ))
        }
        self.blocks = blockList

        let finalDim = channels / (1 << rates.count)
        self._snakeOut.wrappedValue = VoxSnake1d(channels: finalDim)
        self._convOut.wrappedValue = VoxCausalConv1d(
            inChannels: finalDim, outChannels: dOut, kernelSize: 7, padding: 3
        )

        // SR conditioning
        if let boundaries = srBinBoundaries {
            self.srBoundaries = MLXArray(boundaries.map { Int32($0) })
            let srBinBuckets = boundaries.count + 1
            var condLayers: [VoxSRCondLayer?] = []
            for block in blockList {
                condLayers.append(VoxSRCondLayer(
                    inputDim: block.inputChannels,
                    srBinBuckets: srBinBuckets,
                    condType: condType,
                    condDim: condDim,
                    outLayer: condOutLayer
                ))
            }
            self.srCondLayers = condLayers
        } else {
            self.srBoundaries = nil
            self.srCondLayers = []
        }

        super.init()
    }

    func getSRIdx(_ sr: MLXArray) -> MLXArray {
        guard let boundaries = srBoundaries else {
            return MLXArray([Int32(0)])
        }
        return MLX.sum(sr .>= boundaries).asType(.int32).reshaped([1])
    }

    func callAsFunction(_ x: MLXArray, srCond: MLXArray? = nil) -> MLXArray {
        var h: MLXArray
        if let seq = convIn as? VoxSequential {
            h = seq(x)
        } else if let conv = convIn as? VoxCausalConv1d {
            h = conv(x)
        } else {
            fatalError("Unexpected conv_in type")
        }

        if let srCond, srBoundaries != nil {
            let srIdx = getSRIdx(srCond)
            for (block, condLayer) in zip(blocks, srCondLayers) {
                if let condLayer {
                    h = condLayer(h, srCond: srIdx)
                }
                h = block(h)
            }
        } else {
            for block in blocks {
                h = block(h)
            }
        }

        h = snakeOut(h)
        h = convOut(h)
        return MLX.tanh(h)
    }
}

// MARK: - Simple Sequential (for depthwise conv_in)

class VoxSequential: Module {
    let layers: [Module]

    init(_ layers: Module...) {
        self.layers = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let conv = layer as? VoxCausalConv1d {
                h = conv(h)
            }
        }
        return h
    }
}

// MARK: - AudioVAE

class VoxAudioVAE: Module {
    let config: VoxCPM2AudioVAEConfig
    let hopLength: Int
    let latentDim: Int
    let encodeSampleRate: Int
    let outSampleRate: Int
    let chunkSize: Int
    let decodeChunkSize: Int

    @ModuleInfo var encoder: VoxCausalEncoder
    @ModuleInfo var decoder: VoxCausalDecoder

    init(_ config: VoxCPM2AudioVAEConfig) {
        self.config = config

        self.hopLength = config.encoderRates.reduce(1, *)
        self.latentDim = config.latentDim
        self.encodeSampleRate = config.sampleRate
        self.outSampleRate = config.outSampleRate
        self.chunkSize = config.encoderRates.reduce(1, *)
        self.decodeChunkSize = config.decoderRates.reduce(1, *)

        self._encoder.wrappedValue = VoxCausalEncoder(
            dModel: config.encoderDim,
            latentDim: config.latentDim,
            strides: config.encoderRates,
            depthwise: config.depthwise
        )
        self._decoder.wrappedValue = VoxCausalDecoder(
            inputChannel: config.latentDim,
            channels: config.decoderDim,
            rates: config.decoderRates,
            depthwise: config.depthwise,
            dOut: 1,
            useNoiseBlock: config.useNoiseBlock,
            srBinBoundaries: config.srBinBoundaries,
            condType: config.condType,
            condDim: config.condDim,
            condOutLayer: config.condOutLayer
        )

        super.init()
    }

    func encode(_ x: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        var inp = x
        if inp.ndim == 2 {
            inp = inp.expandedDimensions(axis: -1)
        }
        // Ensure (B, T, C) layout
        if inp.dim(1) < inp.dim(2) {
            inp = inp.transposed(0, 2, 1)
        }
        inp = preprocess(inp, sampleRate: sampleRate)
        return encoder(inp)
    }

    func decode(_ z: MLXArray, srCond: MLXArray? = nil) -> MLXArray {
        let sr = srCond ?? MLXArray([Int32(outSampleRate)])
        let out = decoder(z, srCond: sr)
        return out.squeezed(axis: -1)
    }

    private func preprocess(_ audioData: MLXArray, sampleRate: Int?) -> MLXArray {
        let length = audioData.dim(1)
        let padTo = hopLength
        let rightPad = (Int(ceil(Double(length) / Double(padTo))) * padTo) - length
        if rightPad > 0 {
            return MLX.padded(audioData, widths: [IntOrPair((0, 0)), IntOrPair((0, rightPad)), IntOrPair((0, 0))])
        }
        return audioData
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var filtered = weights.filter { !$0.key.contains("fc_logvar") }

        // Fuse weight_norm: weight_g + weight_v → weight
        var fused = [String: MLXArray]()
        var processedKeys = Set<String>()

        for (key, value) in filtered {
            if processedKeys.contains(key) { continue }

            if key.hasSuffix(".weight_g") {
                let base = String(key.dropLast(9))
                let vKey = base + ".weight_v"
                if let g = filtered[key], let v = filtered[vKey] {
                    let vFlat = v.reshaped(v.dim(0), -1)
                    let norm = MLX.sqrt(MLX.sum(vFlat * vFlat, axis: 1)).reshaped(g.shape)
                    let w = g * (v / (norm + 1e-9))
                    fused[base + ".weight"] = w
                    processedKeys.insert(key)
                    processedKeys.insert(vKey)
                    continue
                }
            }
            if key.hasSuffix(".weight_v") { continue }

            fused[key] = value
        }

        // Detect whether keys are raw PyTorch format (encoder.block.N)
        // or already-sanitized MLX Python format (encoder.blocks.layers.N).
        // Quantized models (e.g. VoxCPM2-4bit) ship with pre-sanitized keys.
        let isRawPyTorch = fused.keys.contains { $0.hasPrefix("encoder.block.") || $0.hasPrefix("decoder.model.") }

        var remapped: [String: MLXArray]
        if isRawPyTorch {
            remapped = remapPyTorchKeys(fused)
        } else {
            remapped = fused
        }

        // Python uses nn.Sequential for blocks → adds "layers" level.
        // Swift uses plain [Module] arrays → no "layers" level.
        // Strip "blocks.layers.N" → "blocks.N" for both encoder and decoder.
        var stripped = [String: MLXArray]()
        let blocksLayersPattern = try! NSRegularExpression(pattern: #"\bblocks\.layers\.(\d+)"#)
        for (key, value) in remapped {
            let range = NSRange(key.startIndex..., in: key)
            let newKey = blocksLayersPattern.stringByReplacingMatches(
                in: key, range: range, withTemplate: "blocks.$1"
            )
            stripped[newKey] = value
        }
        remapped = stripped

        // Remap snake_case property names to Swift camelCase.
        // The Swift VoxCausalDecoder uses `let srCondLayers` (camelCase),
        // but safetensors keys use `sr_cond_layers` (snake_case).
        var camelCased = [String: MLXArray]()
        for (key, value) in remapped {
            let newKey = key.replacingOccurrences(of: "sr_cond_layers", with: "srCondLayers")
            camelCased[newKey] = value
        }
        remapped = camelCased

        // Handle sr_boundaries buffer
        if let srBounds = remapped.removeValue(forKey: "decoder._sr_boundaries") {
            remapped["decoder.srBoundaries"] = srBounds
        }
        if let srBounds = remapped.removeValue(forKey: "decoder.sr_boundaries") {
            remapped["decoder.srBoundaries"] = srBounds
        }

        // Swift VoxCausalConv1d/VoxCausalTransposeConv1d wrap inner conv as
        // @ModuleInfo var conv, adding ".conv." to the key path.
        // Python uses inheritance (flat). Insert ".conv." for wrapper types.
        var finalWeights = [String: MLXArray]()
        let convWrapperSuffixes = [
            ".conv_t", ".conv_in", ".conv_out", ".fc_mu",
            ".conv1", ".conv2", ".conv", ".linear", ".out_conv"
        ]
        for (key, value) in remapped {
            var k = key
            for terminal in [".weight", ".bias"] {
                guard k.hasSuffix(terminal) else { continue }
                let stem = String(k.dropLast(terminal.count))
                let isConvWrapper = convWrapperSuffixes.contains { stem.hasSuffix($0) }
                let isSeqLayer = stem.range(
                    of: #"\.conv_in\.layers\.\d+$"#, options: .regularExpression
                ) != nil
                if isConvWrapper || isSeqLayer {
                    k = stem + ".conv" + terminal
                }
            }
            finalWeights[k] = value
        }

        return finalWeights
    }

    private func remapPyTorchKeys(_ fused: [String: MLXArray]) -> [String: MLXArray] {
        var remapped = [String: MLXArray]()
        let numDecBlocks = config.decoderRates.count

        for (key, value) in fused {
            let parts = key.split(separator: ".").map(String.init)
            var newParts: [String]

            if parts[0] == "encoder" {
                if parts[1] == "block" {
                    let idx = Int(parts[2])!
                    if idx == 0 {
                        newParts = ["encoder", "conv_in"] + Array(parts[3...])
                    } else {
                        newParts = ["encoder", "blocks", String(idx - 1)] + Array(parts[3...])
                    }
                } else {
                    newParts = parts
                }
            } else if parts[0] == "decoder" {
                if parts[1] == "model" {
                    let idx = Int(parts[2])!
                    if idx == 0 {
                        newParts = ["decoder", "conv_in", "layers", "0"] + Array(parts[3...])
                    } else if idx == 1 {
                        newParts = ["decoder", "conv_in", "layers", "1"] + Array(parts[3...])
                    } else if idx >= 2 && idx < 2 + numDecBlocks {
                        newParts = ["decoder", "blocks", String(idx - 2)] + Array(parts[3...])
                    } else if idx == 2 + numDecBlocks {
                        newParts = ["decoder", "snake_out"] + Array(parts[3...])
                    } else if idx == 2 + numDecBlocks + 1 {
                        newParts = ["decoder", "conv_out"] + Array(parts[3...])
                    } else {
                        newParts = parts
                    }
                } else if parts[1] == "sr_cond_model" {
                    let ptIdx = Int(parts[2])!
                    let offset = config.depthwise ? 2 : 1
                    let mlxIdx = ptIdx - offset
                    newParts = ["decoder", "sr_cond_layers", String(mlxIdx)] + Array(parts[3...])
                } else if parts[1] == "sr_bin_boundaries" {
                    remapped["decoder.srBoundaries"] = value
                    continue
                } else {
                    newParts = parts
                }
            } else {
                newParts = parts
            }

            // Sub-block remapping: block.N → named components
            var finalParts: [String] = []
            var i = 0
            while i < newParts.count {
                let p = newParts[i]

                if p == "block" && i + 1 < newParts.count, let idx = Int(newParts[i + 1]) {
                    let isEncoder = newParts[..<i].contains("encoder") && newParts[..<i].contains("blocks")
                    let isDecoder = newParts[..<i].contains("decoder") && newParts[..<i].contains("blocks")

                    if isEncoder && finalParts.count == 3 {
                        let mapping = [0: "res1", 1: "res2", 2: "res3", 3: "snake", 4: "conv"]
                        finalParts.append(mapping[idx] ?? "unknown_\(idx)")
                        i += 2
                        continue
                    }

                    if isDecoder && finalParts.count == 3 {
                        let mapping = [0: "snake", 1: "conv_t", 2: "res1", 3: "res2", 4: "res3"]
                        finalParts.append(mapping[idx] ?? "unknown_\(idx)")
                        i += 2
                        continue
                    }

                    let resMapping = [0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"]
                    if let name = resMapping[idx] {
                        finalParts.append(name)
                        i += 2
                        continue
                    }
                }

                finalParts.append(p)
                i += 1
            }

            let newKey = finalParts.joined(separator: ".")
            remapped[newKey] = value
        }

        return remapped
    }
}
