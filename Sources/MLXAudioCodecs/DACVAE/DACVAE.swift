//
//  DACVAE.swift
//  MLXAudioCodecs
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN
import HuggingFace
import MLXAudioCore

// MARK: - Quantizer Projections

/// Quantizer input projection (VAE-style with mean/logvar).
public class DACVAEQuantizerInProj: Module {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray

    public init(inDim: Int, outDim: Int) {
        // Projects to 2*outDim for mean and logvar
        let outChannels = outDim * 2

        let scale = sqrt(1.0 / Float(inDim))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, 1, inDim]
        )
        self._weightG.wrappedValue = dacvaeNormalizeWeight(weightInit)
        self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue + 1e-12)
        self._biasParam.wrappedValue = MLXArray.zeros([outChannels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weightG * weightV / dacvaeNormalizeWeight(weightV)
        var y = MLX.conv1d(x, weight, stride: 1, padding: 0)
        y = y + biasParam
        return y
    }
}

/// Quantizer output projection.
public class DACVAEQuantizerOutProj: Module {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray

    public init(inDim: Int, outDim: Int) {
        let scale = sqrt(1.0 / Float(inDim))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outDim, 1, inDim]
        )
        self._weightG.wrappedValue = dacvaeNormalizeWeight(weightInit)
        self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue + 1e-12)
        self._biasParam.wrappedValue = MLXArray.zeros([outDim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weightG * weightV / dacvaeNormalizeWeight(weightV)
        var y = MLX.conv1d(x, weight, stride: 1, padding: 0)
        y = y + biasParam
        return y
    }
}

// MARK: - Full Decoder with Watermarking

/// DACVAE Decoder with watermarking support.
public class DACVAEFullDecoder: Module {
    let alpha: Float

    @ModuleInfo(key: "conv_in") var convIn: DACVAEWNConv1d
    @ModuleInfo(key: "blocks") var blocks: [DACVAEDecoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: DACVAESnake1d
    @ModuleInfo(key: "conv_out") var convOut: DACVAEWNConv1d
    @ModuleInfo(key: "wm_model") var wmModel: DACVAEWatermarker

    public init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        wmRates: [Int]? = nil,
        wmChannels: Int = 32,
        nbits: Int = 16,
        dOut: Int = 1,
        dWmOut: Int = 128
    ) {
        let wmRatesActual = wmRates ?? [8, 5, 4, 2]

        // First conv layer
        self._convIn.wrappedValue = DACVAEWNConv1d(
            inChannels: inputChannel,
            outChannels: channels,
            kernelSize: 7,
            padding: 3
        )

        // Decoder blocks
        var decoderBlocks: [DACVAEDecoderBlock] = []
        for (i, (stride, wmStride)) in zip(rates, wmRatesActual).enumerated() {
            let inputDim = channels / Int(pow(2.0, Double(i)))
            let outputDim = channels / Int(pow(2.0, Double(i + 1)))
            decoderBlocks.append(DACVAEDecoderBlock(
                inputDim: inputDim,
                outputDim: outputDim,
                stride: stride,
                strideWM: wmStride
            ))
        }
        self._blocks.wrappedValue = decoderBlocks

        // Final output layers (shared with watermark encoder)
        let finalDim = channels / Int(pow(2.0, Double(rates.count)))
        self._snakeOut.wrappedValue = DACVAESnake1d(channels: finalDim)
        self._convOut.wrappedValue = DACVAEWNConv1d(
            inChannels: finalDim,
            outChannels: dOut,
            kernelSize: 7,
            padding: 3
        )

        // Watermarking (uses snake_out/conv_out as shared layers)
        self._wmModel.wrappedValue = DACVAEWatermarker(
            dOut: dOut,
            dLatent: dWmOut,
            channels: wmChannels,
            hidden: 512,
            nbits: nbits,
            lstmLayers: 2
        )

        self.alpha = Float(wmChannels) / Float(dWmOut)

        // Set shared layers after initialization
        super.init()
        wmModel.setSharedLayers(snakeOut: snakeOut, convOut: convOut)
    }

    /// Decode latent features to audio (without final output layers).
    public func callAsFunction(_ x: MLXArray, message: MLXArray? = nil) -> MLXArray {
        var h = convIn(x)
        for block in blocks {
            h = block(h)
        }
        return h
    }

    /// Decode with optional watermarking.
    public func decodeWithWatermark(_ x: MLXArray, message: MLXArray? = nil) -> MLXArray {
        if let msg = message, alpha > 0.0 {
            return watermark(x, message: msg)
        } else {
            // Standard path: snake -> conv -> tanh
            var h = snakeOut(x)
            h = MLX.tanh(convOut(h))
            return h
        }
    }

    /// Apply watermarking to the decoder output.
    private func watermark(_ x: MLXArray, message: MLXArray) -> MLXArray {
        // Watermark encoder: snake_out -> conv_out -> tanh -> wm_conv
        var h = wmModel.encoderBlock(x)

        // Upsample through decoder blocks (watermark path)
        for block in blocks.reversed() {
            h = block.upsampleGroup(h)
        }

        // Post-process: LSTM -> ELU -> conv
        h = wmModel.encoderBlock.postProcess(h)

        // Apply message embedding
        // Transpose h to (B, C, T) for msg_processor
        var hT = h.transposed(0, 2, 1)
        hT = wmModel.msgProcessor(hT, msg: message)
        h = hT.transposed(0, 2, 1)

        // Watermark decoder: conv -> LSTM
        h = wmModel.decoderBlock(h)

        // Downsample through decoder blocks (watermark path)
        for block in blocks {
            h = block.downsampleGroup(h)
        }

        // Post-process: ELU -> conv
        h = wmModel.decoderBlock.postProcess(h)

        // Blend: snake_out(x) -> conv_out -> tanh + alpha * watermark
        let xBase = wmModel.encoderBlock.forwardNoWMConv(x)
        let result = xBase + alpha * h

        return result
    }
}

// MARK: - DACVAE Model

/// DACVAE audio codec for SAM-Audio.
///
/// This is a VAE-style audio codec that encodes audio to a latent space
/// and decodes it back. Unlike the standard DAC, this uses continuous
/// latent representations instead of discrete codes.
public class DACVAE: Module {
    public let config: DACVAEConfig
    public let sampleRate: Int
    public let hopLength: Int

    @ModuleInfo(key: "encoder") var encoder: DACVAEEncoder
    @ModuleInfo(key: "quantizer_in_proj") var quantizerInProj: DACVAEQuantizerInProj
    @ModuleInfo(key: "quantizer_out_proj") var quantizerOutProj: DACVAEQuantizerOutProj
    @ModuleInfo(key: "decoder") var decoder: DACVAEFullDecoder

    public init(config: DACVAEConfig) {
        self.config = config
        self.sampleRate = config.sampleRate
        self.hopLength = config.hopLength

        // Encoder
        self._encoder.wrappedValue = DACVAEEncoder(
            dModel: config.encoderDim,
            strides: config.encoderRates,
            dLatent: config.latentDim
        )

        // Quantizer projections (VAE-style)
        self._quantizerInProj.wrappedValue = DACVAEQuantizerInProj(
            inDim: config.latentDim,
            outDim: config.codebookDim
        )
        self._quantizerOutProj.wrappedValue = DACVAEQuantizerOutProj(
            inDim: config.codebookDim,
            outDim: config.latentDim
        )

        // Decoder with watermarking
        self._decoder.wrappedValue = DACVAEFullDecoder(
            inputChannel: config.latentDim,
            channels: config.decoderDim,
            rates: config.decoderRates
        )
    }

    /// Pad waveform to be divisible by hop_length.
    private func pad(_ wavs: MLXArray) -> MLXArray {
        let length = wavs.shape[1]
        if length % hopLength != 0 {
            let padAmount = hopLength - (length % hopLength)
            if padAmount > 0 {
                return MLX.padded(wavs, widths: [.init(0), .init((0, padAmount)), .init(0)])
            }
        }
        return wavs
    }

    /// Encode waveform to latent representation.
    ///
    /// - Parameter waveform: Audio tensor of shape (batch, length, 1)
    /// - Returns: Latent features of shape (batch, channels, frames)
    public func encode(_ waveform: MLXArray) -> MLXArray {
        let wav = pad(waveform)
        let z = encoder(wav)

        // VAE-style: project and take mean
        let proj = quantizerInProj(z)
        let splits = proj.split(parts: 2, axis: -1)
        let mean = splits[0]

        // Transpose to (batch, channels, frames)
        return mean.transposed(0, 2, 1)
    }

    /// Decode latent features back to waveform.
    ///
    /// For SAM-Audio, this accepts features in codebook_dim space (128)
    /// and projects them to latent_dim before decoding.
    ///
    /// - Parameters:
    ///   - encodedFrames: Tensor of shape (batch, codebook_dim, frames)
    ///   - chunkSize: If provided, decode in chunks of this many frames
    /// - Returns: Waveform of shape (batch, length, 1)
    public func decode(_ encodedFrames: MLXArray, chunkSize: Int? = nil) -> MLXArray {
        // Use chunked decoding for memory efficiency if requested
        if let chunk = chunkSize {
            return decodeChunked(encodedFrames, chunkSize: chunk)
        }

        // Transpose to (batch, frames, codebook_dim)
        let encodedT = encodedFrames.transposed(0, 2, 1)

        // Project from codebook_dim to latent_dim
        let emb = quantizerOutProj(encodedT)

        // Decode
        var out = decoder(emb)

        // Apply final output
        out = decoder.snakeOut(out)
        out = MLX.tanh(decoder.convOut(out))

        return out
    }

    private func decodeChunkFeatures(_ encodedChunk: MLXArray) -> MLXArray {
        let emb = quantizerOutProj(encodedChunk)
        var out = decoder(emb)
        out = decoder.snakeOut(out)
        out = MLX.tanh(decoder.convOut(out))
        eval(out)
        return out
    }

    private func linearRamp(start: Float, end: Float, count: Int) -> MLXArray {
        if count <= 0 {
            return MLXArray([] as [Float])
        }
        if count == 1 {
            return MLXArray([start])
        }

        let denom = Float(count - 1)
        var values: [Float] = []
        values.reserveCapacity(count)
        for i in 0..<count {
            let t = Float(i) / denom
            values.append(start + (end - start) * t)
        }
        return MLXArray(values)
    }

    /// Decode in chunks to reduce peak memory usage.
    private func decodeChunked(_ encodedFrames: MLXArray, chunkSize: Int, overlap: Int = 4) -> MLXArray {
        let totalFrames = encodedFrames.shape[2]
        let encodedT = encodedFrames.transposed(0, 2, 1)  // (B, T, C)
        let samplesPerFrame = hopLength
        let overlapSamples = overlap * samplesPerFrame

        var chunks: [MLXArray] = []
        var start = 0

        while start < totalFrames {
            let end = min(start + chunkSize, totalFrames)
            let chunk = encodedT[0..., start..<end, 0...]
            let out = decodeChunkFeatures(chunk)
            chunks.append(out)

            if end >= totalFrames {
                break
            }
            start = end - overlap
            Memory.clearCache()
        }

        if chunks.count == 1 {
            return chunks[0]
        }

        // Crossfade blend chunk overlaps to match python reference behavior.
        var resultParts: [MLXArray] = []
        for (i, chunk) in chunks.enumerated() {
            let chunkSamples = chunk.shape[1]

            if i == 0 {
                if chunks.count > 1 && overlapSamples > 0 && chunkSamples > overlapSamples {
                    let fadeOutStart = chunkSamples - overlapSamples
                    let fade = linearRamp(start: 1.0, end: 0.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                    let chunkMain = chunk[0..., 0..<fadeOutStart, 0...]
                    let chunkFade = chunk[0..., fadeOutStart..<chunkSamples, 0...] * fade
                    resultParts.append(chunkMain)
                    resultParts.append(chunkFade)
                } else {
                    resultParts.append(chunk)
                }
                continue
            }

            if i == chunks.count - 1 {
                if overlapSamples > 0 && chunkSamples >= overlapSamples && !resultParts.isEmpty {
                    let fadeIn = linearRamp(start: 0.0, end: 1.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                    let chunkFade = chunk[0..., 0..<overlapSamples, 0...] * fadeIn
                    let chunkRest = chunk[0..., overlapSamples..<chunkSamples, 0...]
                    resultParts[resultParts.count - 1] = resultParts[resultParts.count - 1] + chunkFade
                    resultParts.append(chunkRest)
                } else {
                    resultParts.append(chunk)
                }
                continue
            }

            if overlapSamples > 0 && chunkSamples > 2 * overlapSamples && !resultParts.isEmpty {
                let fadeIn = linearRamp(start: 0.0, end: 1.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                let fadeOut = linearRamp(start: 1.0, end: 0.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                let fadeOutStart = chunkSamples - overlapSamples

                let chunkFadeIn = chunk[0..., 0..<overlapSamples, 0...] * fadeIn
                let chunkMiddle = chunk[0..., overlapSamples..<fadeOutStart, 0...]
                let chunkFadeOut = chunk[0..., fadeOutStart..<chunkSamples, 0...] * fadeOut

                resultParts[resultParts.count - 1] = resultParts[resultParts.count - 1] + chunkFadeIn
                resultParts.append(chunkMiddle)
                resultParts.append(chunkFadeOut)
            } else {
                resultParts.append(chunk)
            }
        }

        return MLX.concatenated(resultParts, axis: 1)
    }

    /// Streaming decode that yields chunked audio with overlap blending.
    public func decodeStreaming(
        _ encodedFrames: MLXArray,
        chunkSize: Int = 50,
        overlap: Int = 4
    ) -> AnyIterator<(MLXArray, Bool)> {
        let totalFrames = encodedFrames.shape[2]
        if totalFrames == 0 {
            return AnyIterator { nil }
        }

        let encodedT = encodedFrames.transposed(0, 2, 1)  // (B, T, C)
        eval(encodedT)

        let overlapSamples = overlap * hopLength
        var prevFadeOut: MLXArray? = nil
        var start = 0
        var chunkIdx = 0
        var finished = false

        return AnyIterator {
            if finished || start >= totalFrames {
                return nil
            }

            let end = min(start + chunkSize, totalFrames)
            let isLast = end >= totalFrames

            let chunk = encodedT[0..., start..<end, 0...]
            let out = self.decodeChunkFeatures(chunk)
            let outSamples = out.shape[1]

            if chunkIdx == 0 {
                if !isLast && overlapSamples > 0 && outSamples > overlapSamples {
                    let fadeOutStart = outSamples - overlapSamples
                    let fadeOut = self.linearRamp(start: 1.0, end: 0.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                    prevFadeOut = out[0..., fadeOutStart..<outSamples, 0...] * fadeOut
                    if let prev = prevFadeOut { eval(prev) }

                    let result = out[0..., 0..<fadeOutStart, 0...]
                    eval(result)

                    start = end - overlap
                    chunkIdx += 1
                    Memory.clearCache()
                    return (result, false)
                }

                finished = true
                return (out, true)
            }

            if isLast {
                finished = true
                if overlapSamples > 0, let prev = prevFadeOut, outSamples >= overlapSamples {
                    let fadeIn = self.linearRamp(start: 0.0, end: 1.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                    let blended = prev + out[0..., 0..<overlapSamples, 0...] * fadeIn
                    eval(blended)

                    let finalChunk = MLX.concatenated(
                        [blended, out[0..., overlapSamples..<outSamples, 0...]],
                        axis: 1
                    )
                    eval(finalChunk)
                    return (finalChunk, true)
                }
                return (out, true)
            }

            if overlapSamples > 0, let prev = prevFadeOut, outSamples > 2 * overlapSamples {
                let fadeIn = self.linearRamp(start: 0.0, end: 1.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                let fadeOut = self.linearRamp(start: 1.0, end: 0.0, count: overlapSamples).reshaped([1, overlapSamples, 1])
                let fadeOutStart = outSamples - overlapSamples

                let blended = prev + out[0..., 0..<overlapSamples, 0...] * fadeIn
                eval(blended)

                prevFadeOut = out[0..., fadeOutStart..<outSamples, 0...] * fadeOut
                if let newPrev = prevFadeOut { eval(newPrev) }

                let middleChunk = MLX.concatenated(
                    [blended, out[0..., overlapSamples..<fadeOutStart, 0...]],
                    axis: 1
                )
                eval(middleChunk)

                start = end - overlap
                chunkIdx += 1
                Memory.clearCache()
                return (middleChunk, false)
            }

            start = end - overlap
            chunkIdx += 1
            Memory.clearCache()
            return (out, false)
        }
    }

    /// Stream decode with callback for each emitted audio chunk.
    @discardableResult
    public func decodeStream(
        _ encodedFrames: MLXArray,
        callback: (_ audioChunk: MLXArray, _ chunkIndex: Int, _ isLast: Bool) -> Void,
        chunkSize: Int = 50,
        overlap: Int = 4
    ) -> Int {
        var totalSamples = 0
        var chunkIndex = 0
        let stream = decodeStreaming(encodedFrames, chunkSize: chunkSize, overlap: overlap)

        while let (audioChunk, isLast) = stream.next() {
            callback(audioChunk, chunkIndex, isLast)
            totalSamples += audioChunk.shape[1]
            chunkIndex += 1
        }

        return totalSamples
    }

    /// Encode waveform to codebook space (for SAM-Audio).
    ///
    /// This returns VAE mean features in codebook_dim (128) which is what
    /// SAM-Audio uses for flow matching.
    ///
    /// - Parameter waveform: Audio tensor of shape (batch, 1, length)
    /// - Returns: Latent features of shape (batch, codebook_dim, frames)
    public func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        // Transpose from (batch, 1, length) to (batch, length, 1)
        var wav = waveform.transposed(0, 2, 1)
        wav = pad(wav)

        // Encode to latent space
        let z = encoder(wav)  // (B, T, latent_dim)

        // Project to codebook space and take VAE mean
        let proj = quantizerInProj(z)  // (B, T, 2*codebook_dim)
        let splits = proj.split(parts: 2, axis: -1)
        let mean = splits[0]  // (B, T, codebook_dim)

        // Transpose to (batch, codebook_dim, frames)
        return mean.transposed(0, 2, 1)
    }

    /// Convert waveform sample index to feature frame index.
    public func wavIdxToFeatureIdx(_ wavIdx: Int, sampleRate: Int? = nil) -> Int {
        let srcRate = sampleRate ?? self.sampleRate
        let targetLength = Int(ceil(Float(self.sampleRate * wavIdx) / Float(srcRate)))
        return Int(ceil(Float(targetLength) / Float(hopLength)))
    }

    /// Convert feature frame index to waveform sample index.
    public func featureIdxToWavIdx(_ featureIdx: Int, sampleRate: Int? = nil) -> Int {
        let srcRate = sampleRate ?? self.sampleRate
        let wavChunkLen = Float(featureIdx * hopLength) * (Float(srcRate) / Float(self.sampleRate))
        return Int(wavChunkLen)
    }

    /// Load a pretrained DACVAE model from HuggingFace Hub.
    public static func fromPretrained(
        _ repoId: String,
        cache: HubCache = .default
    ) async throws -> DACVAE {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "DACVAE",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }

        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        // Load config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(DACVAEConfig.self, from: configData)

        // Create model
        let model = DACVAE(config: config)

        // Load weights
        let weightsURL = modelURL.appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: weightsURL)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)

        eval(model.parameters())

        return model
    }
}

extension DACVAE: AudioCodecModel {
    public typealias EncodedAudio = MLXArray

    public var codecSampleRate: Double? { Double(sampleRate) }

    public func encodeAudio(_ waveform: MLXArray) -> MLXArray {
        encode(waveform)
    }

    public func decodeAudio(_ input: MLXArray) -> MLXArray {
        decode(input)
    }
}
