//
//  DACVAE.swift
//  MLXAudioCodecs
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN
import Hub

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

    /// Decode in chunks to reduce peak memory usage.
    private func decodeChunked(_ encodedFrames: MLXArray, chunkSize: Int, overlap: Int = 4) -> MLXArray {
        let totalFrames = encodedFrames.shape[2]

        // Transpose to (batch, frames, codebook_dim)
        let encodedT = encodedFrames.transposed(0, 2, 1)

        var chunks: [MLXArray] = []
        var start = 0

        while start < totalFrames {
            let end = min(start + chunkSize, totalFrames)

            // Extract chunk
            let chunk = encodedT[0..., start..<end, 0...]

            // Project from codebook_dim to latent_dim
            let emb = quantizerOutProj(chunk)

            // Decode
            var out = decoder(emb)
            out = decoder.snakeOut(out)
            out = MLX.tanh(decoder.convOut(out))
            eval(out)

            chunks.append(out)

            // Move to next chunk (with overlap for blending)
            if end >= totalFrames {
                break
            }
            start = end - overlap
        }

        // Simple concatenation (without complex crossfade for now)
        if chunks.count == 1 {
            return chunks[0]
        }

        return MLX.concatenated(chunks, axis: 1)
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
    public static func fromPretrained(_ repoId: String) async throws -> DACVAE {
        let hub = HubApi()
        let repo = Hub.Repo(id: repoId)

        // Download model files
        let modelURL = try await hub.snapshot(from: repo, matching: ["*.json", "*.safetensors"])

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
