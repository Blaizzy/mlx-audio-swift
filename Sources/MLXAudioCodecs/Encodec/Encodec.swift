//
//  Encodec.swift
//  MLXAudioCodecs
//
//  Ported from mlx-audio Python implementation
//

import Foundation
import MLX
import MLXNN
import Hub

// MARK: - Encodec Encoder

/// SEANet encoder as used by EnCodec.
public class EncodecEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    public init(config: EncodecConfig) {
        var model: [Module] = []

        // Initial conv
        model.append(EncodecConv1d(
            config: config,
            inChannels: config.audioChannels,
            outChannels: config.numFilters,
            kernelSize: config.kernelSize
        ))

        var scaling = 1

        // Downsampling blocks
        for ratio in config.upsamplingRatios.reversed() {
            let currentScale = scaling * config.numFilters

            // Residual blocks
            for j in 0..<config.numResidualLayers {
                let dilation = Int(pow(Double(config.dilationGrowthRate), Double(j)))
                model.append(EncodecResnetBlock(
                    config: config,
                    dim: currentScale,
                    dilations: [dilation, 1]
                ))
            }

            model.append(ELU())
            model.append(EncodecConv1d(
                config: config,
                inChannels: currentScale,
                outChannels: currentScale * 2,
                kernelSize: ratio * 2,
                stride: ratio
            ))

            scaling *= 2
        }

        // LSTM
        model.append(EncodecLSTMBlock(config: config, dimension: scaling * config.numFilters))

        // Final layers
        model.append(ELU())
        model.append(EncodecConv1d(
            config: config,
            inChannels: scaling * config.numFilters,
            outChannels: config.hiddenSize,
            kernelSize: config.lastKernelSize
        ))

        self._layers.wrappedValue = model
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            if let conv = layer as? EncodecConv1d {
                h = conv(h)
            } else if let resnet = layer as? EncodecResnetBlock {
                h = resnet(h)
            } else if let elu = layer as? ELU {
                h = elu(h)
            } else if let lstm = layer as? EncodecLSTMBlock {
                h = lstm(h)
            }
        }
        return h
    }
}

// MARK: - Encodec Decoder

/// SEANet decoder as used by EnCodec.
public class EncodecDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    public init(config: EncodecConfig) {
        var scaling = Int(pow(2.0, Double(config.upsamplingRatios.count)))
        var model: [Module] = []

        // Initial conv
        model.append(EncodecConv1d(
            config: config,
            inChannels: config.hiddenSize,
            outChannels: scaling * config.numFilters,
            kernelSize: config.kernelSize
        ))

        // LSTM
        model.append(EncodecLSTMBlock(config: config, dimension: scaling * config.numFilters))

        // Upsampling blocks
        for ratio in config.upsamplingRatios {
            let currentScale = scaling * config.numFilters

            model.append(ELU())
            model.append(EncodecConvTranspose1dLayer(
                config: config,
                inChannels: currentScale,
                outChannels: currentScale / 2,
                kernelSize: ratio * 2,
                stride: ratio
            ))

            // Residual blocks
            for j in 0..<config.numResidualLayers {
                let dilation = Int(pow(Double(config.dilationGrowthRate), Double(j)))
                model.append(EncodecResnetBlock(
                    config: config,
                    dim: currentScale / 2,
                    dilations: [dilation, 1]
                ))
            }

            scaling /= 2
        }

        // Final layers
        model.append(ELU())
        model.append(EncodecConv1d(
            config: config,
            inChannels: config.numFilters,
            outChannels: config.audioChannels,
            kernelSize: config.lastKernelSize
        ))

        self._layers.wrappedValue = model
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            if let conv = layer as? EncodecConv1d {
                h = conv(h)
            } else if let convT = layer as? EncodecConvTranspose1dLayer {
                h = convT(h)
            } else if let resnet = layer as? EncodecResnetBlock {
                h = resnet(h)
            } else if let elu = layer as? ELU {
                h = elu(h)
            } else if let lstm = layer as? EncodecLSTMBlock {
                h = lstm(h)
            }
        }
        return h
    }
}

// MARK: - Encodec

/// EnCodec neural audio codec.
public class Encodec: Module {
    public let config: EncodecConfig

    @ModuleInfo(key: "encoder") var encoder: EncodecEncoder
    @ModuleInfo(key: "decoder") var decoder: EncodecDecoder
    @ModuleInfo(key: "quantizer") var quantizer: EncodecResidualVectorQuantizer

    public init(config: EncodecConfig) {
        self.config = config
        self._encoder.wrappedValue = EncodecEncoder(config: config)
        self._decoder.wrappedValue = EncodecDecoder(config: config)
        self._quantizer.wrappedValue = EncodecResidualVectorQuantizer(config: config)
    }

    // MARK: - Properties

    public var channels: Int {
        return config.audioChannels
    }

    public var samplingRate: Int {
        return config.samplingRate
    }

    public var chunkLength: Int? {
        guard let chunkLengthS = config.chunkLengthS else {
            return nil
        }
        return Int(chunkLengthS * Float(config.samplingRate))
    }

    public var chunkStride: Int? {
        guard let _ = config.chunkLengthS, let overlap = config.overlap, let length = chunkLength else {
            return nil
        }
        return max(1, Int((1.0 - overlap) * Float(length)))
    }

    // MARK: - Encoding

    private func encodeFrame(
        inputValues: MLXArray,
        bandwidth: Float,
        paddingMask: MLXArray
    ) -> (MLXArray, MLXArray?) {
        let length = inputValues.shape[1]
        let duration = Float(length) / Float(config.samplingRate)

        if let chunkLengthS = config.chunkLengthS, duration > 1e-5 + chunkLengthS {
            fatalError("Duration of frame (\(duration)) is longer than chunk \(chunkLengthS)")
        }

        var values = inputValues
        var scale: MLXArray? = nil

        if config.normalize {
            // Normalize if the padding is non zero
            values = values * paddingMask.expandedDimensions(axis: -1)
            let mono = values.sum(axis: 2, keepDims: true) / Float(values.shape[2])
            scale = MLX.sqrt(mono.square().mean(axis: 1, keepDims: true)) + 1e-8
            values = values / scale!
        }

        let embeddings = encoder(values)
        let codes = quantizer.encode(embeddings, bandwidth: bandwidth)
        return (codes, scale)
    }

    /// Encodes the input audio waveform into discrete codes.
    ///
    /// - Parameters:
    ///   - inputValues: The input audio waveform with shape (batch_size, sequence_length, channels).
    ///   - paddingMask: Optional padding mask.
    ///   - bandwidth: The target bandwidth in kbps. Must be one of config.targetBandwidths.
    ///
    /// - Returns: A tuple of (codes, scales) where codes has shape (num_chunks, batch_size, num_codebooks, frames).
    public func encode(
        _ inputValues: MLXArray,
        paddingMask: MLXArray? = nil,
        bandwidth: Float? = nil
    ) -> (MLXArray, [MLXArray?]) {
        let bw = bandwidth ?? config.targetBandwidths[0]

        guard config.targetBandwidths.contains(bw) else {
            fatalError("This model doesn't support bandwidth \(bw). Select one of \(config.targetBandwidths)")
        }

        let inputLength = inputValues.shape[1]
        let channels = inputValues.shape[2]

        guard channels >= 1 && channels <= 2 else {
            fatalError("Number of audio channels must be 1 or 2, but got \(channels)")
        }

        var chunkLen = chunkLength ?? inputLength
        let stride = chunkStride ?? inputLength

        let mask = paddingMask ?? MLXArray.ones([inputValues.shape[0], inputLength]).asType(.bool)

        var encodedFrames: [MLXArray] = []
        var scales: [MLXArray?] = []

        let step = chunkLen - stride
        if chunkLength == nil {
            chunkLen = inputLength
        }

        var offset = 0
        while offset < inputLength - step {
            let frameMask = mask[0..., offset..<(offset + chunkLen)].asType(.bool)
            let frame = inputValues[0..., offset..<(offset + chunkLen), 0...]
            let (encodedFrame, scale) = encodeFrame(inputValues: frame, bandwidth: bw, paddingMask: frameMask)
            encodedFrames.append(encodedFrame)
            scales.append(scale)
            offset += stride
        }

        let stackedFrames = MLX.stacked(encodedFrames, axis: 0)
        return (stackedFrames, scales)
    }

    // MARK: - Decoding

    private func decodeFrame(_ codes: MLXArray, scale: MLXArray? = nil) -> MLXArray {
        let embeddings = quantizer.decode(codes)
        var outputs = decoder(embeddings)
        if let s = scale {
            outputs = outputs * s
        }
        return outputs
    }

    private static func linearOverlapAdd(_ frames: [MLXArray], hopStride: Int) -> MLXArray {
        guard !frames.isEmpty else {
            fatalError("`frames` cannot be an empty list.")
        }

        let N = frames[0].shape[0]
        let frameLength = frames[0].shape[1]
        let C = frames[0].shape[2]
        let totalSize = hopStride * (frames.count - 1) + frames.last!.shape[1]

        // Create weight vector
        let timeValues = (0..<frameLength).map { Float($0 + 1) / Float(frameLength + 1) }
        let weightValues = timeValues.map { 0.5 - abs($0 - 0.5) }

        var outData = [Float](repeating: 0, count: N * totalSize * C)
        var sumWeightData = [Float](repeating: 0, count: totalSize)

        var offset = 0
        for frame in frames {
            let fLen = frame.shape[1]
            let frameData = frame.asArray(Float.self)

            for b in 0..<N {
                for t in 0..<fLen {
                    for c in 0..<C {
                        let outIdx = b * totalSize * C + (offset + t) * C + c
                        let frameIdx = b * fLen * C + t * C + c
                        outData[outIdx] += weightValues[t] * frameData[frameIdx]
                    }
                }
            }

            for t in 0..<fLen {
                sumWeightData[offset + t] += weightValues[t]
            }

            offset += hopStride
        }

        // Normalize by weight sum
        for b in 0..<N {
            for t in 0..<totalSize {
                for c in 0..<C {
                    let idx = b * totalSize * C + t * C + c
                    if sumWeightData[t] != 0 {
                        outData[idx] /= sumWeightData[t]
                    }
                }
            }
        }

        return MLXArray(outData).reshaped([N, totalSize, C])
    }

    /// Decodes the given codes into an output audio waveform.
    ///
    /// - Parameters:
    ///   - audioCodes: Discrete code embeddings of shape (num_chunks, batch_size, num_codebooks, frames).
    ///   - audioScales: Scaling factor for each input chunk.
    ///   - paddingMask: Optional padding mask.
    ///
    /// - Returns: The decoded audio waveform.
    public func decode(
        _ audioCodes: MLXArray,
        _ audioScales: [MLXArray?],
        paddingMask: MLXArray? = nil
    ) -> MLXArray {
        var audioValues: MLXArray

        if chunkLength == nil {
            guard audioCodes.shape[0] == 1 else {
                fatalError("Expected one frame, got \(audioCodes.shape[0])")
            }
            audioValues = decodeFrame(audioCodes[0], scale: audioScales[0])
        } else {
            var decodedFrames: [MLXArray] = []

            for i in 0..<audioCodes.shape[0] {
                let frame = audioCodes[i]
                let scale = audioScales[i]
                let decoded = decodeFrame(frame, scale: scale)
                decodedFrames.append(decoded)
            }

            audioValues = Self.linearOverlapAdd(decodedFrames, hopStride: chunkStride ?? 1)
        }

        // Truncate based on padding mask
        if let mask = paddingMask, mask.shape[1] < audioValues.shape[1] {
            audioValues = audioValues[0..., 0..<mask.shape[1], 0...]
        }

        return audioValues
    }

    // MARK: - Loading

    /// Load a pretrained Encodec model from HuggingFace Hub.
    public static func fromPretrained(_ pathOrRepo: String) async throws -> Encodec {
        let hub = HubApi()
        let repo = Hub.Repo(id: pathOrRepo)

        // Download model files
        let modelURL = try await hub.snapshot(from: repo, matching: ["*.json", "*.safetensors"])

        // Load config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(EncodecConfig.self, from: configData)

        // Create model
        let model = Encodec(config: config)

        // Load weights
        let weightsURL = modelURL.appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: weightsURL)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)

        return model
    }
}

// MARK: - Audio Preprocessing

/// Preprocess audio for EnCodec model.
///
/// - Parameters:
///   - rawAudio: The audio waveform(s) to process.
///   - samplingRate: The sampling rate.
///   - chunkLength: Optional chunk length.
///   - chunkStride: Optional chunk stride.
///
/// - Returns: A tuple of (inputs, masks) where inputs has shape (batch, length, channels).
public func preprocessEncodecAudio(
    _ rawAudio: [MLXArray],
    samplingRate: Int = 24000,
    chunkLength: Int? = nil,
    chunkStride: Int? = nil
) -> (MLXArray, MLXArray) {
    let audio = rawAudio.map { x -> MLXArray in
        if x.ndim == 1 {
            return x.expandedDimensions(axis: -1)
        }
        return x
    }

    var maxLength = audio.map { $0.shape[0] }.max() ?? 0
    if let chunk = chunkLength, let stride = chunkStride {
        maxLength += chunk - (maxLength % stride)
    }

    var inputs: [MLXArray] = []
    var masks: [MLXArray] = []

    for x in audio {
        let length = x.shape[0]
        var mask = MLXArray.ones([length]).asType(.bool)
        var padded = x

        let difference = maxLength - length
        if difference > 0 {
            mask = MLX.padded(mask, widths: [.init((0, difference))])
            padded = MLX.padded(x, widths: [.init((0, difference)), .init(0)])
        }

        inputs.append(padded)
        masks.append(mask)
    }

    return (MLX.stacked(inputs, axis: 0), MLX.stacked(masks, axis: 0))
}
