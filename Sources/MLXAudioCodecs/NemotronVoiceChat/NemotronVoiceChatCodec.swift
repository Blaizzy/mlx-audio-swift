// Adapted from mlx-audio's Nemotron VoiceChat codec implementation.
import Foundation
import MLX
import MLXAudioCore
import MLXNN

/// Per-stream state for incremental decode. Use one cache per stream and reset
/// it before reusing it for another utterance. A cache is not thread safe.
public final class NemotronVoiceChatCodecCache {
    private var states: [String: MLXArray] = [:]

    public init() {}
    public func reset() { states.removeAll() }
    public var isEmpty: Bool { states.isEmpty }

    func update(_ input: MLXArray, key: String, padding: Int, flush: Bool) -> MLXArray {
        guard padding > 0 else { return input }
        let previous = states[key] ?? MLXArray.zeros(
            [input.dim(0), padding, input.dim(2)], dtype: input.dtype)
        precondition(previous.shape == [input.dim(0), padding, input.dim(2)],
                     "A decode cache cannot change batch size or channel layout")
        let joined = concatenated([previous, input], axis: 1)
        if flush {
            states.removeValue(forKey: key)
        } else {
            states[key] = joined[0..., (joined.dim(1) - padding)..., 0...]
        }
        return joined
    }
}

final class NemotronCodecChannelNorm: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray

    init(_ channels: Int) {
        weight = .ones([channels])
        bias = .zeros([channels])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let mean = MLX.mean(input, axis: -1, keepDims: true)
        let variance = MLX.variance(input, axis: -1, keepDims: true)
        return (input - mean) * rsqrt(variance + 1e-6) * weight + bias
    }
}

final class NemotronCodecConvNeXt: Module, UnaryLayer {
    let kernelSize: Int
    let layerID: Int
    @ModuleInfo var dwconv: Conv1d
    @ModuleInfo var norm: NemotronCodecChannelNorm
    @ModuleInfo var pwconv1: Conv1d
    @ModuleInfo var pwconv2: Conv1d

    init(channels: Int, kernelSize: Int, layerID: Int) {
        self.kernelSize = kernelSize
        self.layerID = layerID
        _dwconv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                     kernelSize: kernelSize, groups: channels)
        _norm.wrappedValue = NemotronCodecChannelNorm(channels)
        _pwconv1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: 4 * channels, kernelSize: 1)
        _pwconv2.wrappedValue = Conv1d(inputChannels: 4 * channels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        callAsFunction(input, cache: nil, flush: false)
    }

    func callAsFunction(_ input: MLXArray, cache: NemotronVoiceChatCodecCache?, flush: Bool) -> MLXArray {
        let padded = cache?.update(input, key: "conv.\(layerID)", padding: kernelSize - 1, flush: flush)
            ?? MLX.padded(input, widths: [.init(0), .init((kernelSize - 1, 0)), .init(0)])
        return input + pwconv2(gelu(pwconv1(norm(dwconv(padded)))))
    }
}

final class NemotronCodecEncoder: Module, UnaryLayer {
    @ModuleInfo var layers: [Module]

    init(_ config: NemotronVoiceChatCodecConfig) {
        let channels = config.channelMultipliers.map { config.baseChannels * $0 }
        var result: [Module] = [Conv1d(inputChannels: config.stftChannels,
                                      outputChannels: channels[0], kernelSize: 1, bias: false)]
        var blockID = 0
        for (index, rate) in config.downsampleRates.enumerated() {
            for _ in 0..<config.blocksPerStage {
                result.append(NemotronCodecConvNeXt(channels: channels[index],
                                                   kernelSize: config.blockKernelSize, layerID: blockID))
                blockID += 1
            }
            let next = index + 1 < channels.count ? channels[index + 1] : config.latentDim
            result.append(Conv1d(inputChannels: channels[index], outputChannels: next,
                                 kernelSize: rate, stride: rate, bias: false))
        }
        _layers.wrappedValue = result
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers.reduce(input) { ($1 as! any UnaryLayer)($0) }
    }
}

final class NemotronCodecDecoder: Module {
    @ModuleInfo var layers: [Module]

    init(_ config: NemotronVoiceChatCodecConfig) {
        let channels = config.channelMultipliers.map { config.baseChannels * $0 }
        var result: [Module] = []
        var sourceChannels = config.latentDim
        var blockID = 0
        for (channels, rate) in zip(channels.reversed(), config.downsampleRates.reversed()) {
            result.append(ConvTransposed1d(inputChannels: sourceChannels, outputChannels: channels,
                                           kernelSize: rate, stride: rate, bias: false))
            for _ in 0..<config.blocksPerStage {
                result.append(NemotronCodecConvNeXt(channels: channels,
                                                   kernelSize: config.blockKernelSize, layerID: blockID))
                blockID += 1
            }
            sourceChannels = channels
        }
        result.append(Conv1d(inputChannels: channels[0], outputChannels: config.stftChannels,
                             kernelSize: 1, bias: false))
        _layers.wrappedValue = result
    }

    func callAsFunction(_ input: MLXArray, cache: NemotronVoiceChatCodecCache?, flush: Bool) -> MLXArray {
        var hidden = input
        for layer in layers {
            if let block = layer as? NemotronCodecConvNeXt {
                hidden = block(hidden, cache: cache, flush: flush)
            } else {
                hidden = (layer as! any UnaryLayer)(hidden)
            }
        }
        return hidden
    }
}

final class NemotronCodecVariance: Module {
    let variance = MLXArray(Float(1))
}

final class NemotronCodecPRVQ: Module {
    @ParameterInfo(key: "mus_list") var means: [MLXArray]
    @ModuleInfo(key: "variance_list") var variances: [NemotronCodecVariance]

    init(_ config: NemotronVoiceChatCodecConfig) {
        _means.wrappedValue = (0..<config.numQuantizers).map { _ in
            MLXArray.zeros([config.codebookSize, config.latentDim])
        }
        _variances.wrappedValue = (0..<config.numQuantizers).map { _ in NemotronCodecVariance() }
    }

    func encode(_ latents: MLXArray) -> MLXArray {
        var residual = latents
        var codes: [MLXArray] = []
        for mean in means {
            let distances = sum(residual * residual, axis: -1, keepDims: true)
                - 2 * matmul(residual, mean.T)
                + sum(mean * mean, axis: -1).reshaped([1, 1, -1])
            let indices = argMin(distances, axis: -1)
            codes.append(indices)
            residual = residual - mean[indices]
        }
        return stacked(codes, axis: 1)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        precondition(codes.ndim == 3 && codes.dim(1) > 0 && codes.dim(1) <= means.count,
                     "Expected nonempty codebook prefix shaped (batch, codebooks, frames)")
        precondition(codes.dtype.isInteger, "Codec IDs must be integers")
        var latents = MLXArray.zeros([codes.dim(0), codes.dim(2), means[0].dim(1)], dtype: means[0].dtype)
        for i in 0..<codes.dim(1) { latents = latents + means[i][codes[0..., i, 0...]] }
        return latents
    }
}

/// Codes and original sample length used by the shared roundtrip API.
public struct NemotronVoiceChatEncodedAudio {
    /// Integer codes with shape (batch, codebooks, frames).
    public let codes: MLXArray
    public let originalLength: Int
    public init(codes: MLXArray, originalLength: Int) {
        self.codes = codes
        self.originalLength = originalLength
    }
}

/// NemotronLabs VoiceChat's neural audio codec, independent of the language model.
public final class NemotronVoiceChatCodec: Module {
    public let config: NemotronVoiceChatCodecConfig
    public var sampleRate: Int { config.sampleRate }
    public var frameRate: Double { config.frameRate }
    public var waveformToTokenRatio: Int { config.waveformToTokenRatio }
    /// Initial waveform delay carried by the reference's streaming ISTFT.
    /// The final nonempty flush emits this many extra samples.
    public var streamingLookaheadSamples: Int {
        let left = (config.nFFT - config.hopLength) / 2
        return ((left + config.hopLength - 1) / config.hopLength) * config.hopLength
    }
    @ModuleInfo var encoder: NemotronCodecEncoder
    @ModuleInfo var decoder: NemotronCodecDecoder
    @ModuleInfo var prvq: NemotronCodecPRVQ

    public init(config: NemotronVoiceChatCodecConfig = .init()) {
        precondition((try? config.validate()) != nil, "Invalid Nemotron codec configuration")
        self.config = config
        _encoder.wrappedValue = NemotronCodecEncoder(config)
        _decoder.wrappedValue = NemotronCodecDecoder(config)
        _prvq.wrappedValue = NemotronCodecPRVQ(config)
    }

    /// Encode mono (B,T) or (B,1,T) audio at sampleRate to (B,frames,latentDim).
    /// Native encoding drops incomplete token frames, matching the reference.
    /// Use encodeAudio for padded, length-preserving roundtrips and short inputs.
    public func encodeLatents(_ waveform: MLXArray) -> MLXArray {
        precondition(waveform.ndim == 2 || (waveform.ndim == 3 && waveform.dim(1) == 1),
                     "Expected mono waveform (B,T) or (B,1,T)")
        let audio = waveform.ndim == 3 ? waveform[0..., 0, 0...] : waveform
        precondition(audio.dim(1) >= waveformToTokenRatio, "Native encode requires at least one token frame")
        return encoder(spectrogram(audio))
    }

    /// Encode mono (B,T) or (B,1,T) to integer (B,Q,frames) codes.
    public func encode(_ waveform: MLXArray) -> MLXArray { prvq.encode(encodeLatents(waveform)) }

    /// Decode integer (B,Q,frames) codes into mono waveform (B,1,samples).
    public func decode(_ codes: MLXArray) -> MLXArray { decodeLatents(prvq.decode(codes)) }

    /// Decode latent (B,frames,latentDim) values into mono (B,1,samples).
    public func decodeLatents(
        _ latents: MLXArray, cache: NemotronVoiceChatCodecCache? = nil, flush: Bool = false
    ) -> MLXArray {
        precondition(latents.ndim == 3 && latents.dim(2) == config.latentDim)
        guard latents.dim(1) > 0 else {
            if flush { cache?.reset() }
            return MLXArray.zeros([latents.dim(0), 1, 0])
        }
        let features = decoder(latents, cache: cache, flush: flush)
        let bins = config.nFFT / 2 + 1
        let logits = features[0..., 0..., 0..<bins]
        let phase = features[0..., 0..., bins...]
        let magnitude = 100 * exp(-softplus(-logits + Float(log(100.0))))
        var real = magnitude * cos(phase)
        let imaginary = magnitude * sin(phase)
        var imag = concatenated([
            MLXArray.zeros([features.dim(0), features.dim(1), 1], dtype: imaginary.dtype),
            imaginary[0..., 0..., 1..<(bins - 1)],
            MLXArray.zeros([features.dim(0), features.dim(1), 1], dtype: imaginary.dtype),
        ], axis: -1)
        let left = (config.nFFT - config.hopLength) / 2
        let right = config.nFFT - config.hopLength - left
        let halfSpecPadding = (left + config.hopLength - 1) / config.hopLength
        if let cache {
            real = cache.update(real, key: "istft.real", padding: 2 * halfSpecPadding, flush: flush)
            imag = cache.update(imag, key: "istft.imag", padding: 2 * halfSpecPadding, flush: flush)
            if flush {
                let tail = MLXArray.zeros([real.dim(0), halfSpecPadding, bins], dtype: real.dtype)
                real = concatenated([real, tail], axis: 1)
                imag = concatenated([imag, tail], axis: 1)
            }
        }
        var audio = inverseSpectrogram(real: real, imag: imag)
        audio = audio[0..., left..<(audio.dim(1) - right)]
        if cache != nil {
            let trim = halfSpecPadding * config.hopLength
            if trim > 0 { audio = audio[0..., trim..<(audio.dim(1) - trim)] }
        }
        return audio.expandedDimensions(axis: 1)
    }

    /// Decode only new code frames with per-stream convolution and STFT state.
    /// The initial waveform includes streamingLookaheadSamples of delay. Set
    /// flush on the final nonempty chunk to emit that many additional tail
    /// samples. Initial ISTFT normalization differs from offline decoding.
    public func decodeStep(
        _ codes: MLXArray, cache: NemotronVoiceChatCodecCache, flush: Bool = false
    ) -> MLXArray {
        decodeLatents(prvq.decode(codes), cache: cache, flush: flush)
    }

    private func spectrogram(_ audio: MLXArray) -> MLXArray {
        let left = (config.nFFT - config.hopLength) / 2
        let right = config.nFFT - config.hopLength - left
        let padded = MLX.padded(audio.asType(.float32), widths: [.init(0), .init((left, right))])
        let frames = 1 + (padded.dim(1) - config.nFFT) / config.hopLength
        let view = asStrided(padded, [padded.dim(0), frames, config.nFFT],
                             strides: [padded.dim(1), config.hopLength, 1])
        let spectrum = MLXFFT.rfft(view * periodicWindow(), axis: -1)
        return concatenated([spectrum.realPart(), spectrum.imaginaryPart()], axis: -1)
    }

    private func inverseSpectrogram(real: MLXArray, imag: MLXArray) -> MLXArray {
        let window = periodicWindow()
        let spectrum = real + MLXArray(real: Float(0), imaginary: Float(1)) * imag
        var frames = MLXFFT.irfft(spectrum, axis: -1)
        frames = clip(frames, min: -window, max: window) * window
        let count = real.dim(1)
        let outputLength = (count - 1) * config.hopLength + config.nFFT
        let positions = (MLXArray(0..<count).expandedDimensions(axis: 1) * config.hopLength
                         + MLXArray(0..<config.nFFT).expandedDimensions(axis: 0)).flattened()
        var norm = MLXArray.zeros([outputLength])
        norm = norm.at[positions].add(tiled(window * window, repetitions: [count]))
        norm = maximum(norm, MLXArray(Float(1e-10)))
        let batchOffsets = (MLXArray(0..<real.dim(0)) * outputLength).expandedDimensions(axis: 1)
        let indices = (positions.expandedDimensions(axis: 0) + batchOffsets).flattened()
        var output = MLXArray.zeros([real.dim(0) * outputLength])
        output = output.at[indices].add(frames.flattened())
        return output.reshaped([real.dim(0), outputLength]) / norm
    }

    private func periodicWindow() -> MLXArray {
        // The reference computes its scalar cosine in double precision before
        // constructing the FP32 window; keep that order for the small FFT too.
        MLXArray((0..<config.nFFT).map {
            Float(0.5 * (1 - Foundation.cos(2 * Double.pi * Double($0) / Double(config.nFFT))))
        })
    }
}

extension NemotronVoiceChatCodec: AudioCodecModel {
    public typealias EncodedAudio = NemotronVoiceChatEncodedAudio
    public var codecSampleRate: Double? { Double(sampleRate) }

    /// Shared protocol waveform layout: (batch, samples, 1), at sampleRate.
    public func encodeAudio(_ waveform: MLXArray) -> NemotronVoiceChatEncodedAudio {
        precondition(waveform.ndim == 3 && waveform.dim(2) == 1 && waveform.dim(1) > 0,
                     "Expected nonempty waveform (batch, samples, 1)")
        let length = waveform.dim(1)
        let pad = (waveformToTokenRatio - length % waveformToTokenRatio) % waveformToTokenRatio
        let padded = MLX.padded(waveform, widths: [.init(0), .init((0, pad)), .init(0)])
        return .init(codes: encode(padded.transposed(0, 2, 1)), originalLength: length)
    }

    /// Shared protocol returns (batch, originalSamples, 1), removing encode padding.
    public func decodeAudio(_ input: NemotronVoiceChatEncodedAudio) -> MLXArray {
        precondition(input.originalLength >= 0, "Original waveform length cannot be negative")
        let audio = decode(input.codes).transposed(0, 2, 1)
        return audio[0..., 0..<min(input.originalLength, audio.dim(1)), 0...]
    }
}
